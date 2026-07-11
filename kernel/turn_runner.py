"""Kernel turn runner — the single model+tool loop shared by lightweight hosts and the full app.

This module provides `run_turn()` and `TurnHooks`, the core conversational turn
loop that executes one mode's pipeline: pre-turn operations, model completion with
tool dispatch, and post-turn side effects.

DESIGN — Extension via TurnHooks, not replacement:
    Little Lisa and the production app both call `run_turn()`. They differ only
    in which `TurnHooks` they provide. Every hook defaults to a sensible no-op or
    stub, so a minimal host participates with zero overrides.

    The full app (`app/agent.py`) can eventually delegate its chat and workflow
    completion paths to `run_turn()` by registering `AppTurnRunnerHooks` that
    override model resolution, context assembly, tool dispatch, and post-turn
    persistence — without duplicating the tool loop.

Gaps (use the full `app/agent.py` turn loop when you need):
    - FSM stage transitions (transition_stage, session state machines)
    - Mode delegation (delegate_mode tool dispatch)
    - Live system state, passive intelligence, reflection context
    - Self-model, architecture atlas, capability vibes
    - Thread persistence, handoff artifacts, conversation digest refresh
    - Model restrict/unavailable gating, provider fallback
    - Proactive/autopilot surface reflection
    - Deterministic intent handling (notes, reminders, memory recall/disputes)
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from kernel.ipsar_fsm import (
    apply_transition,
    get_mode_system_intro,
    get_transition_tool_schema,
    get_workflow_catalog,
    is_terminal_stage,
    run_mode_entry_operations,
    run_mode_pre_turn_operations,
)
from kernel.models import (
    CallLedger,
    OperationExecutionContext,
    OperationResult,
    SurfaceContext,
    ToolExecutionContext,
    ToolResult,
    TurnBudget,
    WorkflowSessionState,
    WorkflowTransitionRequest,
)
from kernel.operations.registry import (
    build_operation_tool_schemas,
    execute_operation,
    get_operation_catalog,
)
from kernel.openai_function_tools import build_strict_openai_function_tool
from kernel.provider_core import StreamedChunk, StreamedCompletion, _aggregate_streamed_chunks
from kernel.seams.protocols import (
    ContentRegistry,
    DataSanitizer,
    EventBus,
    ExtensionEvidenceBuilder,
    InputGuard,
    ModelPolicy,
    OutputGuard,
    ProviderGatewayProvider,
)
from kernel.slice import (
    SliceMode,
    SliceOutcome,
    SlicePolicy,
    build_policy_for_trigger,
    resolve_mode_for_trigger,
)
from kernel.tools.mcp_client import MCPClientConfig, MCPToolPool
from kernel.tools.registry import execute_tool, get_tool_catalog
from kernel.tools.schema_translator import mcp_tool_to_openai_function, openai_tool_call_to_mcp

logger = logging.getLogger(__name__)

# The tool name the model calls to request a workflow stage transition.
# Intercepted by the turn loop before the operation/tool/MCP dispatch branches.
_TRANSITION_TOOL_NAME = "transition_stage"

# Stuck-session guard: after this many turns in the same stage without a
# transition, the loop injects a nudge message prompting the model to
# transition or ask the user. After the cap, the loop forces a final
# no-tools completion to stop burning tokens.
_STAGE_NUDGE_THRESHOLD = 4
_STAGE_NUDGE_CAP = 6

# ── Public types ───────────────────────────────────────────────────────────


@dataclass
class TurnResult:
    """Result of one call to `run_turn()`."""

    response_text: str
    tool_call_count: int = 0
    iteration_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    duration_ms: float = 0.0
    latency_ms: int = 0
    slice_outcomes: list[SliceOutcome] = field(default_factory=list)
    """SliceOutcome records for every opportunistic slice that ran
    during this turn. Populated only when the host has wired the
    corresponding slice slot. Empty list when opportunistic
    house-keeping is off (``slice_tuning=None`` or all triggers set
    to ``never``)."""
    blocked: bool = False
    """True when an input or output guard blocked the turn. Input blocks
    short-circuit before the model call; output blocks replace the response
    with a safe fallback and emit a ``guardrail.audit`` event."""
    transitioned_to: str | None = None
    """When the model called ``transition_stage`` during this turn, the new
    ``stage_id``. None when no transition occurred or the mode is not a
    workflow. The host reads this to persist the updated session and update
    its UI route preview."""
    workflow_terminated: bool = False
    """True when the transition moved the session to a terminal stage. The
    loop forces one final no-tools completion using the terminal stage's
    instruction as the system prompt so the model produces a closing summary."""


def _lazy_model_policy() -> ModelPolicy:
    from kernel.context import ManifestModelPolicy

    return ManifestModelPolicy()


def _lazy_provider_gateways() -> ProviderGatewayProvider:
    from kernel.context import KernelProviderGatewayProvider

    return KernelProviderGatewayProvider()


def _lazy_event_bus() -> EventBus:
    from kernel.context import InProcessEventBus

    return InProcessEventBus()


def _lazy_data_sanitizer() -> DataSanitizer:
    from kernel.context import PassThroughSanitizer

    return PassThroughSanitizer()


def _lazy_extension_evidence_builder() -> ExtensionEvidenceBuilder:
    from kernel.context import NoOpExtensionEvidenceBuilder

    return NoOpExtensionEvidenceBuilder()


def _lazy_input_guard() -> InputGuard:
    from kernel.context import PassThroughInputGuard

    return PassThroughInputGuard()


def _lazy_output_guard() -> OutputGuard:
    from kernel.context import PassThroughOutputGuard

    return PassThroughOutputGuard()


@dataclass
class ContextSeams:
    """Bundle of context-layer seams injected through ``TurnHooks``.

    These six seams invert the private ``app.*`` dependencies that
    ``kernel/context/`` previously took. Defaults wire the public
    implementations shipped under ``kernel.context``; the production app
    overrides any of them with its own without a shim layer.
    """

    content_registry: ContentRegistry = field(default=None)  # type: ignore[assignment]
    model_policy: ModelPolicy = field(default_factory=_lazy_model_policy)
    provider_gateways: ProviderGatewayProvider = field(default_factory=_lazy_provider_gateways)
    event_bus: EventBus = field(default_factory=_lazy_event_bus)
    data_sanitizer: DataSanitizer = field(default_factory=_lazy_data_sanitizer)
    extension_evidence_builder: ExtensionEvidenceBuilder = field(
        default_factory=_lazy_extension_evidence_builder
    )
    input_guard: InputGuard = field(default_factory=_lazy_input_guard)
    """Validates user input before the model call (Phase 1). Default:
    ``PassThroughInputGuard`` allows everything. Hosts override with a real
    implementation (Azure AI Content Safety, Bedrock Guardrails, regex, etc.)."""
    output_guard: OutputGuard = field(default_factory=_lazy_output_guard)
    """Validates model output before it reaches the user (Phase 3). Default:
    ``PassThroughOutputGuard`` allows everything. Hosts override with a real
    implementation. Blocked outputs are replaced with a safe fallback and
    audited via ``event_bus.publish("guardrail.audit", ...)``."""


@dataclass
class TurnHooks:
    """Extension points for the kernel turn runner.

    The host (lightweight host or production app) provides concrete implementations.
    Every callable defaults to a no-op or sensible stub, so a minimal host can
    participate with zero overrides.

    DESIGN: Hooks are named for *what they do*, not *why the app needs them*.
    This keeps the kernel free of app-specific concepts like "passive
    intelligence," "capability vibes," or "self-model context."
    """

    # ── Model resolution ────────────────────────────────────────────────
    resolve_model: Callable[[str], Any] | None = None

    complete: Callable[..., Awaitable[Any]] | None = None
    """Core completion function. Must accept messages, model_def, max_tokens,
    temperature, tools (optional), and response_format (optional). Returns
    an object with `.content` (str) and `.tool_calls` (list[dict]).

    A host MUST provide this. If None, `run_turn` raises NotImplementedError.
    """

    complete_stream: Callable[..., Any] | None = None
    """Streaming completion function. When set, the turn runner uses this
    instead of ``complete`` for the model call. Must be an async generator
    that yields ``StreamedChunk`` objects as tokens arrive.

    v1 streams tokens only — tool-call streaming is deferred to v2. When
    the model wants to call tools, the stream's ``finish_reason`` will be
    ``'tool_calls'`` and the host should fall back to ``complete`` for that
    iteration. The turn runner handles this fallback automatically.
    """

    # ── Pre-turn context assembly ────────────────────────────────────────
    assemble_system_prompt: Callable[[str, str | None], str] | None = None
    """Given mode_instruction: str and optional stage_instruction: str|None,
    return the full system prompt.  Default: mode_instruction only.

    The app injects SYSTEM_PROMPT, turn-input contract, and planner block here.
    """

    assemble_user_context: Callable[[str, str, str], str] | None = None
    """Given mode_id: str, query: str, and pre_turn_briefing: str, return
    additional user-context content injected before the query in the final
    user message.  Default: empty string.

    The app injects surface blocks, user blocks, live context, memory recall,
    self-model, architecture atlas, etc. here.
    """

    # ── MCP tool servers ──────────────────────────────────────────────────
    mcp_tool_servers: list[MCPClientConfig] = field(default_factory=list)
    """MCP server configurations to spawn for tool discovery and dispatch.
    Default: empty list (no external tools). The host declares which MCP
    servers to run (e.g. ``lisa-context``, ``lisa-guard``)
    and the kernel manages the subprocess lifecycle via ``MCPToolPool``.

    The kernel speaks MCP tool schema internally. At the provider edge,
    ``schema_translator`` converts to whatever the provider expects.
    """

    mcp_tool_pool: Any = None
    """Optional host-owned long-lived :class:`MCPToolPool`.

    When set, ``run_turn`` reuses this pool across turns instead of
    spawning/tearing down a per-turn pool. The host owns the lifecycle:
    it calls ``await pool.start()`` once at runtime init and
    ``await pool.aclose()`` at process exit. This is the eager-start /
    warm-server path — ``mcp_tool_servers`` should then contain only the
    cold (per-turn) configs; the warm configs live inside this pool.

    When ``None`` (default), ``run_turn`` builds a per-turn pool from
    ``mcp_tool_servers`` as before — existing hosts and tests keep
    working unchanged.
    """

    # ── Operation dispatch ────────────────────────────────────────────────
    execute_operation: Callable[..., Awaitable[OperationResult]] | None = None
    """Execute a named operation with an OperationExecutionContext. Default:
    delegates to `kernel.operations.registry.execute_operation`.

    Operations are internal kernel capabilities (pre-turn briefings, entry
    actions, declarative prompts). They are also advertised to the LLM as
    tools directly from the in-process catalog and dispatched via this hook
    so they receive full host callbacks (complete_fn, resolve_model_fn, etc.).
    """

    # ── Post-turn side effects ───────────────────────────────────────────
    on_turn_complete: Callable[[str, str, str, list[dict[str, Any]]], Awaitable[None]] | None = None
    """Called after every successful turn with mode_id, query, response_text,
    and the full message list. Default: no-op.

    The app hooks thread persistence, conversation digest refresh, provenance,
    and context-usage events here.
    """

    # ── Opportunistic house-keeping (turn-anchored slices) ───────────────
    # See kernel/slice.py for the full contract. Each slice slot is
    # independently optional. A host that wants only post-turn slices
    # leaves pre_turn_slice and explicit_slice as None. A host that
    # wants nothing (purposeful neglect) leaves all three as None.
    slice_tuning: Any = None
    """SliceTuning instance (from kernel.slice) that controls budget
    shape. Default: ``None`` means "use SliceTuning() defaults and
    disable all three triggers." A host picks the trigger modes by
    setting the appropriate enabled flags.
    """

    pre_turn_slice: Callable[[Any], Awaitable[Any]] | None = None
    """Opportunistic slice fired before the model call. Receives a
    ``SlicePolicy`` (mode=PRE_TURN) and returns a ``SliceOutcome``.

    The framework short-circuits the call if the budget is zero (e.g.
    the pressure curve has collapsed to nothing on this turn).
    """

    post_turn_slice: Callable[[Any], Awaitable[Any]] | None = None
    """Opportunistic slice fired after the response is committed.
    Receives a ``SlicePolicy`` (mode=POST_TURN) and returns a
    ``SliceOutcome``.
    """

    explicit_slice: Callable[[Any], Awaitable[Any]] | None = None
    """Slice fired only when an operation or CLI command invokes it.
    Receives a ``SlicePolicy`` (mode=EXPLICIT). The caller's budget
    is honored verbatim — pressure does not apply."""

    last_slice_finished_at: float | None = None
    """Monotonic timestamp of the last slice that completed on this
    host. Used by the framework's pressure math. The host can also
    read/write this to coordinate multiple slices."""

    context_seams: ContextSeams = field(default_factory=ContextSeams)
    """Context-layer seams (content registry, model policy, provider gateways,
    event bus, data sanitizer). Defaults wire the public implementations
    shipped under ``kernel.context``; the production app overrides any of
    them with its own."""

    # ── Workshop child-workflow execution ──────────────────────────────────
    workflow_turn_runner: Any | None = None
    """Callable matching ``WorkflowTurnRunner`` protocol for workshop child
    workflow execution. When set, injected into ``OperationExecutionContext``
    so workshop operations can fan out child turns. Default: None (workshop
    operations will report the runner is missing)."""

    workflow_session_factory: Any | None = None
    """Callable matching ``(user_id, user_slug, mode_id) -> WorkflowSessionState``
    for creating isolated child workflow sessions. When set, injected into
    ``OperationExecutionContext`` so workshop operations can create child
    sessions. Default: None."""

    workflow_session: Any | None = None
    """Callable returning the current ``WorkflowSessionState`` for the active
    user/mode, or ``None`` when no workflow session is active. When set, the
    turn loop reads the session's ``stage_id`` and injects the current
    stage's ``instruction_body`` into the system prompt alongside the mode
    instruction. This is the seam that makes workflow FSM stages live at
    runtime — without it, the loaded stage instructions are never injected
    and the mode runs as if it were a flat chat mode. Default: None (stage
    instructions are omitted; chat and action modes are unaffected)."""

    workflow_session_updater: Any | None = None
    """Callable receiving an updated ``WorkflowSessionState`` after a
    successful ``transition_stage`` tool call. The host persists the session
    (file, DB, in-memory map) so the next turn's ``workflow_session`` hook
    returns the new stage. Default: None (transitions are applied in-process
    but not persisted across turns)."""

    # ── Host state provider ───────────────────────────────────────────────
    state_provider: Any | None = None
    """Callable returning a ``UnifiedState`` for the current host session.
    When set, injected into ``OperationExecutionContext`` so operations that
    need cached runtime state (e.g. workshop) can access it. Default: None."""

    # ── Workshop context bundle builder ────────────────────────────────────
    build_workshop_context_bundle_fn: Any | None = None
    """Callable building a ``WorkshopContextBundle`` from a plan and state.
    When set, injected into ``OperationExecutionContext`` so the workshop
    operation can curate per-panel context bundles. Default: None."""

    # ── Live introspection ─────────────────────────────────────────────────
    introspect: Any | None = None
    """Async callable returning an ``IntrospectSnapshot`` from the host's
    live runtime (process memory, asyncio tasks, MCP pool health, token
    usage, log tail). When set, injected into
    ``OperationExecutionContext.introspect_fn`` so the self-model operation
    and the reflect_self operation can access live runtime state. Default:
    None (self-model telemetry fields stay empty)."""


# ── Streaming adapter ──────────────────────────────────────────────────────


class StreamedCompletionAdapter:
    """Adapter that makes a ``StreamedCompletion`` look like a ``CompletionResult``.

    The turn runner's tool loop expects ``completion.content``,
    ``completion.tool_calls``, and ``completion.usage``. When the streaming
    path produces a ``StreamedCompletion`` (no tool calls in v1), this adapter
    bridges the two shapes without forcing the streaming code to construct a
    full ``CompletionResult``.
    """

    __slots__ = ("_aggregated", "content", "tool_calls", "usage")

    def __init__(self, aggregated: StreamedCompletion) -> None:
        self._aggregated = aggregated
        self.content = aggregated.content
        self.tool_calls = aggregated.tool_calls
        self.usage = aggregated.usage

    def __repr__(self) -> str:
        return f"StreamedCompletionAdapter(content={self.content!r}, finish_reason={self._aggregated.finish_reason!r})"


# ── Default hook implementations ───────────────────────────────────────────


async def _noop_complete(**kwargs: Any) -> Any:
    raise NotImplementedError(
        "TurnHooks.complete is required. Provide a completion function that "
        "returns an object with .content (str) and .tool_calls (list[dict])."
    )


def _default_resolve_model(model_id: str) -> Any:
    """Return a minimal stub model definition."""
    return _StubModelDef(model_id=model_id or "default")


def _default_assemble_system_prompt(
    mode_instruction: str,
    _stage_instruction: str | None,
) -> str:
    return mode_instruction


def _default_assemble_user_context(
    _mode_id: str,
    _query: str,
    _pre_turn_briefing: str,
) -> str:
    return ""


async def _default_execute_operation(
    operation_id: str,
    ctx: OperationExecutionContext,
) -> OperationResult:
    return await execute_operation(operation_id, ctx)


async def _default_on_turn_complete(
    _mode_id: str,
    _query: str,
    _response_text: str,
    _messages: list[dict[str, Any]],
) -> None:
    return


async def _default_slice(_policy: SlicePolicy) -> SliceOutcome:
    return SliceOutcome()


# ── Internal helpers ────────────────────────────────────────────────────────


async def _fire_slice(
    *,
    hooks: TurnHooks,
    trigger: SliceMode,
    slice_outcomes: list[SliceOutcome],
) -> SliceOutcome | None:
    """Dispatch one opportunistic house-keeping slice.

    Returns the outcome if the slice actually ran, ``None`` otherwise.
    The hook is invoked only when the host has both wired the
    corresponding slot AND enabled the trigger in its ``SliceTuning``.

    The caller is responsible for updating
    ``hooks.last_slice_finished_at`` after the call returns; that
    field is on the host's ``TurnHooks`` instance so it survives across
    turns of the same session.
    """
    slot = {
        SliceMode.PRE_TURN: hooks.pre_turn_slice,
        SliceMode.POST_TURN: hooks.post_turn_slice,
        SliceMode.EXPLICIT: hooks.explicit_slice,
    }.get(trigger)
    if slot is None:
        return None

    tuning = hooks.slice_tuning
    if tuning is None:
        return None
    if resolve_mode_for_trigger(tuning=tuning, trigger=trigger) == SliceMode.NEVER:
        return None

    has_pre_turn = (
        hooks.pre_turn_slice is not None
        and resolve_mode_for_trigger(tuning=tuning, trigger=SliceMode.PRE_TURN) != SliceMode.NEVER
    )
    has_post_turn = (
        hooks.post_turn_slice is not None
        and resolve_mode_for_trigger(tuning=tuning, trigger=SliceMode.POST_TURN) != SliceMode.NEVER
    )
    policy = build_policy_for_trigger(
        trigger=trigger,
        tuning=tuning,
        last_slice_finished_at=hooks.last_slice_finished_at,
        has_pre_turn=has_pre_turn,
        has_post_turn=has_post_turn,
    )
    if policy.budget_ms <= 0:
        return None

    started = time.monotonic()
    try:
        outcome = await slot(policy)
    except Exception as exc:  # pylint: disable=broad-except
        elapsed_ms = (time.monotonic() - started) * 1000.0
        logger.warning(
            "%s slice failed: %s: %s",
            trigger.value,
            type(exc).__name__,
            exc,
        )
        outcome = SliceOutcome(
            work_items_completed=0,
            elapsed_ms=elapsed_ms,
            last_item="<error>",
            notes=(f"{type(exc).__name__}: {exc}",),
        )
    if not isinstance(outcome, SliceOutcome):
        # Defensive: a host that returns a non-SliceOutcome should not
        # break the turn. Coerce silently and log once.
        logger.debug(
            "%s slice returned %s instead of SliceOutcome; coercing",
            trigger.value,
            type(outcome).__name__,
        )
        outcome = SliceOutcome(
            work_items_completed=0,
            elapsed_ms=(time.monotonic() - started) * 1000.0,
        )
    # Backfill elapsed_ms if the host didn't measure.
    elif outcome.elapsed_ms <= 0:
        outcome = SliceOutcome(
            work_items_completed=outcome.work_items_completed,
            elapsed_ms=(time.monotonic() - started) * 1000.0,
            last_item=outcome.last_item,
            notes=outcome.notes,
        )
    slice_outcomes.append(outcome)
    return outcome


@dataclass(frozen=True, slots=True)
class _StubModelDef:
    model_id: str
    reasoning_model: bool = False


def _extract_pre_turn_briefing(result: OperationResult) -> str:
    """Return the briefing text that should be injected into the user payload.

    Declarative operations may fall back to rendering their full structured payload
    as JSON when `output_text` is empty. That is useful for direct operation
    callers, but pre-turn prompt assembly should only inject the authored briefing
    text field itself. When the structured payload explicitly says `output_text`
    is empty, treat the pre-turn briefing as absent.
    """

    payload = result.output_json
    if isinstance(payload, dict):
        for field_name in ("output_text", "body_markdown", "markdown", "answer_summary", "summary"):
            candidate = payload.get(field_name)
            if isinstance(candidate, str):
                return candidate.strip()

    return result.output_text.strip()


# ── Core turn runner ────────────────────────────────────────────────────────


async def run_turn(
    *,
    mode_id: str,
    query: str,
    hooks: TurnHooks,
    max_tool_iterations: int = 5,
    preferred_model_id: str | None = None,
    turn_budget: TurnBudget | None = None,
    logger_override: logging.Logger | None = None,
) -> TurnResult:
    """Execute one conversational turn with the full model+tool loop.

    This is the single turn loop shared by lightweight hosts and the production app.
    Both call the same function — they differ only in which `TurnHooks` they
    provide.

    Phases:
    1. **PRE-TURN**: Resolve mode config, run pre-turn/entry operations,
       assemble system prompt and user context, recall memory snippet.
    2. **TOOL LOOP**: Call hooks.complete → dispatch MCP tool calls via
       MCPToolPool → feed results back → repeat until the model returns
       content without tools or max_tool_iterations is reached.
    3. **POST-TURN**: Call hooks.on_turn_complete.

    Args:
        mode_id: The route mode string (e.g. 'default', 'discuss', 'search').
        query: The user's message text after command-prefix stripping.
        hooks: Extension-point callbacks provided by the host.
        max_tool_iterations: Maximum number of tool-calling loop iterations.
        preferred_model_id: Optional per-turn model alias override supplied by
            the host when routing needs an explicit front-door model.
        turn_budget: Optional whole-turn resource budget. When set, the loop
            tracks tool calls, tokens, and wall time against the budget and
            forces a final no-tools completion when any limit is reached.
            When None, only ``max_tool_iterations`` bounds the loop (the
            historical behavior). The budget is the "ratio" knob: the host
            expresses the mix of inference and compute as one envelope.
        logger_override: Optional logger for test capture.

    Returns:
        TurnResult with the final response text and loop statistics.

    Raises:
        NotImplementedError: If hooks.complete is None.
    """
    _log = logger_override or logger
    mode_config = get_workflow_catalog().get(mode_id)

    # Resolve hook callables with defaults
    resolve_model = hooks.resolve_model or _default_resolve_model
    complete = hooks.complete or _noop_complete
    assemble_system = hooks.assemble_system_prompt or _default_assemble_system_prompt
    assemble_user = hooks.assemble_user_context or _default_assemble_user_context
    exec_op = hooks.execute_operation or _default_execute_operation
    on_complete = hooks.on_turn_complete or _default_on_turn_complete

    # ── MCP tool pool ──────────────────────────────────────────────────
    # The kernel speaks MCP tool schema internally. Tool discovery and
    # dispatch go through MCPToolPool; provider-specific schemas are
    # translated at the provider edge by schema_translator.
    #
    # When the host provides a long-lived warm pool (``hooks.mcp_tool_pool``),
    # reuse it across turns without spawning/tearing down subprocesses per
    # turn. The host owns that pool's lifecycle (start at init, aclose at
    # exit). Otherwise fall back to a per-turn pool built from
    # ``hooks.mcp_tool_servers`` (cold servers only when a warm pool is
    # present, or all servers when no warm pool is set).
    host_pool = hooks.mcp_tool_pool
    if host_pool is not None:
        return await _run_turn_inner(
            mode_id=mode_id,
            query=query,
            hooks=hooks,
            pool=host_pool,
            resolve_model=resolve_model,
            complete=complete,
            assemble_system=assemble_system,
            assemble_user=assemble_user,
            exec_op=exec_op,
            on_complete=on_complete,
            mode_config=mode_config,
            max_tool_iterations=max_tool_iterations,
            preferred_model_id=preferred_model_id,
            turn_budget=turn_budget,
            logger_override=logger_override,
        )
    pool = MCPToolPool(hooks.mcp_tool_servers)
    await pool.start()
    try:
        return await _run_turn_inner(
            mode_id=mode_id,
            query=query,
            hooks=hooks,
            pool=pool,
            resolve_model=resolve_model,
            complete=complete,
            assemble_system=assemble_system,
            assemble_user=assemble_user,
            exec_op=exec_op,
            on_complete=on_complete,
            mode_config=mode_config,
            max_tool_iterations=max_tool_iterations,
            preferred_model_id=preferred_model_id,
            turn_budget=turn_budget,
            logger_override=logger_override,
        )
    finally:
        await pool.aclose()


async def _run_turn_inner(
    *,
    mode_id: str,
    query: str,
    hooks: TurnHooks,
    pool: MCPToolPool,
    resolve_model: Callable[[str], Any],
    complete: Callable[..., Awaitable[Any]],
    assemble_system: Callable[[str, str | None], str],
    assemble_user: Callable[[str, str, str], str],
    exec_op: Callable[..., Awaitable[OperationResult]],
    on_complete: Callable[[str, str, str, list[dict[str, Any]]], Awaitable[None]],
    mode_config: Any,
    max_tool_iterations: int,
    preferred_model_id: str | None,
    turn_budget: TurnBudget | None,
    logger_override: logging.Logger | None,
) -> TurnResult:
    """Inner turn loop that owns the MCPToolPool lifecycle via the caller."""

    _log = logger_override or logger

    # Per-turn call ledger shared across all dispatch branches (operations,
    # pack tools, MCP tools). Enforces ``max_calls_per_turn`` uniformly; the
    # declarative operation runtime keeps its own in-loop counter for its
    # bounded sub-agent, but the ledger is the authoritative cross-branch cap.
    ledger = CallLedger()
    # Wall-clock anchor for the turn budget. Captured once so every budget
    # check measures from the same origin.
    turn_start_mono = time.monotonic()

    async def _mcp_tool_executor(tool_id: str, _ctx: Any) -> ToolResult:
        """Bridge OperationExecutionContext.execute_tool_fn to in-process tools.

        Catalog-driven: if the tool_id is in the internal tool catalog
        (discovered from packs), dispatch in-process via
        ``tools.registry.execute_tool``. Otherwise fall through to the MCP
        pool. Declarative operations that need to call tools route through
        here.
        """
        from kernel.tools.registry import get_tool_catalog as _get_tools

        tool_catalog = _get_tools()
        definition = tool_catalog.get(tool_id)
        if definition is not None and definition.implementation_module is not None:
            tool_ctx = _to_tool_context(tool_id, _ctx)
            return await execute_tool(tool_id, tool_ctx)

        params = getattr(_ctx, "parameters", {}) or {}
        mcp_result = await pool.call_tool(tool_id, params)
        return ToolResult(
            tool_id=tool_id,
            output_text=mcp_result.output_text,
            error=mcp_result.output_text if mcp_result.is_error else "",
        )

    # ── Phase 1: Pre-turn ─────────────────────────────────────────────
    pre_briefing = ""
    if (
        mode_config is not None
        and mode_config.mode_kind == "action"
        and mode_config.entry_operation_ids
    ):
        # Action modes run entry operations and return immediately
        ctx = _build_minimal_context(
            mode_id=mode_id,
            query=query,
            resolve_model_fn=resolve_model,
            complete_fn=complete,
            execute_tool_fn=_mcp_tool_executor,
            execute_operation_fn=exec_op,
            workflow_turn_runner=hooks.workflow_turn_runner,
            workflow_session_factory=hooks.workflow_session_factory,
            state_provider=hooks.state_provider,
            build_workshop_context_bundle_fn=hooks.build_workshop_context_bundle_fn,
        )
        result = await run_mode_entry_operations(mode_id, ctx, ledger=ledger)
        return TurnResult(
            response_text=result.output_text.strip() or "Done",
            tool_call_count=0,
            iteration_count=1,
        )

    # Chat/workflow modes: run pre-turn operations
    ctx = _build_minimal_context(
        mode_id=mode_id,
        query=query,
        resolve_model_fn=resolve_model,
        complete_fn=complete,
        execute_tool_fn=_mcp_tool_executor,
        execute_operation_fn=exec_op,
        workflow_turn_runner=hooks.workflow_turn_runner,
        workflow_session_factory=hooks.workflow_session_factory,
        state_provider=hooks.state_provider,
        build_workshop_context_bundle_fn=hooks.build_workshop_context_bundle_fn,
    )
    pre_result = await run_mode_pre_turn_operations(mode_id, ctx, ledger=ledger)
    pre_briefing = _extract_pre_turn_briefing(pre_result)
    # ── Phase 1.0: Input guard ─────────────────────────────────────
    # Runs before the pre-turn slice so a blocked input does not consume
    # house-keeping budget. A sanitizing guard rewrites the query for the
    # remainder of the turn.
    guard_surface = SurfaceContext(
        surface_type="personal_dm",
        surface_id="lite-host-console",
        memory_scope="personal",
        thread_owner_id="lite-user",
    )
    input_guard_result = await hooks.context_seams.input_guard.validate(query, guard_surface)
    if not input_guard_result.allowed:
        blocked_reason = input_guard_result.blocked_reason or "Input blocked by safety policy."
        _log.warning("Input guard blocked turn for mode=%s: %s", mode_id, blocked_reason)
        return TurnResult(
            response_text=blocked_reason,
            blocked=True,
        )
    if input_guard_result.sanitized_text is not None:
        query = input_guard_result.sanitized_text

    # ── Phase 1.5: Pre-turn opportunistic slice ────────────────────────
    # Fires before the model call so any state the recall/assemble
    # hooks will read is fresh. Bounded by the pressure curve.
    slice_outcomes: list[SliceOutcome] = []
    pre_turn_outcome = await _fire_slice(
        hooks=hooks,
        trigger=SliceMode.PRE_TURN,
        slice_outcomes=slice_outcomes,
    )
    if pre_turn_outcome is not None:
        hooks.last_slice_finished_at = time.monotonic()

    # Assemble prompts — workflow modes inject the current stage's
    # instruction_body alongside the mode-level intro. Without this, the
    # loaded stage instructions (inquire.md, reason.md, etc.) are never
    # injected and the FSM is dead at runtime. The host wires
    # ``hooks.workflow_session`` to return the current session; when absent
    # or not a workflow mode, the loop falls back to mode-level only (the
    # historical behavior for chat and action modes).
    mode_instruction = get_mode_system_intro(mode_id)
    stage_instruction: str | None = None
    if (
        mode_config is not None
        and mode_config.mode_kind == "workflow"
        and hooks.workflow_session is not None
    ):
        session = hooks.workflow_session()
        if session is not None and session.stage_id in mode_config.stages:
            stage_instruction = (
                mode_config.stages[session.stage_id].instruction_body or None
            )
    system_prompt = assemble_system(mode_instruction, stage_instruction)
    user_context = assemble_user(mode_id, query, pre_briefing)
    memory_snippet = ""

    # ── Phase 2: Tool loop ────────────────────────────────────────────
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]

    # Build user message
    user_parts: list[str] = []
    if pre_briefing:
        user_parts.append(f"## Pre-turn Briefing\n\n{pre_briefing}")
    if memory_snippet:
        user_parts.append(f"## Memory Recall\n\n{memory_snippet}")
    if user_context:
        user_parts.append(user_context)
    user_parts.append(f"## User Query\n\n{query}")
    user_content = "\n\n".join(user_parts)
    messages.append({"role": "user", "content": user_content})

    # Build tool schemas for this turn — MCP tools + in-process operations.
    # MCP tools are translated from the pool's discovered tool surface.
    # Operations are advertised directly from the in-process catalog so the
    # model sees their real Parameters schema (not an arguments_json wrapper
    # from a subprocess). Both are dispatched in-process: MCP tools via
    # pool.call_tool, operations via hooks.execute_operation.
    mcp_tools = await pool.list_tools()
    # Apply mode-level tool allowlist.
    # When tools_authoritative is True, empty allowed_tool_ids means no
    # tools are advertised. When False (default), empty means all pass.
    if mode_config and (mode_config.allowed_tool_ids or mode_config.tools_authoritative):
        allowed = set(mode_config.allowed_tool_ids)
        mcp_tools = [t for t in mcp_tools if t.name in allowed]
    tool_schemas: list[dict[str, Any]] = [
        mcp_tool_to_openai_function(
            name=spec.name,
            description=spec.description,
            input_schema=spec.input_schema,
        )
        for spec in mcp_tools
    ]
    operation_catalog = get_operation_catalog()
    tool_catalog = get_tool_catalog()
    tool_schemas.extend(
        build_operation_tool_schemas(
            mode_id,
            tools_authoritative=bool(
                mode_config and mode_config.tools_authoritative
            ),
        )
    )
    # Advertise in-process (pack-contributed) tools alongside MCP tools and
    # operations. Pack tools are dispatched in-process via execute_tool, not
    # through the MCP pool.
    tool_schemas.extend(_build_in_process_tool_schemas(mode_config))

    # Workflow modes: advertise the transition_stage tool so the model can
    # advance the FSM. The turn loop intercepts this tool call before the
    # operation/tool/MCP dispatch branches and calls apply_transition.
    is_workflow = (
        mode_config is not None
        and mode_config.mode_kind == "workflow"
        and hooks.workflow_session is not None
    )
    if is_workflow:
        tool_schemas.extend(get_transition_tool_schema())

    # Resolve preferred model
    preferred_model_id = (
        preferred_model_id.strip()
        if isinstance(preferred_model_id, str) and preferred_model_id.strip()
        else mode_config.default_model
        if mode_config and mode_config.default_model
        else "default"
    )
    model_def = resolve_model(preferred_model_id)

    content = ""
    transitioned_to: str | None = None
    workflow_terminated = False
    total_tool_calls = 0
    iteration = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    total_cached_tokens = 0
    total_reasoning_tokens = 0
    total_duration_ms = 0.0
    total_latency_ms = 0

    def _budget_forces_final() -> bool:
        """Return True when the turn budget is exhausted and the loop must
        force a final no-tools completion.

        A budget field set to 0 means unlimited and never forces. When any
        nonzero field is exhausted, the model gets one more call — but with
        no tools advertised so it must produce a final text response.
        """
        if turn_budget is None:
            return False
        if turn_budget.max_tool_calls and total_tool_calls >= turn_budget.max_tool_calls:
            return True
        if turn_budget.max_total_tokens and total_tokens >= turn_budget.max_total_tokens:
            return True
        if turn_budget.max_wall_ms:
            elapsed_ms = int((time.monotonic() - turn_start_mono) * 1000)
            if elapsed_ms >= turn_budget.max_wall_ms:
                return True
        return False

    # Per-iteration model selection (item 5). When the host wires
    # ``hooks.model_policy``, the loop consults it each iteration instead of
    # resolving one model for the whole turn. The policy can upgrade or
    # downgrade the model based on whether tools have run and how much budget
    # remains. This lets a host express "cheap model routes, expensive model
    # synthesizes" without the operation indirection.
    def _resolve_iteration_model(iteration_num: int) -> Any:
        policy = hooks.context_seams.model_policy
        choose_fn = getattr(policy, "choose", None)
        if callable(choose_fn):
            chosen_id = policy.choose(  # type: ignore[attr-defined]
                default_model_id=preferred_model_id,
                iteration=iteration_num,
                tool_calls_so_far=total_tool_calls,
                budget_remaining=(
                    None
                    if turn_budget is None
                    else max(0, turn_budget.max_tool_calls - total_tool_calls)
                    if turn_budget.max_tool_calls
                    else None
                ),
            )
            if isinstance(chosen_id, str) and chosen_id.strip():
                return resolve_model(chosen_id.strip())
        return model_def

    while iteration < max_tool_iterations:
        iteration += 1

        # Stuck-session guard: if the session has been in the same stage
        # for too many turns, inject a nudge. After the cap, strip tools to
        # force a final response and stop burning tokens.
        if is_workflow and hooks.workflow_session is not None:
            wf_session = hooks.workflow_session()
            if wf_session is not None:
                turns_in_stage = wf_session.turn_count
                if turns_in_stage >= _STAGE_NUDGE_CAP:
                    _log.warning(
                        "workflow mode=%s stuck in stage=%s for %d turns; forcing final",
                        mode_id,
                        wf_session.stage_id,
                        turns_in_stage,
                    )
                    budget_exhausted = True
                elif turns_in_stage >= _STAGE_NUDGE_THRESHOLD:
                    messages.append({
                        "role": "user",
                        "content": (
                            f"You have been in the '{wf_session.stage_id}' stage "
                            f"for {turns_in_stage} turns. If you have enough evidence, "
                            f"call transition_stage to advance. Otherwise, ask the user "
                            f"a clarifying question."
                        ),
                    })

        # Budget enforcement: when exhausted, strip tools so the model must
        # produce a final text response. This is the "ratio" stop condition.
        budget_exhausted = _budget_forces_final()
        effective_tools = None if budget_exhausted else (tool_schemas if tool_schemas else None)
        if budget_exhausted:
            _log.debug(
                "turn budget exhausted (mode=%s, tool_calls=%d, tokens=%d); forcing final",
                mode_id,
                total_tool_calls,
                total_tokens,
            )
        effective_model_def = _resolve_iteration_model(iteration)
        if hooks.complete_stream is not None:
            # ── Streaming path ──────────────────────────────────────────
            # v1: stream tokens to the host. When the stream indicates
            # tool_calls are needed (finish_reason='tool_calls'), fall
            # back to the non-streaming complete() for that iteration to
            # get the structured tool-call payload.
            chunks: list[StreamedChunk] = []
            async for chunk in hooks.complete_stream(
                messages=messages,
                model_def=effective_model_def,
                max_tokens=4096,
                temperature=0.4,
                tools=effective_tools,
            ):
                chunks.append(chunk)
            aggregated = _aggregate_streamed_chunks(chunks)
            if aggregated.finish_reason == "tool_calls":
                # Fall back to non-streaming for tool-call payload
                completion = await complete(
                    messages=messages,
                    model_def=effective_model_def,
                    max_tokens=4096,
                    temperature=0.4,
                    tools=effective_tools,
                )
            else:
                # Build a completion-like object from the aggregated stream
                completion = StreamedCompletionAdapter(aggregated)
        else:
            completion = await complete(
                messages=messages,
                model_def=effective_model_def,
                max_tokens=4096,
                temperature=0.4,
                tools=effective_tools,
            )
        (
            prompt_tokens,
            completion_tokens,
            completion_total_tokens,
            cached_tokens,
            reasoning_tokens,
            duration_ms,
            latency_ms,
        ) = _extract_completion_usage(completion)
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        total_tokens += completion_total_tokens
        total_cached_tokens += cached_tokens
        total_reasoning_tokens += reasoning_tokens
        total_duration_ms += duration_ms
        total_latency_ms += latency_ms

        assistant_msg: dict[str, Any] = {"role": "assistant"}
        if getattr(completion, "content", None):
            assistant_msg["content"] = completion.content
        tool_calls = getattr(completion, "tool_calls", None) or []
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        if not tool_calls:
            content = getattr(completion, "content", "") or ""
            content = content.strip()
            break

        # Dispatch tool calls — MCP tools via pool, operations in-process.
        tools_dispatched = False
        for tool_call in tool_calls:
            tool_name, tool_arguments = openai_tool_call_to_mcp(tool_call)
            tool_call_id = str(tool_call.get("id") or "")

            # Per-target ``max_calls_per_turn`` enforcement (item 2). The
            # ledger is the single enforcement point across operations and
            # pack tools. MCP tools have no catalog definition, so they are
            # bounded only by the global iteration count.
            target_def = operation_catalog.get(tool_name) or tool_catalog.get(tool_name)
            if target_def is not None:
                max_calls = getattr(target_def, "max_calls_per_turn", 0)
                if ledger.is_exhausted(tool_name, max_calls):
                    tool_output = (
                        f"Tool '{tool_name}' has reached its per-turn call limit "
                        f"({max_calls})."
                    )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
                        "content": tool_output,
                    })
                    continue

            tools_dispatched = True
            try:
                # Transition interception: the model called transition_stage.
                # This is handled in-loop, not through the operation/tool/MCP
                # dispatch branches. The kernel validates the transition
                # against the current stage's allowed_next_stages, updates
                # the session, and notifies the host via
                # workflow_session_updater.
                if tool_name == _TRANSITION_TOOL_NAME and is_workflow:
                    target_stage = str(tool_arguments.get("target_stage_id") or "").strip()
                    rationale = str(tool_arguments.get("rationale") or "").strip()
                    missing = tool_arguments.get("required_missing_inputs") or []
                    if not target_stage:
                        tool_output = "transition_stage requires target_stage_id."
                    else:
                        wf_session_fn = hooks.workflow_session
                        current_session = wf_session_fn() if wf_session_fn is not None else None
                        if current_session is None:
                            tool_output = "No active workflow session; cannot transition."
                        else:
                            try:
                                transition_req = WorkflowTransitionRequest(
                                    target_stage_id=target_stage,
                                    rationale=rationale,
                                    required_missing_inputs=(
                                        list(missing) if isinstance(missing, list) else []
                                    ),
                                )
                                updated_session = apply_transition(
                                    current_session, transition_req
                                )
                                if hooks.workflow_session_updater is not None:
                                    hooks.workflow_session_updater(updated_session)
                                transitioned_to = updated_session.stage_id
                                tool_output = (
                                    f"Transitioned to stage: {updated_session.stage_id}."
                                    + (f" Rationale: {rationale}" if rationale else "")
                                )
                            except Exception as trans_exc:  # pylint: disable=broad-except
                                tool_output = (
                                    f"Transition failed: "
                                    f"{type(trans_exc).__name__}: {trans_exc}"
                                )
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "name": tool_name,
                        "content": tool_output,
                    })
                    continue

                if tool_name in operation_catalog:
                    # Operation: dispatch in-process via the host's
                    # execute_operation hook, which injects all host
                    # callbacks (complete_fn, resolve_model_fn,
                    # execute_tool_fn, introspect_fn, etc.).
                    op_ctx = ctx.model_copy(
                        update={"operation_id": tool_name, "params": tool_arguments}
                    )
                    op_result = await exec_op(tool_name, op_ctx)
                    tool_output = _stringify_result(op_result)
                elif tool_name in tool_catalog:
                    # In-process tool (pack-contributed): dispatch via the
                    # tool registry, which imports the implementation module
                    # and calls execute(ctx) under a timeout.
                    tool_def = tool_catalog[tool_name]
                    if tool_def.implementation_module is not None:
                        tool_ctx = _to_tool_context(tool_name, ctx)
                        tool_ctx.parameters = tool_arguments
                        tool_result = await execute_tool(tool_name, tool_ctx)
                        tool_output = tool_result.output_text or (
                            _serialize_tool_json(tool_result) if tool_result.output_json else "Done"
                        )
                    else:
                        mcp_result = await pool.call_tool(tool_name, tool_arguments)
                        tool_output = mcp_result.output_text or "Done"
                else:
                    mcp_result = await pool.call_tool(tool_name, tool_arguments)
                    tool_output = mcp_result.output_text or "Done"
                total_tool_calls += 1
                ledger.record(tool_name)
            except Exception as exc:  # pylint: disable=broad-except
                _log.warning("Tool '%s' failed in turn runner: %s", tool_name, exc)
                tool_output = f"Tool execution failed: {type(exc).__name__}: {exc}"

            if not tool_output.strip():
                tool_output = "Done"

            tool_result_msg: dict[str, Any] = {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": str(tool_output),
            }
            # Preserve thought_signature for Gemini thinking mode round-trips.
            # Google requires it inside the function object ("functionCall parts").
            thought_sig = tool_call.get("thought_signature")
            if thought_sig:
                tool_result_msg["function"] = {"thought_signature": thought_sig}
            messages.append(tool_result_msg)

        if not tools_dispatched:
            # No tool calls were dispatched — model returned text
            content = getattr(completion, "content", "") or ""
            content = content.strip()
            break

    if not content:
        content = "The model did not produce a final response."

    # Terminal-stage handling: if the model transitioned to a terminal
    # stage during this turn, force one final no-tools completion using the
    # terminal stage's instruction so the model produces a closing summary.
    if transitioned_to is not None and is_workflow:
        terminal_session = hooks.workflow_session() if hooks.workflow_session else None
        if terminal_session is not None and is_terminal_stage(terminal_session):
            terminal_stage_config = mode_config.stages.get(terminal_session.stage_id)
            if terminal_stage_config and terminal_stage_config.instruction_body:
                terminal_prompt = assemble_system(
                    mode_instruction,
                    terminal_stage_config.instruction_body,
                )
                messages.append({
                    "role": "user",
                    "content": "Provide a closing summary for this workflow.",
                })
                messages[0] = {"role": "system", "content": terminal_prompt}
                try:
                    final_completion = await complete(
                        messages=messages,
                        model_def=effective_model_def,
                        max_tokens=4096,
                        temperature=0.4,
                        tools=None,
                    )
                    final_content = getattr(final_completion, "content", "") or ""
                    if final_content.strip():
                        content = final_content.strip()
                except Exception as exc:  # pylint: disable=broad-except
                    _log.warning("terminal-stage summary failed: %s", exc)
            workflow_terminated = True

    # ── Phase 3: Post-turn ────────────────────────────────────────────
    # Output guard runs before on_complete/store_mem so the persisted and
    # delivered text is the sanitized/safe-fallback version. Blocked
    # outputs emit a guardrail.audit event and skip the post-turn slice.
    output_guard_result = await hooks.context_seams.output_guard.validate(content, guard_surface)
    turn_blocked = False
    if not output_guard_result.allowed:
        original_len = len(content)
        safe_fallback = "I'm unable to provide that response. Please rephrase your request."
        _log.warning(
            "Output guard blocked turn for mode=%s (original_len=%d)",
            mode_id,
            original_len,
        )
        try:
            await hooks.context_seams.event_bus.publish(
                "guardrail.audit",
                {
                    "surface_id": guard_surface.surface_id,
                    "mode_id": mode_id,
                    "blocked_reason": output_guard_result.blocked_reason or "",
                    "original_text_len": original_len,
                },
            )
        except Exception as audit_exc:  # pylint: disable=broad-except
            _log.warning("guardrail.audit publish failed: %s", audit_exc)
        content = safe_fallback
        turn_blocked = True
    elif output_guard_result.sanitized_text is not None:
        content = output_guard_result.sanitized_text

    await on_complete(mode_id, query, content, messages)

    # ── Phase 3.5: Post-turn opportunistic slice ─────────────────────
    # Fires after the response is committed. The user has already seen
    # the answer; this slice consumes the deferred pressure budget.
    # Skipped when the output guard blocked the turn — abnormal turns do
    # not run opportunistic house-keeping.
    if not turn_blocked:
        post_turn_outcome = await _fire_slice(
            hooks=hooks,
            trigger=SliceMode.POST_TURN,
            slice_outcomes=slice_outcomes,
        )
        if post_turn_outcome is not None:
            hooks.last_slice_finished_at = time.monotonic()

    return TurnResult(
        response_text=content,
        tool_call_count=total_tool_calls,
        iteration_count=iteration,
        prompt_tokens=total_prompt_tokens,
        completion_tokens=total_completion_tokens,
        total_tokens=total_tokens,
        cached_tokens=total_cached_tokens,
        reasoning_tokens=total_reasoning_tokens,
        duration_ms=total_duration_ms,
        latency_ms=total_latency_ms,
        slice_outcomes=slice_outcomes,
        blocked=turn_blocked,
        transitioned_to=transitioned_to,
        workflow_terminated=workflow_terminated,
    )


def _build_minimal_context(
    *,
    mode_id: str,
    query: str,
    resolve_model_fn: Any,
    complete_fn: Any,
    execute_tool_fn: Any,
    execute_operation_fn: Any,
    workflow_turn_runner: Any = None,
    workflow_session_factory: Any = None,
    state_provider: Any = None,
    build_workshop_context_bundle_fn: Any = None,
) -> OperationExecutionContext:
    """Build a minimal OperationExecutionContext for the turn runner."""
    return OperationExecutionContext(
        workspace_id="lite-host",
        user_id="lite-user",
        user_slug="lite-user",
        user_name="Lite User",
        conversation_id="lite-host-console",
        surface_type="personal_dm",
        surface_id="lite-host-console",
        memory_scope="personal",
        mode_id=mode_id,
        query=query,
        model_id="default",
        complete_fn=complete_fn,
        resolve_model_fn=resolve_model_fn,
        execute_tool_fn=execute_tool_fn,
        execute_operation_fn=execute_operation_fn,
        workflow_turn_runner=workflow_turn_runner,
        workflow_session_factory=workflow_session_factory,
        state_provider=state_provider,
        build_workshop_context_bundle_fn=build_workshop_context_bundle_fn,
    )


def _stringify_result(result: OperationResult) -> str:
    """Convert an operation result into plain text for model consumption."""
    output_text = getattr(result, "output_text", "") or ""
    output_json = getattr(result, "output_json", None) or {}
    if output_text.strip():
        return output_text.strip()
    if output_json:
        import json as _json

        return _json.dumps(output_json)
    return "Done"


def _serialize_tool_json(result: ToolResult) -> str:
    """Serialize a ToolResult's output_json for model consumption."""
    import json as _json

    return _json.dumps(result.output_json)


def _to_tool_context(tool_id: str, ctx: Any) -> ToolExecutionContext:
    """Build a ToolExecutionContext from an OperationExecutionContext or similar.

    Carries the ambient identity/scope fields so pack-contributed tools can
    use them without a host-specific context object.
    """
    return ToolExecutionContext(
        tool_id=tool_id,
        workspace_id=getattr(ctx, "workspace_id", "default"),
        user_id=getattr(ctx, "user_id", ""),
        user_slug=getattr(ctx, "user_slug", "default"),
        surface_type=getattr(ctx, "surface_type", "personal_dm"),
        surface_id=getattr(ctx, "surface_id", ""),
        memory_scope=getattr(ctx, "memory_scope", "personal"),
        mode_id=getattr(ctx, "mode_id", ""),
        query=getattr(ctx, "query", ""),
        parameters=getattr(ctx, "parameters", {}) or {},
        event_bus=getattr(ctx, "event_bus", None),
        state_provider=getattr(ctx, "state_provider", None),
    )


def _build_in_process_tool_schemas(
    mode_config: Any,
) -> list[dict[str, Any]]:
    """Build OpenAI tool schemas for in-process (pack-contributed) tools.

    Advertises tools from the internal tool catalog (discovered from packs)
    that are allowed for the current mode. Mirrors
    ``build_operation_tool_schemas`` but for the tool catalog.
    """
    catalog = get_tool_catalog()
    if not catalog:
        return []

    allowed_tool_ids: set[str] | None = None
    if mode_config and (mode_config.allowed_tool_ids or mode_config.tools_authoritative):
        allowed_tool_ids = set(mode_config.allowed_tool_ids)

    schemas: list[dict[str, Any]] = []
    for tool_id, definition in catalog.items():
        if allowed_tool_ids is not None and tool_id not in allowed_tool_ids:
            continue
        if definition.mode_allowlist is not None and mode_config:
            if mode_config.mode_id not in definition.mode_allowlist:
                continue
        schemas.append(
            build_strict_openai_function_tool(
                name=tool_id,
                description=definition.description or f"Execute {tool_id}.",
                parameters_schema=definition.parameters_schema,
            )
        )
    return schemas


def _coerce_non_negative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _coerce_non_negative_float(value: Any) -> float:
    try:
        return max(0.0, float(value or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _extract_completion_usage(
    completion: Any,
) -> tuple[int, int, int, int, int, float, int]:
    """Extract token, sub-counter, duration, and latency counts from one completion.

    Returns a 7-tuple ``(prompt_tokens, completion_tokens, total_tokens,
    cached_tokens, reasoning_tokens, duration_ms, latency_ms)``.

    The function reads both the flat attributes some hosts set on the
    completion object (``prompt_tokens`` etc.) and the nested ``usage``
    object the kernel gateways populate. Sub-counters (``cached_tokens``,
    ``reasoning_tokens``) and ``duration_ms`` are read defensively so a
    missing field never raises; they default to 0 when the provider did
    not surface them.
    """

    prompt_tokens = _coerce_non_negative_int(getattr(completion, "prompt_tokens", 0))
    completion_tokens = _coerce_non_negative_int(getattr(completion, "completion_tokens", 0))
    total_tokens = _coerce_non_negative_int(getattr(completion, "total_tokens", 0))
    cached_tokens = _coerce_non_negative_int(getattr(completion, "cached_tokens", 0))
    reasoning_tokens = _coerce_non_negative_int(getattr(completion, "reasoning_tokens", 0))
    duration_ms = _coerce_non_negative_float(getattr(completion, "duration_ms", 0.0))
    latency_ms = _coerce_non_negative_int(getattr(completion, "latency_ms", 0))

    usage = getattr(completion, "usage", None)
    usage_prompt = 0
    usage_completion = 0
    usage_total = 0
    usage_cached = 0
    usage_reasoning = 0
    if isinstance(usage, dict):
        usage_prompt = _coerce_non_negative_int(
            usage.get("prompt_tokens") or usage.get("input_tokens")
        )
        usage_completion = _coerce_non_negative_int(
            usage.get("completion_tokens") or usage.get("output_tokens")
        )
        usage_total = _coerce_non_negative_int(usage.get("total_tokens"))
        usage_cached = _coerce_non_negative_int(usage.get("cached_tokens"))
        usage_reasoning = _coerce_non_negative_int(usage.get("reasoning_tokens"))
    elif usage is not None:
        usage_prompt = _coerce_non_negative_int(
            getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None)
        )
        usage_completion = _coerce_non_negative_int(
            getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None)
        )
        usage_total = _coerce_non_negative_int(getattr(usage, "total_tokens", None))
        usage_cached = _coerce_non_negative_int(getattr(usage, "cached_tokens", None))
        usage_reasoning = _coerce_non_negative_int(getattr(usage, "reasoning_tokens", None))

    prompt = prompt_tokens or usage_prompt
    completion_value = completion_tokens or usage_completion
    total_value = total_tokens or usage_total
    if total_value <= 0:
        total_value = prompt + completion_value
    cached = cached_tokens or usage_cached
    reasoning = reasoning_tokens or usage_reasoning

    return prompt, completion_value, total_value, cached, reasoning, duration_ms, latency_ms

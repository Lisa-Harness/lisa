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
    get_mode_system_intro,
    get_workflow_catalog,
    run_mode_entry_operations,
    run_mode_pre_turn_operations,
)
from kernel.models import (
    OperationExecutionContext,
    OperationResult,
    SurfaceContext,
    ToolResult,
)
from kernel.operations.registry import (
    execute_operation,
)
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
from kernel.tools.schema_translator import mcp_tool_to_openai_function, openai_tool_call_to_mcp

logger = logging.getLogger(__name__)

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


def _lazy_content_registry() -> ContentRegistry:
    from kernel.context import DiskContentRegistry

    return DiskContentRegistry()


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


def _lazy_memory_store() -> Any:
    return None


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

    content_registry: ContentRegistry = field(default_factory=_lazy_content_registry)
    model_policy: ModelPolicy = field(default_factory=_lazy_model_policy)
    provider_gateways: ProviderGatewayProvider = field(default_factory=_lazy_provider_gateways)
    event_bus: EventBus = field(default_factory=_lazy_event_bus)
    data_sanitizer: DataSanitizer = field(default_factory=_lazy_data_sanitizer)
    extension_evidence_builder: ExtensionEvidenceBuilder = field(
        default_factory=_lazy_extension_evidence_builder
    )
    memory_store: Any = field(default_factory=_lazy_memory_store)
    """Durable conversation memory. Default: None (memory is MCP-only)."""
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
    servers to run (e.g. ``lisa-context``, ``lisa-search``, ``lisa-operations``)
    and the kernel manages the subprocess lifecycle via ``MCPToolPool``.

    The kernel speaks MCP tool schema internally. At the provider edge,
    ``schema_translator`` converts to whatever the provider expects.
    """

    # ── Operation dispatch ────────────────────────────────────────────────
    execute_operation: Callable[..., Awaitable[OperationResult]] | None = None
    """Execute a named operation with an OperationExecutionContext. Default:
    delegates to `kernel.operations.registry.execute_operation`.

    Operations are internal kernel capabilities (pre-turn briefings, entry
    actions, declarative prompts). They are not advertised to the model as
    tools; if an operation should be LLM-callable, expose it through the
    ``lisa-operations`` MCP server.
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
    logger_override: logging.Logger | None,
) -> TurnResult:
    """Inner turn loop that owns the MCPToolPool lifecycle via the caller."""

    _log = logger_override or logger

    async def _mcp_tool_executor(tool_id: str, _ctx: Any) -> ToolResult:
        """Bridge OperationExecutionContext.execute_tool_fn to MCPToolPool.

        Declarative operations that need to call tools route through here.
        The MCP pool owns the actual dispatch; we convert MCPToolResult to
        ToolResult for compatibility with the operation execution contract.
        """

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
        result = await run_mode_entry_operations(mode_id, ctx)
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
    pre_result = await run_mode_pre_turn_operations(mode_id, ctx)
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

    # Assemble prompts
    mode_instruction = get_mode_system_intro(mode_id)
    system_prompt = assemble_system(mode_instruction, None)
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

    # Build tool schemas for this turn — MCP tools translated to OpenAI format.
    # Operations are no longer advertised as tools; they are internal kernel
    # capabilities. If an operation should be LLM-callable, expose it through
    # the lisa-operations MCP server.
    mcp_tools = await pool.list_tools()
    tool_schemas: list[dict[str, Any]] = [
        mcp_tool_to_openai_function(
            name=spec.name,
            description=spec.description,
            input_schema=spec.input_schema,
        )
        for spec in mcp_tools
    ]

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
    total_tool_calls = 0
    iteration = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    total_cached_tokens = 0
    total_reasoning_tokens = 0
    total_duration_ms = 0.0
    total_latency_ms = 0

    while iteration < max_tool_iterations:
        iteration += 1
        if hooks.complete_stream is not None:
            # ── Streaming path ──────────────────────────────────────────
            # v1: stream tokens to the host. When the stream indicates
            # tool_calls are needed (finish_reason='tool_calls'), fall
            # back to the non-streaming complete() for that iteration to
            # get the structured tool-call payload.
            chunks: list[StreamedChunk] = []
            async for chunk in hooks.complete_stream(
                messages=messages,
                model_def=model_def,
                max_tokens=4096,
                temperature=0.4,
                tools=tool_schemas if tool_schemas else None,
            ):
                chunks.append(chunk)
            aggregated = _aggregate_streamed_chunks(chunks)
            if aggregated.finish_reason == "tool_calls":
                # Fall back to non-streaming for tool-call payload
                completion = await complete(
                    messages=messages,
                    model_def=model_def,
                    max_tokens=4096,
                    temperature=0.4,
                    tools=tool_schemas if tool_schemas else None,
                )
            else:
                # Build a completion-like object from the aggregated stream
                completion = StreamedCompletionAdapter(aggregated)
        else:
            completion = await complete(
                messages=messages,
                model_def=model_def,
                max_tokens=4096,
                temperature=0.4,
                tools=tool_schemas if tool_schemas else None,
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

        # Dispatch MCP tool calls
        tools_dispatched = False
        for tool_call in tool_calls:
            tool_name, tool_arguments = openai_tool_call_to_mcp(tool_call)
            tool_call_id = str(tool_call.get("id") or "")

            tools_dispatched = True
            try:
                mcp_result = await pool.call_tool(tool_name, tool_arguments)
                tool_output = mcp_result.output_text or "Done"
                total_tool_calls += 1
            except Exception as exc:  # pylint: disable=broad-except
                _log.warning("MCP tool '%s' failed in turn runner: %s", tool_name, exc)
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

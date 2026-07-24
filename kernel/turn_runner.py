"""Kernel turn runner — the single model+tool loop.

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
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, cast

from kernel.approval import ApprovalInvocation, ApprovalRequired, render_approval_request
from kernel.capability_manifest import compile_capability_manifest
from kernel.context.models import ContextBudgetSnapshot
from kernel.ipsar_fsm import (
    apply_transition,
    get_mode_system_intro,
    get_required_mode_intro,
    get_required_stage_instruction,
    get_routing_targets,
    get_transition_tool_schema,
    get_workflow_catalog,
    is_fork_target,
    is_routable_mode,
    is_terminal_stage,
    parse_fork_target_string,
    run_mode_entry_operations,
    run_mode_pre_turn_operations,
)
from kernel.models import (
    CallLedger,
    CapabilityManifest,
    CheckpointBundle,
    ConversationSurface,
    ForkChildResult,
    ForkTarget,
    MemoryScope,
    OperationExecutionContext,
    OperationResult,
    SurfaceContext,
    ToolExecutionContext,
    ToolResult,
    TurnBudget,
    WorkflowModeConfig,
    WorkflowTransitionRequest,
)
from kernel.openai_function_tools import build_strict_openai_function_tool
from kernel.operations.registry import (
    build_operation_tool_schemas,
    execute_operation,
    get_operation_catalog,
)
from kernel.provider_core import StreamedChunk, StreamedCompletion, _aggregate_streamed_chunks
from kernel.seams.protocols import (
    ContentRegistry,
    DataSanitizer,
    EventBus,
    ExtensionEvidenceBuilder,
    InputGuard,
    ModelPolicy,
    ModelRouter,
    OutputGuard,
    ProviderGatewayProvider,
    RouterDecision,
    RouterRequest,
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
    handoff_to: str | None = None
    """When the model called ``transition_stage`` with a ``mode_id`` target
    (routing FSM handoff), the new ``mode_id``. The host reads this to update
    ``current_mode_id`` and persist the mode switch. None when no handoff
    occurred. See ``docs/ROUTING_RFC_HANDOFF.md``."""
    fork_results: list[ForkChildResult] | None = None
    """When the model called ``transition_stage`` with a ``fork:`` target,
    the collected results from all parallel child runs. None when no fork
    occurred. The host and subsequent stages read this to synthesize from
    the fan-out outputs."""
    workflow_terminated: bool = False
    """True when the transition moved the session to a terminal stage. The
    loop forces one final no-tools completion using the terminal stage's
    instruction as the system prompt so the model produces a closing summary."""
    pending_approval: ApprovalInvocation | None = None
    """The exact action that paused this turn, when an approval is required."""


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
    model_router: ModelRouter | None = None
    """Per-iteration model router. When ``None`` (default), the loop falls
    back to ``ModelPolicy.choose`` if present, then to the statically-resolved
    ``preferred_model_id`` — preserving today's behavior. Hosts wire a
    discovered ``ModelRouter`` (from ``lisa.extensions.model_routers``) here
    to enable per-iteration model selection without extra model calls."""
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


@dataclass(slots=True)
class CheckpointTuning:
    """Host-level knobs for automatic checkpoint creation.

    Checkpoints are turn-anchored, not wall-clock: automatic checkpoints
    fire in the post-turn phase (Phase 3.6), piggybacking on the existing
    turn cadence. No background scheduler, no cron. Stage transitions are
    the primary trigger (cheap, infrequent, high-value); every-N-turns is
    secondary and off by default.

    All fields have sensible defaults so a host can construct
    ``CheckpointTuning()`` and let the framework fill in the rest. The
    default is ``enabled=False`` — a host must opt in.
    """

    enabled: bool = False
    on_stage_transition: bool = True
    every_n_turns: int = 0
    explicit: bool = True

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None) -> CheckpointTuning:
        """Build a tuning from a host config dict. Unknown keys are ignored."""
        if not mapping:
            return cls()
        return cls(
            enabled=bool(mapping.get("enabled", False)),
            on_stage_transition=bool(mapping.get("on_stage_transition", True)),
            every_n_turns=int(mapping.get("every_n_turns", 0) or 0),
            explicit=bool(mapping.get("explicit", True)),
        )


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

    assemble_user_context: Callable[[str, str, str, CapabilityManifest | None], str] | None = None
    """Given mode_id: str, query: str, pre_turn_briefing: str, and an optional
    CapabilityManifest, return additional user-context content injected before
    the query in the final user message.  Default: empty string.

    The manifest is compiled from the live tool schemas for this turn — it is
    the ground truth for what the model can actually call. Hosts should use it
    to render a capability context that matches the injected function schema,
    eliminating the "Tooling Paradox" where self-knowledge diverges from the
    real tool surface.

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
    need cached runtime state can access it. Default: None."""

    pack_configuration_resolver: Any | None = None
    """Session-owned resolver for immutable, target-pack configuration."""

    approval_service: Any | None = None
    """Host-owned persistence and receipt service for exact-action approval."""

    code_change_artifact_store: Any | None = None
    """Host-owned store for pack-neutral local coding workflow artifacts."""

    last_context_snapshot: ContextBudgetSnapshot | None = None
    """Most recent ``ContextBudgetSnapshot`` produced by the host's
    context-mapper pass on this turn. Mutated by the host (or its
    ``assemble_user_context``) after each prompt assembly so slice
    providers can read actual context-window utilization without
    re-running the mapper. Default: None (no snapshot; utilization-aware
    slices treat this as "unknown" and no-op). Read-only from the
    kernel's perspective — the kernel never writes this field."""

    write_turn_history_fn: Callable[[list[dict[str, Any]]], None] | None = None
    """Host-owned write-back seam for the live session's turn history.
    Receives a fully-formed replacement turn-history list (typically a
    single summary record followed by retained recent turns) and is
    responsible for persisting it to the session's thread store. Default:
    None means "host does not support turn-history write-back";
    compaction slices and the ``/summarize-history`` command detect this
    and no-op with a warning instead of silently dropping the result."""

    workspace_id: str = "lite-host"
    user_id: str = "lite-user"
    user_slug: str = "lite-user"
    user_name: str = "Lite User"
    principal_key: str = "lite-user"
    conversation_id: str = "lite-host-console"
    surface_type: str = "personal_dm"
    surface_id: str = "lite-host-console"
    memory_scope: str = "personal"
    session_plan_digest: str = ""

    # ── Live introspection ─────────────────────────────────────────────────
    introspect: Any | None = None
    """Async callable returning an ``IntrospectSnapshot`` from the host's
    live runtime (process memory, asyncio tasks, MCP pool health, token
    usage, log tail). When set, injected into
    ``OperationExecutionContext.introspect_fn`` so the self-model operation
    and the reflect_self operation can access live runtime state. Default:
    None (self-model telemetry fields stay empty)."""

    # ── Checkpoints (turn-anchored, not wall-clock) ──────────────────────
    checkpoint_tuning: CheckpointTuning | None = None
    """``CheckpointTuning`` instance that controls automatic checkpoint
    creation. Default: ``None`` means "no automatic checkpoints." A host
    opts in by setting ``CheckpointTuning(enabled=True, ...)``. Explicit
    checkpoints (``/checkpoint <label>``) are always available when the
    ``on_checkpoint`` hook is wired, regardless of this tuning."""

    on_checkpoint: Callable[[CheckpointBundle], Awaitable[None]] | None = None
    """Called after a qualifying turn (stage transition or every-N-turns
    boundary) with a ``CheckpointBundle`` built from the current
    ``UnifiedState`` and ``WorkflowSessionState``. The host persists it —
    typically by calling ``save_checkpoint`` on ``lisa-storage`` through
    the MCP tool pool. Default: None (no checkpointing).

    Checkpoint failures (exceptions in this hook) are caught and logged;
    they do not break the turn. The kernel builds the bundle; the host
    owns persistence and turn-history population."""

    # ── Live stage telemetry ─────────────────────────────────────────────
    on_stage: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None
    """Async callable invoked at key points during the turn so the host
    transport can render a frequently-updated, detailed stage line.

    The kernel emits ``(label, data)`` pairs at:
      - ``"model_call"`` / ``"model_call_complete"`` — around each model
        completion (carries ``mode_id``, ``model_id``, ``provider``,
        ``iteration``, ``tool_calls_so_far``).
      - ``"tool_dispatch"`` / ``"tool_dispatch_complete"`` — around each
        tool/operation dispatch (carries ``tool``, ``tool_kind`` one of
        ``"operation"|"pack_tool"|"mcp_tool"|"transition"``).
      - ``"workflow_stage"`` — when the workflow session's stage is read
        (carries ``stage_id``, ``turn_count``).
      - ``"fork_start"`` / ``"fork_child_start"`` /
        ``"fork_child_complete"`` / ``"fork_complete"`` — around the fork
        fan-out, one event per child so a transport can show a line per
        active call.

    Failures in this hook are caught and logged; they never break the
    turn. Default: None (no stage telemetry; the host's own
    ``complete_live`` stage callbacks still fire separately when the
    provider streaming mode is ``stage``/``hybrid``)."""


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
    _capability_manifest: CapabilityManifest | None = None,
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


# ── Checkpoint helpers ──────────────────────────────────────────────────────


def _should_checkpoint(
    *,
    tuning: CheckpointTuning | None,
    transitioned_to: str | None,
    is_workflow: bool,
    hooks: TurnHooks,
) -> bool:
    """Return True when this turn qualifies for an automatic checkpoint.

    Rules (all require ``tuning.enabled`` and ``hooks.on_checkpoint`` wired):
    - Stage transition: ``transitioned_to is not None`` and
      ``tuning.on_stage_transition`` is True.
    - Every-N-turns: ``tuning.every_n_turns > 0`` and the current workflow
      session's ``turn_count`` is a positive multiple of ``every_n_turns``.
    """
    if tuning is None or not tuning.enabled:
        return False
    if transitioned_to is not None and tuning.on_stage_transition:
        return True
    if tuning.every_n_turns > 0 and is_workflow and hooks.workflow_session is not None:
        session = hooks.workflow_session()
        if session is not None:
            count = getattr(session, "turn_count", 0)
            if count > 0 and count % tuning.every_n_turns == 0:
                return True
    return False


def _build_checkpoint_bundle(
    *,
    mode_id: str,
    transitioned_to: str | None,
    is_workflow: bool,
    hooks: TurnHooks,
) -> CheckpointBundle | None:
    """Build a CheckpointBundle from the current host state.

    Reads ``UnifiedState`` via ``hooks.state_provider`` (if wired) and
    ``WorkflowSessionState`` via ``hooks.workflow_session`` (if wired).
    Returns None when neither is available — there's nothing to checkpoint.
    """
    import uuid
    from datetime import UTC, datetime

    unified_state_dict: dict[str, Any] = {}
    if hooks.state_provider is not None:
        state = hooks.state_provider()
        if state is not None:
            try:
                unified_state_dict = state.model_dump(mode="json")
            except Exception as exc:  # pylint: disable=broad-except
                _log.warning("checkpoint: state_provider dump failed: %s", exc)

    workflow_session_dict: dict[str, Any] = {}
    stage_id = ""
    turn_count = 0
    if is_workflow and hooks.workflow_session is not None:
        session = hooks.workflow_session()
        if session is not None:
            try:
                workflow_session_dict = session.model_dump(mode="json")
            except Exception as exc:  # pylint: disable=broad-except
                _log.warning("checkpoint: workflow_session dump failed: %s", exc)
            stage_id = str(getattr(session, "stage_id", "") or "")
            turn_count = int(getattr(session, "turn_count", 0) or 0)

    if not unified_state_dict and not workflow_session_dict:
        return None

    return CheckpointBundle(
        checkpoint_id=uuid.uuid4().hex[:16],
        created_at=datetime.now(UTC).isoformat(),
        mode_id=mode_id,
        stage_id=stage_id or (transitioned_to or ""),
        turn_count=turn_count,
        unified_state=unified_state_dict,
        workflow_session_state=workflow_session_dict,
    )


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
    max_tool_iterations: int = 15,
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
            the host when routing needs an explicit model designation
            (e.g. ``"deep"`` for ``/deep``).
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
        return await _run_turn_with_approval(
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
        return await _run_turn_with_approval(
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


async def _run_turn_with_approval(
    *,
    mode_id: str,
    query: str,
    hooks: TurnHooks,
    pool: MCPToolPool,
    resolve_model: Callable[[str], Any],
    complete: Callable[..., Awaitable[Any]],
    assemble_system: Callable[[str, str | None], str],
    assemble_user: Callable[[str, str, str, CapabilityManifest | None], str],
    exec_op: Callable[..., Awaitable[OperationResult]],
    on_complete: Callable[[str, str, str, list[dict[str, Any]]], Awaitable[None]],
    mode_config: Any,
    max_tool_iterations: int,
    preferred_model_id: str | None,
    turn_budget: TurnBudget | None,
    logger_override: logging.Logger | None,
) -> TurnResult:
    """Convert an approval interruption into a host-renderable turn result."""

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
    except ApprovalRequired as required:
        return TurnResult(
            response_text=render_approval_request(required.invocation),
            pending_approval=required.invocation,
        )


async def execute_approved_invocation(
    *,
    invocation: ApprovalInvocation,
    hooks: TurnHooks,
) -> ToolResult | OperationResult:
    """Execute one host-approved invocation without replaying a model turn.

    The registry consumes the receipt while rebuilding the same digest from
    the saved invocation, live session policy, and owner-scoped pack config.
    """

    if invocation.session_plan_digest != hooks.session_plan_digest:
        raise ValueError("Approval request belongs to a different session plan.")

    async def _dispatch(pool: MCPToolPool) -> ToolResult | OperationResult:
        async def _nested_tool_executor(tool_id: str, source_ctx: Any) -> ToolResult:
            tool_catalog = get_tool_catalog()
            definition = tool_catalog.get(tool_id)
            if definition is not None and definition.implementation_module is not None:
                tool_ctx = _to_tool_context(tool_id, source_ctx)
                tool_ctx.parameters = (
                    getattr(source_ctx, "parameters", None)
                    or getattr(source_ctx, "params", {})
                    or {}
                )
                return await execute_tool(tool_id, tool_ctx)
            mcp_result = await pool.call_tool(
                tool_id,
                getattr(source_ctx, "parameters", {}) or {},
            )
            return ToolResult(
                tool_id=tool_id,
                output_text=mcp_result.output_text,
                error=mcp_result.output_text if mcp_result.is_error else "",
            )

        identity = invocation.identity
        common: dict[str, Any] = {
            "workspace_id": identity.workspace_id,
            "user_id": identity.user_id,
            "user_slug": identity.user_slug,
            "user_name": hooks.user_name,
            "principal_key": hooks.principal_key,
            "conversation_id": hooks.conversation_id,
            "surface_type": cast(ConversationSurface, identity.surface_type),
            "surface_id": identity.surface_id,
            "memory_scope": cast(MemoryScope, identity.memory_scope),
            "mode_id": identity.mode_id,
            "query": "",
            "pack_configuration_resolver": hooks.pack_configuration_resolver,
            "approval_service": hooks.approval_service,
            "approval_request_id": invocation.request_id,
            "session_plan_digest": hooks.session_plan_digest,
            "code_change_artifact_store": hooks.code_change_artifact_store,
        }
        if invocation.target_kind == "tool":
            return await execute_tool(
                invocation.target_id,
                ToolExecutionContext(
                    tool_id=invocation.target_id,
                    parameters=invocation.parameters,
                    **common,
                ),
            )

        operation_context = OperationExecutionContext(
            operation_id=invocation.target_id,
            params=invocation.parameters,
            complete_fn=hooks.complete or _noop_complete,
            resolve_model_fn=hooks.resolve_model or _default_resolve_model,
            read_authored_text_fn=getattr(
                hooks.context_seams.content_registry,
                "read_authored_text",
                None,
            ),
            execute_tool_fn=_nested_tool_executor,
            execute_operation_fn=hooks.execute_operation or _default_execute_operation,
            state_provider=hooks.state_provider,
            **common,
        )
        execute_operation_fn = hooks.execute_operation or _default_execute_operation
        return await execute_operation_fn(invocation.target_id, operation_context)

    if hooks.mcp_tool_pool is not None:
        return await _dispatch(hooks.mcp_tool_pool)

    pool = MCPToolPool(hooks.mcp_tool_servers)
    await pool.start()
    try:
        return await _dispatch(pool)
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
    assemble_user: Callable[[str, str, str, CapabilityManifest | None], str],
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

    # ── Stage telemetry helper ──────────────────────────────────────────
    # Wraps the optional ``hooks.on_stage`` callback so the loop can fire
    # fine-grained stage events without each call site repeating the
    # None-check and the try/except. Failures are logged and swallowed — a
    # transport error never breaks the turn.
    _on_stage_hook = hooks.on_stage

    async def _emit_stage(label: str, data: dict[str, Any] | None = None) -> None:
        if _on_stage_hook is None:
            return
        try:
            await _on_stage_hook(label, data or {})
        except Exception as stage_exc:  # pylint: disable=broad-except
            _log.debug("on_stage(%s) hook failed: %s", label, stage_exc)

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
            read_authored_text_fn=getattr(
                hooks.context_seams.content_registry,
                "read_authored_text",
                None,
            ),
            execute_tool_fn=_mcp_tool_executor,
            execute_operation_fn=exec_op,
            state_provider=hooks.state_provider,
            pack_configuration_resolver=hooks.pack_configuration_resolver,
            approval_service=hooks.approval_service,
            code_change_artifact_store=hooks.code_change_artifact_store,
            workspace_id=hooks.workspace_id,
            user_id=hooks.user_id,
            user_slug=hooks.user_slug,
            user_name=hooks.user_name,
            principal_key=hooks.principal_key,
            conversation_id=hooks.conversation_id,
            surface_type=hooks.surface_type,
            surface_id=hooks.surface_id,
            memory_scope=hooks.memory_scope,
            session_plan_digest=hooks.session_plan_digest,
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
        read_authored_text_fn=getattr(
            hooks.context_seams.content_registry,
            "read_authored_text",
            None,
        ),
        execute_tool_fn=_mcp_tool_executor,
        execute_operation_fn=exec_op,
        state_provider=hooks.state_provider,
        pack_configuration_resolver=hooks.pack_configuration_resolver,
        approval_service=hooks.approval_service,
        code_change_artifact_store=hooks.code_change_artifact_store,
        workspace_id=hooks.workspace_id,
        user_id=hooks.user_id,
        user_slug=hooks.user_slug,
        user_name=hooks.user_name,
        principal_key=hooks.principal_key,
        conversation_id=hooks.conversation_id,
        surface_type=hooks.surface_type,
        surface_id=hooks.surface_id,
        memory_scope=hooks.memory_scope,
        session_plan_digest=hooks.session_plan_digest,
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
    #
    # Workflow modes require non-empty mode and stage instruction bodies —
    # an empty body would silently run the model modeless/stageless. Chat
    # and action modes use the lenient ``get_mode_system_intro`` which
    # returns "" for unknown modes (membership-check contract) so a chat
    # mode with no authored body can still rely on the base system prompt.
    is_workflow = mode_config is not None and mode_config.mode_kind == "workflow"
    if is_workflow:
        mode_instruction = get_required_mode_intro(mode_id)
    else:
        mode_instruction = get_mode_system_intro(mode_id)
    stage_instruction: str | None = None
    if is_workflow and hooks.workflow_session is not None:
        session = hooks.workflow_session()
        if session is not None and session.stage_id in mode_config.stages:  # type: ignore[union-attr]
            stage_instruction = get_required_stage_instruction(mode_id, session.stage_id)
    system_prompt = assemble_system(mode_instruction, stage_instruction)

    # Build tool schemas for this turn — MCP tools + in-process operations.
    # Done BEFORE assemble_user_context so the capability manifest reflects
    # exactly what the model will see this turn (closes the Tooling Paradox:
    # self-knowledge context now matches the injected function schema).
    # MCP tools are translated from the pool's discovered tool surface.
    # Operations are advertised directly from the in-process catalog so the
    # model sees their real Parameters schema (not an arguments_json wrapper
    # from a subprocess). Both are dispatched in-process: MCP tools via
    # pool.call_tool, operations via hooks.execute_operation.
    # Compute which cold servers could contribute allowed tools so we only
    # spawn servers the active mode can use. When the mode has no
    # allowed_tool_ids and is not authoritative (all tools pass), connect
    # every cold server as before. When authoritative with an allowlist,
    # only connect servers whose namespaced prefix appears in the allowlist.
    allowed_server_ids: set[str] | None = None
    if mode_config and mode_config.allowed_tool_ids:
        allowed_set = set(mode_config.allowed_tool_ids)
        allowed_server_ids = {
            c.server_id
            for c in getattr(pool, "configs", [])
            if any(t.startswith(c.server_id + ".") for t in allowed_set)
        }
    elif mode_config and mode_config.tools_authoritative and not mode_config.allowed_tool_ids:
        # Authoritative with empty allowlist = no tools; no cold servers needed.
        allowed_server_ids = set()

    mcp_tools = await pool.list_tools(allowed_server_ids=allowed_server_ids)
    # Apply mode-level tool allowlist.
    # When tools_authoritative is True (default), empty allowed_tool_ids
    # means no tools are advertised. When False, empty means all pass.
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
            tools_authoritative=bool(mode_config and mode_config.tools_authoritative),
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
    # Unified FSM routing: also advertise transition_stage for chat/action
    # modes that have routing targets (mode handoffs) or fork targets. This
    # lets the model hand off between modes mid-turn via the same mechanism
    # as workflow stage advancement. See docs/ROUTING_RFC_HANDOFF.md.
    has_routing_targets = bool(get_routing_targets(mode_id)) if mode_config else False
    if is_workflow or has_routing_targets:
        tool_schemas.extend(get_transition_tool_schema())

    # Apply one final model-facing prune pass so the payload sent to the
    # provider cannot include stale/duplicate/disallowed tool schemas.
    tool_schemas, tool_prune_stats = _prune_model_tool_schemas(
        tool_schemas=tool_schemas,
        mode_config=mode_config,
        is_workflow=is_workflow,
        has_routing_targets=has_routing_targets,
    )
    _log.info(
        "tool schema mode=%s in=%d out=%d dup=%d disallowed=%d invalid=%d "
        "is_workflow=%s has_routing=%s names=[%s]",
        mode_id,
        tool_prune_stats["input_count"],
        tool_prune_stats["output_count"],
        tool_prune_stats["dropped_duplicate"],
        tool_prune_stats["dropped_disallowed"],
        tool_prune_stats["dropped_invalid"],
        is_workflow,
        has_routing_targets,
        ", ".join(_tool_schema_name(s) for s in tool_schemas),
    )

    # Compile a deterministic capability manifest from the final tool payload
    # so the model's self-knowledge context matches what it can actually call.
    capability_manifest = compile_capability_manifest(tool_schemas, mode_config=mode_config)

    # Unified FSM routing: capability-manifest-aware proactive hint injection.
    # Compare the manifest against the user's query — if the current mode
    # lacks tools the query seems to need, inject a routing hint into the
    # system prompt. This is proactive (zero wasted iterations) and zero-cost
    # when the current mode can handle the request. See docs/ROUTING_RFC_HANDOFF.md.
    routing_hint = _compute_capability_routing_hint(
        mode_id=mode_id,
        query=query,
        capability_manifest=capability_manifest,
        routing_targets=get_routing_targets(mode_id) if mode_config else [],
        mode_config=mode_config,
    )
    if routing_hint:
        _log.info(
            "routing hint injected mode=%s hint_len=%d hint=%s",
            mode_id,
            len(routing_hint),
            routing_hint[:200],
        )
        system_prompt = system_prompt + "\n\n" + routing_hint
    else:
        _log.info(
            "no routing hint mode=%s (satisfies_intents=%s routing_targets=%s)",
            mode_id,
            getattr(mode_config, "satisfies_intents", None),
            get_routing_targets(mode_id) if mode_config else [],
        )

    user_context = assemble_user(mode_id, query, pre_briefing, capability_manifest)
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

    # Resolve preferred model
    preferred_model_id = (
        preferred_model_id.strip()
        if isinstance(preferred_model_id, str) and preferred_model_id.strip()
        else mode_config.default_model
        if mode_config and mode_config.default_model
        else "default"
    )
    model_def = resolve_model(preferred_model_id)

    # ── Capability gate ────────────────────────────────────────────────
    # When the mode declares ``required_capabilities`` (pack-authored
    # ``ModelCapability`` strings) and a model router is wired, validate the
    # resolved model's declared capabilities before the tool loop starts. If
    # the model is missing a required capability, return a clear user-facing
    # block instead of letting a weak model reach for write-effect tools it
    # cannot use reliably (e.g. gemma4:cloud calling ``coding_create_worktree``
    # for a "write a roadmap" request it can't actually complete). When no
    # router is wired, the gate is advisory (no block) — a host without a
    # router has no escalation path, so blocking would dead-end. The gate is
    # the safety net behind the router: the router's ``required_capabilities
    # includes "code"`` rule should have already escalated to a capable model;
    # the gate catches the cases where the router declined or returned None.
    _cap_block = _validate_model_capabilities(
        mode_config=mode_config,
        model_def=model_def,
        preferred_model_id=preferred_model_id,
        router_wired=hooks.context_seams.model_router is not None,
    )
    if _cap_block is not None:
        _log.warning(
            "capability gate blocked turn mode=%s model=%s missing=%s",
            mode_id,
            getattr(model_def, "model_id", preferred_model_id),
            _cap_block.missing,
        )
        return _cap_block.result

    content = ""
    transitioned_to: str | None = None
    handoff_to: str | None = None
    fork_results: list[ForkChildResult] | None = None
    workflow_terminated = False
    total_tool_calls = 0
    iteration = 0

    # Mutable turn-state read by ``_resolve_iteration_model`` so the router
    # can see what has happened so far this turn. ``last_tool_result_text``
    # is updated after each tool dispatch; ``required_capabilities`` and
    # ``roles_needed`` are derived once from the mode config and the
    # preferred-model resolution. They default to empty — a router that does
    # not need them ignores them.
    last_tool_result_text: str | None = None
    required_capabilities: tuple[str, ...] = ()
    if mode_config is not None:
        required_capabilities = tuple(getattr(mode_config, "required_capabilities", ()) or ())
    roles_needed: tuple[str, ...] = ()
    if isinstance(preferred_model_id, str) and preferred_model_id.strip():
        roles_needed = (preferred_model_id.strip(),)
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

    # Per-iteration model selection. Three-tier fallback:
    #   1. ``ContextSeams.model_router`` — the formal ``ModelRouter`` Protocol
    #      with a widened ``RouterRequest`` carrying query, last_tool_result,
    #      required_capabilities, mode_id, and roles_needed. This is the
    #      pluggable router path (``lisa.extensions.model_routers``).
    #   2. ``ModelPolicy.choose`` — the legacy duck-typed optional method with
    #      the narrow ``(default_model_id, iteration, tool_calls_so_far,
    #      budget_remaining)`` signature. Preserved for backward compatibility.
    #   3. Static ``model_def`` — the once-resolved ``preferred_model_id``.
    #      This is today's default behavior when no router and no ``choose``
    #      are wired.
    def _resolve_iteration_model(iteration_num: int) -> Any:
        router = hooks.context_seams.model_router
        if router is not None:
            req = RouterRequest(
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
                query=query,
                last_tool_result=last_tool_result_text,
                required_capabilities=required_capabilities,
                mode_id=mode_id,
                roles_needed=roles_needed,
            )
            try:
                decision = router.choose(req)
            except Exception:  # pylint: disable=broad-except
                _log.warning(
                    "model_router.choose failed (iteration=%d): %s",
                    iteration_num,
                    exc_info=True,
                )
                decision = None
            if isinstance(decision, RouterDecision) and decision.model_id.strip():
                resolved = resolve_model(decision.model_id.strip())
                if resolved is not None:
                    return resolved
                _log.warning(
                    "model_router returned unknown model_id %r; falling back",
                    decision.model_id,
                )
            elif isinstance(decision, str) and decision.strip():
                # Tolerate routers that return a bare model_id string.
                resolved = resolve_model(decision.strip())
                if resolved is not None:
                    return resolved
        # Legacy duck-typed ``choose`` on ModelPolicy.
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

    tools_dispatched_ever = False  # tracks if *any* tool was called this turn
    _narration_nudge_count = 0  # guards against infinite narration retries

    while iteration < max_tool_iterations:
        iteration += 1

        # Stuck-session guard: if the session has been in the same stage
        # for too many turns, inject a nudge. After the cap, strip tools to
        # force a final response and stop burning tokens.
        if is_workflow and hooks.workflow_session is not None:
            wf_session = hooks.workflow_session()
            if wf_session is not None:
                turns_in_stage = wf_session.turn_count
                await _emit_stage(
                    "workflow_stage",
                    {
                        "mode_id": mode_id,
                        "stage_id": wf_session.stage_id,
                        "turn_count": turns_in_stage,
                        "iteration": iteration,
                    },
                )
                if turns_in_stage >= _STAGE_NUDGE_CAP:
                    _log.warning(
                        "workflow mode=%s stuck in stage=%s for %d turns; forcing final",
                        mode_id,
                        wf_session.stage_id,
                        turns_in_stage,
                    )
                    budget_exhausted = True
                elif turns_in_stage >= _STAGE_NUDGE_THRESHOLD:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"You have been in the '{wf_session.stage_id}' stage "
                                f"for {turns_in_stage} turns. If you have enough evidence, "
                                f"call transition_stage to advance. Otherwise, ask the user "
                                f"a clarifying question."
                            ),
                        }
                    )

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
        await _emit_stage(
            "model_call",
            {
                "mode_id": mode_id,
                "model_id": getattr(effective_model_def, "model_id", ""),
                "provider": getattr(effective_model_def, "provider", ""),
                "iteration": iteration,
                "tool_calls_so_far": total_tool_calls,
                "tools_offered": bool(effective_tools),
            },
        )
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

        await _emit_stage(
            "model_call_complete",
            {
                "mode_id": mode_id,
                "model_id": getattr(effective_model_def, "model_id", ""),
                "iteration": iteration,
                "tool_calls": len(getattr(completion, "tool_calls", None) or []),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": completion_total_tokens,
                "duration_ms": duration_ms,
            },
        )

        assistant_msg: dict[str, Any] = {"role": "assistant"}
        if getattr(completion, "content", None):
            assistant_msg["content"] = completion.content
        tool_calls = getattr(completion, "tool_calls", None) or []
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        if hasattr(completion, "model"):
            assistant_msg["extra"] = {
                "provenance": {"source": "observed", "model": getattr(completion, "model", "")}
            }
        messages.append(assistant_msg)

        if not tool_calls:
            content = getattr(completion, "content", "") or ""
            content = content.strip()

            # ── Hallucinated-transition guard ──────────────────────────
            # Weak models (e.g. Gemma/GLM via Ollama) sometimes *narrate* a
            # stage transition or handoff in their text output ("I have
            # transitioned to the search stage…") instead of actually
            # calling the ``transition_stage`` tool. When a routing hint
            # was injected into the system prompt, detect transition-like
            # language and inject a correction nudge so the model retries
            # with the actual tool call. Without this guard the user sees
            # a false "I transitioned" message and no work is done.
            #
            # The guard fires when:
            #   - A routing hint was injected (the model was told to hand off)
            #   - The model has NOT yet called transition_stage this turn
            #     (``transitioned_to`` is still None)
            #   - The model has NOT yet dispatched any other tool
            #     (``tools_dispatched_ever`` is False — once the model does
            #     real work, a narrated transition in a summary is harmless)
            #   - We haven't exhausted the retry cap (prevents infinite
            #     loops with models that never learn to call the tool)
            #
            # After the cap, the guard gives up and lets the narrated
            # response through so the user sees something rather than a
            # silent hang.
            _NARRATION_NUDGE_CAP = 3
            if (
                routing_hint
                and transitioned_to is None
                and not tools_dispatched_ever
                and _narration_nudge_count < _NARRATION_NUDGE_CAP
            ):
                lowered = content.lower()
                transition_markers = (
                    "i have transitioned",
                    "i'll transition",
                    "i'll hand off",
                    "i will now transition",
                    "i am now transitioning",
                    "i will now hand off",
                    "i have moved to",
                    "i am now in the",
                    "moving to the",
                    "transitioning to the",
                )
                if any(m in lowered for m in transition_markers):
                    _narration_nudge_count += 1
                    _log.warning(
                        "model narrated a transition without calling "
                        "transition_stage; injecting correction nudge "
                        "(mode=%s, iteration=%d, nudge=%d/%d)",
                        mode_id,
                        iteration,
                        _narration_nudge_count,
                        _NARRATION_NUDGE_CAP,
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "You wrote that you are transitioning, but you "
                                "did not actually call the `transition_stage` "
                                "tool. A transition only happens when you call "
                                "the tool — writing about it does nothing. "
                                "Call `transition_stage` now with the "
                                "appropriate `target_stage_id`. If you cannot "
                                "or should not transition, respond directly to "
                                "the user without claiming a transition."
                            ),
                        }
                    )
                    continue

            break

        # Dispatch tool calls — MCP tools via pool, operations in-process.
        tools_dispatched = False
        for tool_call in tool_calls:
            tool_name, tool_arguments = openai_tool_call_to_mcp(tool_call)
            tool_call_id = str(tool_call.get("id") or "")

            # ── Routing-hint write-tool guard (defense in depth) ───────
            # When a routing hint was injected this turn (the kernel told the
            # model to hand off because the current mode lacks the tools the
            # query needs), a model that ignores the hint and instead calls a
            # write-effect tool from the current mode is almost always wrong
            # — e.g. a weak model in ``coding`` mode calling
            # ``coding_create_worktree`` for a "write a roadmap" request
            # instead of handing off to ``markdown``. Inject a correction
            # nudge (mirroring the hallucinated-transition guard) and skip
            # the dispatch so the model retries with ``transition_stage`` or
            # a read-only tool. Bounded by ``_NARRATION_NUDGE_CAP`` so a model
            # that never learns still gets to produce a final response.
            if (
                routing_hint
                and tool_name != _TRANSITION_TOOL_NAME
                and _narration_nudge_count < _NARRATION_NUDGE_CAP
            ):
                _hint_effect = _tool_effect(
                    tool_name,
                    operation_catalog=operation_catalog,
                    tool_catalog=tool_catalog,
                )
                if _hint_effect == "write" or _hint_effect == "external_write":
                    _narration_nudge_count += 1
                    _log.warning(
                        "model called write-effect tool %r after a routing "
                        "hint; injecting correction nudge (mode=%s, "
                        "iteration=%d, nudge=%d/%d)",
                        tool_name,
                        mode_id,
                        iteration,
                        _narration_nudge_count,
                        _NARRATION_NUDGE_CAP,
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"You called the write-effect tool `{tool_name}` "
                                "but a routing hint told you this mode lacks "
                                "the tools needed for this request. Do not "
                                "write from this mode — call `transition_stage` "
                                "with the suggested `target_stage_id` to hand "
                                "off, or use a read-only tool and respond "
                                "directly to the user."
                            ),
                        }
                    )
                    continue

            # Per-target ``max_calls_per_turn`` enforcement (item 2). The
            # ledger is the single enforcement point across operations and
            # pack tools. MCP tools have no catalog definition, so they are
            # bounded only by the global iteration count.
            from kernel.capability import resolve_capability_for_turn_limit

            target_def = resolve_capability_for_turn_limit(
                tool_name,
                operation_catalog=operation_catalog,
                tool_catalog=tool_catalog,
            )
            if target_def is not None:
                max_calls = getattr(target_def, "max_calls_per_turn", 0)
                if ledger.is_exhausted(tool_name, max_calls):
                    tool_output = (
                        f"Tool '{tool_name}' has reached its per-turn call limit ({max_calls})."
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "name": tool_name,
                            "content": tool_output,
                        }
                    )
                    continue

            tools_dispatched = True
            tools_dispatched_ever = True
            # Determine the tool kind for stage telemetry. Computed here so
            # the dispatch branches below stay unchanged; the value is only
            # read by ``_emit_stage``. ``"transition"`` is special-cased so
            # the transport can distinguish a workflow/routing/fork action
            # from a plain tool call.
            if tool_name == _TRANSITION_TOOL_NAME:
                _tool_kind = "transition"
            elif tool_name in operation_catalog:
                _tool_kind = "operation"
            elif tool_name in tool_catalog:
                _tool_kind = "pack_tool"
            else:
                _tool_kind = "mcp_tool"
            await _emit_stage(
                "tool_dispatch",
                {
                    "tool": tool_name,
                    "tool_kind": _tool_kind,
                    "iteration": iteration,
                    "tool_calls_so_far": total_tool_calls + 1,
                    "mode_id": mode_id,
                },
            )
            _tool_dispatch_started = time.monotonic()
            try:
                # Transition interception: the model called transition_stage.
                # This is handled in-loop, not through the operation/tool/MCP
                # dispatch branches. The kernel validates the transition
                # against the current stage's allowed_next_stages, updates
                # the session, and notifies the host via
                # workflow_session_updater.
                #
                # Unified FSM routing (see docs/ROUTING_RFC_HANDOFF.md):
                # transition_stage now handles three target kinds:
                #   1. fork:mode/stage,... — parallel fan-out to heterogeneous modes
                #   2. mode_id (routable) — mid-turn mode switch (handoff)
                #   3. workflow stage — existing apply_transition path
                if tool_name == _TRANSITION_TOOL_NAME:
                    target_stage = str(tool_arguments.get("target_stage_id") or "").strip()
                    rationale = str(tool_arguments.get("rationale") or "").strip()
                    missing = tool_arguments.get("required_missing_inputs") or []
                    if not target_stage:
                        tool_output = "transition_stage requires target_stage_id."
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "name": tool_name,
                                "content": tool_output,
                            }
                        )
                        await _emit_stage(
                            "tool_dispatch_complete",
                            {
                                "tool": tool_name,
                                "tool_kind": "transition",
                                "iteration": iteration,
                                "tool_calls_so_far": total_tool_calls,
                                "ok": False,
                                "transition": "missing_target",
                                "duration_ms": int(
                                    (time.monotonic() - _tool_dispatch_started) * 1000
                                ),
                            },
                        )
                        continue

                    # ── Case A: fork fan-out ──────────────────────────────
                    if is_fork_target(target_stage):
                        fork_targets_raw = parse_fork_target_string(target_stage)
                        fork_results_list: list[ForkChildResult] = []
                        if not fork_targets_raw:
                            tool_output = "Fork target has no valid mode/stage pairs."
                        else:
                            catalog = get_workflow_catalog()
                            # Build ForkTarget list, validating mode_ids
                            fork_targets: list[ForkTarget] = []
                            queries = tool_arguments.get("queries") or []
                            if not isinstance(queries, list):
                                queries = []
                            for idx, (f_mode_id, f_stage_id) in enumerate(fork_targets_raw):
                                f_mode = catalog.get(f_mode_id)
                                if f_mode is None:
                                    tool_output = f"Fork target mode '{f_mode_id}' not found."
                                    break
                                resolved_stage = f_stage_id or f_mode.start_stage_id
                                f_query = (
                                    queries[idx]
                                    if idx < len(queries) and isinstance(queries[idx], str)
                                    else query
                                )
                                fork_targets.append(
                                    ForkTarget(
                                        mode_id=f_mode_id,
                                        stage_id=resolved_stage,
                                        query=f_query,
                                    )
                                )
                            else:
                                # All targets valid — execute fan-out
                                fork_results_list = await _execute_fork_fanout(
                                    fork_targets=fork_targets,
                                    hooks=hooks,
                                    pool=pool,
                                    resolve_model=resolve_model,
                                    complete=complete,
                                    exec_op=exec_op,
                                    turn_budget=turn_budget,
                                    turn_start_mono=turn_start_mono,
                                    logger_override=_log,
                                    emit_stage=_emit_stage,
                                )
                                fork_results = fork_results_list
                                succeeded = sum(1 for r in fork_results_list if r.status == "ok")
                                # Surface failed children (timeout, error,
                                # no_tools, cancelled) explicitly so the
                                # parent model does not claim false success or
                                # hallucinate a deferred response it cannot
                                # deliver. The framework is strictly
                                # turn-by-turn — there is no background
                                # scheduler, so a failed child's work is lost
                                # unless the user re-issues the request.
                                failed_children = [r for r in fork_results_list if r.status != "ok"]
                                if failed_children:
                                    failure_parts: list[str] = []
                                    for r in failed_children:
                                        if r.status == "no_tools":
                                            failure_parts.append(
                                                f"{r.target.mode_id} (no tool calls; "
                                                f"{r.error_message})"
                                            )
                                        elif r.status == "timeout":
                                            failure_parts.append(
                                                f"{r.target.mode_id} (timed out; "
                                                f"{r.error_message or 'child turn timed out'})"
                                            )
                                        elif r.status == "cancelled":
                                            failure_parts.append(
                                                f"{r.target.mode_id} (cancelled; "
                                                f"{r.error_message or 'child turn was cancelled'})"
                                            )
                                        else:
                                            failure_parts.append(
                                                f"{r.target.mode_id} ({r.status}; "
                                                f"{r.error_message})"
                                            )
                                    tool_output = (
                                        f"Fork fan-out: {succeeded}/{len(fork_targets)} "
                                        f"children succeeded. The following children failed "
                                        f"(their results are NOT available — the framework is "
                                        f"turn-by-turn with no background scheduler, so a "
                                        f"failed child's work is lost; re-issue the request "
                                        f"to retry): {'; '.join(failure_parts)}. "
                                        f"Do not tell the user you will report back later — "
                                        f"you cannot. Report the failure and suggest re-issuing."
                                    )
                                else:
                                    tool_output = (
                                        f"Fork fan-out completed: "
                                        f"{succeeded}/{len(fork_targets)} children "
                                        f"succeeded. Results are in "
                                        f"stage_data.fork_results."
                                    )
                        await _emit_stage(
                            "tool_dispatch_complete",
                            {
                                "tool": tool_name,
                                "tool_kind": "transition",
                                "iteration": iteration,
                                "tool_calls_so_far": total_tool_calls,
                                "ok": True,
                                "transition": "fork",
                                "fork_children": len(fork_results_list) if fork_results_list else 0,
                                "duration_ms": int(
                                    (time.monotonic() - _tool_dispatch_started) * 1000
                                ),
                            },
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "name": tool_name,
                                "content": tool_output,
                            }
                        )
                        continue

                    # ── Case B: mode handoff (routing FSM) ────────────────
                    if is_routable_mode(target_stage):
                        routing_targets = get_routing_targets(mode_id)
                        if target_stage not in routing_targets and target_stage != "default":
                            tool_output = (
                                f"Mode handoff not allowed: '{mode_id}' cannot "
                                f"transition to '{target_stage}'."
                            )
                        else:
                            # Mid-turn mode switch: swap system prompt + tool surface
                            target_mode_config = get_workflow_catalog().get(target_stage)
                            if target_mode_config is None:
                                tool_output = f"Target mode '{target_stage}' not found."
                            else:
                                # Swap system prompt
                                target_mode_instruction = get_mode_system_intro(target_stage)
                                messages[0] = {
                                    "role": "system",
                                    "content": assemble_system(target_mode_instruction, None),
                                }
                                # Rebuild tool schemas for the new mode
                                tool_schemas = await _rebuild_tool_schemas_for_mode(
                                    target_stage,
                                    target_mode_config,
                                    pool=pool,
                                    hooks=hooks,
                                    is_workflow_for_target=(
                                        target_mode_config.mode_kind == "workflow"
                                        and hooks.workflow_session is not None
                                    ),
                                )
                                # Recompile capability manifest for new surface
                                capability_manifest = compile_capability_manifest(
                                    tool_schemas, mode_config=target_mode_config
                                )
                                # Update runtime state. The ctx built in
                                # Phase 1 carries the *entry* mode_id, so
                                # post-handoff operation/tool dispatches
                                # (op_ctx = ctx.model_copy(...), _to_tool_context)
                                # would inherit the stale mode and be rejected
                                # by execute_operation / execute_tool with
                                # "... is not allowed for mode '<entry>'".
                                # Patch ctx.mode_id so subsequent dispatches
                                # authorize against the new mode.
                                mode_id = target_stage
                                mode_config = target_mode_config
                                handoff_to = target_stage
                                try:
                                    ctx = ctx.model_copy(update={"mode_id": target_stage})
                                except Exception:  # pylint: disable=broad-except
                                    # Fallback for non-pydantic ctx shapes.
                                    try:
                                        ctx.mode_id = target_stage
                                    except Exception:  # pylint: disable=broad-except
                                        _log.warning(
                                            "Could not update ctx.mode_id after "
                                            "handoff to '%s'; post-handoff "
                                            "operations may be mis-authorized.",
                                            target_stage,
                                        )
                                tool_output = f"Handed off to mode: {target_stage}." + (
                                    f" Rationale: {rationale}" if rationale else ""
                                )
                        await _emit_stage(
                            "tool_dispatch_complete",
                            {
                                "tool": tool_name,
                                "tool_kind": "transition",
                                "iteration": iteration,
                                "tool_calls_so_far": total_tool_calls,
                                "ok": True,
                                "transition": "handoff",
                                "handoff_to": target_stage,
                                "duration_ms": int(
                                    (time.monotonic() - _tool_dispatch_started) * 1000
                                ),
                            },
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "name": tool_name,
                                "content": tool_output,
                            }
                        )
                        continue

                    # ── Case C: workflow stage transition (existing path) ──
                    if is_workflow:
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
                                updated_session = apply_transition(current_session, transition_req)
                                if hooks.workflow_session_updater is not None:
                                    hooks.workflow_session_updater(updated_session)
                                transitioned_to = updated_session.stage_id
                                tool_output = (
                                    f"Transitioned to stage: {updated_session.stage_id}."
                                    + (f" Rationale: {rationale}" if rationale else "")
                                )
                            except Exception as trans_exc:  # pylint: disable=broad-except
                                tool_output = (
                                    f"Transition failed: {type(trans_exc).__name__}: {trans_exc}"
                                )
                    else:
                        tool_output = (
                            f"Unknown transition target: '{target_stage}'. "
                            f"It is not a routable mode, a fork target, or a workflow stage."
                        )
                    await _emit_stage(
                        "tool_dispatch_complete",
                        {
                            "tool": tool_name,
                            "tool_kind": "transition",
                            "iteration": iteration,
                            "tool_calls_so_far": total_tool_calls,
                            "ok": transitioned_to is not None,
                            "transition": "workflow_stage" if is_workflow else "unknown",
                            "transitioned_to": transitioned_to,
                            "duration_ms": int((time.monotonic() - _tool_dispatch_started) * 1000),
                        },
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "name": tool_name,
                            "content": tool_output,
                        }
                    )
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
                        tool_output = _stringify_pack_tool_result(tool_result)
                    else:
                        mcp_result = await pool.call_tool(tool_name, tool_arguments)
                        tool_output = mcp_result.output_text or "Done"
                else:
                    mcp_result = await pool.call_tool(tool_name, tool_arguments)
                    tool_output = mcp_result.output_text or "Done"
                total_tool_calls += 1
                ledger.record(tool_name)
            except ApprovalRequired:
                raise
            except Exception as exc:  # pylint: disable=broad-except
                _log.warning("Tool '%s' failed in turn runner: %s", tool_name, exc)
                tool_output = f"Tool execution failed: {type(exc).__name__}: {exc}"

            await _emit_stage(
                "tool_dispatch_complete",
                {
                    "tool": tool_name,
                    "tool_kind": _tool_kind,
                    "iteration": iteration,
                    "tool_calls_so_far": total_tool_calls,
                    "ok": not tool_output.startswith("Tool execution failed:"),
                    "duration_ms": int((time.monotonic() - _tool_dispatch_started) * 1000),
                },
            )

            if not tool_output.strip():
                tool_output = "Done"

            # Track the latest tool result for the model router.
            last_tool_result_text = str(tool_output)

            tool_result_msg: dict[str, Any] = {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": str(tool_output),
            }
            # Preserve thought_signature for Gemini thinking mode round-trips.
            # The signature lives at the top level (REST gateway flat shape) or
            # nested under function (OpenAI-compatible shape).
            thought_sig = tool_call.get("thought_signature")
            if not thought_sig:
                fn_payload = tool_call.get("function")
                if isinstance(fn_payload, dict):
                    thought_sig = fn_payload.get("thought_signature")
            if thought_sig:
                tool_result_msg["thought_signature"] = thought_sig
            messages.append(tool_result_msg)

        if not tools_dispatched:
            # No tool calls were dispatched — model returned text
            content = getattr(completion, "content", "") or ""
            content = content.strip()
            break

    if not content:
        # The loop exited due to max_tool_iterations while the model was
        # still calling tools. Force one final no-tools completion so the
        # model synthesizes its findings into a coherent response instead
        # of the generic "did not produce a final response" fallback.
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
                final_usage = _extract_completion_usage(final_completion)
                total_prompt_tokens += final_usage[0]
                total_completion_tokens += final_usage[1]
                total_tokens += final_usage[2]
                total_duration_ms += final_usage[5]
        except Exception as exc:  # pylint: disable=broad-except
            _log.warning("iteration-cap final completion failed: %s", exc)
        content = content or "The model did not produce a final response."

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
                messages.append(
                    {
                        "role": "user",
                        "content": "Provide a closing summary for this workflow.",
                    }
                )
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

    # ── Phase 3.6: Checkpoint evaluation ─────────────────────────────
    # Turn-anchored, not wall-clock. Fires after the post-turn slice on
    # non-blocked turns when the host has wired checkpointing. The kernel
    # builds a CheckpointBundle from current state + workflow session;
    # the host's on_checkpoint hook persists it. Checkpoint failures are
    # caught — they must not break the turn.
    if not turn_blocked and hooks.on_checkpoint is not None:
        tuning = hooks.checkpoint_tuning
        should_checkpoint = _should_checkpoint(
            tuning=tuning,
            transitioned_to=transitioned_to,
            is_workflow=is_workflow,
            hooks=hooks,
        )
        if should_checkpoint:
            try:
                bundle = _build_checkpoint_bundle(
                    mode_id=mode_id,
                    transitioned_to=transitioned_to,
                    is_workflow=is_workflow,
                    hooks=hooks,
                )
                if bundle is not None:
                    await hooks.on_checkpoint(bundle)
            except Exception as exc:  # pylint: disable=broad-except
                _log.warning("on_checkpoint hook failed: %s", exc)

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
        handoff_to=handoff_to,
        fork_results=fork_results,
        workflow_terminated=workflow_terminated,
    )


# ---------------------------------------------------------------------------
# Unified FSM routing helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _CapabilityGateBlock:
    """Result of a failed capability-gate check."""

    result: TurnResult
    missing: tuple[str, ...]


def _validate_model_capabilities(
    *,
    mode_config: Any,
    model_def: Any,
    preferred_model_id: str,
    router_wired: bool,
) -> _CapabilityGateBlock | None:
    """Validate the resolved model against the mode's required capabilities.

    Returns ``None`` when the turn may proceed. Returns a
    ``_CapabilityGateBlock`` (carrying a user-facing ``TurnResult``) when the
    model is missing a required capability AND a router is wired — the gate
    hard-blocks so a weak model cannot reach for write-effect tools it cannot
    use reliably. When no router is wired, the gate is advisory (returns
    ``None``) because a host without a router has no escalation path.

    Backward compatible: a mode with no ``required_capabilities`` (the default)
    never blocks. A ``model_def`` that does not carry ``capabilities`` (e.g. a
    test stub) is treated as having no declared capabilities, which only
    matters when the mode requires capabilities.
    """
    if mode_config is None:
        return None
    required = getattr(mode_config, "required_capabilities", None) or ()
    if not required:
        return None
    # The gate only hard-blocks when a router is wired (escalation path exists).
    if not router_wired:
        return None
    model_caps = set(getattr(model_def, "capabilities", ()) or ())
    missing = tuple(
        cap for cap in required if cap not in model_caps
    )
    if not missing:
        return None
    model_id = getattr(model_def, "model_id", preferred_model_id)
    mode_id = getattr(mode_config, "mode_id", "(unknown)")
    missing_str = ", ".join(sorted(missing))
    message = (
        f"Model '{model_id}' is missing the capability(ies) required by mode "
        f"'{mode_id}' ({missing_str}). Use a model that declares those "
        f"capabilities, or switch to a mode that does not require them. No "
        f"tools were called."
    )
    return _CapabilityGateBlock(
        result=TurnResult(response_text=message, blocked=True),
        missing=missing,
    )


def _detect_query_intents(
    *,
    query: str,
    routing_targets: list[str],
) -> set[str]:
    """Detect which intents the user's query expresses, using the routing
    targets' ``fast_path_patterns`` as the pattern source.

    Returns the set of intent tag strings (lowercased) that the query matches.
    Each routing target's ``routing_intent_tags`` are attributed to the query
    when any of that target's ``fast_path_patterns`` matches the query
    (case-insensitive). This is data-driven: the kernel does not hardcode
    which keywords map to which intents — the mode front-matter declares both
    the patterns and the tags.
    """
    if not routing_targets:
        return set()
    catalog = get_workflow_catalog()
    query_lower = query.lower()
    detected: set[str] = set()
    for target_id in routing_targets:
        mode_config = catalog.get(target_id)
        if mode_config is None:
            continue
        # Check fast_path_patterns (anchored regex) first.
        matched = False
        for pattern in mode_config.fast_path_patterns:
            try:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    detected.update(mode_config.routing_intent_tags)
                    matched = True
                    break
            except re.error:
                pass
        if matched:
            continue
        # Check routing_intent_keywords (per-tag substring match). Each
        # keyword only attributes its associated tag, so a keyword match for
        # one tag does not falsely trigger a different tag the same mode
        # declares. The tag vocabulary is pack-authored and opaque to the
        # kernel.
        for tag, keywords in mode_config.routing_intent_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    detected.add(tag)
                    break
        # Also check the command_pattern — a query that explicitly names the
        # mode (e.g. "learn that...") is a strong signal.
        if not mode_config.routing_intent_keywords and mode_config.command_pattern:
            try:
                if re.search(mode_config.command_pattern, query, re.IGNORECASE):
                    detected.update(mode_config.routing_intent_tags)
            except re.error:
                pass
    return detected


def _tool_effect(
    tool_name: str,
    *,
    operation_catalog: dict[str, Any],
    tool_catalog: dict[str, Any],
) -> str:
    """Return the ``EffectKind`` of a capability id, or ``"read"`` when
    unknown.

    Resolves against the operation catalog first, then the tool catalog
    (matching ``resolve_capability`` precedence). MCP-native tools have no
    catalog definition and default to ``"read"`` — the routing-hint write-tool
    guard only fires on declared write-effect tools, so MCP tools without a
    definition are left alone.
    """
    op_def = operation_catalog.get(tool_name)
    if op_def is not None:
        return str(getattr(op_def, "effect", "read") or "read")
    tool_def = tool_catalog.get(tool_name)
    if tool_def is not None:
        return str(getattr(tool_def, "effect", "read") or "read")
    return "read"


def _compute_capability_routing_hint(
    *,
    mode_id: str,
    query: str,
    capability_manifest: CapabilityManifest,  # retained for API stability
    routing_targets: list[str],
    mode_config: WorkflowModeConfig | None = None,
) -> str:
    """Compute a proactive routing hint when the current mode lacks tools.

    Tag-blind and prose-blind: the kernel does not name, hardcode, or
    hard-author any intent tag (research, reasoning, write, learn, ...) or
    any steering prose. Intent tags are opaque pack-authored vocabulary;
    routing prose is pack-authored via each target's ``routing_hint``
    template. Each routing target declares its own ``fast_path_patterns``
    (what queries it handles) and ``routing_intent_tags`` (what intents it
    covers) in its _mode.md front matter. The *current* mode symmetrically
    declares ``satisfies_intents`` (what intents it handles in-place). This
    function:

    1. Detects query intents by matching against all routing targets' patterns.
       The returned set is opaque strings — the kernel never inspects its
       members.
    2. Computes the gap as the set difference of detected intents minus the
       current mode's ``satisfies_intents``. If the gap is empty, no hint
       (zero tokens). The live capability manifest is NOT consulted for the
       gap check: the coarse ``kind == "tool"`` heuristic it implied treated
       any tool-bearing mode as satisfying every intent, which suppressed
       handoffs from tool-rich modes that lacked the specific intent's tools
       (e.g. coding mode falling back to coding_create_worktree instead of
       handing off to markdown for a write-a-document request).
    3. Finds routing targets whose ``routing_intent_tags`` cover all detected
       intents — if exactly one does, suggest a single handoff.
    4. If multiple targets each cover a disjoint subset of intents, suggest a
       ``fork:`` fan-out so each child runs in its own mode with its own tools.
    5. If no target fully covers the detected intents but a gap exists,
       fall back to listing all routing targets.

    Returns an empty string when no hint is needed (zero tokens).
    """
    if not routing_targets:
        return ""
    catalog = get_workflow_catalog()

    # ── 1. Detect query intents from routing target patterns ──────────
    detected_intents = _detect_query_intents(query=query, routing_targets=routing_targets)

    # ── 2. Check if the current mode already satisfies detected intents ──
    # Declarative and tag-blind: the kernel works over opaque intent-tag
    # sets. It never names any specific intent (research, reasoning, write,
    # learn, ...) — those are pack-authored vocabulary. The current mode
    # authors ``satisfies_intents`` in its _mode.md front matter; the
    # detected intents it does not claim to satisfy are the gap. Default
    # empty (e.g. ``default`` mode) means every detected intent is a gap,
    # so a handoff hint is offered. The live capability manifest is NOT
    # consulted for the gap check: the coarse ``kind == "tool"`` heuristic
    # it implied treated any tool-bearing mode as satisfying every intent,
    # which suppressed handoffs from tool-rich modes that lacked the
    # specific intent's tools (e.g. coding mode falling back to
    # coding_create_worktree instead of handing off to markdown for a
    # write-a-document request).
    current_satisfied: set[str] = set(
        mode_config.satisfies_intents if mode_config is not None else []
    )
    gap_intents = detected_intents - current_satisfied
    has_gap = bool(gap_intents)
    if not detected_intents and not has_gap:
        return ""
    if detected_intents and not has_gap:
        # The current mode already satisfies all detected intents.
        return ""

    # ── 3. Find targets that cover all detected intents (single handoff) ──
    covering_targets: list[str] = []
    for target_id in routing_targets:
        mode_config = catalog.get(target_id)
        if mode_config is None:
            continue
        target_tags = set(mode_config.routing_intent_tags)
        if detected_intents and detected_intents.issubset(target_tags):
            covering_targets.append(target_id)

    # ── 4. Multi-intent fork detection ─────────────────────────────────
    # When no single target covers all detected intents but multiple targets
    # each cover a disjoint subset, suggest a fork fan-out. This is the
    # generic form of the learn+write case: learn covers "learn", markdown
    # covers "write", neither covers both → fork:learn,markdown.
    if not covering_targets and len(detected_intents) >= 2:
        partial_targets: list[str] = []
        covered_intents: set[str] = set()
        for target_id in routing_targets:
            mode_config = catalog.get(target_id)
            if mode_config is None:
                continue
            target_tags = set(mode_config.routing_intent_tags)
            overlap = target_tags & detected_intents
            if overlap:
                partial_targets.append(target_id)
                covered_intents.update(overlap)
        # Only suggest a fork when the partial targets collectively cover
        # all detected intents and there are at least 2 of them.
        if len(partial_targets) >= 2 and covered_intents == detected_intents:
            fork_target = f"fork:{','.join(partial_targets)}"
            # Purely mechanical: name the tool and the fork target id. The
            # kernel does not narrate tool-policy semantics ("mode-scoped
            # tools", "rejected") or the fork's internal mechanics — the
            # model reads each child mode's own _mode.md body after the
            # fork. Same prose-blind contract as the single-handoff branch.
            hint = (
                "## Routing Hint\n\n"
                f"Hand off by calling `transition_stage` with "
                f"target_stage_id `{fork_target}`."
            )
            return hint

    # ── 5. Single handoff to the covering target ──────────────────────
    if covering_targets:
        preferred = covering_targets[0]
        # The kernel owns no routing prose. The handoff instruction is purely
        # mechanical: name the tool (transition_stage) and the target id. Any
        # semantic steering ("researches then reasons", "do not use the
        # one-shot search action mode", ...) is pack-authored routing *input*
        # — the target's routing_intent_tags / fast_path_patterns that drove
        # the decision to pick this target — not kernel-authored routing
        # *output*. The model reads the chosen mode's own _mode.md body after
        # the handoff; the kernel does not narrate the mode's purpose here.
        hint = (
            "## Routing Hint\n\n"
            f"Your current mode may lack the tools needed for this request. "
            f"Hand off to `{preferred}` by calling `transition_stage` with "
            f"target_stage_id `{preferred}`."
        )
        return hint

    # ── 6. Fallback: list all routing targets ──────────────────────────
    if not has_gap:
        return ""
    targets_str = ", ".join(routing_targets)
    hint = (
        "## Routing Hint\n\n"
        "Your current mode may lack the tools needed for this request. "
        f"You can hand off to a better-suited mode by calling "
        f"`transition_stage` with one of these target_stage_id values: "
        f"{targets_str}."
    )
    return hint


async def _rebuild_tool_schemas_for_mode(
    target_mode_id: str,
    target_mode_config: Any,
    *,
    pool: MCPToolPool,
    hooks: TurnHooks,
    is_workflow_for_target: bool,
) -> list[dict[str, Any]]:
    """Rebuild the tool schema list for a new mode after a mid-turn handoff.

    Mirrors the tool assembly logic in ``_run_turn_inner`` Phase 1 but
    without re-running pre-turn operations or the input guard. The new
    mode's ``allowed_tool_ids`` and ``tools_authoritative`` determine the
    tool surface.
    """
    allowed_server_ids: set[str] | None = None
    if target_mode_config and target_mode_config.allowed_tool_ids:
        allowed_set = set(target_mode_config.allowed_tool_ids)
        allowed_server_ids = {
            c.server_id
            for c in getattr(pool, "configs", [])
            if any(t.startswith(c.server_id + ".") for t in allowed_set)
        }
    elif (
        target_mode_config
        and target_mode_config.tools_authoritative
        and not target_mode_config.allowed_tool_ids
    ):
        allowed_server_ids = set()

    mcp_tools = await pool.list_tools(allowed_server_ids=allowed_server_ids)
    if target_mode_config and (
        target_mode_config.allowed_tool_ids or target_mode_config.tools_authoritative
    ):
        allowed = set(target_mode_config.allowed_tool_ids)
        mcp_tools = [t for t in mcp_tools if t.name in allowed]
    new_tool_schemas: list[dict[str, Any]] = [
        mcp_tool_to_openai_function(
            name=spec.name,
            description=spec.description,
            input_schema=spec.input_schema,
        )
        for spec in mcp_tools
    ]
    new_tool_schemas.extend(
        build_operation_tool_schemas(
            target_mode_id,
            tools_authoritative=bool(target_mode_config and target_mode_config.tools_authoritative),
        )
    )
    new_tool_schemas.extend(_build_in_process_tool_schemas(target_mode_config))
    if is_workflow_for_target:
        new_tool_schemas.extend(get_transition_tool_schema())
    target_has_routing = bool(get_routing_targets(target_mode_id)) if target_mode_config else False
    new_tool_schemas, _ = _prune_model_tool_schemas(
        tool_schemas=new_tool_schemas,
        mode_config=target_mode_config,
        is_workflow=is_workflow_for_target,
        has_routing_targets=target_has_routing,
    )
    return new_tool_schemas


def _fork_mode_requires_tools(mode_id: str) -> bool:
    """Return True when a fork child's mode declares action operations/tools.

    Modes like ``learn`` and ``markdown`` exist to *take action* (distill +
    persist memories, write documents). Their ``_mode.md`` front matter
    declares non-empty ``allowed_operation_ids`` or ``allowed_tool_ids``.
    A fork child running such a mode that completes with **zero tool calls**
    almost certainly produced a hallucinated "Done — wrote it to disk" text
    response without invoking its write/learn tool, leaving nothing on disk
    or in memory. The caller marks these ``status="no_tools"`` instead of
    ``"ok"`` so the parent does not report a false success.

    Chat-only modes (``default``, ``ponder``) with empty allowlists
    legitimately complete without tools, so this returns False for them.
    """
    mode_config = get_workflow_catalog().get(mode_id)
    if mode_config is None:
        return False
    # Use the *authored* allowlists, not the merged (effective) ones, so a
    # chat-only mode that inherits ambient capabilities (e.g. recall_memory
    # via default_available) is not misflagged as requiring tools to complete.
    return bool(mode_config.authored_allowed_operation_ids) or bool(
        mode_config.authored_allowed_tool_ids
    )


async def _run_fork_child_turn(
    *,
    target: ForkTarget,
    hooks: TurnHooks,
    child_budget: TurnBudget | None,
    started: float,
    logger_override: logging.Logger | None,
) -> tuple[ForkChildResult, str, int]:
    """Run one fork child turn and map the outcome to a ``ForkChildResult``.

    Returns ``(result, status, elapsed_ms)`` so the caller can emit a
    ``fork_child_complete`` stage event without re-deriving the status.
    Extracted from ``_execute_fork_fanout`` to keep that function under
    the McCabe complexity cap.

    A child whose mode declares action operations/tools (``learn``,
    ``markdown``, etc.) but completes with **zero tool calls** is marked
    ``status="no_tools"`` rather than ``"ok"``. Such a child typically
    produced a hallucinated "Done — wrote it to disk" text response
    without invoking its write/learn tool, leaving nothing on disk or
    in the memory store. The parent must not report these as succeeded.
    """
    import time as _time

    _flog = logger_override or logger
    try:
        child_result = await run_turn(
            mode_id=target.mode_id,
            query=target.query,
            hooks=hooks,
            max_tool_iterations=5,
            turn_budget=child_budget,
            logger_override=_flog,
        )
        elapsed_ms = int((_time.monotonic() - started) * 1000)
        if child_result.tool_call_count == 0 and _fork_mode_requires_tools(target.mode_id):
            _flog.warning(
                "fork child %s completed with 0 tool calls but its mode declares "
                "action operations/tools; marking no_tools (likely hallucinated success)",
                target.mode_id,
            )
            return (
                ForkChildResult(
                    target=target,
                    status="no_tools",
                    output_text=child_result.response_text,
                    error_message=(
                        f"fork child {target.mode_id} produced no tool calls; its mode "
                        "declares action operations/tools, so its reported success is "
                        "unverified (nothing written to disk or memory)."
                    ),
                    duration_ms=elapsed_ms,
                ),
                "no_tools",
                elapsed_ms,
            )
        return (
            ForkChildResult(
                target=target,
                status="ok",
                output_text=child_result.response_text,
                duration_ms=elapsed_ms,
            ),
            "ok",
            elapsed_ms,
        )
    except TimeoutError:
        elapsed_ms = int((_time.monotonic() - started) * 1000)
        return (
            ForkChildResult(
                target=target,
                status="timeout",
                error_message="Child turn timed out.",
                duration_ms=elapsed_ms,
            ),
            "timeout",
            elapsed_ms,
        )
    except Exception as exc:  # pylint: disable=broad-except
        elapsed_ms = int((_time.monotonic() - started) * 1000)
        _flog.warning("fork child %s failed: %s: %s", target.mode_id, type(exc).__name__, exc)
        return (
            ForkChildResult(
                target=target,
                status="error",
                error_message=f"{type(exc).__name__}: {exc}",
                duration_ms=elapsed_ms,
            ),
            "error",
            elapsed_ms,
        )


def _fork_child_budget(turn_budget: TurnBudget | None, n: int) -> TurnBudget | None:
    """Divide the parent turn budget equally across ``n`` children.

    A field set to ``0`` means unlimited and is passed through as ``0``.
    Returns ``None`` when the parent has no budget.
    """
    if turn_budget is None:
        return None
    return TurnBudget(
        max_tool_calls=(turn_budget.max_tool_calls // n if turn_budget.max_tool_calls else 0),
        max_total_tokens=(turn_budget.max_total_tokens // n if turn_budget.max_total_tokens else 0),
        max_wall_ms=(turn_budget.max_wall_ms // n if turn_budget.max_wall_ms else 0),
    )


async def _run_fork_child(
    *,
    idx: int,
    target: ForkTarget,
    results: list[ForkChildResult | None],
    turn_budget: TurnBudget | None,
    n: int,
    hooks: TurnHooks,
    emit_stage: Callable[[str, dict[str, Any]], Awaitable[None]] | None,
    logger_override: logging.Logger | None,
) -> None:
    """Run one fork child: emit start, run the child turn, store result, emit complete.

    Extracted from ``_execute_fork_fanout`` to keep that orchestrator under
    the McCabe complexity cap. ``results`` is mutated in place at ``idx``.
    """
    import time as _time

    _flog = logger_override or logger
    started = _time.monotonic()
    child_budget = _fork_child_budget(turn_budget, n)
    if emit_stage is not None:
        await emit_stage(
            "fork_child_start",
            {
                "child_index": idx,
                "mode_id": target.mode_id,
                "stage_id": target.stage_id,
                "query_preview": (target.query or "")[:80],
            },
        )
    result, status, elapsed_ms = await _run_fork_child_turn(
        target=target,
        hooks=hooks,
        child_budget=child_budget,
        started=started,
        logger_override=_flog,
    )
    results[idx] = result
    if emit_stage is not None:
        await emit_stage(
            "fork_child_complete",
            {
                "child_index": idx,
                "mode_id": target.mode_id,
                "stage_id": target.stage_id,
                "status": status,
                "duration_ms": elapsed_ms,
            },
        )


async def _execute_fork_fanout(
    *,
    fork_targets: list[ForkTarget],
    hooks: TurnHooks,
    pool: MCPToolPool,
    resolve_model: Callable[[str], Any],
    complete: Callable[..., Awaitable[Any]],
    exec_op: Callable[..., Awaitable[OperationResult]],
    turn_budget: TurnBudget | None,
    turn_start_mono: float,
    logger_override: logging.Logger | None,
    emit_stage: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None,
) -> list[ForkChildResult]:
    """Execute a parallel fork fan-out to heterogeneous modes.

    Each child runs in an isolated session with its own mode's system prompt
    and tool surface. Children are fanned out via ``asyncio.TaskGroup`` and
    collected into a list of ``ForkChildResult``.

    The parent turn's ``TurnBudget`` is divided equally across N children.
    Each child gets a sub-budget. If a child exhausts its sub-budget, it
    returns a partial result.

    ``emit_stage`` is the parent turn's ``_emit_stage`` helper. When set,
    the fan-out emits ``fork_start`` / ``fork_child_start`` (one per
    child) / ``fork_child_complete`` (one per child) / ``fork_complete``
    so a transport can render one line per active child. Child turns
    run with ``on_stage`` suppressed on their hooks so their internal
    stage events do not mix into the parent's stage line.
    """
    import asyncio

    _flog = logger_override or logger
    n = len(fork_targets)
    if n == 0:
        return []

    if emit_stage is not None:
        await emit_stage(
            "fork_start",
            {
                "children": n,
                "targets": [{"mode_id": t.mode_id, "stage_id": t.stage_id} for t in fork_targets],
            },
        )

    results: list[ForkChildResult | None] = [None] * n

    async def _run_child(idx: int, target: ForkTarget) -> None:
        await _run_fork_child(
            idx=idx,
            target=target,
            results=results,
            turn_budget=turn_budget,
            n=n,
            hooks=hooks,
            emit_stage=emit_stage,
            logger_override=_flog,
        )

    try:
        async with asyncio.TaskGroup() as tg:
            for idx, target in enumerate(fork_targets):
                tg.create_task(_run_child(idx, target))
    except ExceptionGroup as eg:  # pylint: disable=broad-except
        _flog.warning("fork fan-out TaskGroup had errors: %s", eg.exceptions)

    # Fill in any None slots as cancelled
    final_results: list[ForkChildResult] = []
    for idx, r in enumerate(results):
        if r is not None:
            final_results.append(r)
        else:
            final_results.append(
                ForkChildResult(
                    target=fork_targets[idx],
                    status="cancelled",
                    error_message="Child turn was cancelled.",
                )
            )
    if emit_stage is not None:
        await emit_stage(
            "fork_complete",
            {
                "children": n,
                "ok": sum(1 for r in final_results if r.status == "ok"),
                "failed": sum(1 for r in final_results if r.status != "ok"),
            },
        )
    return final_results


def _build_minimal_context(
    *,
    mode_id: str,
    query: str,
    resolve_model_fn: Any,
    complete_fn: Any,
    read_authored_text_fn: Any,
    execute_tool_fn: Any,
    execute_operation_fn: Any,
    state_provider: Any = None,
    pack_configuration_resolver: Any = None,
    approval_service: Any = None,
    code_change_artifact_store: Any = None,
    workspace_id: str = "lite-host",
    user_id: str = "lite-user",
    user_slug: str = "lite-user",
    user_name: str = "Lite User",
    principal_key: str = "lite-user",
    conversation_id: str = "lite-host-console",
    surface_type: str = "personal_dm",
    surface_id: str = "lite-host-console",
    memory_scope: str = "personal",
    session_plan_digest: str = "",
) -> OperationExecutionContext:
    """Build a minimal OperationExecutionContext for the turn runner."""
    return OperationExecutionContext(
        workspace_id=workspace_id,
        user_id=user_id,
        user_slug=user_slug,
        user_name=user_name,
        principal_key=principal_key,
        conversation_id=conversation_id,
        surface_type=cast(ConversationSurface, surface_type),
        surface_id=surface_id,
        memory_scope=cast(MemoryScope, memory_scope),
        mode_id=mode_id,
        query=query,
        model_id="default",
        complete_fn=complete_fn,
        resolve_model_fn=resolve_model_fn,
        read_authored_text_fn=read_authored_text_fn,
        execute_tool_fn=execute_tool_fn,
        execute_operation_fn=execute_operation_fn,
        state_provider=state_provider,
        pack_configuration_resolver=pack_configuration_resolver,
        approval_service=approval_service,
        session_plan_digest=session_plan_digest,
        code_change_artifact_store=code_change_artifact_store,
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


def _stringify_pack_tool_result(result: ToolResult) -> str:
    """Render a pack-contributed tool result for model consumption.

    Pack tools (coding_list_directory, coding_find_files, coding_read_file, etc.)
    return BOTH a terse human summary in ``output_text`` (e.g. "Listed 5 entries.")
    AND the structured payload in ``output_json`` (the entry list itself). Emitting
    only the summary starves the model of the data it called the tool to get. When
    both are present, concatenate the summary and the JSON payload.

    When the tool returns an error (``result.error`` is non-empty), the error
    MUST be surfaced to the model. Otherwise the model sees only "Done" and
    cannot understand why the tool produced no data, leading it to report
    that the tool "returned a generic Done message" instead of diagnosing
    the actual failure (e.g. missing pack configuration, bad workspace root).
    """
    import json as _json

    summary = result.output_text.strip()
    payload = result.output_json
    if result.error.strip():
        error_text = f"Error: {result.error.strip()}"
        if summary:
            return f"{summary}\n{error_text}"
        if payload:
            return f"{error_text}\n{_json.dumps(payload, indent=2, sort_keys=True)}"
        return error_text
    if summary and payload:
        return f"{summary}\n{_json.dumps(payload, indent=2, sort_keys=True)}"
    if summary:
        return summary
    if payload:
        return _json.dumps(payload, indent=2, sort_keys=True)
    return "Done"


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
        pack_config=getattr(ctx, "pack_config", {}),
        pack_configuration_resolver=getattr(ctx, "pack_configuration_resolver", None),
        approval_service=getattr(ctx, "approval_service", None),
        approval_request_id=getattr(ctx, "approval_request_id", ""),
        session_plan_digest=getattr(ctx, "session_plan_digest", ""),
        code_change_artifact_store=getattr(ctx, "code_change_artifact_store", None),
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


def _tool_schema_name(schema: dict[str, Any]) -> str:
    """Return the function name from an OpenAI tool schema."""
    function = schema.get("function")
    if not isinstance(function, dict):
        return ""
    name = function.get("name")
    return str(name).strip() if isinstance(name, str) else ""


def _mode_bound_operation_ids(mode_config: Any) -> set[str]:
    """Return operation ids explicitly bound by one mode definition."""
    if mode_config is None:
        return set()
    return set(
        list(getattr(mode_config, "entry_operation_ids", []) or [])
        + list(getattr(mode_config, "pre_turn_operation_ids", []) or [])
        + list(getattr(mode_config, "post_turn_operation_ids", []) or [])
        + list(getattr(mode_config, "allowed_operation_ids", []) or [])
    )


def _operation_allowed_for_mode(
    *,
    operation_id: str,
    definition: Any,
    mode_config: Any,
    tools_authoritative: bool,
    mode_bound_ids: set[str],
) -> bool:
    """Return whether one operation should be advertised for this mode."""
    if mode_config is None:
        return True

    mode_id = str(getattr(mode_config, "mode_id", "") or "")
    mode_allowlist = getattr(definition, "mode_allowlist", None)
    if mode_allowlist and mode_id in mode_allowlist:
        return True
    if operation_id in mode_bound_ids:
        return True

    if tools_authoritative:
        return False
    if mode_allowlist:
        return False
    return True


def _prune_model_tool_schemas(
    *,
    tool_schemas: list[dict[str, Any]],
    mode_config: Any,
    is_workflow: bool,
    has_routing_targets: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Apply one final model-facing prune pass over assembled tool schemas.

    Assembly-time filtering can evolve across MCP/operation/tool catalog paths.
    This final pass is the payload contract: only the allowed tool set reaches
    the model completion call.
    """

    operation_catalog = get_operation_catalog()
    tool_catalog = get_tool_catalog()
    tools_authoritative = bool(mode_config and getattr(mode_config, "tools_authoritative", True))
    allowed_tool_ids: set[str] | None = None
    if mode_config and (getattr(mode_config, "allowed_tool_ids", []) or tools_authoritative):
        allowed_tool_ids = set(getattr(mode_config, "allowed_tool_ids", []) or [])

    mode_id = str(getattr(mode_config, "mode_id", "") or "")
    mode_bound_ids = _mode_bound_operation_ids(mode_config)
    seen_names: set[str] = set()
    pruned: list[dict[str, Any]] = []
    dropped_duplicate = 0
    dropped_disallowed = 0
    dropped_invalid = 0

    for schema in tool_schemas:
        name = _tool_schema_name(schema)
        if not name:
            dropped_invalid += 1
            continue
        if name in seen_names:
            dropped_duplicate += 1
            continue
        seen_names.add(name)

        if name == _TRANSITION_TOOL_NAME:
            if is_workflow or has_routing_targets:
                pruned.append(schema)
            else:
                dropped_disallowed += 1
            continue

        op_def = operation_catalog.get(name)
        if op_def is not None:
            if _operation_allowed_for_mode(
                operation_id=name,
                definition=op_def,
                mode_config=mode_config,
                tools_authoritative=tools_authoritative,
                mode_bound_ids=mode_bound_ids,
            ):
                pruned.append(schema)
            else:
                dropped_disallowed += 1
            continue

        tool_def = tool_catalog.get(name)
        if tool_def is not None:
            mode_allowlist = getattr(tool_def, "mode_allowlist", None)
            if allowed_tool_ids is not None and name not in allowed_tool_ids:
                dropped_disallowed += 1
                continue
            if mode_allowlist and mode_id and mode_id not in mode_allowlist:
                dropped_disallowed += 1
                continue
            pruned.append(schema)
            continue

        # Unknown tools are MCP tools. Restrict them by mode allowlist when
        # one is active (including authoritative-empty => no unknown tools).
        if allowed_tool_ids is not None and name not in allowed_tool_ids:
            dropped_disallowed += 1
            continue
        pruned.append(schema)

    stats = {
        "input_count": len(tool_schemas),
        "output_count": len(pruned),
        "dropped_duplicate": dropped_duplicate,
        "dropped_disallowed": dropped_disallowed,
        "dropped_invalid": dropped_invalid,
    }
    return pruned, stats


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

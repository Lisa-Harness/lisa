"""Kernel turn runner — the single model+tool loop.

This module provides `run_turn()` and `TurnHooks`, the core conversational turn
loop that executes one mode's pipeline: pre-turn operations, model completion
with tool dispatch, and post-turn side effects.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Literal, cast

from kernel.approval import ApprovalInvocation, ApprovalRequired, render_approval_request
from kernel.capability_manifest import compile_capability_manifest
from kernel.config import settings as _kernel_settings
from kernel.context.models import ContextBudgetSnapshot
from kernel.execution_policy import ExecutionPolicy, resolve_capability_invocation_limit
from kernel.ipsar_fsm import (
    apply_transition,
    get_mode_system_intro,
    get_required_mode_intro,
    get_required_stage_instruction,
    get_routing_targets,
    get_transition_tool_schema,
    get_workflow_catalog,
    get_yield_tool_schema,
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
    WorkflowTransitionRequest,
    YieldAction,
    YieldPolicy,
)
from kernel.operations.registry import (
    build_operation_tool_schemas,
    execute_operation,
    get_operation_catalog,
)
from kernel.operations.structured_output import completion_content
from kernel.provider_core import StreamedChunk, StreamedCompletion, _aggregate_streamed_chunks
from kernel.recovery import RecoveryIdentity, RecoveryRecord
from kernel.seams.protocols import (
    ContentRegistry,
    CoordinatorRouteDecision,
    DataSanitizer,
    EventBus,
    ExtensionEvidenceBuilder,
    InputGuard,
    ModelPolicy,
    ModelRouter,
    OutputGuard,
    ProviderGatewayProvider,
    RouteCoordinator,
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
from kernel.workflow_instance import (
    WorkflowInstance,
    WorkflowStep,
    _step_id_for,
)

logger = logging.getLogger(__name__)

# Turn-runner guard caps live in ``kernel.config.Settings`` (fields
# ``narration_nudge_cap`` / ``forced_fork_non_fork_tool_cap`` /
# ``stage_nudge_threshold`` / ``stage_nudge_cap`` / ``max_handoffs_per_turn``
# / ``duplicate_hint_limit`` / ``hint_cycle_min_len`` / ``hint_override_limit``)
# so hosts can tune them via env vars without editing kernel source. They
# were previously module-level constants here; see the git-blame history
# for the incidents that motivated each cap (the 2026-07-27 fork-refusal,
# the 2026-07-30 plan→ponder→markdown bounce, etc.). ``_run_turn_inner``
# reads them from the shared ``_kernel_settings`` singleton and, for the
# few that are referenced multiple times per turn, binds a local alias
# once at the top of the loop to avoid repeated attribute access.


def _extract_hint_target(hint_text: str) -> str:
    """Extract the ``target_stage_id`` value from a routing-hint string.

    Hint prose is pack-authored but the kernel emits the mechanical
    ``target_stage_id `<id>`` suffix itself (see
    ``_compute_capability_routing_hint``). This extracts that id so the
    hint-cycle guard can track the target sequence without parsing
    arbitrary prose. Returns ``""`` when no target is found (e.g. the
    generic fallback hint that lists multiple targets).
    """
    if not hint_text:
        return ""
    # The hint always ends with: target_stage_id `<id>`.
    marker = "target_stage_id `"
    idx = hint_text.rfind(marker)
    if idx < 0:
        return ""
    rest = hint_text[idx + len(marker) :]
    end = rest.find("`")
    if end < 0:
        return ""
    return rest[:end].strip()


def _build_budget_briefing(
    *,
    state: Any,
    max_tool_iterations: int,
    per_step_max_iterations: int,
    per_step_max_tool_calls: int,
    tool_schemas: list[dict[str, Any]],
    operation_catalog: dict[str, Any],
    tool_catalog: dict[str, Any],
    execution_policy: Any,
    budget_exhausted: bool,
) -> str:
    """Build a compact, per-iteration lifecycle status block for the system prompt.

    The unified turn lifecycle replaces the old ``## Turn Budget`` block with a
    ``## Turn Lifecycle`` block that speaks the model's new vocabulary:
    ``continue`` / ``transition_stage`` / ``yield`` / ``finish``. Only
    **discrete, actionable** budgets are surfaced: iterations remaining,
    this step's iteration/tool-call sub-budget, and per-target call limits
    for tools that declare a non-zero ``max_calls_per_turn``. Continuous
    budgets (tokens, wall time) are deliberately omitted — the model cannot
    reliably convert "50k tokens left" into a concrete decision, and
    advertising a large token budget risks anchoring the model to *use* it.

    When ``budget_exhausted`` is True, the block tells the model the harness
    is about to yield (not "produce your answer now" — the old instruction
    that fought the yield). The model sees this on the iteration where the
    budget fires; the harness then breaks with a ``YieldAction`` regardless
    of what the model produces.

    The block is designed to be **super-brief** so it can be replaced in
    the system prompt every iteration without token bloat. The full
    contract (all per-target limits) is only emitted on iteration 1; later
    iterations emit only the counts that changed. Callers handle the
    full-vs-delta distinction by passing the right ``state`` snapshot.

    Returns an empty string when there is nothing actionable to report
    (e.g. unlimited everything, iteration 1 with no per-target limits).
    """
    lines: list[str] = []
    # ── Iteration budget ──
    remaining_iterations = max_tool_iterations - state.iteration
    if remaining_iterations > 0:
        lines.append(f"iters: {remaining_iterations}/{max_tool_iterations}")
    # ── Per-step sub-budget (only when a WorkflowInstance is wired) ──
    if per_step_max_iterations > 0 and state.step_iteration > 0:
        step_remaining = per_step_max_iterations - state.step_iteration
        lines.append(f"step: {step_remaining}/{per_step_max_iterations} iters")
    if per_step_max_tool_calls > 0 and state.step_tool_calls > 0:
        step_tc_remaining = per_step_max_tool_calls - state.step_tool_calls
        lines.append(f"step: {step_tc_remaining}/{per_step_max_tool_calls} calls")
    # ── Per-target call limits (only tools with a non-zero cap) ──
    # Resolve the effective per-turn cap for each advertised tool. Only
    # tools that declare a finite ``max_calls_per_turn`` (tightened by the
    # execution policy) are surfaced — unlimited tools add noise. On
    # iteration 1, emit the full contract (used/max); on later iterations,
    # emit only tools that are *partially consumed* (used > 0 and remaining
    # > 0) so the block stays super-brief.
    from kernel.capability import resolve_capability_for_turn_limit

    per_target_parts: list[str] = []
    for schema in tool_schemas:
        name = schema.get("function", {}).get("name", "")
        if not name or name == _TRANSITION_TOOL_NAME or name == _YIELD_TOOL_NAME:
            continue
        target_def = resolve_capability_for_turn_limit(
            name,
            operation_catalog=operation_catalog,
            tool_catalog=tool_catalog,
        )
        if target_def is None:
            continue
        authored = int(getattr(target_def, "max_calls_per_turn", 0) or 0)
        if authored <= 0:
            continue
        max_calls = resolve_capability_invocation_limit(execution_policy, name, authored)
        if max_calls <= 0:
            continue
        used = state.call_ledger_counts.get(name, 0) if hasattr(state, "call_ledger_counts") else 0
        remaining = max(0, max_calls - used)
        # Iteration 1: emit the full contract (used=0 → remaining=max).
        # Later iterations: emit only partially-consumed tools (used > 0).
        if state.iteration <= 1 or used > 0:
            per_target_parts.append(f"{name}:{remaining}/{max_calls}")
    if per_target_parts:
        lines.append("calls: " + ", ".join(per_target_parts))
    # ── Lifecycle state ──
    # When the harness is about to yield (budget exhausted, stuck-in-stage
    # cap, or recovery stall), tell the model the harness will ask the user —
    # NOT "produce your answer now" (the old instruction that fought the
    # yield). The model sees this on the iteration where the budget fires;
    # the harness then breaks with a YieldAction regardless of what the
    # model produces.
    if budget_exhausted:
        lines.append("YIELD: budget exhausted — harness will ask the user")
    if not lines:
        return ""
    return "## Turn Lifecycle\n\n" + " · ".join(lines)


def _is_workflow_instance_complete(instance: Any) -> bool:
    """Return True when the workflow instance has no remaining steps.

    Thin wrapper around ``kernel.workflow_instance.is_complete`` with
    graceful degradation: any exception or missing attribute returns
    ``True`` so the per-step cap falls through to the existing final
    completion rather than synthesizing a handoff to a step that may not
    exist.
    """
    try:
        from kernel.workflow_instance import is_complete  # noqa: PLC0415

        return bool(is_complete(instance))
    except Exception:  # pylint: disable=broad-except
        return True


def _next_remaining_step(instance: Any) -> Any:
    """Return the first uncompleted ``WorkflowStep`` after the current one.

    Wraps ``kernel.workflow_instance.remaining_steps`` and selects the
    step immediately following ``current_step_id`` in declaration order
    (falling back to the first remaining step when the current step is
    already advanced or missing). Returns ``None`` when there are no
    remaining steps or on any exception.
    """
    try:
        from kernel.workflow_instance import remaining_steps  # noqa: PLC0415

        remaining = remaining_steps(instance)
        if not remaining:
            return None
        current = getattr(instance, "current_step_id", "") or ""
        # Prefer the step immediately after the current one in
        # declaration order; fall back to the first remaining step.
        for step in remaining:
            if step.step_id != current:
                return step
        return remaining[0]
    except Exception:  # pylint: disable=broad-except
        return None


def _resolve_yield_policy(mode_config: Any) -> YieldPolicy:
    """Return the ``YieldPolicy`` for one mode, tolerating missing config.

    The policy is authored in ``_mode.md`` front matter and loaded onto
    ``WorkflowModeConfig.yield_policy`` by ``kernel.ipsar_fsm``. When the
    config or policy is absent (e.g. a minimal test fixture), return the
    default-empty policy (inline asks allowed, no fallback) so the turn
    degrades to the legacy "ask in place" behavior rather than crashing.
    """
    if mode_config is None:
        return YieldPolicy()
    policy = getattr(mode_config, "yield_policy", None)
    if isinstance(policy, YieldPolicy):
        return policy
    return YieldPolicy()


def _next_remaining_step_mode_id(hooks: Any) -> str:
    """Return the mode id of the next uncompleted workflow step, or "".

    Thin wrapper around ``_next_remaining_step`` that reads the active
    ``WorkflowInstance`` from ``hooks.workflow_instance``. Used by the
    stuck-in-stage yield path to route a forced yield to the next step's
    mode when the current mode forbids inline asks (action modes).
    """
    if hooks.workflow_instance is None:
        return ""
    try:
        instance = hooks.workflow_instance()
    except Exception:  # pylint: disable=broad-except
        return ""
    step = _next_remaining_step(instance)
    return step.mode_id if step is not None else ""


# ── Public types ───────────────────────────────────────────────────────────


@dataclass
class TurnResult:
    """Result of one call to `run_turn()`."""

    response_text: str
    tool_call_count: int = 0
    called_tool_names: set[str] = field(default_factory=set)
    """Names of every tool actually dispatched this turn. Used by the
    fork-child validator to confirm an action mode's child invoked at
    least one of its authored action tools, not just read-only fetches."""
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
    per_model_usage: list[dict[str, Any]] = field(default_factory=list)
    """Per-model token breakdown for every model that was actually called
    during this turn. Each entry is keyed by the model's ``model_id`` and
    carries ``prompt_tokens`` / ``completion_tokens`` / ``total_tokens`` /
    ``cached_tokens`` / ``reasoning_tokens`` / ``duration_ms`` / ``latency_ms``.
    Entries for the same model_id across iterations are merged into one row.
    The host's token accounting reads this to populate ``model_token_totals``
    so ``/tokens`` reports per-model usage rather than attributing the whole
    turn to the default mode's resolved model. Empty when no model was
    called (e.g. input-guard block, pure operation mode)."""
    per_mode_usage: list[dict[str, Any]] = field(default_factory=list)
    """Per-mode token breakdown for every mode that actually ran during this
    turn. Mirrors ``per_model_usage`` but keyed by ``mode_id``. The host's
    token accounting reads this to populate ``mode_token_totals`` so
    ``/tokens`` reports the actual mode(s) that did the work rather than
    attributing the whole turn to the entry mode — the previous behavior
    misattributed kernel-driven recursive handoffs (plan → search → coding)
    to the entry mode. Empty when ``record_model_usage`` was never called
    with a mode_id, or no model was called this turn."""


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

    workflow_instance: Any | None = None
    """Callable returning the active ``WorkflowInstance`` for the current
    user/surface, or ``None`` when no multi-stage workflow is in flight. The
    instance is the durable inter-mode object that survives across modes,
    turns, and host restarts; ``chain_satisfied`` is hydrated from its
    ``completed_steps`` at turn start so a resumed turn does not re-suggest
    already-completed steps (the fix for the 2026-07-28 glm-5.2
    self-assessment stall where ``coding`` was never reached after resume).
    Default: None (ephemeral ``chain_satisfied`` behavior is preserved)."""

    workflow_instance_updater: Any | None = None
    """Callable receiving an updated ``WorkflowInstance`` after a successful
    ``transition_stage`` to a new mode, persisting progress so the next
    turn's ``workflow_instance`` hook returns the advanced instance. The
    host owns the instance lifecycle (create on turn 1, advance on handoff,
    clear on completion, load on boot). Default: None (transitions advance
    in-process ``chain_satisfied`` but the instance is not persisted)."""

    route_coordinator: RouteCoordinator | None = None
    """Optional route-coordinator backend that persists the next handoff
    target across turns. The kernel consults it for continuation prompts such
    as ``proceed`` and saves a target after a successful mid-turn handoff."""

    # ── Host state provider ───────────────────────────────────────────────
    state_provider: Any | None = None
    """Callable returning a ``UnifiedState`` for the current host session.
    When set, injected into ``OperationExecutionContext`` so operations that
    need cached runtime state can access it. Default: None."""

    pack_configuration_resolver: Any | None = None
    """Session-owned resolver for immutable, target-pack configuration."""

    approval_service: Any | None = None
    """Host-owned persistence and receipt service for exact-action approval."""

    recovery_service: Any | None = None
    """Host-owned durable service for interrupted dispatch work items."""

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

    # ── Memory recall (pre-turn injection) ────────────────────────────────
    memory_recall_enabled: bool = True
    """When True, the turn loop calls ``recall_memory`` during Phase 1 and
    injects the result into the user message as a ``## Memory Recall`` block.
    Gated independently of ``memory_enabled`` so hosts can opt out of the
    per-turn recall cost while keeping the memory backend active."""

    memory_recall_pre_turn_limit: int = 3
    """Maximum number of recall results to inject into the user message."""

    memory_recall_min_query_chars: int = 3
    """Skip pre-turn recall when the query is shorter than this many
    characters (e.g. ``"ok"``, ``"thanks"``)."""

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

    For gpt-5.x reasoning models the visible ``content`` can be empty while
    the model's substantive answer lives in ``reasoning_content``. We fold
    reasoning into ``content`` here (when content is empty) so the rest of
    the turn runner — which reads ``completion.content`` — sees the model's
    actual answer instead of falling through to "did not produce a final
    response." The raw reasoning is still available via
    ``self.reasoning_content`` for callers that want to distinguish the two.
    """

    __slots__ = ("_aggregated", "content", "reasoning_content", "tool_calls", "usage")

    def __init__(self, aggregated: StreamedCompletion) -> None:
        self._aggregated = aggregated
        self.reasoning_content = aggregated.reasoning_content
        # Fold reasoning into content when the visible answer is empty.
        self.content = aggregated.content or aggregated.reasoning_content
        self.tool_calls = aggregated.tool_calls
        self.usage = aggregated.usage

    def __repr__(self) -> str:
        return (
            f"StreamedCompletionAdapter(content={self.content!r}, "
            f"reasoning_content={self.reasoning_content!r}, "
            f"finish_reason={self._aggregated.finish_reason!r})"
        )


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


def _looks_like_continuation_query(query: str) -> bool:
    """Return True for continuation-style prompts such as ``proceed``.

    Matches the lowered query as a prefix against the shared resume-phrase
    vocabulary (:func:`kernel.routing_vocabulary.get_resume_phrases`), so
    ``"proceed with the current step in the plan"`` matches ``"proceed"``.
    The vocabulary is session-authored via ``routing.resume_phrases`` in
    ``session_context.md``; the historical default is used when no session
    vocabulary is wired. This keeps the detector in lock-step with the
    host's affirmative-resume path (``hosts.core.handle_message``), which
    reads the same registry.
    """
    lowered = (query or "").strip().lower()
    if not lowered:
        return False
    from kernel.routing_vocabulary import get_resume_phrases  # noqa: PLC0415

    return any(lowered.startswith(phrase) for phrase in get_resume_phrases())


def _coordinator_routing_hint(
    *,
    hooks: TurnHooks,
    mode_id: str,
    query: str,
) -> str:
    """Return a handoff hint from a persisted route coordinator when applicable.

    When a ``plan_workflow_op`` WorkflowInstance is active and has a
    current step, the instance is the source of truth for the next target
    (not the coordinator's saved decision, which points at the first step
    and goes stale after advancement). The instance's ``current_step_id``
    was advanced by ``advance_step`` when the previous step completed, so
    it names the step that should run next on a ``/proceed`` resume.
    """
    if hooks.route_coordinator is None or not _looks_like_continuation_query(query):
        return ""
    # ── Plan-instance-driven resume ──────────────────────────────────
    # When a plan_workflow_op instance is active, read the next target
    # from the instance's current step (the coordinator's saved
    # next_target is the FIRST step and goes stale after advancement).
    if hooks.workflow_instance is not None:
        try:
            _instance = hooks.workflow_instance()
        except Exception:  # pylint: disable=broad-except
            _instance = None
        if (
            _instance is not None
            and getattr(_instance, "spawned_from", "") == "plan_workflow_op"
            and getattr(_instance, "current_step_id", "")
        ):
            _step = (
                _instance.step_by_id(_instance.current_step_id)
                if hasattr(_instance, "step_by_id")
                else None
            )
            if _step is not None and _step.mode_id:
                return (
                    "## Routing Hint\n\n"
                    f"Continue with the next step in the planned workflow. "
                    f"Hand off to `{_step.mode_id}` by calling "
                    f"`transition_stage` with target_stage_id "
                    f"`{_step.mode_id}`."
                )
    # ── Coordinator-driven resume (legacy / non-plan) ────────────────
    request = {
        "mode_id": mode_id,
        "query": query,
        "surface_id": hooks.surface_id,
        "conversation_id": hooks.conversation_id,
        "user_id": hooks.user_id,
        "session_plan_digest": hooks.session_plan_digest,
    }
    try:
        decision = hooks.route_coordinator.load(request)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("route_coordinator.load failed: %s", exc)
        return ""
    target = (getattr(decision, "next_target", "") or "").strip()
    if not target:
        return ""
    return (
        "## Routing Hint\n\n"
        f"Continue with the previously planned next step. Hand off to `{target}` by "
        f"calling `transition_stage` with target_stage_id `{target}`."
    )


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
    lifecycle_state: Literal["working", "yielded", "committed", "stalled"] = "working",
    awaiting_input_question: str = "",
) -> CheckpointBundle | None:
    """Build a CheckpointBundle from the current host state.

    Reads ``UnifiedState`` via ``hooks.state_provider`` (if wired) and
    ``WorkflowSessionState`` via ``hooks.workflow_session`` (if wired).
    Returns None when neither is available — there's nothing to checkpoint.

    ``lifecycle_state`` / ``awaiting_input_question`` carry the unified turn
    lifecycle's terminal action so a yielded checkpoint is resumable by
    ``/proceed`` (the host loads the bundle, sees ``yielded``, injects the
    user's reply, and resumes the turn against the same ``WorkflowInstance``
    step). ``working`` is the legacy default (in-progress snapshot);
    ``committed`` marks a clean finish.
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
                logger.warning("checkpoint: state_provider dump failed: %s", exc)

    workflow_session_dict: dict[str, Any] = {}
    stage_id = ""
    turn_count = 0
    if is_workflow and hooks.workflow_session is not None:
        session = hooks.workflow_session()
        if session is not None:
            try:
                workflow_session_dict = session.model_dump(mode="json")
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("checkpoint: workflow_session dump failed: %s", exc)
            stage_id = str(getattr(session, "stage_id", "") or "")
            turn_count = int(getattr(session, "turn_count", 0) or 0)

    # ── Durable workflow-instance snapshot ───────────────────────────
    # Capture the active ``WorkflowInstance`` so a restore rehydrates the
    # inter-mode chain state (completed_steps, current_step_id) — the fix
    # for the glm-5.2 stall where a restored turn had no memory of which
    # chain steps had already run.
    workflow_instance_dict: dict[str, Any] = {}
    if hooks.workflow_instance is not None:
        try:
            instance = hooks.workflow_instance()
        except Exception:  # pylint: disable=broad-except
            instance = None
        if instance is not None:
            try:
                workflow_instance_dict = instance.model_dump(mode="json")
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("checkpoint: workflow_instance dump failed: %s", exc)

    if not unified_state_dict and not workflow_session_dict and not workflow_instance_dict:
        return None

    return CheckpointBundle(
        checkpoint_id=uuid.uuid4().hex[:16],
        created_at=datetime.now(UTC).isoformat(),
        mode_id=mode_id,
        stage_id=stage_id or (transitioned_to or ""),
        turn_count=turn_count,
        unified_state=unified_state_dict,
        workflow_session_state=workflow_session_dict,
        workflow_instance=workflow_instance_dict,
        lifecycle_state=lifecycle_state,
        awaiting_input_question=awaiting_input_question,
    )


def _build_recovery_identity(hooks: TurnHooks) -> RecoveryIdentity:
    """Build the exact identity scope for recovery records."""

    return RecoveryIdentity(
        workspace_id=hooks.workspace_id,
        user_id=hooks.user_id,
        user_slug=hooks.user_slug,
        surface_type=cast(ConversationSurface, hooks.surface_type),
        surface_id=hooks.surface_id,
        memory_scope=cast(MemoryScope, hooks.memory_scope),
        session_plan_digest=hooks.session_plan_digest,
    )


def _current_recovery_stage_id(*, hooks: TurnHooks, is_workflow: bool) -> str:
    """Return current workflow stage id when available, else empty."""

    if not is_workflow or hooks.workflow_session is None:
        return ""
    try:
        session = hooks.workflow_session()
    except Exception:  # pylint: disable=broad-except
        return ""
    if session is None:
        return ""
    return str(getattr(session, "stage_id", "") or "")


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


def _plan_steps_to_workflow_steps(
    plan_steps: list[dict[str, Any]],
) -> tuple[WorkflowStep, ...]:
    """Convert plan_workflow's ``steps`` dicts into ``WorkflowStep`` objects.

    Each plan step has ``mode_id``, ``description``, and ``rationale``. The
    ``WorkflowStep`` only carries ``mode_id`` and ``step_id`` (the
    description/rationale are for the user-visible plan block, not the
    instance). The last step is marked terminal. Step ids are deterministic
    via ``_step_id_for`` so a recompiled plan preserves ``completed_steps``
    by id intersection.
    """
    if not plan_steps:
        return ()
    steps: list[WorkflowStep] = []
    for idx, step in enumerate(plan_steps):
        mode_id = str(step.get("mode_id", "")).strip()
        if not mode_id:
            continue
        steps.append(
            WorkflowStep(
                step_id=_step_id_for(mode_id, idx),
                mode_id=mode_id,
                is_terminal=(idx == len(plan_steps) - 1),
                spawned_from="plan_workflow_op",
            )
        )
    return tuple(steps)


# ── Core turn runner ────────────────────────────────────────────────────────


async def run_turn(
    *,
    mode_id: str,
    query: str,
    hooks: TurnHooks,
    max_tool_iterations: int | None = None,
    per_step_max_iterations: int | None = None,
    per_step_max_tool_calls: int | None = None,
    preferred_model_id: str | None = None,
    turn_budget: TurnBudget | None = None,
    execution_policy: ExecutionPolicy | None = None,
    logger_override: logging.Logger | None = None,
    suppress_routing_hint: bool = False,
) -> TurnResult:
    """Execute one conversational turn with the full model+tool loop.

    ``suppress_routing_hint`` is set by the fork fan-out so child turns do
    not recompute and inject their own routing hints. A fork child already
    runs in its target mode by design; a hint telling it to hand off again
    creates a loop (the child sees the original multi-intent query, detects
    intents its mode doesn't satisfy, and is told to re-route — which a
    weak/strict model turns into an invalid-content 500). Fork children
    must run their mode to completion without re-routing.

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
            When ``None`` (the default), reads ``kernel.config.Settings.max_tool_iterations``
            (env var ``MAX_TOOL_ITERATIONS``, default 15). The execution
            policy may further tighten this via ``min()``.
        preferred_model_id: Optional per-turn model alias override supplied by
            the host when routing needs an explicit model designation
            (e.g. ``"discuss"`` for /discuss routing).
        turn_budget: Optional whole-turn resource budget. When set, the loop
            tracks tool calls, tokens, and wall time against the budget and
            forces a final no-tools completion when any limit is reached.
            When None, only ``max_tool_iterations`` bounds the loop (the
            historical behavior). The budget is the "ratio" knob: the host
            expresses the mix of inference and compute as one envelope.
        execution_policy: Optional host-resolved execution policy. In this
            migration phase, it is adapted to legacy knobs by tightening
            ``max_tool_iterations`` and populating an equivalent
            ``TurnBudget`` when one is not explicitly provided.
        logger_override: Optional logger for test capture.

    Returns:
        TurnResult with the final response text and loop statistics.

    Raises:
        NotImplementedError: If hooks.complete is None.
    """
    _log = logger_override or logger

    if max_tool_iterations is None:
        max_tool_iterations = _kernel_settings.max_tool_iterations
    if per_step_max_iterations is None:
        per_step_max_iterations = _kernel_settings.per_step_max_iterations
    if per_step_max_tool_calls is None:
        per_step_max_tool_calls = _kernel_settings.per_step_max_tool_calls

    if execution_policy is not None:
        if execution_policy.max_model_calls is not None:
            max_tool_iterations = min(max_tool_iterations, execution_policy.max_model_calls)
        if turn_budget is None:
            turn_budget = TurnBudget(
                max_tool_calls=execution_policy.max_capability_calls or 0,
                max_total_tokens=execution_policy.max_total_tokens or 0,
                max_wall_ms=execution_policy.max_wall_ms or 0,
            )

    mode_config = get_workflow_catalog().get(mode_id)

    if hooks.recovery_service is not None:
        try:
            await hooks.recovery_service.mark_running_indeterminate(_build_recovery_identity(hooks))
        except Exception as exc:  # pylint: disable=broad-except
            _log.warning("recovery startup reconciliation failed: %s", exc)

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
            per_step_max_iterations=per_step_max_iterations,
            per_step_max_tool_calls=per_step_max_tool_calls,
            preferred_model_id=preferred_model_id,
            turn_budget=turn_budget,
            execution_policy=execution_policy,
            logger_override=logger_override,
            suppress_routing_hint=suppress_routing_hint,
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
            per_step_max_iterations=per_step_max_iterations,
            per_step_max_tool_calls=per_step_max_tool_calls,
            preferred_model_id=preferred_model_id,
            turn_budget=turn_budget,
            execution_policy=execution_policy,
            logger_override=logger_override,
            suppress_routing_hint=suppress_routing_hint,
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
    per_step_max_iterations: int,
    per_step_max_tool_calls: int,
    preferred_model_id: str | None,
    turn_budget: TurnBudget | None,
    execution_policy: ExecutionPolicy | None,
    logger_override: logging.Logger | None,
    suppress_routing_hint: bool = False,
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
            per_step_max_iterations=per_step_max_iterations,
            per_step_max_tool_calls=per_step_max_tool_calls,
            preferred_model_id=preferred_model_id,
            turn_budget=turn_budget,
            execution_policy=execution_policy,
            logger_override=logger_override,
            suppress_routing_hint=suppress_routing_hint,
        )
    except ApprovalRequired as required:
        # Preserve the mid-turn handoff across the approval boundary. The
        # pending invocation carries the identity of the mode that owned
        # the effectful action (e.g. ``coding`` after a default -> coding
        # handoff). Without this, ``run_turn`` returns ``handoff_to=None``
        # and the host leaves ``current_mode_id`` at the turn's entry mode
        # (``default``). The subsequent ``/approve`` rebuilds the approval
        # identity from ``current_mode_id`` and the exact-match in
        # ``_identity_matches`` fails, stranding the approval. Surface the
        # owning mode as ``handoff_to`` so the host advances
        # ``current_mode_id`` before the user resumes — but only when it
        # differs from the entry mode, so an approval paused in the entry
        # mode does not fabricate a self-handoff.
        owning_mode = required.invocation.identity.mode_id or ""
        return TurnResult(
            response_text=render_approval_request(required.invocation),
            pending_approval=required.invocation,
            handoff_to=owning_mode if owning_mode and owning_mode != mode_id else None,
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
    per_step_max_iterations: int,
    per_step_max_tool_calls: int,
    preferred_model_id: str | None,
    turn_budget: TurnBudget | None,
    execution_policy: ExecutionPolicy | None,
    logger_override: logging.Logger | None,
    suppress_routing_hint: bool = False,
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

    # ── Mutable turn state (extracted to _TurnState) ───────────────────
    # Promoted from nested closures to a dataclass + module-level functions
    # in kernel.turn_phases._turn_state.  The state object is threaded
    # through every helper call instead of being captured by closures.
    from kernel.turn_phases._turn_state import (
        _TurnState,
    )
    from kernel.turn_phases._turn_state import (
        budget_forces_final as _budget_forces_final_fn,
    )
    from kernel.turn_phases._turn_state import (
        emit_stage as _emit_stage_fn,
    )
    from kernel.turn_phases._turn_state import (
        mcp_tool_executor as _mcp_tool_executor_fn,
    )
    from kernel.turn_phases._turn_state import (
        record_model_usage as _record_model_usage_fn,
    )
    from kernel.turn_phases._turn_state import (
        resolve_iteration_model as _resolve_iteration_model_fn,
    )

    _state = _TurnState()
    # Expose the budget caps on the turn state so ``emit_stage`` stamps
    # them on every stage payload. The transport uses them to render
    # graphical counters (pip rows / sparklines) against the correct
    # denominator. ``max_tool_iterations`` is the global iteration cap;
    # ``max_tool_calls`` comes from the per-turn ``TurnBudget`` (0 =
    # unlimited, in which case the transport falls back to a saturating
    # counter).
    _state.max_iterations = max_tool_iterations
    _state.max_tool_calls = turn_budget.max_tool_calls if turn_budget is not None else 0
    _state.max_handoffs = _kernel_settings.max_handoffs_per_turn
    _on_stage_hook = hooks.on_stage

    # Turn-runner guard caps, read once from the kernel settings singleton
    # (``kernel.config.Settings``). The three referenced multiple times
    # per turn are bound to locals to avoid repeated attribute lookups in
    # the hot loop; the rest are read directly from ``_kernel_settings``
    # at their single use site. See the Settings fields for env-var
    # overrides and the incidents each cap guards against.
    _max_handoffs_per_turn = _kernel_settings.max_handoffs_per_turn
    _hint_override_limit = _kernel_settings.hint_override_limit

    async def _emit_stage(label: str, data: dict[str, Any] | None = None) -> None:
        await _emit_stage_fn(_state, _on_stage_hook, _log, label, data)

    async def _mcp_tool_executor(tool_id: str, _ctx: Any) -> ToolResult:
        return await _mcp_tool_executor_fn(tool_id, _ctx, pool)

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
        result = await run_mode_entry_operations(
            mode_id,
            ctx,
            ledger=ledger,
            execution_policy=execution_policy,
        )
        # ── Kernel-driven handoff for planner action modes ──────────────
        # When an action mode's entry operation returns a routing target
        # (``output_json["target_stage_id"]``), the kernel drives the
        # handoff itself instead of asking the model to call
        # ``transition_stage``. This is the hard cutover from the old
        # contract (``plan`` mode's ``_mode.md`` told the model to call
        # ``transition_stage`` after ``plan_workflow``) to a kernel-driven
        # one: weak models that stop after planning — emitting a summary
        # like "the next step is `search`" without calling the tool — left
        # no persisted route state, so a follow-up ``proceed`` had nothing
        # to load and the chain stalled (the 2026-07-28 incident).
        #
        # The kernel persists the coordinator decision *from the plan
        # result* (not from a successful ``transition_stage`` tool call),
        # then recursively runs the turn against the target mode with
        # routing hints suppressed so the target does not re-plan. The
        # target mode's own entry operations run in the recursive call.
        # An empty ``target_stage_id`` (no routable mode matched, no model
        # wiring, or a parse failure) falls through to the user-facing
        # text return — the planner legitimately reports it cannot route.
        planned_target = str((result.output_json or {}).get("target_stage_id", "") or "").strip()
        # The visible plan block (rendered markdown) from plan_workflow's
        # output_text. When the model returned a ``steps`` array, this is a
        # ``## Plan`` block the user sees before the first step executes.
        # When the model returned no steps (weak-model fallback), this is
        # the single-target rationale (no plan block) — same as pre-steps.
        plan_block = (result.output_text or "").strip()
        plan_steps_raw = (result.output_json or {}).get("steps") or []
        if not isinstance(plan_steps_raw, list):
            plan_steps_raw = []
        if planned_target:
            # ── Persist the plan as a WorkflowInstance ───────────────────
            # When the model returned a steps array, create a
            # WorkflowInstance with spawned_from="plan_workflow_op" so the
            # plan survives across turns and is resumable with /proceed.
            # The instance's declared_steps drive the chain (replacing the
            # emergent post-handoff hint one-step-at-a-time chain). When
            # no steps were returned (weak-model fallback), skip instance
            # creation — the ad-hoc chain_satisfied path handles it.
            workflow_steps = _plan_steps_to_workflow_steps(plan_steps_raw)
            if workflow_steps and hooks.workflow_instance_updater is not None:
                try:
                    instance = WorkflowInstance(
                        declared_steps=workflow_steps,
                        current_step_id=workflow_steps[0].step_id,
                        current_mode_id=workflow_steps[0].mode_id,
                        spawned_from="plan_workflow_op",
                    )
                    # Persist via the updater hook (host owns lifecycle).
                    # The existing mid-turn handoff path calls the updater
                    # with just the instance (identity is host-internal);
                    # match that contract.
                    hooks.workflow_instance_updater(instance)
                    _log.info(
                        "persisted plan_workflow_op WorkflowInstance "
                        "instance_id=%s steps=%d first=%s",
                        instance.instance_id,
                        len(workflow_steps),
                        workflow_steps[0].mode_id,
                    )
                except Exception as wi_exc:  # pylint: disable=broad-except
                    _log.warning(
                        "WorkflowInstance creation/persist failed for plan: %s",
                        wi_exc,
                    )
            if hooks.route_coordinator is not None:
                try:
                    hooks.route_coordinator.save(
                        CoordinatorRouteDecision(
                            next_target=planned_target,
                            reason=(
                                result.output_json.get("rationale")
                                or f"planned handoff from {mode_id}"
                            ),
                        ),
                        {
                            "mode_id": planned_target,
                            "query": query,
                            "surface_id": hooks.surface_id,
                            "conversation_id": hooks.conversation_id,
                            "user_id": hooks.user_id,
                            "session_plan_digest": hooks.session_plan_digest,
                        },
                    )
                except Exception as coord_exc:  # pylint: disable=broad-except
                    _log.warning(
                        "route_coordinator.save failed for planned target %s: %s",
                        planned_target,
                        coord_exc,
                    )
            _log.info(
                "kernel-driven handoff from planner mode=%s target=%s "
                "(planned by entry op, not model transition_stage)",
                mode_id,
                planned_target,
            )
            # Run the first step. The plan block is prepended to the
            # recursive turn's response_text so the user sees the plan at
            # the top, followed by the first step's output.
            step_result = await run_turn(
                mode_id=planned_target,
                query=query,
                hooks=hooks,
                max_tool_iterations=max_tool_iterations,
                per_step_max_iterations=per_step_max_iterations,
                per_step_max_tool_calls=per_step_max_tool_calls,
                preferred_model_id=preferred_model_id,
                turn_budget=turn_budget,
                execution_policy=execution_policy,
                logger_override=logger_override,
                suppress_routing_hint=True,
            )
            # Prepend the plan block to the response so the user sees it
            # before the first step's output. Only prepend when there is
            # an actual plan block (steps were returned); otherwise the
            # rationale is not useful as a prefix to the step output.
            if plan_block and plan_steps_raw and step_result.response_text:
                step_result.response_text = f"{plan_block}\n\n{step_result.response_text}"
            elif plan_block and plan_steps_raw and not step_result.response_text:
                step_result.response_text = plan_block
            return step_result
        # ── Plan failure: hard stop with a structured recovery prompt ──
        # When the planner returned an empty ``target_stage_id`` (parse
        # failure, model error, no wiring), do NOT fall through to the
        # raw error text and do NOT let the post-handoff hint generator
        # re-hallucinate a target on the next turn (the 2026-07-30
        # plan→ponder→markdown bounce: plan failed with ``StubCompletion``,
        # the post-handoff hint re-fired telling the model to go to
        # ``discuss``, and the model improvised to ``ponder``). The turn
        # terminates with a deterministic recovery prompt that tells the
        # user plan failed and offers concrete next steps. Mark the model
        # unreliable for planning so the hint generator stops escalating
        # to ``plan`` for it (see ``routing_vocabulary.is_plan_reliable``).
        plan_reason = (
            str((result.output_json or {}).get("rationale", "") or "").strip()
            or "the planner returned no routing target"
        )
        plan_model_id = str(preferred_model_id or "")
        from kernel.routing_vocabulary import mark_plan_unreliable_for_model  # noqa: PLC0415

        mark_plan_unreliable_for_model(plan_model_id)
        _log.warning(
            "plan_workflow returned no target (mode=%s, model=%s, reason=%s); "
            "marking model unreliable for planning and terminating turn with "
            "recovery prompt",
            mode_id,
            plan_model_id,
            plan_reason,
        )
        # If the planner produced partial steps before failing (e.g. a
        # parse failure on a multi-step plan that partially populated),
        # include them in the recovery message so the user has context on
        # what was attempted. This is rare — most failures produce no
        # steps at all — but when it happens the partial plan is useful
        # diagnostic information for rephrasing.
        partial_steps = (result.output_json or {}).get("steps") or []
        partial_block = ""
        if isinstance(partial_steps, list) and partial_steps:
            partial_lines = ["The planner partially identified these steps before failing:"]
            for idx, s in enumerate(partial_steps, start=1):
                if isinstance(s, dict):
                    partial_lines.append(
                        f"  {idx}. {s.get('mode_id', '?')} — {s.get('description', '')}"
                    )
            partial_block = "\n\n" + "\n".join(partial_lines)
        recovery = (
            f"I attempted to plan this request in `{mode_id}` mode but could "
            f"not produce a routing decision (reason: {plan_reason}). This is "
            f"often a sign of a complex or ambiguous multi-step request."
            f"{partial_block}\n\n"
            f"You can:\n"
            f"  1. Rephrase the request with explicit sequential steps "
            f'(e.g. "search for X, then write a markdown report").\n'
            f"  2. Name the target mode directly (e.g. `/markdown ...`, "
            f"`/ponder ...`) to bypass planning.\n"
            f"  3. Ask me to discuss it first (`/discuss ...`) so I can "
            f"clarify what you want before committing to a workflow."
        )
        return TurnResult(
            response_text=recovery,
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
    pre_result = await run_mode_pre_turn_operations(
        mode_id,
        ctx,
        ledger=ledger,
        execution_policy=execution_policy,
    )
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
        if session is not None and session.stage_id in mode_config.stages:
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
    # ── Unified turn lifecycle: advertise the yield tool ──────────────
    # ``yield`` is a first-class terminal action the model calls to ask the
    # user a question. It is advertised whenever the mode's ``yield_policy``
    # allows it (``allowed=True``, the default for chat modes) OR when the
    # mode has a ``fallback_mode`` (action modes that forbid inline asks but
    # name a redirect target). When the model calls ``yield``, the dispatch
    # loop intercepts it (same pattern as ``transition_stage``) — it does
    # NOT dispatch to any executor; it breaks the loop with the question as
    # the turn content and stamps ``CheckpointBundle.lifecycle_state="yielded"``.
    _yield_policy_for_tools = _resolve_yield_policy(mode_config)
    if _yield_policy_for_tools.allowed or _yield_policy_for_tools.fallback_mode:
        tool_schemas.extend(get_yield_tool_schema())

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

    # ── Durable instance hydration ─────────────────────────────────────
    # When a ``WorkflowInstance`` is wired, hydrate ``chain_satisfied`` from
    # its ``completed_steps`` so a resumed turn does not re-suggest
    # already-completed chain steps. The instance is the durable form of this
    # accumulator; ``chain_satisfied`` is its in-memory projection. This runs
    # BEFORE the routing-hint computation below so the hint sees the hydrated
    # gap (the remaining consumer intents, not the already-done producers).
    # This is the fix for the glm-5.2 self-assessment stall: after
    # ``proceed``, the instance's completed research/write steps are
    # rehydrated and the router names the remaining consumer (coding) as the
    # gap instead of re-offering the first producer (search).
    if hooks.workflow_instance is not None:
        try:
            _active_instance = hooks.workflow_instance()
        except Exception:  # pylint: disable=broad-except
            _active_instance = None
        if _active_instance is not None:
            for _step_id in getattr(_active_instance, "completed_steps", ()):
                _step = (
                    _active_instance.step_by_id(_step_id)
                    if hasattr(_active_instance, "step_by_id")
                    else None
                )
                if _step is not None:
                    _state.chain_satisfied.update(_step.satisfies_intents)

    # Unified FSM routing: capability-manifest-aware proactive hint injection.
    # Compare the manifest against the user's query — if the current mode
    # lacks tools the query seems to need, inject a routing hint into the
    # system prompt. This is proactive (zero wasted iterations) and zero-cost
    # when the current mode can handle the request. See docs/ROUTING_RFC_HANDOFF.md.
    #
    # Suppressed for fork children: a fork child already runs in its target
    # mode by design. A hint telling it to hand off again creates a loop
    # (the child sees the original multi-intent query, detects intents its
    # mode doesn't satisfy, and is told to re-route — which a weak/strict
    # model turns into an invalid-content 500). See _run_fork_child_turn.
    routing_hint = ""
    if not suppress_routing_hint:
        routing_hint = _compute_capability_routing_hint(
            mode_id=mode_id,
            query=query,
            capability_manifest=capability_manifest,
            routing_targets=get_routing_targets(mode_id) if mode_config else [],
            mode_config=mode_config,
            # Pass the hydrated chain memory so a resumed turn's hint names
            # the remaining gap (not already-completed steps). Without this
            # the turn-start hint re-suggests the first producer even when
            # the instance marks it complete (the glm-5.2 stall).
            extra_satisfied=_state.chain_satisfied,
            planner_model_id=str(preferred_model_id or ""),
        )
    coordinator_hint = _coordinator_routing_hint(
        hooks=hooks,
        mode_id=mode_id,
        query=query,
    )
    if coordinator_hint:
        routing_hint = coordinator_hint
    if routing_hint:
        _log.info(
            "routing hint injected mode=%s hint_len=%d hint=%s",
            mode_id,
            len(routing_hint),
            routing_hint[:200],
        )
        system_prompt = system_prompt + "\n\n" + routing_hint
        # ── Forced-fork enforcer: precompute the deterministic fork target.
        # When the routing hint names a ``fork:`` target, the kernel already
        # decided the routing. The model is the executor. If it refuses to
        # call ``transition_stage`` — instead burning iterations on read-only
        # tools or returning empty completions (the 2026-07-27 incident) —
        # the turn loop forces the fork itself rather than begging via
        # nudges. Stored on ``_state`` and consumed in the tool-dispatch
        # branch; re-computed after each handoff by the chained-hint block.
        _state.forced_fork_target = (
            _compute_fork_target_string(
                query=query,
                routing_targets=get_routing_targets(mode_id) if mode_config else [],
                mode_config=mode_config,
            )
            or ""
        )
        _state.non_fork_tool_dispatches = 0
        # ── Forced-plan enforcer: precompute the plan-escalation target.
        # When the routing hint names ``plan`` as the target (3+ gap
        # intents → plan-mode escalation), the kernel already decided the
        # routing. The model is the executor. If it refuses to call
        # ``transition_stage`` — instead burning iterations on read-only
        # tools in the entry mode (the 2026-07-30 400k-token burn: 15
        # iterations of tool-cycling in ``default`` before the model
        # finally handed off to ``plan``, which then failed to parse) —
        # the turn loop forces the plan handoff itself after a small cap
        # of non-transition tool dispatches. This bounds the pre-plan
        # token burn to roughly ``cap × per-iteration-tokens`` instead
        # of the full ``max_tool_iterations`` budget. Mirrors the
        # forced-fork enforcer for the plan-escalation path.
        _state.forced_plan_target = "plan" if _extract_hint_target(routing_hint) == "plan" else ""
        _state.non_plan_tool_dispatches = 0
    else:
        _log.info(
            "no routing hint mode=%s (satisfies_intents=%s routing_targets=%s)",
            mode_id,
            getattr(mode_config, "satisfies_intents", None),
            get_routing_targets(mode_id) if mode_config else [],
        )

    user_context = assemble_user(mode_id, query, pre_briefing, capability_manifest)

    # ── Pre-turn memory recall ────────────────────────────────────────
    # Opportunistic: fires recall_memory before the model call so Lisa
    # surfaces relevant memories in the user message. Gated on
    # memory_recall_enabled and a minimum query length so trivial queries
    # ("ok", "thanks") don't waste a backend round-trip. A down memory
    # backend is logged and swallowed — it never blocks the turn.
    memory_snippet = ""
    if hooks.memory_recall_enabled and query and len(query) >= hooks.memory_recall_min_query_chars:
        try:
            recall_ctx = _build_minimal_context(
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
            recall_ctx.params = {
                "query": query,
                "limit": hooks.memory_recall_pre_turn_limit,
                "deep": False,
            }
            recall_result = await exec_op("recall_memory", recall_ctx)
            if recall_result.output_text:
                memory_snippet = recall_result.output_text
                _log.debug(
                    "pre-turn recall mode=%s chars=%d",
                    mode_id,
                    len(memory_snippet),
                )
        except Exception as exc:  # pylint: disable=broad-except
            _log.warning("pre-turn memory recall failed: %s", exc)

    # ── Phase 2: Tool loop ────────────────────────────────────────────
    # Save the base system prompt (without any budget briefing) so the
    # per-iteration budget briefing can be swapped in/out without
    # accumulating stale copies. The briefing is appended to the base
    # prompt each iteration; the base is the stable contract (mode
    # instruction, stage instruction, routing hint).
    _base_system_prompt = system_prompt
    _budget_briefing_marker = "\n\n## Turn Lifecycle\n\n"
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

    # required_capabilities are derived once from the mode config. They
    # default to empty — a router that does not need them ignores them.
    # Initialized before the capability gate so the router's iteration-1
    # selection (below) can read them off ``_state``.
    required_capabilities: tuple[str, ...] = ()
    if mode_config is not None:
        required_capabilities = tuple(getattr(mode_config, "required_capabilities", ()) or ())
    _state.required_capabilities = required_capabilities

    # ── Pre-gate router escalation (iteration 1) ───────────────────────
    # Run the model router once for iteration 1 *before* the capability
    # gate. The gate is documented as the safety net behind the router:
    # the router's ``"code" in route.required_capabilities → <model>``
    # rule should escalate to a capable model first, and the gate catches
    # the cases where the router declined or returned None. Running the
    # gate on the static preferred model before the router (the previous
    # ordering) hard-blocked turns the router would have rescued — e.g.
    # a default model lacking ``code`` entering coding mode with a router
    # rule that re-routes to a code-capable model. The gate now validates
    # the router-resolved model; if the router is absent or declines,
    # ``resolve_iteration_model`` falls back to ``model_def`` and the
    # gate retains its original advisory/hard-block semantics.
    _state.iteration = 1
    _resolved_for_gate = _resolve_iteration_model_fn(
        _state,
        1,
        hooks,
        preferred_model_id,
        resolve_model,
        model_def,
        query,
        mode_id,
        turn_budget,
        _log,
    )
    if _resolved_for_gate is not None:
        model_def = _resolved_for_gate
    # Cache the pre-gate iteration-1 resolution so the tool loop reuses it
    # instead of re-invoking the router (which would double-run the CEL
    # rules and could re-emit a terminal action). ``_pregate_decision``
    # carries any ``finish``/``transition_to`` decision the router made.
    _pregate_iteration1_model = model_def
    _pregate_decision = _state.pending_router_decision
    _state.iteration = 0
    _state.effective_model = model_def

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
    # ── Unified turn lifecycle ───────────────────────────────────────────
    # ``_yielded`` is set True when the loop breaks via a forced or
    # model-emitted ``YieldAction``. The checkpoint layer reads it to stamp
    # ``CheckpointBundle.lifecycle_state`` (``yielded`` vs ``committed``).
    _yielded = False
    # total_tool_calls, iteration, and per-model usage live on _state
    # (initialized to 0/empty by _TurnState).
    # Cumulative set of intents satisfied by every mode entered during this
    # turn (the entry mode + each handoff target). This prevents a sequential
    # chain from re-suggesting a step that already ran: after default → search
    # → coding, "research" was satisfied by search even though coding's own
    # ``satisfies_intents`` is only ["coding"]. Without this accumulator the
    # post-coding hint would re-detect "research" from the original query text
    # and re-suggest search — an infinite loop. The accumulator is the turn's
    # memory of which chain steps have already completed.
    _state.chain_satisfied = set()
    if mode_config is not None:
        _state.chain_satisfied.update(mode_config.satisfies_intents)
    # Per-step iteration sub-budget counters. Reset to 0 here (the entry
    # mode is step 1; the first loop increment moves them to 1) and reset
    # again on every successful ``transition_stage`` handoff so each step
    # in a chained workflow gets its own budget (Phase 3). When no
    # ``WorkflowInstance`` is wired these are unused — the global
    # ``max_tool_iterations`` alone governs the loop.
    _state.step_iteration = 0
    _state.step_tool_calls = 0

    # ``required_capabilities`` and ``_state.required_capabilities`` were
    # initialized before the pre-gate router escalation (above). The
    # router-relevant ``_state.effective_model`` was seeded there with the
    # router-resolved iteration-1 model so subsequent iterations observe
    # the correct preceding-model metadata.

    def _record_model_usage(
        model_def: Any,
        *,
        p: int,
        c: int,
        t: int,
        ct: int,
        rt: int,
        d: float,
        lat: int,
    ) -> None:
        """Merge one completion's counters into the per-model accumulator.

        Also accumulates into ``per_mode_usage`` keyed by this turn's
        ``mode_id`` so the host can attribute tokens to the mode that
        actually ran the completion (not the entry mode)."""
        _record_model_usage_fn(
            _state, model_def, p=p, c=c, t=t, ct=ct, rt=rt, d=d, lat=lat, mode_id=mode_id
        )

    def _budget_forces_final() -> bool:
        """Return True when the turn budget is exhausted."""
        return _budget_forces_final_fn(_state, turn_budget, turn_start_mono)

    def _resolve_iteration_model(iteration_num: int) -> Any:
        # Reuse the pre-gate iteration-1 resolution so the router runs once
        # per iteration, not twice. The pre-gate pass already evaluated the
        # CEL rules for iteration 1 and captured any terminal action.
        if iteration_num == 1 and _pregate_iteration1_model is not None:
            _state.pending_router_decision = _pregate_decision
            return _pregate_iteration1_model
        return _resolve_iteration_model_fn(
            _state,
            iteration_num,
            hooks,
            preferred_model_id,
            resolve_model,
            model_def,
            query,
            mode_id,
            turn_budget,
            _log,
        )

    tools_dispatched_ever = False  # tracks if *any* tool was called this turn
    # ── Unified turn lifecycle: pending yield ───────────────────────────
    # When the harness forces a yield (stuck-in-stage cap, iteration cap, or
    # recovery stall), it sets ``_pending_yield``. The loop checks this after
    # the model produces its next completion: if set, the loop breaks with the
    # yield's question as the turn's content and stamps
    # ``CheckpointBundle.lifecycle_state="yielded"``. This makes "ask"
    # distinguishable from "done" (the 2026-07-30 bug: both were "no tool
    # calls"). ``None`` means no forced yield is pending; the model works
    # normally.
    _pending_yield: YieldAction | None = None

    while _state.iteration < max_tool_iterations:
        _state.iteration += 1
        _state.step_iteration += 1

        # ── Unified turn lifecycle: recovery-stall check ──────────────
        # Before any other guard fires, check if a stalled ``RecoveryRecord``
        # blocks the workflow (``state="stalled"``, ``retry_policy="confirm"``,
        # ``blocks_workflow=True``). If so, force a yield naming the stall so
        # the user can retry/skip. This is the resolver's "recovery-stall"
        # priority branch made live in the loop — previously the turn loop
        # created and settled records but never read them back.
        if _pending_yield is None and hooks.recovery_service is not None:
            try:
                _open_records = await hooks.recovery_service.list_open(
                    _build_recovery_identity(hooks)
                )
            except Exception:  # pylint: disable=broad-except
                _open_records = []
            _stall_record = next(
                (
                    r
                    for r in _open_records
                    if getattr(r, "state", "") == "stalled"
                    and getattr(r, "retry_policy", "") == "confirm"
                    and getattr(r, "blocks_workflow", False)
                ),
                None,
            )
            if _stall_record is not None:
                _yield_policy = _resolve_yield_policy(mode_config)
                _yield_target = _yield_policy.fallback_mode or (
                    _next_remaining_step_mode_id(hooks) or "discuss"
                    if not _yield_policy.allowed
                    else ""
                )
                _stall_detail = (
                    _stall_record.last_error
                    or f"{_stall_record.target_kind} {_stall_record.target_id} stalled"
                )
                _pending_yield = YieldAction(
                    question=(
                        f"A previous step stalled and needs your input to continue. "
                        f"Stall: {_stall_detail}. Retry, skip, or revise?"
                    ),
                    context=(
                        f"Recovery record blocks workflow step "
                        f"'{getattr(_stall_record, 'step_id', '')}'."
                    ),
                    target_mode_id=_yield_target,
                )
                _log.warning(
                    "turn lifecycle: recovery stall blocks workflow "
                    "(mode=%s, target=%s, step=%s); forcing yield",
                    mode_id,
                    _stall_record.target_id,
                    getattr(_stall_record, "step_id", ""),
                )

        # ── Per-step iteration sub-budget (Phase 3) ───────────────────
        # When the current workflow step has exhausted its per-step
        # iteration or tool-call budget but the global turn cap has not
        # fired, synthesize a handoff to the next remaining step instead
        # of letting the model keep looping in the same mode (the
        # 2026-07-28 glm-5.2 stall: a 6-step chain starved its final
        # ``markdown`` step because earlier steps consumed the whole
        # shared 15-iteration budget). We reuse the existing
        # ``pending_planned_handoff`` synthesis block below: set the next
        # step's mode_id and ``continue`` so the synth fires on the next
        # loop iteration. When there is no active ``WorkflowInstance`` or
        # no remaining step (the instance is complete), fall through to
        # the existing empty-content final-completion fallback — the
        # per-step cap then behaves as a tighter global cap, which is
        # acceptable for a complete workflow (it should produce an
        # answer, not loop forever; the global cap is the backstop).
        _step_budget_exhausted = (
            _state.step_iteration > per_step_max_iterations
            or _state.step_tool_calls >= per_step_max_tool_calls
        )
        if _step_budget_exhausted and hooks.workflow_instance is not None:
            try:
                _wi = hooks.workflow_instance()
            except Exception:  # pylint: disable=broad-except
                _wi = None
            if _wi is not None and not _is_workflow_instance_complete(_wi):
                _next_step = _next_remaining_step(_wi)
                if _next_step is not None:
                    _log.info(
                        "per-step budget exhausted (step_iteration=%d, "
                        "step_tool_calls=%d, mode=%s, global_iteration=%d/%d); "
                        "synthesizing handoff to next step %s",
                        _state.step_iteration,
                        _state.step_tool_calls,
                        mode_id,
                        _state.iteration,
                        max_tool_iterations,
                        _next_step.mode_id,
                    )
                    _state.pending_planned_handoff = _next_step.mode_id
                    # Reset the per-step counters now (rather than waiting
                    # for the post-handoff block) so the cap check does not
                    # re-fire on the next iteration before the synth block
                    # below can process the pending handoff.
                    _state.step_iteration = 0
                    _state.step_tool_calls = 0
                    continue
                # No remaining step but instance not complete — fall
                # through to final completion (decision: B).
            _wi = None  # explicit fall-through

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
                        "_state.iteration": _state.iteration,
                    },
                )
                if turns_in_stage >= _kernel_settings.stage_nudge_cap:
                    # ── Unified turn lifecycle: cap forces a yield, not a silent final ──
                    # The 2026-07-30 fix: hitting the stuck-in-stage cap used to set
                    # ``budget_exhausted = True`` (strip tools → force a truncated final
                    # that cut off the user's experience). The unified lifecycle makes
                    # "ask" distinguishable from "done": the cap now forces a
                    # :class:`kernel.models.YieldAction` so the user gets a question
                    # they can answer to redirect, instead of a truncated response.
                    # The resolver (``resolve_lifecycle_action``) owns this decision in
                    # the full design; this branch is the in-loop application of its
                    # "stuck-in-stage cap" priority until the full cutover lands.
                    _log.warning(
                        "workflow mode=%s stuck in stage=%s for %d turns; forcing yield",
                        mode_id,
                        wf_session.stage_id,
                        turns_in_stage,
                    )
                    _yield_policy = _resolve_yield_policy(mode_config)
                    _yield_target = _yield_policy.fallback_mode or (
                        _next_remaining_step_mode_id(hooks) or "discuss"
                        if not _yield_policy.allowed
                        else ""
                    )
                    _yield_question = _yield_policy.default_question or (
                        f"I've been working in the '{wf_session.stage_id}' stage "
                        f"for {turns_in_stage} turns and haven't converged. "
                        f"Can you clarify what you need?"
                    )
                    _pending_yield = YieldAction(
                        question=_yield_question,
                        context=f"stuck in stage '{wf_session.stage_id}' for {turns_in_stage} turns (cap)",
                        target_mode_id=_yield_target,
                    )
                    # ``budget_exhausted`` is NOT set here: the ``_pending_yield``
                    # takes over at the loop break and surfaces the question. The
                    # old strip-tools path was redundant with the yield.
                elif turns_in_stage >= _kernel_settings.stage_nudge_threshold:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"You have been in the '{wf_session.stage_id}' stage "
                                f"for {turns_in_stage} turns. If you have enough evidence, "
                                f"call transition_stage to advance. Otherwise, call yield "
                                f"with a clarifying question for the user."
                            ),
                        }
                    )

        # ── Unified turn lifecycle: resource-budget exhaustion forces a yield ──
        # The old ``_budget_forces_final()`` path stripped tools to force a silent
        # final when ``TurnBudget`` limits (tool calls, tokens, wall time) were
        # exhausted. The unified lifecycle makes this consistent with the
        # iteration cap: exhaustion forces a ``YieldAction`` so the user can
        # redirect, instead of a truncated response. When no ``TurnBudget`` is
        # wired (the common case — ``turn_budget`` is None), this is a no-op and
        # the loop runs under ``max_tool_iterations`` alone.
        budget_exhausted = False
        if _budget_forces_final() and _pending_yield is None:
            _log.info(
                "turn budget exhausted (mode=%s, tool_calls=%d, tokens=%d); forcing yield",
                mode_id,
                _state.total_tool_calls,
                _state.total_tokens,
            )
            _yield_policy = _resolve_yield_policy(mode_config)
            _yield_target = _yield_policy.fallback_mode or (
                _next_remaining_step_mode_id(hooks) or "discuss"
                if not _yield_policy.allowed
                else ""
            )
            _pending_yield = YieldAction(
                question=(
                    _yield_policy.default_question
                    or "I've hit a resource budget for this turn. How would you like me to proceed?"
                ),
                context="turn budget exhausted (tool calls / tokens / wall time)",
                target_mode_id=_yield_target,
            )
            # Strip tools on the final completion so the model produces the
            # yield question text (the break applies ``_pending_yield``).
            budget_exhausted = True
        effective_tools = None if budget_exhausted else (tool_schemas if tool_schemas else None)

        # ── Per-iteration budget briefing ───────────────────────────
        # Inject a compact, per-iteration budget status block into the
        # system prompt so the model can self-scope its work within the
        # remaining iteration / tool-call / per-target envelope. Only
        # discrete, actionable budgets are surfaced (iterations, tool
        # calls, per-target call limits) — never continuous budgets
        # (tokens, wall time) which the model cannot reliably convert
        # into action and which risk anchoring the model to "use" the
        # budget it sees. The base system prompt is preserved; the
        # briefing is appended fresh each iteration (the previous
        # iteration's briefing is stripped via the marker so it does not
        # accumulate). Gated by ``budget_briefing_enabled`` (default
        # True) so hosts can opt out. See ``_build_budget_briefing``.
        if _kernel_settings.budget_briefing_enabled:
            _briefing = _build_budget_briefing(
                state=_state,
                max_tool_iterations=max_tool_iterations,
                per_step_max_iterations=per_step_max_iterations,
                per_step_max_tool_calls=per_step_max_tool_calls,
                tool_schemas=tool_schemas,
                operation_catalog=operation_catalog,
                tool_catalog=tool_catalog,
                execution_policy=execution_policy,
                budget_exhausted=budget_exhausted,
            )
            # Strip any previous briefing (by marker) and append the
            # fresh one. The base prompt is the stable contract; the
            # briefing is the per-iteration delta. The marker includes
            # the leading ``\n\n`` separator so the split cleanly
            # separates the base from the briefing.
            _current_sys = str(messages[0].get("content", ""))
            if _budget_briefing_marker in _current_sys:
                _current_sys = _current_sys.split(_budget_briefing_marker, 1)[0]
            if _briefing:
                messages[0]["content"] = _current_sys + "\n\n" + _briefing
            else:
                messages[0]["content"] = _current_sys

        effective_model_def = _resolve_iteration_model(_state.iteration)
        # ── Router terminal actions ──────────────────────────────────
        # When the model router returns a ``finish`` or ``transition_to``
        # action, the turn loop forces a final no-tools completion instead
        # of continuing the tool loop. ``finish`` strips tools immediately;
        # ``transition_to`` first moves the active workflow to the named
        # terminal stage (validated against the workflow's allowed
        # transitions), then strips tools so the model produces the stage's
        # closing summary. Both reuse the existing budget-finalization path
        # (tools=None → model must respond with text).
        _router_decision = _state.pending_router_decision
        if _router_decision is not None:
            if _router_decision.finish.strip():
                _log.info(
                    "router finish action (iteration=%d, reason=%s); forcing final",
                    _state.iteration,
                    _router_decision.finish,
                )
                effective_tools = None
            elif _router_decision.transition_to.strip():
                _target_stage = _router_decision.transition_to.strip()
                if is_workflow and hooks.workflow_session is not None:
                    _wf_session = hooks.workflow_session()
                    if _wf_session is not None:
                        try:
                            _transition_req = WorkflowTransitionRequest(
                                target_stage_id=_target_stage,
                                rationale=f"router transition_to: {_router_decision.reason}",
                            )
                            _updated_session = apply_transition(_wf_session, _transition_req)
                            if hooks.workflow_session_updater is not None:
                                hooks.workflow_session_updater(_updated_session)
                            transitioned_to = _updated_session.stage_id
                            _log.info(
                                "router transition_to action (iteration=%d, "
                                "stage=%s); forcing final summary",
                                _state.iteration,
                                _target_stage,
                            )
                        except Exception as trans_exc:  # pylint: disable=broad-except
                            _log.warning(
                                "router transition_to to stage %r failed: %s; "
                                "falling back to normal completion",
                                _target_stage,
                                trans_exc,
                            )
                    else:
                        _log.warning(
                            "router transition_to action but no active workflow session; ignoring"
                        )
                else:
                    _log.warning("router transition_to action but not in a workflow mode; ignoring")
                effective_tools = None

        # ── Kernel-driven planned handoff synthesis ───────────────────
        # When a planner action mode's entry op (e.g. ``plan_workflow``)
        # returned a ``target_stage_id`` during the previous iteration's
        # mid-turn handoff, the kernel drives the next handoff itself
        # instead of calling the model. The model in ``plan`` mode
        # reliably stops after planning (the 2026-07-28 stall), so we
        # synthesize a ``transition_stage`` tool call with the planned
        # target and dispatch it through the existing tool-dispatch path.
        # This mirrors the forced-fork synthesizer but fires
        # deterministically from the plan result. Clear the pending target
        # first so the synthesized call cannot re-enter this branch; if
        # the new handoff target is itself a planner, its own entry op
        # sets a fresh ``pending_planned_handoff`` for the next link.
        _planned_handoff_target = _state.pending_planned_handoff
        _state.pending_planned_handoff = ""
        if _planned_handoff_target:
            _log.info(
                "kernel synthesizing planned transition_stage target=%s (mode=%s, iteration=%d)",
                _planned_handoff_target,
                mode_id,
                _state.iteration,
            )
            _synth_planned_call = {
                "id": f"planned_handoff_{_state.iteration}",
                "type": "function",
                "function": {
                    "name": _TRANSITION_TOOL_NAME,
                    "arguments": json.dumps(
                        {
                            "target_stage_id": _planned_handoff_target,
                            "rationale": (
                                "Kernel-driven handoff from the plan "
                                "result; the model is not asked to call "
                                "transition_stage."
                            ),
                        }
                    ),
                },
            }
            completion = SimpleNamespace(
                content="",
                tool_calls=[_synth_planned_call],
                model=getattr(effective_model_def, "model_id", ""),
            )
            prompt_tokens = completion_tokens = completion_total_tokens = 0
            cached_tokens = reasoning_tokens = 0
            duration_ms = 0.0
            latency_ms = 0
        else:
            await _emit_stage(
                "model_call",
                {
                    "mode_id": mode_id,
                    "model_id": getattr(effective_model_def, "model_id", ""),
                    "provider": getattr(effective_model_def, "provider", ""),
                    "reasoning_model": bool(getattr(effective_model_def, "reasoning_model", False)),
                    "_state.iteration": _state.iteration,
                    "max_iterations": max_tool_iterations,
                    "tool_calls_so_far": _state.total_tool_calls,
                    "tools_offered": bool(effective_tools),
                },
            )
            if hooks.complete_stream is not None:
                # ── Streaming path ──────────────────────────────────────
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
        _state.total_prompt_tokens += prompt_tokens
        _state.total_completion_tokens += completion_tokens
        _state.total_tokens += completion_total_tokens
        _state.total_cached_tokens += cached_tokens
        _state.total_reasoning_tokens += reasoning_tokens
        _state.total_duration_ms += duration_ms
        _state.total_latency_ms += latency_ms
        _record_model_usage(
            effective_model_def,
            p=prompt_tokens,
            c=completion_tokens,
            t=completion_total_tokens,
            ct=cached_tokens,
            rt=reasoning_tokens,
            d=duration_ms,
            lat=latency_ms,
        )

        await _emit_stage(
            "model_call_complete",
            {
                "mode_id": mode_id,
                "model_id": getattr(effective_model_def, "model_id", ""),
                "reasoning_model": bool(getattr(effective_model_def, "reasoning_model", False)),
                "_state.iteration": _state.iteration,
                "max_iterations": max_tool_iterations,
                "tool_calls": len(getattr(completion, "tool_calls", None) or []),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "_state.total_tokens": completion_total_tokens,
                "duration_ms": duration_ms,
            },
        )

        assistant_msg: dict[str, Any] = {"role": "assistant"}
        # Use the canonical extractor so a reasoning model that put its
        # answer in ``reasoning_content`` (empty ``content``) still records
        # the visible text in the message history for subsequent iterations.
        _assistant_text = completion_content(completion)
        if _assistant_text:
            assistant_msg["content"] = _assistant_text
        tool_calls = getattr(completion, "tool_calls", None) or []
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        if hasattr(completion, "model"):
            assistant_msg["extra"] = {
                "provenance": {"source": "observed", "model": getattr(completion, "model", "")}
            }
        messages.append(assistant_msg)

        if not tool_calls:
            # ── Forced-fork enforcer (empty-completion path) ───────────
            # The model returned no tool calls while a deterministic fork
            # target is pending. This is the 2026-07-27 Run B failure:
            # ``plan`` mode returned empty content on iteration 7 with a
            # fork hint active, ending the turn with no fork ever running
            # and the iteration-cap fallback hallucinating the tool call
            # as prose. Synthesize the ``transition_stage(fork:...)`` call
            # ourselves and dispatch it on this iteration instead of
            # accepting the empty turn.
            _synthesized_fork = False
            if _state.forced_fork_target:
                _log.warning(
                    "model returned no tool calls with a forced fork "
                    "pending (mode=%s, iteration=%d); kernel synthesizing "
                    "fork target=%s",
                    mode_id,
                    _state.iteration,
                    _state.forced_fork_target,
                )
                synth_call = {
                    "id": f"forced_fork_empty_{_state.iteration}",
                    "type": "function",
                    "function": {
                        "name": _TRANSITION_TOOL_NAME,
                        "arguments": json.dumps(
                            {
                                "target_stage_id": _state.forced_fork_target,
                                "rationale": (
                                    "Kernel-forced fork: the model returned no "
                                    "tool calls after the routing hint named "
                                    "this fork target."
                                ),
                            }
                        ),
                    },
                }
                # Append the synthesized tool call to the last assistant
                # message (which had no tool_calls) so the tool result
                # message we attach below stays well-formed.
                assistant_msg["tool_calls"] = [synth_call]
                tool_calls = [synth_call]
                _state.forced_fork_target = ""
                _state.non_fork_tool_dispatches = 0
                _synthesized_fork = True

            # ── Forced-plan enforcer (empty-completion path) ──────────
            # The model returned no tool calls while a plan-escalation
            # target is pending. Synthesize the
            # ``transition_stage(plan)`` call ourselves and dispatch it
            # on this iteration instead of accepting the empty turn
            # (mirrors the forced-fork empty-completion synth).
            _synthesized_plan = False
            if _state.forced_plan_target:
                _log.warning(
                    "model returned no tool calls with a forced plan "
                    "pending (mode=%s, iteration=%d); kernel synthesizing "
                    "plan handoff target=%s",
                    mode_id,
                    _state.iteration,
                    _state.forced_plan_target,
                )
                synth_plan_call = {
                    "id": f"forced_plan_empty_{_state.iteration}",
                    "type": "function",
                    "function": {
                        "name": _TRANSITION_TOOL_NAME,
                        "arguments": json.dumps(
                            {
                                "target_stage_id": _state.forced_plan_target,
                                "rationale": (
                                    "Kernel-forced plan handoff: the model "
                                    "returned no tool calls after the routing "
                                    "hint named plan as the target."
                                ),
                            }
                        ),
                    },
                }
                assistant_msg["tool_calls"] = [synth_plan_call]
                tool_calls = [synth_plan_call]
                _state.forced_plan_target = ""
                _state.non_plan_tool_dispatches = 0
                _synthesized_plan = True

            # Use the canonical extractor so reasoning models (gpt-5.x,
            # glm-5.2, kimi-k2.6) that put the answer in ``reasoning_content``
            # while leaving ``content`` empty still produce visible output.
            # Reading ``completion.content`` raw was the 2026-07-30 zero-output
            # bug: the turn loop saw empty content, forced a final no-tools
            # completion that also read only ``.content``, and stayed empty.
            content = completion_content(completion).strip()

            if not _synthesized_fork and not _synthesized_plan:
                # ── Unified turn lifecycle: apply pending yield ──────────
                # A forced yield (stuck-in-stage cap, iteration cap, or
                # recovery stall) was set earlier in this iteration. The
                # model produced no tool calls (it either complied with the
                # stripped-tools force-final path, or it finished naturally
                # at the same iteration the cap fired). Instead of accepting
                # the model's text as the final answer, break the loop with
                # the yield's question as the turn content so the user can
                # redirect. The checkpoint layer stamps
                # ``lifecycle_state="yielded"``.
                #
                # The hallucinated-transition guard, chain-incomplete guard,
                # and write-tool guard that lived here were removed in the
                # unified-lifecycle cutover. They injected ``user`` messages
                # to correct models that narrated transitions instead of
                # calling ``transition_stage``, or stopped silently with a
                # routing hint active. The unified lifecycle replaces that
                # "beg via nudges" pattern: the harness either forces a
                # ``TransitionAction`` (the forced-fork/plan enforcers
                # below, which are kernel-takes-over, not nudge-and-beg) or
                # forces a ``YieldAction`` (the cap paths). The model's
                # narrated transition is now accepted as a final response —
                # if the workflow is incomplete, the user can reply and the
                # routing hint will re-fire on the next turn.
                if _pending_yield is not None:
                    content = _pending_yield.question
                    _yielded = True
                    _log.info(
                        "turn lifecycle: applying forced yield (mode=%s, iteration=%d, target=%s)",
                        mode_id,
                        _state.iteration,
                        _pending_yield.target_mode_id or "(in-place)",
                    )
                    break

                break

        # Dispatch tool calls — MCP tools via pool, operations in-process.
        tools_dispatched = False
        for tool_call in tool_calls:
            tool_name, tool_arguments = openai_tool_call_to_mcp(tool_call)
            tool_call_id = str(tool_call.get("id") or "")

            # ── Forced-fork enforcer ──────────────────────────────────
            # When the kernel has computed a deterministic ``fork:`` target
            # (``_state.forced_fork_target``) and the model refuses to call
            # ``transition_stage`` — instead burning iterations on read-only
            # tools (``fetch_page``, ``web_search``) or returning empty
            # completions — the kernel forces the fork itself. The routing
            # hint already encodes the routing decision; the model is the
            # executor. When the executor won't execute a deterministic
            # next step, the kernel takes over rather than begging via
            # nudges (the 2026-07-27 incident: ``plan`` mode burned 5
            # ``fetch_page`` calls then returned empty, ending the turn
            # with no fork ever running and the model hallucinating the
            # tool call as prose in the iteration-cap fallback).
            #
            # Detection: a forced fork target is set AND the model called a
            # non-``transition_stage`` tool. Each such dispatch increments
            # ``non_fork_tool_dispatches``; at the cap, the kernel
            # synthesizes the ``transition_stage(fork:...)`` call itself by
            # overriding the current tool call's name/arguments and letting
            # the existing transition interception handle the fan-out.
            # ``transition_stage`` calls (fork or otherwise) reset the
            # counter and never trigger the enforcer.
            if _state.forced_fork_target and tool_name != _TRANSITION_TOOL_NAME:
                _state.non_fork_tool_dispatches += 1
                if _state.non_fork_tool_dispatches >= (
                    _kernel_settings.forced_fork_non_fork_tool_cap
                ):
                    _log.warning(
                        "model refused to fork (mode=%s, iteration=%d, "
                        "non_fork_tool_dispatches=%d); kernel forcing fork "
                        "target=%s",
                        mode_id,
                        _state.iteration,
                        _state.non_fork_tool_dispatches,
                        _state.forced_fork_target,
                    )
                    # Synthesize the fork call. Mutate the current
                    # ``tool_call`` dict IN PLACE — it is the same object
                    # referenced by the assistant message already appended
                    # to ``messages`` (``assistant_msg["tool_calls"] =
                    # tool_calls`` above). Overriding ``function.name`` /
                    # ``function.arguments`` here keeps the assistant
                    # message and the tool result message consistent, so
                    # the provider sees a well-formed tool_call → tool
                    # result pair. Keep the original ``id`` so the tool
                    # result message's ``tool_call_id`` matches the
                    # assistant message's ``tool_calls[].id`` (the
                    # 2026-07-27 follow-on: a fresh ``forced_fork_N`` id
                    # was rejected by the provider with ``Invalid
                    # parameter: 'tool_call_id' not found in 'tool_calls'
                    # of previous message``).
                    tool_call["function"]["name"] = _TRANSITION_TOOL_NAME
                    tool_call["function"]["arguments"] = json.dumps(
                        {
                            "target_stage_id": _state.forced_fork_target,
                            "rationale": (
                                "Kernel-forced fork: the model did not call "
                                "transition_stage after the routing hint "
                                "named this fork target."
                            ),
                        }
                    )
                    tool_name = _TRANSITION_TOOL_NAME
                    tool_arguments = {
                        "target_stage_id": _state.forced_fork_target,
                        "rationale": (
                            "Kernel-forced fork: the model did not call "
                            "transition_stage after the routing hint named "
                            "this fork target."
                        ),
                    }
                    # tool_call_id stays the original — already set above
                    # from ``tool_call.get("id")``.
                    # The enforcer fires once per turn; clear the target so
                    # a subsequent model-issued fork on a later iteration
                    # is not re-enforced and so the post-fork hint
                    # recomputation can set a fresh target if needed.
                    _state.forced_fork_target = ""
                    _state.non_fork_tool_dispatches = 0
                    # Fall through: the transition interception below now
                    # sees ``tool_name == _TRANSITION_TOOL_NAME`` and
                    # dispatches the fork.

            # ── Forced-plan enforcer ────────────────────────────────
            # When the kernel has computed a plan-escalation target
            # (``_state.forced_plan_target == "plan"``) and the model
            # refuses to call ``transition_stage`` — instead burning
            # iterations on read-only tools in the entry mode — the
            # kernel forces the plan handoff itself. The routing hint
            # already named ``plan`` as the target; the model is the
            # executor. When the executor won't execute a deterministic
            # next step, the kernel takes over rather than begging via
            # nudges (the 2026-07-30 incident: the model burned 15
            # iterations of tool-cycling in ``default`` before finally
            # handing off to ``plan``, which then failed to parse —
            # 400k tokens lost to the pre-plan loop). Mirrors the
            # forced-fork enforcer: each non-``transition_stage``
            # dispatch increments ``non_plan_tool_dispatches``; at the
            # cap, the kernel synthesizes the
            # ``transition_stage(plan)`` call itself.
            if _state.forced_plan_target and tool_name != _TRANSITION_TOOL_NAME:
                _state.non_plan_tool_dispatches += 1
                if _state.non_plan_tool_dispatches >= (
                    _kernel_settings.forced_plan_non_transition_tool_cap
                ):
                    _log.warning(
                        "model refused to hand off to plan (mode=%s, "
                        "iteration=%d, non_plan_tool_dispatches=%d); "
                        "kernel forcing plan handoff target=%s",
                        mode_id,
                        _state.iteration,
                        _state.non_plan_tool_dispatches,
                        _state.forced_plan_target,
                    )
                    # Synthesize the plan handoff. Mutate the current
                    # ``tool_call`` dict IN PLACE (same rationale as the
                    # forced-fork enforcer: keeps the assistant message
                    # and tool result message consistent so the provider
                    # sees a well-formed pair).
                    tool_call["function"]["name"] = _TRANSITION_TOOL_NAME
                    tool_call["function"]["arguments"] = json.dumps(
                        {
                            "target_stage_id": _state.forced_plan_target,
                            "rationale": (
                                "Kernel-forced plan handoff: the model did "
                                "not call transition_stage after the "
                                "routing hint named plan as the target."
                            ),
                        }
                    )
                    tool_name = _TRANSITION_TOOL_NAME
                    tool_arguments = {
                        "target_stage_id": _state.forced_plan_target,
                        "rationale": (
                            "Kernel-forced plan handoff: the model did "
                            "not call transition_stage after the routing "
                            "hint named plan as the target."
                        ),
                    }
                    # The enforcer fires once per turn; clear the target
                    # so a subsequent model-issued handoff is not
                    # re-enforced.
                    _state.forced_plan_target = ""
                    _state.non_plan_tool_dispatches = 0
                    # Fall through: the transition interception below now
                    # sees ``tool_name == _TRANSITION_TOOL_NAME`` and
                    # dispatches the plan handoff.

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
            max_calls = 0
            if target_def is not None:
                max_calls = int(getattr(target_def, "max_calls_per_turn", 0) or 0)
            max_calls = resolve_capability_invocation_limit(
                execution_policy,
                tool_name,
                max_calls,
            )
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
            elif tool_name == _YIELD_TOOL_NAME:
                _tool_kind = "yield"
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
                    "_state.iteration": _state.iteration,
                    "tool_calls_so_far": _state.total_tool_calls + 1,
                    "max_tool_calls": max_tool_iterations,
                    "mode_id": mode_id,
                },
            )
            _tool_dispatch_started = time.monotonic()
            recovery_item_id = ""
            recovery_retry_policy = "safe"
            if hooks.recovery_service is not None and _tool_kind != "transition":
                try:
                    effect = "read"
                    if tool_name in operation_catalog:
                        effect = str(
                            getattr(operation_catalog[tool_name], "effect", "read") or "read"
                        )
                    elif tool_name in tool_catalog:
                        effect = str(getattr(tool_catalog[tool_name], "effect", "read") or "read")
                    elif _tool_kind == "mcp_tool":
                        effect = "execute"
                    recovery_retry_policy = "safe" if effect == "read" else "confirm"
                    # ── Durable workflow-instance correlation ──────────
                    # When a ``WorkflowInstance`` is active, stamp the
                    # recovery record with its id + current step id so the
                    # host can decide whether a stalled record blocks the
                    # instance from advancing (``step_blocks_instance``).
                    _wi_id = ""
                    _wi_step_id = ""
                    if hooks.workflow_instance is not None:
                        try:
                            _wi = hooks.workflow_instance()
                        except Exception:  # pylint: disable=broad-except
                            _wi = None
                        if _wi is not None:
                            _wi_id = getattr(_wi, "instance_id", "") or ""
                            _wi_step_id = getattr(_wi, "current_step_id", "") or ""
                    started_record = await hooks.recovery_service.begin(
                        RecoveryRecord(
                            identity=_build_recovery_identity(hooks),
                            conversation_id=hooks.conversation_id,
                            mode_id=mode_id,
                            stage_id=_current_recovery_stage_id(
                                hooks=hooks,
                                is_workflow=is_workflow,
                            ),
                            workflow_instance_id=_wi_id,
                            step_id=_wi_step_id,
                            target_kind=cast(Any, _tool_kind),
                            target_id=tool_name,
                            effect=cast(Any, effect),
                            parameters=(tool_arguments if isinstance(tool_arguments, dict) else {}),
                            retry_policy=cast(Any, recovery_retry_policy),
                        )
                    )
                    recovery_item_id = started_record.work_item_id
                except Exception as recovery_exc:  # pylint: disable=broad-except
                    _log.warning("recovery begin failed for %s: %s", tool_name, recovery_exc)
            try:
                # ── Unified turn lifecycle: yield interception ─────────────
                # The model called ``yield`` — a first-class terminal action.
                # This is handled in-loop, not through the operation/tool/MCP
                # dispatch branches (same pattern as ``transition_stage``).
                # The kernel extracts the question/context/target, sets
                # ``_pending_yield`` and ``_yielded``, appends a tool result
                # so the message history stays well-formed, and breaks the
                # loop. The checkpoint layer stamps
                # ``lifecycle_state="yielded"`` with the question.
                if tool_name == _YIELD_TOOL_NAME:
                    _yield_question = str(tool_arguments.get("question") or "").strip()
                    _yield_context = str(tool_arguments.get("context") or "").strip()
                    _yield_target = str(tool_arguments.get("target_mode_id") or "").strip()
                    if not _yield_question:
                        # The model called yield with no question — degenerate.
                        # Use the policy's default question rather than hanging.
                        _yp = _resolve_yield_policy(mode_config)
                        _yield_question = _yp.default_question or (
                            "I need more information to proceed. "
                            "What would you like me to focus on?"
                        )
                    _pending_yield = YieldAction(
                        question=_yield_question,
                        context=_yield_context or "model-emitted yield",
                        target_mode_id=_yield_target,
                    )
                    content = _yield_question
                    _yielded = True
                    _log.info(
                        "turn lifecycle: model emitted yield (mode=%s, iteration=%d, target=%s)",
                        mode_id,
                        _state.iteration,
                        _yield_target or "(in-place)",
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "name": tool_name,
                            "content": f"Yielded to user: {_yield_question}",
                        }
                    )
                    await _emit_stage(
                        "tool_dispatch_complete",
                        {
                            "tool": tool_name,
                            "tool_kind": "yield",
                            "_state.iteration": _state.iteration,
                            "mode_id": mode_id,
                        },
                    )
                    break

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
                                "_state.iteration": _state.iteration,
                                "tool_calls_so_far": _state.total_tool_calls,
                                "ok": False,
                                "transition": "invalid",
                                "duration_ms": int(
                                    (time.monotonic() - _tool_dispatch_started) * 1000
                                ),
                            },
                        )
                        continue
                    # ── Guard: don't route a research/review intent into an
                    # action mode after a search entry op failed this turn.
                    # Once ``_state.entry_failed`` is set, the research
                    # backend is known dead. Letting the model hand off to
                    # ``coding`` lets it author a tangible-but-wrong artifact
                    # (the 2026-07-25 ``crew/`` directory) as the "answer" to
                    # a research question. Reject the handoff and force the
                    # model to surface the failure instead.
                    if (
                        _state.entry_failed
                        and target_stage == "coding"
                        and not target_stage.startswith("fork:")
                    ):
                        tool_output = (
                            "Cannot route to `coding` mode: the search/research "
                            f"backend failed this turn ({_state.entry_failed_mode}: "
                            f"{_state.entry_failed_reason}). A research/review "
                            "request must not be answered by authoring artifacts. "
                            "Report the search failure to the user and suggest "
                            "re-issuing once the provider is reconfigured."
                        )
                        _log.warning(
                            "rejected coding handoff after entry failure (failed_mode=%s)",
                            _state.entry_failed_mode,
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

                    # ── Case A: fork fan-out ──────────────────────────────
                    if is_fork_target(target_stage):
                        # The model honored the routing hint and called
                        # ``transition_stage`` itself. Clear the forced-fork
                        # and forced-plan enforcer state so it does not
                        # re-fire on a later iteration (e.g. the post-fork
                        # synthesis turn that returns text without tool
                        # calls).
                        _state.forced_fork_target = ""
                        _state.non_fork_tool_dispatches = 0
                        _state.forced_plan_target = ""
                        _state.non_plan_tool_dispatches = 0
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
                                    execution_policy=execution_policy,
                                    turn_start_mono=turn_start_mono,
                                    logger_override=_log,
                                    emit_stage=_emit_stage,
                                )
                                fork_results = fork_results_list
                                succeeded = sum(1 for r in fork_results_list if r.status == "ok")
                                # Mirror to ``_state`` so the post-loop
                                # terminal-failure check can detect a
                                # ``no_tools`` child and break the ``while``
                                # loop (the ``for``-loop ``break`` inside the
                                # fork block only exits the inner tool loop).
                                _state.fork_results = fork_results_list
                                # Surface failed children (timeout, error,
                                # no_tools, cancelled) explicitly so the
                                # parent model does not claim false success or
                                # hallucinate a deferred response it cannot
                                # deliver. The framework is strictly
                                # turn-by-turn — there is no background
                                # scheduler, so a failed child's work is lost
                                # unless the user re-issues the request.
                                failed_children = [r for r in fork_results_list if r.status != "ok"]
                                # A ``no_tools`` child is a hallucinated
                                # success: its mode declares action tools
                                # but it made zero tool calls, so nothing
                                # landed on disk/memory. The 2026-07-18 fix
                                # labelled it; the remaining gap (2026-07-25
                                # incident) was that the parent still handed
                                # the label to the model as advisory prose,
                                # which the model could paraphrase into a
                                # false "completed". When any child is
                                # ``no_tools``, abort the user-facing turn
                                # with a deterministic failure instead of
                                # ``continue``-ing. The model cannot be
                                # trusted to surface it honestly.
                                has_no_tools = any(r.status == "no_tools" for r in failed_children)
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
                                    child_parts: list[str] = []
                                    for r in fork_results_list:
                                        child_parts.append(
                                            f"### {r.target.mode_id}\n{r.output_text}"
                                        )
                                    tool_output = (
                                        f"Fork fan-out completed: "
                                        f"{succeeded}/{len(fork_targets)} children "
                                        f"succeeded.\n\n"
                                        f"{chr(10).join(child_parts)}"
                                    )
                        await _emit_stage(
                            "tool_dispatch_complete",
                            {
                                "tool": tool_name,
                                "tool_kind": "transition",
                                "_state.iteration": _state.iteration,
                                "tool_calls_so_far": _state.total_tool_calls,
                                "ok": True,
                                "transition": "fork",
                                "fork_children": len(fork_results_list) if fork_results_list else 0,
                                "duration_ms": int(
                                    (time.monotonic() - _tool_dispatch_started) * 1000
                                ),
                            },
                        )
                        # ── Mark fork children's intents as satisfied ──
                        # Without this, the post-fork routing-hint recomputation
                        # runs against the entry mode's ``satisfies_intents``
                        # (typically empty for ``default``), re-detects the
                        # original query's intents as a gap, and re-emits the
                        # exact ``fork:...`` hint. The model obligingly calls
                        # ``transition_stage`` again → another fan-out → more
                        # duplicate artifacts. Mirrors the handoff branch's
                        # ``_state.chain_satisfied.update(...)`` so the next
                        # hint is computed against the union of every mode the
                        # turn has already run (entry + fork children). Only
                        # ``ok`` children contribute their intents — a
                        # failed/``no_tools`` child satisfied nothing. Tag-
                        # blind: the kernel only unions opaque pack-authored
                        # intent-tag sets.
                        if fork_results_list:
                            for r in fork_results_list:
                                if r.status != "ok":
                                    continue
                                child_mode = catalog.get(r.target.mode_id)
                                if child_mode is None:
                                    continue
                                _state.chain_satisfied.update(child_mode.satisfies_intents)
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "name": tool_name,
                                "content": tool_output,
                            }
                        )
                        # If any fork child is ``no_tools`` (hallucinated
                        # success), abort the user-facing turn with a
                        # deterministic failure. Do NOT ``continue`` and hand
                        # the failure to the model as advice — the model can
                        # paraphrase it into a false "completed" (the
                        # 2026-07-25 incident). Surface the failure directly.
                        if has_no_tools:
                            no_tools_modes = [
                                r.target.mode_id
                                for r in fork_results_list
                                if r.status == "no_tools"
                            ]
                            content = (
                                "Fork fan-out failed: the following children made "
                                "zero tool calls despite their mode declaring action "
                                f"tools ({', '.join(no_tools_modes)}), so their reported "
                                "success is unverified — nothing was written to disk or "
                                "memory. Re-issue the request."
                            )
                            break
                        continue

                    # ── Case B: mode handoff (routing FSM) ────────────────
                    if is_routable_mode(target_stage):
                        # The model honored the routing hint (chose a single
                        # handoff instead of the suggested fork). Clear the
                        # forced-fork and forced-plan enforcer state; the
                        # post-handoff hint recomputation below will set a
                        # fresh target if the new mode still leaves a gap.
                        _state.forced_fork_target = ""
                        _state.non_fork_tool_dispatches = 0
                        _state.forced_plan_target = ""
                        _state.non_plan_tool_dispatches = 0
                        # ── Hint-override tracking ──────────────────────
                        # When the model hands off to a target that differs
                        # from the routing hint's suggested target, it is
                        # overriding the hint. After ``hint_override_limit``
                        # overrides the model has demonstrated its own
                        # routing judgment; continuing to inject hints is
                        # noise the model rationally routes around (the
                        # 2026-07-30 bounce: hint said ``discuss``, model
                        # went to ``ponder`` twice). Track overrides so the
                        # post-handoff hint recomputation can suppress
                        # further hints once the limit is reached.
                        _hint_suggested = _extract_hint_target(routing_hint)
                        if _hint_suggested and target_stage and target_stage != _hint_suggested:
                            _state.hint_override_count += 1
                            _log.info(
                                "routing hint override: hint suggested %s, "
                                "model chose %s (override_count=%d, cap=%d)",
                                _hint_suggested,
                                target_stage,
                                _state.hint_override_count,
                                _hint_override_limit,
                            )
                        # Hard backstop against routing-hint loops. The
                        # ``chain_satisfied`` accumulator is the primary
                        # breaker, but it depends on accurate pack-authored
                        # ``routing_intent_tags``. When attribution fails,
                        # the hint re-fires on every handoff with no end.
                        # Cap total handoffs and force a final completion.
                        _state.handoff_count += 1
                        if _state.handoff_count > _max_handoffs_per_turn:
                            _log.warning(
                                "routing hint loop guard: %d handoffs in turn "
                                "(cap=%d); forcing final completion",
                                _state.handoff_count,
                                _max_handoffs_per_turn,
                            )
                            tool_output = (
                                f"Routing loop guard: reached "
                                f"{_max_handoffs_per_turn} mode handoffs in "
                                f"this turn without completing the chain. "
                                f"Forcing a final response."
                            )
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call_id,
                                    "name": tool_name,
                                    "content": tool_output,
                                }
                            )
                            content = tool_output
                            break
                        # Initialize the post-handoff hint to empty for every
                        # handoff path. The chained-hint recomputation below
                        # (inside the target-mode-found ``else`` branch)
                        # overwrites this when the new mode leaves a residual
                        # intent gap. Branches that reject the handoff (not
                        # routable, target mode missing) leave it empty so the
                        # post-tool-output nudge at the ``if post_handoff_hint``
                        # check below is skipped. Without this initialization
                        # those reject branches reach that check with an
                        # unbound local → ``UnboundLocalError`` (logged as
                        # "Tool 'transition_stage' failed in turn runner").
                        post_handoff_hint = ""
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
                                # ── Structured handoff context ───────────────
                                # Give the receiving mode awareness of its
                                # position in the chain so it knows it is not
                                # the final step and should hand off when done.
                                # ``from_mode`` is the source mode_id (still
                                # the pre-handoff value here); ``chain_position``
                                # is the 1-based position of this handoff target
                                # in the turn's chain; ``remaining_steps`` are
                                # the routing targets whose intent tags still
                                # cover a gap intent after this handoff (i.e.
                                # the steps still to run). Tag-blind: the kernel
                                # only manipulates opaque intent-tag sets and
                                # routing target ids; it names no intents.
                                try:
                                    post_handoff_satisfied = set(_state.chain_satisfied) | set(
                                        target_mode_config.satisfies_intents
                                    )
                                    remaining_gap = (
                                        _detect_query_intents(
                                            query=query,
                                            routing_targets=get_routing_targets(target_stage),
                                        )
                                        - post_handoff_satisfied
                                    )
                                    remaining_steps: list[str] = []
                                    if remaining_gap:
                                        for rid in get_routing_targets(target_stage):
                                            rmode = get_workflow_catalog().get(rid)
                                            if rmode is None or rid == target_stage:
                                                continue
                                            if (
                                                set(rmode.routing_intent_tags) & remaining_gap
                                                and rid not in remaining_steps
                                            ):
                                                remaining_steps.append(rid)
                                    chain_position = len(_state.chain_satisfied) + 1
                                    # Terminal-step directive: when there are no
                                    # remaining steps, tell the model to COMPLETE
                                    # and respond, not to hand off. The previous
                                    # unconditional "hand off to the next step"
                                    # line contradicted the "(none — this is the
                                    # final step)" sentinel and left the terminal
                                    # mode (e.g. markdown) flailing with no
                                    # completion signal (2026-07-30 incident).
                                    if remaining_steps:
                                        next_directive = (
                                            "- After you complete your work, hand off to "
                                            "the next step by calling "
                                            "`transition_stage`.\n"
                                        )
                                    else:
                                        next_directive = (
                                            "- You are the final step in this chain. "
                                            "Complete your work and respond to the user "
                                            "directly — do NOT call `transition_stage`.\n"
                                        )
                                    handoff_context = (
                                        "\n\n## Handoff Context\n"
                                        f"- You were handed off from: `{mode_id}`\n"
                                        f"- Chain position: {chain_position} of "
                                        f"{chain_position + len(remaining_steps)}\n"
                                        f"- Remaining steps after this one: "
                                        f"{_format_remaining_steps(remaining_steps)}\n"
                                        f"{next_directive}"
                                    )
                                    messages[0]["content"] = (
                                        str(messages[0].get("content", "")) + handoff_context
                                    )
                                except Exception as ctx_exc:  # pylint: disable=broad-except
                                    _log.warning(
                                        "failed to build handoff context for mode=%s: %s",
                                        target_stage,
                                        ctx_exc,
                                    )
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
                                # Action modes (e.g. search) run entry
                                # operations (e.g. web_search) that produce
                                # the mode's output. Without running them the
                                # model lands in search mode with no web_search
                                # tool (it's an operation, not a tool) and no
                                # results — it correctly says "I don't have
                                # web_search." Run the entry operations and
                                # inject their output as a tool result so the
                                # model has the data to work with.
                                entry_output = ""
                                if (
                                    target_mode_config.mode_kind == "action"
                                    and target_mode_config.entry_operation_ids
                                ):
                                    try:
                                        entry_ctx = ctx.model_copy(update={"mode_id": target_stage})
                                        entry_result = await run_mode_entry_operations(
                                            target_stage,
                                            entry_ctx,
                                            ledger=ledger,
                                            execution_policy=execution_policy,
                                        )
                                        entry_output = entry_result.output_text.strip() or "Done"
                                        # ── Kernel-driven next handoff ──
                                        # When the handoff target is a
                                        # planner action mode (e.g. ``plan``)
                                        # whose entry op returns a
                                        # ``target_stage_id`` in
                                        # ``output_json``, the kernel drives
                                        # the next handoff itself instead
                                        # of injecting the plan result as
                                        # text and waiting for the model to
                                        # call ``transition_stage``. The
                                        # model in ``plan`` mode reliably
                                        # stops after planning (the
                                        # 2026-07-28 stall), so the kernel
                                        # synthesizes the transition at the
                                        # top of the next iteration. The
                                        # coordinator decision is saved
                                        # from the plan result so a later
                                        # ``proceed`` can resume the chain.
                                        #
                                        # IMPORTANT: only ``plan`` mode's
                                        # entry op (``plan_workflow``)
                                        # returns a ``target_stage_id``.
                                        # Other action modes (e.g.
                                        # ``search`` with ``web_search``)
                                        # return their own output shape
                                        # (search results), which has no
                                        # routing target. Applying the
                                        # kernel-driven handoff to every
                                        # action-mode entry op misreads
                                        # ``web_search`` output as a failed
                                        # plan and hard-stops the turn — the
                                        # 2026-07-30 ``search`` incident
                                        # (``plan_workflow in mid-turn
                                        # handoff returned no target
                                        # (mode=search)``). Gate the planner
                                        # logic on the target actually being
                                        # ``plan`` (or, more generally, on
                                        # the entry op being a planner op
                                        # that returns routing decisions).
                                        _is_planner_handoff = (
                                            target_stage == "plan"
                                            or "plan_workflow"
                                            in (target_mode_config.entry_operation_ids or [])
                                        )
                                        _planned_next = ""
                                        if _is_planner_handoff:
                                            _planned_next = str(
                                                (entry_result.output_json or {}).get(
                                                    "target_stage_id", ""
                                                )
                                                or ""
                                            ).strip()
                                        if _planned_next:
                                            _state.pending_planned_handoff = _planned_next
                                            if hooks.route_coordinator is not None:
                                                try:
                                                    hooks.route_coordinator.save(
                                                        CoordinatorRouteDecision(
                                                            next_target=_planned_next,
                                                            reason=(
                                                                entry_result.output_json.get(
                                                                    "rationale"
                                                                )
                                                                or f"planned handoff from {target_stage}"
                                                            ),
                                                        ),
                                                        {
                                                            "mode_id": _planned_next,
                                                            "query": query,
                                                            "surface_id": hooks.surface_id,
                                                            "conversation_id": hooks.conversation_id,
                                                            "user_id": hooks.user_id,
                                                            "session_plan_digest": hooks.session_plan_digest,
                                                        },
                                                    )
                                                except Exception as coord_exc:  # pylint: disable=broad-except
                                                    _log.warning(
                                                        "route_coordinator.save "
                                                        "failed for planned "
                                                        "next target %s: %s",
                                                        _planned_next,
                                                        coord_exc,
                                                    )
                                            _log.info(
                                                "kernel-driven next handoff "
                                                "from planner mode=%s target=%s "
                                                "(planned by entry op)",
                                                target_stage,
                                                _planned_next,
                                            )
                                        elif _is_planner_handoff:
                                            # ── Plan failure mid-turn ──
                                            # The planner returned no
                                            # ``target_stage_id`` (parse
                                            # failure, model error). Do NOT
                                            # continue in ``plan`` mode — the
                                            # post-handoff hint generator
                                            # would re-hallucinate a target
                                            # and the model would improvise
                                            # around it (the 2026-07-30
                                            # bounce). Mark the model
                                            # unreliable for planning and
                                            # break the loop with a
                                            # deterministic recovery prompt
                                            # so the user re-issues or
                                            # names a target directly.
                                            #
                                            # Only fire for planner handoffs.
                                            # A non-planner action mode (e.g.
                                            # ``search``/``web_search``) has
                                            # no ``target_stage_id`` by
                                            # design — that is NOT a plan
                                            # failure, and must not hard-stop
                                            # the turn (2026-07-30 incident).
                                            _plan_fail_reason = str(
                                                (entry_result.output_json or {}).get(
                                                    "rationale", ""
                                                )
                                                or "the planner returned no routing target"
                                            ).strip()
                                            _state.plan_failed = True
                                            _state.plan_failed_mode = target_stage
                                            _state.plan_failed_reason = _plan_fail_reason
                                            from kernel.routing_vocabulary import (  # noqa: PLC0415
                                                mark_plan_unreliable_for_model,
                                            )

                                            mark_plan_unreliable_for_model(
                                                str(preferred_model_id or "")
                                            )
                                            _log.warning(
                                                "plan_workflow in mid-turn "
                                                "handoff returned no target "
                                                "(mode=%s, reason=%s); marking "
                                                "model unreliable and breaking",
                                                target_stage,
                                                _plan_fail_reason,
                                            )
                                    except Exception as entry_exc:  # pylint: disable=broad-except
                                        entry_output = (
                                            f"Entry operation failed: "
                                            f"{type(entry_exc).__name__}: {entry_exc}"
                                        )
                                        _log.warning(
                                            "entry operation for handoff mode=%s failed: %s",
                                            target_stage,
                                            entry_exc,
                                        )
                                        # Mark the turn as hard-failed. An
                                        # action mode (e.g. ``search``) whose
                                        # entry operation produced no data
                                        # has nothing for the model to work
                                        # on. Continuing in that mode lets the
                                        # model drift into an unrelated action
                                        # mode (e.g. ``coding``) and author a
                                        # tangible-but-wrong artifact as the
                                        # "answer" — the 2026-07-25 ``crew/``
                                        # incident. Break the loop with a
                                        # deterministic user-facing failure
                                        # so the user reconfigures the failed
                                        # provider and re-issues, instead of
                                        # the model hallucinating a solution.
                                        _state.entry_failed = True
                                        _state.entry_failed_mode = target_stage
                                        _state.entry_failed_reason = (
                                            f"{type(entry_exc).__name__}: {entry_exc}"
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
                                if hooks.route_coordinator is not None:
                                    try:
                                        hooks.route_coordinator.save(
                                            CoordinatorRouteDecision(
                                                next_target=target_stage,
                                                reason=rationale or "handoff",
                                            ),
                                            {
                                                "mode_id": target_stage,
                                                "query": query,
                                                "surface_id": hooks.surface_id,
                                                "conversation_id": hooks.conversation_id,
                                                "user_id": hooks.user_id,
                                                "session_plan_digest": hooks.session_plan_digest,
                                            },
                                        )
                                    except Exception as coord_exc:  # pylint: disable=broad-except
                                        _log.warning(
                                            "route_coordinator.save failed for target %s: %s",
                                            target_stage,
                                            coord_exc,
                                        )
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
                                # ── Chained-handoff hint recomputation ──
                                # The proactive routing hint was computed in
                                # Phase 1 against the *entry* mode. After a
                                # mid-turn handoff the new mode may still leave
                                # some of the original query's intents unsatisfied
                                # (e.g. default → search resolves "research" but
                                # "write" remains a gap → the next hint should
                                # name markdown). Without recomputing, the model
                                # lands in the first handoff mode, satisfies that
                                # one intent, and stops — it gets no signal that
                                # the ad-hoc workflow is incomplete. Recompute the
                                # hint against the new mode's ``satisfies_intents``
                                # so the model can chain one handoff per remaining
                                # intent. The kernel remains a pure mapper: it
                                # calls the same opaque-tag set algebra over
                                # pack-authored vocabulary; it names no intents,
                                # tool IDs, or routing prose here.
                                #
                                # Suppressed for fork children: a handoff inside
                                # a fork child is the same re-routing loop the
                                # Phase-1 suppression prevents. Fork children run
                                # their target mode to completion without chaining
                                # further handoffs.
                                post_handoff_hint = ""
                                if not suppress_routing_hint:
                                    # Accumulate the new mode's satisfied
                                    # intents into the chain memory so the
                                    # recomputed hint does not re-suggest
                                    # steps that already ran. The entry
                                    # mode's intents were added at init; each
                                    # handoff adds its target's intents here.
                                    _state.chain_satisfied.update(
                                        target_mode_config.satisfies_intents
                                    )
                                    # Reset the per-step iteration sub-budget
                                    # counters so the new step gets its own
                                    # full budget. Placed before the durable
                                    # instance advancement so a failure there
                                    # does not leave a stale counter. (Phase 3)
                                    _state.step_iteration = 0
                                    _state.step_tool_calls = 0
                                    post_handoff_hint = _compute_capability_routing_hint(
                                        mode_id=target_stage,
                                        query=query,
                                        capability_manifest=capability_manifest,
                                        routing_targets=get_routing_targets(target_stage),
                                        mode_config=target_mode_config,
                                        extra_satisfied=_state.chain_satisfied,
                                        planner_model_id=str(preferred_model_id or ""),
                                    )
                                    # ── Durable instance advancement ──
                                    # Persist the handoff on the active
                                    # ``WorkflowInstance`` so the next turn's
                                    # ``chain_satisfied`` hydration rehydrates
                                    # the completed step. The step that just
                                    # finished is the instance's
                                    # ``current_step_id`` (the mode we handed
                                    # *from*); advancing it moves the pointer
                                    # to the new target. Idempotent: a
                                    # repeated handoff to the same target is
                                    # a no-op. Runs after the hint recomputation
                                    # so a failure here never suppresses it.
                                    if (
                                        hooks.workflow_instance is not None
                                        and hooks.workflow_instance_updater is not None
                                    ):
                                        try:
                                            _wi = hooks.workflow_instance()
                                        except Exception:  # pylint: disable=broad-except
                                            _wi = None
                                        if _wi is not None and getattr(_wi, "current_step_id", ""):
                                            try:
                                                from kernel.workflow_instance import (
                                                    advance_step,
                                                )

                                                _advanced = advance_step(_wi, _wi.current_step_id)
                                                hooks.workflow_instance_updater(_advanced)
                                                # ── Plan-driven next-step hint ──
                                                # When a plan_workflow_op
                                                # instance is active and has
                                                # a remaining step after
                                                # advancement, override the
                                                # emergent routing hint with
                                                # a directive hint naming the
                                                # next step's mode from the
                                                # precomputed plan. This
                                                # replaces the one-step-at-a-
                                                # time emergent hint chain
                                                # with the declared plan
                                                # sequence. The plan is the
                                                # source of truth, not the
                                                # gap-intent recompute.
                                                if (
                                                    _advanced is not None
                                                    and _advanced.spawned_from == "plan_workflow_op"
                                                    and _advanced.current_step_id
                                                ):
                                                    _next_step = (
                                                        _advanced.step_by_id(
                                                            _advanced.current_step_id
                                                        )
                                                        if hasattr(_advanced, "step_by_id")
                                                        else None
                                                    )
                                                    if _next_step is not None:
                                                        post_handoff_hint = (
                                                            "## Routing Hint\n\n"
                                                            f"This is the next step in the "
                                                            f"planned workflow. Call "
                                                            f"`transition_stage` with "
                                                            f"target_stage_id "
                                                            f"`{_next_step.mode_id}`."
                                                        )
                                                        _log.info(
                                                            "plan-driven next step: "
                                                            "instance=%s step=%s "
                                                            "target=%s",
                                                            _advanced.instance_id,
                                                            _next_step.step_id,
                                                            _next_step.mode_id,
                                                        )
                                            except Exception as wi_exc:  # pylint: disable=broad-except
                                                _log.warning(
                                                    "workflow_instance_updater failed after "
                                                    "handoff to '%s': %s",
                                                    target_stage,
                                                    wi_exc,
                                                )
                                if post_handoff_hint:
                                    # ── Hint-override suppression ──────
                                    # After the model overrides the hint
                                    # ``hint_override_limit`` times, stop
                                    # injecting further hints — the model
                                    # has its own routing judgment and the
                                    # hint is noise (the 2026-07-30 bounce:
                                    # model overrode ``discuss``→``ponder``
                                    # twice). The hint stays suppressed for
                                    # the rest of the turn; the handoff
                                    # cap backstop fires if the model loops.
                                    if _state.hint_override_count >= _hint_override_limit:
                                        _log.warning(
                                            "suppressing post-handoff hint: "
                                            "model overrode %d hints this "
                                            "turn (mode=%s)",
                                            _state.hint_override_count,
                                            target_stage,
                                        )
                                        post_handoff_hint = ""
                                        routing_hint = ""
                                    elif post_handoff_hint == _state.last_hint_text:
                                        # Duplicate-hint dedup: when the
                                        # recomputed hint is byte-identical
                                        # to the previous one, the accumulator
                                        # failed to advance and the model is
                                        # in a routing loop. Suppress
                                        # injection after
                                        # ``duplicate_hint_limit`` repeats
                                        # so the max-hops guard can take over
                                        # instead of re-injecting the same
                                        # prose forever.
                                        _state.duplicate_hint_count += 1
                                    else:
                                        _state.duplicate_hint_count = 0
                                    _state.last_hint_text = post_handoff_hint
                                    # ── Hint-cycle guard ────────────────
                                    # A 2-cycle (A→B then B→A) produces two
                                    # *different* hint strings, so the
                                    # byte-identical dedup above never fires.
                                    # The chain_satisfied accumulator is the
                                    # primary breaker, but it depends on
                                    # accurate pack-authored
                                    # ``satisfies_intents``. When a stale
                                    # catalog drops those declarations, the
                                    # gap never shrinks and the hint
                                    # oscillates A↔B until the hard
                                    # ``handoff_count`` cap (wasting model
                                    # round-trips). Extract the target from
                                    # the hint and detect a repeat of a
                                    # target seen 2 handoffs ago — that is
                                    # the A→B→A signature. Suppress the hint
                                    # so the ``handoff_count`` backstop
                                    # completes the turn immediately.
                                    _hint_target = _extract_hint_target(post_handoff_hint)
                                    _cycle_detected = False
                                    if _hint_target:
                                        _state.hint_target_history.append(_hint_target)
                                        # A 2-cycle: the latest target
                                        # matches the one before the
                                        # previous (A, B, A).
                                        if (
                                            len(_state.hint_target_history)
                                            >= _kernel_settings.hint_cycle_min_len
                                            and _state.hint_target_history[-1]
                                            == _state.hint_target_history[-3]
                                        ):
                                            _cycle_detected = True
                                    if _cycle_detected:
                                        _log.warning(
                                            "suppressing routing hint 2-cycle "
                                            "(mode=%s, history=%s)",
                                            target_stage,
                                            _state.hint_target_history[-3:],
                                        )
                                        post_handoff_hint = ""
                                        routing_hint = ""
                                    elif _state.duplicate_hint_count >= (
                                        _kernel_settings.duplicate_hint_limit
                                    ):
                                        _log.warning(
                                            "suppressing duplicate routing hint "
                                            "(mode=%s, repeats=%d)",
                                            target_stage,
                                            _state.duplicate_hint_count,
                                        )
                                        post_handoff_hint = ""
                                        routing_hint = ""
                                    else:
                                        _log.info(
                                            "post-handoff routing hint mode=%s hint_len=%d hint=%s",
                                            target_stage,
                                            len(post_handoff_hint),
                                            post_handoff_hint[:200],
                                        )
                                        # Append to the system prompt so the guards
                                        # (hallucinated-transition, write-tool) key
                                        # off the new hint. Strip any per-iteration
                                        # budget briefing first (by marker) so the
                                        # hint lands on the base prompt; the next
                                        # iteration's briefing injection will
                                        # append after the hint.
                                        _pre_hint_sys = str(messages[0].get("content", ""))
                                        if _budget_briefing_marker in _pre_hint_sys:
                                            _pre_hint_sys = _pre_hint_sys.split(
                                                _budget_briefing_marker, 1
                                            )[0]
                                        messages[0]["content"] = (
                                            _pre_hint_sys + "\n\n" + post_handoff_hint
                                        )
                                        routing_hint = post_handoff_hint
                                        # Refresh the forced-fork target for the
                                        # new mode so the enforcer tracks the
                                        # post-handoff routing decision.
                                        _state.forced_fork_target = (
                                            _compute_fork_target_string(
                                                query=query,
                                                routing_targets=get_routing_targets(target_stage),
                                                mode_config=target_mode_config,
                                                extra_satisfied=_state.chain_satisfied,
                                            )
                                            or ""
                                        )
                                        _state.non_fork_tool_dispatches = 0
                                        # Refresh the forced-plan target for
                                        # the new mode so the enforcer
                                        # tracks the post-handoff routing
                                        # decision.
                                        _state.forced_plan_target = (
                                            "plan"
                                            if _extract_hint_target(post_handoff_hint) == "plan"
                                            else ""
                                        )
                                        _state.non_plan_tool_dispatches = 0
                                else:
                                    # The new mode satisfies all remaining
                                    # intents; clear the stale hint so the
                                    # guards don't fire on the old one.
                                    routing_hint = ""
                                    _state.duplicate_hint_count = 0
                                    _state.last_hint_text = ""
                                    # No remaining gap → no forced fork or
                                    # forced plan.
                                    _state.forced_fork_target = ""
                                    _state.non_fork_tool_dispatches = 0
                                    _state.forced_plan_target = ""
                                    _state.non_plan_tool_dispatches = 0
                                tool_output = f"Handed off to mode: {target_stage}." + (
                                    f" Rationale: {rationale}" if rationale else ""
                                )
                                if entry_output:
                                    tool_output += f"\n\nEntry operation results:\n{entry_output}"
                        await _emit_stage(
                            "tool_dispatch_complete",
                            {
                                "tool": tool_name,
                                "tool_kind": "transition",
                                "_state.iteration": _state.iteration,
                                "tool_calls_so_far": _state.total_tool_calls,
                                "max_tool_calls": max_tool_iterations,
                                "ok": True,
                                "transition": "handoff",
                                "handoff_to": target_stage,
                                "max_handoffs": _max_handoffs_per_turn,
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
                        # When a post-handoff hint exists (the chain is not
                        # yet complete), inject it as a user-message nudge
                        # AFTER the tool output so the model treats it as a
                        # direct instruction to act on now — not background
                        # system-prompt context it can ignore in favor of the
                        # mode's own "summarize the results" directive. A
                        # user message is the strongest signal short of
                        # calling the tool ourselves; without it the model
                        # in an action mode (e.g. search) does its one tool
                        # call, summarizes, and ends the turn — the chain
                        # stalls even though the hint is in the system prompt.
                        if post_handoff_hint:
                            messages.append(
                                {
                                    "role": "user",
                                    "content": (
                                        "CRITICAL: You are in a multi-step "
                                        "workflow. You are now in the `"
                                        + target_stage
                                        + "` mode. Complete this mode's work, "
                                        "then you MUST call "
                                        "`transition_stage` to continue to the "
                                        "next step. Do NOT end the turn "
                                        "without calling `transition_stage`. " + post_handoff_hint
                                    ),
                                }
                            )
                        # If the entry operation for this handoff target
                        # failed, do NOT ``continue`` into the data-less
                        # mode. The model would have nothing to work on and
                        # would drift into an unrelated action mode. Break
                        # the loop with a deterministic failure so the user
                        # sees the provider failure and re-issues.
                        if _state.entry_failed:
                            content = (
                                f"The `{_state.entry_failed_mode}` mode's entry "
                                f"operation failed ({_state.entry_failed_reason}). "
                                f"Reconfigure the failed provider and re-issue the "
                                f"request."
                            )
                            break
                        if _state.plan_failed:
                            # The planner (e.g. ``plan``) returned no
                            # routing target. Surface a structured recovery
                            # prompt instead of continuing in the data-less
                            # planner mode (the 2026-07-30 bounce).
                            content = (
                                f"I attempted to plan this request in "
                                f"`{_state.plan_failed_mode}` mode but could "
                                f"not produce a routing decision (reason: "
                                f"{_state.plan_failed_reason}). This is often "
                                f"a sign of a complex or ambiguous multi-step "
                                f"request.\n\n"
                                f"You can:\n"
                                f"  1. Rephrase the request with explicit "
                                f"sequential steps "
                                f'(e.g. "search for X, then write a markdown '
                                f'report").\n'
                                f"  2. Name the target mode directly (e.g. "
                                f"`/markdown ...`, `/ponder ...`) to bypass "
                                f"planning.\n"
                                f"  3. Ask me to discuss it first "
                                f"(`/discuss ...`) so I can clarify what you "
                                f"want before committing to a workflow."
                            )
                            break
                        continue
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
                            "_state.iteration": _state.iteration,
                            "tool_calls_so_far": _state.total_tool_calls,
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
                _state.total_tool_calls += 1
                _state.step_tool_calls += 1
                _state.called_tool_names.add(tool_name)
                ledger.record(tool_name)
                # Sync the ledger counts to _state so the per-iteration
                # budget briefing can report per-target remaining calls.
                _state.call_ledger_counts = dict(ledger.counts)
                if recovery_item_id and hooks.recovery_service is not None:
                    try:
                        await hooks.recovery_service.settle(
                            recovery_item_id,
                            _build_recovery_identity(hooks),
                            state="succeeded",
                            result_preview=str(tool_output)[:240],
                        )
                    except Exception as recovery_exc:  # pylint: disable=broad-except
                        _log.warning(
                            "recovery settle success failed for %s: %s",
                            tool_name,
                            recovery_exc,
                        )
            except ApprovalRequired:
                if recovery_item_id and hooks.recovery_service is not None:
                    try:
                        await hooks.recovery_service.settle(
                            recovery_item_id,
                            _build_recovery_identity(hooks),
                            state="stalled",
                            last_error="approval required",
                            failure_scope="workflow",
                            retry_policy="confirm",
                            blocks_workflow=True,
                        )
                    except Exception as recovery_exc:  # pylint: disable=broad-except
                        _log.warning(
                            "recovery settle approval failed for %s: %s",
                            tool_name,
                            recovery_exc,
                        )
                raise
            except Exception as exc:  # pylint: disable=broad-except
                _log.warning("Tool '%s' failed in turn runner: %s", tool_name, exc)
                tool_output = f"Tool execution failed: {type(exc).__name__}: {exc}"
                if recovery_item_id and hooks.recovery_service is not None:
                    try:
                        await hooks.recovery_service.settle(
                            recovery_item_id,
                            _build_recovery_identity(hooks),
                            state="stalled",
                            last_error=str(exc),
                            failure_scope="item",
                            retry_policy=cast(Any, recovery_retry_policy),
                            blocks_workflow=False,
                        )
                    except Exception as recovery_exc:  # pylint: disable=broad-except
                        _log.warning(
                            "recovery settle failure failed for %s: %s",
                            tool_name,
                            recovery_exc,
                        )

            await _emit_stage(
                "tool_dispatch_complete",
                {
                    "tool": tool_name,
                    "tool_kind": _tool_kind,
                    "_state.iteration": _state.iteration,
                    "tool_calls_so_far": _state.total_tool_calls,
                    "ok": not tool_output.startswith("Tool execution failed:"),
                    "duration_ms": int((time.monotonic() - _tool_dispatch_started) * 1000),
                },
            )

            if not tool_output.strip():
                tool_output = "Done"

            # Track the latest tool result for the model router.
            _state.last_tool_result_text = str(tool_output)

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

        # ── Hard-failure / loop-guard while-break ──────────────────────
        # The handoff, fork, and max-hops guards above set ``content`` and
        # ``break`` out of the inner ``for tool_call in tool_calls:`` loop.
        # But ``tools_dispatched`` is True (a tool was dispatched), so the
        # ``if not tools_dispatched: break`` below would NOT break the
        # outer ``while`` loop — the model would get another completion and
        # could drift / hallucinate. These guards set a terminal-failure
        # flag on ``_state``; honor it here by breaking the ``while`` so
        # the deterministic failure ``content`` reaches the output guard.
        if (
            _state.entry_failed
            or _state.plan_failed
            or _state.handoff_count > _max_handoffs_per_turn
            or (
                _state.fork_results is not None
                and any(r.status == "no_tools" for r in _state.fork_results)
            )
        ):
            break

        if not tools_dispatched:
            # No tool calls were dispatched — model returned text. Use the
            # canonical extractor so reasoning models that put the answer in
            # ``reasoning_content`` (empty ``content``) still surface it.
            content = completion_content(completion).strip()
            break

    if not content:
        # ── Unified turn lifecycle: iteration-cap with no content → yield ──
        # The loop exited due to ``max_tool_iterations`` while the model was
        # still calling tools and produced no usable text. The unified
        # lifecycle prefers a yield when the mode allows it (the user can
        # redirect instead of getting a possibly-incoherent synthesis). When
        # the mode forbids yields (action modes), fall through to the legacy
        # forced-final (preserving the truncate as the last resort).
        if _pending_yield is not None:
            content = _pending_yield.question
            _yielded = True
        elif _resolve_yield_policy(mode_config).allowed:
            _yield_policy = _resolve_yield_policy(mode_config)
            _pending_yield = YieldAction(
                question=(
                    _yield_policy.default_question
                    or (
                        "I've hit my iteration budget for this turn. "
                        "How would you like me to proceed?"
                    )
                ),
                context=f"iteration cap {max_tool_iterations} reached with no final content",
                target_mode_id=_yield_policy.fallback_mode,
            )
            content = _pending_yield.question
            _yielded = True
        else:
            # Action mode with no yield escape: preserve the legacy forced final.
            # Diagnostic: log why the loop exited with no usable content so this
            # class of failure is diagnosable without re-reading the full log.
            # Common exit causes: max_tool_iterations reached, a hard-failure
            # guard (entry_failed / handoff cap / fork no_tools) set ``content``,
            # or the model returned an empty/whitespace completion (often a
            # reasoning model whose visible answer landed in reasoning_content).
            _log.warning(
                "turn loop exited with empty content mode=%s iteration=%d/%d "
                "tool_calls_total=%d handoffs=%d entry_failed=%s fork_no_tools=%s "
                "transitioned_to=%s routing_hint=%s",
                mode_id,
                _state.iteration,
                max_tool_iterations,
                _state.total_tool_calls,
                _state.handoff_count,
                _state.entry_failed,
                bool(
                    _state.fork_results is not None
                    and any(r.status == "no_tools" for r in _state.fork_results)
                ),
                transitioned_to,
                bool(routing_hint),
            )
            # The loop exited due to max_tool_iterations while the model was
            # still calling tools. Force one final no-tools completion so the
            # model synthesizes its findings into a coherent response instead
            # of the generic "did not produce a final response" fallback.
            final_reasoning_tokens = 0
            try:
                final_completion = await complete(
                    messages=messages,
                    model_def=effective_model_def,
                    max_tokens=4096,
                    temperature=0.4,
                    tools=None,
                )
                # Canonical extractor: a reasoning model may put the synthesis
                # in ``reasoning_content``; reading only ``.content`` here would
                # repeat the zero-output bug on the final forced completion.
                final_content = completion_content(final_completion)
                if final_content.strip():
                    content = final_content.strip()
                    final_usage = _extract_completion_usage(final_completion)
                    final_reasoning_tokens = final_usage[4]
                    _state.total_prompt_tokens += final_usage[0]
                    _state.total_completion_tokens += final_usage[1]
                    _state.total_tokens += final_usage[2]
                    _state.total_duration_ms += final_usage[5]
                    _record_model_usage(
                        effective_model_def,
                        p=final_usage[0],
                        c=final_usage[1],
                        t=final_usage[2],
                        ct=final_usage[3],
                        rt=final_usage[4],
                        d=final_usage[5],
                        lat=final_usage[6],
                    )
                else:
                    _log.warning(
                        "iteration-cap final completion also returned empty "
                        "content (mode=%s, finish_reason=%s, reasoning_tokens=%d); "
                        "falling back to generic message",
                        mode_id,
                        getattr(final_completion, "finish_reason", "unknown"),
                        final_reasoning_tokens,
                    )
            except Exception as exc:  # pylint: disable=broad-except
                _log.warning("_state.iteration-cap final completion failed: %s", exc)
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
                        terminal_usage = _extract_completion_usage(final_completion)
                        _state.total_prompt_tokens += terminal_usage[0]
                        _state.total_completion_tokens += terminal_usage[1]
                        _state.total_tokens += terminal_usage[2]
                        _state.total_duration_ms += terminal_usage[5]
                        _record_model_usage(
                            effective_model_def,
                            p=terminal_usage[0],
                            c=terminal_usage[1],
                            t=terminal_usage[2],
                            ct=terminal_usage[3],
                            rt=terminal_usage[4],
                            d=terminal_usage[5],
                            lat=terminal_usage[6],
                        )
                except Exception as exc:  # pylint: disable=broad-except
                    _log.warning("terminal-stage summary failed: %s", exc)
            workflow_terminated = True

    # ── Phase 3: Post-turn ────────────────────────────────────────────
    # Output guard runs before on_complete/store_mem so the persisted and
    # delivered text is the sanitized/safe-fallback version. Blocked
    # outputs emit a guardrail.audit event and skip the post-turn slice.
    output_guard_result = await hooks.context_seams.output_guard.validate(content, guard_surface)
    turn_blocked = False
    if (
        _state.entry_failed
        or _state.plan_failed
        or (
            _state.fork_results and any(r.status == "no_tools" for r in (_state.fork_results or []))
        )
    ):
        # A handoff entry operation failed, or a fork child was marked
        # ``no_tools`` (hallucinated success). Either way the turn broke
        # with a deterministic failure message. Mark blocked so the host
        # skips opportunistic post-turn house-keeping (same contract as
        # an output guard block) and the user sees a hard failure rather
        # than a hallucinated success.
        turn_blocked = True
    elif not output_guard_result.allowed:
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
                # ── Unified turn lifecycle: stamp the checkpoint ──────────
                # ``_yielded`` is set when the loop broke via a forced or
                # model-emitted ``YieldAction``. A yielded checkpoint is
                # resumable by ``/proceed`` (the host loads the bundle, sees
                # ``lifecycle_state="yielded"``, injects the user's reply,
                # and resumes the turn against the same ``WorkflowInstance``
                # step). Otherwise the turn committed cleanly.
                _ckpt_state: Literal["working", "yielded", "committed", "stalled"] = (
                    "yielded" if _yielded else "committed"
                )
                _ckpt_question = content if _yielded else ""
                bundle = _build_checkpoint_bundle(
                    mode_id=mode_id,
                    transitioned_to=transitioned_to,
                    is_workflow=is_workflow,
                    hooks=hooks,
                    lifecycle_state=_ckpt_state,
                    awaiting_input_question=_ckpt_question,
                )
                if bundle is not None:
                    await hooks.on_checkpoint(bundle)
            except Exception as exc:  # pylint: disable=broad-except
                _log.warning("on_checkpoint hook failed: %s", exc)

    # ── Plan progress rendering ──────────────────────────────────────
    # When a plan_workflow_op instance is active and has remaining steps,
    # append a short progress summary so the user sees which step just
    # finished and what's next. Only appended when there is visible
    # content (don't append to an empty error response).
    if content and hooks.workflow_instance is not None:
        try:
            _progress_instance = hooks.workflow_instance()
        except Exception:  # pylint: disable=broad-except
            _progress_instance = None
        if (
            _progress_instance is not None
            and getattr(_progress_instance, "spawned_from", "") == "plan_workflow_op"
            and getattr(_progress_instance, "declared_steps", ())
        ):
            try:
                from kernel.workflow_instance import (
                    render_workflow_progress,
                )

                _progress = render_workflow_progress(_progress_instance)
                if _progress:
                    content = f"{content}\n\n{_progress}"
            except Exception:  # pylint: disable=broad-except
                _log.warning("render_workflow_progress failed: %s", exc)

    return TurnResult(
        response_text=content,
        tool_call_count=_state.total_tool_calls,
        called_tool_names=set(_state.called_tool_names),
        iteration_count=_state.iteration,
        prompt_tokens=_state.total_prompt_tokens,
        completion_tokens=_state.total_completion_tokens,
        total_tokens=_state.total_tokens,
        cached_tokens=_state.total_cached_tokens,
        reasoning_tokens=_state.total_reasoning_tokens,
        duration_ms=_state.total_duration_ms,
        latency_ms=_state.total_latency_ms,
        slice_outcomes=slice_outcomes,
        blocked=turn_blocked,
        transitioned_to=transitioned_to,
        handoff_to=handoff_to,
        fork_results=fork_results,
        workflow_terminated=workflow_terminated,
        per_model_usage=list(_state.per_model_usage.values()),
        per_mode_usage=list(_state.per_mode_usage.values()),
    )


# ---------------------------------------------------------------------------
# Re-exports from extracted turn_phases submodules.
#
# These symbols were historically defined in this module. They have been
# extracted to kernel.turn_phases.* for complexity management but are
# re-exported here so the 21 test files that import them via
# ``from kernel.turn_runner import _compute_capability_routing_hint`` etc.
# continue to resolve without edits.
# ---------------------------------------------------------------------------

# Routing-hint cluster (extracted in Phase 1)
# Fork fan-out cluster (extracted in Phase 2)
from kernel.turn_phases._fork import (  # noqa: F401
    _build_minimal_context,
    _execute_fork_fanout,
    _fork_child_budget,
    _fork_mode_action_tool_ids,
    _fork_mode_requires_tools,
    _run_fork_child,
    _run_fork_child_turn,
)

# Post-turn / capability-gate / serialization cluster (extracted in Phase 4)
from kernel.turn_phases._post_turn import (  # noqa: F401
    _CapabilityGateBlock,
    _coerce_non_negative_float,
    _coerce_non_negative_int,
    _extract_completion_usage,
    _serialize_tool_json,
    _stringify_pack_tool_result,
    _stringify_result,
    _to_tool_context,
    _validate_model_capabilities,
)
from kernel.turn_phases._routing_hint import (  # noqa: F401
    _compute_capability_routing_hint,
    _compute_fork_target_string,
    _detect_query_intents,
    _first_target_for_earliest_intent,
    _format_remaining_steps,
    _query_is_sequential,
    _single_handoff_target_for_gap,
    _tool_effect,
)

# Tool-schema cluster (extracted in Phase 3)
from kernel.turn_phases._tool_schemas import (  # noqa: F401
    _TRANSITION_TOOL_NAME,
    _YIELD_TOOL_NAME,
    _build_in_process_tool_schemas,
    _mode_bound_operation_ids,
    _operation_allowed_for_mode,
    _prune_model_tool_schemas,
    _rebuild_tool_schemas_for_mode,
    _tool_schema_name,
)

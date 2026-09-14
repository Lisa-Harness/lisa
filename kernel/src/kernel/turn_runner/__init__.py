"""Kernel turn runner — the single model+tool loop.

This module provides `run_turn()` and `TurnHooks`, the core conversational turn
loop that executes one mode's pipeline: pre-turn operations, model completion
with tool dispatch, and post-turn side effects.
"""

from __future__ import annotations

__all__ = [
    # Public surface
    "ContextSeams",
    "StreamedCompletionAdapter",
    "TurnHooks",
    "TurnResult",
    "execute_approved_invocation",
    "run_turn",
    # Satellite internals re-exported for the historical module namespace
    # (tests patch/mock these via ``kernel.turn_runner.<name>``).
    "_StubModelDef",
    "_aggregate_child_usage",
    "_build_budget_briefing",
    "_build_recovery_identity",
    "_build_sub_turn_contract",
    "_coordinator_routing_hint",
    "_current_recovery_stage_id",
    "_default_assemble_system_prompt",
    "_default_assemble_user_context",
    "_default_delegate_fn",
    "_default_execute_operation",
    "_default_fork_fn",
    "_default_on_turn_complete",
    "_default_resolve_model",
    "_default_slice",
    "_extract_hint_target",
    "_extract_pre_turn_briefing",
    "_fire_slice",
    "_looks_like_continuation_query",
    "_noop_complete",
    "_plan_steps_to_workflow_steps",
    "_synthesized_call_missing_all_required",
]



import json
import logging
import re
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Literal, cast

from kernel.approval import (
    ApprovalInvocation,
    ApprovalRequired,
    render_approval_request,
)
from kernel.capability_manifest import compile_capability_manifest
from kernel.config import settings as _kernel_settings
from kernel.context.sub_turn_contract import (
    render_no_write_warning,
    render_sub_turn_contract,
    render_sub_turn_midpoint_nudge,
)
from kernel.execution_policy import ExecutionPolicy, resolve_capability_invocation_limit
from kernel.ipsar_fsm import (
    get_mode_system_intro,
    get_routing_targets,
    get_workflow_catalog,
    is_routable_mode,
    is_terminal_stage,
    run_mode_entry_operations,
    run_mode_pre_turn_operations,
)
from kernel.models import (
    BlockDiagnostic,
    CallLedger,
    CapabilityManifest,
    ConversationSurface,
    DelegationResult,
    DelegationSpec,
    DelegationStatus,
    EffectReceipt,
    ForkChildResult,
    ForkResult,
    ForkSpec,
    MemoryScope,
    OperationExecutionContext,
    OperationResult,
    SurfaceContext,
    ToolExecutionContext,
    ToolResult,
    TurnBudget,
    YieldAction,
    YieldPolicy,
)
from kernel.operations.registry import (
    execute_operation,
    get_operation_catalog,
    mode_completion_effects,
)
from kernel.operations.structured_output import completion_content
from kernel.provider_contracts import (
    CompletionRequest,
    StreamedChunk,
    StreamedCompletion,
    _aggregate_streamed_chunks,
)
from kernel.recovery import RecoveryIdentity, RecoveryRecord
from kernel.seams.host_adapter import build_callable_host
from kernel.seams.protocols import (
    ContentRegistry,
    CoordinatorRouteDecision,
    DataSanitizer,
    EventBus,
    ExtensionItemBuilder,
    InputGuard,
    ModelPolicy,
    ModelRouter,
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
from kernel.system_segments import (
    N_HANDOFF_CONTEXT,
    N_LIFECYCLE_BRIEFING,
    N_ROUTING_HINT,
    SystemPromptSegments,
)
from kernel.token_usage import build_token_usage_event as _build_token_usage_event_fn
from kernel.token_usage import publish_token_usage as _publish_token_usage_fn
from kernel.tools.mcp_client import MCPToolPool, bare_tool_name
from kernel.tools.registry import execute_tool, get_tool_catalog
from kernel.tools.schema_translator import (
    openai_tool_call_to_mcp,
)
from kernel.turn_hooks import TurnHooks
from kernel.turn_phases._child_turn import classify_child_turn
from kernel.turn_phases._completion_contract import (
    AcceptTurn,
    BlockNoVerifiedWrite,
    ChatReadFootnote,
    ChatSoftAccept,
    PreexistingFailureBlock,
    SurfaceProseWithFootnote,
    resolve_completion_contract,
)
from kernel.turn_phases._dispatch import dispatch_capability
from kernel.turn_phases._entry import execute_entry_phase
from kernel.turn_phases._fork import _build_minimal_context, _execute_fork_fanout
from kernel.turn_phases._lifecycle import (
    ContinueAction as _ContinueAction,
)
from kernel.turn_phases._lifecycle import (
    FinishAction as _FinishAction,
)
from kernel.turn_phases._lifecycle import (
    LifecycleResolverSettings as _LRS,
)
from kernel.turn_phases._lifecycle import (
    PauseAction as _PauseAction,
)
from kernel.turn_phases._lifecycle import (
    StopAction as _StopAction,
)
from kernel.turn_phases._lifecycle import (
    TerminateAction as _TerminateAction,
)
from kernel.turn_phases._lifecycle import (
    TransitionAction as _TransitionAction,
)
from kernel.turn_phases._lifecycle import (
    _TurnStateView as _TSV,
)
from kernel.turn_phases._lifecycle import (
    resolve_lifecycle_action as _resolve_lifecycle,
)
from kernel.turn_phases._post_turn import (
    _extract_completion_usage,
    _to_tool_context,
    _validate_model_capabilities,
)
from kernel.turn_phases._routing import RoutingContext, apply_routing
from kernel.turn_phases._routing_hint import (
    _compute_capability_routing_hint,
    _compute_fork_targets,
    _detect_query_intents,
    _format_remaining_steps,
    emit_transition_hint,
)
from kernel.turn_phases._tool_schemas import (
    _TRANSITION_TOOL_NAME,
    _YIELD_TOOL_NAME,
    _rebuild_tool_schemas_for_mode,
    _tool_schema_name,
    build_tool_surface,
)
from kernel.workflow_position import WorkflowStep, _step_id_for

logger = logging.getLogger(__name__)


# ── Package satellites (split from the former single-file turn_runner) ────
# The engine imports these back so ``kernel.turn_runner`` keeps its historical
# module namespace: ``mock.patch("kernel.turn_runner.<name>")`` and
# ``from kernel.turn_runner import <name>`` both resolve.
from ._defaults import (
    _aggregate_child_usage,
    _build_recovery_identity,
    _coordinator_routing_hint,
    _current_recovery_stage_id,
    _default_assemble_system_prompt,
    _default_assemble_user_context,
    _default_delegate_fn,
    _default_execute_operation,
    _default_fork_fn,
    _default_on_turn_complete,
    _default_resolve_model,
    _default_slice,
    _looks_like_continuation_query,
    _noop_complete,
)
from ._helpers import (
    _extract_pre_turn_briefing,
    _fire_slice,
    _plan_steps_to_workflow_steps,
    _StubModelDef,
)
from ._result import TurnResult
from ._seams import ContextSeams
from ._streaming import StreamedCompletionAdapter

# ── Stalled-write rescue: announcement detection ───────────────────────────
# Prose that ANNOUNCES a write in future/present-continuous tense while the
# receipt ledger shows zero write attempts is the signature of the
# write-announcement loop (the 2026-09-02 incident: eight consecutive turns
# of "Writing both files now" with only read tool dispatches). The patterns
# are intentionally narrow — they must match commitment language, not
# discussion of writing in general — and are compiled once at import.
_STALLED_WRITE_PATTERNS = re.compile(
    r"(?:"
    r"\bwriting\s+(?:both\s+)?(?:the\s+)?(?:files?|scripts?|it|them|now)\b"
    r"|\b(?:i'?ll|i\s+will|let\s+me|i'm\s+going\s+to)\s+(?:now\s+)?"
    r"(?:write|create|save|generate|produce|build)\b"
    r"|\bwriting\s+(?:the\s+)?(?:\w+\s+)?(?:script|toolkit|cheat\s?sheet)s?\b"
    r"|\bno\s+more\s+stalling\b"
    r")",
    re.IGNORECASE,
)


def _synthesized_call_missing_all_required(
    tool_name: str,
    tool_arguments: dict[str, Any],
    *,
    operation_catalog: dict[str, Any],
    tool_catalog: dict[str, Any],
) -> bool:
    """Return True when a *kernel-synthesized* call is missing every required
    parameter of its target tool/operation.

    The forced-action enforcer and the do-the-work backstop synthesize tool
    calls with empty (``{}``) arguments when the model returns no tool calls.
    For routing-only syntheses (``transition_stage``, ``fork``) that is
    correct — those tools take optional/derived args. For an action-tool
    synthesis (the write-contract backstop picking the mode's first action
    tool), empty args are a guaranteed validation failure: action tools
    require user-supplied parameters (``cs_apply_cloud_init`` needs 7). The
    dispatch raises a pydantic ``ValidationError``, the result is fed back to
    the model, and the backstop can re-arm and loop — burning iterations on a
    call that can never succeed because the kernel has no values to fill.

    This guard lets the synthesis sites detect that case and abort the turn
    with the model's prose instead of dispatching a guaranteed failure. A
    turn that routed into a write-contract mode the user never actually
    requested (the 2026-08-27 cloudservers misroute) should surface the
    model's answer, not fire a destructive action with empty params.

    Returns False when the tool is unknown (MCP-native tools have no kernel
    definition — let the normal dispatch path handle them), when the tool
    has no ``parameters_schema`` (no validation contract — dispatch), or
    when at least one required parameter is supplied.
    """
    bare = bare_tool_name(tool_name)
    definition = operation_catalog.get(tool_name) or tool_catalog.get(tool_name)
    if definition is None:
        definition = operation_catalog.get(bare) or tool_catalog.get(bare)
    if definition is None:
        return False
    schema = getattr(definition, "parameters_schema", None)
    if schema is None:
        return False
    model_fields = getattr(schema, "model_fields", None)
    if not model_fields:
        return False
    required_names = [
        name
        for name, field in model_fields.items()
        if getattr(field, "is_required", lambda: False)()
    ]
    if not required_names:
        return False
    supplied = {k for k, v in (tool_arguments or {}).items() if v is not None}
    return not any(name in supplied for name in required_names)


#: Maximum characters of the operator-visible tool display label emitted in
#: the ``tool_dispatch`` stage payload. Long enough to read a command, short
#: enough that hosts don't need to truncate again for a one-line status.
_DISPLAY_LABEL_MAX = 80


def _resolve_tool_display_label(
    template: str,
    params: dict[str, Any],
) -> str:
    """Render a tool's ``display_template`` against the call parameters.

    Same tolerant ``format_map`` semantics as ``_format_read_abbreviation``:
    missing keys render as ``{key}`` instead of raising, so a template that
    references an absent optional parameter still produces a readable label.
    Any format error (or an empty template) returns an empty string so the
    host falls back to its own primary-argument preview. The label is
    newline-flattened and truncated to ``_DISPLAY_LABEL_MAX``.

    This is display-only: it surfaces on host progress UIs (CLI once mode,
    REPL transcript) and never enters the model's context.
    """
    if not template or not isinstance(template, str) or not template.strip():
        return ""

    class _MissingKeyDict(dict):
        def __missing__(self, key: str) -> str:  # type: ignore[override]
            return "{" + key + "}"

    try:
        rendered = template.format_map(_MissingKeyDict(dict(params or {})))
    except Exception:  # pylint: disable=broad-except
        return ""
    rendered = str(rendered).replace("\n", " ").strip()
    if len(rendered) > _DISPLAY_LABEL_MAX:
        return rendered[: _DISPLAY_LABEL_MAX - 1] + "…"
    return rendered


# Turn-runner guard caps live in ``kernel.config.Settings`` (fields
# ``max_tool_iterations`` / ``fork_child_max_tool_iterations`` /
# ``per_step_max_iterations`` / ``per_step_max_tool_calls`` /
# ``forced_action_non_compliant_cap`` /
# ``stage_nudge_threshold`` / ``stage_nudge_cap`` / ``max_handoffs_per_turn`` /
# ``duplicate_hint_limit`` / ``hint_cycle_min_len`` / ``hint_override_limit``)
# so hosts can tune them via env vars without editing kernel source. For
# the complete registry of invariants, provenance incidents, pinning tests,
# and scenario matrix cells, see ``docs/lessons/turn-loop-invariants.md``.
# ``_run_turn_inner`` reads them from the shared ``_kernel_settings``
# singleton and, for the few that are referenced multiple times per turn,
# binds a local alias once at the top of the loop.


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
    stage_nudge_cap: int,
    tool_schemas: list[dict[str, Any]],
    operation_catalog: dict[str, Any],
    tool_catalog: dict[str, Any],
    execution_policy: Any,
    budget_exhausted: bool,
    guidance: tuple[str, ...] = (),
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

    **Sub-turn contract** (fork / delegate children): when
    ``state.sub_turn_kind`` is set and ``state.action_tool_ids`` is
    non-empty, a ``## Sub-Turn Contract`` block is prepended to the
    briefing. It frames the iteration budget as a write-deadline and
    names the action tools the child must invoke to avoid a ``no_tools``
    abort — the 2026-08-03 incident: a fork child saw only ``iters: 49/50``
    and read-looped to the cap without ever writing. The contract is
    generic over ``sub_turn_kind`` so delegate (``"delegate"``) reuses the
    same seam without reworking the briefing.

    **Guidance lines** (Phase 7: one control channel): soft nudges — the
    resolver's ``ContinueAction.nudge``, the sub-turn midpoint write-nudge,
    and the persistent post-handoff chain-advance directive — ride this
    block as extra lines. They replace the old synthesized ``user``
    messages; the turn loop never appends ``role: "user"`` messages
    (user messages come only from the host's initial query).

    Returns an empty string when there is nothing actionable to report
    (e.g. unlimited everything, iteration 1 with no per-target limits, and
    no guidance lines).
    """
    lines: list[str] = []
    # ── Iteration budget ──
    # Uses the 1-indexed iteration counter so the model never guesses
    # whether "47/50" means "47 remaining" or "47 used": ``iteration: 3
    # of 50`` is unambiguously "I am on iteration 3 of a 50-iteration
    # budget." The previous ``remaining/total`` format (``iters: 47/50``)
    # was consistently misread as a progress bar during metaanalysis.
    remaining_iterations = max_tool_iterations - state.iteration
    if max_tool_iterations > 0 and state.iteration > 0:
        lines.append(f"iteration: {state.iteration} of {max_tool_iterations}")
    # ── Stage budget (the per-mode ceiling that actually fires) ────────
    # ``stage_nudge_cap`` is the real bound on iterations in one mode
    # before the harness pauses the turn. The global ``max_tool_iterations``
    # (300) is never reached in practice; the stage cap (default 9) is.
    # Without surfacing it, the model sees only ``iter 3/300`` and has no
    # way to reconcile that with prior cap-fire messages in the thread
    # ("stuck in explainer for 6 iterations (cap)"). The format is
    # explicitly ``N completed with M remaining`` to prevent the
    # completed/remaining confusion that the old ``N/M`` format caused.
    turns_in_stage = getattr(state, "turns_in_stage", 0) or 0
    if stage_nudge_cap > 0 and turns_in_stage > 0:
        stage_remaining = max(0, stage_nudge_cap - turns_in_stage)
        lines.append(
            f"stage budget: {turns_in_stage} completed with {stage_remaining} remaining"
        )
    # ── Per-step sub-budget (only when a WorkflowPosition is wired) ──
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
    # ── Sub-turn contract (fork / delegate children) ──────────────────
    # A fork/delegate child whose mode declares action tools must be told
    # the budget is a write-deadline, not exploration room — and that
    # reads do not count toward completion. Without this the child sees
    # only ``iters: 49/50`` and read-loops to the cap (the 2026-08-03
    # incident). The contract is generic over ``sub_turn_kind`` so
    # delegate reuses the seam.
    sub_contract = _build_sub_turn_contract(
        state=state,
        remaining_iterations=remaining_iterations,
        budget_exhausted=budget_exhausted,
    )
    if not lines and not sub_contract and not guidance:
        return ""
    sections: list[str] = []
    if lines:
        sections.append(" · ".join(lines))
    # Guidance lines (Phase 7: one control channel) ride the same
    # lifecycle block so the per-iteration marker strip removes them with
    # the briefing — no second marker, no stale accumulation.
    sections.extend(line for line in guidance if line)
    lifecycle_block = "## Turn Lifecycle\n\n" + "\n\n".join(sections) if sections else ""
    if sub_contract and lifecycle_block:
        return sub_contract + "\n\n" + lifecycle_block
    if sub_contract:
        return sub_contract
    return lifecycle_block


def _build_sub_turn_contract(
    *,
    state: Any,
    remaining_iterations: int,
    budget_exhausted: bool,
) -> str:
    """Build the ``## Sub-Turn Contract`` block for fork/delegate children.

    Returns an empty string when this is the parent turn, the mode has no
    action tools (read-only modes legitimately complete without writing),
    or the child already wrote (at least one action tool was called).

    The block names the action tools, frames the remaining iterations as a
    write-deadline, and previews the abort: if the budget exhausts without
    a write, the child's output is discarded as unverified. The framing
    adapts to ``sub_turn_kind`` — a fork child's output is discarded;
    a delegate child (planned) returns an honest ``no_tools`` status to
    its parent. The parent-turn case (``sub_turn_kind is None``) returns
    ``""`` so the parent briefing is unchanged.

    The framing prose, deadline sentence, and budget-exhausted tail are
    authored Markdown loaded through ``kernel.context.sub_turn_contract``
    (``prompts/sub_turn/{fork,delegate}_contract.md`` and
    ``prompts/sub_turn/budget_exhausted.md``). This function keeps only the
    ``sub_turn_kind`` discriminator, the action-tool set, and the
    ``str.format_map`` substitution glue — no model-facing prose lives here.
    """
    kind = getattr(state, "sub_turn_kind", None)
    if not kind:
        return ""
    action_ids = getattr(state, "action_tool_ids", None) or set()
    if not action_ids:
        return ""
    # Once the child has called at least one action tool, the success
    # criterion is met — drop the contract so it stops dominating the
    # prompt for the rest of the child's iterations.
    called = getattr(state, "called_tool_names", set()) or set()
    if called & action_ids:
        return ""
    tools_list = ", ".join(sorted(action_ids))
    return render_sub_turn_contract(
        kind=kind,
        tools_list=tools_list,
        remaining_iterations=remaining_iterations,
        budget_exhausted=budget_exhausted,
    )


def _is_workflow_position_complete(position: Any) -> bool:
    """Return True when the workflow position has no remaining work.

    Thin wrapper around ``kernel.workflow_position.is_complete`` with
    graceful degradation: any exception or missing attribute returns
    ``True`` so the per-step cap falls through to the existing final
    completion rather than synthesizing a handoff to a step that may not
    exist.
    """
    try:
        from kernel.workflow_view import is_complete

        return bool(is_complete(position))
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug("is_complete check raised (degrading to complete=True): %s", exc)
        return True


def _next_remaining_step(position: Any) -> Any:
    """Return the first uncompleted ``WorkflowStep`` after the current one.

    Wraps ``kernel.workflow_position.remaining_steps`` and selects the
    step immediately following ``current_step_id`` in declaration order
    (falling back to the first remaining step when the current step is
    already advanced or missing). Returns ``None`` when there are no
    remaining steps or on any exception.
    """
    try:
        from kernel.workflow_view import remaining_steps

        remaining = remaining_steps(position)
        if not remaining:
            return None
        current = getattr(position, "current_step_id", "") or ""
        # Prefer the step immediately after the current one in
        # declaration order; fall back to the first remaining step.
        for step in remaining:
            if step.step_id != current:
                return step
        return remaining[0]
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug("remaining_steps check raised (degrading to None): %s", exc)
        return None


def _resolve_yield_policy(mode_config: Any, override: YieldPolicy | None = None) -> YieldPolicy:
    """Return the ``YieldPolicy`` for one mode, tolerating missing config.

    The policy is authored in ``_mode.md`` front matter and loaded onto
    ``WorkflowModeConfig.yield_policy`` by ``kernel.ipsar_fsm``. When the
    config or policy is absent (e.g. a minimal test fixture), return the
    default-empty policy (inline asks allowed, no fallback) so the turn
    degrades to the legacy "ask in place" behavior rather than crashing.

    ``override`` is an optional ``YieldPolicy`` supplied by the caller (the
    fork fan-out passes ``YieldPolicy(allowed=False)`` unconditionally so a
    fork child's ``yield`` tool surface and lifecycle resolution follow the
    no-ask invariant). When non-None, it takes precedence over the mode's
    authored policy — this is the single seam by which a fork child suppresses
    ``yield`` even when the foreground mode allows it.
    """
    if override is not None:
        return override
    if mode_config is None:
        return YieldPolicy()
    policy = getattr(mode_config, "yield_policy", None)
    if isinstance(policy, YieldPolicy):
        return policy
    return YieldPolicy()


def _next_remaining_step_mode_id(hooks: Any) -> str:
    """Return the mode id of the next uncompleted workflow step, or "".

    Thin wrapper around ``_next_remaining_step`` that reads the active
    ``WorkflowPosition`` from ``hooks.workflow.workflow_position``. Used by the
    stuck-in-stage yield path to route a forced yield to the next step's
    mode when the current mode forbids inline asks (action modes).
    """
    if hooks.workflow.workflow_position is None:
        return ""
    try:
        position = hooks.workflow.workflow_position()
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug("hooks.workflow.workflow_position() raised in next step lookup: %s", exc)
        return ""
    step = _next_remaining_step(position)
    return step.mode_id if step is not None else ""


def _promises_routing_contract(mode_config: Any) -> bool:
    """Return True when the mode declares ``entry_op_kind: routing``.

    The declaration governs *failure diagnosis only*: when a
    routing-declared entry op returns an empty ``target_stage_id``, the
    turn terminates with a deterministic recovery prompt and the model is
    marked unreliable for planning. The handoff itself is data-driven —
    any non-empty ``target_stage_id`` in an entry-op result is honored,
    declared or not. Work entry ops (the default; e.g. ``web_search``)
    return their own output shape and are never misread as failed
    planners (the 2026-08-06 ``plan → search`` stall). ``plan_workflow``
    is the sole routing implementor today; a future ``delegate`` op joins
    by declaring ``entry_op_kind: routing`` in its mode front matter — no
    kernel code change.
    """
    return getattr(mode_config, "entry_op_kind", "") == "routing"


def _active_position_resume_target(hooks: Any) -> str:
    """Return the resume target mode from an active workflow position, or "".

    Origin-blind: reads the active ``WorkflowPosition`` (when wired) and,
    if it is non-complete with a remaining step, returns that step's
    ``mode_id``. Used by the mid-turn handoff path so re-entering the
    originating mode of a chain (e.g. ``plan``) while a broader workflow
    is in flight resumes that position instead of blindly re-running the
    entry op against a continuation/status query (the 2026-08-01
    plan→markdown→learn→re-plan failure). Returns "" when no resumable
    position is active, so the caller falls through to the normal path.
    """
    if hooks.workflow.workflow_position is None:
        return ""
    try:
        position = hooks.workflow.workflow_position()
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug("hooks.workflow.workflow_position() raised in resume lookup: %s", exc)
        return ""
    if position is None:
        return ""
    if _is_workflow_position_complete(position):
        return ""
    step = _next_remaining_step(position)
    return step.mode_id if step is not None else ""


def _resume_in_flight_target(
    *,
    hooks: TurnHooks,
    mode_id: str,
    query: str,
) -> str:
    """Return the resume target when a continuation query resumes a plan.

    The resume-in-flight detector: when the user replies with a bare
    imperative ("proceed", "continue", "go ahead") while parked on a
    work entry-op mode (e.g. ``search`` after a model-driven handoff),
    the literal phrase must NOT be fed to that mode's entry op — it has
    no search/research intent and would be sent to the backend as the
    query (the 2026-08-08 ``proceed`` → Brave ``q=proceed+meaning``
    incident). Instead, the active ``WorkflowPosition`` is the source
    of truth: return its next remaining step's ``mode_id`` so Phase 1
    can skip the entry op and let the coordinator routing hint drive the
    handoff to the next planned step.

    Returns "" when the query is not a continuation phrase, no position
    is active, the position is complete, or the next step's mode equals
    the current mode (the parked step still owns this turn — the entry
    op should run for it). Callers fall through to the normal entry-op
    path on "".
    """
    if not _looks_like_continuation_query(query):
        return ""
    target = _active_position_resume_target(hooks)
    if not target or target == mode_id:
        return ""
    return target



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
    yield_policy_override: YieldPolicy | None = None,
    sub_turn_kind: str | None = None,
    delegate_depth: int = 0,
    stage_id_override: str | None = None,
) -> TurnResult:
    """Execute one conversational turn with the full model+tool loop.

    ``suppress_routing_hint`` is set by the fork fan-out so child turns do
    not recompute and inject their own routing hints. A fork child already
    runs in its target mode by design; a hint telling it to hand off again
    creates a loop (the child sees the original multi-intent query, detects
    intents its mode doesn't satisfy, and is told to re-route — which a
    weak/strict model turns into an invalid-content 500). Fork children
    must run their mode to completion without re-routing.

    ``yield_policy_override`` is set by the fork fan-out to
    ``YieldPolicy(allowed=False)`` so a fork child's ``yield`` tool surface
    and lifecycle resolution enforce the no-ask invariant. A fork child runs
    with no interactive channel — the user cannot answer a mid-turn question
    — so the ``yield`` tool is never advertised to the child. A mode that
    declares ``fork_allowed: true`` commits to completing or failing without
    asking.

    ``sub_turn_kind`` is set by the fork fan-out (``"fork"``) and, once the
    delegate operation lands, by ``_default_delegate_fn`` (``"delegate"``).
    ``None`` for the parent turn. When set, the budget briefing emits a
    ``## Sub-Turn Contract`` block that frames the iteration budget as a
    write-deadline and names the action tools the child must invoke to
    avoid a ``no_tools`` abort, and the turn loop injects a one-shot
    midpoint write-nudge (see ``_build_sub_turn_contract`` and the
    midpoint nudge in ``_run_turn_inner``). The action-tool set is
    resolved from the mode's declared completion effects via
    ``mode_completion_effects`` (shared with the child-turn classifier).

    This is the single turn loop shared by lightweight hosts and the production app.
    Both call the same function — they differ only in which `TurnHooks` they
    provide.

    Phases:
    1. **PRE-TURN**: Resolve mode config, run pre-turn/entry operations,
       assemble system prompt and user context, recall memory snippet.
    2. **TOOL LOOP**: Call hooks.models.complete → dispatch MCP tool calls via
       MCPToolPool → feed results back → repeat until the model returns
       content without tools or max_tool_iterations is reached.
    3. **POST-TURN**: Call hooks.lifecycle.on_turn_complete.

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
        NotImplementedError: If hooks.models.complete is None.
    """
    _log = logger_override or logger

    if max_tool_iterations is None:
        max_tool_iterations = _kernel_settings.max_tool_iterations
    if per_step_max_iterations is None:
        per_step_max_iterations = _kernel_settings.per_step_max_iterations
    if per_step_max_tool_calls is None:
        per_step_max_tool_calls = _kernel_settings.per_step_max_tool_calls

    # ── Ratio-based iteration cap scaling ────────────────────────────
    # When a workflow position with N declared steps is active, scale the
    # global iteration ceiling to N × per_step_max_iterations so each step
    # gets its full sub-budget without starving later steps. Only scales
    # *up* — the base default covers ad-hoc single turns that may burn
    # iterations on read-only exploration. A 5-step plan with per_step=75
    # gets 375 total iterations instead of the flat 300 ceiling.
    if hooks.workflow.workflow_position is not None:
        try:
            _ratio_pos = hooks.workflow.workflow_position()
        except Exception:  # pylint: disable=broad-except
            _ratio_pos = None
        if _ratio_pos is not None:
            _declared = getattr(_ratio_pos, "declared_steps", ())
            if _declared:
                _step_count = max(1, len(_declared))
                _scaled = _step_count * per_step_max_iterations
                max_tool_iterations = max(max_tool_iterations, _scaled)
                _log.debug(
                    "ratio cap: %d steps × %d per_step = %d global (base=%d)",
                    _step_count,
                    per_step_max_iterations,
                    max_tool_iterations,
                    _kernel_settings.max_tool_iterations,
                )

    if execution_policy is not None:
        if execution_policy.max_model_calls is not None:
            max_tool_iterations = min(max_tool_iterations, execution_policy.max_model_calls)
        if turn_budget is None:
            turn_budget = TurnBudget(
                max_tool_calls=execution_policy.max_capability_calls or 0,
                max_total_tokens=execution_policy.max_total_tokens or 0,
                max_completion_tokens=execution_policy.max_completion_tokens or 0,
                max_wall_ms=execution_policy.max_wall_ms or 0,
            )

    mode_config = get_workflow_catalog().get(mode_id)

    if hooks.services.recovery_service is not None:
        try:
            await hooks.services.recovery_service.mark_running_indeterminate(_build_recovery_identity(hooks))
        except Exception as exc:  # pylint: disable=broad-except
            _log.warning("recovery startup reconciliation failed: %s", exc)

    # Resolve hook callables with defaults
    resolve_model = hooks.models.resolve_model or _default_resolve_model
    complete = hooks.models.complete or _noop_complete
    assemble_system = hooks.context.assemble_system_prompt or _default_assemble_system_prompt
    assemble_user = hooks.context.assemble_user_context or _default_assemble_user_context
    exec_op = hooks.operations.execute_operation or _default_execute_operation
    on_complete = hooks.lifecycle.on_turn_complete or _default_on_turn_complete

    # ── MCP tool pool ──────────────────────────────────────────────────
    # The kernel speaks MCP tool schema internally. Tool discovery and
    # dispatch go through MCPToolPool; provider-specific schemas are
    # translated at the provider edge by schema_translator.
    #
    # When the host provides a long-lived warm pool (``hooks.tools.mcp_tool_pool``),
    # reuse it across turns without spawning/tearing down subprocesses per
    # turn. The host owns that pool's lifecycle (start at init, aclose at
    # exit). Otherwise fall back to a per-turn pool built from
    # ``hooks.tools.mcp_tool_servers`` (cold servers only when a warm pool is
    # present, or all servers when no warm pool is set).
    host_pool = hooks.tools.mcp_tool_pool
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
            yield_policy_override=yield_policy_override,
            sub_turn_kind=sub_turn_kind,
            delegate_depth=delegate_depth,
            stage_id_override=stage_id_override,
        )
    pool = MCPToolPool(hooks.tools.mcp_tool_servers)
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
            sub_turn_kind=sub_turn_kind,
            delegate_depth=delegate_depth,
            stage_id_override=stage_id_override,
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
    yield_policy_override: YieldPolicy | None = None,
    sub_turn_kind: str | None = None,
    delegate_depth: int = 0,
    stage_id_override: str | None = None,
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
            yield_policy_override=yield_policy_override,
            sub_turn_kind=sub_turn_kind,
            delegate_depth=delegate_depth,
            stage_id_override=stage_id_override,
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

    if invocation.session_plan_digest != hooks.identity.session_plan_digest:
        raise ValueError("Approval request belongs to a different session plan.")

    async def _dispatch(pool: MCPToolPool) -> ToolResult | OperationResult:
        # Tools that declare ``stream_output`` publish incremental lines on
        # the ``lisa.tool_output`` bus topic; the same turn-level subscription
        # pattern as _run_turn_inner applies here (approval resume also runs
        # inside the host process with the transport attached).
        _tool_output_bus = getattr(hooks.context_seams, "event_bus", None)

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
                tool_ctx.stream_output = bool(getattr(definition, "stream_output", False))
                tool_ctx.event_bus = _tool_output_bus
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
            "user_name": hooks.identity.user_name,
            "principal_key": hooks.identity.principal_key,
            "conversation_id": hooks.identity.conversation_id,
            "surface_type": cast(ConversationSurface, identity.surface_type),
            "surface_id": identity.surface_id,
            "memory_scope": cast(MemoryScope, identity.memory_scope),
            "mode_id": identity.mode_id,
            "query": "",
            "pack_configuration_resolver": hooks.services.pack_configuration_resolver,
            "approval_service": hooks.services.approval_service,
            "approval_request_id": invocation.request_id,
            "session_plan_digest": hooks.identity.session_plan_digest,
            "code_change_artifact_store": hooks.services.code_change_artifact_store,
            "data_dir": hooks.services.data_dir,
            "model_router": hooks.context_seams.model_router,
        }
        if invocation.target_kind == "tool":
            _tool_def = get_tool_catalog().get(invocation.target_id)
            return await execute_tool(
                invocation.target_id,
                ToolExecutionContext(
                    tool_id=invocation.target_id,
                    parameters=invocation.parameters,
                    stream_output=bool(getattr(_tool_def, "stream_output", False)) if _tool_def else False,
                    event_bus=_tool_output_bus,
                    **common,
                ),
            )

        operation_context = OperationExecutionContext(
            operation_id=invocation.target_id,
            params=invocation.parameters,
            state_provider=hooks.services.state_provider,
            host=build_callable_host(
                complete_fn=hooks.models.complete or _noop_complete,
                resolve_model_fn=hooks.models.resolve_model or _default_resolve_model,
                read_authored_text_fn=getattr(
                    hooks.context_seams.content_registry,
                    "read_authored_text",
                    None,
                ),
                execute_tool_fn=_nested_tool_executor,
                execute_operation_fn=hooks.operations.execute_operation or _default_execute_operation,
                gateway_provider_fn=lambda: hooks.context_seams.provider_gateways,
            ),
            **common,
        )
        execute_operation_fn = hooks.operations.execute_operation or _default_execute_operation
        return await execute_operation_fn(invocation.target_id, operation_context)

    if hooks.tools.mcp_tool_pool is not None:
        return await _dispatch(hooks.tools.mcp_tool_pool)

    pool = MCPToolPool(hooks.tools.mcp_tool_servers)
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
    yield_policy_override: YieldPolicy | None = None,
    sub_turn_kind: str | None = None,
    delegate_depth: int = 0,
    stage_id_override: str | None = None,
) -> TurnResult:
    """Inner turn loop that owns the MCPToolPool lifecycle via the caller."""

    _log = logger_override or logger

    # ── Delegate hook wiring ──────────────────────────────────────────
    # Build the ``delegate_fn`` closure that the pack-level ``delegate``
    # operation calls via ``ctx.host.delegate(spec)``. The closure captures
    # the turn's hooks, budget, execution policy, and the current depth so
    # the delegate child runs as a ``sub_turn_kind="delegate"`` turn at
    # ``depth + 1``. When the host provides its own ``hooks.operations.delegate_fn``,
    # that takes precedence (custom delegation backend). Otherwise the
    # kernel's ``_default_delegate_fn`` is used. The closure is built
    # once per turn so every operation dispatched in this turn sees the
    # same delegate hook.
    _delegate_depth = delegate_depth
    if hooks.operations.delegate_fn is not None:
        _delegate_fn: Callable[..., Awaitable[DelegationResult]] = hooks.operations.delegate_fn
    else:

        async def _delegate_fn(
            spec: DelegationSpec,
            *,
            _hooks: TurnHooks = hooks,
            _budget: TurnBudget | None = turn_budget,
            _policy: ExecutionPolicy | None = execution_policy,
            _depth: int = _delegate_depth,
            _logr: logging.Logger | None = logger_override,
        ) -> DelegationResult:
            return await _default_delegate_fn(
                spec,
                hooks=_hooks,
                turn_budget=_budget,
                execution_policy=_policy,
                depth=_depth + 1,
                logger_override=_logr,
            )

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
        ForcedAction,
        _TurnState,
    )
    from kernel.turn_phases._turn_state import (
        budget_forces_final as _budget_forces_final_fn,
    )
    from kernel.turn_phases._turn_state import (
        emit_stage as _emit_stage_fn,
    )
    from kernel.turn_phases._turn_state import (
        emit_tool_output as _emit_tool_output_fn,
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
    # ── Completion-contract state (sub-turns AND parent) ──────────
    # Resolve the mode's required side-effecting capabilities once at
    # turn start so the budget briefing and the midpoint write-nudge can
    # name them without re-resolving every iteration, and so the Phase-3
    # parent-turn receipt check (Phase 8) validates the parent against
    # the same contract fork children are held to.
    # ``mode_completion_effects`` is the canonical resolver shared with
    # the child-turn classifier, so the briefing names the same action
    # tools the validator checks — including ``mode_allowlist``-admitted
    # operations the old authored-only resolver missed. Refreshed on
    # every mid-turn mode handoff so the contract follows the mode the
    # turn actually ends in.
    if sub_turn_kind:
        _state.sub_turn_kind = sub_turn_kind
    try:
        _state.action_tool_ids = mode_completion_effects(mode_id, mcp_pool=pool)
    except Exception:  # pylint: disable=broad-except
        _state.action_tool_ids = set()
    _on_stage_hook = hooks.lifecycle.on_stage

    # Turn-runner guard caps, read once from the kernel settings singleton
    # (``kernel.config.Settings``). The three referenced multiple times
    # per turn are bound to locals to avoid repeated attribute lookups in
    # the hot loop; the rest are read directly from ``_kernel_settings``
    # at their single use site. See the Settings fields for env-var
    # overrides and the incidents each cap guards against.
    _max_handoffs_per_turn = _kernel_settings.max_handoffs_per_turn
    _hint_override_limit = _kernel_settings.hint_override_limit
    # The fork fan-out passes ``YieldPolicy(allowed=False)`` here so a fork
    # child's ``yield`` tool surface and lifecycle resolution enforce the
    # no-ask invariant. ``None`` (foreground / non-fork path) means
    # ``_resolve_yield_policy`` reads the mode's authored ``yield_policy``.
    _yield_policy_override = yield_policy_override

    # ── Unified routing applier (Phase 3) ─────────────────────────────
    # The routing context bundles the host/session identity once so every
    # handoff origin (planner entry op, model transition_stage, forced
    # synthesis) saves the coordinator decision under the same identity via
    # ``kernel.turn_phases._routing``.
    _routing_ctx = RoutingContext(
        query=query,
        surface_id=str(hooks.identity.surface_id or ""),
        conversation_id=str(hooks.identity.conversation_id or ""),
        user_id=str(hooks.identity.user_id or ""),
        session_plan_digest=str(hooks.identity.session_plan_digest or ""),
    )

    async def _emit_stage(label: str, data: dict[str, Any] | None = None) -> None:
        await _emit_stage_fn(_state, _on_stage_hook, _log, label, data)

    # ── Tool-output streaming wiring ────────────────────────────────
    # Pack tools that declare ``stream_output: true`` publish incremental
    # operator-visible lines on the ``lisa.tool_output`` event-bus topic
    # (``ctx.event_bus``). When a bus is present, this host-side subscriber
    # re-emits each line through the same ``on_stage`` hook as a
    # ``tool_output`` stage event so transports can render a live scrolling
    # region. Without a bus (headless tests, MCP server) this is a no-op —
    # the tool's publish call simply finds no subscribers.
    _tool_output_bus = getattr(hooks.context_seams, "event_bus", None)
    if _tool_output_bus is not None and hasattr(_tool_output_bus, "subscribe"):

        async def _on_tool_output_event(payload: dict[str, Any]) -> None:
            await _emit_tool_output_fn(_state, _on_stage_hook, _log, payload)

        _tool_output_bus.subscribe("lisa.tool_output", _on_tool_output_event)

    async def _emit_tool_output(data: dict[str, Any]) -> None:
        """Direct (non-bus) forwarder, for tests and fork children."""
        await _emit_tool_output_fn(_state, _on_stage_hook, _log, data)

    # ── Fork hook wiring ──────────────────────────────────────────────
    # Build the ``fork_fn`` closure that the pack-level ``fork`` operation
    # calls via ``ctx.host.fork(spec)``. Built after ``_emit_stage`` is
    # defined (unlike ``_delegate_fn``) because the fan-out emits fork
    # stage events through it.
    if hooks.operations.fork_fn is not None:
        _fork_fn: Callable[..., Awaitable[ForkResult]] = hooks.operations.fork_fn
    else:

        async def _fork_fn(
            spec: ForkSpec,
            *,
            _hooks: TurnHooks = hooks,
            _pool: MCPToolPool = pool,
            _resolve_model: Callable[[str], Any] = resolve_model,
            _complete: Callable[..., Awaitable[Any]] = complete,
            _exec_op: Callable[..., Awaitable[OperationResult]] = exec_op,
            _budget: TurnBudget | None = turn_budget,
            _policy: ExecutionPolicy | None = execution_policy,
            _turn_start: float = turn_start_mono,
            _logr: logging.Logger | None = logger_override,
            _emit: Callable[[str, dict[str, Any]], Awaitable[None]] | None = _emit_stage,
            _max_iters: int = max_tool_iterations,
        ) -> ForkResult:
            return await _default_fork_fn(
                spec,
                hooks=_hooks,
                pool=_pool,
                resolve_model=_resolve_model,
                complete=_complete,
                exec_op=_exec_op,
                turn_budget=_budget,
                execution_policy=_policy,
                turn_start_mono=_turn_start,
                logger_override=_logr,
                emit_stage=_emit,
                max_tool_iterations_override=_max_iters,
            )

    async def _mcp_tool_executor(tool_id: str, _ctx: Any) -> ToolResult:
        return await _mcp_tool_executor_fn(tool_id, _ctx, pool)

    # ── Phase 1: Pre-turn (entry operations and plan execution) ───────
    def _build_entry_ctx(**overrides: Any) -> Any:
        base_ctx_kwargs = dict(
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
            state_provider=hooks.services.state_provider,
            pack_configuration_resolver=hooks.services.pack_configuration_resolver,
            approval_service=hooks.services.approval_service,
            code_change_artifact_store=hooks.services.code_change_artifact_store,
            gateway_provider_fn=lambda: hooks.context_seams.provider_gateways,
            workspace_id=hooks.identity.workspace_id,
            user_id=hooks.identity.user_id,
            user_slug=hooks.identity.user_slug,
            user_name=hooks.identity.user_name,
            principal_key=hooks.identity.principal_key,
            conversation_id=hooks.identity.conversation_id,
            surface_type=hooks.identity.surface_type,
            surface_id=hooks.identity.surface_id,
            memory_scope=hooks.identity.memory_scope,
            session_plan_digest=hooks.identity.session_plan_digest,
            data_dir=hooks.services.data_dir,
            delegate_fn=_delegate_fn,
            fork_fn=_fork_fn,
            model_router=hooks.context_seams.model_router,
            event_bus=_tool_output_bus,
        )
        base_ctx_kwargs.update(overrides)
        return _build_minimal_context(**base_ctx_kwargs)

    _entry_result = await execute_entry_phase(
        mode_id=mode_id,
        query=query,
        mode_config=mode_config,
        hooks=hooks,
        ledger=ledger,
        execution_policy=execution_policy,
        routing_ctx=_routing_ctx,
        preferred_model_id=preferred_model_id,
        assemble_user_fn=assemble_user,
        build_minimal_context_fn=_build_entry_ctx,
        plan_steps_to_workflow_steps_fn=_plan_steps_to_workflow_steps,
        promises_routing_contract_fn=_promises_routing_contract,
        resume_in_flight_target_fn=_resume_in_flight_target,
        run_mode_entry_operations_fn=run_mode_entry_operations,
    )
    if _entry_result.early_return is not None:
        return _entry_result.early_return

    if _entry_result.pending_planned_handoff:
        _state.pending_planned_handoff = _entry_result.pending_planned_handoff
    if _entry_result.plan_prefix:
        _state.plan_prefix = _entry_result.plan_prefix
    if _entry_result.suppress_routing_hint:
        suppress_routing_hint = True
    if _entry_result.first_step_query:
        query = _entry_result.first_step_query
    _early_user_context = _entry_result.early_user_context

    # Chat/workflow modes: run pre-turn operations
    ctx = _build_entry_ctx(conversation_excerpt_markdown=_early_user_context)
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
        surface_type=cast(ConversationSurface, hooks.identity.surface_type),
        surface_id=str(hooks.identity.surface_id or ""),
        memory_scope=cast(MemoryScope, hooks.identity.memory_scope),
        thread_owner_id=str(hooks.identity.user_id or ""),
    )
    input_guard_result = await hooks.context_seams.input_guard.validate(query, guard_surface)
    if not input_guard_result.allowed:
        blocked_reason = input_guard_result.blocked_reason or "Input blocked by safety policy."
        _log.warning("Input guard blocked turn for mode=%s: %s", mode_id, blocked_reason)
        return TurnResult(
            response_text=blocked_reason,
            blocked=True,
            block_origin="input_guard",
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
        hooks.lifecycle.last_slice_finished_at = time.monotonic()

    # Assemble prompts — one mode model (Phase 6c). Every catalog mode
    # carries a stage graph (synthesized for stageless modes) and every
    # turn has a position, so instruction injection is uniform: the mode
    # body is the mode intro, and the current stage's body injects when
    # the stage carries one. Synthesized stages have no body, so
    # chat/action prompts are byte-identical to before. Workflow-mode
    # body invariants are enforced at catalog load (validate_workflow_mode),
    # so the reads here are lenient.
    mode_instruction = get_mode_system_intro(mode_id)
    stage_instruction: str | None = None
    if mode_config is not None:
        # Resolve the stage for the instruction body. An explicit child
        # ``stage_id_override`` (from a fork/delegate target) wins over the
        # live position's stage so a child can be pinned to a
        # specific stage. Without an override, read the position's current
        # stage — only when the position is parked in this turn's mode
        # (a stale or child-scoped hook must not inject another mode's
        # stage).
        _stage_for_instruction = stage_id_override or ""
        if not _stage_for_instruction and hooks.workflow.workflow_position is not None:
            _position_for_instruction = hooks.workflow.workflow_position()
            if (
                _position_for_instruction is not None
                and getattr(_position_for_instruction, "mode_id", "") == mode_id
            ):
                _stage_for_instruction = str(
                    getattr(_position_for_instruction, "stage_id", "") or ""
                )
        if _stage_for_instruction in mode_config.stages:
            _stage_body = mode_config.stages[_stage_for_instruction].instruction_body
            if _stage_body.strip():
                stage_instruction = _stage_body
    system_prompt = assemble_system(mode_instruction, stage_instruction)

    # Build the tool surface for this turn via the single shared assembler
    # (kernel.turn_phases._tool_schemas.build_tool_surface). The post-handoff
    # rebuild path calls the same function, so a mode's surface at turn start
    # is byte-identical to its surface after a mid-turn handoff — the
    # 2026-08-06 ``okf`` loop (rebuild dropped ``transition_stage``) and the
    # latent yield-drop (rebuild never advertised ``yield``) both came from
    # the two paths drifting apart.
    #
    # Done BEFORE assemble_user_context so the capability manifest reflects
    # exactly what the model will see this turn (closes the Tooling Paradox:
    # self-knowledge context now matches the injected function schema).
    #
    # ``yield`` advertisement follows the resolved policy: advertised whenever
    # the mode's ``yield_policy`` allows it (``allowed=True``, the default for
    # chat modes) OR when the mode names a ``fallback_mode``. When the model
    # calls ``yield``, the dispatch loop intercepts it (same pattern as
    # ``transition_stage``) — it does NOT dispatch to any executor; it breaks
    # the loop with the question as the turn content. The thread entry is
    # stamped ``yielded=True`` + ``yield_question`` for durable resume.
    #
    # Unified FSM routing: ``transition_stage`` is advertised when the
    # mode's stage graph has edges (intra-mode advancement) or the mode
    # has routing targets (inter-mode handoffs) — computed inside the
    # assembler. See docs/ROUTING_RFC_HANDOFF.md.
    has_routing_targets = bool(get_routing_targets(mode_id)) if mode_config else False
    _yield_policy_for_tools = _resolve_yield_policy(mode_config, _yield_policy_override)
    tool_schemas, tool_prune_stats = await build_tool_surface(
        mode_config,
        pool=pool,
        yield_policy=_yield_policy_for_tools,
    )
    operation_catalog = get_operation_catalog()
    tool_catalog = get_tool_catalog()
    _log.info(
        "tool schema mode=%s in=%d out=%d dup=%d disallowed=%d invalid=%d "
        "has_routing=%s names=[%s]",
        mode_id,
        tool_prune_stats["input_count"],
        tool_prune_stats["output_count"],
        tool_prune_stats["dropped_duplicate"],
        tool_prune_stats["dropped_disallowed"],
        tool_prune_stats["dropped_invalid"],
        has_routing_targets,
        ", ".join(_tool_schema_name(s) for s in tool_schemas),
    )

    # Compile a deterministic capability manifest from the final tool payload
    # so the model's self-knowledge context matches what it can actually call.
    capability_manifest = compile_capability_manifest(tool_schemas, mode_config=mode_config)

    # ── Durable position hydration ─────────────────────────────────────
    # When a ``WorkflowPosition`` is wired, hydrate ``chain_satisfied`` from
    # its ``completed_steps`` so a resumed turn does not re-suggest
    # already-completed chain steps. The position is the durable form of this
    # accumulator; ``chain_satisfied`` is its in-memory projection. This runs
    # BEFORE the routing-hint computation below so the hint sees the hydrated
    # gap (the remaining consumer intents, not the already-done producers).
    # This is the fix for the glm-5.2 self-assessment stall: after
    # ``proceed``, the position's completed research/write steps are
    # rehydrated and the router names the remaining consumer (coding) as the
    # gap instead of re-offering the first producer (search).
    if hooks.workflow.workflow_position is not None:
        try:
            _active_position = hooks.workflow.workflow_position()
        except Exception:  # pylint: disable=broad-except
            _active_position = None
        if _active_position is not None:
            for _step_id in getattr(_active_position, "completed_steps", ()):
                _step = (
                    _active_position.step_by_id(_step_id)
                    if hasattr(_active_position, "step_by_id")
                    else None
                )
                if _step is not None:
                    _state.chain_satisfied.update(_step.satisfies_intents)

    # Seed the typed prompt-segment registry from the stable harness +
    # session + customization + mode + stage base assembled above. Any
    # routing hint below lands as an ephemeral segment on top of it.
    _sys_segments = SystemPromptSegments.from_base(system_prompt)

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
    # Phase 7 (one control channel): the post-handoff chain-advance
    # directive rides the per-iteration lifecycle briefing instead of a
    # synthesized ``user`` message. Set when a handoff leaves a non-empty
    # routing hint (the chain is in flight); cleared when the chain
    # completes or the hint is suppressed. Persistent across iterations —
    # the briefing rebuild re-injects it every iteration until cleared.
    _chain_advance_guidance = ""
    # One-shot stalled-write rescue directive (the 2026-09-02
    # write-announcement loop): set when a natural completion announces a
    # write while the receipt ledger shows zero write attempts. Rides the
    # per-iteration lifecycle briefing for exactly one iteration; cleared
    # after the next model call regardless of outcome.
    _stalled_write_guidance = ""
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
    # The coordinator hint is also suppressed when ``suppress_routing_hint``
    # is set by the resume-in-flight gate: that gate already set
    # ``pending_planned_handoff`` to the next step's mode, so the
    # coordinator hint (which reads ``current_step_id`` — the step we are
    # parked on, not the next one) would compete with the synthesized
    # handoff and re-point at the current mode (the 2026-08-10 planning
    # loop: ``current_step_id=plan_1`` → coordinator hint says "go to
    # plan" → self-handoff → re-run ``plan_workflow`` → dead-end).
    coordinator_hint = (
        _coordinator_routing_hint(
            hooks=hooks,
            mode_id=mode_id,
            query=query,
        )
        if not suppress_routing_hint
        else ""
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
        # Dynamic segments (routing hint, per-iteration lifecycle briefing) are
        # registered alongside the stable base on ``_sys_segments``; the flattened
        # prompt remains byte-identical to the pre-segmenting single string.
        # The hint goes ONLY into the dynamic segment — not also into the stable
        # base string — so the flattened output carries it exactly once.
        system_prompt = system_prompt + "\n\n" + routing_hint
        _sys_segments.set_dynamic(N_ROUTING_HINT, routing_hint)
        # ── Forced-action enforcer: precompute the deterministic routing
        # decision (Phase 5: one enforcer). When the routing hint encodes
        # the routing — a ``fork:<csv>`` fan-out or any single-handoff
        # target (``plan``, ``search``, ``remember`` …) — the kernel has
        # already decided; the model is the executor. If it refuses to
        # call the expected tool — instead burning iterations on read-only
        # tools or returning empty completions (the 2026-07-27 fork
        # incident; the 2026-07-30 400k-token pre-plan burn; the
        # 2026-08-07 ``plan → remember → search`` stall) — the turn loop
        # forces the action itself after a small cap of non-compliant
        # dispatches rather than begging via nudges. Stored on ``_state``
        # and consumed in the tool-dispatch branch; re-computed after each
        # handoff by the chained-hint block. The fork list wins when both
        # shapes are present (it is the more specific decision).
        _fork_targets = _compute_fork_targets(
            query=query,
            routing_targets=get_routing_targets(mode_id) if mode_config else [],
            mode_config=mode_config,
        )
        _hint_target_id = _extract_hint_target(routing_hint)
        _transition_target = (
            _hint_target_id
            if _hint_target_id and not _hint_target_id.startswith("fork")
            else ""
        )
        _forced_targets = _fork_targets or ([_transition_target] if _transition_target else [])
        _state.forced_action = (
            ForcedAction(targets=list(_forced_targets)) if _forced_targets else None
        )
        _log.info(
            "forced action computed: kind=%s targets=%s",
            _state.forced_action.kind if _state.forced_action else None,
            _state.forced_action.targets if _state.forced_action else [],
        )
    else:
        _log.info(
            "no routing hint mode=%s (satisfies_intents=%s routing_targets=%s)",
            mode_id,
            getattr(mode_config, "satisfies_intents", None),
            get_routing_targets(mode_id) if mode_config else [],
        )

    # Reassemble user context with the full capability manifest (built
    # after tool surface assembly). The early pass used ``None`` so the
    # planner entry op could see the thread excerpt; this pass adds the
    # capability block for the model's self-knowledge context.
    user_context = assemble_user(mode_id, query, pre_briefing, capability_manifest)

    # ── Pre-turn memory recall ────────────────────────────────────────
    # Opportunistic: fires recall_memory before the model call so Lisa
    # surfaces relevant memories in the user message. Gated on
    # memory_recall_enabled and a minimum query length so trivial queries
    # ("ok", "thanks") don't waste a backend round-trip. A down memory
    # backend is logged and swallowed — it never blocks the turn.
    memory_snippet = ""
    if hooks.recall.enabled and query and len(query) >= hooks.recall.min_query_chars:
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
                gateway_provider_fn=lambda: hooks.context_seams.provider_gateways,
                state_provider=hooks.services.state_provider,
                pack_configuration_resolver=hooks.services.pack_configuration_resolver,
                approval_service=hooks.services.approval_service,
                code_change_artifact_store=hooks.services.code_change_artifact_store,
                workspace_id=hooks.identity.workspace_id,
                user_id=hooks.identity.user_id,
                user_slug=hooks.identity.user_slug,
                user_name=hooks.identity.user_name,
                principal_key=hooks.identity.principal_key,
                conversation_id=hooks.identity.conversation_id,
                surface_type=hooks.identity.surface_type,
                surface_id=hooks.identity.surface_id,
                memory_scope=hooks.identity.memory_scope,
                session_plan_digest=hooks.identity.session_plan_digest,
                data_dir=hooks.services.data_dir,
                delegate_fn=_delegate_fn,
                fork_fn=_fork_fn,
                model_router=hooks.context_seams.model_router,
            )
            recall_ctx.params = {
                "query": query,
                "limit": hooks.recall.limit,
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
    # The segment registry was seeded from the stable base (harness + session
    # + customizations + mode + stage) before the routing-hint block; the hint
    # and any per-iteration lifecycle briefing live as keyed dynamic segments on
    # it. ``messages[0].content`` is always produced by ``_sys_segments.flatten()``
    # so the rendered prompt stays byte-identical to the historical single
    # string while the stable prefix is never rewritten per-iteration.
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _sys_segments.flatten()}
    ]

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
    _log.debug(
        "capability gate passed: mode=%s model=%s",
        mode_id,
        getattr(model_def, "model_id", preferred_model_id),
    )

    content = ""
    transitioned_to: str | None = None
    handoff_to: str | None = None
    fork_results: list[ForkChildResult] | None = None
    workflow_terminated = False
    # ── Unified turn lifecycle ───────────────────────────────────────────
    # ``_yielded`` is set True when the loop breaks via a forced or
    # model-emitted ``YieldAction``. The thread entry records it for
    # durable resume via ``/resume``.
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
    # ``WorkflowPosition`` is wired these are unused — the global
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
        cc: int = 0,
    ) -> None:
        """Merge one completion's counters into the per-model accumulator.

        Also accumulates into ``per_mode_usage`` keyed by this turn's
        ``mode_id`` so the host can attribute tokens to the mode that
        actually ran the completion (not the entry mode)."""
        _record_model_usage_fn(
            _state, model_def, p=p, c=c, t=t, ct=ct, rt=rt, cc=cc, d=d, lat=lat, mode_id=mode_id
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
    # ``yielded=True on the thread entry``. This makes "ask"
    # distinguishable from "done" (the 2026-07-30 bug: both were "no tool
    # calls"). ``None`` means no forced yield is pending; the model works
    # normally.
    _pending_yield: YieldAction | None = None
    # Operator-exit discriminator (Phase: pause/stop/terminate). Set when the
    # resolver produced a ``PauseAction`` / ``StopAction`` / ``TerminateAction``
    # (operator signal OR a cap path that degrades to one). The transport
    # reads ``TurnResult.operator_exit`` to decide whether to return the
    # prompt. ``""`` means not an operator exit.
    _pending_operator_exit: Literal["", "paused", "stopped", "terminated"] = ""
    # Cap-pause reason: when the stuck-in-stage cap (NOT an operator signal)
    # produces a ``PauseAction``, its reason is saved here so the post-loop
    # code can surface a diagnostic instead of a silent empty response. An
    # operator-initiated pause leaves this empty — only the cap path sets it.
    _cap_pause_reason: str = ""

    while _state.iteration < max_tool_iterations:
        _state.iteration += 1
        _state.step_iteration += 1

        # ── Unified turn lifecycle: resolver-driven decisions ──────────
        # The pure ``resolve_lifecycle_action()`` resolver owns every
        # iteration-level decision: recovery stall, handoff-loop backstop,
        # forced-fork/plan enforcers, iteration cap, resource-budget
        # exhaustion, per-step budget, stuck-in-stage cap/threshold. The
        # loop builds the resolver inputs from ``_state`` / ``hooks``,
        # calls the resolver, and applies the returned action. This
        # replaces the ~10 scattered in-loop guards (Phases 1-4 of the
        # cutover) with one decision point.

        # Fetch recovery records once per iteration (the resolver needs them).
        _open_records: list[Any] = []
        if hooks.services.recovery_service is not None:
            try:
                _open_records = await hooks.services.recovery_service.list_open(
                    _build_recovery_identity(hooks)
                )
            except Exception:  # pylint: disable=broad-except
                _open_records = []

        # Build the turn-state view for the resolver.
        # ``turns_in_stage`` is the unified stall counter: the workflow
        # session's per-stage turn count for workflow modes, the per-step
        # iteration counter (reset on every ``transition_stage``) for
        # chat/action modes. One counter drives the resolver's single
        # stuck-in-stage guard pair regardless of mode kind.
        # Build the turn-state view for the resolver.
        # ``turns_in_stage`` is the unified stall counter (Phase 6c): the
        # position's durable per-stage turn count (previous turns) plus
        # this turn's per-step iteration counter. One counter across turns
        # and iterations drives the resolver's single stuck-in-stage guard
        # pair — no mode-kind gate.
        _turns_in_stage = _state.step_iteration
        if hooks.workflow.workflow_position is not None:
            _wf_position = hooks.workflow.workflow_position()
            if _wf_position is not None:
                _turns_in_stage = max(0, _wf_position.turn_count - 1) + _state.step_iteration
                await _emit_stage(
                    "workflow_stage",
                    {
                        "mode_id": mode_id,
                        "stage_id": _wf_position.stage_id,
                        "turn_count": _wf_position.turn_count,
                        "_state.iteration": _state.iteration,
                    },
                )

        _resolver_settings = _LRS(
            max_tool_iterations=max_tool_iterations,
            per_step_max_iterations=per_step_max_iterations,
            per_step_max_tool_calls=per_step_max_tool_calls,
            stage_nudge_threshold=_kernel_settings.stage_nudge_threshold,
            stage_nudge_cap=_kernel_settings.stage_nudge_cap,
            forced_action_non_compliant_cap=_kernel_settings.forced_action_non_compliant_cap,
            max_handoffs_per_turn=_max_handoffs_per_turn,
            write_rescue_mode=_kernel_settings.write_rescue_mode,
        )
        _forced = _state.forced_action
        # ── Operator signal poll (Phase: pause/stop/terminate) ──────────
        # Poll the host once per iteration at the top of the loop, before
        # the resolver. ``None`` (host did not wire it) is zero overhead.
        # The signal is terminal: the resolver produces the matching action
        # this iteration, the loop breaks before the model call. In-flight
        # tool dispatches from the prior iteration already completed.
        _operator_signal = ""
        if hooks.lifecycle.operator_signal is not None:
            try:
                _operator_signal = hooks.lifecycle.operator_signal() or ""
            except Exception:  # pylint: disable=broad-except
                _operator_signal = ""
        _resolver_state = _TSV(
            iteration=_state.iteration,
            step_iteration=_state.step_iteration,
            step_tool_calls=_state.step_tool_calls,
            total_tool_calls=_state.total_tool_calls,
            turns_in_stage=_turns_in_stage,
            budget_exhausted=_budget_forces_final() if _pending_yield is None else False,
            forced_action_target=(
                f"fork:{_forced.target_csv}" if _forced is not None and _forced.kind == "fork"
                else (_forced.target_csv if _forced is not None else "")
            ),
            forced_action_non_compliant_dispatches=(
                _forced.non_compliant_dispatches if _forced is not None else 0
            ),
            handoff_count=_state.handoff_count,
            operator_signal=_operator_signal,
            action_tool_ids=frozenset(_state.action_tool_ids),
            has_write_receipt=any(
                r.capability_id in _state.action_tool_ids
                or bare_tool_name(r.capability_id) in _state.action_tool_ids
                for r in _state.effect_receipts
                if r.effect in ("write", "external_write")
            ),
            has_read_receipt=any(
                r.effect in ("read", "execute") for r in _state.effect_receipts
            ),
            consecutive_tool_errors=_state.consecutive_tool_errors,
            last_error_tool=_state.last_error_tool,
        )
        _active_position_for_resolver: Any = None
        if hooks.workflow.workflow_position is not None:
            try:
                _active_position_for_resolver = hooks.workflow.workflow_position()
            except Exception:  # pylint: disable=broad-except
                _active_position_for_resolver = None

        _lifecycle_action = _resolve_lifecycle(
            state=_resolver_state,
            mode_id=mode_id,
            mode_yield_policy=_resolve_yield_policy(mode_config, _yield_policy_override),
            settings=_resolver_settings,
            router_decision=_state.pending_router_decision,
            workflow_position=_active_position_for_resolver,
            recovery_records=_open_records,
        )

        # ── Turn decision telemetry ──────────────────────────────────
        if hooks.context_seams.event_bus is not None:
            try:
                _action_type = type(_lifecycle_action).__name__
                _action_reason = getattr(_lifecycle_action, "reason", "") or getattr(
                    _lifecycle_action, "rationale", ""
                )
                await hooks.context_seams.event_bus.publish(
                    "turn.decision",
                    {
                        "mode_id": mode_id,
                        "iteration": _state.iteration,
                        "turns_in_stage": _turns_in_stage,
                        "action_type": _action_type,
                        "action_reason": _action_reason,
                        "total_tool_calls": _state.total_tool_calls,
                        "write_receipts": sum(
                            1
                            for r in _state.effect_receipts
                            if r.effect in ("write", "external_write")
                        ),
                        "read_receipts": sum(
                            1
                            for r in _state.effect_receipts
                            if r.effect in ("read", "execute")
                        ),
                    },
                )
            except Exception as _bus_exc:  # pylint: disable=broad-except
                _log.debug("turn.decision telemetry publish failed: %s", _bus_exc)

        # ── Apply the lifecycle action ───────────────────────────────
        # The resolver returns a tagged union; the loop applies the
        # side effects. This is the single application point that replaces
        # the ~10 scattered in-loop guards.

        if isinstance(_lifecycle_action, _FinishAction):
            if _lifecycle_action.strip_tools:
                effective_tools = None
            # If the finish is due to a handoff-loop backstop, set content
            # so the post-loop break path surfaces it.
            if "handoff" in _lifecycle_action.reason:
                content = (
                    f"Routing loop guard: reached {_max_handoffs_per_turn} "
                    f"mode handoffs in this turn without completing the chain. "
                    f"Forcing a final response."
                )
            break

        # ── Operator-exit actions (Phase: pause/stop/terminate) ──────────
        # Pause/stop/terminate are terminal: break before the model call.
        # The MCP pool is NOT cancelled — any in-flight tool dispatch from the
        # prior iteration already completed; the poll is at the top of the
        # loop, not mid-``await``.
        if isinstance(_lifecycle_action, _PauseAction):
            _pending_operator_exit = "paused"
            # Distinguish cap-pause from operator-signal pause: the cap's
            # reason starts with "stuck in" (set by the stuck-in-stage cap
            # in section 5 of the resolver), while an operator signal's
            # reason is "operator pause" (section 0). Save the cap reason
            # so the post-loop code can surface a diagnostic instead of a
            # silent empty response.
            if _lifecycle_action.reason != "operator pause":
                _cap_pause_reason = _lifecycle_action.reason
            _log.info(
                "turn lifecycle: pause (mode=%s, iteration=%d, reason=%s, cap=%s)",
                mode_id,
                _state.iteration,
                _lifecycle_action.reason,
                bool(_cap_pause_reason),
            )
            break

        if isinstance(_lifecycle_action, _StopAction):
            _pending_operator_exit = "stopped"
            _log.info(
                "turn lifecycle: operator stop (mode=%s, iteration=%d, reason=%s)",
                mode_id,
                _state.iteration,
                _lifecycle_action.reason,
            )
            break

        if isinstance(_lifecycle_action, _TerminateAction):
            _pending_operator_exit = "terminated"
            _log.info(
                "turn lifecycle: operator terminate (mode=%s, iteration=%d, reason=%s)",
                mode_id,
                _state.iteration,
                _lifecycle_action.reason,
            )
            break

        if isinstance(_lifecycle_action, YieldAction):
            _pending_yield = _lifecycle_action
            _log.debug(
                "turn lifecycle: resolver forced yield (mode=%s, iteration=%d, target=%s)",
                mode_id,
                _state.iteration,
                _lifecycle_action.target_mode_id or "(in-place)",
            )
            # Strip tools on the final completion so the model produces
            # the yield question text.
            effective_tools = None
            # Fall through to the model call — the yield is applied after
            # the model produces its completion (the existing break path).

        if isinstance(_lifecycle_action, _TransitionAction):
            if _lifecycle_action.forced:
                # Forced transition (per-step budget, forced-fork/plan
                # enforcer): set ``pending_planned_handoff`` so the
                # existing synthesis block below builds the tool call.
                _state.pending_planned_handoff = _lifecycle_action.target_mode_id
                # Reset per-step counters for the new step.
                _state.step_iteration = 0
                _state.step_tool_calls = 0
                _log.info(
                    "turn lifecycle: resolver forced transition target=%s (mode=%s, iteration=%d, reason=%s)",
                    _lifecycle_action.target_mode_id,
                    mode_id,
                    _state.iteration,
                    _lifecycle_action.rationale,
                )
                # Clear the forced-action enforcer state so it does not
                # re-fire after the forced transition is applied.
                _state.forced_action = None
            # Non-forced transitions (router transition_to) are handled
            # below in the router terminal-actions block (they need the
            # workflow session side effect).

        # ``ContinueAction.nudge`` is NOT appended to the messages here —
        # it rides the per-iteration lifecycle briefing below (Phase 7:
        # one control channel). The turn loop never synthesizes ``user``
        # messages; user messages come only from the host.

        # ── Midpoint write-nudge (sub-turn child OR parent turn) ───────
        # A turn whose mode declares action tools gets a ONE-SHOT nudge
        # when it has done only read-only calls so far, telling it to
        # commit to a write or yield. Two trigger shapes share one latch
        # (``write_nudge_fired`` on ``_TurnState``):
        #
        # - **Sub-turn child** (fork/delegate): fires at the iteration
        #   midpoint (``iteration > max_tool_iterations // 2``). Converts
        #   the silent read-loop → no_tools abort (2026-08-03 incident)
        #   into either a real write or an honest yield.
        # - **Parent turn**: fires ``write_deadline_margin`` iterations
        #   *before* the stuck-in-stage cap would pause the turn, using
        #   the unified ``turns_in_stage`` counter. This gives a
        #   capture mode (``researcher``/``reviewer``/``executor``) a
        #   final chance to call ``markdown_forge_document`` /
        #   ``okf_emit_concept`` rather than ending blocked as
        #   ``no_tools`` (hallucinated success) when the cap fires. The
        #   pre-cap margin is the dial — ``0`` disables the parent
        #   nudge; ``1`` fires one iteration before the cap.
        #
        # The nudge fires exactly once per turn (``write_nudge_fired``);
        # the sub-turn contract block in the briefing carries the same
        # signal every iteration, this is the escalated mid-run version.
        # Transport (Phase 7: one control channel): this iteration's
        # lifecycle briefing — never a synthesized ``user`` message.
        _midpoint_nudge_text = ""
        _parent_write_deadline = (
            not _state.sub_turn_kind
            and _kernel_settings.write_deadline_margin > 0
            and _turns_in_stage
            >= _kernel_settings.stage_nudge_cap - _kernel_settings.write_deadline_margin
        )
        if (
            _state.action_tool_ids
            and not _state.write_nudge_fired
            and not (_state.called_tool_names & _state.action_tool_ids)
            and not any(
                r.capability_id in _state.action_tool_ids
                for r in _state.effect_receipts
                if r.effect in ("write", "external_write")
            )
            # A turn with read/execute receipts did real read-only work;
            # a read-only turn in a write-capable mode does not need a
            # "commit or yield" nudge (the 2026-08-27 read-receipt fix —
            # the cloudservers read query was nudged toward the destructive
            # write tools it had no reason to call).
            and not any(r.effect in ("read", "execute") for r in _state.effect_receipts)
            and (
                _parent_write_deadline
                or (
                    _state.sub_turn_kind
                    and _state.iteration > max_tool_iterations // 2
                )
            )
            and _pending_yield is None
        ):
            _state.write_nudge_fired = True
            _action_list = ", ".join(sorted(_state.action_tool_ids))
            _log.info(
                "write-nudge (kind=%s mode=%s iteration=%d turns_in_stage=%d "
                "remaining=%d zero_writes); nudging to commit or yield",
                _state.sub_turn_kind or "parent",
                mode_id,
                _state.iteration,
                _turns_in_stage,
                max_tool_iterations - _state.iteration,
            )
            # The nudge prose is authored Markdown loaded through
            # ``kernel.context.sub_turn_contract`` (``prompts/
            # sub_turn/midpoint_write_nudge.md``); only the action-list
            # substitution glue lives here. The same prose applies to
            # parent turns and sub-turn children — both are "past the
            # deadline with zero writes."
            _midpoint_nudge_text = render_sub_turn_midpoint_nudge(
                action_list=_action_list,
            )

        # Determine effective_tools from the budget/yield state.
        if _pending_yield is not None:
            effective_tools = None
        elif not isinstance(_lifecycle_action, _FinishAction):
            effective_tools = tool_schemas if tool_schemas else None

        # Derive budget_exhausted for the budget-briefing block below.
        budget_exhausted = _pending_yield is not None

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
        #
        # ── Per-iteration guidance (Phase 7: one control channel) ─────
        # Soft nudges ride the lifecycle briefing (system prompt); the
        # turn loop never synthesizes ``user`` messages — user messages
        # come only from the host (the initial query). Sources: the
        # resolver's ``ContinueAction.nudge``, this iteration's sub-turn
        # midpoint write-nudge, and the persistent post-handoff
        # chain-advance directive (``_chain_advance_guidance``).
        _guidance_lines: list[str] = []
        if isinstance(_lifecycle_action, _ContinueAction) and _lifecycle_action.nudge:
            _guidance_lines.append(_lifecycle_action.nudge)
        if _midpoint_nudge_text:
            _guidance_lines.append(_midpoint_nudge_text)
        if _chain_advance_guidance:
            _guidance_lines.append(_chain_advance_guidance)
        if _stalled_write_guidance:
            _guidance_lines.append(_stalled_write_guidance)
            # One-shot: clear after the briefing carries it so the
            # directive cannot accumulate across iterations.
            _stalled_write_guidance = ""
        if _kernel_settings.budget_briefing_enabled:
            _briefing = _build_budget_briefing(
                state=_state,
                max_tool_iterations=max_tool_iterations,
                per_step_max_iterations=per_step_max_iterations,
                per_step_max_tool_calls=per_step_max_tool_calls,
                stage_nudge_cap=_kernel_settings.stage_nudge_cap,
                tool_schemas=tool_schemas,
                operation_catalog=operation_catalog,
                tool_catalog=tool_catalog,
                execution_policy=execution_policy,
                budget_exhausted=budget_exhausted,
                guidance=tuple(_guidance_lines),
            )
            # Replace the per-iteration lifecycle briefing segment. The stable
            # base is never rewritten; when the briefing is empty the segment is
            # cleared so the flattened prompt matches the pre-segment string.
            _sys_segments.set_dynamic(N_LIFECYCLE_BRIEFING, _briefing)
            messages[0]["content"] = _sys_segments.flatten()

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
        _log.debug(
            "resolver action: type=%s mode=%s iteration=%d",
            type(_lifecycle_action).__name__,
            mode_id,
            _state.iteration,
        )
        _router_decision = _state.pending_router_decision
        if _router_decision is not None:
            if _router_decision.finish.strip():
                _log.debug(
                    "router finish action (iteration=%d, reason=%s); forcing final",
                    _state.iteration,
                    _router_decision.finish,
                )
                effective_tools = None
            elif _router_decision.transition_to.strip():
                _target_stage = _router_decision.transition_to.strip()
                # The transition target is recorded on the TurnResult
                # (``transitioned_to``) and written to the thread; the
                # WorkflowView derives the stage from it on the next turn.
                # No mutable position is updated here.
                transitioned_to = _target_stage
                _log.debug(
                    "router transition_to action (iteration=%d, "
                    "stage=%s); forcing final summary",
                    _state.iteration,
                    _target_stage,
                )
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
            _log.debug(
                "kernel synthesizing planned transition_stage target=%s (mode=%s, iteration=%d)",
                _planned_handoff_target,
                mode_id,
                _state.iteration,
            )
            # Carry the active plan's first-step query as the
            # ``delegated_goal`` so the handoff context builder injects
            # it as ``## Your Current Task`` — the executor needs the
            # specific step it is running, not just the mode id. The
            # plan block itself is injected from the position's
            # ``plan_text`` in the handoff context builder.
            _synth_delegated_goal = ""
            try:
                _synth_pos = (
                    hooks.workflow.workflow_position()
                    if hooks.workflow.workflow_position
                    else None
                )
            except Exception:
                _synth_pos = None
            if _synth_pos is not None and getattr(_synth_pos, "current_step_id", ""):
                _synth_step = _synth_pos.step_by_id(_synth_pos.current_step_id)
                if _synth_step and _synth_step.step_query:
                    _synth_delegated_goal = _synth_step.step_query
            _synth_args: dict[str, Any] = {
                "target_stage_id": _planned_handoff_target,
                "rationale": (
                    "Kernel-driven handoff from the plan "
                    "result; the model is not asked to call "
                    "transition_stage."
                ),
            }
            if _synth_delegated_goal:
                _synth_args["delegated_goal"] = _synth_delegated_goal
            _synth_planned_call = {
                "id": f"planned_handoff_{_state.iteration}",
                "type": "function",
                "function": {
                    "name": _TRANSITION_TOOL_NAME,
                    "arguments": json.dumps(_synth_args),
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

            # ── Provider-gateway dispatch (single live completion path) ──
            # Every model call routes through ``ContextSeams.provider_gateways``.
            # A model call
            # with no gateway wired for the resolved provider fails with a
            # clear error (this is the error the ``provider_gateways``
            # docstring has always promised).
            _provider_gateways = hooks.context_seams.provider_gateways
            if _provider_gateways is None:
                raise NotImplementedError(
                    "No provider gateway wired: TurnHooks.context_seams.provider_gateways "
                    "is None. Wire a ProviderGatewayProvider (e.g. the SDK's, discovered "
                    "via lisa.extensions.provider_gateways) into ContextSeams."
                )
            _provider_name = str(getattr(effective_model_def, "provider", "") or "")
            _gateway = _provider_gateways.gateway_for(_provider_name)
            if _gateway is None:
                raise NotImplementedError(
                    f"No provider gateway registered for provider {_provider_name!r}. "
                    f"Known to the active gateway provider: model "
                    f"{getattr(effective_model_def, 'model_id', '')!r} cannot be completed."
                )
            completion_request = CompletionRequest(
                messages=messages,
                model_def=effective_model_def,
                max_tokens=4096,
                temperature=0.4,
                tools=effective_tools,
                # Carry the typed segment view alongside ``messages`` so the
                # provider gateway can express the stable-vs-dynamic boundary
                # on the wire (Anthropic ``cache_control`` on the stable
                # prefix). Gateways that do not consume segments fall back to
                # flattening the system message from ``messages``, which is
                # byte-identical to the segment-flattened text.
                system_prompt_segments=_sys_segments.as_segments_copy(),
            )

            _stream_gateway_fn = getattr(_gateway, "complete_stream", None)
            _complete_started = time.perf_counter()
            if hooks.models.complete_stream is not None and callable(_stream_gateway_fn):
                # ── Streaming path ──────────────────────────────────────
                # v1: stream tokens to the host. When the stream indicates
                # tool_calls are needed (finish_reason='tool_calls'), fall
                # back to the non-streaming complete() for that iteration to
                # get the structured tool-call payload.
                chunks: list[StreamedChunk] = []
                async for chunk in _stream_gateway_fn(completion_request):
                    chunks.append(chunk)
                _stream_ms = (time.perf_counter() - _complete_started) * 1000.0
                aggregated = _aggregate_streamed_chunks(chunks)
                _fallback_request = completion_request

                # Truncated-stream rescue retry: when tools were offered and
                # the stream stops by length, the generation was cut off
                # mid-generation — quite possibly mid tool-call payload (v1
                # streaming does not parse deltas, so a truncated payload is
                # indistinguishable from a narration-only completion and the
                # announced write is silently lost; the 2026-09-02
                # write-announcement loop: glm-5.2:cloud narrated "Writing
                # both files now" every turn while a multi-KB write_file
                # payload blew the 4096-token budget). Retry once with a
                # doubled budget (bounded by the model's context window)
                # before accepting the narration-only aggregate as final.
                if aggregated.finish_reason == "length" and effective_tools:
                    _base_max_tokens = int(completion_request.max_tokens or 0)
                    _retry_max_tokens = _base_max_tokens * 2
                    _cap_window = int(getattr(effective_model_def, "context_window", 0) or 0)
                    if _cap_window > 0:
                        _retry_max_tokens = min(_retry_max_tokens, _cap_window)
                    if _retry_max_tokens > _base_max_tokens:
                        _log.warning(
                            "stream truncated with tools offered; retrying once "
                            "with higher max_tokens (mode=%s, model=%s, from=%d, to=%d)",
                            mode_id,
                            getattr(effective_model_def, "model_id", ""),
                            _base_max_tokens,
                            _retry_max_tokens,
                        )
                        _fallback_request = CompletionRequest(
                            messages=completion_request.messages,
                            model_def=completion_request.model_def,
                            max_tokens=_retry_max_tokens,
                            temperature=completion_request.temperature,
                            tools=completion_request.tools,
                            response_format=completion_request.response_format,
                            system_prompt_segments=completion_request.system_prompt_segments,
                        )
                        _retry_started = time.perf_counter()
                        _retry_chunks: list[StreamedChunk] = []
                        async for chunk in _stream_gateway_fn(_fallback_request):
                            _retry_chunks.append(chunk)
                        _stream_ms += (time.perf_counter() - _retry_started) * 1000.0
                        aggregated = _aggregate_streamed_chunks(_retry_chunks)

                if aggregated.finish_reason == "tool_calls":
                    # Fall back to non-streaming for tool-call payload
                    _fallback_started = time.perf_counter()
                    completion = await _gateway.complete(_fallback_request)
                    if completion.duration_ms <= 0.0:
                        completion.duration_ms = (
                            time.perf_counter() - _fallback_started
                        ) * 1000.0
                else:
                    # Build a completion-like object from the aggregated stream
                    completion = StreamedCompletionAdapter(aggregated)
                    if not aggregated.usage:
                        # The stream carried no usage; stamp measured duration so
                        # duration_ms survives the flip for time-billed providers.
                        completion.duration_ms = _stream_ms
            else:
                completion = await _gateway.complete(completion_request)
                if completion.duration_ms <= 0.0:
                    completion.duration_ms = (time.perf_counter() - _complete_started) * 1000.0
        usage = _extract_completion_usage(completion)
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        completion_total_tokens = usage.total_tokens
        cached_tokens = usage.cached_tokens
        reasoning_tokens = usage.reasoning_tokens
        cache_creation_tokens = usage.cache_creation_tokens
        duration_ms = usage.duration_ms
        latency_ms = usage.latency_ms
        _state.total_prompt_tokens += prompt_tokens
        _state.total_completion_tokens += completion_tokens
        _state.total_tokens += completion_total_tokens
        _state.total_cached_tokens += cached_tokens
        _state.total_reasoning_tokens += reasoning_tokens
        _state.total_cache_creation_tokens += cache_creation_tokens
        _state.total_duration_ms += duration_ms
        _state.total_latency_ms += latency_ms
        _record_model_usage(
            effective_model_def,
            p=prompt_tokens,
            c=completion_tokens,
            t=completion_total_tokens,
            ct=cached_tokens,
            rt=reasoning_tokens,
            cc=cache_creation_tokens,
            d=duration_ms,
            lat=latency_ms,
        )
        # Durable per-call token telemetry. The host subscribes its
        # TokenUsageTelemetryHandler to ``lisa.token_usage`` so every model
        # call survives process restarts (the /tokens lifetime view and the
        # introspection 24h summary read the resulting JSONL). Best-effort:
        # a publish failure never breaks the model-call path.
        _publish_token_usage_fn(
            hooks.context_seams.event_bus,
            _build_token_usage_event_fn(
                provider=str(getattr(effective_model_def, "provider", "") or ""),
                model_id=str(getattr(effective_model_def, "model_id", "") or ""),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=completion_total_tokens,
                cached_tokens=cached_tokens,
                reasoning_tokens=reasoning_tokens,
                cache_creation_tokens=cache_creation_tokens,
                duration_ms=duration_ms,
                latency_ms=latency_ms,
                request_id=str(getattr(completion, "request_id", "") or ""),
                mode_id=mode_id,
                stage_id=str(getattr(_state, "current_stage_id", "") or ""),
                call_site="turn_loop",
                conversation_id=hooks.identity.conversation_id,
                user_slug=hooks.identity.user_slug,
                surface_id=str(hooks.identity.surface_id or ""),
            ),
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

        # Initialize before the tool-calls branch: the dispatch path
        # (real tool calls) reaches the synthesized-call required-params
        # guard below and reads ``_synthesized_forced``. When the model
        # returns its own calls, this stays ``False`` — no synth happened.
        # When the empty-completion path synthesizes a forced call, the
        # branch below flips it to ``True``. Without this default the
        # dispatch path raises ``UnboundLocalError`` on a real-tool-call
        # iteration (2026-08-27 planner transition turn).
        _synthesized_forced = False
        if not tool_calls:
            # ── Forced-action enforcer (empty-completion path) ───────
            # The model returned no tool calls while a deterministic
            # routing decision is pending. Synthesize the expected call
            # (``fork`` for a fan-out, ``transition_stage`` for a single
            # handoff) ourselves and dispatch it on this iteration instead
            # of accepting the empty turn (the 2026-07-27 fork incident;
            # the 2026-08-07 chained-pipeline stall). One synth site for
            # both shapes (Phase 5).
            if _state.forced_action:
                _fa = _state.forced_action
                _log.warning(
                    "model returned no tool calls with a forced %s "
                    "pending (mode=%s, iteration=%d); kernel synthesizing "
                    "targets=%s",
                    _fa.kind,
                    mode_id,
                    _state.iteration,
                    _fa.targets,
                )
                if _fa.kind == "fork":
                    _forced_arguments: dict[str, Any] = {
                        "targets": [
                            {"mode_id": mid, "query": query} for mid in _fa.targets
                        ],
                    }
                else:
                    _forced_arguments = {
                        "target_stage_id": _fa.targets[0],
                        "rationale": (
                            "Kernel-forced transition handoff: the "
                            "model returned no tool calls after the "
                            "routing hint named a target."
                        ),
                    }
                synth_call = {
                    "id": f"forced_{_fa.kind}_empty_{_state.iteration}",
                    "type": "function",
                    "function": {
                        "name": _fa.expected_tool,
                        "arguments": json.dumps(_forced_arguments),
                    },
                }
                # Append the synthesized tool call to the last assistant
                # message (which had no tool_calls) so the tool result
                # message we attach below stays well-formed.
                assistant_msg["tool_calls"] = [synth_call]
                tool_calls = [synth_call]
                _state.forced_action = None
                _synthesized_forced = True
            elif _state.forced_work_tool:
                # ── Do-the-work backstop (empty-completion path) ──────
                # A guidance-needing model returned no tool calls on a
                # plan-self-handoff iteration where the routing hint and
                # the plan-driven hint both correctly went silent. Without
                # this backstop the turn ends with the model's prose and
                # blocks as ``no_tools`` (hallucinated success). Synthesize
                # a call to the mode's action tool so the turn does real
                # work instead of claiming it. Clear immediately so a
                # subsequent empty completion is not re-synthesized into a
                # loop (the 2026-08-15 researcher no-verified-write block).
                _work_tool = _state.forced_work_tool
                _log.warning(
                    "model returned no tool calls with a do-the-work "
                    "backstop pending (mode=%s, iteration=%d); kernel "
                    "synthesizing tool=%s",
                    mode_id,
                    _state.iteration,
                    _work_tool,
                )
                _work_call = {
                    "id": f"forced_work_empty_{_state.iteration}",
                    "type": "function",
                    "function": {
                        "name": _work_tool,
                        "arguments": json.dumps({}),
                    },
                }
                assistant_msg["tool_calls"] = [_work_call]
                tool_calls = [_work_call]
                _state.forced_work_tool = ""
                _synthesized_forced = True

            # Use the canonical extractor so reasoning models (gpt-5.x,
            # glm-5.2, kimi-k2.6) that put the answer in ``reasoning_content``
            # while leaving ``content`` empty still produce visible output.
            # Reading ``completion.content`` raw was the 2026-07-30 zero-output
            # bug: the turn loop saw empty content, forced a final no-tools
            # completion that also read only ``.content``, and stayed empty.
            content = completion_content(completion).strip()

            if not _synthesized_forced:
                # ── Unified turn lifecycle: apply pending yield ──────────
                # A forced yield (stuck-in-stage cap, iteration cap, or
                # recovery stall) was set earlier in this iteration. The
                # model produced no tool calls (it either complied with the
                # stripped-tools force-final path, or it finished naturally
                # at the same iteration the cap fired). Instead of accepting
                # the model's text as the final answer, break the loop with
                # the yield's question as the turn content so the user can
                # redirect. The thread entry records
                # ``yielded=True`` on the thread entry.
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
                    _log.debug(
                        "turn lifecycle: applying forced yield (mode=%s, iteration=%d, target=%s)",
                        mode_id,
                        _state.iteration,
                        _pending_yield.target_mode_id or "(in-place)",
                    )
                    break

                # ── Write-contract empty-completion backstop ──────────────
                # A chat/action mode with a side-effecting completion
                # contract (``action_tool_ids`` non-empty) that ends with
                # ZERO tool calls and no write receipt returned prose
                # instead of doing its work. The post-loop ``_parent_no_tools``
                # check blocks it as hallucinated success — and the
                # chat-mode softening cannot rescue it because every
                # softening branch is gated on ``_did_work`` (the 2026-08-14
                # proceed-loop guard, intentionally strict on zero-tool
                # turns). Without this backstop the turn throws away the
                # model's prose and the user sees a generic "no verified
                # write" block instead of the answer they asked for.
                #
                # Reuse the existing ``forced_work_tool`` synthesizer: arm
                # it with the mode's first action tool (sorted) and
                # ``continue`` so the next iteration synthesizes a real
                # dispatch. The synthesizer at the top of the empty-
                # completion block consumes it and clears it (one-shot);
                # the dispatch path clears it on any real call (``:3987``),
                # so it cannot loop. The model's prose is discarded on the
                # synth iteration and replaced with the tool's result, after
                # which ``_did_work=True`` and the chat-mode softening can
                # accept the follow-up response.
                #
                # Gated narrowly to avoid masking genuine hallucination:
                #   - ``mode_config`` present (mode resolved)
                #   - ``action_tool_ids`` non-empty (the mode HAS a write
                #     contract — read-only chat modes legitimately complete
                #     with zero tools)
                #   - ``not has_write_receipt`` (the contract is unmet)
                #   - ``total_tool_calls == 0`` (zero tools this turn — a
                #     turn that already did read work is handled by the
                #     chat-mode softening downstream)
                #   - ``_pending_yield is None`` (checked above; a forced
                #     yield takes precedence)
                #   - ``_state.forced_work_tool`` empty (not already armed —
                #     the synth block is one-shot, re-arming would loop)
                #   - budget not exhausted (no point arming a synth the
                #     cap will override next iteration)
                _has_write_receipt = any(
                    r.capability_id in _state.action_tool_ids
                    or bare_tool_name(r.capability_id) in _state.action_tool_ids
                    for r in _state.effect_receipts
                    if r.effect in ("write", "external_write")
                )
                _has_read_receipt = any(
                    r.effect in ("read", "execute") for r in _state.effect_receipts
                )
                if (
                    mode_config is not None
                    and _state.action_tool_ids
                    and not _has_write_receipt
                    and not _has_read_receipt
                    and _state.total_tool_calls == 0
                    and not _state.forced_work_tool
                    and not _budget_forces_final()
                ):
                    _work_target = next(iter(sorted(_state.action_tool_ids)))
                    _state.forced_work_tool = _work_target
                    _log.info(
                        "write-contract backstop armed (mode=%s "
                        "iteration=%d tool=%s); model returned prose with "
                        "zero tool calls — synthesizing an action-tool "
                        "dispatch instead of accepting a hallucinated "
                        "success",
                        mode_id,
                        _state.iteration,
                        _work_target,
                    )
                    continue

                # ── Stalled-write rescue (read-work, no write attempt) ────
                # A write-contract mode did read-only work this turn but
                # ended with prose that either explicitly ANNOUNCES a write
                # that never happened (the 2026-09-02 write-announcement
                # loop), or stalled in a read loop across multiple iterations
                # without committing to a write (the 2026-09-04 read-loop burn).
                #
                # The empty-completion backstop above cannot fire (it
                # requires ``total_tool_calls == 0``) and the chat-mode
                # softening downstream accepts the turn. The rescue is a
                # one-shot Phase-7 guidance line (rides the next iteration's
                # lifecycle briefing): it names the mode's action tool and
                # directs immediate action. No kernel-synthesized dispatch
                # here — an MCP-native action tool requires parameters the
                # kernel cannot fabricate.
                #
                # Trigger is structured:
                # 1. Mode declares action tools and has 0 write receipts.
                # 2. Read receipts are present (real non-write work was done).
                # 3. 0 writes were attempted.
                # 4. Either prose announced a write (_STALLED_WRITE_PATTERNS)
                #    OR the mode has stalled in stage >= stage_nudge_threshold
                #    without producing dense content.
                _has_write_attempt = any(
                    bare_tool_name(n) in _state.action_tool_ids or n in _state.action_tool_ids
                    for n in _state.called_tool_names
                )
                _announcement_match = (
                    bool(_STALLED_WRITE_PATTERNS.search(content)) if content else False
                )
                _is_chat = mode_config.mode_kind == "chat" if mode_config else False
                _has_dense = (
                    len(content.strip()) >= _kernel_settings.chat_mode_dense_content_min_chars
                )
                _stalled_at_threshold = (
                    _turns_in_stage >= _kernel_settings.stage_nudge_threshold
                    and not (_is_chat and _has_dense)
                )

                if (
                    mode_config is not None
                    and _state.action_tool_ids
                    and not _has_write_receipt
                    and _has_read_receipt
                    and not _has_write_attempt
                    and not _state.stalled_write_rescue_fired
                    and (_announcement_match or _stalled_at_threshold)
                ):
                    _state.stalled_write_rescue_fired = True
                    _rescue_action_list = ", ".join(sorted(_state.action_tool_ids))
                    if _announcement_match:
                        _stalled_write_guidance = (
                            "STOP ANNOUNCING AND DISPATCH. Your previous "
                            "response described a write that never happened — "
                            "no write tool was called this turn. You have all "
                            "the information you need. Call one of "
                            f"[{_rescue_action_list}] NOW with complete "
                            "arguments in this iteration. Do NOT narrate the "
                            "write before making the tool call; narration "
                            "instead of the tool call wastes the user's time."
                        )
                    else:
                        _stalled_write_guidance = (
                            "COMMIT TO A WRITE OR CONCLUDE. You have gathered "
                            "evidence across read calls but have not called any "
                            f"action tool. Call one of [{_rescue_action_list}] "
                            "NOW with complete arguments in this iteration to land "
                            "your work, or synthesize your final findings."
                        )
                    _log.warning(
                        "stalled-write rescue armed (mode=%s iteration=%d "
                        "tools=%s announcement=%s): read-only work with zero "
                        "write attempts — one-shot dispatch directive injected "
                        "into the next iteration's briefing",
                        mode_id,
                        _state.iteration,
                        sorted(_state.action_tool_ids),
                        _announcement_match,
                    )
                    continue

                _log.debug(
                    "turn loop: natural completion (mode=%s, iteration=%d, no tool calls, no pending yield)",
                    mode_id,
                    _state.iteration,
                )
                if not content:
                    # Natural completion with zero tool calls AND empty
                    # content: the provider produced nothing usable. This
                    # break path bypasses the ``if not tools_dispatched``
                    # warning further down (that block only runs when
                    # tool_calls existed but none dispatched), so without
                    # this diagnostic the post-loop fallback yields the
                    # "empty response" question with NO logged cause. The
                    # 2026-09-02 incident: glm-5.3-flash:cloud via Ollama
                    # burned the whole completion budget on reasoning the
                    # gateway did not extract, produced ``content=""``, and
                    # every such turn exited silently here.
                    # ``finish_reason`` distinguishes truncation
                    # (``length``) from a clean-but-empty stop.
                    _log.warning(
                        "turn loop: natural completion returned empty content "
                        "(mode=%s, iteration=%d/%d, finish_reason=%s, "
                        "completion_tokens=%d, reasoning_tokens=%d) — "
                        "harness will yield for redirect; if "
                        "finish_reason=length, reasoning consumed the "
                        "completion budget before any visible text",
                        mode_id,
                        _state.iteration,
                        max_tool_iterations,
                        getattr(completion, "finish_reason", "unknown"),
                        _state.total_completion_tokens,
                        _state.total_reasoning_tokens,
                    )
                break

        # Dispatch tool calls — MCP tools via pool, operations in-process.
        tools_dispatched = False
        _log.debug(
            "tool dispatch: iteration=%d tool_count=%d names=%s",
            _state.iteration,
            len(tool_calls),
            [tc.get("function", {}).get("name", "?") for tc in tool_calls],
        )
        # The do-the-work backstop is a one-shot: if the model dispatched
        # any tool this iteration, it does not need the kernel to
        # synthesize one. Clear before the per-call loop so a real
        # dispatch never coexists with a pending synth (the 2026-08-15
        # self-handoff no-verified-write fix).
        if tool_calls:
            _state.forced_work_tool = ""
        for tool_call in tool_calls:
            tool_name, tool_arguments = openai_tool_call_to_mcp(tool_call)
            tool_call_id = str(tool_call.get("id") or "")

            # ── Synthesized-call required-params guard ─────────────────
            # A kernel-synthesized call (forced-action enforcer or
            # do-the-work backstop) with empty arguments cannot succeed
            # against an action tool that requires user-supplied params.
            # Dispatching it raises a pydantic ValidationError, feeds the
            # error back to the model, and the backstop can re-arm and
            # loop — burning iterations on a call the kernel has no
            # values to fill. This is the 2026-08-27 cloudservers misroute:
            # the write-contract backstop synthesized ``cs_apply_cloud_init``
            # with ``{}``, it failed 7-missing-required, re-armed, and
            # looped while the user had asked to *leave* the topic.
            #
            # Abort the turn with the model's prose instead of
            # dispatching a guaranteed failure. Routing-only syntheses
            # (``transition_stage``, ``fork``) take optional/derived args
            # and are never missing all required params, so they pass
            # through unchanged. A turn that routed into a write-contract
            # mode the user never requested should surface the model's
            # answer, not fire a destructive action with empty params.
            if _synthesized_forced and _synthesized_call_missing_all_required(
                tool_name,
                tool_arguments,
                operation_catalog=operation_catalog,
                tool_catalog=tool_catalog,
            ):
                _log.warning(
                    "synthesized call %s is missing all required "
                    "parameters; aborting dispatch to avoid a guaranteed "
                    "validation failure and backstop loop (mode=%s, "
                    "iteration=%d) — routing misfire, surfacing model prose",
                    tool_name,
                    mode_id,
                    _state.iteration,
                )
                _state.forced_action = None
                _state.forced_work_tool = ""
                if _state.total_tool_calls == 0:
                    # ── Routing-misfire prose surfacing ───────────────────
                    # The abort happened on the FIRST dispatch attempt of a
                    # turn that dispatched nothing real (the 2026-09-02
                    # executor misroute: a conversational question routed
                    # planner→thinker→reviewer→executor, the model answered
                    # in prose with zero tool calls, the write-contract
                    # backstop synthesized its first action tool with ``{}``,
                    # and this guard aborted). Without a marker here the
                    # post-loop ``_parent_no_tools`` check sees zero
                    # receipts + zero calls, classifies the turn as
                    # hallucinated success, and REPLACES the model's prose
                    # with the no-verified-write block — directly
                    # contradicting this guard's "surfacing model prose"
                    # contract. The prose may be a genuine answer (the
                    # routing put the user's question in front of a
                    # write-contract mode); it may also be a claim of work.
                    # Distinguish honestly: keep the prose and append the
                    # no-write warning footnote (same authored asset the
                    # chat-mode softening uses) instead of a hard block.
                    # Bounded to zero-dispatch turns: a turn that already
                    # dispatched real tools keeps the strict block (its
                    # prose rides on top of actual work, so a failed write
                    # attempt must still block — the 2026-08-14 rule).
                    _state.routing_misfire_prose = True
                break

            # ── Forced-action enforcer (dispatch path) ────────────────
            # When the kernel has computed a deterministic routing
            # decision (``_state.forced_action``) and the model refuses to
            # call the expected tool — instead burning iterations on
            # read-only tools (``fetch_page``, ``web_search``) or returning
            # empty completions — the kernel forces the action itself. The
            # routing hint already encodes the decision; the model is the
            # executor. When the executor won't execute a deterministic
            # next step, the kernel takes over rather than begging via
            # nudges (the 2026-07-27 fork incident: 5 ``fetch_page`` calls
            # then an empty completion; the 2026-07-30 400k-token pre-plan
            # tool-cycling burn; the 2026-08-07 ``plan → remember →
            # search`` stall where ``search`` never ran).
            #
            # Detection: a forced action is set AND the model called a
            # non-expected tool. Each such dispatch increments
            # ``non_compliant_dispatches``; at the cap, the kernel
            # synthesizes the expected call itself by overriding the
            # current tool call's name/arguments and letting the dispatch
            # below handle it. The expected tool's calls reset the state
            # and never trigger the enforcer. Mutates the current
            # ``tool_call`` dict IN PLACE — it is the same object
            # referenced by the assistant message already appended to
            # ``messages``, so overriding ``function.name`` /
            # ``function.arguments`` here keeps the assistant message and
            # the tool result message consistent (the 2026-07-27
            # follow-on: a fresh ``forced_fork_N`` id was rejected by the
            # provider with ``tool_call_id not found``).
            _forced = _state.forced_action
            if _forced is not None and tool_name != _forced.expected_tool:
                _forced.non_compliant_dispatches += 1
                
                # Trust-Aware Enforcer Cap
                reliability = float(getattr(_state.effective_model, "structured_reliability", 0.0) or 0.0)
                is_frontier = getattr(_state.effective_model, "cost_tier", "") == "frontier"
                is_reasoning = getattr(_state.effective_model, "reasoning_model", False)
                
                if is_frontier or is_reasoning or reliability >= 0.95:
                    effective_cap = float('inf')
                else:
                    effective_cap = _kernel_settings.forced_action_non_compliant_cap

                if _forced.non_compliant_dispatches >= effective_cap:
                    _log.warning(
                        "model refused to %s (mode=%s, iteration=%d, "
                        "non_compliant_dispatches=%d); kernel forcing %s "
                        "targets=%s",
                        _forced.expected_tool,
                        mode_id,
                        _state.iteration,
                        _forced.non_compliant_dispatches,
                        _forced.kind,
                        _forced.targets,
                    )
                    if _forced.kind == "fork":
                        tool_arguments = {
                            "targets": [
                                {"mode_id": mid, "query": query}
                                for mid in _forced.targets
                            ],
                        }
                    else:
                        tool_arguments = {
                            "target_stage_id": _forced.targets[0],
                            "rationale": (
                                "Kernel-forced transition handoff: the model "
                                "did not call transition_stage after the "
                                "routing hint named a target."
                            ),
                        }
                    # Normalize the override to the provider's tool_call
                    # shape. OpenAI wraps name/arguments inside a
                    # ``function`` sub-dict; Ollama and other flat-shape
                    # providers put them at the top level. ``openai_tool_call_to_mcp``
                    # already handles both on read; this override path must
                    # too, or it raises ``KeyError: 'function'`` for the
                    # flat shape (2026-08-25 glm-5.2:cloud crash).
                    if isinstance(tool_call.get("function"), dict):
                        tool_call["function"]["name"] = _forced.expected_tool
                        tool_call["function"]["arguments"] = json.dumps(tool_arguments)
                    else:
                        tool_call["name"] = _forced.expected_tool
                        tool_call["arguments"] = json.dumps(tool_arguments)
                    tool_name = _forced.expected_tool
                    # The enforcer fires once per decision; clear the state
                    # so a subsequent model-issued call is not re-enforced
                    # and so the post-handoff hint recomputation can set a
                    # fresh target if needed.
                    _state.forced_action = None
                    _forced = None
                    # Fall through: the dispatch below now sees the
                    # expected tool name and executes the routing action.

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
            # Operator-visible display label: render the tool's
            # ``display_template`` against the call params (display-only;
            # never enters model context). Empty when the tool has no
            # template — hosts fall back to their own arg preview.
            _display_label = ""
            _tool_streams_output = False
            if isinstance(tool_arguments, dict):
                _tool_definition = tool_catalog.get(tool_name)
                _display_template = getattr(_tool_definition, "display_template", "") or ""
                if _display_template:
                    _display_label = _resolve_tool_display_label(
                        _display_template,
                        tool_arguments,
                    )
                # Streamed-output posture: transports open a live output
                # region only for tools whose ``.tool.yml`` declares
                # ``stream_output`` — the flag travels on the dispatch event
                # so the region is armed before the first line arrives.
                _tool_streams_output = bool(
                    getattr(_tool_definition, "stream_output", False)
                )
            await _emit_stage(
                "tool_dispatch",
                {
                    "tool": tool_name,
                    "tool_kind": _tool_kind,
                    "arguments": tool_arguments if isinstance(tool_arguments, dict) else {},
                    "display": _display_label,
                    "stream_output": _tool_streams_output,
                    "_state.iteration": _state.iteration,
                    "tool_calls_so_far": _state.total_tool_calls + 1,
                    "max_tool_calls": max_tool_iterations,
                    "mode_id": mode_id,
                },
            )
            _tool_dispatch_started = time.monotonic()
            recovery_item_id = ""
            recovery_retry_policy = "safe"
            # Skip recovery records for first-class terminal/routing actions
            # (``transition`` and ``yield``). Their target kinds are not part
            # of ``RecoveryTargetKind`` (recovery.py), so constructing a
            # ``RecoveryRecord`` for them raises a pydantic ``literal_error``
            # and only logs a misleading "recovery begin failed" warning.
            if hooks.services.recovery_service is not None and _tool_kind not in ("transition", "yield"):
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
                    # ── Durable workflow-position correlation ──────────
                    # When a ``WorkflowPosition`` is active, stamp the
                    # recovery record with the current step id so the host
                    # can decide whether a stalled record blocks the
                    # position from advancing (``step_blocks_instance``).
                    _wi_step_id = ""
                    if hooks.workflow.workflow_position is not None:
                        try:
                            _position_for_record = hooks.workflow.workflow_position()
                        except Exception:  # pylint: disable=broad-except
                            _position_for_record = None
                        if _position_for_record is not None:
                            _wi_step_id = getattr(_position_for_record, "current_step_id", "") or ""
                    started_record = await hooks.services.recovery_service.begin(
                        RecoveryRecord(
                            identity=_build_recovery_identity(hooks),
                            mode_id=mode_id,
                            stage_id=_current_recovery_stage_id(
                                hooks=hooks,
                            ),
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
                # loop. The thread entry records
                # ``yielded=True`` on the thread entry with the question.
                if tool_name == _YIELD_TOOL_NAME:
                    _yield_question = str(tool_arguments.get("question") or "").strip()
                    _yield_context = str(tool_arguments.get("context") or "").strip()
                    _yield_target = str(tool_arguments.get("target_mode_id") or "").strip()
                    if not _yield_question:
                        # The model called yield with no question — degenerate.
                        # Use the policy's default question rather than hanging.
                        _yp = _resolve_yield_policy(mode_config, _yield_policy_override)
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
                    _log.debug(
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
                # the position, and notifies the host via
                # workflow_position_updater.
                #
                # Unified FSM routing (see docs/ROUTING_RFC_HANDOFF.md):
                # transition_stage handles two target kinds:
                #   1. mode_id (routable) — mid-turn mode switch (handoff)
                #   2. workflow stage — existing apply_stage_transition path
                # Parallel fan-out is owned by the ``fork`` operation
                # (dispatched as a tool below, not via transition_stage).
                if tool_name == _TRANSITION_TOOL_NAME:
                    target_stage = str(tool_arguments.get("target_stage_id") or "").strip()
                    rationale = str(tool_arguments.get("rationale") or "").strip()
                    delegated_goal = str(tool_arguments.get("delegated_goal") or "").strip()
                    context_payload = str(tool_arguments.get("context_payload") or "").strip()
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
                    if _state.entry_failed and target_stage == "coding":
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

                    # ── Unified target resolution (Phase 6c) ──────────
                    # One resolution order for every mode kind: the target
                    # resolves against the current mode's stage graph first
                    # (intra-mode stage advance, validated against the
                    # current stage's allowed_next_stages), then against
                    # routable mode ids (inter-mode handoff). Both advance
                    # the same WorkflowPosition. Stage ids shadow mode ids
                    # within their own mode (ponder's ``exit`` wins over a
                    # hypothetical ``exit`` mode).
                    if mode_config is not None and target_stage in mode_config.stages:
                        # Intra-mode stage transition. The transition target
                        # is recorded on the TurnResult (``transitioned_to``)
                        # and written to the thread; the WorkflowView derives
                        # the stage from it on the next turn. No mutable
                        # position is updated here.
                        transitioned_to = target_stage
                        tool_output = (
                            f"Transitioned to stage: {target_stage}."
                            + (f" Rationale: {rationale}" if rationale else "")
                        )
                        await _emit_stage(
                            "tool_dispatch_complete",
                            {
                                "tool": tool_name,
                                "tool_kind": "transition",
                                "_state.iteration": _state.iteration,
                                "tool_calls_so_far": _state.total_tool_calls,
                                "ok": transitioned_to is not None,
                                "transition": "workflow_stage",
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
                        if tool_call != tool_calls[-1]:
                            _log.warning(
                                "Model batched %d trailing tool call(s) after stage transition. "
                                "Breaking the dispatch loop to cleanly boot the new stage.",
                                len(tool_calls) - tool_calls.index(tool_call) - 1
                            )
                        break

                    # ── Case B: mode handoff (routing FSM) ────────────────
                    if is_routable_mode(target_stage):
                        # The forced-fork / forced-transition enforcer state
                        # and per-step counters are reset by ``apply_routing``
                        # below — one applier owns that bookkeeping for every
                        # handoff origin (Phase 3).
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
                            _log.debug(
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
                                # Swap system prompt: replace the entire
                                # stable base for the post-handoff mode. The
                                # routing hint for the new mode is re-added
                                # below as a dynamic segment.
                                target_mode_instruction = get_mode_system_intro(target_stage)
                                _new_base = assemble_system(target_mode_instruction, None)
                                _sys_segments.clear_dynamic(N_ROUTING_HINT)
                                _sys_segments.clear_dynamic(N_HANDOFF_CONTEXT)
                                if _new_base.strip():
                                    _sys_segments.replace_base(_new_base)
                                else:
                                    # Pre-segment behavior allowed an empty
                                    # target prompt (bare mode with no intro);
                                    # preserve it by clearing the stable segment
                                    # rather than raising mid-turn.
                                    _sys_segments.clear_base()
                                messages[0] = {
                                    "role": "system",
                                    "content": _new_base,
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
                                    
                                    # Use the payload parsed from transition_stage arguments
                                    dg = delegated_goal if 'delegated_goal' in locals() else ""
                                    cp = context_payload if 'context_payload' in locals() else ""

                                    # Inject the active plan's rendered block
                                    # so the receiving mode knows the plan it
                                    # is executing — not just its mode id and
                                    # chain position. Without this the executor
                                    # sees a bare ``step_query`` and has no
                                    # visibility into the plan that was
                                    # presented (the 2026-08-09 plan-amnesia
                                    # incident).
                                    _handoff_plan_text = ""
                                    try:
                                        _handoff_pos = (
                                            hooks.workflow.workflow_position()
                                            if hooks.workflow.workflow_position
                                            else None
                                        )
                                    except Exception:
                                        _handoff_pos = None
                                    if _handoff_pos is not None:
                                        _handoff_plan_text = (
                                            getattr(_handoff_pos, "plan_text", "") or ""
                                        ).strip()

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
                                    )
                                    if _handoff_plan_text:
                                        handoff_context += (
                                            f"\n## Plan You Are Executing\n"
                                            f"{_handoff_plan_text}\n"
                                        )
                                    if dg:
                                        handoff_context += f"\n## Your Current Task\n{dg}\n"
                                    if cp:
                                        handoff_context += f"\n## Provided Context\n{cp}\n"
                                    handoff_context += f"\n{next_directive}"
                                    # Route handoff context through segments
                                    # so the stable base stays untouched and
                                    # this per-turn content is replaceable.
                                    _sys_segments.set_dynamic(
                                        N_HANDOFF_CONTEXT, handoff_context
                                    )
                                    messages[0]["content"] = _sys_segments.flatten()
                                except Exception as ctx_exc:  # pylint: disable=broad-except
                                    _log.warning(
                                        "failed to build handoff context for mode=%s: %s",
                                        target_stage,
                                        ctx_exc,
                                    )
                                # Rebuild tool schemas for the new mode via
                                # the same assembler Phase 1 used. Resolve the
                                # yield policy exactly as Phase 1 does
                                # (override wins, else the target mode's
                                # authored policy) so the rebuilt surface
                                # advertises (or omits) ``yield`` identically
                                # to a turn that started in this mode —
                                # previously the rebuild never added
                                # ``yield``, so a mode handed off mid-turn
                                # silently lost the ability to ask.
                                tool_schemas = await _rebuild_tool_schemas_for_mode(
                                    target_stage,
                                    target_mode_config,
                                    pool=pool,
                                    yield_policy=_resolve_yield_policy(
                                        target_mode_config, _yield_policy_override
                                    ),
                                )
                                # Recompile capability manifest for new surface
                                capability_manifest = compile_capability_manifest(
                                    tool_schemas, mode_config=target_mode_config
                                )
                                # Phase 8: the completion contract follows
                                # the mode the turn is actually in —
                                # refresh the action-tool set so the
                                # parent-turn receipt check (Phase 3)
                                # evaluates the handoff target, not the
                                # entry mode.
                                try:
                                    _state.action_tool_ids = mode_completion_effects(
                                        target_stage,
                                        mcp_pool=pool,
                                    )
                                except Exception:  # pylint: disable=broad-except
                                    _state.action_tool_ids = set()
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
                                # ── Plan resume-awareness ──────────────
                                # When the handoff target is ``plan`` while a
                                # broader ``plan_workflow_op`` instance is
                                # still in flight, resume that instance's next
                                # step instead of blindly re-running
                                # ``plan_workflow`` against a continuation/
                                # status query. The planner has no awareness
                                # of the existing plan and fails with "No
                                # installed mode provides a continue/status
                                # capability" (the 2026-08-01
                                # plan→markdown→learn→re-plan failure). This
                                # mirrors the turn-start resume path
                                # (``_coordinator_routing_hint``) for the
                                # mid-turn handoff case.
                                _promises_routing = _promises_routing_contract(target_mode_config)
                                _plan_resume_target = (
                                    _active_position_resume_target(hooks)
                                    if _promises_routing
                                    else ""
                                )
                                if _plan_resume_target:
                                    # Resuming an active plan: do NOT run the
                                    # planner entry op (it would refuse the
                                    # continuation query). Drive the handoff
                                    # to the instance's next step directly.
                                    entry_output = (
                                        f"Resuming the active plan; next step "
                                        f"`{_plan_resume_target}`."
                                    )
                                    _state.pending_planned_handoff = _plan_resume_target
                                    _log.info(
                                        "plan mid-turn handoff resuming active "
                                        "instance; target=%s (skip re-planning)",
                                        _plan_resume_target,
                                    )
                                elif (
                                    target_mode_config.mode_kind == "action"
                                    and target_mode_config.entry_operation_ids
                                ):
                                    try:
                                        # When the handoff targets a step in
                                        # the active plan, use its derived
                                        # step_query instead of the original
                                        # user query.
                                        _step_query = ""
                                        try:
                                            _active_pos = (
                                                hooks.workflow.workflow_position()
                                                if hooks.workflow.workflow_position
                                                else None
                                            )
                                        except Exception:
                                            _active_pos = None

                                        if _active_pos and getattr(
                                            _active_pos, "current_step_id", ""
                                        ):
                                            _next_step = _active_pos.step_by_id(
                                                _active_pos.current_step_id
                                            )
                                            if _next_step and _next_step.mode_id == target_stage:
                                                _step_query = _next_step.step_query

                                        entry_ctx = ctx.model_copy(
                                            update={
                                                "mode_id": target_stage,
                                                "query": _step_query or ctx.query,
                                            }
                                        )
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
                                        # Data-driven: a non-empty
                                        # ``target_stage_id`` in the
                                        # entry-op result IS the routing
                                        # contract, so the kernel drives
                                        # the next handoff for any op that
                                        # returns one — no op-id list. Work
                                        # ops (``web_search``) return their
                                        # own output shape with no routing
                                        # target and are never misread as
                                        # failed plans (the 2026-07-30
                                        # ``search`` incident).
                                        _planned_next = str(
                                            (entry_result.output_json or {}).get(
                                                "target_stage_id", ""
                                            )
                                            or ""
                                        ).strip()
                                        if _planned_next:
                                            _state.pending_planned_handoff = _planned_next
                                            # The plan prose lives in the
                                            # thread's ``response_text``; the
                                            # WorkflowView derives declared/
                                            # completed steps from it on the
                                            # next turn. No mutable position
                                            # is persisted.
                                            if hooks.workflow.route_coordinator is not None:
                                                try:
                                                    hooks.workflow.route_coordinator.save(
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
                                                            "surface_id": hooks.identity.surface_id,
                                                            "conversation_id": hooks.identity.conversation_id,
                                                            "user_id": hooks.identity.user_id,
                                                            "session_plan_digest": hooks.identity.session_plan_digest,
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
                                            _log.debug(
                                                "kernel-driven next handoff "
                                                "from planner mode=%s target=%s "
                                                "(planned by entry op)",
                                                target_stage,
                                                _planned_next,
                                            )
                                        elif _promises_routing:
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
                                            # Only fire for modes that
                                            # declared a routing contract.
                                            # A work action mode (e.g.
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
                                            from kernel.routing_vocabulary import (
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
                                        # When the failed mode is a routing
                                        # entry op (e.g. ``plan_workflow``
                                        # timed out), mark the model
                                        # unreliable for that op so the
                                        # routing-hint generator stops
                                        # re-escalating to it on subsequent
                                        # turns. The 2026-07-31 incident:
                                        # ``plan_workflow`` timed out, this
                                        # branch set ``entry_failed`` but
                                        # did NOT mark the model unreliable,
                                        # so the hint generator re-escalated
                                        # to ``plan`` on the next turn and
                                        # timed out again identically. The
                                        # "no target" branch above already
                                        # marks unreliable; this mirrors it
                                        # for the exception path.
                                        if _promises_routing:
                                            from kernel.routing_vocabulary import (
                                                mark_plan_unreliable_for_model,
                                            )

                                            mark_plan_unreliable_for_model(
                                                str(preferred_model_id or "")
                                            )
                                            _log.warning(
                                                "plan entry operation in mid-turn "
                                                "handoff raised (mode=%s, exc=%s); "
                                                "marking model %s unreliable for "
                                                "planning and breaking",
                                                target_stage,
                                                type(entry_exc).__name__,
                                                preferred_model_id,
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
                                _departing_mode_id = mode_id
                                mode_id = target_stage
                                mode_config = target_mode_config
                                handoff_to = target_stage
                                # Apply the unified routing side effects
                                # (per-step counter reset, forced-enforcer
                                # clear, coordinator save) via the single
                                # applier — Phase 3. The chain_satisfied
                                # accumulation happens below at the hint
                                # recomputation site, which owns the
                                # catalog read for the target's intents.
                                apply_routing(
                                    target=target_stage,
                                    source="model_tool_call",
                                    mode_id=mode_id,
                                    state=_state,
                                    hooks=hooks,
                                    ctx=_routing_ctx,
                                    rationale=rationale,
                                    log=_log,
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
                                # ── Mid-turn pre-turn operations ────────────
                                # Run the new mode's ``pre_turn_operation_ids``
                                # exactly as the turn-start path does. Without
                                # this, lifecycle-bound prewarms (the bash
                                # pack's ``sandbox_facts`` session probe) never
                                # ran for a mode reached mid-turn, so the first
                                # effectful gate in that mode saw a MISSING
                                # containment fact and the host policy engine
                                # fail-closed to an approval prompt even though
                                # the command would have run sandboxed (the
                                # 2026-09-10 scratch-dir approval prompt).
                                # Non-approval failures stay isolated inside
                                # ``run_mode_pre_turn_operations``
                                # (``failure_policy="skip"``); an
                                # ``ApprovalRequired`` propagates to the turn
                                # wrapper's existing handler like at turn
                                # start. ``_build_entry_ctx`` reads the
                                # reassigned ``mode_id`` closure, so the ctx
                                # already carries the new mode.
                                if target_mode_config.pre_turn_operation_ids:
                                    try:
                                        pre_ctx = _build_entry_ctx(
                                            mode_id=target_stage,
                                        )
                                        await run_mode_pre_turn_operations(
                                            target_stage,
                                            pre_ctx,
                                            ledger=ledger,
                                            execution_policy=execution_policy,
                                        )
                                    except ApprovalRequired:
                                        raise
                                    except Exception as pre_exc:  # pylint: disable=broad-except
                                        _log.warning(
                                            "mid-turn pre_turn targets failed for "
                                            "mode '%s': %s",
                                            target_stage,
                                            pre_exc,
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
                                    #
                                    # **Work-gate**: only accumulate if the
                                    # departing mode produced at least one
                                    # relevant receipt (read/execute for
                                    # research/gather modes, write for
                                    # code/write modes). A mode that was
                                    # visited but did nothing — forced out
                                    # by the forced-action enforcer after
                                    # zero productive calls — must NOT mark
                                    # its intents as satisfied. Otherwise the
                                    # next mode inherits a false "research
                                    # already done" and never redirects back
                                    # (the 2026-09-04 incident: researcher
                                    # was forced to executor after 1 failed
                                    # fetch_page, but chain_satisfied already
                                    # contained {research,gather,learn}, so
                                    # executor saw no gap and crawled to the
                                    # cap).
                                    _departing_has_work = any(
                                        r.effect in ("read", "execute", "write", "external_write")
                                        for r in _state.effect_receipts
                                    )
                                    if _departing_has_work:
                                        _state.chain_satisfied.update(
                                            target_mode_config.satisfies_intents
                                        )
                                    # Per-step counters were already reset by
                                    # ``apply_routing`` above (before the
                                    # durable instance advancement so a
                                    # failure there does not leave a stale
                                    # counter).
                                    post_handoff_hint = _compute_capability_routing_hint(
                                        mode_id=target_stage,
                                        query=query,
                                        capability_manifest=capability_manifest,
                                        routing_targets=get_routing_targets(target_stage),
                                        mode_config=target_mode_config,
                                        extra_satisfied=_state.chain_satisfied,
                                        planner_model_id=str(preferred_model_id or ""),
                                    )
                                    # ── Declared next-step hint (from derived view) ──
                                    # The handoff is recorded on the TurnResult
                                    # (``handoff_to``) and written to the thread;
                                    # the WorkflowView derives progress from
                                    # receipts in later thread entries. Here we
                                    # read the turn-start view snapshot to emit a
                                    # directive hint naming the next step's mode
                                    # from the declared sequence, so the model
                                    # knows where to go next. No mutable
                                    # position is advanced — the view is
                                    # read-only and re-derived next turn.
                                    if hooks.workflow.workflow_position is not None:
                                        try:
                                            _position = hooks.workflow.workflow_position()
                                        except Exception:  # pylint: disable=broad-except
                                            _position = None
                                        if _position is not None and getattr(
                                            _position, "current_step_id", ""
                                        ):
                                            _next_step = _position.step_by_id(
                                                _position.current_step_id
                                            )
                                            if _next_step is not None:
                                                # Emit the plan-driven hint via
                                                # the shared self-handoff guard.
                                                # When the next step's mode is
                                                # the one the turn just handed
                                                # off to, the guard returns ""
                                                # (a self-handoff hint causes
                                                # the forced-action enforcer to
                                                # synthesize a no-op
                                                # transition_stage to the
                                                # current mode, looping until
                                                # the iteration cap — the
                                                # 2026-08-11 executor→executor
                                                # stuck loop).
                                                post_handoff_hint = emit_transition_hint(
                                                    target_mode_id=_next_step.mode_id,
                                                    current_mode_id=target_stage,
                                                    prose_template=(
                                                        "## Routing Hint\n\n"
                                                        "This is the next step "
                                                        "in the planned "
                                                        "workflow. Call "
                                                        "`transition_stage` "
                                                        "with target_stage_id "
                                                        "`{target}`."
                                                    ),
                                                )
                                                if post_handoff_hint:
                                                    _log.info(
                                                        "plan-driven next step: "
                                                        "position=%s step=%s "
                                                        "target=%s",
                                                        _position.position_id,
                                                        _next_step.step_id,
                                                        _next_step.mode_id,
                                                    )
                                                # ── Do-the-work backstop ──
                                                # The routing hint and the
                                                # plan-driven hint both
                                                # correctly go silent here,
                                                # leaving a guidance-needing
                                                # model with no directive.
                                                # When the effective model
                                                # declared ``needs_guidance``
                                                # and this mode has action
                                                # tools, arm a do-the-work
                                                # backstop: the empty-
                                                # completion enforcer will
                                                # synthesize a call to one
                                                # of the mode's action tools
                                                # instead of accepting prose
                                                # and ending blocked as
                                                # ``no_tools`` (the
                                                # 2026-08-15 researcher
                                                # no-verified-write block).
                                                # Cleared as soon as any
                                                # tool dispatches, so it
                                                # cannot loop.
                                                elif getattr(
                                                    _state.effective_model,
                                                    "needs_guidance",
                                                    False,
                                                ):
                                                    _work_tools = (
                                                        mode_completion_effects(
                                                            target_stage,
                                                            mcp_pool=pool,
                                                        )
                                                        if pool is not None
                                                        else _state.action_tool_ids
                                                    )
                                                    if _work_tools:
                                                        _state.forced_work_tool = next(
                                                            iter(sorted(_work_tools))
                                                        )
                                                        _log.info(
                                                            "self-handoff "
                                                            "do-the-work "
                                                            "backstop armed "
                                                            "(mode=%s "
                                                            "tool=%s)",
                                                            target_stage,
                                                            _state.forced_work_tool,
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
                                    # the rest of the turn.
                                    #
                                    # 2026-08-05: suppression alone was not
                                    # enough — with ``hint_override_limit=50``
                                    # the model kept cycling for 16
                                    # round-trips at ~48k prompt tokens
                                    # after suppression, burning 62% of the
                                    # session's tokens in markdown mode. Now
                                    # that the limit is bounded (5), also
                                    # arm a pending yield when the mode
                                    # allows it so the next iteration halts
                                    # with a redirect question instead of
                                    # burning to the 300-iteration ceiling.
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
                                        _chain_advance_guidance = ""
                                        # Arm a halt: the model has
                                        # demonstrated it will not follow
                                        # routing hints this turn. Yield on
                                        # the next iteration (when the mode
                                        # allows it) so the user can
                                        # redirect, instead of burning the
                                        # remaining iteration budget.
                                        if _pending_yield is None:
                                            _halt_yp = _resolve_yield_policy(
                                                mode_config, _yield_policy_override
                                            )
                                            if _halt_yp.allowed or _halt_yp.fallback_mode:
                                                _pending_yield = YieldAction(
                                                    question=_halt_yp.default_question or (
                                                        "I've overridden the suggested routing "
                                                        f"{_state.hint_override_count} times and "
                                                        "am not converging. How would you like "
                                                        "me to proceed?"
                                                    ),
                                                    context=(
                                                        f"hint override limit reached "
                                                        f"({_state.hint_override_count} "
                                                        f"overrides, mode={target_stage})"
                                                    ),
                                                    target_mode_id=_halt_yp.fallback_mode or "",
                                                )
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
                                        _chain_advance_guidance = ""
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
                                        _chain_advance_guidance = ""
                                    else:
                                        _log.debug(
                                            "post-handoff routing hint mode=%s hint_len=%d hint=%s",
                                            target_stage,
                                            len(post_handoff_hint),
                                            post_handoff_hint[:200],
                                        )
                                        # The lifecycle briefing is already a
                                        # keyed segment (``N_LIFECYCLE_BRIEFING``),
                                        # so it is replaced by the next iteration's
                                        # briefing automatically — no marker
                                        # splitting needed here. Just append the
                                        # post-handoff hint as a fresh dynamic
                                        # segment named after the new mode's
                                        # residual chain direction.
                                        _sys_segments.set_dynamic(
                                            N_ROUTING_HINT, post_handoff_hint
                                        )
                                        messages[0]["content"] = _sys_segments.flatten()
                                        routing_hint = post_handoff_hint
                                        # Phase 7 (one control channel):
                                        # the chain-advance directive rides
                                        # the per-iteration lifecycle
                                        # briefing until the chain
                                        # completes — replacing the old
                                        # synthesized ``user``-message
                                        # escalation. Cleared when the
                                        # chain completes or the hint is
                                        # suppressed.
                                        _chain_advance_guidance = (
                                            "WORKFLOW CHAIN IN FLIGHT: "
                                            "You are now in the `"
                                            + target_stage
                                            + "` mode — one step of a "
                                            "multi-step workflow. Complete "
                                            "this mode's work, then you "
                                            "MUST call `transition_stage` "
                                            "to continue to the next step. "
                                            "Do NOT end the turn without "
                                            "calling `transition_stage`. "
                                            + post_handoff_hint
                                        )
                                        # Refresh the forced action for the
                                        # new mode so the enforcer tracks
                                        # the post-handoff routing decision
                                        # (Phase 5: one enforcer; the fork
                                        # list wins when both shapes are
                                        # present).
                                        _ph_fork_targets = _compute_fork_targets(
                                            query=query,
                                            routing_targets=get_routing_targets(target_stage),
                                            mode_config=target_mode_config,
                                            extra_satisfied=_state.chain_satisfied,
                                        )
                                        _post_handoff_target = _extract_hint_target(
                                            post_handoff_hint
                                        )
                                        _ph_transition_target = (
                                            _post_handoff_target
                                            if _post_handoff_target
                                            and not _post_handoff_target.startswith("fork")
                                            else ""
                                        )
                                        _ph_targets = _ph_fork_targets or (
                                            [_ph_transition_target]
                                            if _ph_transition_target
                                            else []
                                        )
                                        _state.forced_action = (
                                            ForcedAction(targets=list(_ph_targets))
                                            if _ph_targets
                                            else None
                                        )
                                else:
                                    # The new mode satisfies all remaining
                                    # intents; clear the stale hint so the
                                    # guards don't fire on the old one.
                                    routing_hint = ""
                                    _chain_advance_guidance = ""
                                    _state.duplicate_hint_count = 0
                                    _state.last_hint_text = ""
                                    # No remaining gap → no forced action.
                                    _state.forced_action = None
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
                        # Phase 7 (one control channel): when a
                        # post-handoff hint exists (the chain is not yet
                        # complete), the chain-advance directive rides the
                        # per-iteration lifecycle briefing
                        # (``_chain_advance_guidance``, set above) instead
                        # of a synthesized ``user`` message — the briefing
                        # is rebuilt before the next model call, so the
                        # model sees the directive at the same point it
                        # used to see the escalation message.
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
                        if tool_call != tool_calls[-1]:
                            _log.warning(
                                "Model batched %d trailing tool call(s) after transition_stage. "
                                "Breaking the dispatch loop to cleanly boot the new mode.",
                                len(tool_calls) - tool_calls.index(tool_call) - 1
                            )
                        break
                    tool_output = (
                        f"Unknown transition target: '{target_stage}'. "
                        f"It is not a stage in mode '{mode_id}' or a routable mode."
                    )
                    await _emit_stage(
                        "tool_dispatch_complete",
                        {
                            "tool": tool_name,
                            "tool_kind": "transition",
                            "_state.iteration": _state.iteration,
                            "tool_calls_so_far": _state.total_tool_calls,
                            "ok": False,
                            "transition": "unknown",
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

                # Capability dispatch (operation, pack tool, or MCP tool)
                _outcome = await dispatch_capability(
                    tool_name=tool_name,
                    tool_arguments=tool_arguments if isinstance(tool_arguments, dict) else {},
                    tool_kind=_tool_kind,
                    ctx=ctx,
                    state=_state,
                    ledger=ledger,
                    pool=pool,
                    operation_catalog=operation_catalog,
                    tool_catalog=tool_catalog,
                    exec_op=exec_op,
                    hooks=hooks,
                    recovery_item_id=recovery_item_id,
                    recovery_retry_policy=recovery_retry_policy,
                    build_recovery_identity_fn=_build_recovery_identity,
                    log=_log,
                )
                tool_output = _outcome.tool_output

                await _emit_stage(
                    "tool_dispatch_complete",
                    {
                        "tool": tool_name,
                        "tool_kind": _tool_kind,
                        "_state.iteration": _state.iteration,
                        "tool_calls_so_far": _state.total_tool_calls,
                        "ok": _outcome.is_ok,
                        "duration_ms": _outcome.duration_ms,
                        "result_preview": str(tool_output)[:200],
                    },
                )

                if not tool_output.strip():
                    tool_output = "Done"

                # Track the latest tool result for the model router.
                _state.last_tool_result_text = str(tool_output)
                _state.last_operation_result = _outcome.operation_result
                _state.tool_results_received += 1

                if _outcome.delegate_yield is not None:
                    _pending_yield = _outcome.delegate_yield
                    content = _pending_yield.question
                    _yielded = True
                    _log.info(
                        "delegate child model_yield propagated to user "
                        "(mode=%s, iteration=%d)",
                        mode_id,
                        _state.iteration,
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "name": tool_name,
                            "content": f"Yielded to user: {_pending_yield.question}",
                        }
                    )
                    break
            except ApprovalRequired:
                raise

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
                and any(r.status != "ok" for r in _state.fork_results)
            )
        ):
            break

        if not tools_dispatched:
            # No tool calls were dispatched — model returned text. Use the
            # canonical extractor so reasoning models that put the answer in
            # ``reasoning_content`` (empty ``content``) still surface it.
            content = completion_content(completion).strip()
            if not content:
                # Empty final completion with zero tool calls: the provider
                # produced nothing usable (provider-empty response, all
                # tokens spent on unextracted reasoning, or a truncation that
                # dropped the visible text). Log the cause so remote boxes
                # can diagnose WHY without the local jsonl artifacts —
                # ``finish_reason`` distinguishes truncation (``length``)
                # from a clean-but-empty stop.
                _log.warning(
                    "turn loop: model returned empty final completion "
                    "(mode=%s, iteration=%d/%d, finish_reason=%s, "
                    "tool_calls_total=%d) — harness will yield for redirect, "
                    "not claim a budget hit",
                    mode_id,
                    _state.iteration,
                    max_tool_iterations,
                    getattr(completion, "finish_reason", "unknown"),
                    _state.total_tool_calls,
                )
            break

    if not content:
        # ── Cap-pause diagnostic ────────────────────────────────────────
        # The stuck-in-stage cap (NOT an operator signal) broke the loop
        # before the model call. Surface a diagnostic so the user knows why
        # the turn paused and can redirect. This is NOT an operator-initiated
        # pause — the cap fired because the model was stuck in a loop (often
        # retrying a failing tool call). The diagnostic lets the user
        # intervene with the right path or instruction.
        if _cap_pause_reason:
            # Build a richer diagnostic: include the last tool call and a
            # snippet of its result so the user can see *why* the model was
            # stuck (e.g. "markdown_forge_document kept returning truncated
            # documents" or "read_file failed: file not found"). The last
            # receipt's capability_id names the tool; ``last_tool_result_text``
            # carries its output (truncated for the diagnostic).
            _last_tool_name = ""
            if _state.effect_receipts:
                _last_tool_name = (
                    _state.effect_receipts[-1].capability_id
                    if hasattr(_state.effect_receipts[-1], "capability_id")
                    else ""
                )
            _last_tool_snippet = ""
            if _state.last_tool_result_text:
                _last_tool_snippet = _state.last_tool_result_text[:300]
                if len(_state.last_tool_result_text) > 300:
                    _last_tool_snippet += "…"
            _why_clause = ""
            if _last_tool_name:
                _why_clause = f" The last tool I called was `{_last_tool_name}`"
                # When the last dispatch was an in-process operation,
                # surface structured context from ``OperationResult.extra``
                # so the user sees *why* it kept failing (e.g. "truncated
                # after 3 continuation attempts, finish_reason=length")
                # rather than just the model-facing summary string.
                _op_ctx_parts: list[str] = []
                if _state.last_operation_result is not None:
                    _op_extra = getattr(_state.last_operation_result, "extra", {}) or {}
                    _cont = _op_extra.get("continuation") if isinstance(_op_extra, dict) else None
                    if isinstance(_cont, dict):
                        if _cont.get("was_truncated"):
                            _attempts = _cont.get("attempts", 0)
                            _finish = _cont.get("final_finish_reason", "unknown")
                            _op_ctx_parts.append(
                                f"truncated after {_attempts} continuation "
                                f"attempt(s) (finish_reason={_finish})"
                            )
                    _trunc = _op_extra.get("truncated") if isinstance(_op_extra, dict) else None
                    if _trunc and not _op_ctx_parts:
                        _op_ctx_parts.append("persisted truncated")
                    _lint = _op_extra.get("lint_warnings") if isinstance(_op_extra, dict) else None
                    if _lint:
                        _op_ctx_parts.append(f"{_lint} lint warning(s)")
                if _op_ctx_parts:
                    _why_clause += f" which returned: {'; '.join(_op_ctx_parts)}"
                elif _last_tool_snippet:
                    _why_clause += f" which returned: {_last_tool_snippet}"
                _why_clause += "."
            content = (
                f"I've hit my iteration budget in {mode_id} "
                f"({_cap_pause_reason}).{_why_clause} "
                f"I've paused so you can redirect me — reply with instructions "
                f"or a correction and I'll resume from here."
            )
            _log.info(
                "turn loop: surfacing cap-pause diagnostic (mode=%s, reason=%s, last_tool=%s)",
                mode_id,
                _cap_pause_reason,
                _last_tool_name,
            )
        # ── Operator exit with no content ──────────────────────────────
        # A PauseAction / StopAction / TerminateAction broke the loop before
        # the model call. The transport reads ``operator_exit`` to decide
        # prompt behavior; the operator exit is encoded in
        # ``terminated``. Do NOT synthesize a yield question — operator
        # exits do not surface a question (only model-emitted yields and
        # recovery stalls do). The cap-pause diagnostic above already set
        # ``content`` for the stuck-in-stage cap case, so skip the fallback
        # yield paths entirely when the cap diagnostic has already surfaced.
        elif _pending_operator_exit:
            content = ""
        elif _pending_yield is not None:
            content = _pending_yield.question
            _yielded = True
        elif _resolve_yield_policy(mode_config, _yield_policy_override).allowed:
            _yield_policy = _resolve_yield_policy(mode_config, _yield_policy_override)
            # Be honest about WHY the turn ended with no content. The budget
            # claim is only true when the iteration cap actually fired; an
            # empty final completion at iteration N of N (the 2026-08-30
            # remote-box relapse: kimi-k3:cloud returned empty visible
            # content on 1-2 model calls) must NOT be labeled "I've hit my
            # iteration budget" — that fabricated claim is persisted to the
            # thread, the model later reads it back (via the thread manifest
            # or the lisa-thread MCP server) as ground truth, and the
            # pattern re-fires every turn. An authored
            # ``yield_policy.default_question`` still wins in both branches.
            _budget_actually_exhausted = _state.iteration >= max_tool_iterations
            if _budget_actually_exhausted:
                _fallback_question = _yield_policy.default_question or (
                    "I've hit my iteration budget for this turn. "
                    "How would you like me to proceed?"
                )
                _fallback_context = (
                    f"iteration cap {max_tool_iterations} reached "
                    f"with no final content"
                )
            else:
                _fallback_question = _yield_policy.default_question or (
                    "The model returned an empty response before reaching "
                    "any budget limit. How would you like me to proceed?"
                )
                _fallback_context = (
                    f"empty final completion at iteration "
                    f"{_state.iteration} of {max_tool_iterations} — budget "
                    f"not exhausted; check the provider response "
                    f"(finish_reason/reasoning extraction)"
                )
            _pending_yield = YieldAction(
                question=_fallback_question,
                context=_fallback_context,
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
                "fork_yielded=%s transitioned_to=%s routing_hint=%s",
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
                bool(
                    _state.fork_results is not None
                    and any(r.status == "yielded" for r in _state.fork_results)
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
                    final_reasoning_tokens = final_usage.reasoning_tokens
                    _state.total_prompt_tokens += final_usage.prompt_tokens
                    _state.total_completion_tokens += final_usage.completion_tokens
                    _state.total_tokens += final_usage.total_tokens
                    _state.total_cache_creation_tokens += final_usage.cache_creation_tokens
                    _state.total_duration_ms += final_usage.duration_ms
                    _record_model_usage(
                        effective_model_def,
                        p=final_usage.prompt_tokens,
                        c=final_usage.completion_tokens,
                        t=final_usage.total_tokens,
                        ct=final_usage.cached_tokens,
                        rt=final_usage.reasoning_tokens,
                        cc=final_usage.cache_creation_tokens,
                        d=final_usage.duration_ms,
                        lat=final_usage.latency_ms,
                    )
                    _publish_token_usage_fn(
                        hooks.context_seams.event_bus,
                        _build_token_usage_event_fn(
                            provider=str(getattr(effective_model_def, "provider", "") or ""),
                            model_id=str(getattr(effective_model_def, "model_id", "") or ""),
                            prompt_tokens=final_usage.prompt_tokens,
                            completion_tokens=final_usage.completion_tokens,
                            total_tokens=final_usage.total_tokens,
                            cached_tokens=final_usage.cached_tokens,
                            reasoning_tokens=final_usage.reasoning_tokens,
                            cache_creation_tokens=final_usage.cache_creation_tokens,
                            duration_ms=final_usage.duration_ms,
                            latency_ms=final_usage.latency_ms,
                            request_id=str(getattr(final_completion, "request_id", "") or ""),
                            mode_id=mode_id,
                            call_site="forced_final",
                            conversation_id=hooks.identity.conversation_id,
                            user_slug=hooks.identity.user_slug,
                            surface_id=str(hooks.identity.surface_id or ""),
                        ),
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

    # Terminal-stage handling: if the turn ended parked on a terminal
    # stage, force one final no-tools completion using the
    # terminal stage's instruction so the model produces a closing summary.
    if transitioned_to is not None:
        terminal_position = hooks.workflow.workflow_position() if hooks.workflow.workflow_position else None
        if terminal_position is not None and is_terminal_stage(terminal_position):
            terminal_stage_config = mode_config.stages.get(terminal_position.stage_id)
            if terminal_stage_config and terminal_stage_config.instruction_body:
                # The closing directive rides the system prompt (Phase 7:
                # one control channel) — no synthesized ``user`` message.
                terminal_prompt = (
                    assemble_system(
                        mode_instruction,
                        terminal_stage_config.instruction_body,
                    )
                    + "\n\nProvide a closing summary for this workflow."
                )
                if terminal_prompt.strip():
                    _sys_segments.replace_base(terminal_prompt)
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
                        _state.total_prompt_tokens += terminal_usage.prompt_tokens
                        _state.total_completion_tokens += terminal_usage.completion_tokens
                        _state.total_tokens += terminal_usage.total_tokens
                        _state.total_cache_creation_tokens += terminal_usage.cache_creation_tokens
                        _state.total_duration_ms += terminal_usage.duration_ms
                        _record_model_usage(
                            effective_model_def,
                            p=terminal_usage.prompt_tokens,
                            c=terminal_usage.completion_tokens,
                            t=terminal_usage.total_tokens,
                            ct=terminal_usage.cached_tokens,
                            rt=terminal_usage.reasoning_tokens,
                            cc=terminal_usage.cache_creation_tokens,
                            d=terminal_usage.duration_ms,
                            lat=terminal_usage.latency_ms,
                        )
                        _publish_token_usage_fn(
                            hooks.context_seams.event_bus,
                            _build_token_usage_event_fn(
                                provider=str(getattr(effective_model_def, "provider", "") or ""),
                                model_id=str(getattr(effective_model_def, "model_id", "") or ""),
                                prompt_tokens=terminal_usage.prompt_tokens,
                                completion_tokens=terminal_usage.completion_tokens,
                                total_tokens=terminal_usage.total_tokens,
                                cached_tokens=terminal_usage.cached_tokens,
                                reasoning_tokens=terminal_usage.reasoning_tokens,
                                cache_creation_tokens=terminal_usage.cache_creation_tokens,
                                duration_ms=terminal_usage.duration_ms,
                                latency_ms=terminal_usage.latency_ms,
                                request_id=str(getattr(final_completion, "request_id", "") or ""),
                                mode_id=mode_id,
                                call_site="terminal_stage",
                                conversation_id=hooks.identity.conversation_id,
                                user_slug=hooks.identity.user_slug,
                                surface_id=str(hooks.identity.surface_id or ""),
                            ),
                        )
                except Exception as exc:  # pylint: disable=broad-except
                    _log.warning("terminal-stage summary failed: %s", exc)
            workflow_terminated = True

    # ── Phase 3: Post-turn ────────────────────────────────────────────
    # ── Parent-turn completion contract (Phase 8) ────────────────────
    contract_decision = resolve_completion_contract(
        mode_id=mode_id,
        mode_kind=mode_config.mode_kind if mode_config else "chat",
        content=content,
        action_tool_ids=_state.action_tool_ids,
        effect_receipts=_state.effect_receipts,
        called_tool_names=_state.called_tool_names,
        total_tool_calls=_state.total_tool_calls,
        sub_turn_kind=_state.sub_turn_kind,
        yielded=_yielded,
        routing_misfire_prose=_state.routing_misfire_prose,
        dense_content_min_chars=_kernel_settings.chat_mode_dense_content_min_chars,
        preexisting_block=bool(
            _state.entry_failed
            or _state.plan_failed
            or (
                _state.fork_results
                and any(r.status != "ok" for r in (_state.fork_results or []))
            )
        ),
    )

    turn_blocked = False
    _block_origin = ""

    if isinstance(contract_decision, PreexistingFailureBlock):
        turn_blocked = True
    elif isinstance(contract_decision, SurfaceProseWithFootnote):
        _log.warning(
            "parent turn mode=%s aborted a synthesized action-tool "
            "dispatch (missing required params) on a zero-dispatch "
            "turn; surfacing the model's prose with the no-write "
            "warning instead of blocking as no_tools",
            mode_id,
        )
        _action_names = ", ".join(
            f"`{t}`" for t in sorted(contract_decision.action_tool_ids)
        )
        _warning = render_no_write_warning(action_names=_action_names)
        if _warning:
            content = content.rstrip() + "\n\n" + _warning
    elif isinstance(contract_decision, ChatSoftAccept):
        _log.info(
            "parent turn mode=%s completed with no verified write, but "
            "is a chat mode that did read-only work and produced dense "
            "content (%d chars); accepting as successful.",
            mode_id,
            contract_decision.content_len,
        )
    elif isinstance(contract_decision, ChatReadFootnote):
        _log.warning(
            "parent turn mode=%s completed with no verified write for its "
            "mode's action tools %s; appending warning footnote",
            mode_id,
            sorted(contract_decision.action_tool_ids),
        )
        _action_names = ", ".join(
            f"`{t}`" for t in sorted(contract_decision.action_tool_ids)
        )
        _warning = render_no_write_warning(action_names=_action_names)
        if _warning:
            content = content.rstrip() + "\n\n" + _warning
    elif isinstance(contract_decision, BlockNoVerifiedWrite):
        _log.warning(
            "parent turn mode=%s completed with no verified write for its "
            "mode's action tools %s; blocking as no_tools (hallucinated success)",
            mode_id,
            sorted(contract_decision.action_tool_ids),
        )
        content = contract_decision.block_message
        turn_blocked = True
        _block_origin = "no_tools"
    elif isinstance(contract_decision, AcceptTurn):
        pass

    # Output guard runs before on_complete/store_mem so the persisted and
    # delivered text is the sanitized/safe-fallback version. Blocked
    # outputs emit a guardrail.audit event and skip the post-turn slice.
    output_guard_result = await hooks.context_seams.output_guard.validate(content, guard_surface)
    if not turn_blocked:
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
            _block_origin = "output_guard"
        elif output_guard_result.sanitized_text is not None:
            content = output_guard_result.sanitized_text

    await on_complete(mode_id, query, content, messages, list(_state.effect_receipts))

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
            hooks.lifecycle.last_slice_finished_at = time.monotonic()

    # ── Workflow progress rendering ───────────────────────────────
    # When a position is active and has remaining steps, append a short
    # progress summary so the user sees which step just finished and
    # what's next. Origin-blind: applies to ad-hoc, session_plan, and
    # plan_workflow_op positions alike — ``render_workflow_progress`` is
    # already origin-blind. Only appended when there is visible content
    # (don't append to an empty error response).
    if content and hooks.workflow.workflow_position is not None:
        try:
            _progress_position = hooks.workflow.workflow_position()
        except Exception:  # pylint: disable=broad-except
            _progress_position = None
        if _progress_position is not None and getattr(_progress_position, "declared_steps", ()):
            try:
                from kernel.workflow_view import (
                    is_complete as _position_is_complete,
                )
                from kernel.workflow_view import (
                    render_workflow_progress,
                )

                if not _position_is_complete(_progress_position):
                    _progress = render_workflow_progress(_progress_position)
                    if _progress:
                        content = f"{content}\n\n{_progress}"
            except Exception as exc:  # pylint: disable=broad-except
                _log.warning("render_workflow_progress failed: %s", exc)

    # ── Plan prefix (in-loop planner handoff) ─────────────────────────
    # The Phase-1 planner path stages the rendered plan block on
    # ``_state.plan_prefix`` instead of prepending it to a recursive child
    # turn's response (the recursion is deleted — the first step runs
    # in-loop via ``pending_planned_handoff``). Prepend it here so the user
    # sees the plan at the top, followed by the first step's output. Only
    # when there is content to prefix (a plan-failure recovery prompt or an
    # empty turn leaves the prefix unstaged).
    if _state.plan_prefix and content:
        content = f"{_state.plan_prefix}\n\n{content}"

    return TurnResult(
        response_text=content,
        tool_call_count=_state.total_tool_calls,
        called_tool_names=set(_state.called_tool_names),
        effect_receipts=list(_state.effect_receipts),
        iteration_count=_state.iteration,
        prompt_tokens=_state.total_prompt_tokens,
        completion_tokens=_state.total_completion_tokens,
        total_tokens=_state.total_tokens,
        cached_tokens=_state.total_cached_tokens,
        reasoning_tokens=_state.total_reasoning_tokens,
        cache_creation_tokens=_state.total_cache_creation_tokens,
        duration_ms=_state.total_duration_ms,
        latency_ms=_state.total_latency_ms,
        slice_outcomes=slice_outcomes,
        blocked=turn_blocked,
        block_origin=_block_origin,
        transitioned_to=transitioned_to,
        handoff_to=handoff_to,
        fork_results=_state.fork_results if _state.fork_results else fork_results,
        workflow_terminated=workflow_terminated,
        operator_exit=_pending_operator_exit,
        yielded=_pending_yield if _yielded else None,
        per_model_usage=list(_state.per_model_usage.values()),
        per_mode_usage=list(_state.per_mode_usage.values()),
    )

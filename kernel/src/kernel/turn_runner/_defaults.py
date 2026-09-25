"""Default hook implementations for the kernel turn runner.

Each ``_default_*`` is the no-op or minimal concrete fallback used when the
host leaves a ``TurnHooks`` extension point unset. ``_default_delegate_fn``
recurses into :func:`kernel.turn_runner.run_turn` — imported lazily inside
the function because the circular import ``defaults -> loop -> defaults``
must be deferred to call time.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable, Iterable
from typing import Any, cast

from kernel.config import settings as _kernel_settings
from kernel.execution_policy import ExecutionPolicy
from kernel.models import (
    CapabilityManifest,
    ConversationSurface,
    DelegationResult,
    DelegationSpec,
    DelegationStatus,
    ForkChildResult,
    ForkResult,
    ForkSpec,
    MemoryScope,
    OperationExecutionContext,
    OperationResult,
    TurnBudget,
)
from kernel.operations.registry import execute_operation, mode_completion_effects
from kernel.recovery import RecoveryIdentity
from kernel.slice import SliceOutcome, SlicePolicy
from kernel.tools.mcp_client import MCPToolPool
from kernel.turn_hooks import TurnHooks
from kernel.turn_phases._child_turn import classify_child_turn
from kernel.turn_phases._fork import _execute_fork_fanout
from kernel.turn_phases._routing_hint import emit_transition_hint
from kernel.turn_runner._helpers import _StubModelDef

logger = logging.getLogger(__name__)


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


async def _default_delegate_fn(
    spec: DelegationSpec,
    *,
    hooks: TurnHooks,
    turn_budget: TurnBudget | None,
    execution_policy: ExecutionPolicy | None,
    depth: int,
    logger_override: logging.Logger | None = None,
) -> DelegationResult:
    """Run one delegate sub-turn by recursing into ``run_turn``.

    The default implementation reuses the fork infrastructure (budget,
    status mapping, action-tool validation) without fanning out in
    parallel. The child runs as a ``sub_turn_kind="delegate"`` turn so
    the budget briefing emits the delegate-variant sub-turn contract
    (``status=no_tools`` framing, not the fork "your output is
    discarded" framing) and the midpoint write-nudge fires.

    Per spec §6.1, the ``status`` predicate reuses the fork-child
    action-tool-floor helper (``mode_completion_effects``) — a child that
    made 17 read-only calls but never invoked a side-effecting action tool
    is ``no_tools``, not ``ok``.

    Per spec §6.2, the child passes ``sub_turn_kind="delegate"`` so the
    seam already in place emits the delegate-variant briefing.

    Per spec §6.3, delegate children suppress routing hints (matching
    fork children) so a delegate child does not re-route and create a
    loop. The seam is independent of this decision; the briefing
    function handles either case.
    """

    _dlog = logger_override or logger
    # Bound recursion. The root parent is depth 0; a direct delegate
    # child is depth 1. When the cap is hit, return ``max_depth`` so the
    # parent can surface the limit honestly instead of recursing until
    # the stack overflows.
    if depth > _kernel_settings.delegate_max_depth:
        _dlog.warning(
            "delegate child %s exceeded max depth %d (depth=%d); returning max_depth status",
            spec.target_mode_id,
            _kernel_settings.delegate_max_depth,
            depth,
        )
        return DelegationResult(
            target_mode_id=spec.target_mode_id,
            status="max_depth",
            error_message=(
                f"delegate child {spec.target_mode_id} exceeded the "
                f"maximum recursion depth ({_kernel_settings.delegate_max_depth}); "
                "re-issue the request with a shallower delegation chain or "
                "raise DELEGATE_MAX_DEPTH."
            ),
            depth=depth,
        )

    started = time.monotonic()
    child_budget = turn_budget
    child_max_iters = (
        spec.max_tool_iterations or _kernel_settings.delegate_child_max_tool_iterations
    )
    try:
        # Lazy package-root import: the engine lives in ``kernel.turn_runner``
        # (this package's ``__init__``), and ``defaults ──► package ──► defaults``
        # would be a circular import at module load. Resolving at call time also
        # lets tests patching ``kernel.turn_runner.run_turn`` intercept here.
        import kernel.turn_runner as _engine_pkg

        child_result = await _engine_pkg.run_turn(
            mode_id=spec.target_mode_id,
            query=spec.query,
            hooks=hooks,
            max_tool_iterations=child_max_iters,
            turn_budget=child_budget,
            execution_policy=execution_policy,
            logger_override=_dlog,
            # Delegate children suppress routing hints (matching fork
            # children) so a child does not re-route and create a loop.
            suppress_routing_hint=True,
            # Mark this turn as a delegate child so the budget briefing
            # emits the delegate-variant sub-turn contract (the
            # ``status=no_tools`` framing, not the fork "your output is
            # discarded" framing) and the midpoint write-nudge fires.
            sub_turn_kind="delegate",
            # Thread the recursion depth so the child's own delegate
            # children are depth+1 (bounded by delegate_max_depth).
            delegate_depth=depth,
            # Apply the delegate target's stage and model-alias overrides
            # so a delegate child runs the requested stage and model rather
            # than the target mode's defaults. Previously both
            # ``DelegationSpec.stage_id`` and ``DelegationSpec.model_alias``
            # were parsed but never applied to the child turn.
            stage_id_override=spec.stage_id or None,
            preferred_model_id=spec.model_alias or None,
        )
        elapsed_ms = int((time.monotonic() - started) * 1000)

        # ── Shared child-turn classification ───────────────────────────
        # Delegate to ``classify_child_turn`` so the delegate path shares
        # one classifier with the fork path. This fixes the same audit
        # findings (3.1: ``blocked`` is checked first; 4.1:
        # ``mode_completion_effects`` replaces ``_fork_mode_action_tool_ids``;
        # 7.1: the duplicated classifier is removed).
        action_ids = mode_completion_effects(
            spec.target_mode_id,
            mcp_pool=hooks.tools.mcp_tool_pool,
        )
        outcome = classify_child_turn(
            child_result=child_result,
            mode_id=spec.target_mode_id,
            elapsed_ms=elapsed_ms,
            action_tool_ids=action_ids,
        )
        # ``ChildStatus`` and ``DelegationStatus`` share every status except
        # ``complete`` (child) vs ``ok`` (delegate); map ``complete`` to ``ok``.
        _delegate_status: DelegationStatus = (
            "ok" if outcome.status == "complete" else outcome.status
        )
        return DelegationResult(
            target_mode_id=spec.target_mode_id,
            status=_delegate_status,
            output_text=outcome.output_text,
            error_message=outcome.error_message,
            tool_call_count=outcome.tool_call_count,
            called_tool_names=list(outcome.called_tool_names),
            duration_ms=outcome.duration_ms,
            yield_action=outcome.yield_action,
            depth=depth + 1,
            per_model_usage=[dict(e) for e in outcome.per_model_usage],
            per_mode_usage=[dict(e) for e in outcome.per_mode_usage],
        )
    except TimeoutError:
        elapsed_ms = int((time.monotonic() - started) * 1000)
        return DelegationResult(
            target_mode_id=spec.target_mode_id,
            status="timeout",
            error_message="Delegate child turn timed out.",
            duration_ms=elapsed_ms,
            depth=depth + 1,
        )
    except Exception as exc:  # pylint: disable=broad-except
        elapsed_ms = int((time.monotonic() - started) * 1000)
        _dlog.warning(
            "delegate child %s failed: %s: %s",
            spec.target_mode_id,
            type(exc).__name__,
            exc,
        )
        return DelegationResult(
            target_mode_id=spec.target_mode_id,
            status="error",
            error_message=f"{type(exc).__name__}: {exc}",
            duration_ms=elapsed_ms,
            depth=depth + 1,
        )


_USAGE_AGG_FIELDS: tuple[str, ...] = (
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cached_tokens",
    "reasoning_tokens",
    "cache_creation_tokens",
    "duration_ms",
    "latency_ms",
)


def _aggregate_child_usage(
    sources: Iterable[list[dict[str, Any]]],
    *,
    key_field: str,
) -> list[dict[str, Any]]:
    """Merge per-child usage lists into one list keyed by ``key_field``.

    ``key_field`` is ``"model_id"`` for per-model usage, ``"mode_id"`` for
    per-mode. Entries with the same key across children (and across a
    child's own iterations) sum their counters into one row, mirroring the
    per-model accumulator on ``_TurnState``. Returns a list (not a dict) so
    it serializes cleanly into ``ForkResult.per_model_usage`` /
    ``ForkResult.per_mode_usage`` and from there into the fork operation's
    ``output_json``.
    """
    merged: dict[str, dict[str, Any]] = {}
    for source in sources:
        for entry in source or []:
            key = str(entry.get(key_field, "") or "-").strip() or "-"
            target = merged.setdefault(key, {key_field: key})
            for field_name in _USAGE_AGG_FIELDS:
                current = target.get(field_name, 0) or 0
                incoming = entry.get(field_name, 0) or 0
                if isinstance(incoming, float) or isinstance(current, float):
                    target[field_name] = float(current) + float(incoming)
                else:
                    target[field_name] = int(current) + int(incoming)
    return list(merged.values())


async def _default_fork_fn(
    spec: ForkSpec,
    *,
    hooks: TurnHooks,
    pool: MCPToolPool,
    resolve_model: Callable[[str], Any],
    complete: Callable[..., Awaitable[Any]],
    exec_op: Callable[..., Awaitable[OperationResult]],
    turn_budget: TurnBudget | None,
    execution_policy: ExecutionPolicy | None,
    turn_start_mono: float,
    logger_override: logging.Logger | None = None,
    emit_stage: Callable[[str, dict[str, Any]], Awaitable[None]] | None = None,
    max_tool_iterations_override: int = 0,
) -> ForkResult:
    """Run a parallel fork fan-out by wrapping ``_execute_fork_fanout``.

    The default implementation reuses the existing fork orchestrator from
    :mod:`kernel.turn_phases._fork`: budget division, ``asyncio.TaskGroup``
    fan-out, per-child ``run_turn`` recursion as ``sub_turn_kind="fork"``,
    and the shared ``classify_child_turn`` classifier. The pack-level
    ``fork`` operation calls this via ``ctx.host.fork(spec)``.

    Unlike :func:`_default_delegate_fn`, fork does not bound recursion
    depth — fork children do not themselves fork. The per-child
    iteration cap comes from ``spec.max_tool_iterations`` when non-zero,
    else falls back to ``fork_child_max_tool_iterations``. An explicit
    ``max_tool_iterations_override`` (threaded from the parent turn's
    own cap when wired) takes precedence over both so a fork child can
    never exceed the parent turn's iteration envelope.
    """
    _flog = logger_override or logger
    if not spec.targets:
        return ForkResult(children=[], succeeded=0)

    # Apply the per-child iteration cap. Precedence: explicit override
    # (parent turn cap) > spec > kernel default. A non-zero override
    # always wins so a fork child cannot exceed the parent's envelope.
    child_max_iters = (
        max_tool_iterations_override
        or spec.max_tool_iterations
        or _kernel_settings.fork_child_max_tool_iterations
    )

    try:
        children = await _execute_fork_fanout(
            fork_targets=spec.targets,
            hooks=hooks,
            pool=pool,
            resolve_model=resolve_model,
            complete=complete,
            exec_op=exec_op,
            turn_budget=turn_budget,
            execution_policy=execution_policy,
            turn_start_mono=turn_start_mono,
            logger_override=_flog,
            emit_stage=emit_stage,
            max_tool_iterations=child_max_iters,
        )
    except TimeoutError:
        # ``_execute_fork_fanout`` catches per-child timeouts and maps
        # them to ``ForkChildResult(status="timeout")``; a bare
        # ``TimeoutError`` here means the orchestrator itself was
        # cancelled. Surface every child as cancelled.
        children = [
            ForkChildResult(
                target=t,
                status="cancelled",
                error_message="Fork fan-out was cancelled.",
            )
            for t in spec.targets
        ]
    succeeded = sum(1 for r in children if r.status == "ok")
    # Aggregate each child's per-model / per-mode token usage so the parent
    # turn can merge them into its ``_state`` ledger. Without this, a
    # multi-model fork reports only the parent's resolved default model in
    # ``/tokens`` — the 2026-08-06 regression. Same-model entries across
    # children (and across a child's own iterations) merge into one row.
    return ForkResult(
        children=children,
        succeeded=succeeded,
        per_model_usage=_aggregate_child_usage(
            (r.per_model_usage for r in children), key_field="model_id"
        ),
        per_mode_usage=_aggregate_child_usage(
            (r.per_mode_usage for r in children), key_field="mode_id"
        ),
    )


async def _default_on_turn_complete(
    _mode_id: str,
    _query: str,
    _response_text: str,
    _messages: list[dict[str, Any]],
    _effect_receipts: list[Any],
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
    host's affirmative-resume path (``host.handle_message``), which
    reads the same registry.
    """
    lowered = (query or "").strip().lower()
    if not lowered:
        return False
    from kernel.routing_vocabulary import get_resume_phrases

    return any(lowered.startswith(phrase) for phrase in get_resume_phrases())


def _coordinator_routing_hint(
    *,
    hooks: TurnHooks,
    mode_id: str,
    query: str,
) -> str:
    """Return a handoff hint from an active workflow position on a resume.

    Origin-blind: when a ``WorkflowPosition`` is active and has a current
    step, the position is the source of truth for the next target. The
    position's ``current_step_id`` was advanced by ``advance_step`` when
    the previous step completed, so it names the step that should run
    next on a ``/proceed`` resume.

    The legacy ``route_coordinator``-driven branch was deleted: the
    production host never wires ``route_coordinator`` (a legacy
    single-target persistence whose ``next_target`` pointed at the first
    step and went stale after advancement). The durable position is the
    sole resume source. Tests that exercised the coordinator branch are
    removed alongside it (hard cutover).
    """
    if not _looks_like_continuation_query(query):
        return ""
    # ── Position-driven resume (origin-blind) ─────────────────────────
    # Read the next target from the position's current step. Applies to
    # any active non-complete position (ad-hoc, session_plan, or
    # plan_workflow_op) — the chain's declared order is authoritative.
    # ``current_step_id`` is advanced by ``advance_step`` when a step
    # completes, so it names the step that should run next on a
    # ``/proceed`` resume. The Phase 1 resume-in-flight gate handles the
    # distinct parked-on-a-work-entry-op case (where ``current_step_id``
    # names a step whose entry op already ran this turn) by skipping that
    # entry op; this hint still names ``current_step_id``'s mode so the
    # model can run it when the step genuinely hasn't executed yet.
    if hooks.workflow.workflow_position is None:
        return ""
    try:
        _position = hooks.workflow.workflow_position()
    except Exception:  # pylint: disable=broad-except
        return ""
    if _position is None or not getattr(_position, "current_step_id", ""):
        return ""
    _step = (
        _position.step_by_id(_position.current_step_id)
        if hasattr(_position, "step_by_id")
        else None
    )
    if _step is None or not _step.mode_id:
        return ""
    # Guard against self-handoff via the shared helper contract: a
    # self-handoff hint causes the forced-action enforcer to synthesize a
    # no-op transition_stage to the current mode, looping until the
    # iteration cap (the 2026-08-11 executor→executor and
    # researcher→researcher stuck loops).
    return emit_transition_hint(
        target_mode_id=_step.mode_id,
        current_mode_id=mode_id,
        prose_template=(
            "## Routing Hint\n\n"
            "Continue with the next step in the workflow. "
            "Hand off to `{target}` by calling "
            "`transition_stage` with target_stage_id `{target}`."
        ),
    )


def _build_recovery_identity(hooks: TurnHooks) -> RecoveryIdentity:
    """Build the exact identity scope for recovery records."""

    return RecoveryIdentity(
        workspace_id=hooks.identity.workspace_id,
        user_id=hooks.identity.user_id,
        user_slug=hooks.identity.user_slug,
        surface_type=cast(ConversationSurface, hooks.identity.surface_type),
        surface_id=hooks.identity.surface_id,
        memory_scope=cast(MemoryScope, hooks.identity.memory_scope),
    )


def _current_recovery_stage_id(*, hooks: TurnHooks) -> str:
    """Return the position's current stage id when available, else empty."""

    if hooks.workflow.workflow_position is None:
        return ""
    try:
        position = hooks.workflow.workflow_position()
    except Exception:  # pylint: disable=broad-except
        return ""
    if position is None:
        return ""
    return str(getattr(position, "stage_id", "") or "")

"""Phase 1 entry-operation and plan execution for the turn runner.

This module encapsulates the pre-turn stage-entry operations, resume-in-flight
gate, and planner execution that occur before the tool loop starts.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from kernel.ipsar_fsm import run_mode_entry_operations
from kernel.models import CallLedger
from kernel.routing_vocabulary import mark_plan_unreliable_for_model
from kernel.turn_phases._routing import RoutingContext, save_route_decision

if TYPE_CHECKING:
    from kernel.execution_policy import ExecutionPolicy
    from kernel.turn_runner import TurnHooks, TurnResult

_log = logging.getLogger(__name__)

__all__ = [
    "EntryPhaseResult",
    "execute_entry_phase",
]


@dataclass(frozen=True, slots=True)
class EntryPhaseResult:
    """Outcome of Phase 1 entry-operation execution."""

    early_return: TurnResult | None = None
    """When non-None, the entry phase resolved the turn (work entry op or plan failure)."""
    pending_planned_handoff: str = ""
    """Planned handoff target mode id to synthesize in the first iteration."""
    plan_prefix: str = ""
    """Staged plan block to prepend to the turn's final content."""
    suppress_routing_hint: bool = False
    """Whether the planner/resume path suppresses emergent routing hints."""
    first_step_query: str = ""
    """Sub-query for the first planned step if derived from workflow steps."""
    early_user_context: str = ""
    """Pre-built conversation excerpt for entry ops."""


def _build_plan_failure_recovery(
    *,
    mode_id: str,
    plan_reason: str,
    partial_steps: list[Any],
) -> str:
    """Render a structured recovery message when plan_workflow returns no target."""
    partial_block = ""
    if partial_steps:
        partial_lines = ["The planner partially identified these steps before failing:"]
        for idx, s in enumerate(partial_steps, start=1):
            if isinstance(s, dict):
                partial_lines.append(
                    f"  {idx}. {s.get('mode_id', '?')} — {s.get('description', '')}"
                )
        partial_block = "\n\n" + "\n".join(partial_lines)

    return (
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


async def execute_entry_phase(
    *,
    mode_id: str,
    query: str,
    mode_config: Any,
    hooks: TurnHooks,
    ledger: CallLedger,
    execution_policy: ExecutionPolicy | None,
    routing_ctx: RoutingContext,
    preferred_model_id: str | None,
    assemble_user_fn: Callable[[str, str, str, Any], str],
    build_minimal_context_fn: Callable[..., Any],
    plan_steps_to_workflow_steps_fn: Callable[[list[Any]], list[Any]],
    promises_routing_contract_fn: Callable[[Any], bool],
    resume_in_flight_target_fn: Callable[..., str],
    run_mode_entry_operations_fn: Callable[..., Any] = run_mode_entry_operations,
) -> EntryPhaseResult:
    """Execute Phase 1 entry-operation logic.

    Returns an :class:`EntryPhaseResult` with either an ``early_return`` TurnResult
    (for work ops or plan failures) or the staged planned handoff state.
    """
    pre_briefing = ""
    has_entry_ops = mode_config is not None and bool(mode_config.entry_operation_ids)
    early_user_context = (
        assemble_user_fn(mode_id, query, pre_briefing, None) if has_entry_ops else ""
    )
    _log.debug(
        "early user context: built=%s has_entry_ops=%s mode=%s",
        bool(early_user_context),
        has_entry_ops,
        mode_id,
    )

    if mode_config is None or not mode_config.entry_operation_ids:
        return EntryPhaseResult(early_user_context=early_user_context)

    # ── Resume-in-flight gate (work + routing entry ops) ──────────
    resume_target = resume_in_flight_target_fn(hooks=hooks, mode_id=mode_id, query=query)
    if resume_target:
        _log.info(
            "resume-in-flight: skipping work entry op for mode=%s "
            "on continuation query; plan resumes to %s",
            mode_id,
            resume_target,
        )
        return EntryPhaseResult(
            pending_planned_handoff=resume_target,
            suppress_routing_hint=True,
            early_user_context=early_user_context,
        )

    ctx = build_minimal_context_fn(
        mode_id=mode_id,
        query=query,
        conversation_excerpt_markdown=early_user_context,
    )
    result = await run_mode_entry_operations_fn(
        mode_id,
        ctx,
        ledger=ledger,
        execution_policy=execution_policy,
    )

    planned_target = str((result.output_json or {}).get("target_stage_id", "") or "").strip()
    promises_routing = promises_routing_contract_fn(mode_config)

    if not planned_target and not promises_routing:
        # Work entry op: the result IS the turn output.
        from kernel.turn_runner import TurnResult

        entry_text = (result.output_text or "").strip()
        # Detect the query-discard hazard: a work entry op replaces the
        # ENTIRE turn result, so any operator instruction in the query is
        # dropped before a model call. That contract is right when the
        # op's output genuinely IS the answer (web_search), but binding a
        # prewarm/status snapshot as a work entry op silently swallows
        # user commands (the 2026-09-06 shell/sandbox_facts and
        # 2026-09-23 finances/finances_status incidents). Surface a loud
        # diagnostic so pack-side misbindings are visible in the host log
        # instead of manifesting as "the model did nothing."
        _entry_ops = ", ".join(mode_config.entry_operation_ids)
        _log.warning(
            "entry-op turn short-circuit: mode=%s entry_ops=[%s] discarded "
            "operator query %r before any model call — work entry ops end "
            "the turn with their own output; prewarm/status ops must use "
            "pre_turn_operation_ids",
            mode_id,
            _entry_ops,
            (query or "")[:120],
        )
        return EntryPhaseResult(
            early_return=TurnResult(
                response_text=entry_text,
                tool_call_count=0,
                iteration_count=1,
            ),
            early_user_context=early_user_context,
        )

    plan_block = (result.output_text or "").strip()
    plan_steps_raw = (result.output_json or {}).get("steps") or []
    if not isinstance(plan_steps_raw, list):
        plan_steps_raw = []

    if planned_target:
        workflow_steps = plan_steps_to_workflow_steps_fn(plan_steps_raw)
        first_step_query = ""
        if workflow_steps and workflow_steps[0].mode_id == planned_target:
            sub_q = workflow_steps[0].step_query
            if sub_q:
                first_step_query = sub_q

        save_route_decision(
            hooks,
            routing_ctx,
            target_mode_id=planned_target,
            reason=(
                (result.output_json or {}).get("rationale") or f"planned handoff from {mode_id}"
            ),
        )
        _log.debug(
            "kernel-driven handoff from planner mode=%s target=%s "
            "(planned by entry op, not model transition_stage)",
            mode_id,
            planned_target,
        )
        staged_prefix = plan_block if (plan_block and plan_steps_raw) else ""
        return EntryPhaseResult(
            pending_planned_handoff=planned_target,
            plan_prefix=staged_prefix,
            suppress_routing_hint=True,
            first_step_query=first_step_query,
            early_user_context=early_user_context,
        )

    # ── Plan failure: hard stop with a structured recovery prompt ──
    plan_reason = (
        str((result.output_json or {}).get("rationale", "") or "").strip()
        or "the planner returned no routing target"
    )
    plan_model_id = str(preferred_model_id or "")
    mark_plan_unreliable_for_model(plan_model_id)
    _log.warning(
        "plan_workflow returned no target (mode=%s, model=%s, reason=%s); "
        "marking model unreliable for planning and terminating turn with "
        "recovery prompt",
        mode_id,
        plan_model_id,
        plan_reason,
    )
    raw_steps = (result.output_json or {}).get("steps") or []
    partial_steps = [s for s in raw_steps if isinstance(s, dict)]
    recovery = _build_plan_failure_recovery(
        mode_id=mode_id,
        plan_reason=plan_reason,
        partial_steps=partial_steps,
    )
    from kernel.turn_runner import TurnResult

    return EntryPhaseResult(
        early_return=TurnResult(
            response_text=recovery,
            tool_call_count=0,
            iteration_count=1,
        ),
        early_user_context=early_user_context,
    )

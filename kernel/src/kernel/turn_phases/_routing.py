"""Unified routing applier for the turn runner.

One place owns every durable side effect of a mode/stage advance, regardless
of who decided it:

- the model calling ``transition_stage`` (a mode handoff or an intra-mode
  stage transition),
- the planner's entry op returning a routing contract
  (``output_json["target_stage_id"]``),
- the forced-action enforcer synthesizing the call the model refused to make.

Before this module existed, the three origins each performed a different
subset of the bookkeeping (instance persistence, coordinator save,
per-step counter reset, chain_satisfied accumulation) — the 2026-08-01
plan→markdown→re-plan loop and the 2026-08-07 plan→remember→search stall
both trace to one origin missing a side effect another origin had.

The applier is deliberately small: it validates the target, persists the
``WorkflowPosition`` (creating one from a plan's steps when given), saves
the coordinator decision, advances the durable position, and resets the
per-step counters. It does NOT rebuild the tool surface or run entry ops —
those stay in the turn loop's existing handoff continuation, which every
origin now funnels through.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

from kernel.models import WorkflowTransitionRequest
from kernel.seams.protocols import CoordinatorRouteDecision

__all__ = [
    "RoutingSource",
    "TurnDecision",
    "TurnDecisionResult",
    "apply_routing",
    "save_route_decision",
]

logger = logging.getLogger(__name__)


class TurnDecision(Enum):
    """The three possible outcomes of an execution loop evaluation."""

    CONTINUE = "continue"
    HANDOFF = "handoff"
    FINISH = "finish"


@dataclass(frozen=True, slots=True)
class TurnDecisionResult:
    """The result of the centralized loop decision."""

    decision: TurnDecision
    target: str | None = None
    reason: str = ""


RoutingSource = Literal["model_tool_call", "entry_op_contract", "forced_enforcer"]
"""Who decided the routing. Diagnostics only — behavior is identical."""


@dataclass(frozen=True, slots=True)
class RoutingContext:
    """The host/session identity a route decision is saved under.

    Built once per turn from ``hooks``; the applier never reads hooks
    directly for identity fields.
    """

    query: str
    surface_id: str
    conversation_id: str
    user_id: str
    session_plan_digest: str


def save_route_decision(
    hooks: Any,
    ctx: RoutingContext,
    *,
    target_mode_id: str,
    reason: str,
) -> None:
    """Persist the coordinator route decision for ``/proceed`` resumption.

    Failure-tolerant by design: a coordinator save failure must never abort
    the handoff (the durable ``WorkflowInstance`` is the authoritative chain
    state; the coordinator is the resume hint).
    """
    if hooks.workflow.route_coordinator is None:
        return
    try:
        hooks.workflow.route_coordinator.save(
            CoordinatorRouteDecision(next_target=target_mode_id, reason=reason),
            {
                "mode_id": target_mode_id,
                "query": ctx.query,
                "surface_id": ctx.surface_id,
                "conversation_id": ctx.conversation_id,
                "user_id": ctx.user_id,
                "session_plan_digest": ctx.session_plan_digest,
            },
        )
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning(
            "route_coordinator.save failed for target %s: %s",
            target_mode_id,
            exc,
            exc_info=True,
        )


def apply_routing(
    *,
    target: str,
    source: RoutingSource,
    mode_id: str,
    state: Any,
    hooks: Any,
    ctx: RoutingContext,
    rationale: str = "",
    log: logging.Logger | None = None,
) -> str:
    """Apply the shared side effects of a mode/stage advance. Returns the target.

    Every handoff origin calls this exactly once per advance. It owns:

    1. Resetting the per-step iteration/tool-call counters so the new step
       gets its own sub-budget (placed before any durable write so a failure
       there does not leave stale counters).
    2. Clearing the forced-action enforcer state so a completed handoff does
       not re-fire the enforcer on the next iteration.
    3. Accumulating the target mode's ``satisfies_intents`` into the turn's
       ``chain_satisfied`` (the caller passes the target config's intents via
       the turn loop, which owns the catalog read).
    4. Saving the coordinator route decision (``/proceed`` resumption).

    Instance advancement (``advance_step`` + ``enter_mode``) stays in the
    turn loop's handoff continuation because it needs the live position
    from ``hooks``; the applier keeps the counter/enforcer/coordinator
    bookkeeping that all three origins must share.
    """
    _log = log or logger
    # 1. Per-step counters: the new step gets a full sub-budget.
    state.step_iteration = 0
    state.step_tool_calls = 0
    # 2. Forced-action enforcer state: clear so a completed handoff does not
    # re-fire on the next iteration.
    state.forced_action = None
    # 3. Coordinator decision: persist for /proceed resumption.
    save_route_decision(
        hooks,
        ctx,
        target_mode_id=target,
        reason=rationale or f"{source} handoff from {mode_id}",
    )
    _log.debug(
        "routing applied: %s -> %s (source=%s)",
        mode_id,
        target,
        source,
    )
    return target


def build_transition_request(
    target: str,
    *,
    rationale: str = "",
    required_missing_inputs: list[str] | None = None,
) -> WorkflowTransitionRequest:
    """Build the ``WorkflowTransitionRequest`` for an intra-mode stage advance."""
    return WorkflowTransitionRequest(
        target_stage_id=target,
        rationale=rationale,
        required_missing_inputs=required_missing_inputs or [],
    )


# ── Centralized loop decision ────────────────────────────────────────


def evaluate_turn_decision(
    *,
    has_tool_calls: bool,
    has_transition_request: bool,
    transition_target: str | None,
    is_terminal: bool,
    budget_exhausted: bool,
    has_content: bool,
) -> TurnDecisionResult:
    """Classify one iteration's outcome into the three-outcome loop contract.

    This is the single decision point that reduces every model iteration
    to one of:

    - ``CONTINUE`` — the model called tools (but no transition) and the
      budget is not exhausted. The loop should iterate again.
    - ``HANDOFF`` — the model (or the enforcer) produced a
      ``transition_stage`` call. The loop should advance the mode/stage
      via ``apply_routing`` and then iterate again with the new target's
      tool surface.
    - ``FINISH`` — the model produced a final answer (content with no
      tool calls), the position is terminal, or the budget is exhausted
      with no pending transition. The loop should break and emit the
      response.

    The function is intentionally pure — it reads only the booleans the
    turn loop already computes and returns a ``TurnDecisionResult``. The
    caller (``run_turn``) performs the side effects. This keeps the
    decision testable in isolation and prevents policy from scattering
    across the loop body.

    Args:
        has_tool_calls: True when the model completion included at least
            one tool call that is not a ``transition_stage`` call.
        has_transition_request: True when the model (or the forced-action
            enforcer) produced a ``transition_stage`` tool call.
        transition_target: The target stage/mode id from the transition
            request, or ``None`` when there is no transition.
        is_terminal: True when the current workflow position is complete
            (all declared steps satisfied and the intra-mode pointer is
            on a terminal stage).
        budget_exhausted: True when the iteration or tool-call budget has
            been consumed.
        has_content: True when the model produced a non-empty content
            string (a visible answer for the user).

    Returns:
        A ``TurnDecisionResult`` whose ``decision`` is one of
        ``CONTINUE``, ``HANDOFF``, or ``FINISH``.
    """
    # 1. A transition request always wins — the model (or enforcer) is
    #    explicitly handing off to another mode/stage.
    if has_transition_request and transition_target:
        return TurnDecisionResult(
            decision=TurnDecision.HANDOFF,
            target=transition_target,
            reason="model requested transition_stage",
        )

    # 2. A terminal position means the workflow is done — finish
    #    regardless of remaining budget. This is the receipt-backed
    #    completion path: ``is_terminal`` is only True when the workflow
    #    view says all steps are satisfied.
    if is_terminal:
        return TurnDecisionResult(
            decision=TurnDecision.FINISH,
            reason="workflow position is terminal (all steps satisfied)",
        )

    # 3. Tool calls without a transition mean the model is still working
    #    — continue the loop unless the budget is gone.
    if has_tool_calls and not budget_exhausted:
        return TurnDecisionResult(
            decision=TurnDecision.CONTINUE,
            reason="model dispatched tools; iterating",
        )

    # 4. Budget exhausted with no transition and no terminal position:
    #    the harness must yield. This is the stall / cap path.
    if budget_exhausted:
        return TurnDecisionResult(
            decision=TurnDecision.FINISH,
            reason="budget exhausted with no pending transition",
        )

    # 5. No tool calls, no transition, budget remaining, and not
    #    terminal: the model produced a final answer (or an empty
    #    response). Either way, the turn is done.
    return TurnDecisionResult(
        decision=TurnDecision.FINISH,
        reason="model produced final answer"
        if has_content
        else "model produced no tool calls and no content",
    )

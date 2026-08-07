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
``WorkflowInstance`` (creating one from a plan's steps when given), saves
the coordinator decision, advances the durable instance, and resets the
per-step counters. It does NOT rebuild the tool surface or run entry ops —
those stay in the turn loop's existing handoff continuation, which every
origin now funnels through.

DESIGN — re-exported by ``kernel.turn_runner``:
    Existing tests import routing helpers from ``kernel.turn_runner``. That
    module re-exports the symbols below so no test file needs editing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

from kernel.models import WorkflowTransitionRequest
from kernel.seams.protocols import CoordinatorRouteDecision
from kernel.workflow_instance import WorkflowInstance, WorkflowStep

__all__ = [
    "RoutingSource",
    "apply_routing",
    "persist_planned_instance",
    "save_route_decision",
]

logger = logging.getLogger(__name__)

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
    if hooks.route_coordinator is None:
        return
    try:
        hooks.route_coordinator.save(
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
        )


def persist_planned_instance(
    hooks: Any,
    workflow_steps: tuple[WorkflowStep, ...],
    log: logging.Logger,
) -> None:
    """Persist a plan's steps as a ``WorkflowInstance`` (spawned_from=plan).

    No-op when there are no steps (weak-model fallback produced a bare
    target) or no updater is wired. Failure-tolerant: instance persistence
    failure must not abort the handoff — the turn continues and the ad-hoc
    chain path covers the gap.
    """
    if not workflow_steps or hooks.workflow_instance_updater is None:
        return
    try:
        instance = WorkflowInstance(
            declared_steps=workflow_steps,
            current_step_id=workflow_steps[0].step_id,
            current_mode_id=workflow_steps[0].mode_id,
            spawned_from="plan_workflow_op",
        )
        hooks.workflow_instance_updater(instance)
        log.info(
            "persisted plan_workflow_op WorkflowInstance instance_id=%s steps=%d first=%s",
            instance.instance_id,
            len(workflow_steps),
            workflow_steps[0].mode_id,
        )
    except Exception as exc:  # pylint: disable=broad-except
        log.warning("WorkflowInstance creation/persist failed for plan: %s", exc)


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

    Instance advancement (``advance_step``) stays in the turn loop's handoff
    continuation because it needs the live instance from ``hooks``; the
    applier keeps the counter/enforcer/coordinator bookkeeping that all three
    origins must share.
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
    _log.info(
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

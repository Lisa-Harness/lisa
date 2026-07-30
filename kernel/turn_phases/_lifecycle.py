"""Unified turn-lifecycle action resolver.

This module owns the single decision function that the turn loop consults each
iteration to decide whether to *continue* working, *transition* to another
mode, *yield* (ask the user a question), or *finish* (produce final output and
end the turn cleanly).

It replaces the organically-grown collage of scattered guards in
``kernel.turn_runner`` (``stage_nudge_threshold``/``cap`` injection,
``budget_forces_final`` tool-stripping, ``forced_fork``/``forced_plan``
post-hoc synthesis, the hallucinated-transition nudge, and the
chain-incomplete nudge) with one pure function that returns a tagged
:class:`LifecycleAction`.

Design constraints (mirroring the codebase conventions):

- Pure: takes the turn state, the active workflow instance, open recovery
  records, the model router's decision, and the kernel settings; returns an
  action. No I/O, no mutation. The turn loop applies the action.
- Tag-blind and prose-blind: the resolver reads opaque mode ids and step ids
  from the instance/policy — it names no mode in code (``discuss`` only
  appears as a default yield target when the policy's ``fallback_mode`` and
  ``is_default_yield_target`` are empty, and even then it is a fallback the
  caller may override).
- No compat: this is the hard cutover from the guard collage to one resolver.
  The guard constants (``stage_nudge_threshold``, ``stage_nudge_cap``,
  ``per_step_max_iterations``, ``per_step_max_tool_calls``,
  ``forced_fork_non_fork_tool_cap``, ``forced_plan_non_transition_tool_cap``, ``duplicate_hint_limit``,
  ``hint_cycle_min_len``, ``hint_override_limit``, ``max_handoffs_per_turn``)
  remain on ``kernel.config.Settings`` but their semantics flip from
  "force a silent final" to "force a :class:`YieldAction` or
  :class:`TransitionAction`". The loop's only true ceiling is
  ``max_tool_iterations``, which forces a yield rather than truncating.

See :doc:`/memories/session/plan` for the full design and the three-phase
cutover.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from kernel.models import YieldAction, YieldPolicy

__all__ = [
    "ContinueAction",
    "FinishAction",
    "LifecycleAction",
    "LifecycleResolverSettings",
    "TransitionAction",
    "resolve_lifecycle_action",
]


# ── Action tagged union ────────────────────────────────────────────────────
#
# A lightweight tagged union (dataclass subclasses) instead of a Pydantic
# model: these are per-iteration loop control values, not serialized state.
# Pydantic is for durable contracts (YieldAction, WorkflowInstance, ...);
# loop control is a pure-Python sum type.


@dataclass(frozen=True, slots=True)
class ContinueAction:
    """Keep the tool loop running. Optionally carry a soft nudge for the model.

    ``nudge`` is a short, actionable hint appended to the lifecycle briefing
    (never an injected ``user`` message — that path is what made "ask"
    indistinguishable from "done"). Empty means no nudge; the model works
    normally. Non-empty nudges are rare and advisory (e.g. "consider yielding
    if blocked"), not corrective.
    """

    nudge: str = ""


@dataclass(frozen=True, slots=True)
class TransitionAction:
    """Hand off to another mode mid-turn.

    ``target_mode_id`` is the mode to route to. ``rationale`` is logged and
    surfaced in the lifecycle briefing so the model understands the handoff.
    ``forced`` is True when the harness synthesized this transition because the
    model refused to call ``transition_stage`` (the old forced-fork/forced-plan
    cases); False when the model called it explicitly.
    """

    target_mode_id: str
    rationale: str = ""
    forced: bool = False


@dataclass(frozen=True, slots=True)
class FinishAction:
    """Produce final output and end the turn cleanly.

    ``strip_tools`` is True when the harness should advertise no tools on the
    final completion (the old ``budget_exhausted`` path); False when the model
    voluntarily finished with tools still available. ``reason`` is logged.
    """

    strip_tools: bool = False
    reason: str = ""


# ``YieldAction`` (from kernel.models) is the fourth variant — reused directly
# so the durable checkpoint contract and the loop-control union share one type.


LifecycleAction = ContinueAction | TransitionAction | YieldAction | FinishAction
"""The four terminal/continuation actions the harness and model share.

- ``ContinueAction``: keep iterating (the model wants to call tools).
- ``TransitionAction``: hand off to another mode (the model called
  ``transition_stage``, or the resolver forced a handoff on budget/stall).
- ``YieldAction``: pause and ask the user a question (the model emitted
  ``yield``, or the resolver forced a yield on cap/stall). Terminal — breaks
  the loop like ``FinishAction``.
- ``FinishAction``: produce final output and end the turn.
"""


# ── Resolver inputs ────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class LifecycleResolverSettings:
    """The settings thresholds the resolver consults, pre-extracted from
    ``kernel.config.Settings`` so the resolver stays pure and testable.

    Each field mirrors a ``kernel.config.Settings`` field; the turn loop
    reads them once at turn start (or per-iteration for the hot few) and
    passes them in. Defaults match the historical settings values.
    """

    max_tool_iterations: int = 15
    per_step_max_iterations: int = 8
    per_step_max_tool_calls: int = 12
    stage_nudge_threshold: int = 4
    stage_nudge_cap: int = 6


@runtime_checkable
class _WorkflowInstanceLike(Protocol):
    """Structural shape the resolver needs from a ``WorkflowInstance``.

    The resolver does not import ``kernel.workflow_instance.WorkflowInstance``
    to avoid a circular dependency; it reads the three attributes it needs via
    duck typing. The real model satisfies this protocol.
    """

    current_step_id: str
    current_mode_id: str
    declared_steps: tuple[Any, ...]


@runtime_checkable
class _RecoveryRecordLike(Protocol):
    """Structural shape the resolver needs from a ``RecoveryRecord``."""

    state: str
    blocks_workflow: bool
    retry_policy: str
    target_id: str
    target_kind: str
    last_error: str
    step_id: str


@runtime_checkable
class _RouterDecisionLike(Protocol):
    """Structural shape the resolver needs from a ``RouterDecision``."""

    finish: str
    transition_to: str


@dataclass(frozen=True, slots=True)
class _TurnStateView:
    """Read-only view of the per-turn counters the resolver consults.

    Mirrors the relevant fields on ``kernel.turn_phases._turn_state._TurnState``
    without importing that private module. The turn loop constructs this view
    from its state object each iteration.
    """

    iteration: int
    step_iteration: int
    step_tool_calls: int
    total_tool_calls: int
    turns_in_stage: int


def _has_next_step(instance: _WorkflowInstanceLike | None) -> bool:
    """Return True when the instance has a remaining step after the current one.

    Uses ``kernel.workflow_instance.is_complete`` when available (the real
    model); falls back to a structural check (``declared_steps`` non-empty and
    ``current_step_id`` set) so the resolver stays import-light.
    """
    if instance is None:
        return False
    try:
        from kernel.workflow_instance import is_complete  # noqa: PLC0415

        return not is_complete(instance)
    except Exception:  # pylint: disable=broad-except
        return bool(instance.declared_steps) and bool(instance.current_step_id)


def _next_step_mode_id(instance: _WorkflowInstanceLike | None) -> str:
    """Return the mode id of the step following the current one, or "".

    Mirrors the selection in ``kernel.turn_runner._next_remaining_step``: the
    step immediately after ``current_step_id`` in declaration order, falling
    back to the first remaining step. Returns "" when there is no next step.
    """
    if instance is None:
        return ""
    try:
        from kernel.workflow_instance import remaining_steps  # noqa: PLC0415

        remaining = remaining_steps(instance)
        if not remaining:
            return ""
        current = getattr(instance, "current_step_id", "") or ""
        for step in remaining:
            if step.step_id != current:
                return step.mode_id
        return remaining[0].mode_id
    except Exception:  # pylint: disable=broad-except
        return ""


def _stalled_blocking_record(
    recovery_records: list[_RecoveryRecordLike] | None,
) -> _RecoveryRecordLike | None:
    """Return the first stalled record that blocks the workflow, or None.

    A record blocks when its state is ``stalled`` (open for operator) AND
    ``blocks_workflow`` is True AND its ``retry_policy`` is ``confirm``
    (``safe`` records are auto-retried by the dispatcher before the resolver
    ever sees them; ``never`` records are skipped, not yielded). This is the
    Phase-3 wiring that makes recovery stalls lifecycle inputs.
    """
    if not recovery_records:
        return None
    for record in recovery_records:
        if (
            getattr(record, "state", "") == "stalled"
            and getattr(record, "blocks_workflow", False)
            and getattr(record, "retry_policy", "") == "confirm"
        ):
            return record
    return None


def _resolve_yield_target(
    policy: YieldPolicy,
    *,
    instance: _WorkflowInstanceLike | None,
    fallback_default: str = "discuss",
) -> str:
    """Resolve the mode id a yield should route to.

    Priority: the policy's ``fallback_mode`` (explicit per-mode override) →
    the instance's next-step mode (yield-in-place to continue the chain) →
    a hardcoded ``discuss`` last resort (only when the policy opted out of
    inline asks via ``allowed: false`` and named no fallback).

    The ``fallback_default`` is the only place a mode id appears in this
    module; it is a last-resort default the caller may override and is not
    authored in code anywhere a pack could control. Packs own the routing via
    ``yield_policy.fallback_mode``; this default only fires when a pack shipped
    an action mode with ``allowed: false`` and no ``fallback_mode`` (a
    misconfiguration the catalog validator should catch, but we degrade
    gracefully).
    """
    if policy.fallback_mode:
        return policy.fallback_mode
    if not policy.allowed:
        # Action mode that forbids inline asks but named no fallback. Use the
        # instance's next step if there is one (yield to continue the chain),
        # else the hardcoded default. This is a misconfiguration surface; the
        # validator should warn, but we never hang the turn.
        next_mode = _next_step_mode_id(instance)
        return next_mode or fallback_default
    # Chat mode with inline asks allowed: yield in place (empty target means
    # the harness surfaces the question in the current mode).
    return ""


def resolve_lifecycle_action(
    *,
    state: _TurnStateView,
    mode_id: str,
    mode_yield_policy: YieldPolicy,
    settings: LifecycleResolverSettings,
    router_decision: _RouterDecisionLike | None = None,
    workflow_instance: _WorkflowInstanceLike | None = None,
    recovery_records: list[_RecoveryRecordLike] | None = None,
    is_workflow: bool = False,
) -> LifecycleAction:
    """Decide the turn's next action for one iteration.

    Priority order (highest first):

    1. **Router decision** — the formal ``ModelRouter`` returned ``finish`` or
       ``transition_to``. The router is the highest authority; it can see
       query semantics the harness cannot.
    2. **Recovery stall blocks workflow** — a stalled ``RecoveryRecord`` with
       ``retry_policy="confirm"`` blocks the instance. Yield with the stall
       detail so the user can retry/skip. (Phase 3 wiring.)
    3. **Global iteration cap** — ``max_tool_iterations`` is the only true
       ceiling. Hitting it forces a yield (not a silent final) so the user can
       redirect, unless the mode forbids yields (action mode), in which case
       we finish (the old truncate behavior, preserved as the last resort).
    4. **Per-step budget exhausted** — the current workflow step consumed its
       ``per_step_max_iterations`` / ``per_step_max_tool_calls``. If there is a
       next step, force a ``TransitionAction`` to it (the old
       ``pending_planned_handoff`` synth); if not, yield (or finish for
       action modes).
    5. **Stuck-in-stage cap** — ``stage_nudge_cap`` turns in the same stage.
       Forces a yield (or finish for action modes). This is the 2026-07-30
       "loop max guard 6" fix: yield, don't truncate.
    6. **Stuck-in-stage threshold** — ``stage_nudge_threshold`` turns. Soft
       ``ContinueAction`` with a nudge ("consider yielding if blocked"). The
       model keeps working; no force.
    7. **Default** — ``ContinueAction`` (no nudge). The model works normally.

    The resolver never injects a ``user`` message (that was the old collage's
    mistake — it made "ask" indistinguishable from "done"). Nudges are advisory
    strings carried on ``ContinueAction`` for the lifecycle briefing; forces
    are terminal ``YieldAction`` / ``TransitionAction`` / ``FinishAction``.
    """
    # ── 1. Router decision (highest priority) ──────────────────────────
    if router_decision is not None:
        if getattr(router_decision, "finish", "").strip():
            return FinishAction(
                strip_tools=True,
                reason=f"router finish: {router_decision.finish.strip()}",
            )
        target = getattr(router_decision, "transition_to", "").strip()
        if target:
            return TransitionAction(
                target_mode_id=target,
                rationale=f"router transition_to",
                forced=False,
            )

    # ── 2. Recovery stall blocks the workflow ───────────────────────────
    blocking = _stalled_blocking_record(recovery_records)
    if blocking is not None:
        target = _resolve_yield_target(mode_yield_policy, instance=workflow_instance)
        detail = blocking.last_error or f"{blocking.target_kind} {blocking.target_id} stalled"
        return YieldAction(
            question=(
                f"A previous step stalled and needs your input to continue. "
                f"Stall: {detail}. Retry, skip, or revise?"
            ),
            context=f"Recovery record blocks workflow step '{blocking.step_id}'.",
            target_mode_id=target,
        )

    # ── 3. Global iteration cap (the only true ceiling) ─────────────────
    if state.iteration >= settings.max_tool_iterations:
        if mode_yield_policy.allowed or mode_yield_policy.fallback_mode:
            target = _resolve_yield_target(mode_yield_policy, instance=workflow_instance)
            question = (
                mode_yield_policy.default_question
                or "I've hit my iteration budget for this turn. How would you like me to proceed?"
            )
            return YieldAction(
                question=question,
                context=f"iteration cap {settings.max_tool_iterations} reached",
                target_mode_id=target,
            )
        # Action mode with no yield escape: preserve the old truncate as the
        # last resort (don't hang the turn).
        return FinishAction(
            strip_tools=True,
            reason=f"iteration cap {settings.max_tool_iterations} (action mode, no yield)",
        )

    # ── 4. Per-step budget exhausted ────────────────────────────────────
    step_budget_exhausted = (
        state.step_iteration > settings.per_step_max_iterations
        or state.step_tool_calls >= settings.per_step_max_tool_calls
    )
    if step_budget_exhausted and workflow_instance is not None and _has_next_step(workflow_instance):
        next_mode = _next_step_mode_id(workflow_instance)
        if next_mode:
            return TransitionAction(
                target_mode_id=next_mode,
                rationale=(
                    f"per-step budget exhausted "
                    f"(step_iteration={state.step_iteration}, "
                    f"step_tool_calls={state.step_tool_calls})"
                ),
                forced=True,
            )

    # ── 5. Stuck-in-stage cap (the "loop max guard 6" fix) ───────────────
    # Only consulted for workflow modes (``turns_in_stage`` is meaningless
    # for chat modes). The cap forces a yield, not a silent final.
    if is_workflow and state.turns_in_stage >= settings.stage_nudge_cap:
        if mode_yield_policy.allowed or mode_yield_policy.fallback_mode:
            target = _resolve_yield_target(mode_yield_policy, instance=workflow_instance)
            question = (
                mode_yield_policy.default_question
                or (
                    f"I've been working in this stage for {state.turns_in_stage} turns "
                    f"and haven't converged. Can you clarify what you need?"
                )
            )
            return YieldAction(
                question=question,
                context=f"stuck in stage for {state.turns_in_stage} turns (cap)",
                target_mode_id=target,
            )
        return FinishAction(
            strip_tools=True,
            reason=f"stuck in stage for {state.turns_in_stage} turns (action mode, no yield)",
        )

    # ── 6. Stuck-in-stage threshold (soft nudge) ────────────────────────
    if is_workflow and state.turns_in_stage >= settings.stage_nudge_threshold:
        return ContinueAction(
            nudge=(
                f"You have been in this stage for {state.turns_in_stage} turns. "
                f"If you have enough evidence, call transition_stage to advance. "
                f"Otherwise, call yield with a clarifying question."
            )
        )

    # ── 7. Default: keep working ───────────────────────────────────────
    return ContinueAction()
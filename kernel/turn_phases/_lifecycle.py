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
  ``forced_action_non_compliant_cap``, ``duplicate_hint_limit``,
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
from typing import Any, Protocol, runtime_checkable

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


@dataclass(frozen=True, slots=True)
class PauseAction:
    """Operator-initiated freeze. Break the loop WITHOUT returning the prompt.

    Checkpoint ``lifecycle_state="paused"``; the ``WorkflowPosition`` is
    preserved unchanged. The transport keeps the turn open; resume re-enters
    the same turn against the preserved position with a fresh iteration budget.

    Pause and stop write the **identical** ``CheckpointBundle`` — the only
    difference is whether the transport returns the prompt. If the terminal
    dies mid-pause, boot loads the paused checkpoint and the user lands in
    the stop state (prompt returned). One durable substrate, two live-session
    presentations.

    Also the terminal action the cap paths (global iteration cap, per-step
    exhausted with no next step, resource-budget exhausted) produce instead of
    ``YieldAction`` — the operator did not ask a question, the kernel ran out
    of budget. Freeze instead of surfacing "How would you like me to proceed?".
    """

    reason: str = ""


@dataclass(frozen=True, slots=True)
class StopAction:
    """Operator-initiated stop. Same checkpoint as pause, prompt returned.

    Checkpoint ``lifecycle_state="paused"`` (same as pause); the
    ``WorkflowPosition`` is preserved. The transport returns the prompt; the
    next message starts a fresh turn against the preserved position.
    """

    reason: str = ""


@dataclass(frozen=True, slots=True)
class TerminateAction:
    """Operator-initiated hard terminate. Workflow is killed.

    Checkpoint ``lifecycle_state="terminated"`` for audit/``/undo`` only (no
    resume). Mark the ``WorkflowPosition`` complete (clear ``current_step_id``)
    before building the checkpoint so the persisted snapshot reflects the
    termination. The transport returns the prompt; the next message is a
    fresh turn with no active workflow.

    Also the terminal action the action-mode cap last resort produces instead
    of silently finishing — pausing would hang an action mode that cannot
    resume, so terminate is the honest answer.
    """

    reason: str = ""


# ``YieldAction`` (from kernel.models) is the fourth variant — reused directly
# so the durable checkpoint contract and the loop-control union share one type.


LifecycleAction = (
    ContinueAction
    | TransitionAction
    | YieldAction
    | FinishAction
    | PauseAction
    | StopAction
    | TerminateAction
)
"""The terminal/continuation actions the harness, the model, and the operator share.

- ``ContinueAction``: keep iterating (the model wants to call tools).
- ``TransitionAction``: hand off to another mode (the model called
  ``transition_stage``, or the resolver forced a handoff on budget/stall).
- ``YieldAction``: pause and ask the user a question (the model emitted
  ``yield``, or the resolver forced a yield on a recovery stall). Terminal —
  breaks the loop like ``FinishAction``.
- ``FinishAction``: produce final output and end the turn.
- ``PauseAction``: operator freeze — checkpoint paused, no prompt returned.
- ``StopAction``: operator stop — same checkpoint as pause, prompt returned.
- ``TerminateAction``: operator terminate — checkpoint terminated (audit
  only), workflow marked complete, prompt returned.
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
    # Unified forced-action enforcer cap (Phase 5): non-compliant tool
    # dispatches before the kernel synthesizes the expected fork/transition
    # call itself.
    forced_action_non_compliant_cap: int = 3
    max_handoffs_per_turn: int = 5


@runtime_checkable
class _WorkflowPositionLike(Protocol):
    """Structural shape the resolver needs from a ``WorkflowPosition``.

    The resolver does not import ``kernel.workflow_position.WorkflowPosition``
    to avoid a circular dependency; it reads the three attributes it needs via
    duck typing. The real model satisfies this protocol.
    """

    current_step_id: str
    mode_id: str
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
    # ── Phase 4 extensions ───────────────────────────────────────────
    # Resource-budget exhaustion flag (computed by the loop's
    # ``_budget_forces_final()`` — the resolver stays pure and just reads
    # the boolean). True when TurnBudget tool-call / token / wall-time
    # limits are exceeded.
    budget_exhausted: bool = False
    # Unified forced-action enforcer view (Phase 5): a comma-joined
    # target string for a pending deterministic routing decision
    # (``fork:a,b`` for a fan-out, ``plan``/``search``/… for a single
    # handoff) plus the count of consecutive tool dispatches that were
    # not the expected tool. The enforcer itself runs in
    # ``turn_runner.py``'s tool-dispatch loop (it synthesizes the
    # ``fork``/``transition_stage`` call); these fields are carried for
    # diagnostic visibility only.
    forced_action_target: str = ""
    forced_action_non_compliant_dispatches: int = 0
    # Handoff-loop backstop: total mid-turn mode handoffs this turn.
    handoff_count: int = 0
    # Operator-initiated terminal signal (Phase: pause/stop/terminate).
    # One of ``""``, ``"pause"``, ``"stop"``, ``"terminate"``. Empty means no
    # operator signal is pending; the resolver ignores it. Polled once per
    # iteration by the turn loop from ``TurnHooks.operator_signal``.
    operator_signal: str = ""


def _has_next_step(position: _WorkflowPositionLike | None) -> bool:
    """Return True when the position has a remaining step after the current one.

    Uses ``kernel.workflow_position.is_complete`` when available (the real
    model); falls back to a structural check (``declared_steps`` non-empty and
    ``current_step_id`` set) so the resolver stays import-light.
    """
    if position is None:
        return False
    try:
        from kernel.workflow_position import is_complete  # noqa: PLC0415

        return not is_complete(position)
    except Exception:  # pylint: disable=broad-except
        return bool(position.declared_steps) and bool(position.current_step_id)


def _next_step_mode_id(position: _WorkflowPositionLike | None) -> str:
    """Return the mode id of the step following the current one, or "".

    Mirrors the selection in ``kernel.turn_runner._next_remaining_step``: the
    step immediately after ``current_step_id`` in declaration order, falling
    back to the first remaining step. Returns "" when there is no next step.
    """
    if position is None:
        return ""
    try:
        from kernel.workflow_position import remaining_steps  # noqa: PLC0415

        remaining = remaining_steps(position)
        if not remaining:
            return ""
        current = getattr(position, "current_step_id", "") or ""
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
    position: _WorkflowPositionLike | None,
    fallback_default: str = "discuss",
) -> str:
    """Resolve the mode id a yield should route to.

    Priority: the policy's ``fallback_mode`` (explicit per-mode override) →
    the position's next-step mode (yield-in-place to continue the chain) →
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
        # position's next step if there is one (yield to continue the chain),
        # else the hardcoded default. This is a misconfiguration surface; the
        # validator should warn, but we never hang the turn.
        next_mode = _next_step_mode_id(position)
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
    workflow_position: _WorkflowPositionLike | None = None,
    recovery_records: list[_RecoveryRecordLike] | None = None,
) -> LifecycleAction:
    """Decide the turn's next action for one iteration.

    Priority order (highest first):

    1. **Router decision** — the formal ``ModelRouter`` returned ``finish`` or
       ``transition_to``. The router is the highest authority; it can see
       query semantics the harness cannot.
    2. **Recovery stall blocks workflow** — a stalled ``RecoveryRecord`` with
       ``retry_policy="confirm"`` blocks the instance. Yield with the stall
       detail so the user can retry/skip. (Phase 3 wiring.)
    2b. **Handoff-loop backstop** — ``max_handoffs_per_turn`` exceeded. Finish
       to prevent infinite routing-hint cycles. (Phase 4.)
    2c. **Forced-fork enforcer** — the kernel has a deterministic ``fork:``
       target the model refused to call ``transition_stage`` for. After the
       cap of non-transition dispatches, force a ``TransitionAction``. (Phase 4.)
    2d. **Forced-plan enforcer** — mirrors forced-fork for the ``plan``
       routing-escalation target. (Phase 4.)
    3. **Global iteration cap** — ``max_tool_iterations`` is the only true
       ceiling. Hitting it forces a yield (not a silent final) so the user can
       redirect, unless the mode forbids yields (action mode), in which case
       we finish (the old truncate behavior, preserved as the last resort).
    3b. **Resource-budget exhaustion** — the ``TurnBudget`` (tool calls /
       tokens / wall time) is exhausted. Forces a yield (or finish for action
       modes). (Phase 4.)
    4. **Per-step budget exhausted** — the current workflow step consumed its
       ``per_step_max_iterations`` / ``per_step_max_tool_calls``. If there is a
       next step, force a ``TransitionAction`` to it (the old
       ``pending_planned_handoff`` synth); if not, yield (or finish for
       action modes).
    5. **Stuck-in-stage cap** — ``stage_nudge_cap`` iterations in the same
       stage/mode without a transition. Forces a yield (or finish for action
       modes). Applies uniformly to every mode kind: the loop feeds
       ``turns_in_stage`` from the workflow session for workflow modes and
       from the per-step counter (reset on every ``transition_stage``) for
       chat/action modes, so one counter drives one guard. This is the
       2026-07-30 "loop max guard 6" fix (yield, don't truncate) unified
       with the 2026-08-05 stuck-in-chat-mode guard — the split into
       workflow-only and chat-only guards was the drift the unification
       removes.
    6. **Stuck-in-stage threshold** — ``stage_nudge_threshold`` iterations.
       Soft ``ContinueAction`` with a nudge ("consider yielding if blocked").
       The model keeps working; no force.
    7. **Default** — ``ContinueAction`` (no nudge). The model works normally.

    The resolver never injects a ``user`` message (that was the old collage's
    mistake — it made "ask" indistinguishable from "done"). Nudges are advisory
    strings carried on ``ContinueAction`` for the lifecycle briefing; forces
    are terminal ``YieldAction`` / ``TransitionAction`` / ``FinishAction`` /
    ``PauseAction`` / ``StopAction`` / ``TerminateAction``.
    """
    # ── 0. Operator signal (highest priority) ──────────────────────────
    # The operator is the highest authority: pause/stop/terminate override
    # the router decision and every cap path. Pause and stop share one
    # checkpoint (``lifecycle_state="paused"``); the difference is whether
    # the transport returns the prompt. Terminate marks the workflow
    # complete and writes a ``terminated`` checkpoint for audit/``/undo``.
    _op = state.operator_signal
    if _op == "pause":
        return PauseAction(reason="operator pause")
    if _op == "stop":
        return StopAction(reason="operator stop")
    if _op == "terminate":
        return TerminateAction(reason="operator terminate")

    # ── 1. Router decision ──────────────────────────────────────────────
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
                rationale="router transition_to",
                forced=False,
            )

    # ── 2. Recovery stall blocks the workflow ───────────────────────────
    blocking = _stalled_blocking_record(recovery_records)
    if blocking is not None:
        target = _resolve_yield_target(mode_yield_policy, position=workflow_position)
        detail = blocking.last_error or f"{blocking.target_kind} {blocking.target_id} stalled"
        work_item_id = str(getattr(blocking, "work_item_id", "") or "")
        # Print the literal slash commands with the work-item id so the user
        # has a concrete, copy-pasteable escape. The historical prompt
        # ("Retry, skip, or revise?") implied bare words work, but only the
        # explicit ``/retry|/skip|/cancel <id>`` forms are real commands, and
        # ``/retry`` on a confirm-policy record directs the user to re-issue
        # the request or /skip it. Keep the prompt actionable, not a riddle.
        commands_hint = (
            f"Run /skip {work_item_id} or /cancel {work_item_id} to dismiss, "
            f"or /retry {work_item_id} to retry. Otherwise rephrase your request."
            if work_item_id
            else "Run /stalls to see open work items, or rephrase your request."
        )
        return YieldAction(
            question=(
                f"A previous step stalled and needs your input to continue. "
                f"Stall: {detail}. {commands_hint}"
            ),
            context=f"Recovery record blocks workflow step '{blocking.step_id}'.",
            target_mode_id=target,
        )

    # ── 2b. Handoff-loop backstop (Phase 4) ──────────────────────────────
    # The ``handoff_count`` hard backstop prevents infinite routing-hint
    # cycles when ``chain_satisfied`` attribution fails. Fires before the
    # iteration cap so a tight handoff loop (A→B→A→B...) is caught early
    # rather than burning the full iteration budget.
    if state.handoff_count > settings.max_handoffs_per_turn:
        return FinishAction(
            strip_tools=True,
            reason=(
                f"handoff loop guard: {state.handoff_count} handoffs "
                f"(cap={settings.max_handoffs_per_turn})"
            ),
        )

    # ── 2d. Forced-transition enforcer (Phase 4) ───────────────────────
    # Bounds the pre-handoff token burn to roughly
    # ``cap x per-iteration-tokens`` instead of the full iteration budget
    # when the model refuses to call ``transition_stage`` for a hinted
    # single-handoff target (``plan``, ``search``, ``remember`` …).
    # Generalized 2026-08-07 from the plan-only enforcer so a weak model
    # that refuses to call ``transition_stage`` for ANY next step is
    # forced after the cap. The fork (multi-target) shape has no resolver
    # guard — the dispatch-loop enforcer is the single synthesis site
    # (Phase 5; the pre-Phase-5 fork guard here and the loop enforcer
    # were dual paths for one decision).
    if (
        state.forced_action_target
        and not state.forced_action_target.startswith("fork:")
        and state.forced_action_non_compliant_dispatches
        >= settings.forced_action_non_compliant_cap
    ):
        return TransitionAction(
            target_mode_id=state.forced_action_target,
            rationale=(
                "kernel-forced transition handoff: the model did not call "
                "transition_stage after the routing hint named a target."
            ),
            forced=True,
        )

    # ── 3. Global iteration cap (the only true ceiling) ────────────────
    # Caps pause, they do not yield. The operator did not ask a question;
    # the kernel ran out of budget. Freeze the turn in place with a paused
    # checkpoint so the operator can resume for a fresh budget, or stop /
    # terminate. The old "How would you like me to proceed?" question is
    # gone — only model-emitted yields and recovery stalls surface a
    # question.
    if state.iteration >= settings.max_tool_iterations:
        if mode_yield_policy.allowed or mode_yield_policy.fallback_mode:
            return PauseAction(
                reason=(
                    f"iteration cap {settings.max_tool_iterations} reached"
                ),
            )
        # Action mode with no yield escape: pausing would hang a mode that
        # cannot resume. Terminate is the honest answer — mark the workflow
        # done, checkpoint for audit, return the prompt. The old silent
        # truncate (FinishAction) is gone; the operator gets a real terminal
        # state they can see.
        return TerminateAction(
            reason=(
                f"iteration cap {settings.max_tool_iterations} "
                f"(action mode, no yield escape)"
            ),
        )

    # ── 3b. Resource-budget exhaustion (Phase 4) ────────────────────
    # The TurnBudget (tool calls / tokens / wall time) is exhausted. Pause
    # so the operator can resume for a fresh budget, instead of surfacing a
    # question or truncating. When no TurnBudget is wired (the common case),
    # ``budget_exhausted`` is False and this is a no-op.
    if state.budget_exhausted:
        if mode_yield_policy.allowed or mode_yield_policy.fallback_mode:
            return PauseAction(
                reason="turn budget exhausted (tool calls / tokens / wall time)",
            )
        return TerminateAction(
            reason="turn budget exhausted (action mode, no yield escape)",
        )

    # ── 4. Per-step budget exhausted ────────────────────────────────────
    step_budget_exhausted = (
        state.step_iteration > settings.per_step_max_iterations
        or state.step_tool_calls >= settings.per_step_max_tool_calls
    )
    if (
        step_budget_exhausted
        and workflow_position is not None
        and _has_next_step(workflow_position)
    ):
        next_mode = _next_step_mode_id(workflow_position)
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

    # ── 5. Stuck-in-stage cap (unified across mode kinds) ──────────────
    # ``turns_in_stage`` is fed uniformly by the loop: the workflow
    # session's per-stage turn count for workflow modes, the per-step
    # iteration counter (reset on every ``transition_stage``) for
    # chat/action modes. One counter, one cap. The cap pauses the turn —
    # the operator can resume, stop, or terminate. The old 2026-07-30
    # "loop max guard 6" fix (yield, don't truncate) is now a pause: same
    # state preservation, no spurious question, now covering the
    # chat-mode case the 2026-08-05 markdown burn exposed.
    if state.turns_in_stage >= settings.stage_nudge_cap:
        if mode_yield_policy.allowed or mode_yield_policy.fallback_mode:
            return PauseAction(
                reason=(
                    f"stuck in {mode_id} for {state.turns_in_stage} "
                    f"iterations (cap)"
                ),
            )
        return TerminateAction(
            reason=(
                f"stuck in {mode_id} for {state.turns_in_stage} iterations "
                f"(action mode, no yield escape)"
            ),
        )

    # ── 6. Stuck-in-stage threshold (soft nudge) ────────────────────────
    if state.turns_in_stage >= settings.stage_nudge_threshold:
        return ContinueAction(
            nudge=(
                f"You have been in {mode_id} for {state.turns_in_stage} iterations "
                f"without advancing. If you have enough evidence, call "
                f"transition_stage to advance. Otherwise, call yield with a "
                f"clarifying question."
            )
        )

    # ── 7. Default: keep working ───────────────────────────────────────
    return ContinueAction()

"""Tests for the unified turn-lifecycle action resolver.

``resolve_lifecycle_action`` is the pure decision function that replaces the
organically-grown collage of scattered guards (stage_nudge_threshold/cap,
budget_forces_final, forced_fork/forced_plan, chain-incomplete nudge) with
one tagged-union return: ``ContinueAction`` / ``TransitionAction`` /
``YieldAction`` / ``FinishAction``.

These tests cover each priority branch (router, recovery-blocks, iteration
cap, per-step budget, stage cap, stage threshold, default) in isolation so
the resolver's contract is locked independent of the turn loop.
"""

from __future__ import annotations

from dataclasses import dataclass

from kernel.models import YieldAction, YieldPolicy
from kernel.recovery import (
    RECOVERY_APPROVAL_PENDING_ERROR,
    RecoveryIdentity,
    RecoveryRecord,
)
from kernel.turn_phases._lifecycle import (
    ContinueAction,
    FinishAction,
    LifecycleResolverSettings,
    PauseAction,
    StopAction,
    TerminateAction,
    TransitionAction,
    _RecoveryRecordLike,
    resolve_lifecycle_action,
)


@dataclass
class _State:
    iteration: int = 1
    step_iteration: int = 1
    step_tool_calls: int = 0
    total_tool_calls: int = 0
    turns_in_stage: int = 0
    # Phase 4 extension fields (defaults match _TurnStateView).
    budget_exhausted: bool = False
    # Unified forced-action enforcer view (Phase 5): comma-joined target
    # string (``fork:a,b`` fan-out or a single-handoff mode id) plus the
    # count of consecutive non-compliant tool dispatches.
    forced_action_target: str = ""
    forced_action_non_compliant_dispatches: int = 0
    handoff_count: int = 0
    # Operator-signal field (pause/stop/terminate). Defaults to "" so
    # existing tests ignore it; the resolver only acts on non-empty.
    operator_signal: str = ""
    # Write-rescue inputs (forced-transition-before-cap).
    action_tool_ids: frozenset[str] = frozenset()
    has_write_receipt: bool = False
    has_read_receipt: bool = False
    # Error-streak inputs (stage nudge enrichment).
    consecutive_tool_errors: int = 0
    last_error_tool: str = ""


@dataclass
class _Step:
    step_id: str
    mode_id: str
    write_capability: str | None = None


@dataclass
class _Instance:
    current_step_id: str = "s1"
    current_mode_id: str = "search"
    declared_steps: tuple[_Step, ...] = (
        _Step("s1", "search"),
        _Step("s2", "coding"),
    )
    completed_steps: tuple[str, ...] = ()


def _settings(**overrides: object) -> LifecycleResolverSettings:
    """Build a fully-populated ``LifecycleResolverSettings`` for tests.

    The resolver dataclass has no defaults (every field mirrors a
    ``kernel.config.Settings`` field); tests supply explicit values via
    this helper so the resolver never second-guesses the configured
    settings. Override only the field the branch under test exercises.
    """
    base = {
        "max_tool_iterations": 15,
        "per_step_max_iterations": 8,
        "per_step_max_tool_calls": 12,
        "stage_nudge_threshold": 4,
        "stage_nudge_cap": 6,
        "forced_action_non_compliant_cap": 3,
        "max_handoffs_per_turn": 5,
        "write_rescue_mode": "explainer",
    }
    base.update(overrides)
    return LifecycleResolverSettings(**base)  # type: ignore[arg-type]


@dataclass
class _Router:
    finish: str = ""
    transition_to: str = ""


def _record_identity() -> RecoveryIdentity:
    return RecoveryIdentity(
        workspace_id="w",
        user_id="u",
        user_slug="u",
        surface_type="personal_dm",
        surface_id="s",
        memory_scope="personal",
    )


def _record(
    *,
    state: str = "running",
    retry_policy: str = "safe",
    last_error: str = "",
    step_id: str = "",
    mode_id: str = "search",
) -> RecoveryRecord:
    """Build a real ``RecoveryRecord`` — the turn runner passes real models
    from ``recovery_service.list_open`` into the resolver, so fixtures must
    match the production shape (a ghost attribute here once disabled the
    stall gate silently)."""
    return RecoveryRecord(
        identity=_record_identity(),
        mode_id=mode_id,
        target_kind="pack_tool",
        target_id="web_search",
        state=state,
        retry_policy=retry_policy,
        last_error=last_error,
        step_id=step_id,
    )


# ── Default: continue ──────────────────────────────────────────────────────


def test_default_is_continue_with_no_nudge() -> None:
    action = resolve_lifecycle_action(
        state=_State(),
        mode_id="default",
        mode_yield_policy=YieldPolicy(),
        settings=_settings(),
    )
    assert isinstance(action, ContinueAction)
    assert action.nudge == ""


# ── Router decision (highest priority) ──────────────────────────────────────


def test_router_finish_takes_precedence_over_cap() -> None:
    """Even at the iteration cap, a router finish wins."""
    action = resolve_lifecycle_action(
        state=_State(iteration=15),
        mode_id="default",
        mode_yield_policy=YieldPolicy(),
        settings=_settings(),
        router_decision=_Router(finish="done"),
    )
    assert isinstance(action, FinishAction)
    assert action.strip_tools is True


def test_router_transition_to_returns_transition() -> None:
    action = resolve_lifecycle_action(
        state=_State(),
        mode_id="default",
        mode_yield_policy=YieldPolicy(),
        settings=_settings(),
        router_decision=_Router(transition_to="coding"),
    )
    assert isinstance(action, TransitionAction)
    assert action.target_mode_id == "coding"
    assert action.forced is False


# ── Recovery stall blocks workflow ──────────────────────────────────────────


def test_real_record_satisfies_resolver_protocol() -> None:
    """The resolver protocol must be satisfied by the REAL ``RecoveryRecord``.

    The turn runner passes real models from ``recovery_service.list_open``
    into the resolver. If this drifts, a gate reading a ghost attribute
    (defaulting via ``getattr``) silently disables itself instead of
    failing — exactly what happened when the checkpoint trim removed
    ``blocks_workflow`` while the stall gate still read it.
    """
    record = _record()
    assert isinstance(record, _RecoveryRecordLike)


def test_stalled_blocking_record_forces_yield() -> None:
    """A stalled confirm-policy record pending approval blocks the workflow.

    Regression for the checkpoint-trim regression: the resolver used to
    gate on the removed ``blocks_workflow`` attribute, so real records
    never tripped the priority-2 stall yield.
    """
    record = _record(
        state="stalled",
        retry_policy="confirm",
        last_error=RECOVERY_APPROVAL_PENDING_ERROR,
        step_id="s1",
    )
    action = resolve_lifecycle_action(
        state=_State(),
        mode_id="search",
        mode_yield_policy=YieldPolicy(allowed=True),
        settings=_settings(),
        recovery_records=[record],
    )
    assert isinstance(action, YieldAction)
    assert "blocks workflow" in action.context
    assert RECOVERY_APPROVAL_PENDING_ERROR in action.question
    # Structured remediation carries verbs + the work item id — never slash
    # syntax (pack vocabulary does not cross the kernel boundary).
    assert action.remediation is not None
    verbs = [a.verb for a in action.remediation.actions]
    assert verbs == ["skip", "cancel", "inspect", "retry"]
    assert all(a.target_id == record.work_item_id for a in action.remediation.actions)
    assert "/skip" not in action.question
    assert "/retry" not in action.question
    assert "/stalls" not in action.question
    # A confirm-policy retry is surfaced as unsafe: the host must ask.
    retry_action = next(a for a in action.remediation.actions if a.verb == "retry")
    assert retry_action.safe is False


def test_stall_remediation_omitted_without_work_item_id() -> None:
    """A record with no work item id cannot be remediated by id."""
    record = _record(
        state="stalled",
        retry_policy="confirm",
        last_error=RECOVERY_APPROVAL_PENDING_ERROR,
        step_id="s1",
    )
    record = record.model_copy(update={"work_item_id": ""})
    action = resolve_lifecycle_action(
        state=_State(),
        mode_id="search",
        mode_yield_policy=YieldPolicy(allowed=True),
        settings=_settings(),
        recovery_records=[record],
    )
    assert isinstance(action, YieldAction)
    assert action.remediation is None
    assert action.question.endswith("rephrase your request.")


def test_stall_question_carries_no_command_syntax() -> None:
    """The kernel-side contract: no slash syntax may appear in the question.

    The stall yield's question is kernel-owned prose; verbs + ids travel in
    ``remediation``. Guards the boundary the runtime's rendering relies on,
    since the vocabulary move moves ``/skip`` / ``/cancel`` / ``/retry`` /
    ``/stalls`` out of the kernel prompt entirely.
    """
    from kernel.models import YieldRemediation, YieldRemediationAction

    record = _record(
        state="stalled",
        retry_policy="confirm",
        last_error=RECOVERY_APPROVAL_PENDING_ERROR,
        step_id="s1",
    )
    action = resolve_lifecycle_action(
        state=_State(),
        mode_id="search",
        mode_yield_policy=YieldPolicy(allowed=True),
        settings=_settings(),
        recovery_records=[record],
    )
    assert isinstance(action, YieldAction)
    assert isinstance(action.remediation, YieldRemediation)
    assert action.remediation.actions
    assert isinstance(action.remediation.actions[0], YieldRemediationAction)
    for slash in ("/skip", "/cancel", "/retry", "/stalls", "/stall", "/approve", "/deny"):
        assert slash not in action.question


def test_tool_failure_stall_does_not_block() -> None:
    """An ordinary tool-failure stall must NOT re-yield on later turns.

    The tool error already flowed back to the model as a tool result, so
    the model can adapt; re-yielding would trap the user in a pre-model
    prompt loop. Only the approval-pending sentinel blocks.
    """
    record = _record(
        state="stalled",
        retry_policy="confirm",
        last_error="web_search timed out",
        step_id="s1",
    )
    action = resolve_lifecycle_action(
        state=_State(),
        mode_id="search",
        mode_yield_policy=YieldPolicy(allowed=True),
        settings=_settings(),
        recovery_records=[record],
    )
    assert isinstance(action, ContinueAction)


def test_safe_retry_record_does_not_block() -> None:
    """A stalled record with retry_policy='safe' is auto-retried, not yielded."""
    action = resolve_lifecycle_action(
        state=_State(),
        mode_id="search",
        mode_yield_policy=YieldPolicy(),
        settings=_settings(),
        recovery_records=[_record(state="stalled", retry_policy="safe")],
    )
    assert isinstance(action, ContinueAction)


# ── Global iteration cap ────────────────────────────────────────────────────


def test_iteration_cap_forces_pause_for_chat_mode() -> None:
    """Caps pause, they do not yield. The operator did not ask a question; the
    kernel ran out of budget. Freeze the turn in place."""
    action = resolve_lifecycle_action(
        state=_State(iteration=15),
        mode_id="default",
        mode_yield_policy=YieldPolicy(allowed=True),
        settings=_settings(max_tool_iterations=15),
    )
    assert isinstance(action, PauseAction)
    assert "iteration cap 15" in action.reason


def test_iteration_cap_forces_terminate_for_action_mode_without_escape() -> None:
    """An action mode that forbids yield and names no fallback terminates —
    pausing would hang a mode that cannot resume."""
    action = resolve_lifecycle_action(
        state=_State(iteration=15),
        mode_id="coding",
        mode_yield_policy=YieldPolicy(allowed=False, fallback_mode=""),
        settings=_settings(max_tool_iterations=15),
    )
    assert isinstance(action, TerminateAction)
    assert "action mode, no yield escape" in action.reason


def test_iteration_cap_with_fallback_mode_pauses() -> None:
    """A mode with a fallback_mode still pauses (not yields) at the cap — caps
    freeze, they don't surface a question."""
    action = resolve_lifecycle_action(
        state=_State(iteration=15),
        mode_id="coding",
        mode_yield_policy=YieldPolicy(allowed=False, fallback_mode="discuss"),
        settings=_settings(max_tool_iterations=15),
    )
    assert isinstance(action, PauseAction)


# ── Per-step budget exhausted ───────────────────────────────────────────────


def test_per_step_budget_exhausted_forces_transition_to_next_step() -> None:
    action = resolve_lifecycle_action(
        state=_State(step_iteration=9, step_tool_calls=0),
        mode_id="search",
        mode_yield_policy=YieldPolicy(),
        settings=_settings(per_step_max_iterations=8),
        workflow_position=_Instance(),
    )
    assert isinstance(action, TransitionAction)
    assert action.target_mode_id == "coding"
    assert action.forced is True


def test_per_step_budget_exhausted_no_next_step_falls_through() -> None:
    """When the instance has no next step, the per-step cap doesn't fire
    a transition; the global cap or default continue handles it."""
    action = resolve_lifecycle_action(
        state=_State(step_iteration=9, step_tool_calls=0),
        mode_id="coding",
        mode_yield_policy=YieldPolicy(),
        settings=_settings(per_step_max_iterations=8),
        workflow_position=_Instance(
            current_step_id="s2",
            current_mode_id="coding",
            declared_steps=(_Step("s1", "search"), _Step("s2", "coding")),
            completed_steps=("s1", "s2"),
        ),
    )
    # No next step → not a transition; falls through to default continue
    # (the global cap would catch it if iteration were high).
    assert isinstance(action, ContinueAction)


# ── Stuck-in-stage cap (unified across mode kinds) ─────────────────────────


def test_stage_cap_forces_pause_for_workflow_mode() -> None:
    """Caps pause, they do not yield. The stuck-in-stage cap freezes the turn."""
    action = resolve_lifecycle_action(
        state=_State(turns_in_stage=6),
        mode_id="ponder",
        mode_yield_policy=YieldPolicy(allowed=True),
        settings=_settings(stage_nudge_cap=6),
    )
    assert isinstance(action, PauseAction)
    assert "6 iterations" in action.reason


def test_stage_cap_forces_terminate_for_action_mode_without_escape() -> None:
    """An action mode that forbids yield terminates at the stuck-in-stage cap."""
    action = resolve_lifecycle_action(
        state=_State(turns_in_stage=6),
        mode_id="coding",
        mode_yield_policy=YieldPolicy(allowed=False, fallback_mode=""),
        settings=_settings(stage_nudge_cap=6),
    )
    assert isinstance(action, TerminateAction)


def test_stage_cap_fires_for_chat_mode_too() -> None:
    """The guard is mode-kind blind: a chat mode stuck at the cap pauses.

    The 2026-08-05 markdown burn happened because the workflow guard was
    gated on ``is_workflow`` and chat modes slipped through. The unified
    guard reads the same counter for every mode kind (the loop feeds it
    from the workflow session for workflow modes, the per-step counter for
    chat/action modes)."""
    action = resolve_lifecycle_action(
        state=_State(turns_in_stage=10),
        mode_id="markdown",
        mode_yield_policy=YieldPolicy(allowed=True),
        settings=_settings(stage_nudge_cap=6),
    )
    assert isinstance(action, PauseAction)
    assert "markdown" in action.reason


def test_stage_cap_ignored_below_cap() -> None:
    action = resolve_lifecycle_action(
        state=_State(turns_in_stage=5),
        mode_id="ponder",
        mode_yield_policy=YieldPolicy(),
        settings=_settings(stage_nudge_threshold=4, stage_nudge_cap=6),
    )
    # At 5 with threshold 4: the soft nudge fires, not the cap.
    assert isinstance(action, ContinueAction)
    assert action.nudge != ""


# ── Stuck-in-stage threshold (soft nudge) ───────────────────────────────────


def test_stage_threshold_emits_continue_with_nudge() -> None:
    action = resolve_lifecycle_action(
        state=_State(turns_in_stage=4),
        mode_id="ponder",
        mode_yield_policy=YieldPolicy(),
        settings=_settings(stage_nudge_threshold=4, stage_nudge_cap=6),
    )
    assert isinstance(action, ContinueAction)
    assert action.nudge != ""
    assert "transition_stage" in action.nudge
    assert "yield" in action.nudge


# ── Yield target resolution ─────────────────────────────────────────────────


def test_yield_uses_policy_fallback_mode() -> None:
    """A mode with a fallback_mode still pauses at the cap (caps freeze, they
    don't yield). The fallback_mode would only matter for a model-emitted
    YieldAction or a recovery-stall yield — not for cap paths."""
    action = resolve_lifecycle_action(
        state=_State(iteration=15),
        mode_id="coding",
        mode_yield_policy=YieldPolicy(allowed=False, fallback_mode="discuss"),
        settings=_settings(max_tool_iterations=15),
    )
    assert isinstance(action, PauseAction)


def test_yield_in_place_when_policy_allows_inline() -> None:
    """A chat mode with allowed=True pauses at the cap (not yields in place).
    Caps freeze; only model-emitted yields surface a question."""
    action = resolve_lifecycle_action(
        state=_State(iteration=15),
        mode_id="default",
        mode_yield_policy=YieldPolicy(allowed=True),
        settings=_settings(max_tool_iterations=15),
    )
    assert isinstance(action, PauseAction)


# ── Phase 4: Handoff-loop backstop ───────────────────────────────────────────


def test_handoff_loop_backstop_forces_finish() -> None:
    """When handoff_count exceeds max_handoffs_per_turn, finish immediately."""
    action = resolve_lifecycle_action(
        state=_State(handoff_count=6),
        mode_id="default",
        mode_yield_policy=YieldPolicy(),
        settings=_settings(max_handoffs_per_turn=5),
    )
    assert isinstance(action, FinishAction)
    assert action.strip_tools
    assert "handoff" in action.reason


def test_handoff_loop_backstop_not_triggered_within_cap() -> None:
    """At exactly the cap, the backstop does not fire (>)."""
    action = resolve_lifecycle_action(
        state=_State(handoff_count=5),
        mode_id="default",
        mode_yield_policy=YieldPolicy(),
        settings=_settings(max_handoffs_per_turn=5),
    )
    assert isinstance(action, ContinueAction)


# ── Phase 5: Unified forced-action enforcer ────────────────────────────
#
# One enforcer drives both shapes (fork fan-out and single-handoff
# transition); the resolver guard fires only for the transition shape. The
# fork shape is enforced solely by the dispatch-loop enforcer in
# ``turn_runner.py`` — a pre-Phase-5 dual path (resolver guard AND loop
# enforcer for one decision) was collapsed into the single loop site.


def test_forced_transition_enforcer_fires_at_cap() -> None:
    """When the forced-action target is a single handoff and non-compliant
    dispatches hit the cap, force a TransitionAction to the target."""
    action = resolve_lifecycle_action(
        state=_State(
            forced_action_target="plan",
            forced_action_non_compliant_dispatches=3,
        ),
        mode_id="default",
        mode_yield_policy=YieldPolicy(),
        settings=_settings(forced_action_non_compliant_cap=3),
    )
    assert isinstance(action, TransitionAction)
    assert action.forced
    assert action.target_mode_id == "plan"


def test_forced_transition_enforcer_fires_for_non_plan_target() -> None:
    """The generalized enforcer fires for ANY routing target, not just plan.

    Regression for the 2026-08-07 ``plan → remember → search`` stall: the
    old plan-only enforcer left ``search`` unreachable when the weak model
    in ``remember`` returned text without calling ``transition_stage``.
    """
    action = resolve_lifecycle_action(
        state=_State(
            forced_action_target="search",
            forced_action_non_compliant_dispatches=3,
        ),
        mode_id="remember",
        mode_yield_policy=YieldPolicy(),
        settings=_settings(forced_action_non_compliant_cap=3),
    )
    assert isinstance(action, TransitionAction)
    assert action.forced
    assert action.target_mode_id == "search"


def test_forced_transition_enforcer_not_triggered_below_cap() -> None:
    """Below the cap, the enforcer does not fire."""
    action = resolve_lifecycle_action(
        state=_State(
            forced_action_target="plan",
            forced_action_non_compliant_dispatches=2,
        ),
        mode_id="default",
        mode_yield_policy=YieldPolicy(),
        settings=_settings(forced_action_non_compliant_cap=3),
    )
    assert isinstance(action, ContinueAction)


def test_forced_action_fork_shape_not_resolver_enforced() -> None:
    """A ``fork:``-shaped forced action never fires the resolver guard.

    Replaces the pre-Phase-5 ``test_forced_fork_enforcer_fires_at_cap``
    (a standing baseline failure): that test expected a resolver
    ``TransitionAction`` for a fork target, but the fork enforcer has
    always lived in the turn_runner dispatch loop — it synthesizes a
    ``fork`` operation call, not a ``transition_stage``. The resolver now
    explicitly excludes the fork shape; the loop enforcer is the single
    synthesis site for both shapes.
    """
    action = resolve_lifecycle_action(
        state=_State(
            forced_action_target="fork:search,markdown",
            forced_action_non_compliant_dispatches=3,
        ),
        mode_id="default",
        mode_yield_policy=YieldPolicy(),
        settings=_settings(forced_action_non_compliant_cap=3),
    )
    assert isinstance(action, ContinueAction)


def test_forced_action_enforcer_not_triggered_without_target() -> None:
    """No forced action means no enforcer, even above the cap."""
    action = resolve_lifecycle_action(
        state=_State(forced_action_target="", forced_action_non_compliant_dispatches=10),
        mode_id="default",
        mode_yield_policy=YieldPolicy(),
        settings=_settings(forced_action_non_compliant_cap=3),
    )
    assert isinstance(action, ContinueAction)


# ── Phase 4: Resource-budget exhaustion ─────────────────────────────────────


def test_budget_exhausted_forces_pause_for_chat_mode() -> None:
    """When budget_exhausted is True and the mode allows yields, pause.
    Caps freeze, they don't surface a question."""
    action = resolve_lifecycle_action(
        state=_State(budget_exhausted=True),
        mode_id="default",
        mode_yield_policy=YieldPolicy(allowed=True),
        settings=_settings(),
    )
    assert isinstance(action, PauseAction)
    assert "budget" in action.reason


def test_budget_exhausted_forces_terminate_for_action_mode() -> None:
    """When budget_exhausted is True and the mode forbids yields, terminate."""
    action = resolve_lifecycle_action(
        state=_State(budget_exhausted=True),
        mode_id="search",
        mode_yield_policy=YieldPolicy(allowed=False),
        settings=_settings(),
    )
    assert isinstance(action, TerminateAction)
    assert "action mode" in action.reason


def test_budget_exhausted_not_triggered_when_false() -> None:
    """When budget_exhausted is False, the branch is a no-op."""
    action = resolve_lifecycle_action(
        state=_State(budget_exhausted=False),
        mode_id="default",
        mode_yield_policy=YieldPolicy(),
        settings=_settings(),
    )
    assert isinstance(action, ContinueAction)


# ── Stuck-in-chat-mode cap (2026-08-05, subsumed by the unified guard) ─────
#
# The chat-stuck guards were deleted in the Phase-2 unification: the resolver
# now reads ONE counter (``turns_in_stage``, fed by the loop from the workflow
# session for workflow modes and from the per-step counter for chat/action
# modes) and applies ONE threshold/cap pair to every mode kind. The cases the
# chat-stuck block covered map onto the unified guard as follows:
#
# - routable chat mode past the cap → ``test_stage_cap_fires_for_chat_mode_too``
# - action mode without escape → ``test_stage_cap_forces_finish_for_action_mode_without_escape``
# - below the cap → ``test_stage_threshold_emits_continue_with_nudge``
# - single-mode chat turn (no transitions) → the counter keeps rising, so the
#   unified cap bounds it at ``stage_nudge_cap``; the global iteration cap is
#   the ultimate ceiling either way.
#
# Nothing here needs a chat-specific settings pair; ``chat_stuck_mode_cap`` /
# ``chat_stuck_mode_threshold`` are deleted from ``kernel.config.Settings``.


# ── Operator signals (pause / stop / terminate) ─────────────────────────────
#
# The operator is the highest authority — branch 0 fires before the router
# decision and every cap path. Pause and stop share one checkpoint
# (``lifecycle_state="paused"``); the difference is the transport prompt
# behavior, not the action. Terminate marks the workflow complete.


def test_operator_pause_overrides_router_finish() -> None:
    """A router finish is lower priority than an operator pause."""
    action = resolve_lifecycle_action(
        state=_State(operator_signal="pause"),
        mode_id="default",
        mode_yield_policy=YieldPolicy(),
        settings=_settings(),
        router_decision=_Router(finish="done"),
    )
    assert isinstance(action, PauseAction)


def test_operator_stop_overrides_router_finish() -> None:
    action = resolve_lifecycle_action(
        state=_State(operator_signal="stop"),
        mode_id="default",
        mode_yield_policy=YieldPolicy(),
        settings=_settings(),
        router_decision=_Router(finish="done"),
    )
    assert isinstance(action, StopAction)


def test_operator_terminate_overrides_router_finish() -> None:
    action = resolve_lifecycle_action(
        state=_State(operator_signal="terminate"),
        mode_id="default",
        mode_yield_policy=YieldPolicy(),
        settings=_settings(),
        router_decision=_Router(finish="done"),
    )
    assert isinstance(action, TerminateAction)


def test_operator_pause_overrides_iteration_cap() -> None:
    """At the iteration cap, an operator pause wins over the cap's pause."""
    action = resolve_lifecycle_action(
        state=_State(iteration=15, operator_signal="pause"),
        mode_id="default",
        mode_yield_policy=YieldPolicy(allowed=True),
        settings=_settings(max_tool_iterations=15),
    )
    assert isinstance(action, PauseAction)
    assert action.reason == "operator pause"


def test_operator_stop_overrides_recovery_stall() -> None:
    """An operator stop wins over a recovery-stall yield."""
    action = resolve_lifecycle_action(
        state=_State(operator_signal="stop"),
        mode_id="search",
        mode_yield_policy=YieldPolicy(allowed=True),
        settings=_settings(),
        recovery_records=[
            _record(
                state="stalled",
                retry_policy="confirm",
                last_error=RECOVERY_APPROVAL_PENDING_ERROR,
                step_id="s1",
            )
        ],
    )
    assert isinstance(action, StopAction)


def test_operator_terminate_overrides_handoff_backstop() -> None:
    action = resolve_lifecycle_action(
        state=_State(handoff_count=6, operator_signal="terminate"),
        mode_id="default",
        mode_yield_policy=YieldPolicy(),
        settings=_settings(max_handoffs_per_turn=5),
    )
    assert isinstance(action, TerminateAction)


def test_no_operator_signal_is_noop() -> None:
    """Empty operator_signal means no operator exit — the resolver ignores it."""
    action = resolve_lifecycle_action(
        state=_State(operator_signal=""),
        mode_id="default",
        mode_yield_policy=YieldPolicy(),
        settings=_settings(),
    )
    assert isinstance(action, ContinueAction)


# ── Write-rescue forced transition (before the cap) ────────────────────────


def test_write_rescue_fires_at_cap() -> None:
    """A write-contract mode hitting the stuck-in-stage cap with no receipt
    forces a transition to the configured rescue mode."""
    action = resolve_lifecycle_action(
        state=_State(
            turns_in_stage=6,  # stage_nudge_cap=6
            action_tool_ids=frozenset({"markdown_forge_document"}),
            has_write_receipt=False,
        ),
        mode_id="researcher",
        mode_yield_policy=YieldPolicy(),
        settings=_settings(stage_nudge_cap=6, write_rescue_mode="explainer"),
    )
    assert isinstance(action, TransitionAction)
    assert action.target_mode_id == "explainer"
    assert action.forced is True


def test_write_rescue_targets_planned_writer_when_available() -> None:
    """When the workflow position has a remaining write step, the
    rescue targets that step's mode instead of the fallback."""
    action = resolve_lifecycle_action(
        state=_State(
            turns_in_stage=6,
            action_tool_ids=frozenset({"coding_apply_patch"}),
            has_write_receipt=False,
        ),
        mode_id="researcher",
        mode_yield_policy=YieldPolicy(),
        settings=_settings(stage_nudge_cap=6, write_rescue_mode="explainer"),
        workflow_position=_Instance(
            current_step_id="s1",
            current_mode_id="researcher",
            declared_steps=(
                _Step("s1", "researcher"),
                _Step("s2", "executor", write_capability="coding_apply_patch"),
            ),
        ),
    )
    assert isinstance(action, TransitionAction)
    assert action.target_mode_id == "executor"
    assert action.forced is True


def test_write_rescue_skipped_when_receipt_present() -> None:
    """A mode that has already produced a write receipt does not trigger
    the rescue — the turn can complete normally."""
    action = resolve_lifecycle_action(
        state=_State(
            turns_in_stage=6,
            action_tool_ids=frozenset({"markdown_forge_document"}),
            has_write_receipt=True,
        ),
        mode_id="researcher",
        mode_yield_policy=YieldPolicy(),
        settings=_settings(stage_nudge_cap=6, write_rescue_mode="explainer"),
    )
    # Should NOT be a forced transition — falls through to the cap path.
    assert not isinstance(action, TransitionAction)


def test_write_rescue_skipped_when_no_action_tools() -> None:
    """A mode with no declared action tools (read-only) does not trigger
    the rescue."""
    action = resolve_lifecycle_action(
        state=_State(
            turns_in_stage=6,
            action_tool_ids=frozenset(),
            has_write_receipt=False,
        ),
        mode_id="default",
        mode_yield_policy=YieldPolicy(),
        settings=_settings(stage_nudge_cap=6, write_rescue_mode="explainer"),
    )
    # Should NOT be a forced transition — falls through to the cap path.
    assert not isinstance(action, TransitionAction)


def test_write_rescue_disabled_when_rescue_mode_empty() -> None:
    """Setting ``write_rescue_mode=""`` disables the forced transition."""
    action = resolve_lifecycle_action(
        state=_State(
            turns_in_stage=6,
            action_tool_ids=frozenset({"markdown_forge_document"}),
            has_write_receipt=False,
        ),
        mode_id="researcher",
        mode_yield_policy=YieldPolicy(),
        settings=_settings(stage_nudge_cap=6, write_rescue_mode=""),
    )
    # Should NOT be a forced transition — falls through to the cap path.
    assert not isinstance(action, TransitionAction)


def test_write_rescue_skipped_before_cap() -> None:
    """The rescue only fires at ``stage_nudge_cap``, not earlier."""
    action = resolve_lifecycle_action(
        state=_State(
            turns_in_stage=5,  # cap=6, so 5 is before the cap
            action_tool_ids=frozenset({"markdown_forge_document"}),
            has_write_receipt=False,
        ),
        mode_id="researcher",
        mode_yield_policy=YieldPolicy(),
        settings=_settings(stage_nudge_cap=6, write_rescue_mode="explainer"),
    )
    # Should NOT be a forced transition — the pre-cap nudge gets its chance.
    assert not isinstance(action, TransitionAction)


def test_write_rescue_no_self_handoff_when_rescue_equals_current_mode() -> None:
    """When the configured rescue mode equals the mode already running,
    the resolver must NOT synthesize a self-handoff — that would reset
    ``turns_in_stage`` and re-fire this block every iteration, producing
    an unbounded zero-token self-handoff loop. Fall through to the normal
    stuck-in-stage cap path instead. Regression for the 2026-08-14
    ``explainer → explainer`` hyperloop."""
    action = resolve_lifecycle_action(
        state=_State(
            turns_in_stage=6,
            action_tool_ids=frozenset({"markdown_forge_document"}),
            has_write_receipt=False,
        ),
        mode_id="explainer",
        mode_yield_policy=YieldPolicy(),
        settings=_settings(stage_nudge_cap=6, write_rescue_mode="explainer"),
    )
    # Must NOT be a forced self-transition. Falls through to the cap path
    # (pause or terminate depending on yield policy).
    assert not isinstance(action, TransitionAction)

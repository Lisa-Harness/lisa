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
from kernel.turn_phases._lifecycle import (
    ContinueAction,
    FinishAction,
    LifecycleResolverSettings,
    TransitionAction,
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
    forced_fork_target: str = ""
    non_fork_tool_dispatches: int = 0
    forced_plan_target: str = ""
    non_plan_tool_dispatches: int = 0
    handoff_count: int = 0


@dataclass
class _Step:
    step_id: str
    mode_id: str


@dataclass
class _Instance:
    current_step_id: str = "s1"
    current_mode_id: str = "search"
    declared_steps: tuple[_Step, ...] = (
        _Step("s1", "search"),
        _Step("s2", "coding"),
    )
    completed_steps: tuple[str, ...] = ()


@dataclass
class _Router:
    finish: str = ""
    transition_to: str = ""


@dataclass
class _Record:
    state: str = "running"
    blocks_workflow: bool = False
    retry_policy: str = "safe"
    target_id: str = "web_search"
    target_kind: str = "pack_tool"
    last_error: str = ""
    step_id: str = ""


# ── Default: continue ──────────────────────────────────────────────────────


def test_default_is_continue_with_no_nudge() -> None:
    action = resolve_lifecycle_action(
        state=_State(),
        mode_id="default",
        mode_yield_policy=YieldPolicy(),
        settings=LifecycleResolverSettings(),
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
        settings=LifecycleResolverSettings(),
        router_decision=_Router(finish="done"),
    )
    assert isinstance(action, FinishAction)
    assert action.strip_tools is True


def test_router_transition_to_returns_transition() -> None:
    action = resolve_lifecycle_action(
        state=_State(),
        mode_id="default",
        mode_yield_policy=YieldPolicy(),
        settings=LifecycleResolverSettings(),
        router_decision=_Router(transition_to="coding"),
    )
    assert isinstance(action, TransitionAction)
    assert action.target_mode_id == "coding"
    assert action.forced is False


# ── Recovery stall blocks workflow ──────────────────────────────────────────


def test_stalled_blocking_record_forces_yield() -> None:
    action = resolve_lifecycle_action(
        state=_State(),
        mode_id="search",
        mode_yield_policy=YieldPolicy(allowed=True),
        settings=LifecycleResolverSettings(),
        recovery_records=[
            _Record(
                state="stalled",
                blocks_workflow=True,
                retry_policy="confirm",
                last_error="web_search timed out",
                step_id="s1",
            )
        ],
    )
    assert isinstance(action, YieldAction)
    assert "blocks workflow" in action.context
    assert "web_search timed out" in action.question


def test_safe_retry_record_does_not_block() -> None:
    """A stalled record with retry_policy='safe' is auto-retried, not yielded."""
    action = resolve_lifecycle_action(
        state=_State(),
        mode_id="search",
        mode_yield_policy=YieldPolicy(),
        settings=LifecycleResolverSettings(),
        recovery_records=[_Record(state="stalled", blocks_workflow=True, retry_policy="safe")],
    )
    assert isinstance(action, ContinueAction)


# ── Global iteration cap ────────────────────────────────────────────────────


def test_iteration_cap_forces_yield_for_chat_mode() -> None:
    action = resolve_lifecycle_action(
        state=_State(iteration=15),
        mode_id="default",
        mode_yield_policy=YieldPolicy(allowed=True),
        settings=LifecycleResolverSettings(max_tool_iterations=15),
    )
    assert isinstance(action, YieldAction)
    assert "iteration budget" in action.question


def test_iteration_cap_forces_finish_for_action_mode_without_escape() -> None:
    """An action mode that forbids yield and names no fallback finishes."""
    action = resolve_lifecycle_action(
        state=_State(iteration=15),
        mode_id="coding",
        mode_yield_policy=YieldPolicy(allowed=False, fallback_mode=""),
        settings=LifecycleResolverSettings(max_tool_iterations=15),
    )
    assert isinstance(action, FinishAction)
    assert action.strip_tools is True


def test_iteration_cap_uses_policy_default_question() -> None:
    action = resolve_lifecycle_action(
        state=_State(iteration=15),
        mode_id="default",
        mode_yield_policy=YieldPolicy(allowed=True, default_question="What should I focus on?"),
        settings=LifecycleResolverSettings(max_tool_iterations=15),
    )
    assert isinstance(action, YieldAction)
    assert action.question == "What should I focus on?"


# ── Per-step budget exhausted ───────────────────────────────────────────────


def test_per_step_budget_exhausted_forces_transition_to_next_step() -> None:
    action = resolve_lifecycle_action(
        state=_State(step_iteration=9, step_tool_calls=0),
        mode_id="search",
        mode_yield_policy=YieldPolicy(),
        settings=LifecycleResolverSettings(per_step_max_iterations=8),
        workflow_instance=_Instance(),
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
        settings=LifecycleResolverSettings(per_step_max_iterations=8),
        workflow_instance=_Instance(
            current_step_id="s2",
            current_mode_id="coding",
            declared_steps=(_Step("s1", "search"), _Step("s2", "coding")),
            completed_steps=("s1", "s2"),
        ),
    )
    # No next step → not a transition; falls through to default continue
    # (the global cap would catch it if iteration were high).
    assert isinstance(action, ContinueAction)


# ── Stuck-in-stage cap ──────────────────────────────────────────────────────


def test_stage_cap_forces_yield_for_workflow_mode() -> None:
    action = resolve_lifecycle_action(
        state=_State(turns_in_stage=6),
        mode_id="ponder",
        mode_yield_policy=YieldPolicy(allowed=True),
        settings=LifecycleResolverSettings(stage_nudge_cap=6),
        is_workflow=True,
    )
    assert isinstance(action, YieldAction)
    assert "6 turns" in action.question


def test_stage_cap_forces_finish_for_action_mode_without_escape() -> None:
    action = resolve_lifecycle_action(
        state=_State(turns_in_stage=6),
        mode_id="coding",
        mode_yield_policy=YieldPolicy(allowed=False, fallback_mode=""),
        settings=LifecycleResolverSettings(stage_nudge_cap=6),
        is_workflow=True,
    )
    assert isinstance(action, FinishAction)


def test_stage_cap_ignored_for_non_workflow_mode() -> None:
    """Chat modes don't track turns_in_stage; the cap only fires for workflows."""
    action = resolve_lifecycle_action(
        state=_State(turns_in_stage=6),
        mode_id="default",
        mode_yield_policy=YieldPolicy(),
        settings=LifecycleResolverSettings(stage_nudge_cap=6),
        is_workflow=False,
    )
    assert isinstance(action, ContinueAction)


# ── Stuck-in-stage threshold (soft nudge) ───────────────────────────────────


def test_stage_threshold_emits_continue_with_nudge() -> None:
    action = resolve_lifecycle_action(
        state=_State(turns_in_stage=4),
        mode_id="ponder",
        mode_yield_policy=YieldPolicy(),
        settings=LifecycleResolverSettings(stage_nudge_threshold=4, stage_nudge_cap=6),
        is_workflow=True,
    )
    assert isinstance(action, ContinueAction)
    assert action.nudge != ""
    assert "transition_stage" in action.nudge
    assert "yield" in action.nudge


# ── Yield target resolution ─────────────────────────────────────────────────


def test_yield_uses_policy_fallback_mode() -> None:
    action = resolve_lifecycle_action(
        state=_State(iteration=15),
        mode_id="coding",
        mode_yield_policy=YieldPolicy(allowed=False, fallback_mode="discuss"),
        settings=LifecycleResolverSettings(max_tool_iterations=15),
    )
    assert isinstance(action, YieldAction)
    assert action.target_mode_id == "discuss"


def test_yield_in_place_when_policy_allows_inline() -> None:
    """A chat mode with allowed=True yields in place (empty target)."""
    action = resolve_lifecycle_action(
        state=_State(iteration=15),
        mode_id="default",
        mode_yield_policy=YieldPolicy(allowed=True),
        settings=LifecycleResolverSettings(max_tool_iterations=15),
    )
    assert isinstance(action, YieldAction)
    assert action.target_mode_id == ""


# ── Phase 4: Handoff-loop backstop ───────────────────────────────────────────


def test_handoff_loop_backstop_forces_finish() -> None:
    """When handoff_count exceeds max_handoffs_per_turn, finish immediately."""
    action = resolve_lifecycle_action(
        state=_State(handoff_count=6),
        mode_id="default",
        mode_yield_policy=YieldPolicy(),
        settings=LifecycleResolverSettings(max_handoffs_per_turn=5),
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
        settings=LifecycleResolverSettings(max_handoffs_per_turn=5),
    )
    assert isinstance(action, ContinueAction)


# ── Phase 4: Forced-fork enforcer ───────────────────────────────────────────


def test_forced_fork_enforcer_fires_at_cap() -> None:
    """When the forced-fork target is set and non-fork dispatches hit the cap,
    force a TransitionAction to the fork target."""
    action = resolve_lifecycle_action(
        state=_State(forced_fork_target="fork:search,markdown", non_fork_tool_dispatches=3),
        mode_id="default",
        mode_yield_policy=YieldPolicy(),
        settings=LifecycleResolverSettings(forced_fork_non_fork_tool_cap=3),
    )
    assert isinstance(action, TransitionAction)
    assert action.forced
    assert action.target_mode_id == "fork:search,markdown"


def test_forced_fork_enforcer_not_triggered_below_cap() -> None:
    """Below the cap, the enforcer does not fire."""
    action = resolve_lifecycle_action(
        state=_State(forced_fork_target="fork:search,markdown", non_fork_tool_dispatches=2),
        mode_id="default",
        mode_yield_policy=YieldPolicy(),
        settings=LifecycleResolverSettings(forced_fork_non_fork_tool_cap=3),
    )
    assert isinstance(action, ContinueAction)


def test_forced_fork_enforcer_not_triggered_without_target() -> None:
    """No forced-fork target means no enforcer."""
    action = resolve_lifecycle_action(
        state=_State(forced_fork_target="", non_fork_tool_dispatches=10),
        mode_id="default",
        mode_yield_policy=YieldPolicy(),
        settings=LifecycleResolverSettings(forced_fork_non_fork_tool_cap=3),
    )
    assert isinstance(action, ContinueAction)


# ── Phase 4: Forced-plan enforcer ───────────────────────────────────────────


def test_forced_plan_enforcer_fires_at_cap() -> None:
    """When the forced-plan target is set and non-plan dispatches hit the cap,
    force a TransitionAction to plan."""
    action = resolve_lifecycle_action(
        state=_State(forced_plan_target="plan", non_plan_tool_dispatches=3),
        mode_id="default",
        mode_yield_policy=YieldPolicy(),
        settings=LifecycleResolverSettings(forced_plan_non_transition_tool_cap=3),
    )
    assert isinstance(action, TransitionAction)
    assert action.forced
    assert action.target_mode_id == "plan"


def test_forced_plan_enforcer_not_triggered_below_cap() -> None:
    """Below the cap, the enforcer does not fire."""
    action = resolve_lifecycle_action(
        state=_State(forced_plan_target="plan", non_plan_tool_dispatches=2),
        mode_id="default",
        mode_yield_policy=YieldPolicy(),
        settings=LifecycleResolverSettings(forced_plan_non_transition_tool_cap=3),
    )
    assert isinstance(action, ContinueAction)


# ── Phase 4: Resource-budget exhaustion ─────────────────────────────────────


def test_budget_exhausted_forces_yield_for_chat_mode() -> None:
    """When budget_exhausted is True and the mode allows yields, force a yield."""
    action = resolve_lifecycle_action(
        state=_State(budget_exhausted=True),
        mode_id="default",
        mode_yield_policy=YieldPolicy(allowed=True),
        settings=LifecycleResolverSettings(),
    )
    assert isinstance(action, YieldAction)
    assert "budget" in action.context


def test_budget_exhausted_forces_finish_for_action_mode() -> None:
    """When budget_exhausted is True and the mode forbids yields, finish."""
    action = resolve_lifecycle_action(
        state=_State(budget_exhausted=True),
        mode_id="search",
        mode_yield_policy=YieldPolicy(allowed=False),
        settings=LifecycleResolverSettings(),
    )
    assert isinstance(action, FinishAction)
    assert action.strip_tools


def test_budget_exhausted_not_triggered_when_false() -> None:
    """When budget_exhausted is False, the branch is a no-op."""
    action = resolve_lifecycle_action(
        state=_State(budget_exhausted=False),
        mode_id="default",
        mode_yield_policy=YieldPolicy(),
        settings=LifecycleResolverSettings(),
    )
    assert isinstance(action, ContinueAction)

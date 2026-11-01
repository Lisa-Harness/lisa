"""Loop-control union ordering and exhaustiveness (F2 of the 2026-10-30
structural review).

The turn loop's guard sequence is load-bearing: a single iteration passes
through the lifecycle break, the empty-completion break/continue, the
repeated-batch stop/refuse, and the terminal-dispatch-failure break, in that
order. Nothing states which guard wins when two apply in one iteration — the
ordering is the accumulated result of incident fixes, tested locally but
never at the composition level. These tests pin the union's variant set and
the loop's guard order so the precedence is auditable instead of
archaeological.
"""

from __future__ import annotations

import inspect
import typing

from kernel.turn_phases._loop_control import (
    BreakIteration,
    ContinueIteration,
    DispatchToolBatch,
    LoopControl,
    RefuseToolBatch,
    StopLoop,
)


def test_union_covers_exactly_five_variants() -> None:
    """The union is a closed five-variant sum type.

    A new guard author must pick one of these; a removed variant must be
    deleted from the union, not left as dead weight. ``get_args`` on the
    union alias is the single source of truth for the variant set.
    """

    variants = set(typing.get_args(LoopControl))
    assert variants == {
        ContinueIteration,
        BreakIteration,
        DispatchToolBatch,
        RefuseToolBatch,
        StopLoop,
    }


def test_loop_guard_order_is_stable() -> None:
    """The while-loop's decision points fire in a fixed, documented order.

    Source-lock test (the provider-quota pattern): the loop body must apply
    the lifecycle action before the empty-completion resolution, which must
    precede the repeated-batch guard, which must precede the dispatch-loop
    action. Reordering any pair changes which guard wins when two fire in
    one iteration, so the order is pinned here.
    """

    from kernel.turn_runner import _run_turn_inner

    source = inspect.getsource(_run_turn_inner)
    lifecycle_pos = source.index("_apply_iteration_lifecycle_action(")
    empty_pos = source.index("_resolve_empty_completion_for_dispatch(")
    repeat_pos = source.index("_guard_repeated_tool_batch(")
    dispatch_pos = source.index("_resolve_dispatch_loop_action(")

    assert lifecycle_pos < empty_pos < repeat_pos < dispatch_pos


def test_full_guard_sequence_is_pinned() -> None:
    """F8: all six decision points fire in one fixed order, precondition first.

    The review's F8 list plus the completion-``None`` break the review did
    not enumerate: the while-precondition (``_next_loop_action``) must fire
    before the lifecycle break, which must precede the completion-``None``
    break, the empty-completion resolution, the repeated-batch guard, and
    the dispatch-loop action. The precondition is now a named pure function
    (not an anonymous conjunction in the ``while`` header), so its position
    is pinnable too.
    """

    from kernel.turn_runner import _run_turn_inner

    source = inspect.getsource(_run_turn_inner)
    precondition_pos = source.index("_next_loop_action(")
    lifecycle_pos = source.index("_apply_iteration_lifecycle_action(")
    completion_none_pos = source.index("if completion is None:")
    empty_pos = source.index("_resolve_empty_completion_for_dispatch(")
    repeat_pos = source.index("_guard_repeated_tool_batch(")
    dispatch_pos = source.index("_resolve_dispatch_loop_action(")

    assert (
        precondition_pos
        < lifecycle_pos
        < completion_none_pos
        < empty_pos
        < repeat_pos
        < dispatch_pos
    )


def test_next_loop_action_is_pure_and_ordered() -> None:
    """F8: the while-precondition is one ordered pure function.

    Each condition is a branch returning ``StopLoop`` with its reason; the
    fallthrough is ``ContinueIteration``. The branch order mirrors the
    original conjunction's clause order, so the first matching condition is
    the same one that would have short-circuited the ``and`` chain.
    """

    from kernel.turn_runner import _next_loop_action

    def _state(**overrides: object) -> object:
        base = {
            "iteration": 1,
            "yielded": False,
            "wall_budget_expired": False,
            "stream_stalled": False,
            "provider_quota_exhausted": False,
            "entry_failed": False,
            "plan_failed": False,
            "handoff_count": 0,
            "fork_results": None,
        }
        base.update(overrides)
        return type("S", (), base)()

    # Fallthrough: every condition clear → proceed with the iteration.
    assert isinstance(
        _next_loop_action(state=_state(), max_tool_iterations=50, max_handoffs_per_turn=8),
        ContinueIteration,
    )

    # Each condition alone stops the loop, with its reason.
    cases = [
        ({"iteration": 50}, "iteration cap"),
        ({"yielded": True}, "yielded"),
        ({"wall_budget_expired": True}, "wall budget"),
        ({"stream_stalled": True}, "stream stalled"),
        ({"provider_quota_exhausted": True}, "provider quota"),
        ({"entry_failed": True}, "terminal dispatch failure"),
        ({"plan_failed": True}, "terminal dispatch failure"),
        ({"handoff_count": 9}, "terminal dispatch failure"),
        ({"fork_results": [type("R", (), {"status": "error"})()]}, "terminal dispatch failure"),
    ]
    for overrides, expected_reason in cases:
        action = _next_loop_action(
            state=_state(**overrides), max_tool_iterations=50, max_handoffs_per_turn=8
        )
        assert isinstance(action, StopLoop), f"{overrides} should stop the loop, got {action!r}"
        assert expected_reason in action.reason, (
            f"{overrides} reason {action.reason!r} should mention {expected_reason!r}"
        )

    # Precedence: the first condition in the chain wins when several apply.
    both = _next_loop_action(
        state=_state(iteration=50, yielded=True),
        max_tool_iterations=50,
        max_handoffs_per_turn=8,
    )
    assert isinstance(both, StopLoop)
    assert both.reason == "iteration cap reached (iteration=50)"


def test_payload_variants_carry_their_data() -> None:
    """RefuseToolBatch and StopLoop carry their diagnostic payloads.

    The refusal text and the stop reason are the data a consumer needs to
    log or surface; the no-payload variants stay empty so a consumer cannot
    accidentally depend on data that is not there.
    """

    refusal = RefuseToolBatch(refusal="Not executed: already made 5 times")
    assert refusal.refusal == "Not executed: already made 5 times"

    stop = StopLoop(reason="identical tool batch re-issued past its cap")
    assert stop.reason == "identical tool batch re-issued past its cap"

    assert isinstance(ContinueIteration(), ContinueIteration)
    assert isinstance(BreakIteration(), BreakIteration)
    assert isinstance(DispatchToolBatch(), DispatchToolBatch)

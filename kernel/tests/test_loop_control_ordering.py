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

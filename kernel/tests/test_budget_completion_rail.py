"""Tests for ``budget_forces_final`` completion-based token accounting.

Covers the ``max_completion_tokens`` rail: the per-turn budget measures
actual model work (output tokens) rather than cumulative context re-send,
so a turn with a large prompt re-accumulation does not exhaust the budget
while the model has produced little real output.
"""

from __future__ import annotations

import time

from kernel.models import TurnBudget
from kernel.turn_phases._turn_state import _TurnState, budget_forces_final


def _state(**overrides: int) -> _TurnState:
    defaults: dict[str, int] = {
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "total_tool_calls": 0,
    }
    defaults.update(overrides)
    return _TurnState(**defaults)


def test_completion_rail_fires_when_completion_exceeds_cap() -> None:
    """A high completion count against a low completion cap forces a yield."""
    budget = TurnBudget(max_completion_tokens=1000, max_total_tokens=0, max_wall_ms=0)
    state = _state(total_completion_tokens=1500)

    assert budget_forces_final(state, budget, time.monotonic()) is True


def test_completion_rail_does_not_fire_on_cumulative_prompt_growth() -> None:
    """A huge cumulative total with small completion does NOT trip the
    completion rail — context re-send is not treated as model work."""
    budget = TurnBudget(max_completion_tokens=1000, max_total_tokens=0, max_wall_ms=0)
    state = _state(total_completion_tokens=100, total_tokens=5_000_000)

    assert budget_forces_final(state, budget, time.monotonic()) is False


def test_unlimited_completion_never_fires() -> None:
    """max_completion_tokens=0 means unlimited."""
    budget = TurnBudget(max_completion_tokens=0, max_total_tokens=0, max_wall_ms=0)
    state = _state(total_completion_tokens=99_999_999)

    assert budget_forces_final(state, budget, time.monotonic()) is False


def test_cumulative_rail_still_fires_as_fallback() -> None:
    """The cumulative ``max_total_tokens`` rail still works for hosts that
    want a whole-turn envelope, independent of the completion rail."""
    budget = TurnBudget(max_completion_tokens=0, max_total_tokens=1000, max_wall_ms=0)
    state = _state(total_completion_tokens=100, total_tokens=1500)

    assert budget_forces_final(state, budget, time.monotonic()) is True


def test_none_budget_never_fires() -> None:
    """No budget wired means no forced yield from the token rails."""
    assert (
        budget_forces_final(_state(total_completion_tokens=10_000), None, time.monotonic()) is False
    )

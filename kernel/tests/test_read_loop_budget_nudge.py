"""Tests for the budget-aware trigger of the one-shot read-loop nudge."""

from __future__ import annotations

import logging

from kernel.models import EffectReceipt
from kernel.turn_phases._turn_state import _TurnState
from kernel.turn_runner import _per_iteration_nudges


def _read_only_state(total_tool_calls: int) -> _TurnState:
    state = _TurnState(total_tool_calls=total_tool_calls)
    state.effect_receipts = [EffectReceipt(effect="read", capability_id="some_read")]
    return state


def _read_loop_nudge(state: _TurnState, max_turn_tool_calls: int) -> str:
    _, read_loop = _per_iteration_nudges(
        state=state,
        mode_id="worker",
        max_tool_iterations=50,
        turns_in_stage=1,
        pending_yield=None,
        log=logging.getLogger(__name__),
        max_turn_tool_calls=max_turn_tool_calls,
    )
    return read_loop


def test_read_loop_nudge_fires_near_tool_call_budget() -> None:
    state = _read_only_state(total_tool_calls=7)

    assert _read_loop_nudge(state, max_turn_tool_calls=10)
    assert state.read_loop_nudge_fired
    assert _read_loop_nudge(state, max_turn_tool_calls=10) == ""


def test_read_loop_nudge_waits_while_budget_has_room() -> None:
    assert _read_loop_nudge(_read_only_state(total_tool_calls=2), max_turn_tool_calls=10) == ""


def test_read_loop_nudge_ignores_unlimited_budget() -> None:
    assert _read_loop_nudge(_read_only_state(total_tool_calls=99), max_turn_tool_calls=0) == ""

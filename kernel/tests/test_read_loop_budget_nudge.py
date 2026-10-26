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


class _ReadToolDef:
    effect = "read"


class _WriteToolDef:
    effect = "write"


def _track(state: _TurnState, name: str, args: dict, tool_def: object) -> None:
    from kernel.turn_phases._dispatch import _track_retrieval_call

    _track_retrieval_call(state, name, args, tool_def=tool_def)


def test_retrieval_tracking_counts_pack_read_tools_by_effect() -> None:
    """Pack tools declaring ``effect: read`` count as retrieval calls.

    Reproduction of the 2026-10-26 loop: ``coding_read_file`` /
    ``coding_search_code`` re-read one file window at shrinking ranges;
    only ``thread_*`` tools were tracked, so the read-loop nudge's
    repeat-count basis stayed 0 and the nudge never reported the
    repeats. Membership by the catalog effect covers every read-declaring
    pack tool without enumerating names.
    """
    state = _TurnState()
    _track(state, "coding_read_file", {"path": "a.py", "start_line": 10}, _ReadToolDef())
    _track(state, "coding_search_code", {"query": "x"}, _ReadToolDef())
    _track(state, "coding_edit_files", {"path": "a.py"}, _WriteToolDef())

    assert state.retrieval_calls == 2
    assert len(state.retrieval_distinct_args) == 2


def test_retrieval_tracking_counts_repeated_args_as_repeats() -> None:
    """A re-read with identical arguments registers as a repeat, not a
    distinct call — the cheap signal the nudge reports."""
    state = _TurnState()
    args = {"path": "turn_hooks_builder.py", "start_line": 1470, "end_line": 1482}
    for _ in range(3):
        _track(state, "coding_read_file", dict(args), _ReadToolDef())

    assert state.retrieval_calls == 3
    assert len(state.retrieval_distinct_args) == 1


def test_retrieval_tracking_falls_back_to_suffix_for_uncataloged_tools() -> None:
    """Unknown tools keep the thread-archive suffix membership.

    MCP thread tools can arrive without a catalog definition; the
    suffix list stays as the fallback so they still count.
    """
    state = _TurnState()
    _track(state, "thread.thread_search", {"query": "x"}, None)
    _track(state, "fetch_page", {"url": "https://x"}, None)

    assert state.retrieval_calls == 1
    assert len(state.retrieval_distinct_args) == 1

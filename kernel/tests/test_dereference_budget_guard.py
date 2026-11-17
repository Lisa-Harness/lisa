"""Tests for the per-turn dereference budget guard."""

from __future__ import annotations

import pytest
from kernel.config import settings
from kernel.turn_phases._turn_state import _TurnState
from kernel.turn_runner import _deref_budget_refusal


class _ReadToolDef:
    effect = "read"


def _ref(char: str) -> str:
    return "r" + char * 64


def test_deref_guard_allows_call_at_budget_edge(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "max_dereference_calls_per_turn", 5)
    monkeypatch.setattr(settings, "max_dereference_repeats_per_turn", 1)
    state = _TurnState(dereference_calls=4, dereference_distinct_refs={_ref("a"), _ref("b")})

    refusal = _deref_budget_refusal(
        state=state,
        tool_name="thread.thread_read_content",
        tool_arguments={"ref": _ref("c"), "offset_chars": 0, "limit_chars": 4000},
        tool_def=_ReadToolDef(),
    )

    assert refusal == ""
    assert not state.intra_turn_compaction.finish_now


def test_deref_guard_refuses_repeated_expansion_past_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "max_dereference_calls_per_turn", 5)
    monkeypatch.setattr(settings, "max_dereference_repeats_per_turn", 1)
    state = _TurnState(
        dereference_calls=5,
        dereference_distinct_refs={_ref("a"), _ref("b")},
    )

    refusal = _deref_budget_refusal(
        state=state,
        tool_name="thread.thread_read_content",
        tool_arguments={"ref": _ref("a"), "offset_chars": 4000, "limit_chars": 4000},
        tool_def=_ReadToolDef(),
    )

    assert "Not executed: dereference budget reached" in refusal
    assert state.intra_turn_compaction.finish_now


def test_deref_guard_allows_distinct_refs_even_over_call_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "max_dereference_calls_per_turn", 5)
    monkeypatch.setattr(settings, "max_dereference_repeats_per_turn", 1)
    state = _TurnState(
        dereference_calls=5,
        dereference_distinct_refs={_ref("a"), _ref("b"), _ref("c"), _ref("d"), _ref("e")},
    )

    refusal = _deref_budget_refusal(
        state=state,
        tool_name="thread.thread_read_content",
        tool_arguments={"ref": _ref("f"), "offset_chars": 0, "limit_chars": 4000},
        tool_def=_ReadToolDef(),
    )

    assert refusal == ""
    assert not state.intra_turn_compaction.finish_now


def test_deref_guard_ignores_non_ref_read_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "max_dereference_calls_per_turn", 1)
    monkeypatch.setattr(settings, "max_dereference_repeats_per_turn", 0)
    state = _TurnState(dereference_calls=10, dereference_distinct_refs={_ref("a")})

    refusal = _deref_budget_refusal(
        state=state,
        tool_name="coding_read_file",
        tool_arguments={"path": "a.py", "start_line": 1, "end_line": 20},
        tool_def=_ReadToolDef(),
    )

    assert refusal == ""
    assert not state.intra_turn_compaction.finish_now


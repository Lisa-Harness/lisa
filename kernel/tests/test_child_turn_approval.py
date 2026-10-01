"""Child-turn classification for delegate/fork sub-turns."""

from __future__ import annotations

from types import SimpleNamespace

from kernel.turn_phases._child_turn import classify_child_turn
from kernel.turn_runner._result import TurnResult


def test_child_paused_for_approval_is_blocked_not_complete() -> None:
    """A delegate child that paused on an approval must not report ``ok``.

    The 2026-10-01 e2e run delegated a fix to executor; the child paused on
    ``coding_create_worktree`` and was classified complete, so the parent
    yielded the raw "/approve <id>" text instead of handing off.
    """
    child = TurnResult(
        response_text="Approval required for create an isolated coding worktree.",
        pending_approval=SimpleNamespace(target_id="coding_create_worktree"),  # type: ignore[arg-type]
    )

    outcome = classify_child_turn(
        child_result=child,
        mode_id="executor",
        elapsed_ms=10,
        action_tool_ids=set(),
    )

    assert outcome.status == "blocked"
    assert "coding_create_worktree" in outcome.error_message
    assert "transition_stage" in outcome.error_message


def test_child_without_pending_approval_completes() -> None:
    outcome = classify_child_turn(
        child_result=TurnResult(response_text="done"),
        mode_id="executor",
        elapsed_ms=10,
        action_tool_ids=set(),
    )

    assert outcome.status == "complete"

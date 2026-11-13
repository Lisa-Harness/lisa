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


def test_reasoning_tail_child_is_interrupted_not_complete() -> None:
    """A reasoning-overflow child is interrupted, not a verified ``ok``.

    Regression for the 2026-11-13 turn 1258 incident: two coding_execute
    fork children burned their completion cap on reasoning (no visible
    answer) and their >150-char reasoning-tail caveat counted toward dense
    content, so ``classify_child_turn`` reported both as ``complete``.
    The parent's digest then claimed two successes that the model never
    finished. The reasoning-tail quote still rides ``output_text`` so the
    parent sees what the child was thinking, but the status says the work
    was cut short.
    """
    child = TurnResult(
        response_text="*The model produced no visible answer — reasoning consumed the "
        "completion budget before any visible text — only internal reasoning. "
        "Tail of its thinking, not a response:* > deliberating over the diff hunks",
        content_source="reasoning_tail",
    )

    outcome = classify_child_turn(
        child_result=child,
        mode_id="coding_execute",
        elapsed_ms=10,
        action_tool_ids={"coding_apply_patch", "coding_commit"},
    )

    assert outcome.status == "interrupted"
    assert "reasoning" in outcome.error_message.lower()
    # The tail quote still rides the outcome for the parent's digest.
    assert "deliberating over the diff hunks" in outcome.output_text


def test_reasoning_tail_child_with_write_receipt_still_completes() -> None:
    """A verified write beats the reasoning-tail gate.

    A reasoning model can land its write and then keep thinking past the
    visible-answer window — the receipts prove the work landed, so the
    status is ``complete`` regardless of what the surfaced text looks like.
    """
    from kernel.models import EffectReceipt

    child = TurnResult(
        response_text="(reasoning tail quote — no visible answer)",
        content_source="reasoning_tail",
        effect_receipts=[
            EffectReceipt(effect="write", capability_id="coding_apply_patch", artifact_id="art-1")
        ],
    )

    outcome = classify_child_turn(
        child_result=child,
        mode_id="coding_execute",
        elapsed_ms=10,
        action_tool_ids={"coding_apply_patch", "coding_commit"},
    )

    assert outcome.status == "complete"

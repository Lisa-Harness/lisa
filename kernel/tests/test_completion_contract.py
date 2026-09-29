"""Unit tests for the pure completion contract resolver."""

from __future__ import annotations

from kernel.models import EffectReceipt
from kernel.turn_phases._completion_contract import (
    AcceptTurn,
    BlockNoCheckedCommit,
    BlockNoVerifiedWrite,
    ChatReadFootnote,
    ChatSoftAccept,
    PreexistingFailureBlock,
    SurfaceProseWithFootnote,
    resolve_completion_contract,
)


def test_preexisting_failure_block() -> None:
    """An upstream lifecycle failure (entry op, plan, fork) immediately blocks."""
    decision = resolve_completion_contract(
        mode_id="writer",
        preexisting_block=True,
    )
    assert isinstance(decision, PreexistingFailureBlock)


def test_sub_turn_is_exempt() -> None:
    """Sub-turn children are classified by classify_child_turn, not the parent resolver."""
    decision = resolve_completion_contract(
        mode_id="writer",
        action_tool_ids={"write_file"},
        sub_turn_kind="fork",
    )
    assert isinstance(decision, AcceptTurn)
    assert "sub-turn" in decision.reason


def test_yielded_turn_is_exempt() -> None:
    """Turns that yielded to the user deliberately paused and are exempt."""
    decision = resolve_completion_contract(
        mode_id="writer",
        action_tool_ids={"write_file"},
        yielded=True,
    )
    assert isinstance(decision, AcceptTurn)
    assert "yielded" in decision.reason


def test_checked_commit_goal_precedes_chat_soft_accept_and_write_receipt() -> None:
    decision = resolve_completion_contract(
        mode_id="executor",
        mode_kind="chat",
        content="Looks complete. " * 120,
        action_tool_ids={"create_worktree"},
        effect_receipts=[EffectReceipt(effect="write", capability_id="create_worktree")],
        total_tool_calls=2,
        task_completion="checked_commit",
        checked_commit_verified=False,
        checked_commit_reason="No current checked commit for this task.",
    )
    assert isinstance(decision, BlockNoCheckedCommit)
    assert "No current checked commit" in decision.block_message

    satisfied = resolve_completion_contract(
        mode_id="executor",
        task_completion="checked_commit",
        checked_commit_verified=True,
    )
    assert isinstance(satisfied, AcceptTurn)

    yielded = resolve_completion_contract(
        mode_id="executor",
        task_completion="checked_commit",
        checked_commit_verified=False,
        yielded=True,
    )
    assert isinstance(yielded, AcceptTurn)


def test_mode_without_write_contract_is_exempt() -> None:
    """Modes with no action tools declared (chat/ponder) have no write contract."""
    decision = resolve_completion_contract(
        mode_id="default",
        action_tool_ids=set(),
    )
    assert isinstance(decision, AcceptTurn)
    assert "no write contract" in decision.reason


def test_landed_write_receipt_exact_match() -> None:
    """A write receipt matching the action tool satisfies the contract."""
    receipt = EffectReceipt(effect="write", capability_id="write_file")
    decision = resolve_completion_contract(
        mode_id="writer",
        action_tool_ids={"write_file"},
        effect_receipts=[receipt],
    )
    assert isinstance(decision, AcceptTurn)
    assert "write receipt verified" in decision.reason


def test_landed_write_receipt_bare_name_match() -> None:
    """A namespaced MCP write matches the bare authored action tool name."""
    receipt = EffectReceipt(effect="write", capability_id="lisa-storage.write_file")
    decision = resolve_completion_contract(
        mode_id="writer",
        action_tool_ids={"write_file"},
        effect_receipts=[receipt],
    )
    assert isinstance(decision, AcceptTurn)
    assert "write receipt verified" in decision.reason


def test_honest_read_only_work_without_write_attempt() -> None:
    """Read receipts with zero write attempts prove honest read-only work in write-capable mode."""
    receipt = EffectReceipt(effect="read", capability_id="read_file")
    decision = resolve_completion_contract(
        mode_id="writer",
        action_tool_ids={"write_file"},
        effect_receipts=[receipt],
        called_tool_names={"read_file"},
        total_tool_calls=1,
    )
    assert isinstance(decision, AcceptTurn)
    assert "read-only work verified" in decision.reason


def test_read_work_with_failed_write_attempt_does_not_pass_as_read_only() -> None:
    """A failed write attempt cannot be excused by read work — model may claim false success."""
    receipt = EffectReceipt(effect="read", capability_id="read_file")
    decision = resolve_completion_contract(
        mode_id="writer",
        mode_kind="action",
        action_tool_ids={"write_file"},
        effect_receipts=[receipt],
        called_tool_names={"read_file", "write_file"},
        total_tool_calls=2,
    )
    assert isinstance(decision, BlockNoVerifiedWrite)


def test_routing_misfire_prose_surfaced_with_footnote() -> None:
    """Zero-dispatch turn with aborted synth dispatch surfaces model prose with warning footnote."""
    decision = resolve_completion_contract(
        mode_id="executor",
        mode_kind="chat",
        content="Here is a full explanation of why we shouldn't create a timer.",
        action_tool_ids={"coding_apply_patch"},
        routing_misfire_prose=True,
        total_tool_calls=0,
        called_tool_names=set(),
    )
    assert isinstance(decision, SurfaceProseWithFootnote)
    assert decision.action_tool_ids == frozenset({"coding_apply_patch"})


def test_routing_misfire_never_masks_attempted_write() -> None:
    """Routing misfire cannot surface prose if an action tool was actually attempted."""
    decision = resolve_completion_contract(
        mode_id="executor",
        mode_kind="chat",
        content="I attempted the patch.",
        action_tool_ids={"coding_apply_patch"},
        routing_misfire_prose=True,
        total_tool_calls=1,
        called_tool_names={"coding_apply_patch"},
    )
    assert isinstance(decision, BlockNoVerifiedWrite)


def test_chat_mode_soft_accept_with_dense_content() -> None:
    """A chat mode with a failed write attempt that still produced dense content is softened."""
    # Write was attempted and failed (no write receipt), so it is not pure read work.
    # But because it is a chat mode, did work, and produced dense content, it is accepted.
    decision = resolve_completion_contract(
        mode_id="explainer",
        mode_kind="chat",
        content="A" * 200,
        action_tool_ids={"write_summary"},
        effect_receipts=[],
        called_tool_names={"write_summary"},
        total_tool_calls=1,
        dense_content_min_chars=150,
    )
    assert isinstance(decision, ChatSoftAccept)
    assert decision.content_len == 200


def test_chat_mode_read_footnote_for_substantive_content() -> None:
    """A chat mode with unverified tool work (no receipts) and short content receives a footnote."""
    # Total tool calls > 0, no write attempted, but no read receipts either (e.g. tool errored).
    decision = resolve_completion_contract(
        mode_id="explainer",
        mode_kind="chat",
        content="Short answer.",
        action_tool_ids={"write_summary"},
        effect_receipts=[],
        called_tool_names={"other_tool"},
        total_tool_calls=1,
        dense_content_min_chars=150,
    )
    assert isinstance(decision, ChatReadFootnote)
    assert decision.action_tool_ids == frozenset({"write_summary"})


def test_zero_tool_calls_prose_only_strictly_blocks() -> None:
    """Zero tool calls in a write-contract mode is hallucinated success regardless of mode kind."""
    decision = resolve_completion_contract(
        mode_id="explainer",
        mode_kind="chat",
        content="A" * 500,
        action_tool_ids={"write_summary"},
        total_tool_calls=0,
        called_tool_names=set(),
    )
    assert isinstance(decision, BlockNoVerifiedWrite)


def test_action_mode_read_only_strictly_blocks() -> None:
    """An action mode that did read-only work is blocked because writes land outside thread."""
    receipt = EffectReceipt(effect="read", capability_id="read_file")
    # Honest read work with no write attempt in an action mode was
    # checked in step 3 (read_work and not write_attempted -> AcceptTurn).
    decision_pure_read = resolve_completion_contract(
        mode_id="writer",
        mode_kind="action",
        content="A" * 500,
        action_tool_ids={"write_file"},
        effect_receipts=[receipt],
        called_tool_names={"read_file"},
        total_tool_calls=1,
    )
    assert isinstance(decision_pure_read, AcceptTurn)

    # But if a write was attempted and failed:
    decision_failed_write = resolve_completion_contract(
        mode_id="writer",
        mode_kind="action",
        content="A" * 500,
        action_tool_ids={"write_file"},
        effect_receipts=[receipt],
        called_tool_names={"read_file", "write_file"},
        total_tool_calls=2,
    )
    assert isinstance(decision_failed_write, BlockNoVerifiedWrite)


def test_block_message_formatting() -> None:
    """BlockNoVerifiedWrite produces a complete, deterministic block message."""
    block = BlockNoVerifiedWrite(
        mode_id="code_writer",
        action_tool_ids=frozenset({"apply_patch", "commit"}),
    )
    msg = block.block_message
    assert "The `code_writer` turn produced no verified write" in msg
    assert "apply_patch, commit" in msg
    assert "unverified" in msg

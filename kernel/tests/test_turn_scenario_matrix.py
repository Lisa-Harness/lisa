"""Scenario matrix tests pinning turn-loop guard interactions.

This suite codifies the cross-product of guard inputs across the two pure
decision functions of the kernel turn loop:

1. :func:`kernel.turn_phases._lifecycle.resolve_lifecycle_action` (in-loop)
2. :func:`kernel.turn_phases._completion_contract.resolve_completion_contract` (post-loop)

Each historical incident is encoded as a named scenario cell so guard
interactions cannot silently regress into deadlock or loop states.
"""

from __future__ import annotations

import pytest
from kernel.models import EffectReceipt, YieldPolicy
from kernel.turn_phases._completion_contract import (
    AcceptTurn,
    BlockNoVerifiedWrite,
    ChatSoftAccept,
    SurfaceProseWithFootnote,
    resolve_completion_contract,
)
from kernel.turn_phases._lifecycle import (
    ContinueAction,
    LifecycleResolverSettings,
    PauseAction,
    _TurnStateView,
    resolve_lifecycle_action,
)

# Shared default settings matching kernel.config.Settings defaults
_DEFAULT_SETTINGS = LifecycleResolverSettings(
    max_tool_iterations=300,
    per_step_max_iterations=75,
    per_step_max_tool_calls=150,
    stage_nudge_threshold=4,
    stage_nudge_cap=9,
    forced_action_non_compliant_cap=2,
    max_handoffs_per_turn=50,
    write_rescue_mode="explainer",
)


# ═════════════════════════════════════════════════════════════════════════════
# 1. Historical Incident Matrix Cells
# ═════════════════════════════════════════════════════════════════════════════


def test_cell_2026_08_03_fork_child_read_loop() -> None:
    """Cell: 2026-08-03 fork-child read loop.

    Incident: A fork child in an action mode made 49 read calls without
    writing, then returned text claiming success.
    Invariant: Sub-turns are exempt from the parent contract resolver
    (they are classified by classify_child_turn, which checks receipts).
    Parent turns in action modes that only read without writing are strictly
    blocked unless chat softening applies.
    """
    receipt = EffectReceipt(effect="read", capability_id="read_file")

    # Parent turn in action mode that attempted write and failed -> strictly blocked
    parent_decision = resolve_completion_contract(
        mode_id="writer",
        mode_kind="action",
        content="Done — I wrote the file.",
        action_tool_ids={"write_file"},
        effect_receipts=[receipt],
        called_tool_names={"read_file", "write_file"},
        total_tool_calls=2,
    )
    assert isinstance(parent_decision, BlockNoVerifiedWrite)

    # Sub-turn child is exempt from parent resolver
    child_decision = resolve_completion_contract(
        mode_id="writer",
        mode_kind="action",
        content="Done.",
        action_tool_ids={"write_file"},
        sub_turn_kind="fork",
    )
    assert isinstance(child_decision, AcceptTurn)


def test_cell_2026_08_27_read_receipt_fix() -> None:
    """Cell: 2026-08-27 read-receipt fix.

    Incident: A read-only query ('what VMs are running?') routed to a
    write-capable mode; the turn completed with read receipts and zero
    write attempts.
    Invariant: Read receipts with NO write attempts in a write-capable mode
    is honest read work — accepted as-is without warning footnote or block.
    """
    receipt = EffectReceipt(effect="read", capability_id="list_vms")
    decision = resolve_completion_contract(
        mode_id="cloudservers",
        mode_kind="action",
        content="Here are the active virtual machines: vm-1 (running), vm-2 (stopped).",
        action_tool_ids={"cs_apply_cloud_init", "cs_reboot_vm"},
        effect_receipts=[receipt],
        called_tool_names={"list_vms"},
        total_tool_calls=1,
    )
    assert isinstance(decision, AcceptTurn)
    assert "read-only work verified" in decision.reason


def test_cell_2026_09_02_write_announcement_contract() -> None:
    """Cell: 2026-09-02 write-announcement loop.

    Incident: Eight turns of 'Writing both files now' while dispatching only reads.
    Invariant: If the model announces in prose but never attempts a write,
    and chat softening applies with dense content, it is accepted as soft prose;
    if write was attempted and failed, it is strictly blocked.
    """
    receipt = EffectReceipt(effect="read", capability_id="read_config")

    # Failed write attempt cannot be softened by read receipts
    failed_write = resolve_completion_contract(
        mode_id="writer",
        mode_kind="action",
        content="Writing both files now. No more stalling.",
        action_tool_ids={"write_file"},
        effect_receipts=[receipt],
        called_tool_names={"read_config", "write_file"},
        total_tool_calls=2,
    )
    assert isinstance(failed_write, BlockNoVerifiedWrite)


def test_cell_2026_09_04_fetch_page_read_loop_burn() -> None:
    """Cell: 2026-09-04 fetch_page read-loop burn.

    Incident: 30 consecutive fetch_page calls with 404/402 errors burned to cap.
    Invariant:
    - At threshold with consecutive errors (>= 2): lifecycle resolver produces
      error-enriched ContinueAction telling model to diagnose errors and advance.
    - At cap: lifecycle resolver produces PauseAction (freeze, do not truncate).
    """
    # 1. At threshold with error streak
    state_at_threshold = _TurnStateView(
        iteration=5,
        step_iteration=5,
        step_tool_calls=5,
        total_tool_calls=5,
        turns_in_stage=5,  # > stage_nudge_threshold (4)
        consecutive_tool_errors=3,
        last_error_tool="fetch_page",
        action_tool_ids=frozenset(),
        has_read_receipt=True,
    )
    action_threshold = resolve_lifecycle_action(
        state=state_at_threshold,
        mode_id="researcher",
        mode_yield_policy=YieldPolicy(),
        settings=_DEFAULT_SETTINGS,
    )
    assert isinstance(action_threshold, ContinueAction)
    assert "fetch_page" in action_threshold.nudge
    assert "diagnose why" in action_threshold.nudge.lower()

    # 2. At cap
    state_at_cap = _TurnStateView(
        iteration=9,
        step_iteration=9,
        step_tool_calls=9,
        total_tool_calls=9,
        turns_in_stage=9,  # >= stage_nudge_cap (9)
        action_tool_ids=frozenset(),
        has_read_receipt=True,
    )
    action_cap = resolve_lifecycle_action(
        state=state_at_cap,
        mode_id="researcher",
        mode_yield_policy=YieldPolicy(),
        settings=_DEFAULT_SETTINGS,
    )
    assert isinstance(action_cap, PauseAction)
    assert "stuck in researcher" in action_cap.reason


def test_cell_2026_09_02_turn70_prose_discard() -> None:
    """Cell: 2026-09-02 turn-70 executor misroute prose discard.

    Incident: User asked a conversational question that routed to executor.
    Model answered in prose with 0 tool calls. Write-contract backstop armed,
    synth call with {} was aborted by required-params guard, setting
    routing_misfire_prose=True. The old post-loop check discarded the prose.
    Invariant: The prose is surfaced with the warning footnote and NOT blocked.
    """
    decision = resolve_completion_contract(
        mode_id="executor",
        mode_kind="chat",
        content="Would you like to extend what we created? I can help with systemd timers.",
        action_tool_ids={"coding_apply_patch"},
        routing_misfire_prose=True,
        total_tool_calls=0,
        called_tool_names=set(),
    )
    assert isinstance(decision, SurfaceProseWithFootnote)
    assert decision.action_tool_ids == frozenset({"coding_apply_patch"})


def test_cell_2026_08_30_budget_claim_fabrication() -> None:
    """Cell: 2026-08-30 budget-claim fabrication.

    Incident: At iteration 2 of 300, an empty completion was falsely
    reported as 'I have hit my iteration budget'.
    Invariant: When iteration < max_tool_iterations, the lifecycle resolver
    never reports budget exhaustion.
    """
    state_early = _TurnStateView(
        iteration=2,
        step_iteration=2,
        step_tool_calls=0,
        total_tool_calls=0,
        turns_in_stage=2,
        budget_exhausted=False,
    )
    action = resolve_lifecycle_action(
        state=state_early,
        mode_id="default",
        mode_yield_policy=YieldPolicy(),
        settings=_DEFAULT_SETTINGS,
    )
    assert isinstance(action, ContinueAction)
    assert not action.nudge


# ═════════════════════════════════════════════════════════════════════════════
# 2. Combinatorial Completion Contract Grid
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    ("mode_kind", "has_receipt", "write_attempted", "content_dense", "expected_type"),
    [
        # Landed write always succeeds
        ("chat", True, True, False, AcceptTurn),
        ("action", True, True, False, AcceptTurn),
        # Honest read work (no write attempted) always succeeds
        ("chat", False, False, False, AcceptTurn),
        ("action", False, False, False, AcceptTurn),
        # Failed write in chat mode with dense content -> softened
        ("chat", False, True, True, ChatSoftAccept),
        # Failed write in action mode with dense content -> strictly blocked
        ("action", False, True, True, BlockNoVerifiedWrite),
        # Failed write in chat mode with short content -> strictly blocked
        ("chat", False, True, False, BlockNoVerifiedWrite),
        # Failed write in action mode with short content -> strictly blocked
        ("action", False, True, False, BlockNoVerifiedWrite),
    ],
)
def test_completion_contract_combinatorial_grid(
    mode_kind: str,
    has_receipt: bool,
    write_attempted: bool,
    content_dense: bool,
    expected_type: type,
) -> None:
    """Parametrized grid covering mode kinds, receipts, attempts, and density."""
    receipts = []
    if has_receipt:
        receipts.append(EffectReceipt(effect="write", capability_id="write_file"))
    else:
        receipts.append(EffectReceipt(effect="read", capability_id="read_file"))

    called = {"read_file"}
    if write_attempted:
        called.add("write_file")

    content = ("Dense analysis findings. " * 50) if content_dense else "Short output."

    decision = resolve_completion_contract(
        mode_id="test_mode",
        mode_kind=mode_kind,
        content=content,
        action_tool_ids={"write_file"},
        effect_receipts=receipts,
        called_tool_names=called,
        total_tool_calls=len(called),
        dense_content_min_chars=100,
    )
    assert isinstance(decision, expected_type)


# ═════════════════════════════════════════════════════════════════════════════
# 3. Lifecycle Resolver Stall vs Cap Hierarchy
# ═════════════════════════════════════════════════════════════════════════════


@pytest.mark.parametrize(
    ("turns_in_stage", "has_read", "errors", "expected_action"),
    [
        (1, False, 0, ContinueAction),  # Normal early working iteration
        (4, False, 0, ContinueAction),  # At threshold (generic nudge)
        (5, True, 0, ContinueAction),   # Past threshold with read receipts (synthesize nudge)
        (5, True, 2, ContinueAction),   # Past threshold with error streak (diagnose nudge)
        (9, False, 0, PauseAction),     # At stage cap -> pauses
        (10, True, 3, PauseAction),     # Past stage cap -> pauses
    ],
)
def test_lifecycle_stage_stall_hierarchy(
    turns_in_stage: int,
    has_read: bool,
    errors: int,
    expected_action: type,
) -> None:
    """Verify that stage progression advances strictly from work to nudge to pause."""
    state = _TurnStateView(
        iteration=turns_in_stage,
        step_iteration=turns_in_stage,
        step_tool_calls=turns_in_stage,
        total_tool_calls=turns_in_stage,
        turns_in_stage=turns_in_stage,
        has_read_receipt=has_read,
        consecutive_tool_errors=errors,
        last_error_tool="web_search" if errors else "",
    )
    action = resolve_lifecycle_action(
        state=state,
        mode_id="analyst",
        mode_yield_policy=YieldPolicy(),
        settings=_DEFAULT_SETTINGS,
    )
    assert isinstance(action, expected_action)
    if isinstance(action, ContinueAction):
        if turns_in_stage >= 4:
            assert action.nudge != ""
        else:
            assert action.nudge == ""

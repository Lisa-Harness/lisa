"""One lifecycle-guidance channel, appended at complete tool-batch boundaries."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from kernel.models import EffectReceipt, WorkflowModeConfig
from kernel.turn_hooks import (
    ModelHooks,
)
from kernel.turn_runner import TurnHooks, _build_budget_briefing, run_turn
from kernel.workflow_position import WorkflowPosition

_KERNEL_PKG = Path(__file__).resolve().parents[1] / "src" / "kernel"
_ROLE_USER_RE = re.compile(r"[\"']role[\"']\s*:\s*[\"']user[\"']")


# ── 1. Lint-level source scan ─────────────────────────────────────────


def test_turn_loop_has_only_query_and_lifecycle_user_message_sites() -> None:
    """Only host-query assembly and the central briefing helper add user messages."""
    runner_files = sorted((_KERNEL_PKG / "turn_runner").glob("*.py"))
    phase_files = sorted((_KERNEL_PKG / "turn_phases").glob("*.py"))
    assert runner_files, "kernel/turn_runner package not found; scan would be vacuous"
    assert phase_files, "kernel/turn_phases package not found; scan would be vacuous"

    runner_hits = [
        hit
        for path in runner_files
        for hit in _ROLE_USER_RE.findall(path.read_text(encoding="utf-8"))
    ]
    assert len(runner_hits) == 2, (
        "kernel/turn_runner must contain only query and lifecycle role:'user' "
        f"message literals; found {len(runner_hits)}"
    )
    for phase_file in phase_files:
        hits = _ROLE_USER_RE.findall(phase_file.read_text(encoding="utf-8"))
        assert not hits, (
            f"kernel/turn_phases/{phase_file.name} synthesizes user "
            f"messages ({len(hits)} site(s)); nudges ride the lifecycle "
            "briefing helper"
        )


# ── 2. Briefing guidance transport ────────────────────────────────────


def _briefing_state() -> SimpleNamespace:
    return SimpleNamespace(
        iteration=2,
        step_iteration=0,
        step_tool_calls=0,
        turns_in_stage=0,
        call_ledger_counts={},
        sub_turn_kind=None,
        action_tool_ids=set(),
        called_tool_names=set(),
        effect_receipts=[],
    )


def test_budget_briefing_carries_state_ledger_lines() -> None:
    """The standing state ledger rides the lifecycle block (R3).

    Writes landed/attempted and the refs the turn holds render every
    iteration — deterministic text built from effect receipts and the
    dispatch-time ref capture, zero model tokens, so the model never
    re-scans context to know what it has done or can re-expand.
    """
    state = SimpleNamespace(
        iteration=3,
        step_iteration=0,
        step_tool_calls=0,
        turns_in_stage=2,
        call_ledger_counts={},
        sub_turn_kind=None,
        action_tool_ids=set(),
        called_tool_names={"read_file", "write_file", "patch_file"},
        effect_receipts=[
            EffectReceipt(effect="write", capability_id="write_file", artifact_id="doc-1"),
            EffectReceipt(effect="write", capability_id="patch_file", artifact_id=""),
            EffectReceipt(effect="read", capability_id="read_file", artifact_id=""),
        ],
        active_refs={"r" + "a" * 64, "t391.m20", "t390.m4"},
    )
    briefing = _build_budget_briefing(
        state=state,
        max_tool_iterations=50,
        per_step_max_iterations=0,
        per_step_max_tool_calls=0,
        stage_nudge_cap=9,
        tool_schemas=[],
        operation_catalog={},
        tool_catalog={},
        execution_policy=None,
        budget_exhausted=False,
    )
    assert "## Turn Lifecycle" in briefing
    assert "writes landed: 1, attempted: 2" in briefing
    assert "refs held: 3" in briefing
    # Refs render abbreviated (first ref chars + …), with the expand path.
    assert "thread_read_content" in briefing


def test_budget_briefing_ledger_lines_omitted_when_empty() -> None:
    """A turn with no receipts and no refs renders no ledger lines (R3).

    Iteration 1 before any dispatch has nothing to report; the briefing
    stays byte-identical to the pre-ledger format.
    """
    state = SimpleNamespace(
        iteration=1,
        step_iteration=0,
        step_tool_calls=0,
        turns_in_stage=0,
        call_ledger_counts={},
        sub_turn_kind=None,
        action_tool_ids=set(),
        called_tool_names=set(),
        effect_receipts=[],
        active_refs=set(),
    )
    briefing = _build_budget_briefing(
        state=state,
        max_tool_iterations=50,
        per_step_max_iterations=0,
        per_step_max_tool_calls=0,
        stage_nudge_cap=9,
        tool_schemas=[],
        operation_catalog={},
        tool_catalog={},
        execution_policy=None,
        budget_exhausted=False,
    )
    assert briefing == "## Turn Lifecycle\n\niteration: 1 of 50"


def test_budget_briefing_carries_guidance_lines() -> None:
    """Guidance lines ride the ``## Turn Lifecycle`` block itself.

    They must live under the same marker the per-iteration strip removes,
    so a stale nudge cannot survive into the next iteration.
    """
    briefing = _build_budget_briefing(
        state=_briefing_state(),
        max_tool_iterations=10,
        per_step_max_iterations=0,
        per_step_max_tool_calls=0,
        stage_nudge_cap=9,
        tool_schemas=[],
        operation_catalog={},
        tool_catalog={},
        execution_policy=None,
        budget_exhausted=False,
        guidance=("WORKFLOW CHAIN IN FLIGHT: advance.", "soft nudge"),
    )
    assert briefing.startswith("## Turn Lifecycle")
    assert "WORKFLOW CHAIN IN FLIGHT: advance." in briefing
    assert "soft nudge" in briefing


def test_budget_briefing_guidance_only_still_marks_block() -> None:
    """A guidance-only briefing emits the ``## Turn Lifecycle`` marker.

    Without the marker the next iteration's strip would leave the stale
    guidance in the system prompt.
    """
    briefing = _build_budget_briefing(
        state=_briefing_state(),
        max_tool_iterations=0,  # no iteration line
        per_step_max_iterations=0,
        per_step_max_tool_calls=0,
        stage_nudge_cap=9,
        tool_schemas=[],
        operation_catalog={},
        tool_catalog={},
        execution_policy=None,
        budget_exhausted=False,
        guidance=("soft nudge",),
    )
    assert briefing == "## Turn Lifecycle\n\nsoft nudge"


def test_budget_briefing_empty_when_no_lines_no_guidance() -> None:
    """No budgets and no guidance → empty string (existing contract)."""
    briefing = _build_budget_briefing(
        state=_briefing_state(),
        max_tool_iterations=0,
        per_step_max_iterations=0,
        per_step_max_tool_calls=0,
        stage_nudge_cap=9,
        tool_schemas=[],
        operation_catalog={},
        tool_catalog={},
        execution_policy=None,
        budget_exhausted=False,
    )
    assert briefing == ""


def test_budget_briefing_stage_budget_line() -> None:
    """The stage-budget line surfaces the per-mode ceiling that actually fires.

    Without it the model sees only ``iter 3/300`` and has no way to
    reconcile that with prior cap-fire messages in the thread. The format
    is ``N completed with M remaining`` to prevent the completed/remaining
    confusion that the old ``N/M`` format caused.
    """
    state = SimpleNamespace(
        iteration=3,
        step_iteration=3,
        step_tool_calls=0,
        turns_in_stage=3,
        call_ledger_counts={},
        sub_turn_kind=None,
        action_tool_ids=set(),
        called_tool_names=set(),
        effect_receipts=[],
    )
    briefing = _build_budget_briefing(
        state=state,
        max_tool_iterations=300,
        per_step_max_iterations=0,
        per_step_max_tool_calls=0,
        stage_nudge_cap=9,
        tool_schemas=[],
        operation_catalog={},
        tool_catalog={},
        execution_policy=None,
        budget_exhausted=False,
    )
    assert "## Turn Lifecycle" in briefing
    assert "iteration: 3 of 300" in briefing
    assert "stage budget: 3 completed with 6 remaining" in briefing


def test_budget_briefing_count_lines_state_semantics_explicitly() -> None:
    """Every count line states used-vs-remaining semantics explicitly (F2).

    The 2026-10-14 self-review found ``step: 74/75 iters`` (remaining)
    rendered beside ``iteration: 1 of 300`` (used) and
    ``calls: coding_read_file:46/50`` (remaining) — three count lines,
    two semantics, zero labels. The stage line already spells out
    ``N completed with M remaining``; the per-step and per-target lines
    must do the same instead of relying on positional ``N/M`` that was
    consistently misread as a progress bar.
    """
    state = SimpleNamespace(
        iteration=1,
        step_iteration=1,
        step_tool_calls=16,
        turns_in_stage=0,
        call_ledger_counts={"coding_read_file": 4},
        sub_turn_kind=None,
        action_tool_ids=set(),
        called_tool_names=set(),
        effect_receipts=[],
    )
    tool_schemas = [{"function": {"name": "coding_read_file"}}]
    with patch(
        "kernel.capability.resolve_capability_for_turn_limit",
        return_value=SimpleNamespace(max_calls_per_turn=50),
    ):
        briefing = _build_budget_briefing(
            state=state,
            max_tool_iterations=300,
            per_step_max_iterations=75,
            per_step_max_tool_calls=150,
            stage_nudge_cap=9,
            tool_schemas=tool_schemas,
            operation_catalog={},
            tool_catalog={},
            execution_policy=None,
            budget_exhausted=False,
        )
    # Used-semantics line keeps its explicit "N of M" form.
    assert "iteration: 1 of 300" in briefing
    # Remaining-semantics lines now say "remaining" instead of bare N/M.
    assert "step: 74 of 75 iterations remaining" in briefing
    assert "step: 134 of 150 calls remaining" in briefing
    # Per-target limits state used/max explicitly.
    assert "calls: coding_read_file 4 of 50 used" in briefing
    # No bare remaining/max N/M count format survives.
    assert "74/75" not in briefing
    assert "134/150" not in briefing
    assert "coding_read_file:46/50" not in briefing


def test_budget_briefing_stage_budget_omitted_when_zero_turns() -> None:
    """When ``turns_in_stage`` is 0 the stage-budget line is omitted (iteration 1)."""
    state = SimpleNamespace(
        iteration=1,
        step_iteration=0,
        step_tool_calls=0,
        turns_in_stage=0,
        call_ledger_counts={},
        sub_turn_kind=None,
        action_tool_ids=set(),
        called_tool_names=set(),
        effect_receipts=[],
    )
    briefing = _build_budget_briefing(
        state=state,
        max_tool_iterations=300,
        per_step_max_iterations=0,
        per_step_max_tool_calls=0,
        stage_nudge_cap=9,
        tool_schemas=[],
        operation_catalog={},
        tool_catalog={},
        execution_policy=None,
        budget_exhausted=False,
    )
    assert "stage budget" not in briefing


# ── 3. Runtime invariant across a live chain handoff ──────────────────


@dataclass(frozen=True, slots=True)
class _StubCompletion:
    """Fake model completion for tests (mirrors test_turn_runner)."""

    content: str
    tool_calls: list[dict[str, Any]] | None = None


def _mode_config(
    *,
    mode_id: str,
    satisfies_intents: list[str] | None = None,
    routing_intent_tags: list[str] | None = None,
    routing_intent_keywords: dict[str, list[str]] | None = None,
) -> WorkflowModeConfig:
    return WorkflowModeConfig(
        mode_id=mode_id,
        start_stage_id="none",
        mode_kind="chat",
        routable=True,
        satisfies_intents=list(satisfies_intents) if satisfies_intents else [],
        routing_intent_tags=list(routing_intent_tags) if routing_intent_tags else [],
        routing_intent_keywords=dict(routing_intent_keywords) if routing_intent_keywords else {},
    )


def _catalog_with_chain() -> dict[str, WorkflowModeConfig]:
    """A catalog for the search → coding → markdown chain."""
    return {
        "default": _mode_config(mode_id="default"),
        "search": _mode_config(
            mode_id="search",
            satisfies_intents=["research"],
            routing_intent_tags=["research"],
            routing_intent_keywords={"research": ["search"]},
        ),
        "coding": _mode_config(
            mode_id="coding",
            satisfies_intents=["coding"],
            routing_intent_tags=["coding"],
            routing_intent_keywords={"coding": ["code", "read"]},
        ),
        "markdown": _mode_config(
            mode_id="markdown",
            satisfies_intents=["write"],
            routing_intent_tags=["write"],
            routing_intent_keywords={"write": ["write", "markdown"]},
        ),
    }


@pytest.mark.asyncio
async def test_chain_handoff_guidance_follows_complete_tool_batches() -> None:
    """Chain guidance must not split an assistant's tool-call/result batch."""
    active: list[WorkflowPosition | None] = [None]
    snapshots: list[list[dict[str, Any]]] = []
    _handoff_issued = {"flag": False}

    async def complete(**kwargs: Any) -> _StubCompletion:
        snapshots.append([dict(m) for m in kwargs.get("messages", [])])
        if not _handoff_issued["flag"]:
            _handoff_issued["flag"] = True
            return _StubCompletion(
                content="",
                tool_calls=[
                    {
                        "id": "call_handoff",
                        "type": "function",
                        "function": {
                            "name": "transition_stage",
                            "arguments": '{"target_stage_id": "search"}',
                        },
                    }
                ],
            )
        return _StubCompletion(content="Done.")

    hooks = TurnHooks(
        models=ModelHooks(
            complete=complete,
        ),
    )
    hooks.workflow.workflow_position = lambda: active[0]

    def _updater(position: WorkflowPosition) -> None:
        active[0] = position

    hooks.workflow.workflow_position_updater = _updater

    catalog = _catalog_with_chain()
    with (
        patch("kernel.turn_runner.get_workflow_catalog", return_value=catalog),
        patch("kernel.turn_phases._routing_hint.get_workflow_catalog", return_value=catalog),
        patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=catalog),
    ):
        await run_turn(
            mode_id="default",
            query="search the web then read your code then write a markdown",
            hooks=hooks,
        )

    assert len(snapshots) >= 2, "expected at least the handoff iteration and a continuation"
    for snapshot in snapshots:
        pending: set[str] = set()
        for message in snapshot:
            if message.get("tool_calls"):
                pending.update(call["id"] for call in message["tool_calls"])
            elif message.get("role") == "tool":
                pending.discard(message["tool_call_id"])
            else:
                assert not pending, "guidance split a pending tool batch"
        user_messages = [m for m in snapshot if m.get("role") == "user"]
        assert user_messages
        assert "## User Query" in str(user_messages[0].get("content", ""))
        for message in user_messages[1:]:
            assert "## Turn Lifecycle" in str(message.get("content", ""))
    assert any(
        any("WORKFLOW CHAIN IN FLIGHT" in str(m.get("content", "")) for m in snapshot)
        for snapshot in snapshots
    ), "chain-advance directive missing from the lifecycle briefing"

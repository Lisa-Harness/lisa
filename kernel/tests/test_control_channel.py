"""Phase 7 (one control channel): turn-loop message-role invariants.

The turn loop never appends a synthesized ``role: "user"`` message — user
messages come only from the host (the initial query). Soft nudges (the
resolver's ``ContinueAction.nudge``, the sub-turn midpoint write-nudge)
and the post-handoff chain-advance directive ride the per-iteration
lifecycle briefing in the system prompt (``## Turn Lifecycle``).

Three enforcement layers:
  1. A lint-level source scan: exactly one ``role: "user"`` dict literal
     may exist in ``kernel/turn_runner.py`` (the host query) and none in
     the ``kernel/turn_phases`` modules.
  2. A unit check that ``_build_budget_briefing`` carries guidance lines
     inside the lifecycle block (so the per-iteration marker strip
     removes them with the briefing — no stale accumulation).
  3. A runtime scripted-turn check: across a live ad-hoc chain handoff
     (default → search, chain search → coding → markdown driven by the
     forced-action enforcer), every model call sees exactly one user
     message — the host query — and the chain-advance directive appears
     in the system-prompt briefing while the chain is in flight.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from kernel.models import WorkflowModeConfig
from kernel.turn_hooks import (
    ModelHooks,
)
from kernel.turn_runner import TurnHooks, _build_budget_briefing, run_turn
from kernel.workflow_position import WorkflowPosition

_KERNEL_PKG = Path(__file__).resolve().parents[1] / "src" / "kernel"
_ROLE_USER_RE = re.compile(r"[\"']role[\"']\s*:\s*[\"']user[\"']")


# ── 1. Lint-level source scan ─────────────────────────────────────────


def test_turn_loop_source_has_no_synthesized_user_messages() -> None:
    """Lint-level invariant: one ``role: "user"`` literal in the loop.

    The single permitted site is the host-query message assembly in
    ``_run_turn_inner``. Everything else — nudges, hints, escalations,
    closing directives — rides the system prompt.
    """
    runner_files = sorted((_KERNEL_PKG / "turn_runner").glob("*.py"))
    phase_files = sorted((_KERNEL_PKG / "turn_phases").glob("*.py"))
    assert runner_files, "kernel/turn_runner package not found; scan would be vacuous"
    assert phase_files, "kernel/turn_phases package not found; scan would be vacuous"

    runner_hits = [
        hit
        for path in runner_files
        for hit in _ROLE_USER_RE.findall(path.read_text(encoding="utf-8"))
    ]
    assert len(runner_hits) == 1, (
        "kernel/turn_runner must contain exactly one role:'user' "
        f"message literal (the host query); found {len(runner_hits)}"
    )
    for phase_file in phase_files:
        hits = _ROLE_USER_RE.findall(phase_file.read_text(encoding="utf-8"))
        assert not hits, (
            f"kernel/turn_phases/{phase_file.name} synthesizes user "
            f"messages ({len(hits)} site(s)); nudges ride the lifecycle "
            "briefing (system prompt)"
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
async def test_chain_handoff_turn_has_no_synthesized_user_messages() -> None:
    """Every model call sees exactly one user message: the host query.

    Drives a live ad-hoc chain (``default → search`` by the stub model,
    then the forced-action enforcer advances ``search → coding →
    markdown``). While the chain is in flight the chain-advance directive
    must appear in the system-prompt briefing (the Phase-7 replacement
    for the old synthesized ``user`` escalation); at no point may any
    snapshot contain a synthesized user message.
    """
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
        user_messages = [m for m in snapshot if m.get("role") == "user"]
        assert len(user_messages) == 1, (
            "every model call must see exactly one user message (the host "
            f"query); found {len(user_messages)}: {user_messages!r}"
        )
        assert "## User Query" in str(user_messages[0].get("content", ""))
    # The chain-advance directive rode the system-prompt briefing while
    # the chain was in flight (post-handoff hint non-empty).
    assert any(
        snapshot
        and snapshot[0].get("role") == "system"
        and "WORKFLOW CHAIN IN FLIGHT" in str(snapshot[0].get("content", ""))
        for snapshot in snapshots
    ), "chain-advance directive missing from the lifecycle briefing"

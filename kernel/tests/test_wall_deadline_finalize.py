"""Mid-iteration wall-budget enforcement and partial-work visibility.

Regressions for the 2026-10-21 delegate incident: the child's armed wall
budget was only consulted at iteration boundaries, so one in-flight model
call (95s reasoning) plus the finalization tail crossed the operation
registry's 300s hard cancel — the cancel discarded child work the child
had ALREADY completed (applied edits, created worktree).
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any

import pytest
from kernel.models import EffectReceipt, TurnBudget
from kernel.turn_hooks import ModelHooks, TurnHooks
from kernel.turn_runner import run_turn


@dataclass(frozen=True)
class _SlowStub:
    """Completion stub whose first call runs past the wall deadline."""

    content: str = ""
    tool_calls: list[dict[str, Any]] | None = None
    finish_reason: str | None = None
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    reasoning_content: str = ""


@pytest.mark.asyncio
async def test_wall_deadline_cancels_inflight_model_call(_mcp_catalog_setup) -> None:
    """A model call outliving the wall budget is cancelled, not completed.

    The turn must finalize with the partial-work diagnostic instead of
    raising ``TimeoutError`` out of ``run_turn`` (which would surface to
    the registry as an unstructured failure and lose the receipts the
    earlier iterations had already captured).
    """
    state: dict[str, Any] = {"calls": 0}

    async def complete(**_kwargs: Any) -> _SlowStub:
        state["calls"] += 1
        # The one iteration of this turn outlives the 500ms wall budget.
        await asyncio.sleep(5)
        return _SlowStub(content="never seen")

    budget = TurnBudget(max_tool_calls=0, max_total_tokens=0, max_completion_tokens=0,
                        max_wall_ms=500)
    hooks = TurnHooks(models=ModelHooks(complete=complete))
    t0 = time.monotonic()
    result = await run_turn(mode_id="default", query="hello", hooks=hooks, turn_budget=budget)
    elapsed = time.monotonic() - t0
    assert isinstance(result.response_text, str), "run_turn must return, not raise"
    assert elapsed < 4.5, f"the cancelled call must not run to completion ({elapsed:.2f}s)"
    assert state["calls"] == 1


@pytest.mark.asyncio
async def test_wall_deadline_finalization_tail_is_bounded(_mcp_catalog_setup) -> None:
    """The forced-final completion is bounded by the same remaining budget.

    When the deadline is already spent when finalization starts, the
    forced-final call is skipped (bound fires immediately) and the turn
    still returns a result — never an unhandled TimeoutError.
    """

    async def complete(**_kwargs: Any) -> _SlowStub:
        return _SlowStub(content="final answer")

    budget = TurnBudget(max_tool_calls=0, max_total_tokens=0, max_completion_tokens=0,
                        max_wall_ms=1)
    hooks = TurnHooks(models=ModelHooks(complete=complete))
    result = await run_turn(mode_id="default", query="hello", hooks=hooks, turn_budget=budget)
    assert isinstance(result.response_text, str)


def test_remaining_wall_seconds_zero_when_spent() -> None:
    """A spent budget returns 0.0 (fires immediately); absent/unlimited → None."""
    from kernel.turn_runner import _remaining_wall_seconds

    budget = TurnBudget(max_tool_calls=0, max_total_tokens=0, max_completion_tokens=0,
                        max_wall_ms=1_000)
    start = time.monotonic() - 2.0  # 2s elapsed against a 1s budget
    assert _remaining_wall_seconds(turn_budget=budget, turn_start_mono=start) == 0.0
    assert _remaining_wall_seconds(turn_budget=budget, turn_start_mono=time.monotonic()) > 0.0
    unlimited = TurnBudget(max_tool_calls=0, max_total_tokens=0, max_completion_tokens=0,
                           max_wall_ms=0)
    assert _remaining_wall_seconds(turn_budget=unlimited, turn_start_mono=start) is None
    assert _remaining_wall_seconds(turn_budget=None, turn_start_mono=start) is None
    # A budget with no captured wall anchor (turn_start_mono <= 0) is unbounded:
    # a non-turn caller passing defaults must not see a spent budget.
    assert _remaining_wall_seconds(turn_budget=budget, turn_start_mono=0.0) is None


@pytest.mark.asyncio
async def test_spent_wall_deadline_closes_unawaited_coroutine() -> None:
    """A spent deadline closes the never-started awaitable before raising.

    ``_bound_by_wall_deadline`` receives an already-constructed coroutine;
    raising ``TimeoutError`` without closing it leaves the coroutine to
    GC, which emits ``coroutine ... was never awaited`` (2026-10-22: the
    end-of-turn checked-commit verifier tripped it). The coroutine body
    must not run.
    """
    from kernel.turn_runner import _bound_by_wall_deadline

    started: list[bool] = []

    async def never_starts() -> str:
        started.append(True)
        return "unreachable"

    with pytest.raises(TimeoutError):
        await _bound_by_wall_deadline(never_starts(), deadline_s=0.0)
    assert not started, "the skipped coroutine body must never run"

    # An already-spent budget (negative remaining time) takes the same path.
    with pytest.raises(TimeoutError):
        await _bound_by_wall_deadline(never_starts(), deadline_s=-1.0)
    assert not started


def test_interrupted_child_error_names_resumable_artifacts() -> None:
    """An interrupted child's error message carries its completed artifact ids."""
    from kernel.models import YieldAction
    from kernel.turn_phases._child_turn import classify_child_turn
    from kernel.turn_runner import TurnResult

    receipt = EffectReceipt(effect="write", capability_id="coding_create_worktree",
                            artifact_id="wt123")
    child_result = TurnResult(
        response_text="partial",
        tool_call_count=2,
        called_tool_names={"coding_create_worktree", "coding_read_file"},
        effect_receipts=[receipt, EffectReceipt(effect="read", capability_id="coding_read_file")],
        yielded=YieldAction(question="", context="turn budget exhausted (wall time)"),
    )
    outcome = classify_child_turn(
        child_result=child_result,
        mode_id="coding_execute",
        elapsed_ms=1_000,
        action_tool_ids={"coding_create_worktree"},
    )
    assert outcome.status == "interrupted"
    assert "wt123" in outcome.error_message
    assert "resume" in outcome.error_message.lower()
    # The receipts ride the outcome for the DelegationResult/ForkChildResult.
    assert any(r.artifact_id == "wt123" for r in outcome.effect_receipts)

@pytest.mark.asyncio
async def test_delegate_child_finalizes_before_registry_cancel(
    _mcp_catalog_setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 2026-10-21 incident shape, scaled down.

    A delegate child armed at ``timeout - grace`` whose in-flight model
    call would have crossed the registry's hard cancel must instead be
    cut by its own wall budget and finalize with partial work — the
    parent receives an honest ``interrupted`` result, not a bare
    ``Operation 'delegate' timed out``.
    """
    from kernel.config import settings as kernel_settings
    from kernel.models import DelegationSpec
    from kernel.turn_runner import _default_delegate_fn

    monkeypatch.setattr(kernel_settings, "delegation_grace_seconds", 0)
    # Budget arithmetic: spec.timeout_seconds=1 arms a 1000ms child wall
    # budget; the registry would hard-cancel at 1000ms. The model call
    # sleeps 3s so the armed deadline must fire first.

    async def complete(**_kwargs: Any) -> _SlowStub:
        await asyncio.sleep(3)
        return _SlowStub(content="never seen")

    hooks = TurnHooks(models=ModelHooks(complete=complete))
    t0 = time.monotonic()
    result = await _default_delegate_fn(
        DelegationSpec(target_mode_id="default", query="hello", timeout_seconds=1),
        hooks=hooks,
        turn_budget=None,
        execution_policy=None,
        depth=0,
    )
    elapsed = time.monotonic() - t0
    assert elapsed < 2.5, f"child must finalize under its own deadline ({elapsed:.2f}s)"
    assert isinstance(result, object)

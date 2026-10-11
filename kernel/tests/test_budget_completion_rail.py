"""Tests for ``budget_forces_final`` completion-based token accounting.

Covers the ``max_completion_tokens`` rail: the per-turn budget measures
actual model work (output tokens) rather than cumulative context re-send,
so a turn with a large prompt re-accumulation does not exhaust the budget
while the model has produced little real output.
"""

from __future__ import annotations

import time

import pytest
from kernel.models import TurnBudget
from kernel.turn_phases._turn_state import _TurnState, budget_forces_final


def _state(**overrides: int) -> _TurnState:
    defaults: dict[str, int] = {
        "total_completion_tokens": 0,
        "total_tokens": 0,
        "total_tool_calls": 0,
    }
    defaults.update(overrides)
    return _TurnState(**defaults)


def test_completion_rail_fires_when_completion_exceeds_cap() -> None:
    """A high completion count against a low completion cap forces a yield."""
    budget = TurnBudget(max_completion_tokens=1000, max_total_tokens=0, max_wall_ms=0)
    state = _state(total_completion_tokens=1500)

    assert budget_forces_final(state, budget, time.monotonic()) is True


def test_completion_rail_does_not_fire_on_cumulative_prompt_growth() -> None:
    """A huge cumulative total with small completion does NOT trip the
    completion rail — context re-send is not treated as model work."""
    budget = TurnBudget(max_completion_tokens=1000, max_total_tokens=0, max_wall_ms=0)
    state = _state(total_completion_tokens=100, total_tokens=5_000_000)

    assert budget_forces_final(state, budget, time.monotonic()) is False


def test_unlimited_completion_never_fires() -> None:
    """max_completion_tokens=0 means unlimited."""
    budget = TurnBudget(max_completion_tokens=0, max_total_tokens=0, max_wall_ms=0)
    state = _state(total_completion_tokens=99_999_999)

    assert budget_forces_final(state, budget, time.monotonic()) is False


def test_cumulative_rail_still_fires_as_fallback() -> None:
    """The cumulative ``max_total_tokens`` rail still works for hosts that
    want a whole-turn envelope, independent of the completion rail."""
    budget = TurnBudget(max_completion_tokens=0, max_total_tokens=1000, max_wall_ms=0)
    state = _state(total_completion_tokens=100, total_tokens=1500)

    assert budget_forces_final(state, budget, time.monotonic()) is True


def test_none_budget_never_fires() -> None:
    """No budget wired means no forced yield from the token rails."""
    assert (
        budget_forces_final(_state(total_completion_tokens=10_000), None, time.monotonic()) is False
    )


def test_total_rail_excludes_cached_prefix() -> None:
    """Cached prompt prefixes are not new work: a long cache-hitting turn
    must not exhaust ``max_total_tokens`` (2026-10-10 p920: ~10 calls of
    ~100k mostly-cached prompt tripped the 1M rail without converging)."""
    budget = TurnBudget(max_completion_tokens=0, max_total_tokens=1000, max_wall_ms=0)
    state = _state(total_tokens=5000)
    state.total_cached_tokens = 4500

    assert budget_forces_final(state, budget, time.monotonic()) is False

    state.total_cached_tokens = 3900
    assert budget_forces_final(state, budget, time.monotonic()) is True


@pytest.mark.asyncio
async def test_forced_final_completion_accumulates_cached_tokens() -> None:
    """The forced no-tools final call must fold cached tokens too, or the
    processed-token rail would count its cached prefix as new work."""
    import logging
    from types import SimpleNamespace
    from typing import Any
    from unittest.mock import MagicMock

    from kernel.turn_runner import _force_final_content_completion

    async def complete(**_: Any) -> SimpleNamespace:
        return SimpleNamespace(
            content="Final answer.",
            tool_calls=None,
            prompt_tokens=1000,
            completion_tokens=50,
            total_tokens=1050,
            cached_tokens=900,
            reasoning_tokens=0,
            cache_creation_tokens=0,
            duration_ms=0.0,
            latency_ms=0,
            usage=None,
        )

    state = _state()
    content, _ = await _force_final_content_completion(
        state=state,
        hooks=MagicMock(),
        mode_id="default",
        content="",
        content_source="",
        messages=[],
        effective_model_def=SimpleNamespace(model_id="m", provider="p"),
        complete=complete,
        record_model_usage=lambda *_a, **_k: None,
        log=logging.getLogger(__name__),
    )

    assert content == "Final answer."
    assert state.total_cached_tokens == 900

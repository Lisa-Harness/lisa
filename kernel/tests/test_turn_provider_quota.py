"""Provider-quota graceful finalization in the turn loop.

Regression for the 2026-10-31 00:48 incident: a session-quota 429
(``ProviderGatewayError``) escaped ``run_turn`` whole, failed the turn, and
discarded every receipt the iteration had captured — the same discard class
the stream-stall graceful finalize (2026-10-26 14:20) was written to prevent.
A 429 caught by ``_iteration_completion_or_expiry`` must instead break the
loop to finalization with the partial-work diagnostic, and any other
provider failure must keep the historical propagate contract.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
from kernel.provider_contracts import ProviderGatewayError
from kernel.system_segments import SystemPromptSegments
from kernel.turn_phases._turn_state import _TurnState
from kernel.turn_runner import _iteration_completion_or_expiry


def _gateway_error(status_code: int) -> ProviderGatewayError:
    return ProviderGatewayError(
        provider="ollama",
        model_id="deepseek-v4.1-flash:cloud",
        display_name="deepseek",
        message="you have reached your session usage limit",
        status_code=status_code,
    )


def _hooks_and_seams(monkeypatch: pytest.MonkeyPatch, exc: Exception) -> None:
    async def _failing_completion(**_kwargs: Any) -> None:
        raise exc

    monkeypatch.setattr(
        "kernel.turn_runner._model_completion_for_iteration", _failing_completion
    )


def _call(state: _TurnState) -> Any:
    return _iteration_completion_or_expiry(
        state=state,
        hooks=Any,
        mode_id="coding_review",
        query="q",
        max_tool_iterations=10,
        messages=[],
        effective_model_def=None,
        effective_tools=None,
        sys_segments=SystemPromptSegments(),
        mode_budget_tokens=1024,
        emit_stage=None,  # type: ignore[arg-order]
        log=logging.getLogger("test"),
    )


@pytest.mark.asyncio
async def test_quota_429_finalizes_with_partial_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 429 marks the quota flag and returns None instead of escaping."""

    _hooks_and_seams(monkeypatch, _gateway_error(429))
    state = _TurnState()
    state.iteration = 3
    state.total_tool_calls = 7

    result = await _call(state)

    assert result is None
    assert state.provider_quota_exhausted is True
    assert "ollama:deepseek-v4.1-flash:cloud" in state.provider_quota_detail
    assert state.wall_budget_expired is False
    assert state.stream_stalled is False


@pytest.mark.asyncio
async def test_non_quota_provider_errors_still_propagate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every other status keeps the historical propagate contract."""

    for status in (0, 400, 408, 500, 503):
        _hooks_and_seams(monkeypatch, _gateway_error(status))
        state = _TurnState()
        with pytest.raises(ProviderGatewayError) as caught:
            await _call(state)
        assert caught.value.status_code == status
        assert state.provider_quota_exhausted is False


def test_quota_diagnostic_names_cause_and_partial_work() -> None:
    """The finalization diagnostic reports provider, iteration, and receipts."""

    from kernel.turn_runner import _build_provider_quota_diagnostic

    state = _TurnState()
    state.iteration = 3
    state.total_tool_calls = 7
    state.provider_quota_detail = "ollama:deepseek-v4.1-flash:cloud — session usage limit"

    content = _build_provider_quota_diagnostic(
        state=state, mode_id="coding_review", log=logging.getLogger("test")
    )

    assert "quota or rate limit" in content
    assert "ollama:deepseek-v4.1-flash:cloud" in content
    assert "coding_review" in content
    assert "7 tool call(s)" in content
    assert "preserved in the thread" in content


def test_quota_loop_condition_breaks_to_finalization() -> None:
    """The loop condition breaks on the quota flag next to its siblings."""

    import inspect

    from kernel.turn_runner import _run_turn_inner

    source = inspect.getsource(_run_turn_inner)
    loop_condition = source[
        source.index("while (") : source.index("_state.iteration += 1")
    ]
    assert "not _state.provider_quota_exhausted" in loop_condition
    assert "not _state.wall_budget_expired" in loop_condition
    assert "not _state.stream_stalled" in loop_condition

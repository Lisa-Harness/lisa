"""Tests for the identical tool-batch breaker in the turn loop.

Reproduction of the 2026-10-28 researcher loop: kimi-k3 re-issued one
``thread_read_recent`` + ``thread_search`` batch with unchanged arguments 33
times until the process was killed. Identical calls return identical
results, so the kernel refuses the batch after
``max_identical_tool_batch_repeats`` repeats and stops the loop with a
forced synthesis on the next one.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import pytest
from kernel.config import settings
from kernel.turn_phases._turn_state import _TurnState
from kernel.turn_runner import (
    _guard_repeated_tool_batch,
    _tool_batch_signature,
    run_turn,
)

_LOG = logging.getLogger(__name__)


def _call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {"id": call_id, "function": {"name": name, "arguments": json.dumps(arguments)}}


def _batch(prefix: str) -> list[dict[str, Any]]:
    return [
        _call(f"{prefix}a", "thread.thread_read_recent", {"include_transcript": False, "limit": 3}),
        _call(f"{prefix}b", "thread.thread_search", {"limit": 5, "query": "turn allocation"}),
    ]


def _guard(state: _TurnState, batch: list[dict[str, Any]], messages: list, **kw: Any) -> str:
    return _guard_repeated_tool_batch(
        state=state,
        tool_calls=batch,
        synthesized_forced=kw.get("synthesized_forced", False),
        messages=messages,
        log=_LOG,
    )


def test_signature_ignores_call_ids_key_order_and_call_order() -> None:
    reordered = [
        {
            "id": "z",
            "function": {
                "name": "thread.thread_search",
                "arguments": '{"query": "turn allocation", "limit": 5}',
            },
        },
        _call("y", "thread.thread_read_recent", {"limit": 3, "include_transcript": False}),
    ]
    assert _tool_batch_signature(_batch("1")) == _tool_batch_signature(reordered)


def test_signature_distinguishes_arguments() -> None:
    changed = [_call("a", "thread.thread_search", {"limit": 5, "query": "other"})]
    assert _tool_batch_signature(_batch("1")) != _tool_batch_signature(changed)


def test_guard_dispatches_then_refuses_then_stops(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "max_identical_tool_batch_repeats", 2)
    state = _TurnState()
    messages: list[dict[str, Any]] = []

    assert _guard(state, _batch("1"), messages) == "dispatch"
    assert _guard(state, _batch("2"), messages) == "dispatch"
    assert _guard(state, _batch("3"), messages) == "dispatch"
    assert messages == []

    assert _guard(state, _batch("4"), messages) == "refuse"
    assert [m["tool_call_id"] for m in messages] == ["4a", "4b"]
    assert all(m["role"] == "tool" and "Not executed" in m["content"] for m in messages)
    assert not state.repeated_tool_batch_tripped

    assert _guard(state, _batch("5"), messages) == "stop"
    assert [m["tool_call_id"] for m in messages[2:]] == ["5a", "5b"]
    assert state.repeated_tool_batch_tripped


def test_a_different_batch_resets_the_count(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "max_identical_tool_batch_repeats", 1)
    state = _TurnState()
    messages: list[dict[str, Any]] = []
    other = [_call("o", "thread.thread_search", {"query": "something else"})]

    assert _guard(state, _batch("1"), messages) == "dispatch"
    assert _guard(state, other, messages) == "dispatch"
    assert _guard(state, _batch("2"), messages) == "dispatch"
    assert _guard(state, _batch("3"), messages) == "dispatch"
    assert _guard(state, _batch("4"), messages) == "refuse"


def test_kernel_synthesized_batches_are_never_counted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "max_identical_tool_batch_repeats", 1)
    state = _TurnState()
    for i in range(4):
        assert _guard(state, _batch(str(i)), [], synthesized_forced=True) == "dispatch"


def test_zero_limit_disables_the_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "max_identical_tool_batch_repeats", 0)
    state = _TurnState()
    for i in range(10):
        assert _guard(state, _batch(str(i)), []) == "dispatch"


@dataclass(frozen=True, slots=True)
class _StubCompletion:
    content: str
    tool_calls: list[dict[str, Any]] | None = None
    usage: dict[str, Any] | None = None


class _CountingPool:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def list_tools(self, *, allowed_server_ids: set[str] | None = None) -> list[Any]:
        return []

    async def call_tool(self, tool_id: str, _args: dict[str, Any]) -> Any:
        from kernel.tools.mcp_client import MCPToolResult

        self.calls.append(tool_id)
        return MCPToolResult(tool_id=tool_id, output_text="same result")

    async def __aenter__(self) -> _CountingPool:
        return self

    async def __aexit__(self, *_a: object) -> None:
        pass

    async def start(self) -> None:
        pass

    async def aclose(self) -> None:
        pass


@pytest.mark.asyncio
async def test_looping_model_is_stopped_and_forced_to_synthesize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from turn_hooks_fold import build_hooks

    monkeypatch.setattr(settings, "max_identical_tool_batch_repeats", 2)
    iteration = 0
    tool_free_calls = 0

    async def looping(**kwargs: Any) -> _StubCompletion:
        nonlocal iteration, tool_free_calls
        if kwargs.get("tools") is None:
            tool_free_calls += 1
            return _StubCompletion(content="Synthesized from what I have.")
        iteration += 1
        return _StubCompletion(content="", tool_calls=_batch(str(iteration)))

    pool = _CountingPool()
    hooks = build_hooks(
        defaults={"models": {"complete": looping}, "recall": {"enabled": False}},
        mcp_tool_pool=pool,
    )
    result = await run_turn(mode_id="default", query="loop", hooks=hooks)

    # 1 original + 2 allowed repeats dispatched; the 3rd repeat is refused
    # and the 4th stops the loop.
    assert len(pool.calls) == 6
    assert iteration == 5
    assert tool_free_calls == 1
    assert result.response_text.startswith("Synthesized from what I have.")

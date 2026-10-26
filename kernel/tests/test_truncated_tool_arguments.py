"""Turn-loop guard for length-cut (unparseable) tool-call arguments.

The 2026-09-23 finances incident: a ``finish_reason='length'`` completion
cut a multi-KB ``write_file`` call mid-argument. The translator's old
contract returned ``{}`` for unparseable JSON and the tool DISPATCHED
ANYWAY — the storage backend wrote a truncated ledger fragment and
returned ``ok: true``, so no error surfaced anywhere in the loop.

``openai_tool_call_to_mcp`` now marks the payload with the
``TRUNCATED_TOOL_ARGUMENTS`` sentinel; the turn loop refuses the dispatch,
feeds an explicit error to the model, and counts the attempt so budget
guards still work. These tests pin the turn-loop half of that contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from kernel.turn_hooks import (
    ModelHooks,
    RecallHooks,
    ToolHooks,
)
from kernel.turn_runner import TurnHooks, run_turn


@dataclass(frozen=True, slots=True)
class _StubCompletion:
    content: str
    tool_calls: list[dict[str, Any]] | None = None


def _truncated_tool_call() -> dict[str, Any]:
    """A write_file call whose arguments JSON is cut mid-content-string."""

    cut = '{"path": "ledgers/2024.beancount", "content": "2024-01-02 open Assets:GL1000'
    return {
        "id": "call_cut",
        "function": {"name": "lisa-storage.write_file", "arguments": cut},
    }


@pytest.mark.asyncio
async def test_truncated_arguments_never_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A length-cut tool call must not reach the MCP pool."""

    dispatched: list[str] = []

    class _RecordingPool:
        async def list_tools(self, *args: Any, **kwargs: Any) -> list[Any]:
            return []

        async def call_tool(
            self, tool_id: str, _args: Any, **_kwargs: Any
        ) -> Any:  # pragma: no cover
            dispatched.append(tool_id)
            raise AssertionError("pool dispatch must not run for a truncated call")

    call_log: list[dict[str, Any]] = []

    async def two_step(**kwargs: Any) -> _StubCompletion:
        call_log.append(kwargs)
        if len(call_log) == 1:
            return _StubCompletion(content="", tool_calls=[_truncated_tool_call()])
        return _StubCompletion(content="Recovered.")

    hooks = TurnHooks(
        models=ModelHooks(complete=two_step),
        tools=ToolHooks(mcp_tool_pool=_RecordingPool()),
        recall=RecallHooks(enabled=False),
    )

    # Documented kernel test seam (see kernel/tests/test_streaming.py): the
    # query "write the ledger" triggers the write-intent routing hint, which
    # forces a transition and adds model calls these tests don't stub for.
    with monkeypatch.context() as mp:
        mp.setattr("kernel.turn_runner._compute_capability_routing_hint", lambda *a, **k: "")
        result = await run_turn(mode_id="default", query="write the ledger", hooks=hooks)

    assert dispatched == [], "a truncated-arguments call must never dispatch"
    assert result.response_text == "Recovered."


@pytest.mark.asyncio
async def test_truncated_arguments_surface_explicit_tool_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The model receives an error naming the truncation and instructing a re-issue."""

    call_log: list[dict[str, Any]] = []

    async def two_step(**kwargs: Any) -> _StubCompletion:
        call_log.append(kwargs)
        if len(call_log) == 1:
            return _StubCompletion(content="", tool_calls=[_truncated_tool_call()])
        return _StubCompletion(content="Recovered.")

    hooks = TurnHooks(
        models=ModelHooks(complete=two_step),
        recall=RecallHooks(enabled=False),
    )

    # Same documented seam as above: neutralize the write-intent routing hint.
    monkeypatch.setattr("kernel.turn_runner._compute_capability_routing_hint", lambda *a, **k: "")

    result = await run_turn(mode_id="default", query="write the ledger", hooks=hooks)

    assert len(call_log) == 2, "the turn must continue after the refused dispatch"
    tool_messages = [
        m
        for m in call_log[1].get("messages", [])
        if m.get("role") == "tool" and m.get("tool_call_id") == "call_cut"
    ]
    assert tool_messages, "the refused dispatch must surface as a tool result"
    content = str(tool_messages[0].get("content", ""))
    assert "truncated" in content.lower()
    assert "NOT executed" in content, "the error must state the tool did not run"
    # The refused dispatch counts as an attempt (budget guards key on it).
    assert result.tool_call_count == 1

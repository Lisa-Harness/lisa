"""Memory capture and mining handle dedup backend failures consistently."""

from __future__ import annotations

import json
from typing import Any

import pytest
from kernel.models import OperationExecutionContext, ToolResult
from kernel.seams.host_adapter import build_callable_host
from sdk.packs.memory.operations.memory import mine_thread_segment, synthesize_learning


def _context(execute_tool_fn: Any) -> OperationExecutionContext:
    return OperationExecutionContext(
        operation_id="test",
        workspace_id="workspace",
        user_id="alice",
        user_slug="alice",
        surface_id="console",
        memory_scope="personal",
        host=build_callable_host(execute_tool_fn=execute_tool_fn),
    )


def _mining_params() -> mine_thread_segment.Parameters:
    return mine_thread_segment.Parameters(
        turns=[{"turn": 1, "query": "learning"}],
        workspace_id="workspace",
        user_slug="alice",
        surface_id="console",
        memory_scope="personal",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["capture", "mining"])
@pytest.mark.parametrize("response", ["exception", "tool_error", "invalid_json", "envelope"])
async def test_dedup_search_responses(
    operation: str, response: str, caplog: pytest.LogCaptureFixture
) -> None:
    calls: list[Any] = []

    async def execute_tool(tool_id: str, ctx: Any) -> ToolResult:
        assert tool_id == "memory.search"
        calls.append(ctx)
        if response == "exception":
            raise RuntimeError("search offline")
        if response == "tool_error":
            return ToolResult(tool_id=tool_id, error="search offline")
        if response == "invalid_json":
            return ToolResult(tool_id=tool_id, output_text="{")
        return ToolResult(
            tool_id=tool_id,
            output_text=json.dumps({"results": [{"id": 7, "content": "existing"}, "invalid"]}),
        )

    ctx = _context(execute_tool)
    if operation == "capture":
        candidates = await synthesize_learning._dedup_candidates(ctx, "learning")
    else:
        candidates = await mine_thread_segment._dedup_candidates(ctx, _mining_params(), "learning")

    assert calls[0].parameters["query"] == "learning"
    assert calls[0].parameters["user_slug"] == "alice"
    assert calls[0].parameters["workspace_id"] == "workspace"
    assert calls[0].parameters["surface_id"] == "console"
    assert calls[0].parameters["memory_scope"] == "personal"
    if response == "envelope":
        assert candidates == [{"id": 7, "content": "existing"}]
    else:
        assert candidates == []
        assert ("invalid JSON" if response == "invalid_json" else "search offline") in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["capture", "mining"])
@pytest.mark.parametrize("response", ["exception", "tool_error"])
async def test_retirement_failure_is_reported(
    operation: str, response: str, caplog: pytest.LogCaptureFixture
) -> None:
    calls: list[Any] = []

    async def execute_tool(tool_id: str, ctx: Any) -> ToolResult:
        assert tool_id == "memory.retire"
        calls.append(ctx)
        if response == "exception":
            raise RuntimeError("retire offline")
        return ToolResult(tool_id=tool_id, error="retire offline")

    ctx = _context(execute_tool)
    if operation == "capture":
        retired = await synthesize_learning._retire_record(ctx, 7)
    else:
        retired = await mine_thread_segment._retire_record(ctx, _mining_params(), 7)

    assert retired is False
    assert calls[0].parameters["memory_id"] == 7
    assert calls[0].parameters["user_slug"] == "alice"
    assert "retire offline" in caplog.text

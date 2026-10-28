"""``ToolExecutionContext.mcp_tool_call``: the pack-tool → MCP capability seam.

Pack tools (``fetch_page``) delegate to the session's selected capability
implementation through this seam instead of embedding one. The tool
executors bind it to the turn's pool and it must survive the context copy.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from kernel.models import ToolExecutionContext, ToolResult
from kernel.tools.mcp_client import MCPToolResult
from kernel.turn_phases import _turn_state
from kernel.turn_phases._post_turn import _to_tool_context
from kernel.turn_phases._turn_state import mcp_tool_caller, mcp_tool_executor


class _RecordingPool:
    def __init__(self, result: MCPToolResult) -> None:
        self.result = result
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, tool_id: str, arguments: dict[str, Any]) -> MCPToolResult:
        self.calls.append((tool_id, arguments))
        return self.result


@pytest.mark.asyncio
async def test_mcp_tool_caller_adapts_success() -> None:
    pool = _RecordingPool(MCPToolResult(tool_id="fetch.fetch_url", output_text='{"ok": 1}'))

    result = await mcp_tool_caller(pool)("fetch.fetch_url", {"url": "https://x.test"})  # type: ignore[arg-type]

    assert pool.calls == [("fetch.fetch_url", {"url": "https://x.test"})]
    assert result == ToolResult(tool_id="fetch.fetch_url", output_text='{"ok": 1}')


@pytest.mark.asyncio
async def test_mcp_tool_caller_maps_error_to_tool_error() -> None:
    pool = _RecordingPool(
        MCPToolResult(tool_id="fetch.fetch_url", output_text="HTTP 403 (Forbidden)", is_error=True)
    )

    result = await mcp_tool_caller(pool)("fetch.fetch_url", {})  # type: ignore[arg-type]

    assert result.error == "HTTP 403 (Forbidden)"
    assert result.output_text == ""


def test_to_tool_context_carries_mcp_tool_call() -> None:
    async def _call(_tool_id: str, _args: Any) -> ToolResult:
        return ToolResult()

    source = ToolExecutionContext(tool_id="parent", mcp_tool_call=_call)

    assert _to_tool_context("fetch_page", source).mcp_tool_call is _call


@pytest.mark.asyncio
async def test_mcp_tool_executor_binds_seam_for_internal_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Internal tools dispatched by the executor can reach the turn's pool."""
    captured: list[ToolExecutionContext] = []

    async def _fake_execute_tool(tool_id: str, ctx: ToolExecutionContext) -> ToolResult:
        captured.append(ctx)
        return ToolResult(tool_id=tool_id)

    monkeypatch.setattr(
        "kernel.tools.registry.get_tool_catalog",
        lambda: {"fetch_page": SimpleNamespace(implementation_module="pkg.fetch_page")},
    )
    monkeypatch.setattr(_turn_state, "execute_tool", _fake_execute_tool)
    pool = _RecordingPool(MCPToolResult(tool_id="fetch.fetch_url", output_text="{}"))

    await mcp_tool_executor("fetch_page", ToolExecutionContext(), pool)  # type: ignore[arg-type]

    assert len(captured) == 1
    seam = captured[0].mcp_tool_call
    assert seam is not None
    await seam("fetch.fetch_url", {"url": "https://x.test"})
    assert pool.calls == [("fetch.fetch_url", {"url": "https://x.test"})]

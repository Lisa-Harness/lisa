"""Tests for the three Tier 1 MCP servers (guard, storage, content).

Covers tool registration, dispatch, and behavioral contracts using FastMCP's
in-process ``call_tool`` API.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pytest


def _tool_text(result: Any) -> str:
    """Extract text from FastMCP call_tool result."""
    if isinstance(result, tuple) and len(result) >= 1:
        blocks = result[0]
        if blocks and hasattr(blocks[0], "text"):
            return blocks[0].text
        if blocks:
            return str(blocks[0])
    return str(result)


# ── lisa_guard ──────────────────────────────────────────────────────────────


@pytest.fixture
def guard_setup() -> None:
    from sdk.lisa_guard.server import init_backend

    init_backend("passthrough")


@pytest.mark.asyncio
async def test_guard_lists_three_tools(guard_setup: None) -> None:
    from sdk.lisa_guard.server import mcp

    tools = await mcp.list_tools()
    names = {t.name for t in tools}
    assert {"validate_input", "validate_output", "guard_configured"} == names


@pytest.mark.asyncio
async def test_guard_validate_input_passthrough(guard_setup: None) -> None:
    from sdk.lisa_guard.server import mcp

    result = await mcp.call_tool(
        "validate_input",
        {"query": "hello world", "surface_id": "s1"},
    )
    data = json.loads(_tool_text(result))
    assert data["allowed"] is True
    assert data["blocked_reason"] is None
    assert data["sanitized_text"] is None


@pytest.mark.asyncio
async def test_guard_validate_output_passthrough(guard_setup: None) -> None:
    from sdk.lisa_guard.server import mcp

    result = await mcp.call_tool(
        "validate_output",
        {"text": "safe output", "surface_id": "s1"},
    )
    data = json.loads(_tool_text(result))
    assert data["allowed"] is True


@pytest.mark.asyncio
async def test_guard_configured(guard_setup: None) -> None:
    from sdk.lisa_guard.server import mcp

    result = await mcp.call_tool("guard_configured", {})
    data = json.loads(_tool_text(result))
    assert data["configured"] is True
    assert data["backend"] == "passthrough"


# ── lisa_content ───────────────────────────────────────────────────────────


@pytest.fixture
def content_setup() -> None:
    from sdk.lisa_content.server import init_content

    init_content()


@pytest.mark.asyncio
async def test_content_lists_three_tools(content_setup: None) -> None:
    from sdk.lisa_content.server import mcp

    tools = await mcp.list_tools()
    names = {t.name for t in tools}
    assert {"read_authored_text", "list_content_roots", "invalidate_cache"} == names


@pytest.mark.asyncio
async def test_content_read_absent_returns_empty(content_setup: None) -> None:
    from sdk.lisa_content.server import mcp

    result = await mcp.call_tool("read_authored_text", {"relative_path": "nonexistent/file.md"})
    assert _tool_text(result) == ""


@pytest.mark.asyncio
async def test_content_invalidate_cache(content_setup: None) -> None:
    from sdk.lisa_content.server import mcp

    result = await mcp.call_tool("invalidate_cache", {})
    data = json.loads(_tool_text(result))
    assert data["ok"] is True


@pytest.mark.asyncio
async def test_content_list_roots_returns_array(content_setup: None) -> None:
    from sdk.lisa_content.server import mcp

    result = await mcp.call_tool("list_content_roots", {})
    data = json.loads(_tool_text(result))
    assert isinstance(data, list)


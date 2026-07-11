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
    assert {"read_authored_text", "list_content_roots", "invalidate_cache"}.issubset(names)


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


@pytest.mark.asyncio
async def test_content_lists_all_seven_tools(content_setup: None) -> None:
    from sdk.lisa_content.server import mcp

    tools = await mcp.list_tools()
    names = {t.name for t in tools}
    assert {
        "read_authored_text",
        "list_content_roots",
        "list_modes",
        "list_operations",
        "list_tools",
        "invalidate_cache",
        "invalidate_catalog",
    } <= names


@pytest.mark.asyncio
async def test_content_read_traversal_rejected(tmp_path: Path) -> None:
    """Paths with .. segments must not escape content roots."""
    from sdk.lisa_content.server import init_content, mcp

    secret_dir = tmp_path / "secret"
    secret_dir.mkdir(parents=True, exist_ok=True)
    secret_file = secret_dir / "secret.txt"
    secret_file.write_text("should not be readable", encoding="utf-8")

    root = tmp_path / "root"
    (root / "content").mkdir(parents=True, exist_ok=True)

    init_content(extra_roots=[root])
    result = await mcp.call_tool(
        "read_authored_text",
        {"relative_path": "../secret/secret.txt"},
    )
    assert _tool_text(result) == ""


@pytest.mark.asyncio
async def test_content_read_absolute_path_rejected(tmp_path: Path) -> None:
    """Absolute paths must not be accepted."""
    from sdk.lisa_content.server import init_content, mcp

    root = tmp_path / "root"
    (root / "content").mkdir(parents=True, exist_ok=True)

    init_content(extra_roots=[root])
    abs_path = str(tmp_path / "outside.txt")
    (tmp_path / "outside.txt").write_text("outside", encoding="utf-8")
    result = await mcp.call_tool(
        "read_authored_text",
        {"relative_path": abs_path},
    )
    assert _tool_text(result) == ""


@pytest.mark.asyncio
async def test_content_read_returns_file(tmp_path: Path) -> None:
    """A valid relative path within a root should return file content."""
    from sdk.lisa_content.server import init_content, mcp

    root = tmp_path / "root"
    (root / "modes").mkdir(parents=True, exist_ok=True)
    (root / "modes" / "test.md").write_text("hello content", encoding="utf-8")

    init_content(extra_roots=[root])
    result = await mcp.call_tool(
        "read_authored_text",
        {"relative_path": "modes/test.md"},
    )
    assert _tool_text(result) == "hello content"


@pytest.mark.asyncio
async def test_content_cache_invalidates_on_roots_change(tmp_path: Path) -> None:
    """Changing roots must not return stale cached content."""
    from sdk.lisa_content.server import init_content, mcp

    root_a = tmp_path / "root_a"
    (root_a / "content").mkdir(parents=True, exist_ok=True)
    (root_a / "content" / "file.md").write_text("from root A", encoding="utf-8")

    root_b = tmp_path / "root_b"
    (root_b / "content").mkdir(parents=True, exist_ok=True)
    (root_b / "content" / "file.md").write_text("from root B", encoding="utf-8")

    init_content(extra_roots=[root_a])
    result_a = await mcp.call_tool(
        "read_authored_text",
        {"relative_path": "content/file.md"},
    )
    assert _tool_text(result_a) == "from root A"

    init_content(extra_roots=[root_b])
    result_b = await mcp.call_tool(
        "read_authored_text",
        {"relative_path": "content/file.md"},
    )
    assert _tool_text(result_b) == "from root B"


@pytest.mark.asyncio
async def test_content_invalidate_catalog(content_setup: None) -> None:
    from sdk.lisa_content.server import mcp

    result = await mcp.call_tool("invalidate_catalog", {})
    data = json.loads(_tool_text(result))
    assert data["ok"] is True


@pytest.mark.asyncio
async def test_content_list_modes_returns_json(content_setup: None) -> None:
    from sdk.lisa_content.server import mcp

    result = await mcp.call_tool("list_modes", {})
    data = json.loads(_tool_text(result))
    assert isinstance(data, dict)


@pytest.mark.asyncio
async def test_content_list_operations_returns_json(content_setup: None) -> None:
    from sdk.lisa_content.server import mcp

    result = await mcp.call_tool("list_operations", {})
    data = json.loads(_tool_text(result))
    assert isinstance(data, dict)


@pytest.mark.asyncio
async def test_content_list_tools_returns_json(content_setup: None) -> None:
    from sdk.lisa_content.server import mcp

    result = await mcp.call_tool("list_tools", {})
    data = json.loads(_tool_text(result))
    assert isinstance(data, dict)


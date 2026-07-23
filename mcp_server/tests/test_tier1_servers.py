"""Tests for the three Tier 1 MCP servers (guard, storage, content).

Covers tool registration, dispatch, and behavioral contracts using FastMCP's
in-process ``call_tool`` API.
"""

from __future__ import annotations

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
    from mcp_server.guard.server import init_backend

    init_backend("passthrough")


@pytest.mark.asyncio
async def test_guard_lists_three_tools(guard_setup: None) -> None:
    from mcp_server.guard.server import mcp

    tools = await mcp.list_tools()
    names = {t.name for t in tools}
    assert {"validate_input", "validate_output", "guard_configured"} == names


@pytest.mark.asyncio
async def test_guard_validate_input_passthrough(guard_setup: None) -> None:
    from mcp_server.guard.server import mcp

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
    from mcp_server.guard.server import mcp

    result = await mcp.call_tool(
        "validate_output",
        {"text": "safe output", "surface_id": "s1"},
    )
    data = json.loads(_tool_text(result))
    assert data["allowed"] is True


@pytest.mark.asyncio
async def test_guard_configured(guard_setup: None) -> None:
    from mcp_server.guard.server import mcp

    result = await mcp.call_tool("guard_configured", {})
    data = json.loads(_tool_text(result))
    assert data["configured"] is True
    assert data["backend"] == "passthrough"


# ── lisa_content ───────────────────────────────────────────────────────────


@pytest.fixture
def content_setup() -> None:
    from mcp_server.content.server import init_content, reset_content_state

    reset_content_state()
    init_content()


@pytest.mark.asyncio
async def test_content_lists_three_tools(content_setup: None) -> None:
    from mcp_server.content.server import mcp

    tools = await mcp.list_tools()
    names = {t.name for t in tools}
    assert {"read_authored_text", "list_content_roots", "invalidate_cache"} == names


@pytest.mark.asyncio
async def test_content_read_absent_returns_empty(content_setup: None) -> None:
    from mcp_server.content.server import mcp

    result = await mcp.call_tool("read_authored_text", {"relative_path": "nonexistent/file.md"})
    assert _tool_text(result) == ""


@pytest.mark.asyncio
async def test_content_invalidate_cache(content_setup: None) -> None:
    from mcp_server.content.server import mcp

    result = await mcp.call_tool("invalidate_cache", {})
    data = json.loads(_tool_text(result))
    assert data["ok"] is True


@pytest.mark.asyncio
async def test_content_list_roots_returns_array(content_setup: None) -> None:
    from mcp_server.content.server import mcp

    result = await mcp.call_tool("list_content_roots", {})
    data = json.loads(_tool_text(result))
    assert isinstance(data, list)


@pytest.mark.asyncio
async def test_content_lists_all_three_tools(content_setup: None) -> None:
    from mcp_server.content.server import mcp

    tools = await mcp.list_tools()
    names = {t.name for t in tools}
    assert {
        "read_authored_text",
        "list_content_roots",
        "invalidate_cache",
    } == names


@pytest.mark.asyncio
async def test_content_read_traversal_rejected(tmp_path: Path) -> None:
    """Paths with .. segments must not escape content roots."""
    from mcp_server.content.server import init_content, mcp

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
    from mcp_server.content.server import init_content, mcp

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
    from mcp_server.content.server import init_content, mcp

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
    from mcp_server.content.server import init_content, mcp

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


# ── lisa_catalog ────────────────────────────────────────────────────────────


@pytest.fixture
def catalog_setup() -> None:
    from mcp_server.catalog.server import init_catalog, reset_catalog_state

    reset_catalog_state()
    init_catalog()


@pytest.mark.asyncio
async def test_catalog_lists_all_tools(catalog_setup: None) -> None:
    from mcp_server.catalog.server import mcp

    tools = await mcp.list_tools()
    names = {t.name for t in tools}
    assert {
        "list_content_roots",
        "list_modes",
        "list_operations",
        "list_tools",
        "invalidate_catalog",
        "list_catalog_diagnostics",
    } == names


@pytest.mark.asyncio
async def test_catalog_invalidate_catalog(catalog_setup: None) -> None:
    from mcp_server.catalog.server import mcp

    result = await mcp.call_tool("invalidate_catalog", {})
    data = json.loads(_tool_text(result))
    assert data["ok"] is True


@pytest.mark.asyncio
async def test_catalog_list_modes_returns_json(catalog_setup: None) -> None:
    from mcp_server.catalog.server import mcp

    result = await mcp.call_tool("list_modes", {})
    data = json.loads(_tool_text(result))
    assert isinstance(data, dict)


@pytest.mark.asyncio
async def test_catalog_list_operations_returns_json(catalog_setup: None) -> None:
    from mcp_server.catalog.server import mcp

    result = await mcp.call_tool("list_operations", {})
    data = json.loads(_tool_text(result))
    assert isinstance(data, dict)


@pytest.mark.asyncio
async def test_catalog_list_tools_returns_json(catalog_setup: None) -> None:
    from mcp_server.catalog.server import mcp

    result = await mcp.call_tool("list_tools", {})
    data = json.loads(_tool_text(result))
    assert isinstance(data, dict)


# ── lisa_catalog diagnostics ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_catalog_diagnostics_returns_report(catalog_setup: None) -> None:
    """list_catalog_diagnostics returns a JSON report with expected keys."""
    from mcp_server.catalog.server import mcp

    result = await mcp.call_tool("list_catalog_diagnostics", {})
    data = json.loads(_tool_text(result))
    assert "root_discovery_failures" in data
    assert "mode_issues" in data
    assert "operation_issues" in data
    assert "tool_issues" in data
    assert "summary" in data
    assert "total_roots" in data["summary"]
    assert "total_issues" in data["summary"]


@pytest.mark.asyncio
async def test_catalog_diagnostics_detects_missing_mode_config(tmp_path: Path, monkeypatch) -> None:
    """A mode directory without _mode.md should be reported as an issue."""
    from mcp_server.catalog import server
    from mcp_server.catalog.server import init_catalog, mcp

    root = tmp_path / "root"
    mode_dir = root / "modes" / "broken_mode"
    mode_dir.mkdir(parents=True, exist_ok=True)

    init_catalog(extra_roots=[root])
    monkeypatch.setattr(server, "_all_roots", lambda: [root])
    result = await mcp.call_tool("list_catalog_diagnostics", {})
    data = json.loads(_tool_text(result))
    mode_issues = data["mode_issues"]
    assert len(mode_issues) >= 1
    assert any(issue["mode_dir"] == "broken_mode" for issue in mode_issues)
    assert data["summary"]["total_issues"] >= 1


@pytest.mark.asyncio
async def test_catalog_diagnostics_detects_missing_operation_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    """A .py operation module without a .yml sibling should be reported."""
    from mcp_server.catalog import server
    from mcp_server.catalog.server import init_catalog, mcp

    root = tmp_path / "root"
    ops_dir = root / "operations" / "research"
    ops_dir.mkdir(parents=True, exist_ok=True)
    (ops_dir / "orphan.py").write_text("# no yml sibling\n", encoding="utf-8")

    init_catalog(extra_roots=[root])
    monkeypatch.setattr(server, "_all_roots", lambda: [root])
    result = await mcp.call_tool("list_catalog_diagnostics", {})
    data = json.loads(_tool_text(result))
    op_issues = data["operation_issues"]
    assert len(op_issues) >= 1
    assert any("orphan.py" in issue["path"] for issue in op_issues)


@pytest.mark.asyncio
async def test_catalog_diagnostics_detects_missing_tool_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    """A .py tool module without a .tool.yml sibling should be reported."""
    from mcp_server.catalog import server
    from mcp_server.catalog.server import init_catalog, mcp

    root = tmp_path / "root"
    tools_dir = root / "tools" / "broken_tool"
    tools_dir.mkdir(parents=True, exist_ok=True)
    (tools_dir / "broken_tool.py").write_text("# no tool yml sibling\n", encoding="utf-8")

    init_catalog(extra_roots=[root])
    monkeypatch.setattr(server, "_all_roots", lambda: [root])
    result = await mcp.call_tool("list_catalog_diagnostics", {})
    data = json.loads(_tool_text(result))
    tool_issues = data["tool_issues"]
    assert len(tool_issues) >= 1
    assert any("broken_tool.py" in issue["path"] for issue in tool_issues)


@pytest.mark.asyncio
async def test_catalog_diagnostics_clean_report(tmp_path: Path, monkeypatch) -> None:
    """A valid content root with proper files should produce zero issues."""
    from mcp_server.catalog import server
    from mcp_server.catalog.server import init_catalog, mcp

    root = tmp_path / "root"
    mode_dir = root / "modes" / "good_mode"
    mode_dir.mkdir(parents=True, exist_ok=True)
    (mode_dir / "_mode.md").write_text(
        '---\nmode_id: good_mode\nmode_kind: chat\nstart_stage_id: ""\ndescription: "ok"\n---\n# Good\n',
        encoding="utf-8",
    )

    init_catalog(extra_roots=[root])
    monkeypatch.setattr(server, "_all_roots", lambda: [root])
    result = await mcp.call_tool("list_catalog_diagnostics", {})
    data = json.loads(_tool_text(result))
    assert data["mode_issues"] == []
    assert data["operation_issues"] == []
    assert data["tool_issues"] == []
    assert data["summary"]["total_issues"] == 0

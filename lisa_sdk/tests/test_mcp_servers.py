"""Tests for the four Kernel-as-MCP servers (Part 4).

Covers tool registration and dispatch for lisa-context, lisa-operations,
lisa-search, and lisa-surf using FastMCP's in-process ``call_tool`` API.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


def _tool_text(result: Any) -> str:
    """Extract text from FastMCP call_tool result.

    ``call_tool`` returns ``(content_blocks, result_dict)``. The text is in
    the first content block's ``.text`` attribute, or in ``result[1]['result']``.
    """
    if isinstance(result, tuple) and len(result) >= 1:
        blocks = result[0]
        if blocks and hasattr(blocks[0], "text"):
            return blocks[0].text
        if blocks:
            return str(blocks[0])
    return str(result)


# ── lisa_context ───────────────────────────────────────────────────────────


@pytest.fixture
def context_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data_dir = tmp_path / "ctx_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    # Initialize the module-level store from the resolved data directory.
    # The server derives memory.db from --data-dir; no env-var fallback.
    from lisa_sdk.lisa_context import server as ctx_server

    ctx_server.init_store(data_dir)
    return data_dir / "memory.db"


@pytest.mark.asyncio
async def test_context_add_memory(context_env: Path) -> None:
    from lisa_sdk.lisa_context.server import mcp

    result = await mcp.call_tool(
        "add_memory",
        {"user_slug": "alice", "content": "hello from test"},
    )
    assert "Memory stored with ID" in _tool_text(result)


@pytest.mark.asyncio
async def test_context_search_memory(context_env: Path) -> None:
    from lisa_sdk.lisa_context.server import mcp

    await mcp.call_tool(
        "add_memory",
        {"user_slug": "alice", "content": "the quick brown fox"},
    )
    result = await mcp.call_tool(
        "search_memory",
        {"user_slug": "alice", "query": "quick fox"},
    )
    assert "quick brown fox" in _tool_text(result)


@pytest.mark.asyncio
async def test_context_search_no_match(context_env: Path) -> None:
    from lisa_sdk.lisa_context.server import mcp

    result = await mcp.call_tool(
        "search_memory",
        {"user_slug": "nobody", "query": "zzzzz"},
    )
    assert "No relevant memories" in _tool_text(result)


@pytest.mark.asyncio
async def test_context_list_memories_json(context_env: Path) -> None:
    from lisa_sdk.lisa_context.server import mcp

    await mcp.call_tool(
        "add_memory",
        {"user_slug": "alice", "content": "json test"},
    )
    result = await mcp.call_tool(
        "list_memories",
        {"user_slug": "alice"},
    )
    parsed = json.loads(_tool_text(result))
    assert isinstance(parsed, list)
    assert len(parsed) >= 1
    assert parsed[0]["content"] == "json test"


@pytest.mark.asyncio
async def test_context_get_surface_context(context_env: Path) -> None:
    from lisa_sdk.lisa_context.server import mcp

    await mcp.call_tool(
        "add_memory",
        {"user_slug": "alice", "content": "context line"},
    )
    result = await mcp.call_tool(
        "get_surface_context",
        {"user_slug": "alice"},
    )
    assert "context line" in _tool_text(result)


@pytest.mark.asyncio
async def test_context_lists_four_tools() -> None:
    from lisa_sdk.lisa_context.server import mcp

    tools = await mcp.list_tools()
    names = {t.name for t in tools}
    assert {"add_memory", "search_memory", "list_memories", "get_surface_context"} <= names


# ── lisa_operations ─────────────────────────────────────────────────────────


@pytest.fixture
def operations_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    from kernel.ipsar_fsm import register_content_roots, register_operation_callbacks
    from kernel.operations.registry import execute_operation, get_operation_catalog

    core_root = (Path(__file__).resolve().parents[2] / "sdk_packs" / "core").resolve()
    register_content_roots(core_root)
    register_operation_callbacks(get_operation_catalog, execute_operation)


def test_operations_build_tool_catalog(operations_setup: None) -> None:
    from lisa_sdk.lisa_operations.server import build_tool_catalog

    catalog = build_tool_catalog()
    assert isinstance(catalog, dict)
    assert len(catalog) > 0
    # Every entry has description and input_schema
    for op_id, info in catalog.items():
        assert "description" in info
        assert "input_schema" in info


@pytest.mark.asyncio
async def test_operations_registers_operation_tools(operations_setup: None) -> None:
    from kernel.operations.registry import get_operation_catalog
    from lisa_sdk.lisa_operations.server import mcp

    tools = await mcp.list_tools()
    tool_names = {tool.name for tool in tools}
    operation_ids = set(get_operation_catalog().keys())
    assert operation_ids
    assert operation_ids <= tool_names


@pytest.mark.asyncio
async def test_operations_call_unknown_tool_fails(operations_setup: None) -> None:
    from lisa_sdk.lisa_operations.server import mcp

    with pytest.raises(Exception):
        await mcp.call_tool("nonexistent_op_xyz", {})


@pytest.mark.asyncio
async def test_operations_call_registered_operation(operations_setup: None) -> None:
    from kernel.operations.registry import get_operation_catalog
    from lisa_sdk.lisa_operations.server import mcp

    operation_id = next(iter(get_operation_catalog().keys()))
    result = await mcp.call_tool(operation_id, {})
    # Registered operation tool should execute and return text (possibly an error string
    # if required context/params are missing, but the MCP tool itself is callable).
    assert isinstance(_tool_text(result), str)


# ── lisa_search ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_search_web_search_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    from lisa_sdk.lisa_search import server as search_server
    from lisa_sdk.lisa_search import web_search as ws

    monkeypatch.setattr(ws, "is_configured", lambda: False)

    async def _raise(query: str, client: Any = None) -> Any:
        raise RuntimeError("not configured")

    monkeypatch.setattr(ws, "search", _raise)
    monkeypatch.setattr(search_server, "_web_search", ws)
    from lisa_sdk.lisa_search.server import mcp

    result = await mcp.call_tool("web_search", {"query": "test"})
    assert "not configured" in _tool_text(result)


@pytest.mark.asyncio
async def test_search_configured_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    from lisa_sdk.lisa_search import server as search_server
    from lisa_sdk.lisa_search import web_search as ws

    monkeypatch.setattr(ws, "is_configured", lambda: True)
    monkeypatch.setattr(search_server, "_web_search", ws)
    from lisa_sdk.lisa_search.server import mcp

    result = await mcp.call_tool("web_search_configured", {})
    parsed = json.loads(_tool_text(result))
    assert parsed["configured"] is True


def test_search_lists_tools() -> None:
    from lisa_sdk.lisa_search.server import mcp

    tools = asyncio_get_tools(mcp)
    names = {t.name for t in tools}
    assert {"web_search", "web_search_configured"} <= names


# ── lisa_surf ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_surf_url_creates_session() -> None:
    from lisa_sdk.lisa_surf.server import mcp

    result = await mcp.call_tool(
        "surf_url",
        {
            "workspace_id": "ws-1",
            "surface_id": "surf-1",
            "url": "https://example.com",
        },
    )
    parsed = json.loads(_tool_text(result))
    assert parsed["url"] == "https://example.com"
    assert "session_id" in parsed
    assert parsed["summary"]  # non-empty


@pytest.mark.asyncio
async def test_surf_get_session_no_blob_io() -> None:
    from lisa_sdk.lisa_surf.server import mcp

    result = await mcp.call_tool(
        "get_session_state",
        {"workspace_id": "ws-1", "surface_id": "surf-1"},
    )
    parsed = json.loads(_tool_text(result))
    assert parsed["session"] is None


@pytest.mark.asyncio
async def test_surf_flush_session_no_blob_io() -> None:
    from lisa_sdk.lisa_surf.server import mcp

    result = await mcp.call_tool(
        "flush_session",
        {"workspace_id": "ws-1", "surface_id": "surf-1"},
    )
    parsed = json.loads(_tool_text(result))
    assert parsed["deleted"] is False


def test_surf_lists_three_tools() -> None:
    from lisa_sdk.lisa_surf.server import mcp

    tools = asyncio_get_tools(mcp)
    names = {t.name for t in tools}
    assert {"surf_url", "flush_session", "get_session_state"} <= names


# ── helpers ─────────────────────────────────────────────────────────────────


def asyncio_get_tools(mcp: Any) -> Any:
    """Run ``mcp.list_tools()`` synchronously via a fresh event loop."""
    import asyncio

    return asyncio.run(mcp.list_tools())

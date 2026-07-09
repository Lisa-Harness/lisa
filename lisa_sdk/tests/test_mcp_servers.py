"""Tests for the four Kernel-as-MCP servers (Part 4).

Covers tool registration and dispatch for lisa-context, lisa-memory,
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


# ── lisa_memory ────────────────────────────────────────────────────────────


@pytest.fixture
def memory_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    data_dir = tmp_path / "mem_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    from lisa_sdk.lisa_memory import server as mem_server

    mem_server.init_store(data_dir)
    return data_dir / "memory.db"


@pytest.mark.asyncio
async def test_memory_add(memory_env: Path) -> None:
    from lisa_sdk.lisa_memory.server import mcp

    result = await mcp.call_tool(
        "add_memory",
        {"user_slug": "alice", "content": "hello from test"},
    )
    assert "Memory stored with ID" in _tool_text(result)


@pytest.mark.asyncio
async def test_memory_search(memory_env: Path) -> None:
    from lisa_sdk.lisa_memory.server import mcp

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
async def test_memory_search_no_match(memory_env: Path) -> None:
    from lisa_sdk.lisa_memory.server import mcp

    result = await mcp.call_tool(
        "search_memory",
        {"user_slug": "nobody", "query": "zzzzz"},
    )
    assert "No relevant memories" in _tool_text(result)


@pytest.mark.asyncio
async def test_memory_list_memories_json(memory_env: Path) -> None:
    from lisa_sdk.lisa_memory.server import mcp

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
async def test_memory_get_surface_context(memory_env: Path) -> None:
    from lisa_sdk.lisa_memory.server import mcp

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
async def test_memory_lists_four_tools() -> None:
    from lisa_sdk.lisa_memory.server import mcp

    tools = await mcp.list_tools()
    names = {t.name for t in tools}
    assert {"add_memory", "search_memory", "list_memories", "get_surface_context"} <= names


# ── lisa_context (broker only) ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_context_lists_broker_tool() -> None:
    from lisa_sdk.lisa_context.server import mcp

    tools = await mcp.list_tools()
    names = {t.name for t in tools}
    assert "broker_context" in names
    assert "add_memory" not in names


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

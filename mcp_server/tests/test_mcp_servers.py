"""Tests for the Kernel-as-MCP servers.

Covers tool registration and dispatch for lisa-memory and lisa-guard
using MCPServer's in-process ``call_tool`` API.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from mcp.server.mcpserver.exceptions import ToolError


def _tool_text(result: Any) -> str:
    """Extract text from MCPServer call_tool result.

    MCP 2.0.0 returns a ``CallToolResult`` with a ``.content`` list of
    ``TextContent`` blocks. Older versions returned a
    ``(content_blocks, result_dict)`` tuple. Both shapes are handled.
    """
    # MCP 2.0.0: CallToolResult object with .content list
    if hasattr(result, "content") and result.content:
        blocks = result.content
        if hasattr(blocks[0], "text"):
            return blocks[0].text
        return str(blocks[0])
    # Legacy tuple: (content_blocks, result_dict)
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
    from mcp_server.memory import server as mem_server

    mem_server.init_store(data_dir)
    return data_dir / "memory.db"


def test_memory_init_generic_namespace_defaults_to_readonly_false(tmp_path: Path) -> None:
    """The in-process launcher's generic Namespace omits server-specific flags."""
    from argparse import Namespace

    from mcp_server.memory import server as mem_server

    mem_server._init(
        Namespace(data_dir=str(tmp_path), embedding_model=None, embedding_endpoint=None)
    )

    assert mem_server._cfg.readonly is False
    assert mem_server._deps is not None
    assert mem_server._deps.store.readonly is False


def test_memory_init_honors_readonly_flag(tmp_path: Path) -> None:
    """``--readonly`` reaches the store, and no database file is created."""
    from argparse import Namespace

    from mcp_server.memory import server as mem_server

    mem_server._init(
        Namespace(
            data_dir=str(tmp_path),
            embedding_model=None,
            embedding_endpoint=None,
            readonly=True,
        )
    )

    assert mem_server._cfg.readonly is True
    assert mem_server._deps is not None
    assert mem_server._deps.store.readonly is True
    assert not (tmp_path / "data" / "memory.db").exists()
    assert not (tmp_path / "data").exists()


@pytest.mark.asyncio
async def test_memory_readonly_server_rejects_writes(tmp_path: Path) -> None:
    """``--readonly`` must fail the mutating tools, not silently accept them."""
    from mcp_server.memory import server as mem_server

    writable = tmp_path / "writable"
    mem_server.init_store(writable)
    await mem_server.mcp.call_tool("add_memory", {"user_slug": "alice", "content": "kept"})
    mem_server._deps.store.close()  # type: ignore[union-attr]

    mem_server.init_store(writable, readonly=True)
    for tool, args in (
        ("add_memory", {"user_slug": "alice", "content": "rejected"}),
        ("delete_memory", {"memory_id": 1, "user_slug": "alice"}),
        ("retire_memory", {"memory_id": 1, "user_slug": "alice"}),
    ):
        with pytest.raises(ToolError, match="read-only"):
            await mem_server.mcp.call_tool(tool, args)


@pytest.mark.asyncio
async def test_memory_readonly_server_still_reads(tmp_path: Path) -> None:
    """Reads — including recall, which normally bumps access tracking — work."""
    from mcp_server.memory import server as mem_server

    writable = tmp_path / "writable"
    mem_server.init_store(writable)
    await mem_server.mcp.call_tool(
        "add_memory", {"user_slug": "alice", "content": "the quick brown fox"}
    )
    mem_server._deps.store.close()  # type: ignore[union-attr]

    mem_server.init_store(writable, readonly=True)
    result = await mem_server.mcp.call_tool(
        "search_memory", {"user_slug": "alice", "query": "quick fox"}
    )
    assert "quick brown fox" in _tool_text(result)
    listed = await mem_server.mcp.call_tool("list_memories", {"user_slug": "alice"})
    assert "quick brown fox" in _tool_text(listed)
    counted = await mem_server.mcp.call_tool("count_memories", {"user_slug": "alice"})
    assert _tool_text(counted).strip() == "1"


@pytest.mark.asyncio
async def test_memory_add(memory_env: Path) -> None:
    from mcp_server.memory.server import mcp

    result = await mcp.call_tool(
        "add_memory",
        {"user_slug": "alice", "content": "hello from test"},
    )
    assert "Memory stored with ID" in _tool_text(result)


@pytest.mark.asyncio
async def test_memory_search(memory_env: Path) -> None:
    from mcp_server.memory.server import mcp

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
    from mcp_server.memory.server import mcp

    result = await mcp.call_tool(
        "search_memory",
        {"user_slug": "nobody", "query": "zzzzz"},
    )
    assert _tool_text(result) == "[]"


@pytest.mark.asyncio
async def test_memory_list_memories_json(memory_env: Path) -> None:
    from mcp_server.memory.server import mcp

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
async def test_memory_get_memory_by_id(memory_env: Path) -> None:
    from mcp_server.memory.server import mcp

    add_result = await mcp.call_tool(
        "add_memory",
        {"user_slug": "alice", "content": "context line"},
    )
    add_text = _tool_text(add_result)
    # "Memory stored with ID 1" → parse the trailing integer.
    memory_id = int(add_text.rsplit("ID ", 1)[-1])
    result = await mcp.call_tool(
        "get_memory",
        {"memory_id": memory_id, "user_slug": "alice"},
    )
    parsed = json.loads(_tool_text(result))
    assert parsed["content"] == "context line"
    assert parsed["id"] == memory_id


@pytest.mark.asyncio
async def test_memory_get_memory_not_found(memory_env: Path) -> None:
    from mcp_server.memory.server import mcp

    with pytest.raises(ToolError, match="No memory found"):
        await mcp.call_tool(
            "get_memory",
            {"memory_id": 9999, "user_slug": "alice"},
        )


@pytest.mark.asyncio
async def test_memory_get_memory_cross_surface_is_not_found(memory_env: Path) -> None:
    from mcp_server.memory.server import mcp

    add_result = await mcp.call_tool(
        "add_memory",
        {"user_slug": "alice", "content": "alice's note", "surface_id": "surface-a"},
    )
    memory_id = int(_tool_text(add_result).rsplit("ID ", 1)[-1])
    # A different surface cannot read alice's record.
    with pytest.raises(ToolError, match="No memory found"):
        await mcp.call_tool(
            "get_memory",
            {"memory_id": memory_id, "user_slug": "alice", "surface_id": "surface-b"},
        )


@pytest.mark.asyncio
async def test_memory_delete_memory_by_id(memory_env: Path) -> None:
    from mcp_server.memory.server import mcp

    add_result = await mcp.call_tool(
        "add_memory",
        {"user_slug": "alice", "content": "doomed note"},
    )
    memory_id = int(_tool_text(add_result).rsplit("ID ", 1)[-1])
    delete_result = await mcp.call_tool(
        "delete_memory",
        {"memory_id": memory_id, "user_slug": "alice"},
    )
    assert f"Deleted memory {memory_id}" in _tool_text(delete_result)
    # Subsequent get returns not-found.
    with pytest.raises(ToolError, match="No memory found"):
        await mcp.call_tool(
            "get_memory",
            {"memory_id": memory_id, "user_slug": "alice"},
        )


@pytest.mark.asyncio
async def test_memory_delete_memory_cross_surface_is_not_found(memory_env: Path) -> None:
    from mcp_server.memory.server import mcp

    add_result = await mcp.call_tool(
        "add_memory",
        {"user_slug": "alice", "content": "alice's note", "surface_id": "surface-a"},
    )
    memory_id = int(_tool_text(add_result).rsplit("ID ", 1)[-1])
    # A different surface cannot delete alice's record.
    with pytest.raises(ToolError, match="No memory found"):
        await mcp.call_tool(
            "delete_memory",
            {"memory_id": memory_id, "user_slug": "alice", "surface_id": "surface-b"},
        )


@pytest.mark.asyncio
async def test_memory_retire_memory_by_id(memory_env: Path) -> None:
    from mcp_server.memory.server import mcp

    add_result = await mcp.call_tool(
        "add_memory",
        {"user_slug": "alice", "content": "retire target"},
    )
    memory_id = int(_tool_text(add_result).rsplit("ID ", 1)[-1])
    retire_result = await mcp.call_tool(
        "retire_memory",
        {"memory_id": memory_id, "user_slug": "alice"},
    )
    assert f"Retired memory {memory_id}" in _tool_text(retire_result)

    # list_memories by default excludes retired
    list_result = await mcp.call_tool("list_memories", {"user_slug": "alice"})
    parsed = json.loads(_tool_text(list_result))
    assert all(r["id"] != memory_id for r in parsed)


@pytest.mark.asyncio
async def test_memory_count_memories(memory_env: Path) -> None:
    from mcp_server.memory.server import mcp

    await mcp.call_tool("add_memory", {"user_slug": "alice", "content": "note 1"})
    await mcp.call_tool("add_memory", {"user_slug": "alice", "content": "note 2"})
    count_result = await mcp.call_tool("count_memories", {"user_slug": "alice"})
    assert _tool_text(count_result) == "2"


@pytest.mark.asyncio
async def test_memory_lists_tools() -> None:
    from mcp_server.memory.server import mcp

    tools = await mcp.list_tools()
    names = {t.name for t in tools}
    assert {
        "add_memory",
        "search_memory",
        "list_memories",
        "get_memory",
        "delete_memory",
        "retire_memory",
        "count_memories",
    } <= names


# ── helpers ─────────────────────────────────────────────────────────────────


def asyncio_get_tools(mcp: Any) -> Any:
    """Run ``mcp.list_tools()`` synchronously via a fresh event loop."""
    import asyncio

    return asyncio.run(mcp.list_tools())

"""Tests for the four Kernel-as-MCP servers (Part 4).

Covers tool registration and dispatch for lisa-context, lisa-memory,
and lisa-guard using MCPServer's in-process ``call_tool`` API.
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


# ── lisa_context (broker only) ─────────────────────────────────────────────


@pytest.fixture
def context_setup() -> None:
    from sdk.context.context_broker import reset_extension_item_builder

    reset_extension_item_builder()


@pytest.mark.asyncio
async def test_context_lists_broker_tool(context_setup: None) -> None:
    from mcp_server.context.server import mcp

    tools = await mcp.list_tools()
    names = {t.name for t in tools}
    assert "broker_context" in names
    assert "add_memory" not in names


@pytest.mark.asyncio
async def test_context_broker_context_returns_packet(context_setup: None) -> None:
    """broker_context with a valid typed payload returns brokered markdown."""
    from mcp_server.context.server import mcp

    payload = {
        "query": "what is the current status",
        "mode_id": "default",
        "state": {"compiled_at": "2026-07-11T00:00:00+00:00", "context_markdown": ""},
    }
    result = await mcp.call_tool("broker_context", payload)
    data = json.loads(_tool_text(result))
    assert "packet_markdown" in data
    assert "## Ambient Facts" in data["packet_markdown"]


@pytest.mark.asyncio
async def test_context_broker_context_empty_state_ok(context_setup: None) -> None:
    """broker_context with no state field uses an empty UnifiedState."""
    from mcp_server.context.server import mcp

    payload = {"query": "hello"}
    result = await mcp.call_tool("broker_context", payload)
    data = json.loads(_tool_text(result))
    assert "packet_markdown" in data
    assert "## Ambient Facts" in data["packet_markdown"]


@pytest.mark.asyncio
async def test_context_broker_context_malformed_state_raises(context_setup: None) -> None:
    """broker_context with malformed state should fail, not silently degrade."""
    from mcp_server.context.server import mcp

    payload = {
        "query": "hello",
        "state": {"compiled_at": 12345, "context_markdown": "", "bogus_field": object()},
    }
    # MCPServer wraps tool errors; the call should raise rather than return
    # an empty-state response.
    with pytest.raises(Exception, match="state"):
        await mcp.call_tool("broker_context", payload)


# ── helpers ─────────────────────────────────────────────────────────────────


def asyncio_get_tools(mcp: Any) -> Any:
    """Run ``mcp.list_tools()`` synchronously via a fresh event loop."""
    import asyncio

    return asyncio.run(mcp.list_tools())

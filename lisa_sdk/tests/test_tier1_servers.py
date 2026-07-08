"""Tests for the three Tier 1 MCP servers (guard, storage, content).

Covers tool registration, dispatch, and behavioral contracts using FastMCP's
in-process ``call_tool`` API.
"""

from __future__ import annotations

import base64
import json
import tempfile
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
    from lisa_sdk.lisa_guard.server import init_backend

    init_backend("passthrough")


@pytest.mark.asyncio
async def test_guard_lists_three_tools(guard_setup: None) -> None:
    from lisa_sdk.lisa_guard.server import mcp

    tools = await mcp.list_tools()
    names = {t.name for t in tools}
    assert {"validate_input", "validate_output", "guard_configured"} == names


@pytest.mark.asyncio
async def test_guard_validate_input_passthrough(guard_setup: None) -> None:
    from lisa_sdk.lisa_guard.server import mcp

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
    from lisa_sdk.lisa_guard.server import mcp

    result = await mcp.call_tool(
        "validate_output",
        {"text": "safe output", "surface_id": "s1"},
    )
    data = json.loads(_tool_text(result))
    assert data["allowed"] is True


@pytest.mark.asyncio
async def test_guard_configured(guard_setup: None) -> None:
    from lisa_sdk.lisa_guard.server import mcp

    result = await mcp.call_tool("guard_configured", {})
    data = json.loads(_tool_text(result))
    assert data["configured"] is True
    assert data["backend"] == "passthrough"


# ── lisa_storage ────────────────────────────────────────────────────────────


@pytest.fixture
def storage_env(tmp_path: Path) -> Path:
    from lisa_sdk.lisa_storage.server import init_storage

    init_storage(tmp_path, readonly=False)
    return tmp_path


@pytest.mark.asyncio
async def test_storage_lists_all_blobio_tools(storage_env: Path) -> None:
    from lisa_sdk.lisa_storage.server import mcp

    tools = await mcp.list_tools()
    names = {t.name for t in tools}
    # All 23 BlobIO protocol methods
    expected = {
        "read_unified_state", "write_unified_state",
        "read_blob_bytes", "write_blob_bytes",
        "read_blob_text", "write_blob_text",
        "read_surf_workflow_session", "write_surf_workflow_session",
        "delete_surf_workflow_session",
        "read_surf_screenshot", "write_surf_trace",
        "read_user_context_overlays", "write_user_context_overlays",
        "read_surface_context_overlays", "write_surface_context_overlays",
        "read_proactive_recipients", "write_proactive_recipients",
        "read_user_reminders", "write_user_reminders",
        "read_thread_state", "write_thread_state",
        "read_memory_snapshot", "write_memory_snapshot",
    }
    assert expected <= names, f"Missing tools: {expected - names}"


@pytest.mark.asyncio
async def test_storage_blob_text_round_trip(storage_env: Path) -> None:
    from lisa_sdk.lisa_storage.server import mcp

    await mcp.call_tool(
        "write_blob_text",
        {"container": "test", "blob_path": "a.txt", "payload": "hello world"},
    )
    result = await mcp.call_tool(
        "read_blob_text", {"container": "test", "blob_path": "a.txt"}
    )
    assert _tool_text(result) == "hello world"


@pytest.mark.asyncio
async def test_storage_blob_bytes_round_trip(storage_env: Path) -> None:
    from lisa_sdk.lisa_storage.server import mcp

    payload = base64.b64encode(b"binary data").decode()
    await mcp.call_tool(
        "write_blob_bytes",
        {"container": "test", "blob_path": "b.bin", "payload_b64": payload},
    )
    result = await mcp.call_tool(
        "read_blob_bytes", {"container": "test", "blob_path": "b.bin"}
    )
    assert base64.b64decode(_tool_text(result)) == b"binary data"


@pytest.mark.asyncio
async def test_storage_read_absent_returns_null(storage_env: Path) -> None:
    from lisa_sdk.lisa_storage.server import mcp

    result = await mcp.call_tool(
        "read_blob_text", {"container": "test", "blob_path": "absent.txt"}
    )
    assert _tool_text(result) == "null"


@pytest.mark.asyncio
async def test_storage_path_traversal_blocked(storage_env: Path) -> None:
    from lisa_sdk.lisa_storage.server import mcp

    with pytest.raises(Exception):
        await mcp.call_tool(
            "read_blob_text", {"container": "test", "blob_path": "../escape.txt"}
        )


@pytest.mark.asyncio
async def test_storage_readonly_blocks_writes(tmp_path: Path) -> None:
    from lisa_sdk.lisa_storage.server import init_storage, mcp

    init_storage(tmp_path, readonly=True)
    result = await mcp.call_tool(
        "write_blob_text",
        {"container": "test", "blob_path": "c.txt", "payload": "no"},
    )
    data = json.loads(_tool_text(result))
    assert data["ok"] is False
    assert data["error"] == "readonly"


@pytest.mark.asyncio
async def test_storage_overlays_round_trip(storage_env: Path) -> None:
    from lisa_sdk.lisa_storage.server import mcp

    overlays = {"key": "value", "nested": {"a": 1}}
    await mcp.call_tool(
        "write_user_context_overlays",
        {"overlays_json": json.dumps(overlays)},
    )
    result = await mcp.call_tool("read_user_context_overlays", {})
    data = json.loads(_tool_text(result))
    assert data == overlays


# ── lisa_content ───────────────────────────────────────────────────────────


@pytest.fixture
def content_setup() -> None:
    from lisa_sdk.lisa_content.server import init_content

    init_content()


@pytest.mark.asyncio
async def test_content_lists_three_tools(content_setup: None) -> None:
    from lisa_sdk.lisa_content.server import mcp

    tools = await mcp.list_tools()
    names = {t.name for t in tools}
    assert {"read_authored_text", "list_content_roots", "invalidate_cache"} == names


@pytest.mark.asyncio
async def test_content_read_absent_returns_empty(content_setup: None) -> None:
    from lisa_sdk.lisa_content.server import mcp

    result = await mcp.call_tool(
        "read_authored_text", {"relative_path": "nonexistent/file.md"}
    )
    assert _tool_text(result) == ""


@pytest.mark.asyncio
async def test_content_invalidate_cache(content_setup: None) -> None:
    from lisa_sdk.lisa_content.server import mcp

    result = await mcp.call_tool("invalidate_cache", {})
    data = json.loads(_tool_text(result))
    assert data["ok"] is True


@pytest.mark.asyncio
async def test_content_list_roots_returns_array(content_setup: None) -> None:
    from lisa_sdk.lisa_content.server import mcp

    result = await mcp.call_tool("list_content_roots", {})
    data = json.loads(_tool_text(result))
    assert isinstance(data, list)
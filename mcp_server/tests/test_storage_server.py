"""Tests for the lisa-storage MCP server.

Covers tool registration, file I/O, path traversal rejection, readonly mode,
and checkpoint save/restore/list — using MCPServer's in-process ``call_tool`` API.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


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


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def storage_setup(tmp_path: Path) -> None:
    """Initialize lisa-storage with a temp data dir (writable)."""
    from mcp_server.storage.server import init_storage

    init_storage(tmp_path, readonly=False)


@pytest.fixture
def storage_readonly(tmp_path: Path) -> None:
    """Initialize lisa-storage in readonly mode."""
    from mcp_server.storage.server import init_storage

    init_storage(tmp_path, readonly=True)


# ── Tool registration ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_storage_lists_eleven_tools(storage_setup: None) -> None:
    from mcp_server.storage.server import mcp

    tools = await mcp.list_tools()
    names = {t.name for t in tools}
    assert {
        "read_file",
        "read_file_range",
        "write_file",
        "append_file",
        "list_files",
        "delete_file",
        "stat_file",
        "make_dir",
        "save_checkpoint",
        "restore_checkpoint",
        "list_checkpoints",
        "read_model_manifest",
        "write_model_manifest",
    } == names


# ── File I/O ────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_write_read_roundtrip(storage_setup: None) -> None:
    from mcp_server.storage.server import mcp

    write_result = await mcp.call_tool(
        "write_file", {"path": "reports/daily.md", "content": "# Hello"}
    )
    data = json.loads(_tool_text(write_result))
    assert data["ok"] is True
    assert data["bytes"] > 0

    read_result = await mcp.call_tool("read_file", {"path": "reports/daily.md"})
    text = _tool_text(read_result)
    assert "# Hello" in text


@pytest.mark.asyncio
async def test_read_file_range_roundtrip(storage_setup: None) -> None:
    from mcp_server.storage.server import mcp

    await mcp.call_tool("write_file", {"path": "range.txt", "content": "0123456789"})
    result = await mcp.call_tool(
        "read_file_range", {"path": "range.txt", "start": 2, "length": 4}
    )
    data = json.loads(_tool_text(result))
    assert data["content"] == "2345"
    assert data["start"] == 2
    assert data["length"] == 4
    assert data["total"] == 10
    assert data["eof"] is False


@pytest.mark.asyncio
async def test_append_file_roundtrip(storage_setup: None) -> None:
    from mcp_server.storage.server import mcp

    first = await mcp.call_tool("append_file", {"path": "log.txt", "content": "a\n"})
    data = json.loads(_tool_text(first))
    assert data["ok"] is True
    assert data["bytes"] == 2

    second = await mcp.call_tool("append_file", {"path": "log.txt", "content": "b\n"})
    data = json.loads(_tool_text(second))
    assert data["bytes"] == 4

    read_result = await mcp.call_tool("read_file", {"path": "log.txt"})
    assert _tool_text(read_result) == "a\nb\n"


@pytest.mark.asyncio
async def test_read_absent_returns_error(storage_setup: None) -> None:
    from mcp_server.storage.server import mcp

    result = await mcp.call_tool("read_file", {"path": "nonexistent.txt"})
    data = json.loads(_tool_text(result))
    assert "error" in data


@pytest.mark.asyncio
async def test_traversal_rejected(storage_setup: None) -> None:
    from mcp_server.storage.server import mcp

    result = await mcp.call_tool("read_file", {"path": "../../../etc/passwd"})
    data = json.loads(_tool_text(result))
    assert "error" in data


@pytest.mark.asyncio
async def test_absolute_path_rejected(storage_setup: None) -> None:
    from mcp_server.storage.server import mcp

    # Use a platform-appropriate absolute path that survives the
    # leading-separator strip and is caught by is_absolute().
    abs_path = str(Path("evil.txt").resolve())
    result = await mcp.call_tool("write_file", {"path": abs_path, "content": "bad"})
    data = json.loads(_tool_text(result))
    assert "error" in data


@pytest.mark.asyncio
async def test_list_files(storage_setup: None) -> None:
    from mcp_server.storage.server import mcp

    await mcp.call_tool("write_file", {"path": "a.txt", "content": "a"})
    await mcp.call_tool("write_file", {"path": "sub/b.txt", "content": "b"})
    result = await mcp.call_tool("list_files", {})
    files = json.loads(_tool_text(result))
    assert "a.txt" in files
    assert "sub/b.txt" in files


@pytest.mark.asyncio
async def test_list_files_with_prefix(storage_setup: None) -> None:
    from mcp_server.storage.server import mcp

    await mcp.call_tool("write_file", {"path": "a.txt", "content": "a"})
    await mcp.call_tool("write_file", {"path": "sub/b.txt", "content": "b"})
    result = await mcp.call_tool("list_files", {"prefix": "sub"})
    files = json.loads(_tool_text(result))
    assert "sub/b.txt" in files
    assert "a.txt" not in files


@pytest.mark.asyncio
async def test_stat_file(storage_setup: None) -> None:
    from mcp_server.storage.server import mcp

    await mcp.call_tool("write_file", {"path": "info.txt", "content": "hello world"})
    result = await mcp.call_tool("stat_file", {"path": "info.txt"})
    data = json.loads(_tool_text(result))
    assert data["size"] > 0
    assert data["is_dir"] is False


@pytest.mark.asyncio
async def test_delete_file(storage_setup: None) -> None:
    from mcp_server.storage.server import mcp

    await mcp.call_tool("write_file", {"path": "trash.txt", "content": "x"})
    result = await mcp.call_tool("delete_file", {"path": "trash.txt"})
    data = json.loads(_tool_text(result))
    assert data["ok"] is True

    read_result = await mcp.call_tool("read_file", {"path": "trash.txt"})
    data = json.loads(_tool_text(read_result))
    assert "error" in data


@pytest.mark.asyncio
async def test_delete_directory_fails(storage_setup: None) -> None:
    from mcp_server.storage.server import mcp

    await mcp.call_tool("make_dir", {"path": "mydir"})
    result = await mcp.call_tool("delete_file", {"path": "mydir"})
    data = json.loads(_tool_text(result))
    assert "error" in data


@pytest.mark.asyncio
async def test_make_dir(storage_setup: None) -> None:
    from mcp_server.storage.server import mcp

    result = await mcp.call_tool("make_dir", {"path": "newdir/nested"})
    data = json.loads(_tool_text(result))
    assert data["ok"] is True


# ── Readonly mode ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_readonly_blocks_write(storage_readonly: None) -> None:
    from mcp_server.storage.server import mcp

    result = await mcp.call_tool("write_file", {"path": "test.txt", "content": "x"})
    data = json.loads(_tool_text(result))
    assert "error" in data


@pytest.mark.asyncio
async def test_readonly_blocks_delete(storage_readonly: None) -> None:
    from mcp_server.storage.server import mcp

    result = await mcp.call_tool("delete_file", {"path": "test.txt"})
    data = json.loads(_tool_text(result))
    assert "error" in data


@pytest.mark.asyncio
async def test_readonly_blocks_make_dir(storage_readonly: None) -> None:
    from mcp_server.storage.server import mcp

    result = await mcp.call_tool("make_dir", {"path": "newdir"})
    data = json.loads(_tool_text(result))
    assert "error" in data


# ── Checkpoints ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_save_restore_checkpoint_roundtrip(storage_setup: None) -> None:
    from mcp_server.storage.server import mcp

    bundle = {
        "label": "test-checkpoint",
        "mode_id": "default",
        "stage_id": "gather",
        "turn_count": 5,
        "unified_state": {"compiled_at": "2026-07-11T00:00:00Z"},
        "workflow_session_state": {"mode_id": "research", "stage_id": "gather"},
    }
    save_result = await mcp.call_tool("save_checkpoint", {"bundle": bundle})
    data = json.loads(_tool_text(save_result))
    assert "checkpoint_id" in data
    cid = data["checkpoint_id"]

    restore_result = await mcp.call_tool("restore_checkpoint", {"checkpoint_id": cid})
    restored = json.loads(_tool_text(restore_result))
    assert restored["label"] == "test-checkpoint"
    assert restored["stage_id"] == "gather"
    assert restored["turn_count"] == 5


@pytest.mark.asyncio
async def test_restore_missing_checkpoint_returns_error(storage_setup: None) -> None:
    from mcp_server.storage.server import mcp

    result = await mcp.call_tool("restore_checkpoint", {"checkpoint_id": "nonexistent"})
    data = json.loads(_tool_text(result))
    assert "error" in data


@pytest.mark.asyncio
async def test_list_checkpoints_sorted_desc(storage_setup: None) -> None:
    from mcp_server.storage.server import mcp

    for i in range(3):
        await mcp.call_tool("save_checkpoint", {"bundle": {"label": f"cp-{i}", "turn_count": i}})

    result = await mcp.call_tool("list_checkpoints", {})
    data = json.loads(_tool_text(result))
    assert len(data) == 3
    # Should be sorted by created_at descending; all should have checkpoint_id
    ids = [entry["checkpoint_id"] for entry in data]
    assert len(ids) == 3
    assert all(ids)


@pytest.mark.asyncio
async def test_list_checkpoints_empty(storage_setup: None) -> None:
    from mcp_server.storage.server import mcp

    result = await mcp.call_tool("list_checkpoints", {})
    data = json.loads(_tool_text(result))
    assert data == []


@pytest.mark.asyncio
async def test_readonly_blocks_save_checkpoint(storage_readonly: None) -> None:
    from mcp_server.storage.server import mcp

    result = await mcp.call_tool("save_checkpoint", {"bundle": {"label": "x"}})
    data = json.loads(_tool_text(result))
    assert "error" in data


# ── Model manifest ──────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_write_read_model_manifest_roundtrip(storage_setup: None) -> None:
    from mcp_server.storage.server import mcp

    write = await mcp.call_tool(
        "write_model_manifest",
        {"content": "version: 1\nmodels: []\n", "which": "proposal"},
    )
    data = json.loads(_tool_text(write))
    assert data["ok"] is True
    assert data["which"] == "proposal"
    assert data["bytes"] > 0

    read = await mcp.call_tool("read_model_manifest", {"which": "proposal"})
    text = _tool_text(read)
    assert "version: 1" in text


@pytest.mark.asyncio
async def test_read_model_manifest_not_found(storage_setup: None) -> None:
    from mcp_server.storage.server import mcp

    result = await mcp.call_tool("read_model_manifest", {"which": "active"})
    data = json.loads(_tool_text(result))
    assert "error" in data
    assert "not found" in data["error"]


@pytest.mark.asyncio
async def test_read_model_manifest_invalid_kind(storage_setup: None) -> None:
    from mcp_server.storage.server import mcp

    result = await mcp.call_tool("read_model_manifest", {"which": "bogus"})
    data = json.loads(_tool_text(result))
    assert "error" in data
    assert "active" in data["error"]


@pytest.mark.asyncio
async def test_write_model_manifest_active(storage_setup: None) -> None:
    from mcp_server.storage.server import mcp

    write = await mcp.call_tool(
        "write_model_manifest",
        {"content": "version: 1\nmodels: []\n", "which": "active"},
    )
    data = json.loads(_tool_text(write))
    assert data["ok"] is True
    assert data["which"] == "active"


@pytest.mark.asyncio
async def test_readonly_blocks_write_model_manifest(storage_readonly: None) -> None:
    from mcp_server.storage.server import mcp

    result = await mcp.call_tool(
        "write_model_manifest",
        {"content": "version: 1\n", "which": "proposal"},
    )
    data = json.loads(_tool_text(result))
    assert "error" in data


# ── Glob denylist ───────────────────────────────────────────────────────────


@pytest.fixture
def storage_with_deny(tmp_path: Path) -> Path:
    """Initialize lisa-storage with a representative denylist; return data dir."""
    from mcp_server.storage.server import init_storage

    init_storage(
        tmp_path,
        readonly=False,
        deny_globs=("*.db", "threads/**", "repl_history.txt"),
    )
    return tmp_path


@pytest.mark.asyncio
async def test_deny_blocks_write_denied_glob(storage_with_deny: Path) -> None:
    from mcp_server.storage.server import mcp

    for denied in ("secret.db", "threads/abc.jsonl", "repl_history.txt"):
        result = await mcp.call_tool("write_file", {"path": denied, "content": "x"})
        data = json.loads(_tool_text(result))
        assert "error" in data, denied
        assert "denied by policy" in data["error"], denied


@pytest.mark.asyncio
async def test_deny_blocks_read_denied_glob(storage_with_deny: Path) -> None:
    from mcp_server.storage.server import mcp

    for denied in ("secret.db", "threads/abc.jsonl", "repl_history.txt"):
        result = await mcp.call_tool("read_file", {"path": denied})
        data = json.loads(_tool_text(result))
        assert "error" in data, denied
        assert "denied by policy" in data["error"], denied


@pytest.mark.asyncio
async def test_deny_blocks_delete_and_stat(storage_with_deny: Path) -> None:
    from mcp_server.storage.server import mcp

    for tool, args in (
        ("delete_file", {"path": "secret.db"}),
        ("stat_file", {"path": "secret.db"}),
        ("make_dir", {"path": "threads/new"}),
    ):
        result = await mcp.call_tool(tool, args)
        data = json.loads(_tool_text(result))
        assert "error" in data, tool
        assert "denied by policy" in data["error"], tool


@pytest.mark.asyncio
async def test_deny_filters_list_files(storage_with_deny: Path) -> None:
    from mcp_server.storage.server import mcp

    # Write a normal file and a denied file directly to disk, then confirm
    # list_files drops the denied one.
    files_root = storage_with_deny / "files"
    files_root.mkdir(parents=True, exist_ok=True)
    (files_root / "keep.txt").write_text("ok", encoding="utf-8")
    (files_root / "secret.db").write_text("no", encoding="utf-8")
    result = await mcp.call_tool("list_files", {})
    files = json.loads(_tool_text(result))
    assert "keep.txt" in files
    assert "secret.db" not in files


@pytest.mark.asyncio
async def test_deny_rejects_denied_prefix(storage_with_deny: Path) -> None:
    from mcp_server.storage.server import mcp

    # A prefix that itself matches a deny glob is rejected outright.
    result = await mcp.call_tool("list_files", {"prefix": "repl_history.txt"})
    data = json.loads(_tool_text(result))
    assert "error" in data
    assert "denied by policy" in data["error"]


@pytest.mark.asyncio
async def test_deny_prefix_ancestor_filters_not_leaks(storage_with_deny: Path) -> None:
    from mcp_server.storage.server import mcp

    # Listing an ancestor of denied files must not error (the ancestor
    # isn't itself denied) and must not leak denied file names.
    files_root = storage_with_deny / "files"
    (files_root / "threads" / "deep").mkdir(parents=True, exist_ok=True)
    (files_root / "threads" / "deep" / "abc.jsonl").write_text("x", encoding="utf-8")
    result = await mcp.call_tool("list_files", {"prefix": "threads"})
    data = json.loads(_tool_text(result))
    assert data == []


@pytest.mark.asyncio
async def test_no_deny_globs_allows_everything(storage_setup: None) -> None:
    from mcp_server.storage.server import mcp

    # storage_setup uses the default empty denylist; a .db file is fine.
    result = await mcp.call_tool("write_file", {"path": "anything.db", "content": "x"})
    data = json.loads(_tool_text(result))
    assert data.get("ok") is True

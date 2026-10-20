"""Tests for the lisa-storage MCP server.

Covers tool registration, file I/O, path traversal rejection, readonly mode,
and model manifest read/write — using MCPServer's in-process ``call_tool`` API.
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
    # The receipt states that the stored text is newline-terminated.
    assert data["trailing_newline_added"] is True

    read_result = await mcp.call_tool("read_file", {"path": "reports/daily.md"})
    text = _tool_text(read_result)
    assert "# Hello" in text


@pytest.mark.asyncio
async def test_write_file_newline_terminates_text(storage_setup: None) -> None:
    """The tool's end-to-end write contract: text lands newline-terminated.

    A file whose final newline is missing reads back as if its last line
    were cut off — ``cat`` glued the shell prompt onto the last line of a
    session plan on 2026-10-19. Storage owns file semantics, so it
    enforces the rule and reports that it did.
    """
    from mcp_server.storage.server import mcp

    result = await mcp.call_tool(
        "write_file", {"path": "plan.md", "content": "ends the turn."}
    )
    data = json.loads(_tool_text(result))
    assert data["ok"] is True
    assert data["trailing_newline_added"] is True
    assert data["bytes"] == len(b"ends the turn.\n")

    read_result = await mcp.call_tool("read_file", {"path": "plan.md"})
    assert _tool_text(read_result) == "ends the turn.\n"


@pytest.mark.asyncio
async def test_write_file_leaves_binary_content_exact(storage_setup: None) -> None:
    """A binary format is not text; its trailing bytes stay untouched."""
    from mcp_server.storage.server import mcp

    result = await mcp.call_tool(
        "write_file", {"path": "image.png", "content": "PNG-not-really"}
    )
    data = json.loads(_tool_text(result))
    assert data["ok"] is True
    assert data["trailing_newline_added"] is False
    assert data["bytes"] == len(b"PNG-not-really")


@pytest.mark.asyncio
async def test_append_file_reports_appended_bytes(storage_setup: None) -> None:
    """The append receipt proves an effect landed.

    An empty append changes nothing; ``appended_bytes`` says so, which
    the resulting file size alone cannot.
    """
    from mcp_server.storage.server import mcp

    first = await mcp.call_tool("append_file", {"path": "log.txt", "content": "K1\n"})
    data = json.loads(_tool_text(first))
    assert data["appended_bytes"] == 3
    assert data["bytes"] == 3

    empty = await mcp.call_tool("append_file", {"path": "log.txt", "content": ""})
    data = json.loads(_tool_text(empty))
    assert data["ok"] is True
    assert data["appended_bytes"] == 0
    assert data["bytes"] == 3


@pytest.mark.asyncio
async def test_read_file_range_roundtrip(storage_setup: None) -> None:
    from mcp_server.storage.server import mcp

    await mcp.call_tool("write_file", {"path": "range.txt", "content": "0123456789"})
    result = await mcp.call_tool("read_file_range", {"path": "range.txt", "start": 2, "length": 4})
    data = json.loads(_tool_text(result))
    assert data["content"] == "2345"
    assert data["start"] == 2
    assert data["length"] == 4
    assert data["total"] == 11
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

    with pytest.raises(ToolError):
        await mcp.call_tool("read_file", {"path": "nonexistent.txt"})


@pytest.mark.asyncio
async def test_traversal_rejected(storage_setup: None) -> None:
    from mcp_server.storage.server import mcp

    with pytest.raises(ToolError):
        await mcp.call_tool("read_file", {"path": "../../../etc/passwd"})


@pytest.mark.asyncio
async def test_absolute_path_rejected(storage_setup: None) -> None:
    from mcp_server.storage.server import mcp

    # Use a platform-appropriate absolute path that survives the
    # leading-separator strip and is caught by is_absolute().
    abs_path = str(Path("evil.txt").resolve())
    with pytest.raises(ToolError):
        await mcp.call_tool("write_file", {"path": abs_path, "content": "bad"})


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

    with pytest.raises(ToolError):
        await mcp.call_tool("read_file", {"path": "trash.txt"})


@pytest.mark.asyncio
async def test_delete_directory_fails(storage_setup: None) -> None:
    from mcp_server.storage.server import mcp

    await mcp.call_tool("make_dir", {"path": "mydir"})
    with pytest.raises(ToolError):
        await mcp.call_tool("delete_file", {"path": "mydir"})


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

    with pytest.raises(ToolError, match="read-only"):
        await mcp.call_tool("write_file", {"path": "test.txt", "content": "x"})


@pytest.mark.asyncio
async def test_readonly_blocks_delete(storage_readonly: None) -> None:
    from mcp_server.storage.server import mcp

    with pytest.raises(ToolError):
        await mcp.call_tool("delete_file", {"path": "test.txt"})


@pytest.mark.asyncio
async def test_readonly_blocks_make_dir(storage_readonly: None) -> None:
    from mcp_server.storage.server import mcp

    with pytest.raises(ToolError, match="read-only"):
        await mcp.call_tool("make_dir", {"path": "newdir"})


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

    with pytest.raises(ToolError, match="not found"):
        await mcp.call_tool("read_model_manifest", {"which": "active"})


@pytest.mark.asyncio
async def test_read_model_manifest_invalid_kind(storage_setup: None) -> None:
    from mcp_server.storage.server import mcp

    with pytest.raises(ToolError, match="active"):
        await mcp.call_tool("read_model_manifest", {"which": "bogus"})


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

    with pytest.raises(ToolError, match="read-only"):
        await mcp.call_tool(
            "write_model_manifest",
            {"content": "version: 1\n", "which": "proposal"},
        )


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
        with pytest.raises(ToolError, match="denied by policy"):
            await mcp.call_tool("write_file", {"path": denied, "content": "x"})


@pytest.mark.asyncio
async def test_deny_blocks_read_denied_glob(storage_with_deny: Path) -> None:
    from mcp_server.storage.server import mcp

    for denied in ("secret.db", "threads/abc.jsonl", "repl_history.txt"):
        with pytest.raises(ToolError, match="denied by policy"):
            await mcp.call_tool("read_file", {"path": denied})


@pytest.mark.asyncio
async def test_deny_blocks_delete_and_stat(storage_with_deny: Path) -> None:
    from mcp_server.storage.server import mcp

    for tool, args in (
        ("delete_file", {"path": "secret.db"}),
        ("stat_file", {"path": "secret.db"}),
        ("make_dir", {"path": "threads/new"}),
    ):
        with pytest.raises(ToolError, match="denied by policy"):
            await mcp.call_tool(tool, args)


@pytest.mark.asyncio
async def test_deny_filters_list_files(storage_with_deny: Path) -> None:
    from mcp_server.storage.server import mcp

    # Write a normal file and a denied file directly to disk, then confirm
    # list_files drops the denied one.
    files_root = storage_with_deny / "data" / "files"
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
    with pytest.raises(ToolError, match="denied by policy"):
        await mcp.call_tool("list_files", {"prefix": "repl_history.txt"})


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

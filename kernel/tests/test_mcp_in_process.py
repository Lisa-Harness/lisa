"""Integration test for in-process MCP server connection via memory transport.

Verifies that ``MCPToolPool._connect_in_process`` can wire a
:class:`ClientSession` to a first-party server's :class:`MCPServer`
instance via the MCP SDK's in-memory transport, initialize the session,
and call a tool — all without spawning a subprocess.

This is an integration test (marked ``integration``) because it imports
real server modules and exercises the full MCP protocol stack in-memory.
"""

from __future__ import annotations

import pytest
from kernel.tools.mcp_client import (
    InProcessServerEntry,
    MCPClientConfig,
    MCPToolPool,
)
from mcp_server.registry import get_in_process_server

pytestmark = pytest.mark.integration


def _in_process_config(server_id: str, *, warm: bool = True) -> MCPClientConfig:
    """Build an in-process MCPClientConfig for testing."""
    return MCPClientConfig(
        server_id=server_id,
        command="",
        args=["--data-dir", "/tmp/lisa-test-in-process"],
        env={
            "LISA_TIMEZONE": "UTC",
            "LISA_THREAD_DIR": "/tmp/lisa-test-in-process/threads",
            "LISA_USER_SLUG": "test",
        },
        warm=warm,
        required=False,
        runtime_kind="in_process",
        data_dir_path="/tmp/lisa-test-in-process",
        ensure_data_dir=True,
    )


def _make_test_resolver():
    """Build a resolver that maps server_ids to kernel InProcessServerEntry."""

    def resolver(server_id: str) -> InProcessServerEntry | None:
        entry = get_in_process_server(server_id)
        if entry is None:
            return None
        return InProcessServerEntry(
            server_id=entry.server_id,
            mcp=entry.mcp,
            init=entry.init,
        )

    return resolver


@pytest.mark.asyncio
async def test_in_process_connect_and_list_tools() -> None:
    """An in-process warm server connects, initializes, and lists tools."""
    config = _in_process_config("lisa-content")
    resolver = _make_test_resolver()
    pool = MCPToolPool([config], in_process_resolver=resolver)
    try:
        await pool.start()
        tools = await pool.list_tools()
        tool_names = {t.name for t in tools}
        # lisa-content exposes read_authored_text, list_content_roots, invalidate_cache
        # (namespaced as lisa-content.<tool_name>).
        assert "lisa-content.read_authored_text" in tool_names
        assert "lisa-content.list_content_roots" in tool_names
    finally:
        await pool.aclose()


@pytest.mark.asyncio
async def test_in_process_connect_and_call_tool() -> None:
    """An in-process server accepts tool calls via the memory transport."""
    config = _in_process_config("lisa-content")
    resolver = _make_test_resolver()
    pool = MCPToolPool([config], in_process_resolver=resolver)
    try:
        await pool.start()
        # list_content_roots is read-only and always available.
        # Tools are namespaced as <server_id>.<tool_name>.
        result = await pool.call_tool("lisa-content.list_content_roots", {})
        assert not result.is_error
        # The result should be a JSON array string.
        assert result.output_text  # non-empty
    finally:
        await pool.aclose()


@pytest.mark.asyncio
async def test_in_process_without_resolver_raises() -> None:
    """Connecting an in_process config without a resolver raises MCPConnectionError."""
    from kernel.tools.mcp_client import MCPConnectionError

    config = _in_process_config("lisa-content")
    config = MCPClientConfig(
        server_id=config.server_id,
        command=config.command,
        args=config.args,
        env=config.env,
        warm=True,
        required=True,  # required so the error propagates
        runtime_kind="in_process",
        data_dir_path=config.data_dir_path,
        ensure_data_dir=config.ensure_data_dir,
    )
    pool = MCPToolPool([config])  # no in_process_resolver
    try:
        with pytest.raises(MCPConnectionError, match="no in-process resolver"):
            await pool.start()
    finally:
        await pool.aclose()


@pytest.mark.asyncio
async def test_in_process_unknown_server_raises() -> None:
    """An in_process config with an unknown server_id raises MCPConnectionError."""
    from kernel.tools.mcp_client import MCPConnectionError

    config = MCPClientConfig(
        server_id="nonexistent-server",
        command="",
        args=["--data-dir", "/tmp/lisa-test-in-process"],
        env={},
        warm=True,
        required=True,  # required so the error propagates
        runtime_kind="in_process",
    )

    def resolver(server_id: str) -> InProcessServerEntry | None:
        return None  # unknown

    pool = MCPToolPool([config], in_process_resolver=resolver)
    try:
        with pytest.raises(MCPConnectionError, match="not registered"):
            await pool.start()
    finally:
        await pool.aclose()

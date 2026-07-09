"""MCP-backed ``ContentRegistry`` adapter.

Implements the :class:`~kernel.seams.protocols.ContentRegistry` protocol by
calling the ``read_authored_text`` and ``invalidate_cache`` tools on a
``lisa-content`` MCP server via per-call stdio sessions.
"""

from __future__ import annotations

import asyncio
from typing import Any

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from kernel.tools.mcp_client import MCPClientConfig

__all__ = ["MCPContentRegistry"]


def _extract_text(result: Any) -> str:
    """Extract text from an MCP call_tool result."""
    if result.content and len(result.content) > 0:
        return result.content[0].text  # type: ignore[union-attr]
    return ""


class MCPContentRegistry:
    """ContentRegistry backed by an MCP content server.

    Calls ``read_authored_text`` and ``invalidate_cache`` on a
    ``lisa-content`` server via per-call stdio sessions.
    """

    def __init__(self, config: MCPClientConfig) -> None:
        self._params = StdioServerParameters(
            command=config.command,
            args=config.args,
            env=config.env,
            cwd=config.cwd,
        )

    def _call(self, tool: str, arguments: dict[str, Any]) -> str:
        async def _run() -> str:
            async with stdio_client(self._params) as (read, write_stream):
                async with ClientSession(read, write_stream) as session:
                    await session.initialize()
                    result = await session.call_tool(tool, arguments=arguments)
                    return _extract_text(result)

        # Check for a running event loop BEFORE constructing the coroutine
        # to avoid the "coroutine was never awaited" RuntimeWarning.
        try:
            asyncio.get_running_loop()
            in_loop = True
        except RuntimeError:
            in_loop = False

        if in_loop:
            # Already inside a running event loop — create a new one in a thread.
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(asyncio.run, _run())
                return future.result()
        return asyncio.run(_run())

    def read_authored_text(self, relative_path: str) -> str:
        """Return the first matching authored text asset, or empty string."""
        return self._call("read_authored_text", {"relative_path": relative_path})

    def invalidate(self) -> None:
        """Clear the server-side read cache after content roots change."""
        self._call("invalidate_cache", {})

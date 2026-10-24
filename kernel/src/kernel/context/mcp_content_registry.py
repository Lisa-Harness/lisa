"""MCP-backed ``ContentRegistry`` adapter.

Implements the :class:`~kernel.seams.protocols.ContentRegistry` protocol by
calling the ``read_authored_text`` and ``invalidate_cache`` tools on a
``lisa-content`` MCP server via a shared, long-lived stdio session.
"""

from __future__ import annotations

from typing import Any

from mcp.client.stdio import StdioServerParameters

from kernel.context._shared_stdio import SharedStdioSession
from kernel.tools.mcp_client import MCPClientConfig

__all__ = ["MCPContentRegistry"]


def _extract_text(result: Any) -> str:
    """Extract text from an MCP call_tool result."""
    if result.content and len(result.content) > 0:
        text = result.content[0].text
        return text if isinstance(text, str) else ""
    return ""


class MCPContentRegistry:
    """ContentRegistry backed by an MCP content server.

    Calls ``read_authored_text`` and ``invalidate_cache`` on a
    ``lisa-content`` server via a shared, long-lived stdio session.
    """

    def __init__(self, config: MCPClientConfig) -> None:
        self._session = SharedStdioSession(
            StdioServerParameters(
                command=config.command,
                args=config.args,
                env=config.env,
                cwd=config.cwd,
            )
        )

    def prewarm(self) -> None:
        """Start the shared ``lisa-content`` session in the background."""
        self._session.prewarm()

    def _call(self, tool: str, arguments: dict[str, Any]) -> str:
        return _extract_text(self._session.call(tool, arguments))

    def read_authored_text(self, relative_path: str) -> str:
        """Return the first matching authored text asset, or empty string."""
        return self._call("read_authored_text", {"relative_path": relative_path})

    def invalidate(self) -> None:
        """Clear the server-side read cache after content roots change."""
        self._call("invalidate_cache", {})

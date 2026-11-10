"""MCP-backed ``ContentRegistry`` adapter.

Implements the :class:`~kernel.seams.protocols.ContentRegistry` protocol.
Default transport: the ``read_authored_text`` / ``invalidate_cache`` tools
on a ``lisa-content`` MCP server via a shared, long-lived stdio session.
When the host injects a :class:`~kernel.context._catalog_sources.ContentSource`
(in-process mode), reads are served directly by the shared SDK service and
no subprocess is started.
"""

from __future__ import annotations

from typing import Any

from mcp.client.stdio import StdioServerParameters

from kernel.context._catalog_sources import ContentSource
from kernel.context._shared_stdio import SharedSessionLease, SharedStdioSession
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

    def __init__(
        self,
        config: MCPClientConfig,
        lease: SharedSessionLease | None = None,
        source: ContentSource | None = None,
    ) -> None:
        self._session = SharedStdioSession(
            StdioServerParameters(
                command=config.command,
                args=config.args,
                env=config.env,
                cwd=config.cwd,
            ),
            lease=lease,
        )
        self._source: ContentSource | None = source

    def prewarm(self) -> None:
        """Start the shared ``lisa-content`` session in the background.

        A no-op in in-process mode — there is no subprocess to start.
        """
        if self._source is not None:
            return
        self._session.prewarm()

    def _call(self, tool: str, arguments: dict[str, Any]) -> str:
        return _extract_text(self._session.call(tool, arguments))

    def read_authored_text(self, relative_path: str) -> str:
        """Return the first matching authored text asset, or empty string."""
        if self._source is not None:
            return self._source.read_authored_text(relative_path)
        return self._call("read_authored_text", {"relative_path": relative_path})

    def invalidate(self) -> None:
        """Clear the read cache after content roots change."""
        if self._source is not None:
            self._source.invalidate()
            return
        self._call("invalidate_cache", {})

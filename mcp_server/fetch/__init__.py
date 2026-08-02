"""Lisa Fetch MCP server — fetch-and-cache remote http(s) content.

Thin wrapper over :class:`sdk.content.ContentFetcher` (the SDK fetch-and-cache
primitive). The server owns transport and ``MCPServer`` wiring; all fetch/cache
logic lives in the SDK class.

Tools:
- ``fetch_url`` — fetch one URL with optional TTL/force-refresh/max-bytes overrides
- ``invalidate_fetch_cache`` — clear the fetch cache (one URL or all)
"""

from __future__ import annotations

__all__ = ["init_fetch", "main", "mcp"]

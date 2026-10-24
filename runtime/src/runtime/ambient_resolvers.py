"""Register host-owned runtime values for MCP proxy policies."""

from __future__ import annotations

from kernel.tools.mcp_client import MCPToolPool


def register_ambient_resolvers(
    pool: MCPToolPool,
    *,
    user_slug: str,
    current_mode_id: str,
    app_id: str,
) -> None:
    """Register live host values used to scope MCP tool calls."""
    pool.set_ambient_resolver("user_slug", lambda: user_slug)
    pool.set_ambient_resolver("mode_id", lambda: current_mode_id)
    pool.set_ambient_resolver("app_id", lambda: app_id)


__all__ = ["register_ambient_resolvers"]

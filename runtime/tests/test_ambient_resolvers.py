"""Tests for host-provided values used by MCP proxy policies."""

from __future__ import annotations

from kernel.tools.mcp_client import MCPToolPool
from runtime.ambient_resolvers import register_ambient_resolvers


def test_registers_runtime_identity_resolvers() -> None:
    pool = MCPToolPool([])

    register_ambient_resolvers(
        pool,
        user_slug="alice",
        current_mode_id="default",
        app_id="test",
    )

    assert pool._resolve_ambient("user_slug") == "alice"
    assert pool._resolve_ambient("mode_id") == "default"
    assert pool._resolve_ambient("app_id") == "test"

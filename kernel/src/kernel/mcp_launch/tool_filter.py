"""Tool allow/deny filtering for the merged MCP tool catalog.

After :class:`~kernel.tools.mcp_client.MCPToolPool.list_tools` returns, the
host filters the tool catalog against each server's
:class:`~kernel.mcp_launch.policy.ToolPolicy` before passing schemas to
the model. This keeps policy enforcement out of the pool and in a single,
testable function.
"""

from __future__ import annotations

from kernel.mcp_launch.policy import ToolPolicy
from kernel.tools.mcp_client import MCPToolSpec

__all__ = ["filter_tools"]


def filter_tools(tools: list[MCPToolSpec], policy: ToolPolicy) -> list[MCPToolSpec]:
    """Filter a tool list by allow/deny policy."""
    result = tools
    if policy.allowed_tools is not None:
        allowed = set(policy.allowed_tools)
        result = [t for t in result if t.name in allowed]
    if policy.denied_tools:
        denied = set(policy.denied_tools)
        result = [t for t in result if t.name not in denied]
    return result

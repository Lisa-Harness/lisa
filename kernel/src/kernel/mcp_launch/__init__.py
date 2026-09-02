"""Launch policy and runtime adapter for MCP servers.

This package owns the controlled-launch contract for MCP servers:

- :mod:`kernel.mcp_launch.policy` — approved server specs, env/cwd/tool
  policies, and the registry that gates which servers may be spawned.
- :mod:`kernel.mcp_launch.compiler` — runtime adapter that compiles an
  :class:`~kernel.mcp_launch.policy.MCPServerSpec` into a concrete
  :class:`~kernel.tools.mcp_client.MCPClientConfig` for
  :class:`~kernel.tools.mcp_client.MCPToolPool`.
- :mod:`kernel.mcp_launch.tool_filter` — allow/deny filtering of the
  merged tool catalog before it reaches the model.

Only registered specs can be compiled into ``MCPClientConfig`` entries.
Unregistered specs are rejected — the host cannot spawn a server that
isn't in the registry.
"""

from kernel.mcp_launch.policy import (
    CwdPolicy,
    EnvPolicy,
    LaunchTimeouts,
    MCPServerRegistry,
    MCPServerSpec,
    RuntimeKind,
    ToolPolicy,
)

__all__ = [
    "CwdPolicy",
    "EnvPolicy",
    "LaunchTimeouts",
    "MCPServerRegistry",
    "MCPServerSpec",
    "RuntimeKind",
    "ToolPolicy",
]

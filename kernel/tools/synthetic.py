"""Synthetic tool builders for host-owned aliases.

A synthetic tool is a host-owned in-process handler exposed in the same
catalog as MCP-discovered tools (see
:class:`~kernel.tools.mcp_client.SyntheticToolSpec`). This module provides
convenience builders for the common case where a synthetic tool is an
**alias** for an MCP server tool — i.e. a well-known name (e.g.
``memory.search``) that redirects to a specific server's tool (e.g.
``mem0.search_memories``) with an optional argument mapping.

The generic synthetic-tool mechanism itself lives in
:mod:`kernel.tools.mcp_client` (``MCPToolPool.add_synthetic_tools``); this
module is a thin layer that produces :class:`SyntheticToolSpec` entries
wired to a live :class:`MCPToolPool`.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from kernel.tools.mcp_client import MCPToolPool, SyntheticToolSpec

__all__ = ["mcp_alias_tool"]

#: The default empty-schema for synthetic alias tools that forward their
#: arguments verbatim. Callers should pass a real schema when possible.
_EMPTY_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}


def mcp_alias_tool(
    *,
    name: str,
    description: str,
    pool: MCPToolPool,
    target_tool: str,
    input_schema: dict[str, Any] | None = None,
    arg_map: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    target_server_id: str | None = None,
) -> SyntheticToolSpec:
    """Build a synthetic tool that redirects to an MCP server tool.

    The returned :class:`SyntheticToolSpec` dispatches through ``pool`` to
    ``target_tool`` (an MCP-discovered tool name on some server in the pool),
    optionally transforming the caller's arguments via ``arg_map`` first.

    This is the generic "well-known name → MCP server tool" redirect. The
    canonical application is memory aliases (``memory.search`` →
    ``search_memories``), but the mechanism is not memory-specific: any
    made-up name can alias any MCP server tool.

    Register the result via ``pool.add_synthetic_tools([...])`` **after**
    ``pool.start()`` so the alias can route to a live server session.

    Args:
        name: The synthetic tool name the model sees (e.g. ``memory.search``).
        description: Human-readable description shown to the model.
        pool: The live :class:`MCPToolPool` the alias dispatches through.
        target_tool: The MCP-discovered tool name to forward to
            (e.g. ``search_memories``).
        input_schema: JSON-Schema dict for the model-facing tool. Defaults to
            an empty object schema when the alias forwards arguments verbatim.
        arg_map: Optional transform applied to the caller's arguments before
            forwarding. When ``None``, arguments are passed through unchanged.
            Use this to adapt a generic argument shape (e.g.
            ``{"query": ...}``) to a server-specific shape (e.g.
            ``{"query": ..., "user_id": "default"}``).
        target_server_id: Optional server id pinning dispatch to that server.
            When set, the call bypasses the pool's merged namespaced index
            and routes to this server's own tool index — so the alias can
            target a specific backend even when multiple servers expose the
            same bare tool name (e.g. ``mem0``'s ``add_memory`` vs
            ``lisa-memory``'s ``add_memory`` both appear in the catalog as
            ``mem0.add_memory`` and ``lisa-memory.add_memory`` respectively,
            but the alias pins to one). When ``None``, dispatch falls back
            to the merged namespaced index.
    """

    async def handler(arguments: dict[str, Any]) -> str:
        mapped = arg_map(arguments) if arg_map is not None else arguments
        result = await pool.call_tool(
            target_tool,
            mapped,
            server_id=target_server_id,
        )
        return result.output_text

    return SyntheticToolSpec(
        name=name,
        description=description,
        input_schema=input_schema or _EMPTY_SCHEMA,
        handler=handler,
    )


# Type re-exported for convenience so callers can build custom synthetic tools
# (not just aliases) without importing from two modules.
_AsyncHandler = Callable[[dict[str, Any]], Awaitable[str]]
"""``/mcp`` command: MCP server introspection readout.

Renders a per-server summary (status, transport, URL/command, tool count,
last error) and a per-server tool detail (name, description, required
schema args) from the host's live :class:`~kernel.tools.mcp_client.MCPToolPool`.

Transport-neutral Markdown output — the Rich analogue for the CLI panel
lives in :mod:`hosts.cli.render`. The dispatcher
(:func:`hosts.core.handle_message._dispatch_host_command`) already awaits
``runtime.ensure_pool_ready()`` before invoking this handler, so the
renderer is pure: it reads the pool's introspection accessors and the
already-warmed tool catalog without spawning or awaiting.

Dependency direction: ``_mcp`` → ``_log`` (``handle_simple_command`` in
``_simple``), ``kernel.*``, ``hosts.core.*``. No imports from other
command submodules.
"""

from __future__ import annotations

import logging

from kernel.tools.mcp_client import MCPToolSpec

from hosts.core.commands._lists import escape_markdown_cell
from hosts.core.commands._simple import handle_simple_command
from hosts.core.introspect import _gather_mcp_pool_health
from hosts.core.runtime import HostRuntime

logger = logging.getLogger(__name__)

_DESCRIPTION_TRUNC = 80
_LAST_ERROR_TRUNC = 200


async def handle_mcp_command(runtime: HostRuntime, stripped: str, message: str) -> str:
    """Handle ``/mcp`` and ``/mcp <server_id>``.

    ``/mcp`` renders the per-server summary table. ``/mcp <server_id>``
    warms the target server's tools on demand (cold servers lazily
    connected via ``await pool.list_tools(allowed_server_ids={server_id})``,
    same pattern as the CLI ``lisa mcp`` subcommand) and renders the
    per-server tool detail.

    Async because the detail path warms cold servers; the summary path
    reads only the already-warmed pool index (the dispatcher awaits
    ``runtime.ensure_pool_ready()`` before invoking this handler).
    """
    arg = stripped[4:].strip()
    if not arg:
        response_text = render_mcp_summary(runtime)
    else:
        server_id = arg.split()[0]
        response_text = await render_mcp_server_detail(runtime, server_id)
    return handle_simple_command(runtime, message, "mcp", response_text)


def render_mcp_summary(runtime: HostRuntime) -> str:
    """Render a Markdown summary table of every configured MCP server.

    Columns: Server | Status | Transport | URL/Command | Tools | Last Error.
    Reads the live pool via :func:`_gather_mcp_pool_health` so the status,
    tool count, and last error reflect the current runtime state, not a
    stale snapshot.
    """
    pool = runtime.mcp_tool_pool
    health = _gather_mcp_pool_health(pool)
    if not health:
        return "No MCP servers are configured."
    lines = [
        "## MCP Servers",
        "",
        "| Server | Status | Transport | URL/Command | Tools | Last Error |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for server_id in sorted(health):
        h = health[server_id]
        endpoint = h.url or h.command or "—"
        last_error = _truncate(h.last_error, _LAST_ERROR_TRUNC) or "—"
        # A cold server that hasn't been connected yet shows tool_count=0
        # because tools/list hasn't run — render an em-dash (like the empty
        # Last Error column) instead of a misleading 0.
        tool_display = "—" if (h.status == "cold" and h.tool_count == 0) else str(h.tool_count)
        lines.append(
            f"| {escape_markdown_cell(server_id)} "
            f"| {h.status} | {escape_markdown_cell(h.transport)} "
            f"| {escape_markdown_cell(endpoint)} | {tool_display} "
            f"| {escape_markdown_cell(last_error)} |"
        )
    return "\n".join(lines)


async def render_mcp_server_detail(runtime: HostRuntime, server_id: str) -> str:
    """Render a per-server detail: status block + tool table.

    The tool table lists each tool's name, truncated description, and
    required schema args. Cold servers are warmed on demand via
    ``await pool.list_tools(allowed_server_ids={server_id})`` so their
    tools appear without a full pool warm-up.
    """
    pool = runtime.mcp_tool_pool
    health_map = _gather_mcp_pool_health(pool)
    h = health_map.get(server_id)
    if h is None:
        return f"No MCP server named {server_id!r} is configured."
    specs = await _collect_server_tools(runtime, server_id)
    lines = [f"## MCP Server: {server_id}", ""]
    lines.append(f"- status: {h.status}")
    lines.append(f"- transport: {h.transport}")
    if h.url:
        lines.append(f"- url: {h.url}")
    elif h.command:
        lines.append(f"- command: {h.command}")
    lines.append(f"- warm: {h.warm}")
    if h.permanently_failed:
        lines.append(f"- permanently_failed: {h.permanently_failed}")
    if h.restart_count:
        lines.append(f"- restart_count: {h.restart_count}")
    if h.last_error:
        lines.append(f"- last_error: {_truncate(h.last_error, _LAST_ERROR_TRUNC)}")
    lines.append(f"- tools: {len(specs)}")
    if not specs:
        lines.append("")
        lines.append("_No tools discovered (server may be cold or failed)._")
        return "\n".join(lines)
    lines.append("")
    lines.append("| Tool | Description | Required args |")
    lines.append("| --- | --- | --- |")
    for spec in specs:
        description = _truncate(spec.description, _DESCRIPTION_TRUNC) or "—"
        required = _required_args(spec)
        lines.append(
            f"| {escape_markdown_cell(spec.name)} "
            f"| {escape_markdown_cell(description)} "
            f"| {escape_markdown_cell(required)} |"
        )
    return "\n".join(lines)


async def _collect_server_tools(runtime: HostRuntime, server_id: str) -> list[MCPToolSpec]:
    """Return the tools for one server, warming it on demand when cold.

    Reuses the pool's ``list_tools(allowed_server_ids=...)`` lazy-connect
    path so a cold server's tools appear without a full pool warm-up. The
    returned specs are the bare (un-namespaced) ``MCPToolSpec`` entries from
    the per-server index. Returns an empty list when the pool is absent or
    the server cannot be reached.
    """
    pool = runtime.mcp_tool_pool
    if pool is None:
        return []
    tools_by_server = pool.tools_by_server()
    if server_id in tools_by_server:
        return list(tools_by_server[server_id])
    # Cold server not yet connected: warm just this one, then re-read.
    try:
        await pool.list_tools(allowed_server_ids={server_id})
    except Exception as exc:  # pylint: disable=broad-except
        logger.debug("MCP cold warm-up for %r failed: %s", server_id, exc)
        return []
    return list(pool.tools_by_server().get(server_id, []))


def _required_args(spec: MCPToolSpec) -> str:
    """Extract the comma-joined required arg names from a tool's input schema.

    Falls back to all property keys when the schema declares no explicit
    ``required`` list (some MCP servers omit it). Returns ``—`` when the
    schema has no properties.
    """
    schema = spec.input_schema or {}
    required = schema.get("required")
    if isinstance(required, list) and required:
        return ", ".join(str(r) for r in required)
    properties = schema.get("properties")
    if isinstance(properties, dict) and properties:
        return ", ".join(str(k) for k in properties)
    return "—"


def _truncate(text: str, limit: int) -> str:
    """Truncate ``text`` to ``limit`` chars, appending an ellipsis when cut."""
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"
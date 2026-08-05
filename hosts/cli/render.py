"""Rich rendering helpers for the CLI host.

These functions build Rich renderables (Tables, Panels, Text) from the
transport-neutral data surfaced by ``host``. They are the CLI-host
analogue of the plain-text renderers in ``host.commands`` — the core
returns Markdown strings, and here we wrap them in Rich panels/tables for
the terminal.

Nothing in this module imports from ``hosts.cli.repl`` or
``hosts.cli.main``; it is a leaf rendering module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kernel.models import McpServerHealth
from kernel.tools.mcp_client import MCPToolSpec
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    # ``HostRuntime`` is used only as a string annotation (stringified by
    # ``from __future__ import annotations``); the runtime symbols are
    # passed in at call time. Deferring this import keeps ``render`` from
    # dragging in ``host.runtime`` -> ``kernel.ipsar_fsm`` -> ``openai``
    # (~2s) at module load.
    from host.runtime import HostRuntime as HostRuntime

# Heavy imports are deferred into the functions that use them so that
# ``--version`` / ``modes`` / ``models`` / ``verify`` don't pay the cost of
# ``rich.markdown`` (-> ``markdown_it`` + ``pygments`` ~= 1.4s) or
# ``kernel.ipsar_fsm`` / operation / tool registries (~= 1.5s) unless a
# render path that needs them actually runs.

# ── Style constants ───────────────────────────────────────────────────
STYLE_BORDER_SESSION = "steel_blue3"
STYLE_BORDER_RESPONSE = "dark_sea_green4"
STYLE_BORDER_CATALOG = "slate_blue1"


def render_modes(runtime: HostRuntime, *, console: Console | None = None) -> Table:
    """Build a Rich Table of supported modes from the live workflow catalog.

    Uses ``kernel.ipsar_fsm.get_workflow_catalog`` (wired by
    ``host.mcp_pool.setup_framework``) and ``mode_support_gap`` to
    separate supported from unsupported modes.
    """
    from host.commands import mode_support_gap
    from kernel.ipsar_fsm import get_workflow_catalog

    _ = runtime  # catalog is global; runtime kept for future per-host scoping
    catalog = get_workflow_catalog()
    table = Table(title="Lisa Modes")
    table.add_column("Mode", style="bold")
    table.add_column("Kind")
    table.add_column("Command")
    table.add_column("Description")

    for mode_id, mode in sorted(catalog.items()):
        gap = mode_support_gap(mode)
        command = mode.command_pattern or "default route"
        description = mode.description or command
        if gap is not None:
            description = f"[yellow]requires {gap}[/yellow]"
        table.add_row(mode_id, mode.mode_kind, command, description)

    if console is not None:
        console.print(table)
    return table


def render_models(runtime: HostRuntime, *, console: Console | None = None) -> Table:
    """Build a Rich Table of configured models from the model manifest.

    Uses ``host.model_manifest.load_model_manifest`` against the
    canonical runtime data-dir path.
    """
    from host.model_manifest import load_model_manifest

    path = runtime.models_path
    manifest = load_model_manifest(path)
    table = Table(title="Lisa Models")
    table.add_column("Model", style="bold")
    table.add_column("Provider")
    table.add_column("Aliases")
    table.add_column("Reasoning")

    for model in manifest.models:
        aliases = ", ".join(sorted(model.aliases)) or "—"
        reasoning = "yes" if model.reasoning_model else "no"
        table.add_row(model.model_id, model.provider, aliases, reasoning)

    if console is not None:
        console.print(table)
    return table


def render_pack_list_panel(runtime: HostRuntime, *, console: Console | None = None) -> Panel:
    """Render the pack activation list from ``host.commands.render_pack_list``.

    The core returns a Markdown string; we wrap it in a Rich Panel for the
    terminal host.
    """
    from host.commands import render_pack_list
    from rich.markdown import Markdown

    markdown_text = render_pack_list(runtime)
    panel = Panel(
        Markdown(markdown_text),
        title="Pack Activation",
        border_style=STYLE_BORDER_SESSION,
    )
    if console is not None:
        console.print(panel)
    return panel


def render_operations(runtime: HostRuntime, *, console: Console | None = None) -> Table:
    """Build a Rich Table of internal operations with IDs for allowlist authoring.

    Lists every operation discovered from content packs via the
    ``lisa-catalog`` MCP server. The ID column is exactly the string to copy
    into a mode's ``allowed_operation_ids`` front-matter list.
    """
    from kernel.operations.registry import get_operation_catalog

    _ = runtime  # catalog is global; runtime kept for future per-host scoping
    catalog = get_operation_catalog()
    table = Table(title="Lisa Operations")
    table.add_column("Operation ID", style="bold")
    table.add_column("Kind")
    table.add_column("Description")
    table.add_column("Tools", overflow="fold")

    for operation_id, definition in sorted(catalog.items()):
        description = definition.description or "—"
        tools = ", ".join(definition.tool_ids) if definition.tool_ids else "—"
        table.add_row(operation_id, definition.execution_kind, description, tools)

    if console is not None:
        console.print(table)
    return table


def render_tools(
    runtime: HostRuntime,
    *,
    console: Console | None = None,
) -> Table:
    """Build a Rich Table of internal tools with IDs for allowlist authoring.

    Lists every internal tool discovered from content packs via the
    ``lisa-catalog`` MCP server. The ID column is exactly the string to copy
    into a mode's ``allowed_tool_ids`` front-matter list. MCP server tools
    are listed by the dedicated ``lisa mcp`` command (and REPL ``/mcp``).
    """
    _ = runtime  # catalog is global; runtime kept for future per-host scoping
    table = Table(title="Lisa Internal Tools")
    table.add_column("Tool ID", style="bold")
    table.add_column("Description")
    table.add_column("Max/Turn")
    table.add_column("Timeout")

    from kernel.tools.registry import get_tool_catalog

    catalog = get_tool_catalog()
    for tool_id, definition in sorted(catalog.items()):
        description = definition.description or "—"
        table.add_row(
            tool_id,
            description,
            str(definition.max_calls_per_turn),
            f"{definition.timeout_seconds}s",
        )

    if console is not None:
        console.print(table)
    return table


def render_mcp_servers(
    health: dict[str, McpServerHealth],
    *,
    console: Console | None = None,
) -> Table:
    """Render a Rich Table summarizing every configured MCP server.

    Columns: Server | Status | Transport | URL/Command | Tools | Last Error.
    Reads the per-server :class:`McpServerHealth` collected by warming the
    MCP pool (see :func:`hosts.cli.main._collect_mcp_tools`). The status,
    tool count, and last error reflect the current runtime state — a cold
    server that failed to start shows ``failed`` with the failure reason,
    not ``cold``.
    """
    table = Table(title="MCP Servers")
    table.add_column("Server", style="bold")
    table.add_column("Status")
    table.add_column("Transport")
    table.add_column("URL/Command", overflow="fold")
    table.add_column("Tools", justify="right")
    table.add_column("Last Error", overflow="fold")

    for server_id in sorted(health):
        h = health[server_id]
        endpoint = h.url or h.command or "—"
        last_error = _truncate(h.last_error, 200) or "—"
        status_style = _status_style(h.status)
        # A cold server that hasn't been connected yet shows tool_count=0
        # because tools/list hasn't run — render an em-dash (like the empty
        # Last Error column) instead of a misleading 0.
        tool_display = "—" if (h.status == "cold" and h.tool_count == 0) else str(h.tool_count)
        table.add_row(
            server_id,
            f"[{status_style}]{h.status}[/]",
            h.transport,
            endpoint,
            tool_display,
            last_error,
        )

    if console is not None:
        console.print(table)
    return table


def render_mcp_server_tools(
    specs: list[MCPToolSpec],
    health: McpServerHealth,
    *,
    console: Console | None = None,
) -> Table:
    """Render a Rich Table of one MCP server's tools with descriptions + required args.

    Used by ``lisa mcp <server_id>`` to drill into a single server. The
    header (status, transport, URL/command) comes from ``health``; the tool
    rows come from ``specs``. Tool names are the bare strings referenced by
    per-server ``tool_policy.allowed_tools`` / ``denied_tools`` in
    ``mcp_servers.yaml``.
    """
    table = Table(title=f"MCP Server: {health.server_id}")
    table.add_column("Tool", style="bold", overflow="fold")
    table.add_column("Description", overflow="fold")
    table.add_column("Required args", overflow="fold")

    for spec in specs:
        description = _truncate(spec.description, 80) or "—"
        required = _required_args(spec)
        table.add_row(spec.name, description, required)

    if console is not None:
        console.print(table)
    return table


def _status_style(status: str) -> str:
    """Return a Rich style name for an MCP server status glyph."""
    if status == "warm":
        return "green"
    if status == "failed":
        return "red"
    if status == "cold":
        return "yellow"
    return "grey58"


def _truncate(text: str, limit: int) -> str:
    """Truncate ``text`` to ``limit`` chars, appending an ellipsis when cut."""
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def _required_args(spec: MCPToolSpec) -> str:
    """Extract the comma-joined required arg names from a tool's input schema."""
    schema = spec.input_schema or {}
    required = schema.get("required")
    if isinstance(required, list) and required:
        return ", ".join(str(r) for r in required)
    properties = schema.get("properties")
    if isinstance(properties, dict) and properties:
        return ", ".join(str(k) for k in properties)
    return "—"


def render_markdown_panel(
    text: str,
    *,
    title: str = "Response",
    border_style: str = STYLE_BORDER_RESPONSE,
    console: Console | None = None,
) -> Panel:
    """Wrap a Markdown string in a Rich Panel.

    Convenience helper for rendering any plain-text/Markdown response from
    ``host`` in a terminal panel.
    """
    from rich.markdown import Markdown

    panel = Panel(Markdown(text), title=title, border_style=border_style)
    if console is not None:
        console.print(panel)
    return panel


__all__ = [
    "render_markdown_panel",
    "render_mcp_server_tools",
    "render_mcp_servers",
    "render_models",
    "render_modes",
    "render_operations",
    "render_pack_list_panel",
    "render_tools",
]

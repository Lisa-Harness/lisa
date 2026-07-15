"""Rich rendering helpers for the CLI host.

These functions build Rich renderables (Tables, Panels, Text) from the
transport-neutral data surfaced by ``hosts.core``. They are the CLI-host
analogue of the plain-text renderers in ``hosts.core.commands`` — the core
returns Markdown strings, and here we wrap them in Rich panels/tables for
the terminal.

Nothing in this module imports from ``hosts.cli.repl`` or
``hosts.cli.main``; it is a leaf rendering module.
"""

from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from hosts.core.commands import (
    mode_support_gap,
    render_pack_list,
)
from hosts.core.model_manifest import load_model_manifest
from hosts.core.runtime import HostRuntime
from kernel.ipsar_fsm import get_workflow_catalog
from kernel.operations.registry import get_operation_catalog
from kernel.tools.registry import get_tool_catalog

# ── Style constants ───────────────────────────────────────────────────
STYLE_BORDER_SESSION = "steel_blue3"
STYLE_BORDER_RESPONSE = "dark_sea_green4"
STYLE_BORDER_CATALOG = "slate_blue1"


def render_modes(runtime: HostRuntime, *, console: Console | None = None) -> Table:
    """Build a Rich Table of supported modes from the live workflow catalog.

    Uses ``kernel.ipsar_fsm.get_workflow_catalog`` (wired by
    ``hosts.core.mcp_pool.setup_framework``) and ``mode_support_gap`` to
    separate supported from unsupported modes.
    """
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

    Uses ``hosts.core.model_manifest.load_model_manifest`` against the
    canonical runtime data-dir path.
    """
    path = runtime.models_path
    manifest = load_model_manifest(path)
    table = Table(title="Lisa Models")
    table.add_column("Model", style="bold")
    table.add_column("Provider")
    table.add_column("Designations")
    table.add_column("Aliases")
    table.add_column("Reasoning")

    for model in manifest.models:
        designations = ", ".join(sorted(model.designations)) or "—"
        aliases = ", ".join(sorted(model.aliases)) or "—"
        reasoning = "yes" if model.reasoning_model else "no"
        table.add_row(model.model_id, model.provider, designations, aliases, reasoning)

    if console is not None:
        console.print(table)
    return table


def render_pack_list_panel(runtime: HostRuntime, *, console: Console | None = None) -> Panel:
    """Render the pack activation list from ``hosts.core.commands.render_pack_list``.

    The core returns a Markdown string; we wrap it in a Rich Panel for the
    terminal host.
    """
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
    mcp_tools: list[tuple[str, str]] | None = None,
    console: Console | None = None,
) -> Table:
    """Build a Rich Table of internal tools with IDs for allowlist authoring.

    Lists every internal tool discovered from content packs via the
    ``lisa-catalog`` MCP server. The ID column is exactly the string to copy
    into a mode's ``allowed_tool_ids`` front-matter list.

    Internal (pack-discovered) tools and MCP-server tools are listed in
    separate tables so authors can tell which allowlist each ID belongs to:
    pack tool IDs go in ``allowed_tool_ids``; MCP tool IDs are governed by
    per-server ``tool_policy`` in ``mcp_servers.yaml``.

    ``mcp_tools`` is an optional pre-collected list of
    ``(server_id, tool_name)`` pairs (collected by warming the MCP pool —
    see :func:`hosts.cli.main._collect_mcp_tools`). When provided and
    non-empty, a second table is rendered for MCP server tools.
    """
    _ = runtime  # catalog is global; runtime kept for future per-host scoping
    table = Table(title="Lisa Internal Tools")
    table.add_column("Tool ID", style="bold")
    table.add_column("Description")
    table.add_column("Max/Turn")
    table.add_column("Timeout")

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

    if mcp_tools:
        render_mcp_tools(mcp_tools, console=console)

    return table


def render_mcp_tools(
    mcp_tools: list[tuple[str, str]],
    *,
    console: Console | None = None,
) -> Table:
    """Render a Rich Table of MCP server tools for allowlist authoring.

    ``mcp_tools`` is a list of ``(server_id, tool_name)`` pairs collected by
    warming the MCP pool (see :func:`hosts.cli.main._collect_mcp_tools`).
    The tool names are the bare strings referenced by per-server
    ``tool_policy.allowed_tools`` / ``denied_tools`` in ``mcp_servers.yaml``.
    """
    table = Table(title="MCP Server Tools (per-server tool_policy)")
    table.add_column("Server", style="bold")
    table.add_column("Tool", overflow="fold")

    for server_id, tool_name in mcp_tools:
        table.add_row(server_id, tool_name)

    if console is not None:
        console.print(table)
    return table


def render_markdown_panel(
    text: str,
    *,
    title: str = "Response",
    border_style: str = STYLE_BORDER_RESPONSE,
    console: Console | None = None,
) -> Panel:
    """Wrap a Markdown string in a Rich Panel.

    Convenience helper for rendering any plain-text/Markdown response from
    ``hosts.core`` in a terminal panel.
    """
    panel = Panel(Markdown(text), title=title, border_style=border_style)
    if console is not None:
        console.print(panel)
    return panel


__all__ = [
    "render_markdown_panel",
    "render_mcp_tools",
    "render_models",
    "render_modes",
    "render_operations",
    "render_pack_list_panel",
    "render_tools",
]
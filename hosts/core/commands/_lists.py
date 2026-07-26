"""Markdown table renderers for ``/modes``, ``/operations``, ``/tools``, ``/pack list``.

These emit plain Markdown pipe tables (transport-neutral) — the Rich
analogue lives in ``hosts.cli.render``. ``render_pack_list`` reaches into
``_packs`` for the lock loader and installed-roots map.

Dependency direction: ``_lists`` → ``_routing`` (``mode_support_gap``),
``_packs`` (``_load_installed_packs_lock``, ``installed_pack_roots_by_name``),
``kernel.*``, ``hosts.core.content_roots``, ``hosts.core.pack_activation``.
"""

from __future__ import annotations

import logging
from typing import Any

from kernel.ipsar_fsm import get_workflow_catalog
from kernel.operations.registry import get_operation_catalog
from kernel.pack_activation import EXCLUDED_PACK_NAMES, normalize_pack_name
from kernel.tools.registry import get_tool_catalog

from hosts.core.commands._packs import (
    _load_installed_packs_lock,
    installed_pack_roots_by_name,
)
from hosts.core.commands._routing import mode_support_gap
from hosts.core.content_roots import (
    always_on_pack_names,
    pack_is_active,
    runtime_enabled_pack_map,
    runtime_installed_pack_names,
)
from hosts.core.runtime import HostRuntime

logger = logging.getLogger(__name__)


def escape_markdown_cell(value: str) -> str:
    """Escape a string for safe use inside a Markdown pipe-table cell.

    Backslashes are escaped first because Markdown treats ``\\`` as a
    literal backslash. Without this, Windows drive-letter paths like
    ``C:\\Code\\lisa\\.venv`` lose their trailing backslash when rendered
    (``\\.venv`` collapses to ``.venv``), producing misleading paths such
    as ``C:\\Code\\lisa.venv`` instead of ``C:\\Code\\lisa\\.venv``.
    """
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def resolve_pack_source_kind(
    name: str,
    entry: dict[str, Any],
    always_on: set[str],
    runtime_installed: set[str],
) -> str:
    """Resolve the source-kind label for a pack entry."""
    source_kind = str(entry.get("source_kind") or "").strip().lower()
    if source_kind:
        return source_kind
    if name in always_on:
        return "always-on"
    if name in runtime_installed:
        return "package"
    return "bundled"


def render_modes_list(runtime: HostRuntime) -> str:
    """Render supported modes as Markdown for ``/modes``.

    Mirrors :func:`hosts.cli.render.render_modes` but emits a plain
    Markdown pipe table (transport-neutral) instead of a Rich ``Table``.
    Each row carries the mode id, kind, command pattern, and description
    (or the support-gap notice for modes this host cannot run).
    """
    _ = runtime  # catalog is global; runtime kept for future per-host scoping
    catalog = get_workflow_catalog()
    if not catalog:
        return "No modes discovered yet."
    lines = [
        "## Lisa Modes",
        "",
        "| Mode | Kind | Command | Description |",
        "| --- | --- | --- | --- |",
    ]
    for mode_id, mode in sorted(catalog.items()):
        gap = mode_support_gap(mode)
        command = escape_markdown_cell(mode.command_pattern or "default route")
        description = escape_markdown_cell(mode.description or mode.command_pattern or "")
        if gap is not None:
            description = f"requires {escape_markdown_cell(gap)}"
        lines.append(
            f"| {escape_markdown_cell(mode_id)} | {escape_markdown_cell(mode.mode_kind)} "
            f"| {command} | {description} |"
        )
    return "\n".join(lines)


def render_operations_list(runtime: HostRuntime) -> str:
    """Render internal operations as Markdown for ``/operations``.

    Mirrors :func:`hosts.cli.render.render_operations` but emits a plain
    Markdown pipe table. The Operation ID column is the string to copy
    into a mode's ``allowed_operation_ids`` front matter.
    """
    _ = runtime  # catalog is global; runtime kept for future per-host scoping
    catalog = get_operation_catalog()
    if not catalog:
        return "No operations discovered yet."
    lines = [
        "## Lisa Operations",
        "",
        "| Operation ID | Kind | Description | Tools |",
        "| --- | --- | --- | --- |",
    ]
    for operation_id, definition in sorted(catalog.items()):
        description = escape_markdown_cell(definition.description or "—")
        tools = ", ".join(definition.tool_ids) if definition.tool_ids else "—"
        lines.append(
            f"| {escape_markdown_cell(operation_id)} | "
            f"{escape_markdown_cell(definition.execution_kind)} "
            f"| {description} | {escape_markdown_cell(tools)} |"
        )
    return "\n".join(lines)


def render_tools_list(runtime: HostRuntime) -> str:
    """Render internal tools as Markdown for ``/tools``.

    Mirrors :func:`hosts.cli.render.render_tools` but emits a plain
    Markdown pipe table covering the internal (pack-discovered) tool
    catalog. MCP server tools are intentionally omitted — they require a
    live pool warm-up and are surfaced by the ``lisa tools --with-mcp``
    CLI subcommand, not by the transport-neutral slash command.
    """
    _ = runtime  # catalog is global; runtime kept for future per-host scoping
    catalog = get_tool_catalog()
    if not catalog:
        return "No internal tools discovered yet."
    lines = [
        "## Lisa Internal Tools",
        "",
        "| Tool ID | Description | Max/Turn | Timeout |",
        "| --- | --- | --- | --- |",
    ]
    for tool_id, definition in sorted(catalog.items()):
        description = escape_markdown_cell(definition.description or "—")
        lines.append(
            f"| {escape_markdown_cell(tool_id)} | {description} "
            f"| {definition.max_calls_per_turn} | {definition.timeout_seconds}s |"
        )
    return "\n".join(lines)


def render_pack_list(runtime: HostRuntime) -> str:
    """Render discovered packs, activation state, and capability-provider selections."""
    lock = _load_installed_packs_lock(runtime)
    entries = [entry for entry in lock.get("packs", []) if isinstance(entry, dict)]
    installed_roots = installed_pack_roots_by_name()
    always_on = always_on_pack_names(
        runtime.config.mcp_manifest_path,
        always_on_packs=runtime.config.always_on_packs,
    )
    rt_installed = runtime_installed_pack_names(lock)
    enabled_map = runtime_enabled_pack_map(lock)
    entry_map = {
        normalize_pack_name(str(entry.get("package_name") or "")): entry
        for entry in entries
        if normalize_pack_name(str(entry.get("package_name") or ""))
    }

    all_packages = set(installed_roots) | set(entry_map) | always_on
    all_packages = {name for name in all_packages if name not in EXCLUDED_PACK_NAMES}
    if not all_packages:
        return "No packs discovered yet."

    lines = ["## Pack Activation", ""]
    lines.append("| Pack | State | Source | Root |")
    lines.append("| --- | --- | --- | --- |")
    for package_name in sorted(all_packages):
        entry = entry_map.get(package_name) or {}
        source_kind = resolve_pack_source_kind(
            package_name,
            entry,
            always_on,
            rt_installed,
        )
        active = pack_is_active(
            package_name,
            always_on_packs=always_on,
            runtime_installed=rt_installed,
            enabled_map=enabled_map,
        )
        if package_name in always_on:
            activation = "always-on"
        elif active:
            activation = "enabled"
        else:
            activation = "disabled"
        root = installed_roots.get(package_name)
        root_text = str(root) if root is not None else "(not discovered in current environment)"
        lines.append(
            f"| {escape_markdown_cell(package_name)} | {escape_markdown_cell(activation)} | "
            f"{escape_markdown_cell(source_kind)} | {escape_markdown_cell(root_text)} |"
        )

    selected = lock.get("selected_capability_providers")
    if isinstance(selected, dict) and selected:
        lines.append("")
        lines.append("## Selected Capability Providers")
        lines.append("")
        lines.append("| Capability | Provider |")
        lines.append("| --- | --- |")
        for capability_id, provider_id in sorted(selected.items()):
            cap = escape_markdown_cell(str(capability_id))
            prov = escape_markdown_cell(str(provider_id))
            lines.append(f"| {cap} | {prov} |")

    lines.append("")
    lines.append("Tip: use /pack enable <name> or /pack disable <name> for bundled/runtime packs.")
    return "\n".join(lines)

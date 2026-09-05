"""Lisa Catalog MCP server — mode/operation/tool catalog and diagnostics.

This server owns the catalog-building surface, split from ``lisa-content``
so that the file-reading server (``lisa-content``) and the catalog server
(``lisa-catalog``) can be consumed independently. The kernel's
``MCPModeCatalog`` / ``MCPOperationCatalog`` / ``MCPToolCatalog`` adapters
call the tools here; ``MCPContentRegistry`` calls the file-reading tools
on ``lisa-content``.

Tools:
- ``list_modes`` — return the full workflow mode catalog as JSON
- ``list_operations`` — return the full operation catalog as JSON
- ``list_tools`` — return the full internal tool catalog as JSON
- ``list_content_roots`` — list all discovered content root paths
- ``invalidate_catalog`` — clear the mode/operation/tool catalog caches
- ``list_catalog_diagnostics`` — return a diagnostics report for root
  discovery and catalog scan failures

Uses :func:`sdk.content._roots.resolve_content_roots` for root discovery
and :mod:`sdk.content._catalog` / :mod:`sdk.content._diagnostics` for the
pure catalog and diagnostics logic.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.context import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations

# Suppress a benign runpy RuntimeWarning + activate packs venv site-packages.
# See ``mcp_server/_entrypoint.py`` for the full rationale. Imported
# side-effect-first, pinned above the ``sdk.content`` import via `isort:
# skip` (autofixers otherwise reorder it after, which is import-order
# dependent breakage: packs venv must be on sys.path before `sdk.content`
# triggers catalog loading).
import mcp_server._entrypoint  # noqa: F401 — shared stdio entrypoint shim  # isort: skip
from sdk.content import (
    CatalogCollisionError,
    build_catalog_diagnostics_json,
    build_mode_catalog_json,
    build_operation_catalog_json,
    build_tool_catalog_json,
    invalidate_catalog_cache,
    resolve_content_roots,
    resolve_mode_roots,
    resolve_operation_roots,
    resolve_tool_roots,
)

logger = logging.getLogger(__name__)

__all__ = ["CatalogDeps", "init_catalog", "main", "mcp", "set_deps_for_test"]

# ── Lifespan deps ───────────────────────────────────────────────────────────


@dataclass
class CatalogDeps:
    """Server-scoped dependencies owned by the lifespan."""

    extra_roots: list[Path] | None = None


# Test seam: populated by the lifespan in production, or by
# ``set_deps_for_test`` in tests (direct ``call_tool`` does not run the
# lifespan). This is an explicit seam, not a hidden production singleton.
_deps: CatalogDeps | None = None


def set_deps_for_test(deps: CatalogDeps) -> None:
    """Test-only seam: inject ``CatalogDeps`` without spawning stdio.

    The lifespan owns this in production; tests bypass it because the
    SDK's direct ``call_tool`` path does not run the lifespan.
    """
    global _deps  # noqa: PLW0603 — test seam; lifespan owns this in production
    _deps = deps


def _resolve_deps(ctx: Context[Any, Any] | None) -> CatalogDeps:
    """Return the active deps, preferring lifespan context when available."""
    if ctx is not None:
        try:
            rc = ctx.request_context
            deps = rc.lifespan_context
            if isinstance(deps, CatalogDeps):
                return deps
        except ValueError:
            pass  # direct call_tool path — no request_context
    if _deps is None:
        raise RuntimeError("lisa-catalog is not initialized. Call init_catalog().")
    return _deps


def _all_roots(ctx: Context[Any, Any] | None = None) -> list[Path]:
    """Return the current discovered content roots."""
    return resolve_content_roots(_resolve_deps(ctx).extra_roots or [])


# ── CLI config holder (populated by _init before run()) ────────────────────


@asynccontextmanager
async def _lifespan(app: MCPServer[Any]) -> AsyncIterator[CatalogDeps]:
    """Yield the resolved extra roots for catalog discovery."""
    yield CatalogDeps()


mcp = MCPServer("Lisa Catalog Server", lifespan=_lifespan)


def init_catalog(extra_roots: list[Path] | None = None) -> None:
    """Initialize the catalog server with optional extra roots.

    Kept as a thin back-compat shim. Tests should prefer
    :func:`set_deps_for_test` with a :class:`CatalogDeps` instance.
    """
    set_deps_for_test(CatalogDeps(extra_roots=list(extra_roots) if extra_roots else None))


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
async def list_content_roots(
    ctx: Context[CatalogDeps, Any] | None = None,
) -> str:
    """Return a JSON array of all discovered content root paths."""
    roots = _all_roots(ctx)
    return json.dumps([str(r) for r in roots])


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
async def list_modes(
    ctx: Context[CatalogDeps, Any] | None = None,
) -> str:
    """Return the full workflow mode catalog as JSON.

    Scans all discovered content roots for ``modes/`` subdirectories, parses
    front-matter from ``_mode.md`` and stage files, and returns a JSON object
    mapping ``mode_id`` to a serializable mode config dict with nested
    ``stages``.

    Ambient MCP-native tool ids declared in the host manifest
    (``default_available_tools`` on each server in ``mcp_servers.yaml``)
    are merged into every mode's ``allowed_tool_ids`` here, alongside the
    pack-tool ``default_available: true`` ids. The host passes the resolved
    namespaced set via the ``LISA_DEFAULT_MCP_TOOLS`` env var (comma-separated
    ``server_id.tool`` ids); this is the only channel for MCP-native ambient
    ids because those tools are not in the pack tool catalog the catalog
    server scans.

    When ``LISA_OVERLAY_ROOT`` is set, mode entries discovered under that
    root take automatic precedence over pack entries with the same
    ``mode_id``.
    """
    mode_roots = resolve_mode_roots(_all_roots(ctx))
    operation_roots = resolve_operation_roots(_all_roots(ctx))
    tool_roots = resolve_tool_roots(_all_roots(ctx))
    try:
        operation_catalog_json = build_operation_catalog_json(operation_roots)
        tool_catalog_json = build_tool_catalog_json(tool_roots)
        default_mcp_tool_ids = _read_default_mcp_tools_env()
        overlay_root = _read_overlay_root_env()
        return build_mode_catalog_json(
            mode_roots,
            operation_catalog_json=operation_catalog_json,
            tool_catalog_json=tool_catalog_json,
            default_mcp_tool_ids=default_mcp_tool_ids,
            overlay_root=overlay_root,
        )
    except CatalogCollisionError as exc:
        raise ToolError(str(exc)) from exc


def _read_default_mcp_tools_env() -> set[str]:
    """Read ambient MCP-native tool ids from ``LISA_DEFAULT_MCP_TOOLS``.

    The host compiles its ``mcp_servers.yaml`` into a namespaced id set
    (``server_id.tool``) and passes it via this comma-separated env var so
    the catalog subprocess can merge them into mode allowlists without
    importing host-only policy modules. Empty/absent → no MCP-native
    ambient tools.
    """
    raw = os.getenv("LISA_DEFAULT_MCP_TOOLS", "")
    return {part.strip() for part in raw.split(",") if part.strip()}


def _read_overlay_root_env() -> Path | None:
    """Read the overlay root path from ``LISA_OVERLAY_ROOT``.

    The host sets this env var when ``--overlays-dir`` is configured and
    the directory exists. The catalog server uses it to identify overlay
    entries for automatic precedence in duplicate resolution. Returns
    ``None`` when the env var is absent or the path does not exist.
    """
    raw = os.getenv("LISA_OVERLAY_ROOT", "").strip()
    if not raw:
        return None
    path = Path(raw)
    return path if path.is_dir() else None


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
async def list_operations(
    ctx: Context[CatalogDeps, Any] | None = None,
) -> str:
    """Return the full operation catalog as JSON.

    Scans all discovered content roots for ``operations/`` subdirectories,
    discovers ``.operation.md`` (prompt) and ``.py``+``.yml`` (runtime)
    operation definitions, and returns a JSON object mapping
    ``operation_id`` to a serializable operation definition dict.
    """
    operation_roots = resolve_operation_roots(_all_roots(ctx))
    try:
        return build_operation_catalog_json(operation_roots)
    except CatalogCollisionError as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
async def list_tools(
    ctx: Context[CatalogDeps, Any] | None = None,
) -> str:
    """Return the full internal tool catalog as JSON.

    Scans all discovered content roots for ``tools/`` subdirectories,
    discovers ``.py``+``.tool.yml`` runtime tool definitions, and returns a
    JSON object mapping ``tool_id`` to a serializable tool definition dict.
    The kernel imports the implementation module locally using the
    ``module_path`` field (same pattern as operations).
    """
    tool_roots = resolve_tool_roots(_all_roots(ctx))
    try:
        return build_tool_catalog_json(tool_roots)
    except CatalogCollisionError as exc:
        raise ToolError(str(exc)) from exc


@mcp.tool(annotations=ToolAnnotations(idempotent_hint=True))
async def invalidate_catalog() -> str:
    """Clear the mode/operation/tool catalog caches after content roots change."""
    invalidate_catalog_cache()
    return json.dumps({"ok": True})


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
async def list_catalog_diagnostics(
    ctx: Context[CatalogDeps, Any] | None = None,
) -> str:
    """Return a diagnostics report for content root discovery and catalog scan failures.

    Re-scans entry points to capture load failures and missing directories,
    checks each mode directory for missing ``_mode.md`` or invalid front-matter,
    and checks each runtime operation/tool module for missing metadata siblings.
    Use this to inspect *why* a pack contributed nothing instead of seeing an
    empty catalog with no explanation.
    """
    all_roots = _all_roots(ctx)
    mode_roots = resolve_mode_roots(all_roots)
    operation_roots = resolve_operation_roots(all_roots)
    tool_roots = resolve_tool_roots(all_roots)
    return build_catalog_diagnostics_json(
        all_roots=all_roots,
        mode_roots=mode_roots,
        operation_roots=operation_roots,
        tool_roots=tool_roots,
    )


def _init(args: Any) -> None:
    """Populate the test seam so the server is usable after init."""
    set_deps_for_test(CatalogDeps())


def main() -> None:
    """Entry point for the lisa-catalog MCP server (stdio transport).

    Parses the standard CLI contract via the shared ``run()`` helper.
    ``--readonly`` is not applicable (catalog server is read-only by design).
    """
    from mcp_server.config.cli import run

    run("lisa-catalog", mcp, _init)


if __name__ == "__main__":
    main()

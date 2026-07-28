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
import warnings
from pathlib import Path

# Suppress a benign runpy RuntimeWarning emitted by the console-script
# launcher. ``lisa-catalog`` is entered via ``runpy.run_module``, which
# imports ``mcp_server.catalog`` (the package) to locate ``server`` and
# then execs ``server`` as ``__main__``. The side-effect import of
# ``mcp_server._packs_venv_site`` below re-enters the package path, and
# runpy warns that the package is in ``sys.modules`` before ``server``
# finished executing. The "unpredictable behaviour" it warns about only
# occurs if ``server`` re-imports itself as ``__main__`` mid-execution —
# it does not — so the warning is a false alarm for this entry-point
# pattern. Suppressed narrowly (one message, one category) so real
# RuntimeWarnings still surface.
warnings.filterwarnings(
    "ignore",
    message=r".*found in sys.modules after import of package.*",
    category=RuntimeWarning,
)

import mcp_server._packs_venv_site  # noqa: F401 — activates packs venv .pth files

from mcp.server.fastmcp import FastMCP

from sdk.content import (
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

__all__ = ["init_catalog", "main", "mcp"]

mcp = FastMCP("Lisa Catalog Server")

_extra_roots: list[Path] = []


def reset_catalog_state() -> None:
    """Reset all mutable server state (for test isolation)."""
    global _extra_roots
    _extra_roots = []


def init_catalog(extra_roots: list[Path] | None = None) -> None:
    """Initialize the catalog server with optional extra roots.

    Called by ``main()`` after arg parsing. Tests may call this directly.
    """
    global _extra_roots
    _extra_roots = list(extra_roots) if extra_roots else []


def _all_roots() -> list[Path]:
    return resolve_content_roots(_extra_roots)


@mcp.tool()
async def list_content_roots() -> str:
    """Return a JSON array of all discovered content root paths."""
    roots = _all_roots()
    return json.dumps([str(r) for r in roots])


@mcp.tool()
async def list_modes() -> str:
    """Return the full workflow mode catalog as JSON.

    Scans all discovered content roots for ``modes/`` subdirectories, parses
    front-matter from ``_mode.md`` and stage files, and returns a JSON object
    mapping ``mode_id`` to a serializable mode config dict with nested
    ``stages``.
    """
    mode_roots = resolve_mode_roots(_all_roots())
    operation_roots = resolve_operation_roots(_all_roots())
    tool_roots = resolve_tool_roots(_all_roots())
    operation_catalog_json = build_operation_catalog_json(operation_roots)
    tool_catalog_json = build_tool_catalog_json(tool_roots)
    return build_mode_catalog_json(
        mode_roots,
        operation_catalog_json=operation_catalog_json,
        tool_catalog_json=tool_catalog_json,
    )


@mcp.tool()
async def list_operations() -> str:
    """Return the full operation catalog as JSON.

    Scans all discovered content roots for ``operations/`` subdirectories,
    discovers ``.operation.md`` (prompt) and ``.py``+``.yml`` (runtime)
    operation definitions, and returns a JSON object mapping
    ``operation_id`` to a serializable operation definition dict.
    """
    operation_roots = resolve_operation_roots(_all_roots())
    return build_operation_catalog_json(operation_roots)


@mcp.tool()
async def list_tools() -> str:
    """Return the full internal tool catalog as JSON.

    Scans all discovered content roots for ``tools/`` subdirectories,
    discovers ``.py``+``.tool.yml`` runtime tool definitions, and returns a
    JSON object mapping ``tool_id`` to a serializable tool definition dict.
    The kernel imports the implementation module locally using the
    ``module_path`` field (same pattern as operations).
    """
    tool_roots = resolve_tool_roots(_all_roots())
    return build_tool_catalog_json(tool_roots)


@mcp.tool()
async def invalidate_catalog() -> str:
    """Clear the mode/operation/tool catalog caches after content roots change."""
    invalidate_catalog_cache()
    return json.dumps({"ok": True})


@mcp.tool()
async def list_catalog_diagnostics() -> str:
    """Return a diagnostics report for content root discovery and catalog scan failures.

    Re-scans entry points to capture load failures and missing directories,
    checks each mode directory for missing ``_mode.md`` or invalid front-matter,
    and checks each runtime operation/tool module for missing metadata siblings.
    Use this to inspect *why* a pack contributed nothing instead of seeing an
    empty catalog with no explanation.
    """
    all_roots = _all_roots()
    mode_roots = resolve_mode_roots(all_roots)
    operation_roots = resolve_operation_roots(all_roots)
    tool_roots = resolve_tool_roots(all_roots)
    return build_catalog_diagnostics_json(
        all_roots=all_roots,
        mode_roots=mode_roots,
        operation_roots=operation_roots,
        tool_roots=tool_roots,
    )


def main() -> None:
    """Entry point for the lisa-catalog MCP server (stdio transport).

    Parses the standard CLI contract via the shared argparse helper.
    ``--readonly`` is not applicable (catalog server is read-only by design).
    """
    from mcp_server.config.cli import build_parser, configure_logging, resolve_data_dir

    parser = build_parser("lisa-catalog")
    args = parser.parse_args()

    log = configure_logging(args.log_level, server_name="lisa-catalog")
    data_dir = resolve_data_dir(args.data_dir)
    log.info("lisa-catalog starting (data_dir=%s)", data_dir)

    init_catalog()
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()

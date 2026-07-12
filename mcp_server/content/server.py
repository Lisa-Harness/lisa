"""Lisa Content MCP server — the sole content mediator for the kernel.

Tools:
- ``read_authored_text`` — read an authored text asset from merged roots
- ``list_content_roots`` — list all discovered content root paths
- ``list_modes`` — return the full workflow mode catalog as JSON
- ``list_operations`` — return the full operation catalog as JSON
- ``list_tools`` — return the full internal tool catalog as JSON
- ``invalidate_cache`` — clear the file-read cache
- ``invalidate_catalog`` — clear the mode/operation/tool catalog caches
- ``list_catalog_diagnostics`` — return a diagnostics report for root discovery and catalog scan failures

Uses ``kernel.content_roots.discover_installed_content_roots()`` +
``discover_configured_content_roots()`` to find content roots at startup.
The kernel never touches the filesystem directly; it calls these tools
and the server does all filesystem access.

Env vars (behavioral):
- ``SDK_CONTENT_ROOTS`` — additional content roots to scan (path-separated).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from kernel.content_roots import (
    discover_configured_content_roots,
    discover_installed_content_roots,
)
from kernel.pack_activation import load_disabled_pack_names_from_env
from sdk.content._catalog import (
    build_mode_catalog_json,
    build_operation_catalog_json,
    build_tool_catalog_json,
    invalidate_catalog_cache,
    resolve_mode_roots,
    resolve_operation_roots,
    resolve_tool_roots,
)
from sdk.content._diagnostics import build_catalog_diagnostics_json

logger = logging.getLogger(__name__)

__all__ = ["init_content", "main", "mcp"]

mcp = FastMCP("Lisa Content Server")


@dataclass
class _ContentState:
    """Mutable server state, encapsulated to avoid module-global leakage.

    Replaces the prior ``_extra_roots`` / ``_read_cache`` module-level globals
    so test isolation is explicit: call ``reset_content_state()`` between
    tests instead of relying on import-order side effects.
    """

    extra_roots: list[Path] = field(default_factory=list)
    read_cache: dict[tuple[str, str], str] = field(default_factory=dict)


_state = _ContentState()


def reset_content_state() -> None:
    """Reset all mutable server state (for test isolation)."""
    global _state
    _state = _ContentState()


def init_content(extra_roots: list[Path] | None = None) -> None:
    """Initialize the content server with optional extra roots.

    Called by ``main()`` after arg parsing. Tests may call this directly.
    """
    global _state
    _state = _ContentState(extra_roots=list(extra_roots) if extra_roots else [])


def _all_roots() -> list[Path]:
    excluded = load_disabled_pack_names_from_env()
    installed = discover_installed_content_roots(excluded_packs=excluded)
    configured = discover_configured_content_roots()
    seen: set[str] = set()
    merged: list[Path] = []
    for root in (*installed, *configured, *_state.extra_roots):
        key = str(root.resolve())
        if key in seen:
            continue
        seen.add(key)
        merged.append(root)
    return merged


def _roots_fingerprint() -> str:
    """Return a stable string key for the current discovered root set."""
    return "\n".join(str(r.resolve()) for r in _all_roots())


def _is_safe_relative_path(normalized: str) -> bool:
    """Reject paths that could escape a content root.

    Blocks absolute paths and any path containing a ``..`` segment so a
    crafted ``relative_path`` cannot read files outside discovered roots.
    """
    if not normalized:
        return False
    if Path(normalized).is_absolute():
        return False
    if ".." in Path(normalized).parts:
        return False
    return True


@mcp.tool()
def read_authored_text(relative_path: str) -> str:
    """Return the first matching authored text asset, or empty string."""
    normalized = relative_path.strip().strip("/\\")
    if not _is_safe_relative_path(normalized):
        return ""
    fp = _roots_fingerprint()
    cache_key = (fp, normalized)
    cached = _state.read_cache.get(cache_key)
    if cached is not None:
        return cached
    for root in _all_roots():
        candidate = root / normalized
        if candidate.is_file():
            try:
                content = candidate.read_text(encoding="utf-8")
                _state.read_cache[cache_key] = content
                return content
            except OSError:
                continue
    _state.read_cache[cache_key] = ""
    return ""


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
    return build_mode_catalog_json(mode_roots)


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
async def invalidate_cache() -> str:
    """Clear the file-read cache after content roots change."""
    _state.read_cache.clear()
    return json.dumps({"ok": True})


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
    """Entry point for the lisa-content MCP server (stdio transport).

    Parses the standard CLI contract via the shared argparse helper.
    ``--readonly`` is not applicable (content server is read-only by design).
    """
    from mcp_server.config.cli import build_parser, configure_logging, resolve_data_dir

    parser = build_parser("lisa-content")
    args = parser.parse_args()

    log = configure_logging(args.log_level, server_name="lisa-content")
    data_dir = resolve_data_dir(args.data_dir)
    log.info("lisa-content starting (data_dir=%s)", data_dir)

    init_content()
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()


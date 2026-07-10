"""Lisa Content MCP server — the sole content mediator for the kernel.

Tools:
- ``read_authored_text`` — read an authored text asset from merged roots
- ``list_content_roots`` — list all discovered content root paths
- ``list_modes`` — return the full workflow mode catalog as JSON
- ``list_operations`` — return the full operation catalog as JSON
- ``list_tools`` — return the full internal tool catalog as JSON
- ``invalidate_cache`` — clear the file-read cache
- ``invalidate_catalog`` — clear the mode/operation catalog caches

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
from functools import lru_cache
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from kernel.content_roots import (
    discover_configured_content_roots,
    discover_installed_content_roots,
)
from lisa_sdk.lisa_content._catalog import (
    build_mode_catalog_json,
    build_operation_catalog_json,
    build_tool_catalog_json,
    resolve_mode_roots,
    resolve_operation_roots,
    resolve_tool_roots,
)

logger = logging.getLogger(__name__)

__all__ = ["init_content", "main", "mcp"]

mcp = FastMCP("Lisa Content Server")

_extra_roots: list[Path] = []


def init_content(extra_roots: list[Path] | None = None) -> None:
    """Initialize the content server with optional extra roots.

    Called by ``main()`` after arg parsing. Tests may call this directly.
    """
    global _extra_roots
    _extra_roots = list(extra_roots) if extra_roots else []


def _all_roots() -> list[Path]:
    installed = discover_installed_content_roots()
    configured = discover_configured_content_roots()
    seen: set[str] = set()
    merged: list[Path] = []
    for root in (*installed, *configured, *_extra_roots):
        key = str(root.resolve())
        if key in seen:
            continue
        seen.add(key)
        merged.append(root)
    return merged


@mcp.tool()
@lru_cache(maxsize=512)
def read_authored_text(relative_path: str) -> str:
    """Return the first matching authored text asset, or empty string."""
    normalized = relative_path.strip().strip("/\\")
    if not normalized:
        return ""
    for root in _all_roots():
        candidate = root / normalized
        if candidate.is_file():
            try:
                return candidate.read_text(encoding="utf-8")
            except OSError:
                continue
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
    read_authored_text.cache_clear()  # type: ignore[attr-defined]
    return json.dumps({"ok": True})


@mcp.tool()
async def invalidate_catalog() -> str:
    """Clear the mode/operation catalog caches after content roots change."""
    # The catalog builders are not lru_cached (they scan fresh each call),
    # but this tool exists so the kernel can signal a cache bust intentually.
    # If caching is added later, this is where the clear goes.
    return json.dumps({"ok": True})


def main() -> None:
    """Entry point for the lisa-content MCP server (stdio transport).

    Parses the standard CLI contract via the shared argparse helper.
    ``--readonly`` is not applicable (content server is read-only by design).
    """
    from lisa_sdk.lisa_mcp_config.cli import build_parser, configure_logging, resolve_data_dir

    parser = build_parser("lisa-content")
    args = parser.parse_args()

    log = configure_logging(args.log_level, server_name="lisa-content")
    data_dir = resolve_data_dir(args.data_dir)
    log.info("lisa-content starting (data_dir=%s)", data_dir)

    init_content()
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()

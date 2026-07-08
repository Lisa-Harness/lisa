"""Lisa Content MCP server — content registry over discovered content roots.

Tools:
- ``read_authored_text`` — read an authored text asset from merged roots
- ``list_content_roots`` — list all discovered content root paths
- ``invalidate_cache`` — clear the read cache

Uses ``kernel.content_roots.discover_installed_content_roots()`` +
``discover_configured_content_roots()`` to find content roots at startup.
``read_authored_text`` searches all discovered roots, returns first match
or empty string (same logic as ``DiskContentRegistry``).

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
async def invalidate_cache() -> str:
    """Clear the read cache after content roots change."""
    read_authored_text.cache_clear()  # type: ignore[attr-defined]
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

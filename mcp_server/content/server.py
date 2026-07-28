"""Lisa Content MCP server — authored-text file reading for the kernel.

This server owns the **file-reading** surface. The catalog-building tools
(``list_modes`` / ``list_operations`` / ``list_tools`` / ``invalidate_catalog``
/ ``list_catalog_diagnostics``) moved to the dedicated ``lisa-catalog`` server
so that consumers who only need file reading don't get the catalog tools and
vice versa.

Tools:
- ``read_authored_text`` — read an authored text asset from merged roots
- ``list_content_roots`` — list all discovered content root paths
- ``invalidate_cache`` — clear the file-read cache

Uses :class:`sdk.content._reader.ContentReader` for the path-safety check,
root traversal, and read cache. The kernel never touches the filesystem
directly; it calls these tools and the server delegates to the SDK.
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

# Suppress a benign runpy RuntimeWarning emitted by the console-script
# launcher. See ``mcp_server/catalog/server.py`` for the full rationale;
# ``lisa-content`` has the same side-effect import + entry-point pattern.
warnings.filterwarnings(
    "ignore",
    message=r".*found in sys.modules after import of package.*",
    category=RuntimeWarning,
)

import mcp_server._packs_venv_site  # noqa: F401 — activates packs venv .pth files

from mcp.server.fastmcp import FastMCP

from sdk.content import ContentReader

logger = logging.getLogger(__name__)

__all__ = ["init_content", "main", "mcp"]

mcp = FastMCP("Lisa Content Server")

_reader: ContentReader | None = None


def reset_content_state() -> None:
    """Reset all mutable server state (for test isolation)."""
    global _reader
    _reader = ContentReader()


def _reader_instance() -> ContentReader:
    global _reader
    if _reader is None:
        _reader = ContentReader()
    reader: ContentReader = _reader
    return reader


def init_content(extra_roots: list[Path] | None = None) -> None:
    """Initialize the content server with optional extra roots.

    Called by ``main()`` after arg parsing. Tests may call this directly.
    """
    global _reader
    _reader = ContentReader(extra_roots=extra_roots)


def _all_roots() -> list[Path]:
    """Return the current discovered content roots (delegates to the reader)."""
    return _reader_instance().roots()


@mcp.tool()
def read_authored_text(relative_path: str) -> str:
    """Return the first matching authored text asset, or empty string."""
    return _reader_instance().read_authored_text(relative_path)


@mcp.tool()
async def list_content_roots() -> str:
    """Return a JSON array of all discovered content root paths."""
    roots = _reader_instance().roots()
    return json.dumps([str(r) for r in roots])


@mcp.tool()
async def invalidate_cache() -> str:
    """Clear the file-read cache after content roots change."""
    _reader_instance().invalidate()
    return json.dumps({"ok": True})


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

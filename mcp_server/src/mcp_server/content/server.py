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
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.context import Context
from mcp_types import ToolAnnotations
from sdk.content import ContentReader

# Suppress a benign runpy RuntimeWarning + activate packs venv site-packages.
# See ``mcp_server/_entrypoint.py`` for the full rationale. Imported
# side-effect-first at the top so the packs venv is on sys.path before any
# `from sdk.content import ...` triggers catalog loading.
import mcp_server._entrypoint  # noqa: F401 — shared stdio entrypoint shim

logger = logging.getLogger(__name__)

__all__ = ["ContentDeps", "init_content", "main", "mcp", "set_deps_for_test"]

# ── Lifespan deps ───────────────────────────────────────────────────────────


@dataclass
class ContentDeps:
    """Server-scoped dependencies owned by the lifespan."""

    reader: ContentReader


# Test seam: populated by the lifespan in production, or by
# ``set_deps_for_test`` in tests (direct ``call_tool`` does not run the
# lifespan). This is an explicit seam, not a hidden production singleton.
_deps: ContentDeps | None = None


def set_deps_for_test(deps: ContentDeps) -> None:
    """Test-only seam: inject ``ContentDeps`` without spawning stdio.

    The lifespan owns this in production; tests bypass it because the
    SDK's direct ``call_tool`` path does not run the lifespan.
    """
    global _deps  # noqa: PLW0603 — test seam; lifespan owns this in production
    _deps = deps


def _resolve_deps(ctx: Context[Any, Any] | None) -> ContentDeps:
    """Return the active deps, preferring lifespan context when available."""
    if ctx is not None:
        try:
            rc = ctx.request_context
            deps = rc.lifespan_context
            if isinstance(deps, ContentDeps):
                return deps
        except ValueError:
            pass  # direct call_tool path — no request_context
    if _deps is None:
        raise RuntimeError("lisa-content is not initialized. Call init_content().")
    return _deps


# ── CLI config holder (populated by _init before run()) ────────────────────


@dataclass
class _ContentConfig:
    extra_roots: list[Path] | None = None


_cfg = _ContentConfig()


@asynccontextmanager
async def _lifespan(app: MCPServer[Any]) -> AsyncIterator[ContentDeps]:
    """Construct the content reader from resolved CLI config and yield it."""
    yield ContentDeps(reader=ContentReader(extra_roots=_cfg.extra_roots))


mcp = MCPServer("Lisa Content Server", lifespan=_lifespan)


def init_content(extra_roots: list[Path] | None = None) -> None:
    """Initialize the content server with optional extra roots.

    Kept as a thin back-compat shim. Tests should prefer
    :func:`set_deps_for_test` with a :class:`ContentDeps` instance.
    """
    set_deps_for_test(ContentDeps(reader=ContentReader(extra_roots=extra_roots)))


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
def read_authored_text(
    relative_path: str,
    ctx: Context[ContentDeps, Any] | None = None,
) -> str:
    """Return the first matching authored text asset, or empty string."""
    return _resolve_deps(ctx).reader.read_authored_text(relative_path)


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
async def list_content_roots(
    ctx: Context[ContentDeps, Any] | None = None,
) -> str:
    """Return a JSON array of all discovered content root paths."""
    roots = _resolve_deps(ctx).reader.roots()
    return json.dumps([str(r) for r in roots])


@mcp.tool(annotations=ToolAnnotations(idempotent_hint=True))
async def invalidate_cache(
    ctx: Context[ContentDeps, Any] | None = None,
) -> str:
    """Clear the file-read cache after content roots change."""
    _resolve_deps(ctx).reader.invalidate()
    return json.dumps({"ok": True})


def _init(args: Any) -> None:
    """Populate ``_cfg`` from parsed args so the lifespan can build the reader."""
    _cfg.extra_roots = None
    # Also populate the test seam so the server is usable immediately after
    # init (the lifespan only runs during a real stdio session).
    set_deps_for_test(ContentDeps(reader=ContentReader()))


def main() -> None:
    """Entry point for the lisa-content MCP server (stdio transport).

    Parses the standard CLI contract via the shared ``run()`` helper.
    ``--readonly`` is not applicable (content server is read-only by design).
    """
    from mcp_server.config.cli import run

    run("lisa-content", mcp, _init)


if __name__ == "__main__":
    main()

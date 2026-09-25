"""Lisa Fetch MCP server — fetch-and-cache remote http(s) content.

Thin wrapper over :class:`sdk.content.ContentFetcher`, mirroring the
``mcp_server/content/server.py`` pattern: module-level ``MCPServer`` instance,
``FetchDeps`` carried by the lifespan, one ``@mcp.tool()`` per operation,
``init_fetch()`` for setup, ``set_deps_for_test()`` for test isolation, and
``main()`` using the shared CLI helpers.

The server is a pure pass-through; all network/cache/SSRF logic lives in the
SDK class, never in the server.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from dataclasses import replace as _policy_replace
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.context import Context
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import ToolAnnotations

# Suppress a benign runpy RuntimeWarning + activate packs venv site-packages.
# See ``mcp_server/_entrypoint.py`` for the full rationale. Imported
# side-effect-first at the top so the packs venv is on sys.path before any
# `from sdk.content import ...` triggers catalog loading.
import mcp_server._entrypoint  # noqa: F401 — shared stdio entrypoint shim
from sdk.content import ContentFetcher, FetchPolicy

logger = logging.getLogger(__name__)

__all__ = ["FetchDeps", "init_fetch", "main", "mcp", "set_deps_for_test"]

# ── Lifespan deps ───────────────────────────────────────────────────────────


@dataclass
class FetchDeps:
    """Server-scoped dependencies owned by the lifespan.

    Carries the fetcher and a copy of its default policy so per-call
    overrides can be built without reaching into the fetcher's private
    ``_default_policy`` attribute (decoupling the server from the SDK
    class internals).
    """

    fetcher: ContentFetcher
    default_policy: FetchPolicy = field(default_factory=FetchPolicy)


# Test seam: populated by the lifespan in production, or by
# ``set_deps_for_test`` in tests (direct ``call_tool`` does not run the
# lifespan). This is an explicit seam, not a hidden production singleton.
_deps: FetchDeps | None = None


def set_deps_for_test(deps: FetchDeps) -> None:
    """Test-only seam: inject ``FetchDeps`` without spawning stdio.

    The lifespan owns this in production; tests bypass it because the
    SDK's direct ``call_tool`` path does not run the lifespan.
    """
    global _deps  # noqa: PLW0603 — test seam; lifespan owns this in production
    _deps = deps


def _resolve_deps(ctx: Context[Any, Any] | None) -> FetchDeps:
    """Return the active deps, preferring lifespan context when available."""
    if ctx is not None:
        try:
            rc = ctx.request_context
            deps = rc.lifespan_context
            if isinstance(deps, FetchDeps):
                return deps
        except ValueError:
            pass  # direct call_tool path — no request_context
    if _deps is None:
        raise ToolError("lisa-fetch is not initialized. Call set_deps_for_test(deps).")
    return _deps


# ── CLI config holder (populated by _init before run()) ────────────────────


@dataclass
class _FetchConfig:
    data_dir: Path = Path()


_cfg = _FetchConfig()


@asynccontextmanager
async def _lifespan(app: MCPServer[Any]) -> AsyncIterator[FetchDeps]:
    """Construct the fetcher from resolved CLI config and yield it.

    The server owns a ``FetchPolicy`` copy in deps so per-call overrides
    can be built via ``dataclasses.replace`` without reaching into the
    fetcher's private ``_default_policy`` attribute.
    """
    fetcher = ContentFetcher(_cfg.data_dir)
    yield FetchDeps(fetcher=fetcher, default_policy=fetcher._default_policy)  # type: ignore[reportPrivateUsage]


mcp = MCPServer("Lisa Fetch Server", lifespan=_lifespan)


@mcp.tool(annotations=ToolAnnotations(read_only_hint=True))
async def fetch_url(
    url: str,
    ttl_seconds: float | None = None,
    force_refresh: bool = False,
    max_bytes: int | None = None,
    ctx: Context[FetchDeps, Any] | None = None,
) -> str:
    """Fetch one http(s) URL and return its text, caching per the effective policy.

    Optional overrides build a per-call :class:`FetchPolicy` on top of the
    server default. SSRF rejection (scheme/host/private-IP) raises a
    :class:`ToolError` (surfaced from the SDK fetcher's ``ValueError``).
    On a network failure with a valid stale cache, the stale content is
    returned.
    """
    deps = _resolve_deps(ctx)
    base = deps.default_policy
    overrides: dict[str, object] = {}
    if ttl_seconds is not None:
        overrides["ttl_seconds"] = ttl_seconds
    if force_refresh:
        overrides["force_refresh"] = True
    if max_bytes is not None:
        overrides["max_bytes"] = max_bytes
    policy = _policy_replace(base, **overrides) if overrides else base
    result = await deps.fetcher.fetch(url, policy=policy)
    return json.dumps(
        {
            "url": result.url,
            "status_code": result.status_code,
            "text": result.text,
            "content_type": result.content_type,
            "etag": result.etag,
            "last_modified": result.last_modified,
            "fetched_at": result.fetched_at,
            "from_cache": result.from_cache,
        }
    )


@mcp.tool(annotations=ToolAnnotations(idempotent_hint=True))
async def invalidate_fetch_cache(
    url: str | None = None,
    ctx: Context[FetchDeps, Any] | None = None,
) -> str:
    """Clear the fetch cache. If ``url`` is given, clear only that entry."""
    _resolve_deps(ctx).fetcher.invalidate(url)
    return json.dumps({"ok": True})


def init_fetch(
    data_dir: Path,
    *,
    default_policy: FetchPolicy | None = None,
    allowed_hosts: set[str] | None = None,
    allow_http: bool = False,
) -> None:
    """Initialize the fetch server.

    Kept as a thin back-compat shim. Tests should prefer
    :func:`set_deps_for_test` with a :class:`FetchDeps` instance.
    """
    fetcher = ContentFetcher(
        data_dir,
        default_policy=default_policy or FetchPolicy(),
        allowed_hosts=allowed_hosts,
        allow_http=allow_http,
    )
    set_deps_for_test(FetchDeps(fetcher=fetcher, default_policy=fetcher._default_policy))  # type: ignore[reportPrivateUsage]


def _init(args: Any) -> None:
    """Populate ``_cfg`` from parsed args so the lifespan can build the fetcher."""
    _cfg.data_dir = Path(args.data_dir)
    # Also populate the test seam so the server is usable immediately after
    # init (the lifespan only runs during a real stdio session).
    fetcher = ContentFetcher(_cfg.data_dir)
    set_deps_for_test(FetchDeps(fetcher=fetcher, default_policy=fetcher._default_policy))  # type: ignore[reportPrivateUsage]


def main() -> None:
    """Entry point for the lisa-fetch MCP server (stdio transport)."""
    from mcp_server.config.cli import run

    run("lisa-fetch", mcp, _init)


if __name__ == "__main__":
    main()

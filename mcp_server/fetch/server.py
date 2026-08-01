"""Lisa Fetch MCP server — fetch-and-cache remote http(s) content.

Thin wrapper over :class:`sdk.content.ContentFetcher`, mirroring the
``mcp_server/content/server.py`` pattern: module-level ``FastMCP`` instance,
lazy singleton holding the SDK fetcher, one ``@mcp.tool()`` per operation,
``init_fetch()`` for setup, ``reset_fetch_state()`` for test isolation, and
``main()`` using the shared CLI helpers.

The server is a pure pass-through; all network/cache/SSRF logic lives in the
SDK class, never in the server.
"""

from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path

# Suppress a benign runpy RuntimeWarning emitted by the console-script
# launcher. See ``mcp_server/catalog/server.py`` for the full rationale.
warnings.filterwarnings(
    "ignore",
    message=r".*found in sys.modules after import of package.*",
    category=RuntimeWarning,
)

from dataclasses import replace as _policy_replace

from mcp.server.fastmcp import FastMCP

import mcp_server._packs_venv_site  # noqa: F401 — activates packs venv .pth files
from sdk.content import ContentFetcher, FetchPolicy

logger = logging.getLogger(__name__)

__all__ = ["init_fetch", "main", "mcp", "reset_fetch_state"]

mcp = FastMCP("Lisa Fetch Server")

_fetcher: ContentFetcher | None = None


def reset_fetch_state() -> None:
    """Reset all mutable server state (for test isolation)."""
    global _fetcher
    _fetcher = None


def _fetcher_instance() -> ContentFetcher:
    """Return the active fetcher, raising if uninitialized.

    Mirrors ``mcp_server.storage._backend_instance``: the server requires an
    explicit ``--data-dir`` via the CLI contract. There is deliberately NO
    CWD-relative ``Path("artifacts")`` fallback — that resolves differently
    depending on the process working directory and re-nests caches under a
    subdirectory (the 2026-07-30 ``artifacts/files/artifacts/files``
    incident). ``init_fetch`` must be called first (as ``main()`` does).
    """
    if _fetcher is None:
        raise RuntimeError("lisa-fetch is not initialized. Call init_fetch(data_dir).")
    f: ContentFetcher = _fetcher
    return f


def init_fetch(
    data_dir: Path,
    *,
    default_policy: FetchPolicy | None = None,
    allowed_hosts: set[str] | None = None,
    allow_http: bool = False,
) -> None:
    """Initialize the fetch server.

    Called by ``main()`` after arg parsing. Tests may call this directly.
    """
    global _fetcher
    _fetcher = ContentFetcher(
        data_dir,
        default_policy=default_policy or FetchPolicy(),
        allowed_hosts=allowed_hosts,
        allow_http=allow_http,
    )


@mcp.tool()
async def fetch_url(
    url: str,
    ttl_seconds: float | None = None,
    force_refresh: bool = False,
    max_bytes: int | None = None,
) -> str:
    """Fetch one http(s) URL and return its text, caching per the effective policy.

    Optional overrides build a per-call :class:`FetchPolicy` on top of the
    server default. SSRF rejection (scheme/host/private-IP) raises a
    ``ValueError`` surfaced as an MCP error. On a network failure with a
    valid stale cache, the stale content is returned.
    """
    base = _fetcher_instance()._default_policy
    overrides: dict[str, object] = {}
    if ttl_seconds is not None:
        overrides["ttl_seconds"] = ttl_seconds
    if force_refresh:
        overrides["force_refresh"] = True
    if max_bytes is not None:
        overrides["max_bytes"] = max_bytes
    policy = _policy_replace(base, **overrides) if overrides else base
    result = await _fetcher_instance().fetch(url, policy=policy)
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


@mcp.tool()
async def invalidate_fetch_cache(url: str | None = None) -> str:
    """Clear the fetch cache. If ``url`` is given, clear only that entry."""
    _fetcher_instance().invalidate(url)
    return json.dumps({"ok": True})


def main() -> None:
    """Entry point for the lisa-fetch MCP server (stdio transport)."""
    from mcp_server.config.cli import build_parser, configure_logging, resolve_data_dir

    parser = build_parser("lisa-fetch")
    args = parser.parse_args()

    log = configure_logging(args.log_level, server_name="lisa-fetch")
    data_dir = resolve_data_dir(args.data_dir)
    log.info("lisa-fetch starting (data_dir=%s)", data_dir)

    init_fetch(data_dir)
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()

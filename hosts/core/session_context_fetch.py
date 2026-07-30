"""Remote session context fetching and caching.

Thin adapter over :class:`sdk.content.ContentFetcher`, the SDK fetch-and-cache
primitive. The fetcher owns SSRF validation, the L1/L2 cache, conditional-GET,
and stale-fallback. This module owns only the session-context-specific
**policy translation** (env getters → :class:`FetchPolicy` / ctor args) and
the legacy ``Path`` return contract: the fetched body is written to
``data_dir / "session_context_cache" / "<hash>.md"`` so existing callers in
:mod:`hosts.core.session_context` and the meta-inspecting tests keep working.

Env getters stay here (in the adapter, not the primitive) per the design:
the primitive takes explicit args and reads no env.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import urllib.parse
from pathlib import Path

from sdk.content import ContentFetcher, FetchPolicy

logger = logging.getLogger(__name__)

LISA_SESSION_CONTEXT_ALLOWED_HOSTS_ENV = "LISA_SESSION_CONTEXT_ALLOWED_HOSTS"
LISA_SESSION_CONTEXT_ALLOW_HTTP_ENV = "LISA_SESSION_CONTEXT_ALLOW_HTTP"
LISA_SESSION_CONTEXT_AUTH_HEADER_ENV = "LISA_SESSION_CONTEXT_AUTH_HEADER"
LISA_SESSION_CONTEXT_TOKEN_ENV = "LISA_SESSION_CONTEXT_TOKEN"
LISA_SESSION_CONTEXT_FETCH_TIMEOUT_ENV = "LISA_SESSION_CONTEXT_FETCH_TIMEOUT"


def session_context_allowed_hosts() -> set[str]:
    """Return allowed hosts for remote session context URIs."""
    configured = os.getenv(LISA_SESSION_CONTEXT_ALLOWED_HOSTS_ENV, "").strip()
    if not configured:
        return {"github.com", "raw.githubusercontent.com"}
    values = {value.strip().lower() for value in configured.split(",") if value.strip()}
    return values or {"github.com", "raw.githubusercontent.com"}


def session_context_allow_http() -> bool:
    """Return whether insecure http:// schema is permitted for session contexts."""
    configured = os.getenv(LISA_SESSION_CONTEXT_ALLOW_HTTP_ENV, "false").strip().lower()
    return configured in {"1", "true", "yes", "on"}


def session_context_auth_header() -> str | None:
    """Return the raw auth header to include when fetching, if any."""
    raw_header = os.getenv(LISA_SESSION_CONTEXT_AUTH_HEADER_ENV, "").strip()
    if raw_header:
        return raw_header
    token = os.getenv(LISA_SESSION_CONTEXT_TOKEN_ENV, "").strip()
    if token:
        return f"Bearer {token}"
    return None


def session_context_fetch_timeout() -> float:
    """Return the timeout in seconds for fetching remote session context."""
    configured = os.getenv(LISA_SESSION_CONTEXT_FETCH_TIMEOUT_ENV, "10").strip()
    try:
        return float(configured)
    except ValueError:
        return 10.0


def _cache_path_for(url: str, data_dir: Path) -> Path:
    """Return the deterministic cache .md file path for a URL."""
    cache_dir = data_dir / "session_context_cache"
    hashed = hashlib.sha256(url.strip().encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{hashed}.md"


def _cache_meta_path(url: str, data_dir: Path) -> Path:
    """Return the deterministic cache .meta.json file path for a URL."""
    cache_dir = data_dir / "session_context_cache"
    hashed = hashlib.sha256(url.strip().encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{hashed}.meta.json"


def fetch_session_context_url(
    url: str,
    *,
    data_dir: Path,
    force_refresh: bool = False,
    timeout: float | None = None,
) -> Path:
    """Fetch a remote session context, caching via the SDK ContentFetcher.

    Delegates to :class:`sdk.content.ContentFetcher` for SSRF validation,
    L1/L2 caching, conditional-GET, and stale-fallback. The fetched body is
    written to the legacy ``data_dir / "session_context_cache" / "<hash>.md"``
    path (and a sibling ``.meta.json``) so callers that expect a ``Path``
    and the meta-inspecting tests keep working.

    Raises ValueError if the scheme or host is rejected, or if the network fetch
    fails with no valid stale cache to fall back upon.
    """
    s = url.strip()

    # Build the fetcher with session-context-specific security policy
    # translated from env. The primitive enforces scheme/host/private-IP.
    fetcher = ContentFetcher(
        data_dir,
        default_policy=FetchPolicy(
            ttl_seconds=86_400.0,
            force_refresh=force_refresh,
            max_bytes=200_000,
            cache=True,
            revalidate_etag=True,
            timeout_seconds=timeout if timeout is not None else session_context_fetch_timeout(),
            auth_header=session_context_auth_header(),
        ),
        allowed_hosts=session_context_allowed_hosts(),
        allow_http=session_context_allow_http(),
    )

    # Bridge async→sync at this narrow boundary. The caller
    # (``hosts.core.session_context``) is sync.
    try:
        result = asyncio.run(fetcher.fetch(s))
    except ValueError:
        # Re-raise SSRF / no-cache errors unchanged for the existing tests.
        raise

    # Write to the legacy cache paths for back-compat with the Path contract
    # and the meta-inspecting tests.
    cache_md = _cache_path_for(s, data_dir)
    cache_meta = _cache_meta_path(s, data_dir)
    cache_md.parent.mkdir(parents=True, exist_ok=True)
    cache_md.write_bytes(result.body)
    new_meta = {
        "source_ref": s,
        "fetched_at": result.fetched_at,
        "etag": result.etag,
        "last_modified": result.last_modified,
    }
    cache_meta.write_text(json.dumps(new_meta), encoding="utf-8")
    return cache_md

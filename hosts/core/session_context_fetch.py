"""Remote session context fetching and caching.

Provides transport-neutral fetching of remote session context files via HTTP(S),
with conditional GET caching and trust-policy enforcement.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import httpx

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
    """Fetch a remote session context, caching via conditional GET.

    Raises ValueError if the scheme or host is rejected, or if the network fetch
    fails with no valid stale cache to fall back upon.
    """
    s = url.strip()
    parsed = urllib.parse.urlparse(s)

    # 1. Scheme checks
    if parsed.scheme == "http" and not session_context_allow_http():
        raise ValueError(
            f"Blocked HTTP session context '{s}'. HTTPS is required by default. "
            f"Set {LISA_SESSION_CONTEXT_ALLOW_HTTP_ENV}=1 to override."
        )
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Session context URL must be http or https, got '{s}'.")

    # 2. Host checks
    host = (parsed.hostname or "").lower()
    allowed = session_context_allowed_hosts()
    if host not in allowed:
        raise ValueError(
            f"Blocked session context from host '{host}'. Allowed hosts are: "
            f"{', '.join(allowed)}."
        )

    # 3. Path setup
    cache_md = _cache_path_for(s, data_dir)
    cache_meta = _cache_meta_path(s, data_dir)
    cache_md.parent.mkdir(parents=True, exist_ok=True)

    have_cache = cache_md.is_file() and cache_meta.is_file()
    
    meta: dict[str, str] = {}
    if have_cache:
        try:
            meta = json.loads(cache_meta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            have_cache = False

    # 4. Fetch conditional GET
    headers = {"User-Agent": "lisa_framework"}
    auth = session_context_auth_header()
    if auth:
        if ":" in auth:
            hk, hv = auth.split(":", 1)
            headers[hk.strip()] = hv.strip()
        else:
            headers["Authorization"] = auth

    if have_cache and not force_refresh:
        etag = meta.get("etag")
        last_modified = meta.get("last_modified")
        if etag:
            headers["If-None-Match"] = etag
        if last_modified:
            headers["If-Modified-Since"] = last_modified

    to = timeout if timeout is not None else session_context_fetch_timeout()
    
    with httpx.Client(timeout=to) as client:
        try:
            response = client.get(s, headers=headers, follow_redirects=True)
            if response.status_code == 304 and have_cache:
                logger.debug("session_context_fetch: 304 Not Modified %s", s)
                # update fetched attr
                meta["fetched_at"] = datetime.now(timezone.utc).isoformat()
                cache_meta.write_text(json.dumps(meta), encoding="utf-8")
                return cache_md

            response.raise_for_status()
            logger.debug("session_context_fetch: 200 OK %s", s)
            
            # Write body
            cache_md.write_bytes(response.content)
            
            # Write meta
            new_meta = {
                "source_ref": s,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "etag": response.headers.get("etag", ""),
                "last_modified": response.headers.get("last-modified", ""),
            }
            cache_meta.write_text(json.dumps(new_meta), encoding="utf-8")
            return cache_md

        except httpx.HTTPError as e:
            if have_cache:
                logger.warning(
                    "Network error fetching %s, falling back to stale cache. Error: %s",
                    s,
                    e,
                )
                return cache_md
            raise ValueError(f"Failed to fetch remote session context '{s}': {e}") from e

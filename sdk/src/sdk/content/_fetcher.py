"""Fetch-and-cache primitive for remote content (http(s) URLs).

Generalizes the fetch + conditional-GET + disk-cache pattern that lived in
``host/session_context_fetch.py`` so any pack, host, or MCP server can
retrieve and cache remote documents (session contexts, GitHub-raw agents,
skills, etc.) through one SDK primitive.

Design (see ``docs/`` and the plan in session memory):

- **Cache policy is caller-owned.** Callers pass a :class:`FetchPolicy` to
  :meth:`ContentFetcher.fetch`. The harness supplies a default and imposes
  NO TTL cap, NO cache on/off clamp. Only security bounds are
  non-bypassable.
- **Security bounds (always enforced, ignore policy):** scheme allow-list,
  optional host allow-list, private-IP blocklist, one absolute
  ``max_bytes`` DoS ceiling applied as
  ``min(policy.max_bytes, _ABSOLUTE_MAX_BYTES)``.
- **Two cache tiers:** L1 in-memory ``dict`` (mirror of
  :class:`sdk.content._reader.ContentReader._read_cache`) and L2
  :class:`diskcache.Cache` (SQLite-backed, survives restarts) rooted at
  ``data_dir / "files" / "_fetch"``.
- **Env stays in tool/host adapters.** This primitive takes explicit args;
  callers translate env → :class:`FetchPolicy` at the call boundary.
- **Async-first.** :meth:`ContentFetcher.fetch` is ``async``; sync callers
  bridge via ``asyncio.run`` at the narrowest boundary.
"""

from __future__ import annotations

import hashlib
import ipaddress
import logging
import urllib.parse
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel

from sdk.layout import CACHE_DIR

try:
    from diskcache import Cache as _DiskCache
except ImportError:  # pragma: no cover - diskcache is a hard dep of lisa-sdk
    _DiskCache = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

__all__ = [
    "ContentFetcher",
    "FetchPolicy",
    "FetchedResult",
    "validate_url",
]

# ── Security constants ──────────────────────────────────────────────────────

_ABSOLUTE_MAX_BYTES: int = 1_000_000
_ALLOWED_SCHEMES: tuple[str, ...] = ("http", "https")
_USER_AGENT: str = "lisa_framework"

# RFC1918 + loopback + link-local + unique-local IPv6. A host on a LAN that
# legitimately fetches internal docs can relax this via
# ``private_ip_blocks=()`` on the ``ContentFetcher`` ctor.
_DEFAULT_PRIVATE_IP_BLOCKS: tuple[str, ...] = (
    "10.0.0.0/8",
    "172.16.0.0/12",
    "192.168.0.0/16",
    "127.0.0.0/8",
    "169.254.0.0/16",
    "0.0.0.0/8",
    "::1/128",
    "fc00::/7",
    "fe80::/10",
)

_L2_CACHE_SUBDIR = Path(CACHE_DIR) / "fetch"


# ── Policy + result models ──────────────────────────────────────────────────


@dataclass(frozen=True)
class FetchPolicy:
    """Caller-owned cache + fetch policy.

    The harness supplies a default instance; callers override per-call by
    passing a custom ``FetchPolicy`` to :meth:`ContentFetcher.fetch`. The
    harness imposes no cap on any field here — only the absolute
    ``max_bytes`` DoS ceiling (``_ABSOLUTE_MAX_BYTES``) is applied as a
    clamp, and only because it is a security bound, not a freshness rule.
    """

    ttl_seconds: float = 86_400.0
    force_refresh: bool = False
    max_bytes: int = 200_000
    cache: bool = True
    revalidate_etag: bool = False
    timeout_seconds: float = 15.0
    follow_redirects: bool = True
    auth_header: str | None = None


class FetchedResult(BaseModel):
    """One fetched document returned by :meth:`ContentFetcher.fetch`."""

    url: str
    status_code: int
    body: bytes
    text: str
    content_type: str = ""
    etag: str = ""
    last_modified: str = ""
    fetched_at: str
    from_cache: bool = False


# ── URL validation (shared, security gate) ──────────────────────────────────


def _is_private_ip(host: str, blocks: tuple[str, ...]) -> bool:
    """Return True if ``host`` resolves to a private/loopback IP."""
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return False
    for block in blocks:
        try:
            network = ipaddress.ip_network(block, strict=False)
        except ValueError:
            continue
        if addr in network:
            return True
    return False


def validate_url(
    url: str,
    *,
    allow_http: bool = False,
    allowed_hosts: set[str] | None = None,
    private_ip_blocks: tuple[str, ...] = _DEFAULT_PRIVATE_IP_BLOCKS,
) -> tuple[str | None, str | None]:
    """Validate a URL against the security gate.

    Returns ``(normalized_url | None, error_reason | None)``. Exactly one is
    non-None. This is the single source of truth for SSRF rejection — every
    caller (fetch_page tool, session_context adapter, MCP fetch tool) routes
    through here instead of reimplementing scheme/host checks.
    """
    cleaned = (url or "").strip()
    if not cleaned:
        return None, "URL is required."
    parsed = urllib.parse.urlparse(cleaned)
    scheme = (parsed.scheme or "").lower()
    if scheme not in _ALLOWED_SCHEMES:
        return None, f"URL must be http or https, got '{cleaned}'."
    if scheme == "http" and not allow_http:
        return None, (f"Blocked HTTP URL '{cleaned}'. HTTPS is required by default.")
    host = (parsed.hostname or "").lower()
    if not host:
        return None, f"URL has no host: {cleaned}"
    if allowed_hosts is not None and host not in allowed_hosts:
        allowed_str = ", ".join(sorted(allowed_hosts)) or "(none)"
        return None, (f"Blocked URL from host '{host}'. Allowed hosts are: {allowed_str}.")
    if _is_private_ip(host, private_ip_blocks):
        return None, (f"Blocked URL to private/loopback host '{host}' (SSRF guard).")
    return cleaned, None


# ── Fetcher ─────────────────────────────────────────────────────────────────


def _cache_key(url: str) -> str:
    """Return a stable cache key for a URL (sha256, 16 hex chars)."""
    return hashlib.sha256(url.strip().encode("utf-8")).hexdigest()[:16]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _decode_body(body: bytes, encoding: str | None) -> str:
    return body.decode(encoding or "utf-8", errors="replace")


def _result_to_dict(result: FetchedResult) -> dict[str, Any]:
    return {
        "url": result.url,
        "status_code": result.status_code,
        "body": result.body,
        "text": result.text,
        "content_type": result.content_type,
        "etag": result.etag,
        "last_modified": result.last_modified,
        "fetched_at": result.fetched_at,
    }


def _dict_to_result(data: dict[str, Any], *, from_cache: bool) -> FetchedResult:
    return FetchedResult(
        url=data["url"],
        status_code=data["status_code"],
        body=data["body"],
        text=data["text"],
        content_type=data.get("content_type", ""),
        etag=data.get("etag", ""),
        last_modified=data.get("last_modified", ""),
        fetched_at=data["fetched_at"],
        from_cache=from_cache,
    )


class ContentFetcher:
    """Async fetch-and-cache primitive for remote http(s) content.

    Args:
        data_dir: Canonical data root. The L2 disk cache lives under
            ``data_dir / "files" / "_fetch"``.
        default_policy: Harness default policy; per-call ``policy`` overrides
            it. The harness imposes no cap on these values.
        allowed_hosts: If set, only these hosts (lowercased) are permitted.
            ``None`` means any public host (private IPs are still blocked).
        allow_http: Permit ``http://`` URLs. Default ``False`` (HTTPS only).
        private_ip_blocks: CIDR blocks blocked by the SSRF guard. Pass ``()``
            to disable the private-IP check (e.g. a host on a LAN fetching
            internal docs).
    """

    def __init__(
        self,
        data_dir: Path,
        *,
        default_policy: FetchPolicy = FetchPolicy(),
        allowed_hosts: set[str] | None = None,
        allow_http: bool = False,
        private_ip_blocks: tuple[str, ...] = _DEFAULT_PRIVATE_IP_BLOCKS,
    ) -> None:
        self._data_dir: Path = Path(data_dir)
        self._default_policy: FetchPolicy = default_policy
        self._allowed_hosts: set[str] | None = (
            {h.lower() for h in allowed_hosts} if allowed_hosts is not None else None
        )
        self._allow_http: bool = allow_http
        self._private_ip_blocks: tuple[str, ...] = private_ip_blocks
        # L1: in-memory cache keyed by URL.
        self._l1: dict[str, FetchedResult] = {}
        # L2: diskcache (SQLite-backed) rooted under data_dir/files/_fetch.
        self._l2: Any = None  # _DiskCache | None

    # ── L2 lazy init ────────────────────────────────────────────────────────

    def _l2_cache(self) -> Any:
        if self._l2 is None and _DiskCache is not None:
            cache_dir = self._data_dir / _L2_CACHE_SUBDIR
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._l2 = _DiskCache(str(cache_dir))
        return self._l2

    # ── Public API ──────────────────────────────────────────────────────────

    async def fetch(
        self,
        url: str,
        *,
        policy: FetchPolicy | None = None,
    ) -> FetchedResult:
        """Fetch one URL with L1 → L2 → conditional-GET → network resolution.

        Security gate runs first and cannot be bypassed by ``policy``.
        On a network error with a valid stale cache, returns the stale
        cache (logged). Raises ``ValueError`` for SSRF rejection or for a
        network failure with no cache.
        """
        effective = policy if policy is not None else self._default_policy

        cleaned, reason = validate_url(
            url,
            allow_http=self._allow_http,
            allowed_hosts=self._allowed_hosts,
            private_ip_blocks=self._private_ip_blocks,
        )
        if reason is not None or cleaned is None:
            raise ValueError(reason or "Invalid URL.")

        max_bytes = min(effective.max_bytes, _ABSOLUTE_MAX_BYTES)

        if effective.cache and not effective.force_refresh:
            cached = self._lookup(cleaned, effective.ttl_seconds)
            if cached is not None:
                return cached

        result = await self._fetch_from_network(cleaned, policy=effective, max_bytes=max_bytes)

        if effective.cache:
            self._store(result)
        return result

    def invalidate(self, url: str | None = None) -> None:
        """Clear the L1 cache and (optionally) one L2 entry.

        Args:
            url: If given, invalidate only that URL's entries. If ``None``,
                clear the entire L1 cache and the whole L2 disk cache.
        """
        if url is None:
            self._l1.clear()
            l2 = self._l2_cache()
            if l2 is not None:
                l2.clear()
            return
        self._l1.pop(url, None)
        l2 = self._l2_cache()
        if l2 is not None:
            l2.delete(_cache_key(url))

    # ── Cache lookups ───────────────────────────────────────────────────────

    def _lookup(self, url: str, ttl_seconds: float) -> FetchedResult | None:
        cached = self._l1.get(url)
        if cached is None:
            l2 = self._l2_cache()
            if l2 is not None:
                data = l2.get(_cache_key(url), default=None)
                if data is not None:
                    cached = _dict_to_result(data, from_cache=True)
                    self._l1[url] = cached
        if cached is None:
            return None
        if self._is_expired(cached, ttl_seconds):
            return None
        # Mark as cache-served on the way out (L1 hits are already flagged).
        if not cached.from_cache:
            cached = cached.model_copy(update={"from_cache": True})
            self._l1[url] = cached
        return cached

    @staticmethod
    def _is_expired(result: FetchedResult, ttl_seconds: float) -> bool:
        if ttl_seconds <= 0:
            return False  # no expiry
        try:
            fetched = datetime.fromisoformat(result.fetched_at)
        except ValueError:
            return True
        age = (datetime.now(UTC) - fetched).total_seconds()
        return age > ttl_seconds

    def _store(self, result: FetchedResult) -> None:
        # Store a from_cache=False copy in L1 so future hits re-flag True.
        fresh = result if not result.from_cache else result.model_copy(update={"from_cache": False})
        self._l1[result.url] = fresh
        l2 = self._l2_cache()
        if l2 is not None:
            l2.set(_cache_key(result.url), _result_to_dict(fresh))

    # ── Network ─────────────────────────────────────────────────────────────

    async def _fetch_from_network(
        self,
        url: str,
        *,
        policy: FetchPolicy,
        max_bytes: int,
    ) -> FetchedResult:
        headers: dict[str, str] = {"User-Agent": _USER_AGENT}
        auth = policy.auth_header
        if auth:
            if ":" in auth:
                hk, hv = auth.split(":", 1)
                headers[hk.strip()] = hv.strip()
            else:
                headers["Authorization"] = auth

        # Conditional GET: attach validators from a stale cache if we have one.
        # force_refresh suppresses conditional headers too (matches the
        # original session_context_fetch semantics: force = full fresh download).
        if policy.revalidate_etag and not policy.force_refresh:
            stale = self._peek_raw(url)
            if stale is not None:
                if stale.etag:
                    headers["If-None-Match"] = stale.etag
                if stale.last_modified:
                    headers["If-Modified-Since"] = stale.last_modified

        try:
            async with httpx.AsyncClient(
                timeout=policy.timeout_seconds,
                follow_redirects=policy.follow_redirects,
            ) as client:
                response = await client.get(url, headers=headers)

                if response.status_code == 304 and policy.revalidate_etag:
                    stale = self._peek_raw(url)
                    if stale is not None:
                        refreshed = stale.model_copy(
                            update={"fetched_at": _now_iso(), "from_cache": False}
                        )
                        if policy.cache:
                            self._store(refreshed)
                        return refreshed
                    # 304 with no cache — treat as a fresh empty-ish response.
                    return FetchedResult(
                        url=url,
                        status_code=304,
                        body=b"",
                        text="",
                        content_type="",
                        etag="",
                        last_modified="",
                        fetched_at=_now_iso(),
                    )

                response.raise_for_status()
                raw = response.content[:max_bytes]
                text = _decode_body(raw, response.encoding)
                result = FetchedResult(
                    url=url,
                    status_code=response.status_code,
                    body=raw,
                    text=text,
                    content_type=response.headers.get("content-type", ""),
                    etag=response.headers.get("etag", ""),
                    last_modified=response.headers.get("last-modified", ""),
                    fetched_at=_now_iso(),
                )
                return result

        except httpx.HTTPError as exc:
            stale = self._peek_raw(url)
            if stale is not None:
                logger.warning(
                    "Network error fetching %s, falling back to stale cache. Error: %s",
                    url,
                    exc,
                )
                return stale.model_copy(update={"from_cache": True})
            raise ValueError(f"Failed to fetch '{url}': {exc}") from exc

    def _peek_raw(self, url: str) -> FetchedResult | None:
        """Return a cached entry ignoring TTL (for stale-fallback / validators)."""
        cached = self._l1.get(url)
        if cached is not None:
            return cached
        l2 = self._l2_cache()
        if l2 is not None:
            data = l2.get(_cache_key(url), default=None)
            if data is not None:
                return _dict_to_result(data, from_cache=True)
        return None

    # ── Convenience ─────────────────────────────────────────────────────────

    def with_policy(self, **overrides: Any) -> ContentFetcher:
        """Return a new fetcher sharing cache state but a different default policy.

        Useful when a host wants one cache dir but per-pack default policies
        without constructing independent fetchers. The L1/L2 caches are shared
        (same ``self``); only the default policy differs.
        """
        new_policy = replace(self._default_policy, **overrides)
        clone = ContentFetcher(
            self._data_dir,
            default_policy=new_policy,
            allowed_hosts=self._allowed_hosts,
            allow_http=self._allow_http,
            private_ip_blocks=self._private_ip_blocks,
        )
        # Share cache state (L1 dict + L2 diskcache) with the original.
        clone._l1 = self._l1
        clone._l2 = self._l2
        return clone

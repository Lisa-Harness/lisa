"""Tests for the SDK ContentFetcher primitive (sdk.content._fetcher).

Covers the security gate (SSRF), L1/L2 cache hit/miss/expiry, force_refresh
bypass, conditional-GET 304 path, stale-fallback on network error, max_bytes
clamping by the absolute ceiling, cache=False bypass, the redirect-hop SSRF
guard, the shared cookie jar, and the status-honest error path.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from sdk.content import ContentFetcher, FetchPolicy, FetchStatusError, validate_url

# ── URL validation ──────────────────────────────────────────────────────────


def test_validate_url_accepts_https() -> None:
    cleaned, reason = validate_url("https://github.com/foo.md")
    assert cleaned == "https://github.com/foo.md"
    assert reason is None


def test_validate_url_rejects_empty() -> None:
    cleaned, reason = validate_url("")
    assert cleaned is None
    assert "required" in reason.lower()


def test_validate_url_rejects_non_http_scheme() -> None:
    cleaned, reason = validate_url("file:///etc/passwd")
    assert cleaned is None
    assert "http or https" in reason


def test_validate_url_rejects_http_by_default() -> None:
    cleaned, reason = validate_url("http://github.com/foo.md")
    assert cleaned is None
    assert "HTTPS is required" in reason


def test_validate_url_allows_http_when_enabled() -> None:
    cleaned, reason = validate_url("http://github.com/foo.md", allow_http=True)
    assert cleaned == "http://github.com/foo.md"
    assert reason is None


def test_validate_url_rejects_disallowed_host() -> None:
    cleaned, reason = validate_url(
        "https://evil.com/foo.md",
        allowed_hosts={"github.com", "raw.githubusercontent.com"},
    )
    assert cleaned is None
    assert "evil.com" in reason
    assert "github.com" in reason


def test_validate_url_rejects_private_ip() -> None:
    for host in ("127.0.0.1", "10.0.0.1", "192.168.1.1", "169.254.169.254"):
        cleaned, reason = validate_url(f"https://{host}/meta")
        assert cleaned is None, f"{host} should be blocked"
        assert "private" in reason.lower() or "loopback" in reason.lower()


def test_validate_url_rejects_no_host() -> None:
    cleaned, reason = validate_url("https:///foo.md")
    assert cleaned is None
    assert "no host" in reason


def test_validate_url_private_ip_check_can_be_disabled() -> None:
    cleaned, reason = validate_url(
        "https://10.0.0.1/internal",
        private_ip_blocks=(),
    )
    assert reason is None
    assert cleaned == "https://10.0.0.1/internal"


# ── Fetcher: successful fetch + cache ───────────────────────────────────────


def _mock_response(
    url: str,
    *,
    content: bytes = b"Hello World",
    status: int = 200,
    etag: str = "",
    last_modified: str = "",
) -> httpx.Response:
    return httpx.Response(
        status,
        request=httpx.Request("GET", url),
        content=content,
        headers={
            "content-type": "text/html",
            **({"etag": etag} if etag else {}),
            **({"last-modified": last_modified} if last_modified else {}),
        },
    )


@pytest.mark.asyncio
async def test_fetch_success_caches_and_returns_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fetcher = ContentFetcher(tmp_path)
    url = "https://github.com/foo.md"

    async def _fake_get(self: httpx.AsyncClient, req_url: str, **kw: Any) -> httpx.Response:
        assert req_url == url
        # The honest, self-identifying UA plus the navigation headers a
        # browser sends on a document request. These headers are what unblock
        # bot-defended sites (the UA string is not — see the benchmark in
        # ``_fetcher``'s ``_NAVIGATION_HEADERS`` comment).
        headers = kw["headers"]
        assert (
            headers.get("User-Agent")
            == "LisaHarness/0.1.1 (+https://github.com/Lisa-Framework/lisa)"
        )
        assert headers.get("Sec-Fetch-Mode") == "navigate"
        assert headers.get("Sec-Fetch-Dest") == "document"
        assert headers.get("Upgrade-Insecure-Requests") == "1"
        return _mock_response(
            req_url,
            content=b"Hello World",
            etag="W/123",
            last_modified="Wed, 21 Oct 2015 07:28:00 GMT",
        )

    monkeypatch.setattr(httpx.AsyncClient, "get", _fake_get)

    result = await fetcher.fetch(url)
    assert result.status_code == 200
    assert result.text == "Hello World"
    assert result.etag == "W/123"
    assert result.from_cache is False

    # Second call: L1 hit, no network.
    calls = {"n": 0}

    async def _counting_get(self: httpx.AsyncClient, req_url: str, **kw: Any) -> httpx.Response:
        calls["n"] += 1
        return _mock_response(req_url)

    monkeypatch.setattr(httpx.AsyncClient, "get", _counting_get)
    result2 = await fetcher.fetch(url)
    assert result2.from_cache is True
    assert calls["n"] == 0  # no network call


@pytest.mark.asyncio
async def test_fetch_force_refresh_bypasses_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fetcher = ContentFetcher(tmp_path)
    url = "https://github.com/foo.md"

    async def _fake_get(self: httpx.AsyncClient, req_url: str, **kw: Any) -> httpx.Response:
        return _mock_response(req_url, content=b"Fresh")

    monkeypatch.setattr(httpx.AsyncClient, "get", _fake_get)
    await fetcher.fetch(url)  # prime cache

    calls = {"n": 0}

    async def _counting_get(self: httpx.AsyncClient, req_url: str, **kw: Any) -> httpx.Response:
        calls["n"] += 1
        return _mock_response(req_url, content=b"Refreshed")

    monkeypatch.setattr(httpx.AsyncClient, "get", _counting_get)
    result = await fetcher.fetch(url, policy=FetchPolicy(force_refresh=True))
    assert calls["n"] == 1
    assert result.text == "Refreshed"
    assert result.from_cache is False


@pytest.mark.asyncio
async def test_fetch_expired_cache_refetches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fetcher = ContentFetcher(tmp_path)
    url = "https://github.com/foo.md"

    async def _fake_get(self: httpx.AsyncClient, req_url: str, **kw: Any) -> httpx.Response:
        return _mock_response(req_url, content=b"V1")

    monkeypatch.setattr(httpx.AsyncClient, "get", _fake_get)
    await fetcher.fetch(url)

    # Manually backdate the L1 entry so it's expired under a 1s TTL.
    stale = fetcher._l1[url].model_copy(
        update={"fetched_at": (datetime.now(UTC) - timedelta(seconds=10)).isoformat()}
    )
    fetcher._l1[url] = stale

    async def _fake_get_v2(self: httpx.AsyncClient, req_url: str, **kw: Any) -> httpx.Response:
        return _mock_response(req_url, content=b"V2")

    monkeypatch.setattr(httpx.AsyncClient, "get", _fake_get_v2)
    result = await fetcher.fetch(url, policy=FetchPolicy(ttl_seconds=1.0))
    assert result.text == "V2"
    assert result.from_cache is False


@pytest.mark.asyncio
async def test_fetch_304_returns_cached_when_revalidate_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # ttl_seconds=0 means "never expire" (L1 serves directly, no revalidation).
    # Use a tiny TTL + a backdated entry so the cache is "expired" and the
    # revalidate_etag conditional-GET path is actually exercised.
    fetcher = ContentFetcher(
        tmp_path, default_policy=FetchPolicy(revalidate_etag=True, ttl_seconds=1.0)
    )
    url = "https://github.com/foo.md"

    async def _fake_get(self: httpx.AsyncClient, req_url: str, **kw: Any) -> httpx.Response:
        return _mock_response(req_url, content=b"Original", etag="W/abc")

    monkeypatch.setattr(httpx.AsyncClient, "get", _fake_get)
    await fetcher.fetch(url)

    # Backdate L1 so the entry is expired under the 1s TTL.
    stale = fetcher._l1[url].model_copy(
        update={"fetched_at": (datetime.now(UTC) - timedelta(seconds=10)).isoformat()}
    )
    fetcher._l1[url] = stale

    async def _fake_304(self: httpx.AsyncClient, req_url: str, **kw: Any) -> httpx.Response:
        assert kw["headers"].get("If-None-Match") == "W/abc"
        return httpx.Response(304, request=httpx.Request("GET", req_url))

    monkeypatch.setattr(httpx.AsyncClient, "get", _fake_304)
    result = await fetcher.fetch(url, policy=FetchPolicy(revalidate_etag=True, ttl_seconds=1.0))
    assert result.text == "Original"
    assert result.from_cache is False  # refreshed via 304, not a stale L1 hit


@pytest.mark.asyncio
async def test_fetch_network_error_returns_stale_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fetcher = ContentFetcher(tmp_path)
    url = "https://github.com/foo.md"

    async def _fake_get(self: httpx.AsyncClient, req_url: str, **kw: Any) -> httpx.Response:
        return _mock_response(req_url, content=b"Stale Cache")

    monkeypatch.setattr(httpx.AsyncClient, "get", _fake_get)
    await fetcher.fetch(url)

    async def _raise(self: httpx.AsyncClient, req_url: str, **kw: Any) -> httpx.Response:
        raise httpx.ConnectError("Connection failed")

    monkeypatch.setattr(httpx.AsyncClient, "get", _raise)
    with caplog.at_level("WARNING"):
        result = await fetcher.fetch(url, policy=FetchPolicy(force_refresh=True))

    assert result.text == "Stale Cache"
    assert result.from_cache is True
    assert "stale cache" in caplog.text.lower()


@pytest.mark.asyncio
async def test_fetch_network_error_raises_when_no_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fetcher = ContentFetcher(tmp_path)
    url = "https://github.com/foo.md"

    async def _raise(self: httpx.AsyncClient, req_url: str, **kw: Any) -> httpx.Response:
        raise httpx.ConnectError("Connection failed")

    monkeypatch.setattr(httpx.AsyncClient, "get", _raise)
    with pytest.raises(ValueError, match="Failed to fetch"):
        await fetcher.fetch(url)


@pytest.mark.asyncio
async def test_fetch_cache_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fetcher = ContentFetcher(tmp_path)
    url = "https://github.com/foo.md"
    calls = {"n": 0}

    async def _fake_get(self: httpx.AsyncClient, req_url: str, **kw: Any) -> httpx.Response:
        calls["n"] += 1
        return _mock_response(req_url, content=f"Call {calls['n']}".encode())

    monkeypatch.setattr(httpx.AsyncClient, "get", _fake_get)
    r1 = await fetcher.fetch(url, policy=FetchPolicy(cache=False))
    r2 = await fetcher.fetch(url, policy=FetchPolicy(cache=False))
    assert calls["n"] == 2  # no caching — both hit the network
    assert r1.text == "Call 1"
    assert r2.text == "Call 2"
    assert r1.from_cache is False
    assert r2.from_cache is False


@pytest.mark.asyncio
async def test_fetch_max_bytes_clamped_by_absolute_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fetcher = ContentFetcher(tmp_path)
    url = "https://github.com/big.md"

    async def _fake_get(self: httpx.AsyncClient, req_url: str, **kw: Any) -> httpx.Response:
        return _mock_response(req_url, content=b"x" * 5_000_000)

    monkeypatch.setattr(httpx.AsyncClient, "get", _fake_get)
    # Request an absurd max_bytes; the primitive clamps to _ABSOLUTE_MAX_BYTES.
    result = await fetcher.fetch(url, policy=FetchPolicy(max_bytes=100_000_000, cache=False))
    # body must be <= the absolute ceiling (1_000_000)
    from sdk.content._fetcher import _ABSOLUTE_MAX_BYTES

    assert len(result.body) <= _ABSOLUTE_MAX_BYTES


@pytest.mark.asyncio
async def test_fetch_rejects_ssrf_before_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fetcher = ContentFetcher(tmp_path)
    called = {"n": 0}

    async def _fake_get(self: httpx.AsyncClient, req_url: str, **kw: Any) -> httpx.Response:
        called["n"] += 1
        return _mock_response(req_url)

    monkeypatch.setattr(httpx.AsyncClient, "get", _fake_get)
    with pytest.raises(ValueError):
        await fetcher.fetch("https://169.254.169.254/latest/meta-data")
    assert called["n"] == 0  # security gate must run before any network


# ── Invalidate ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_invalidate_clears_one_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fetcher = ContentFetcher(tmp_path)
    url = "https://github.com/foo.md"

    async def _fake_get(self: httpx.AsyncClient, req_url: str, **kw: Any) -> httpx.Response:
        return _mock_response(req_url, content=b"V1")

    monkeypatch.setattr(httpx.AsyncClient, "get", _fake_get)
    await fetcher.fetch(url)
    assert url in fetcher._l1

    fetcher.invalidate(url)
    assert url not in fetcher._l1


@pytest.mark.asyncio
async def test_invalidate_all_clears_everything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fetcher = ContentFetcher(tmp_path)

    async def _fake_get(self: httpx.AsyncClient, req_url: str, **kw: Any) -> httpx.Response:
        return _mock_response(req_url, content=b"X")

    monkeypatch.setattr(httpx.AsyncClient, "get", _fake_get)
    await fetcher.fetch("https://github.com/a.md")
    await fetcher.fetch("https://github.com/b.md")
    assert len(fetcher._l1) == 2

    fetcher.invalidate()
    assert len(fetcher._l1) == 0


# ── L2 disk cache persistence ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_l2_cache_survives_new_fetcher_instance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    url = "https://github.com/foo.md"

    async def _fake_get(self: httpx.AsyncClient, req_url: str, **kw: Any) -> httpx.Response:
        return _mock_response(req_url, content=b"Persisted")

    monkeypatch.setattr(httpx.AsyncClient, "get", _fake_get)
    f1 = ContentFetcher(tmp_path)
    await f1.fetch(url)

    # New fetcher instance, same data_dir → L2 hit, no network.
    calls = {"n": 0}

    async def _counting_get(self: httpx.AsyncClient, req_url: str, **kw: Any) -> httpx.Response:
        calls["n"] += 1
        return _mock_response(req_url)

    monkeypatch.setattr(httpx.AsyncClient, "get", _counting_get)
    f2 = ContentFetcher(tmp_path)
    result = await f2.fetch(url)
    assert result.text == "Persisted"
    assert result.from_cache is True
    assert calls["n"] == 0


# ── with_policy ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_with_policy_shares_cache_and_overrides_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = ContentFetcher(tmp_path, default_policy=FetchPolicy(cache=True))
    url = "https://github.com/foo.md"

    async def _fake_get(self: httpx.AsyncClient, req_url: str, **kw: Any) -> httpx.Response:
        return _mock_response(req_url, content=b"Shared")

    monkeypatch.setattr(httpx.AsyncClient, "get", _fake_get)
    await base.fetch(url)

    child = base.with_policy(cache=False)
    assert child._default_policy.cache is False
    # Same L1 dict — cache populated by the base fetch is visible to child.
    assert url in child._l1


# ── Navigation headers (bot-defense success rate) ───────────────────────────


@pytest.mark.asyncio
async def test_fetch_sends_navigation_headers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requests carry the navigation headers a browser sends on a document load.

    A 2026-10-20 benchmark over 17 research targets measured 76% success
    with a bare User-Agent versus 88% with these headers. The ablation
    showed the User-Agent string was irrelevant — the Sec-Fetch pair and
    Upgrade-Insecure-Requests were decisive — so Lisa identifies itself
    honestly rather than impersonating a browser.
    """
    fetcher = ContentFetcher(tmp_path)
    url = "https://example.com/"
    captured: dict[str, Any] = {}

    async def _fake_get(self: httpx.AsyncClient, req_url: str, **kw: Any) -> httpx.Response:
        captured.update(kw["headers"])
        return _mock_response(req_url, content=b"ok")

    monkeypatch.setattr(httpx.AsyncClient, "get", _fake_get)
    await fetcher.fetch(url)

    assert captured["User-Agent"] == "LisaHarness/0.1.1 (+https://github.com/Lisa-Framework/lisa)"
    assert captured["Sec-Fetch-Dest"] == "document"
    assert captured["Sec-Fetch-Mode"] == "navigate"
    assert captured["Sec-Fetch-Site"] == "none"
    assert captured["Upgrade-Insecure-Requests"] == "1"


@pytest.mark.asyncio
async def test_fetch_can_opt_out_of_navigation_headers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A caller fetching an API rather than a page can drop the navigation headers."""
    fetcher = ContentFetcher(tmp_path)
    url = "https://api.example.com/data"
    captured: dict[str, Any] = {}

    async def _fake_get(self: httpx.AsyncClient, req_url: str, **kw: Any) -> httpx.Response:
        captured.update(kw["headers"])
        return _mock_response(req_url, content=b"{}")

    monkeypatch.setattr(httpx.AsyncClient, "get", _fake_get)
    await fetcher.fetch(url, policy=FetchPolicy(browser_navigation_headers=False))

    assert "Sec-Fetch-Mode" not in captured
    assert captured["User-Agent"].startswith("LisaHarness/")


@pytest.mark.asyncio
async def test_fetch_extra_headers_override_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Caller-supplied headers are merged over the defaults."""
    fetcher = ContentFetcher(tmp_path)
    url = "https://api.example.com/data"
    captured: dict[str, Any] = {}

    async def _fake_get(self: httpx.AsyncClient, req_url: str, **kw: Any) -> httpx.Response:
        captured.update(kw["headers"])
        return _mock_response(req_url, content=b"{}")

    monkeypatch.setattr(httpx.AsyncClient, "get", _fake_get)
    await fetcher.fetch(
        url,
        policy=FetchPolicy(extra_headers={"Accept": "application/json"}),
    )

    assert captured["Accept"] == "application/json"


# ── Redirect-hop SSRF guard ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_redirect_to_private_host_is_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A public URL that redirects to a private address is refused.

    The entry URL passes the SSRF gate, but httpx follows the 302 to
    ``http://127.0.0.1/`` without consulting the guard again. The request
    hook re-validates every hop, so the internal address is never reached.
    """
    fetcher = ContentFetcher(tmp_path)

    async def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "http://127.0.0.1:9/secret"})
        return httpx.Response(200, content=b"INTERNAL SECRET")

    transport = httpx.MockTransport(_handler)
    original = httpx.AsyncClient

    def _patched(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return original(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _patched)
    with pytest.raises(ValueError, match="Blocked"):
        await fetcher.fetch("https://public.example/start", policy=FetchPolicy(cache=False))


@pytest.mark.asyncio
async def test_redirect_to_public_host_is_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The redirect guard does not block ordinary public redirects."""
    fetcher = ContentFetcher(tmp_path)

    async def _handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "https://other.example/final"})
        return httpx.Response(200, content=b"FINE")

    transport = httpx.MockTransport(_handler)
    original = httpx.AsyncClient

    def _patched(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return original(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _patched)
    result = await fetcher.fetch("https://public.example/start", policy=FetchPolicy(cache=False))
    assert result.text == "FINE"


# ── Cookie jar (challenge/cookie flows) ─────────────────────────────────────


@pytest.mark.asyncio
async def test_cookie_from_one_fetch_is_replayed_on_the_next(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A cookie set by one response is sent on the next fetch.

    Without a persistent jar each fetch starts with a fresh client, so a
    site that gates content behind a cookie can never be fetched.
    """
    fetcher = ContentFetcher(tmp_path)

    async def _handler(request: httpx.Request) -> httpx.Response:
        if request.headers.get("cookie") is None:
            return httpx.Response(
                200,
                headers={"set-cookie": "session=abc; Path=/"},
                content=b"CHALLENGE",
            )
        return httpx.Response(200, content=b"REAL CONTENT")

    transport = httpx.MockTransport(_handler)
    original = httpx.AsyncClient

    def _patched(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return original(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _patched)
    first = await fetcher.fetch("https://site.example/a", policy=FetchPolicy(cache=False))
    second = await fetcher.fetch("https://site.example/b", policy=FetchPolicy(cache=False))

    assert first.text == "CHALLENGE"
    assert second.text == "REAL CONTENT"


@pytest.mark.asyncio
async def test_cookie_persistence_can_be_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``persist_cookies=False`` keeps the jar empty between fetches."""
    fetcher = ContentFetcher(tmp_path)

    async def _handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"set-cookie": "session=abc; Path=/"},
            content=b"ok",
        )

    transport = httpx.MockTransport(_handler)
    original = httpx.AsyncClient

    def _patched(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs["transport"] = transport
        return original(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _patched)
    await fetcher.fetch(
        "https://site.example/a", policy=FetchPolicy(cache=False, persist_cookies=False)
    )

    assert dict(fetcher._cookies) == {}


# ── Status-honest errors ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_http_error_status_raises_instead_of_serving_stale_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A 403 raises rather than returning 24h-old content as a fresh success.

    ``raise_for_status`` raises ``HTTPStatusError`` (an ``httpx.HTTPError``),
    so the transport-error handler used to swallow a 403 and serve the
    stale cache — making a blocked fetch look like a 200. A status error is
    not a transport error and must surface with its status code.
    """
    fetcher = ContentFetcher(tmp_path)
    url = "https://site.example/p"

    async def _fake_get(self: httpx.AsyncClient, req_url: str, **kw: Any) -> httpx.Response:
        return _mock_response(req_url, content=b"Stale Content", etag="W/1")

    monkeypatch.setattr(httpx.AsyncClient, "get", _fake_get)
    await fetcher.fetch(url)

    async def _forbidden(self: httpx.AsyncClient, req_url: str, **kw: Any) -> httpx.Response:
        return httpx.Response(403, request=httpx.Request("GET", req_url), content=b"blocked")

    monkeypatch.setattr(httpx.AsyncClient, "get", _forbidden)
    with pytest.raises(FetchStatusError) as excinfo:
        await fetcher.fetch(url, policy=FetchPolicy(force_refresh=True))

    assert excinfo.value.status_code == 403


@pytest.mark.asyncio
async def test_transport_error_still_falls_back_to_stale_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A genuine network failure keeps the stale-cache fallback."""
    fetcher = ContentFetcher(tmp_path)
    url = "https://site.example/p"

    async def _fake_get(self: httpx.AsyncClient, req_url: str, **kw: Any) -> httpx.Response:
        return _mock_response(req_url, content=b"Stale Content")

    monkeypatch.setattr(httpx.AsyncClient, "get", _fake_get)
    await fetcher.fetch(url)

    async def _raise(self: httpx.AsyncClient, req_url: str, **kw: Any) -> httpx.Response:
        raise httpx.ConnectError("Connection failed")

    monkeypatch.setattr(httpx.AsyncClient, "get", _raise)
    result = await fetcher.fetch(url, policy=FetchPolicy(force_refresh=True))

    assert result.text == "Stale Content"
    assert result.from_cache is True

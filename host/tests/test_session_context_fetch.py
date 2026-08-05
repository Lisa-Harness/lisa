"""Tests for the session_context_fetch adapter.

The adapter now delegates to :class:`sdk.content.ContentFetcher` for SSRF
validation, L1/L2 caching, conditional-GET, and stale-fallback. These tests
verify the adapter's policy translation (env → FetchPolicy / ctor args) and
the legacy ``Path`` + meta.json return contract.

The env-getter tests (``test_allowed_hosts``, ``test_allow_http``,
``test_auth_header``) are unchanged — the getters still live here as policy
translators. The fetch tests mock ``httpx.AsyncClient.get`` (the primitive's
transport) instead of ``httpx.Client.get``.
"""

import json

import httpx
import pytest

from host.session_context_fetch import (
    _cache_meta_path,
    _cache_path_for,
    fetch_session_context_url,
    session_context_allow_http,
    session_context_allowed_hosts,
    session_context_auth_header,
)


def test_allowed_hosts(monkeypatch):
    monkeypatch.delenv("LISA_SESSION_CONTEXT_ALLOWED_HOSTS", raising=False)
    assert session_context_allowed_hosts() == {"github.com", "raw.githubusercontent.com"}

    monkeypatch.setenv("LISA_SESSION_CONTEXT_ALLOWED_HOSTS", "  foo.com , BAR.com")
    assert session_context_allowed_hosts() == {"foo.com", "bar.com"}


def test_allow_http(monkeypatch):
    monkeypatch.delenv("LISA_SESSION_CONTEXT_ALLOW_HTTP", raising=False)
    assert not session_context_allow_http()

    for v in ["1", "true", "yes", "on", "  TRUE  "]:
        monkeypatch.setenv("LISA_SESSION_CONTEXT_ALLOW_HTTP", v)
        assert session_context_allow_http()

    monkeypatch.setenv("LISA_SESSION_CONTEXT_ALLOW_HTTP", "false")
    assert not session_context_allow_http()


def test_auth_header(monkeypatch):
    monkeypatch.delenv("LISA_SESSION_CONTEXT_AUTH_HEADER", raising=False)
    monkeypatch.delenv("LISA_SESSION_CONTEXT_TOKEN", raising=False)
    assert session_context_auth_header() is None

    monkeypatch.setenv("LISA_SESSION_CONTEXT_TOKEN", "12345")
    assert session_context_auth_header() == "Bearer 12345"

    # header wins over token
    monkeypatch.setenv("LISA_SESSION_CONTEXT_AUTH_HEADER", "X-Custom: foo")
    assert session_context_auth_header() == "X-Custom: foo"


def test_fetch_rejects_http_by_default(tmp_path):
    with pytest.raises(ValueError, match="Blocked HTTP URL"):
        fetch_session_context_url("http://github.com/foo.md", data_dir=tmp_path)


def test_fetch_rejects_invalid_scheme(tmp_path):
    with pytest.raises(ValueError, match="must be http or https"):
        fetch_session_context_url("ftp://github.com/foo.md", data_dir=tmp_path)


def test_fetch_rejects_host(tmp_path):
    with pytest.raises(ValueError, match="Blocked URL from host 'evil.com'"):
        fetch_session_context_url("https://evil.com/foo.md", data_dir=tmp_path)


def test_fetch_success_caches_body_and_meta(tmp_path, monkeypatch):
    async def mock_get(self, url, **kwargs):
        assert url == "https://github.com/foo.md"
        assert kwargs["headers"].get("User-Agent") == "lisa_framework"
        return httpx.Response(
            200,
            request=httpx.Request("GET", url),
            content=b"Hello World",
            headers={"etag": "W/123", "last-modified": "Wed, 21 Oct 2015 07:28:00 GMT"},
        )

    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

    path = fetch_session_context_url("https://github.com/foo.md", data_dir=tmp_path)
    assert path.is_file()
    assert path.read_text(encoding="utf-8") == "Hello World"

    meta_path = _cache_meta_path("https://github.com/foo.md", tmp_path)
    assert meta_path.is_file()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["source_ref"] == "https://github.com/foo.md"
    assert meta["etag"] == "W/123"
    assert meta["last_modified"] == "Wed, 21 Oct 2015 07:28:00 GMT"
    assert "fetched_at" in meta


def test_fetch_200_overwrites_legacy_cache(tmp_path, monkeypatch):
    # The adapter constructs a new ContentFetcher per call, so its L1/L2 is
    # empty — no conditional headers are sent. A 200 overwrites the legacy
    # cache + meta. (The 304 path is exercised in sdk/tests/test_fetcher.py
    # where the primitive's L2 is primed directly.)
    url = "https://github.com/foo.md"
    cache_md = _cache_path_for(url, tmp_path)
    cache_md.parent.mkdir(parents=True, exist_ok=True)
    cache_md.write_text("Hello Cache", encoding="utf-8")

    meta_path = _cache_meta_path(url, tmp_path)
    meta_path.write_text(json.dumps({"etag": "W/123"}), encoding="utf-8")

    async def mock_get(self, req_url, **kwargs):
        assert "If-None-Match" not in kwargs["headers"]
        return httpx.Response(
            200,
            request=httpx.Request("GET", req_url),
            content=b"Fresh Hello",
            headers={"etag": "W/999"},
        )

    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

    path = fetch_session_context_url(url, data_dir=tmp_path)
    assert path.read_text(encoding="utf-8") == "Fresh Hello"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["etag"] == "W/999"


def test_fetch_force_refresh_skips_conditional_headers(tmp_path, monkeypatch):
    url = "https://github.com/foo.md"
    cache_md = _cache_path_for(url, tmp_path)
    cache_md.parent.mkdir(parents=True, exist_ok=True)
    cache_md.write_text("Hello Cache", encoding="utf-8")

    meta_path = _cache_meta_path(url, tmp_path)
    meta_path.write_text(json.dumps({"etag": "W/123"}), encoding="utf-8")

    async def mock_get(self, req_url, **kwargs):
        headers = kwargs.get("headers", {})
        # Should not have conditional headers (force_refresh suppresses them)
        assert "If-None-Match" not in headers
        return httpx.Response(200, request=httpx.Request("GET", req_url), content=b"Fresh Hello")

    monkeypatch.setattr(httpx.AsyncClient, "get", mock_get)

    path = fetch_session_context_url(url, data_dir=tmp_path, force_refresh=True)
    assert path.read_text(encoding="utf-8") == "Fresh Hello"


def test_fetch_network_error_raises_without_primitive_cache(tmp_path, monkeypatch):
    # The adapter delegates to the primitive, which has its own L1/L2 —
    # the legacy cache files are NOT consulted by the primitive. So a
    # network error with only legacy-cache files present still raises.
    # (Stale-fallback through the primitive's L2 is tested directly in
    # sdk/tests/test_fetcher.py::test_fetch_network_error_returns_stale_cache.)
    url = "https://github.com/foo.md"
    cache_md = _cache_path_for(url, tmp_path)
    cache_md.parent.mkdir(parents=True, exist_ok=True)
    cache_md.write_text("Stale Cache", encoding="utf-8")
    meta_path = _cache_meta_path(url, tmp_path)
    meta_path.write_text('{"etag": "W/123"}', encoding="utf-8")

    async def mock_err(self, req_url, **kwargs):
        raise httpx.ConnectError("Connection failed")

    monkeypatch.setattr(httpx.AsyncClient, "get", mock_err)

    with pytest.raises(ValueError, match="Failed to fetch"):
        fetch_session_context_url(url, data_dir=tmp_path)


def test_fetch_network_error_raises_if_no_cache(tmp_path, monkeypatch):
    url = "https://github.com/foo.md"

    async def mock_err(self, req_url, **kwargs):
        raise httpx.ConnectError("Connection failed")

    monkeypatch.setattr(httpx.AsyncClient, "get", mock_err)

    with pytest.raises(ValueError, match="Failed to fetch"):
        fetch_session_context_url(url, data_dir=tmp_path)

import json
import logging

import httpx
import pytest

from hosts.core.session_context_fetch import (
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
    with pytest.raises(ValueError, match="Blocked HTTP session context"):
        fetch_session_context_url("http://github.com/foo.md", data_dir=tmp_path)


def test_fetch_rejects_invalid_scheme(tmp_path):
    with pytest.raises(ValueError, match="must be http or https"):
        fetch_session_context_url("ftp://github.com/foo.md", data_dir=tmp_path)


def test_fetch_rejects_host(tmp_path):
    with pytest.raises(ValueError, match="Blocked session context from host 'evil.com'"):
        fetch_session_context_url("https://evil.com/foo.md", data_dir=tmp_path)


def test_fetch_success_caches_body_and_meta(tmp_path, monkeypatch):
    def mock_get(self, url, **kwargs):
        assert url == "https://github.com/foo.md"
        assert kwargs["headers"].get("User-Agent") == "lisa_framework"
        return httpx.Response(200, request=httpx.Request("GET", url), content=b"Hello World", headers={"etag": "W/123", "last-modified": "Wed, 21 Oct 2015 07:28:00 GMT"})

    monkeypatch.setattr(httpx.Client, "get", mock_get)

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


def test_fetch_304_falls_back_to_cache(tmp_path, monkeypatch):
    # Setup initial cache
    url = "https://github.com/foo.md"
    cache_md = _cache_path_for(url, tmp_path)
    cache_md.parent.mkdir(parents=True, exist_ok=True)
    cache_md.write_text("Hello Cache", encoding="utf-8")
    
    meta_path = _cache_meta_path(url, tmp_path)
    meta_path.write_text(json.dumps({
        "source_ref": url,
        "etag": "W/123",
        "last_modified": "Wed, 21 Oct 2015 07:28:00 GMT",
        "fetched_at": "old"
    }), encoding="utf-8")

    def mock_get(self, req_url, **kwargs):
        headers = kwargs.get("headers", {})
        assert headers.get("If-None-Match") == "W/123"
        assert headers.get("If-Modified-Since") == "Wed, 21 Oct 2015 07:28:00 GMT"
        return httpx.Response(304)

    monkeypatch.setattr(httpx.Client, "get", mock_get)

    path = fetch_session_context_url(url, data_dir=tmp_path)
    assert path == cache_md
    assert path.read_text(encoding="utf-8") == "Hello Cache"
    # fetched_at should be updated
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["fetched_at"] != "old"


def test_fetch_force_refresh_skips_conditional_headers(tmp_path, monkeypatch):
    url = "https://github.com/foo.md"
    cache_md = _cache_path_for(url, tmp_path)
    cache_md.parent.mkdir(parents=True, exist_ok=True)
    cache_md.write_text("Hello Cache", encoding="utf-8")
    
    meta_path = _cache_meta_path(url, tmp_path)
    meta_path.write_text(json.dumps({
        "etag": "W/123",
    }), encoding="utf-8")

    def mock_get(self, req_url, **kwargs):
        headers = kwargs.get("headers", {})
        # Should not have conditional headers
        assert "If-None-Match" not in headers
        return httpx.Response(200, request=httpx.Request("GET", req_url), content=b"Fresh Hello")

    monkeypatch.setattr(httpx.Client, "get", mock_get)

    path = fetch_session_context_url(url, data_dir=tmp_path, force_refresh=True)
    assert path.read_text(encoding="utf-8") == "Fresh Hello"


def test_fetch_network_error_returns_stale_cache(tmp_path, monkeypatch, caplog):
    url = "https://github.com/foo.md"
    cache_md = _cache_path_for(url, tmp_path)
    cache_md.parent.mkdir(parents=True, exist_ok=True)
    cache_md.write_text("Stale Cache", encoding="utf-8")
    meta_path = _cache_meta_path(url, tmp_path)
    meta_path.write_text('{"etag": "W/123"}', encoding="utf-8")

    def mock_err(self, req_url, **kwargs):
        raise httpx.ConnectError("Connection failed")

    monkeypatch.setattr(httpx.Client, "get", mock_err)

    with caplog.at_level(logging.WARNING):
        path = fetch_session_context_url(url, data_dir=tmp_path)
    
    assert path == cache_md
    assert path.read_text(encoding="utf-8") == "Stale Cache"
    assert "falling back to stale cache" in caplog.text


def test_fetch_network_error_raises_if_no_cache(tmp_path, monkeypatch):
    url = "https://github.com/foo.md"
    def mock_err(self, req_url, **kwargs):
        raise httpx.ConnectError("Connection failed")

    monkeypatch.setattr(httpx.Client, "get", mock_err)

    with pytest.raises(ValueError, match="Failed to fetch remote session context"):
        fetch_session_context_url(url, data_dir=tmp_path)

"""Tests for :mod:`host.model_discovery.redact`.

These are the regression guards for the credential-leak class of bugs.
The prior implementation leaked Google's API key verbatim into the UI
because ``str(httpx.HTTPStatusError)`` embeds the full request URL,
which contained ``?key=<plaintext>``.
"""

from __future__ import annotations

import httpx

from host.model_discovery.redact import (
    classify_status,
    redact_url,
    safe_http_error,
)

# ── redact_url ───────────────────────────────────────────────────────


class TestRedactUrl:
    def test_redacts_key_query_param(self) -> None:
        url = "https://generativelanguage.googleapis.com/v1beta/models?key=AIzaSyABCD123"
        redacted = redact_url(url)
        assert "AIzaSyABCD123" not in redacted
        assert "REDACTED" in redacted

    def test_redacts_api_key_query_param(self) -> None:
        url = "https://example.com/v1/models?api_key=secret123&limit=10"
        redacted = redact_url(url)
        assert "secret123" not in redacted
        assert "limit=10" in redacted  # non-secret preserved

    def test_preserves_non_key_params(self) -> None:
        url = "https://example.com/v1/models?api-version=2024-10-21&page=2"
        redacted = redact_url(url)
        assert "api-version=2024-10-21" in redacted
        assert "page=2" in redacted

    def test_url_without_query_unchanged(self) -> None:
        url = "https://example.com/v1/models"
        assert redact_url(url) == url

    def test_case_insensitive_key_detection(self) -> None:
        url = "https://example.com/?API_KEY=secret&KEY=secret2"
        redacted = redact_url(url)
        assert "secret" not in redacted
        assert "secret2" not in redacted

    def test_empty_query_unchanged(self) -> None:
        url = "https://example.com/path?"
        assert redact_url(url) == url


# ── safe_http_error ───────────────────────────────────────────────────


class TestSafeHttpError:
    def test_http_status_error_excludes_url(self) -> None:
        request = httpx.Request(
            "GET",
            "https://generativelanguage.googleapis.com/v1beta/models?key=AIzaSySECRET",
        )
        response = httpx.Response(401, request=request)
        exc = httpx.HTTPStatusError("401", request=request, response=response)
        msg = safe_http_error(exc, "google")
        assert "AIzaSySECRET" not in msg
        assert "google" in msg
        assert "401" in msg
        # URL must never appear
        assert "generativelanguage.googleapis.com" not in msg

    def test_http_status_error_includes_status_and_reason(self) -> None:
        request = httpx.Request("GET", "https://api.openai.com/v1/models")
        response = httpx.Response(403, request=request)
        exc = httpx.HTTPStatusError("403", request=request, response=response)
        msg = safe_http_error(exc, "openai")
        assert "403" in msg
        # reason_phrase may be empty in some httpx versions; status code is the stable check
        assert "openai" in msg

    def test_connect_error(self) -> None:
        exc = httpx.ConnectError("connection refused")
        msg = safe_http_error(exc, "azure_openai")
        assert "azure_openai" in msg
        assert "connection" in msg.lower()

    def test_timeout_error(self) -> None:
        exc = httpx.TimeoutException("timed out")
        msg = safe_http_error(exc, "ollama")
        assert "timed out" in msg.lower() or "timed out" in msg

    def test_generic_exception(self) -> None:
        exc = RuntimeError("something weird")
        msg = safe_http_error(exc, "anthropic")
        assert "anthropic" in msg
        assert "something weird" not in msg  # raw message never surfaces


# ── classify_status ──────────────────────────────────────────────────


class TestClassifyStatus:
    def test_connect_error_is_network(self) -> None:
        exc = httpx.ConnectError("refused")
        assert classify_status(exc) == "network"

    def test_timeout_is_network(self) -> None:
        exc = httpx.ReadTimeout("slow")
        assert classify_status(exc) == "network"

    def test_401_is_auth(self) -> None:
        request = httpx.Request("GET", "https://example.com")
        response = httpx.Response(401, request=request)
        exc = httpx.HTTPStatusError("401", request=request, response=response)
        assert classify_status(exc) == "auth"

    def test_403_is_auth(self) -> None:
        request = httpx.Request("GET", "https://example.com")
        response = httpx.Response(403, request=request)
        exc = httpx.HTTPStatusError("403", request=request, response=response)
        assert classify_status(exc) == "auth"

    def test_404_is_not_found(self) -> None:
        request = httpx.Request("GET", "https://example.com")
        response = httpx.Response(404, request=request)
        exc = httpx.HTTPStatusError("404", request=request, response=response)
        assert classify_status(exc) == "not_found"

    def test_429_is_rate_limited(self) -> None:
        request = httpx.Request("GET", "https://example.com")
        response = httpx.Response(429, request=request)
        exc = httpx.HTTPStatusError("429", request=request, response=response)
        assert classify_status(exc) == "rate_limited"

    def test_500_is_unknown(self) -> None:
        request = httpx.Request("GET", "https://example.com")
        response = httpx.Response(500, request=request)
        exc = httpx.HTTPStatusError("500", request=request, response=response)
        assert classify_status(exc) == "unknown"

    def test_value_error_is_schema(self) -> None:
        assert classify_status(ValueError("bad json")) == "schema"

    def test_key_error_is_schema(self) -> None:
        assert classify_status(KeyError("missing")) == "schema"

    def test_generic_exception_is_unknown(self) -> None:
        assert classify_status(RuntimeError("??")) == "unknown"

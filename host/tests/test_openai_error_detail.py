"""Tests for ``_extract_openai_error_detail``.

Mirrors ``_extract_anthropic_error_detail``: extracts a concise detail
string from an OpenAI SDK exception so a 400 (e.g. the ``gpt-5.2-codex``
temperature rejection) surfaces as one readable line instead of a 2 KB
traceback. The OpenAI error schema is::

    {"error": {"message": "...", "type": "...", "param": "...", "code": "..."}}
"""

from __future__ import annotations

import httpx

from host.provider_completions import _extract_openai_error_detail


def _make_exc_with_response(
    status_code: int = 400,
    body: dict | None = None,
    text: str = "",
) -> Exception:
    """Build a synthetic exception with a mocked ``.response`` attribute."""
    if body is not None:
        request = httpx.Request("POST", "https://example.com/v1/chat/completions")
        response = httpx.Response(status_code, json=body, request=request)
    else:
        request = httpx.Request("POST", "https://example.com/v1/chat/completions")
        response = httpx.Response(status_code, text=text, request=request)
    exc = Exception("Bad Request")  # noqa: TRY004
    exc.response = response  # type: ignore[attr-defined]
    return exc


class TestExtractOpenAIErrorDetail:
    def test_extracts_type_message_and_param(self) -> None:
        """A standard OpenAI 400 body yields ``type: message (param=...)``."""
        exc = _make_exc_with_response(
            body={
                "error": {
                    "message": "Unsupported parameter: 'temperature' is not supported.",
                    "type": "invalid_request_error",
                    "param": "temperature",
                    "code": None,
                }
            },
        )
        detail = _extract_openai_error_detail(exc)
        assert "invalid_request_error" in detail
        assert "Unsupported parameter" in detail
        assert "param=temperature" in detail

    def test_extracts_code_when_present(self) -> None:
        """When the error body carries a code, it's in the detail suffix."""
        exc = _make_exc_with_response(
            body={
                "error": {
                    "message": "Rate limit exceeded",
                    "type": "rate_limit_error",
                    "param": None,
                    "code": "rate_limit_exceeded",
                }
            },
        )
        detail = _extract_openai_error_detail(exc)
        assert "rate_limit_error" in detail
        assert "code=rate_limit_exceeded" in detail

    def test_falls_back_to_str_exc_when_no_response(self) -> None:
        """When the exception has no ``.response``, ``str(exc)`` is returned."""
        exc = Exception("network error")
        detail = _extract_openai_error_detail(exc)
        assert detail == "network error"

    def test_falls_back_to_response_text_when_json_fails(self) -> None:
        """When the response body is not JSON, the text is returned."""
        exc = _make_exc_with_response(status_code=502, text="Bad Gateway")
        detail = _extract_openai_error_detail(exc)
        assert "Bad Gateway" in detail

    def test_falls_back_to_top_level_message_when_no_error_key(self) -> None:
        """When the body has no ``error`` key, a top-level ``message`` is used."""
        exc = _make_exc_with_response(
            body={"message": "internal server error"},
        )
        detail = _extract_openai_error_detail(exc)
        assert "internal server error" in detail

    def test_falls_back_to_str_exc_when_payload_is_not_dict(self) -> None:
        """When the JSON body is not a dict, ``str(exc)`` is returned."""
        request = httpx.Request("POST", "https://example.com/v1/chat/completions")
        response = httpx.Response(400, json=["not", "a", "dict"], request=request)
        exc = Exception("bad")
        exc.response = response  # type: ignore[attr-defined]
        detail = _extract_openai_error_detail(exc)
        assert "bad" in detail
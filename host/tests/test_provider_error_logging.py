"""Tests for structured provider-error logging (``log_completion_error``).

``log_completion_error`` emits a structured ``logger.error`` record so
provider failures (400s, timeouts, auth errors) are visible in the JSONL
turn log with enough context to diagnose — ``exc_type``, a concise detail
string, provider/model, API version, finish_reason, and a prompt-size
estimate. It must never raise (a logging failure must not break the
re-raise that the caller performs after this call).

``log_completion_response`` emits a ``logger.debug`` record for
successful completions so they are visible under debug-level logging.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from host.providers import log_completion_error, log_completion_response


def _make_model_def(
    provider: str = "azure",
    model_id: str = "gpt-5.2-codex",
    api_version: str = "2025-03-01-preview",
) -> Any:
    """Build a minimal StubModelDefinition-like object for logging tests."""
    return SimpleNamespace(
        provider=provider,
        model_id=model_id,
        api_version=api_version,
    )


def _make_bad_request_error(
    status_code: int = 400,
    body: dict[str, Any] | None = None,
) -> Exception:
    """Build an OpenAI-style BadRequestError with a mocked response."""
    if body is None:
        body = {
            "error": {
                "message": "Unsupported parameter: 'temperature' is not supported with this model.",
                "type": "invalid_request_error",
                "param": "temperature",
                "code": None,
            }
        }
    request = httpx.Request("POST", "https://example.com/v1/chat/completions")
    response = httpx.Response(status_code, json=body, request=request)
    exc = Exception("Bad Request")  # noqa: TRY004 — synthetic for testing
    exc.response = response  # type: ignore[attr-defined]
    return exc


class TestLogCompletionError:
    @pytest.mark.asyncio
    async def test_logs_structured_error_for_400(self, caplog: pytest.LogCaptureFixture) -> None:
        """A 400 BadRequestError produces a structured completion.error log record."""
        exc = _make_bad_request_error()
        model_def = _make_model_def()
        with caplog.at_level(logging.ERROR, logger="host.providers"):
            log_completion_error(
                exc=exc,
                model_def=model_def,
                response_format_name="text",
                elapsed_ms=42.5,
                messages=[{"role": "user", "content": "hello world"}],
                tools=None,
            )
        assert any("completion.error" in record.message for record in caplog.records)
        record = next(r for r in caplog.records if "completion.error" in r.message)
        assert "BadRequestError" in record.message or "Exception" in record.message
        assert "azure" in record.message
        assert "gpt-5.2-codex" in record.message
        assert "invalid_request_error" in record.message
        assert "temperature" in record.message

    @pytest.mark.asyncio
    async def test_never_raises_on_logging_failure(self, caplog: pytest.LogCaptureFixture) -> None:
        """A logging failure (e.g. bad model_def) is swallowed, not raised."""
        # Pass a model_def missing attributes — the logger should degrade gracefully.
        with caplog.at_level(logging.DEBUG, logger="host.providers"):
            log_completion_error(
                exc=Exception("boom"),
                model_def=SimpleNamespace(),  # type: ignore[arg-type]
                response_format_name="",
                elapsed_ms=0.0,
                messages=[],
                tools=None,
            )
        # No exception propagated — that's the contract.

    @pytest.mark.asyncio
    async def test_finish_reason_extracted_from_response_body(self, caplog: pytest.LogCaptureFixture) -> None:
        """When the error body carries a finish_reason, it's logged."""
        body = {
            "error": {"message": "rate limited", "type": "rate_limit_error"},
            "finish_reason": "length",
        }
        exc = _make_bad_request_error(status_code=429, body=body)
        model_def = _make_model_def()
        with caplog.at_level(logging.ERROR, logger="host.providers"):
            log_completion_error(
                exc=exc,
                model_def=model_def,
                response_format_name="text",
                elapsed_ms=100.0,
                messages=[{"role": "user", "content": "test"}],
                tools=[{"type": "function"}],
            )
        record = next(r for r in caplog.records if "completion.error" in r.message)
        assert "length" in record.message
        assert "tool_count=1" in record.message


class TestLogCompletionResponse:
    @pytest.mark.asyncio
    async def test_logs_debug_record_on_success(self, caplog: pytest.LogCaptureFixture) -> None:
        """A successful completion produces a debug-level completion.response record."""
        result = SimpleNamespace(
            finish_reason="stop",
            prompt_tokens=100,
            completion_tokens=50,
            cached_tokens=0,
            reasoning_tokens=0,
        )
        model_def = _make_model_def(provider="ollama", model_id="deepseek-v4-flash")
        with caplog.at_level(logging.DEBUG, logger="host.providers"):
            log_completion_response(
                model_def=model_def,
                response_format_name="text",
                elapsed_ms=250.0,
                result=result,  # type: ignore[arg-type]
            )
        record = next(r for r in caplog.records if "completion.response" in r.message)
        assert "ollama" in record.message
        assert "deepseek-v4-flash" in record.message
        assert "prompt_tokens=100" in record.message
        assert "completion_tokens=50" in record.message
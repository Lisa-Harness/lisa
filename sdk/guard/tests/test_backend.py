"""Tests for the guard SDK backend — passthrough validation, result
formatting, blocked-audit logging, and backend resolution.

Covers the same behaviors the lisa-guard MCP server exposes, but
exercised directly against the SDK backend without MCPServer.
"""

from __future__ import annotations

import json
import logging

import pytest

from kernel.seams.protocols import GuardResult
from sdk.guard import (
    GuardBackend,
    PassthroughGuardBackend,
    format_guard_result,
    get_backend,
    log_blocked_guard,
)


def test_passthrough_is_guard_backend() -> None:
    b = PassthroughGuardBackend()
    assert isinstance(b, GuardBackend)
    assert b.name == "passthrough"


def test_passthrough_validate_input_allows() -> None:
    b = PassthroughGuardBackend()
    result = b.validate_input("hello world", surface_id="s1")
    assert result.allowed is True
    assert result.blocked_reason is None
    assert result.sanitized_text is None
    assert result.annotations == {}


def test_passthrough_validate_output_allows() -> None:
    b = PassthroughGuardBackend()
    result = b.validate_output("safe output", surface_id="s1")
    assert result.allowed is True


def test_format_guard_result_shape() -> None:
    result = GuardResult(
        allowed=False,
        blocked_reason="toxic",
        sanitized_text="[redacted]",
        annotations={"score": 0.9},
    )
    data = json.loads(format_guard_result(result))
    assert data == {
        "allowed": False,
        "blocked_reason": "toxic",
        "sanitized_text": "[redacted]",
        "annotations": {"score": 0.9},
    }


def test_format_guard_result_allowed_shape() -> None:
    result = GuardResult(allowed=True, annotations={})
    data = json.loads(format_guard_result(result))
    assert data["allowed"] is True
    assert data["blocked_reason"] is None
    assert data["sanitized_text"] is None
    assert data["annotations"] == {}


def test_get_backend_passthrough() -> None:
    b = get_backend("passthrough")
    assert isinstance(b, PassthroughGuardBackend)


def test_get_backend_unknown_falls_back_to_passthrough() -> None:
    b = get_backend("azure_content_safety")
    assert isinstance(b, PassthroughGuardBackend)


def test_get_backend_empty_name_defaults_passthrough() -> None:
    b = get_backend("")
    assert isinstance(b, PassthroughGuardBackend)


def test_log_blocked_guard_emits_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger="sdk.guard._backend"):
        log_blocked_guard(
            direction="input",
            surface_id="s1",
            blocked_reason="toxic",
            text_len=42,
        )
    assert any("guard.blocked" in r.message for r in caplog.records)


def test_log_blocked_guard_respects_env_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LISA_GUARD_LOG_BLOCKED", "0")
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[assignment]
    log = logging.getLogger("sdk.guard._backend")
    log.addHandler(handler)
    try:
        log_blocked_guard(
            direction="output",
            surface_id="s2",
            blocked_reason="pii",
            text_len=10,
        )
    finally:
        log.removeHandler(handler)
    assert not any("guard.blocked" in r.message for r in records)

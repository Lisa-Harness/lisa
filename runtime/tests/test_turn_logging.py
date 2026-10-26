"""Tests for the record-assembly helpers in ``runtime.turn_logging``."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest

from runtime import turn_logging


def _ctx(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "log_level": "info",
        "debug": False,
        "timezone": "UTC",
        "timezone_source": "default",
        "session_context": "work",
        "mode_id": "default",
        "stage_id": "-",
        "next_model_provider": "openai",
        "next_model_id": "gpt-x",
        "stream_enabled": True,
        "streaming_mode": "hybrid",
        "streaming_char_delay_ms": 5,
    }
    base.update(overrides)
    return base


def _coerce(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _preview(text: str, *, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


# ── build_runtime_log_metadata ───────────────────────────────────────


def test_build_runtime_log_metadata_full() -> None:
    metadata = turn_logging.build_runtime_log_metadata(_ctx())

    runtime = metadata["runtime"]
    assert runtime["log_level"] == "info"
    assert runtime["debug"] is False
    assert runtime["timezone"] == "UTC"
    assert runtime["session_context"] == "work"
    assert runtime["mode_id"] == "default"
    assert runtime["next_model"] == {"provider": "openai", "model_id": "gpt-x"}
    assert runtime["streaming"] == {"enabled": True, "mode": "hybrid", "char_delay_ms": 5}


def test_build_runtime_log_metadata_defaults() -> None:
    runtime = turn_logging.build_runtime_log_metadata({})["runtime"]

    assert runtime["log_level"] == "info"
    assert runtime["debug"] is False
    assert runtime["timezone"] == "UTC"
    assert runtime["timezone_source"] == "default"
    assert runtime["session_context"] == "none"  # falsy mapped to "none"
    assert runtime["mode_id"] == "default"
    assert runtime["stage_id"] == "-"
    assert runtime["next_model"] == {"provider": "-", "model_id": "-"}
    assert runtime["streaming"] == {"enabled": False, "mode": "none", "char_delay_ms": 0}


# ── build_log_record ─────────────────────────────────────────────────


def test_build_log_record_merges_payload_and_metadata() -> None:
    record = turn_logging.build_log_record(
        level="info",
        event="cmd.tokens",
        payload={"answer": 42},
        ctx=_ctx(),
        debug_payload={"raw": True},
    )

    assert record is not None
    assert record["event"] == "cmd.tokens"
    assert record["answer"] == 42
    assert "logged_at_utc" in record
    assert record["runtime"]["mode_id"] == "default"
    # log_level is not debug → the debug payload is dropped.
    assert "debug" not in record


def test_build_log_record_attaches_debug_only_at_debug_level() -> None:
    record = turn_logging.build_log_record(
        level="debug",
        event="evt",
        payload={},
        ctx=_ctx(log_level="debug"),
        debug_payload={"extra": "detail"},
    )

    assert record is not None
    assert record["debug"] == {"extra": "detail"}


def test_build_log_record_debug_level_empty_payload_omits_debug_key() -> None:
    record = turn_logging.build_log_record(
        level="debug",
        event="evt",
        payload={},
        ctx=_ctx(log_level="debug"),
        debug_payload=None,
    )

    assert record is not None
    assert "debug" not in record


# ── build_turn_log_record ────────────────────────────────────────────


def test_build_turn_log_record_full_context() -> None:
    context = {
        "mode_id": "default",
        "mode_kind": "ipsar",
        "query": "hello world",
        "handoff": "",
        "resolved_model": {
            "model_id": "gpt-x",
            "provider": "openai",
            "reasoning_model": True,
        },
        "completion": {
            "tool_call_count": 2,
            "iteration_count": 3,
            "prompt_tokens": 100,
            "completion_tokens": 40,
            "total_tokens": 0,  # recomputed as prompt + completion
        },
    }

    record = turn_logging.build_turn_log_record(
        context=context,
        response_text="answer",
        coerce_int=_coerce,
        preview_fn=_preview,
    )

    assert record["type"] == "little_lisa_turn"
    assert record["mode_model"] == "openai:gpt-x"
    assert record["mode_model_id"] == "gpt-x"
    assert record["mode_model_provider"] == "openai"
    assert record["mode_reasoning_model"] is True
    assert record["mode_total_tokens"] == 140
    assert record["response"] == {"text": "answer", "chars": 6}
    assert record["route"] == {
        "mode_id": "default",
        "mode_kind": "ipsar",
        "query_chars": 11,
        "handoff": "",
    }
    assert record["completion_summary"]["tool_call_count"] == 2
    assert record["completion_summary"]["iteration_count"] == 3
    assert record["completion_summary"]["tokens"] == {
        "prompt_tokens": 100,
        "completion_tokens": 40,
        "total_tokens": 140,
    }
    assert record["response_preview"] == "answer"


def test_build_turn_log_record_minimal_context() -> None:
    record = turn_logging.build_turn_log_record(
        context={},
        response_text="",
        coerce_int=_coerce,
        preview_fn=_preview,
    )

    assert record["mode_model"] == ""  # bare model_id; no provider prefix
    assert record["mode_model_provider"] == ""
    assert record["mode_reasoning_model"] is False
    assert record["mode_prompt_tokens"] == 0
    assert record["route"]["query_chars"] == 0
    assert record["response"]["chars"] == 0


def test_build_turn_log_record_preview_truncates() -> None:
    record = turn_logging.build_turn_log_record(
        context={"completion": {"total_tokens": 50}},
        response_text="y" * 400,
        coerce_int=_coerce,
        preview_fn=_preview,
    )

    assert record["mode_total_tokens"] == 50  # explicit positive total is kept
    assert len(record["response_preview"]) == 323  # limit + "..."
    assert record["response"]["chars"] == 400


# ── build_turn_completed_payload ─────────────────────────────────────


def test_build_turn_completed_payload_from_result_attrs() -> None:
    result = SimpleNamespace(
        handoff_to="code",
        tool_call_count=4,
        iteration_count=5,
        blocked=True,
        block_origin="policy",
        content_source="reasoning_tail",
    )

    payload = turn_logging.build_turn_completed_payload(
        mode_id="research",
        result=result,
        coerce_int=_coerce,
    )

    assert payload == {
        "mode_id": "research",
        "turn_index": 0,
        "task_id": "",
        "handoff": "code",
        "tool_call_count": 4,
        "iteration_count": 5,
        "blocked": True,
        "block_origin": "policy",
        "content_source": "reasoning_tail",
    }


def test_build_turn_completed_payload_missing_attrs_default() -> None:
    payload = turn_logging.build_turn_completed_payload(
        mode_id="m",
        result=object(),
        coerce_int=_coerce,
    )

    assert payload == {
        "mode_id": "m",
        "turn_index": 0,
        "task_id": "",
        "handoff": "",
        "tool_call_count": 0,
        "iteration_count": 0,
        "blocked": False,
        "block_origin": "",
        "content_source": "",
    }


# ── build_turn_log_debug_payload ─────────────────────────────────────


def test_build_turn_log_debug_payload() -> None:
    context = {"zebra": 1, "apple": 2, "turn_input": "q", "completion": {"x": 1}}
    state = {
        "turn_prompt_tokens": "7",
        "session_total_tokens": 900,
    }

    payload = turn_logging.build_turn_log_debug_payload(
        context=context,
        response_text="full",
        state=state,
        coerce_int=_coerce,
        thread_dir="/threads",
        state_snapshot_loaded=True,
    )

    assert payload["context_keys"] == ["apple", "completion", "turn_input", "zebra"]
    assert payload["turn_input"] == "q"
    assert payload["response_full_text"] == "full"
    assert payload["thread_dir"] == "/threads"
    assert payload["state_snapshot_loaded"] is True
    tokens = payload["session_tokens"]
    assert tokens["turn_prompt_tokens"] == 7
    assert tokens["session_total_tokens"] == 900
    assert tokens["turn_completion_tokens"] == 0  # missing key coerced to 0


# ── emit_log_record / write_log_record ───────────────────────────────


class _CaptureLogger:
    def __init__(self) -> None:
        self.calls: list[tuple[int, str, dict[str, Any]]] = []

    def log(self, level: int, message: str, *, extra: dict[str, Any]) -> None:
        self.calls.append((level, message, extra))


def test_emit_log_record_splits_fields() -> None:
    target = _CaptureLogger()
    record = {
        "level": "warn",
        "event": "cmd.turn",
        "type": "little_lisa_turn",
        "answer": 42,
        "runtime": {"mode_id": "default"},
    }

    turn_logging.emit_log_record(record, target=target)  # type: ignore[arg-type]

    level, message, extra = target.calls[0]
    assert level == logging.WARNING  # WARN normalises to WARNING
    assert message == "little_lisa_turn"  # record type wins as the message
    assert extra["event"] == "cmd.turn"
    assert "level" not in extra and "type" not in extra
    assert extra["runtime"] == {"mode_id": "default"}


def test_emit_log_record_event_only_message_and_unknown_level() -> None:
    target = _CaptureLogger()

    turn_logging.emit_log_record(
        {"level": "totally-bogus", "event": "evt", "payload": 1},
        target=target,  # type: ignore[arg-type]
    )

    level, message, extra = target.calls[0]
    assert level == logging.INFO  # unknown level falls back
    assert message == "evt"
    assert extra == {"payload": 1, "event": "evt"}


def test_write_log_record_ignores_log_path(monkeypatch: pytest.MonkeyPatch) -> None:
    from pathlib import Path

    captured: list[dict[str, Any]] = []

    def _capture(record: dict[str, Any], *, target: Any) -> None:
        captured.append(record)

    monkeypatch.setattr(turn_logging, "emit_log_record", _capture)
    turn_logging.write_log_record({"level": "info", "event": "x"}, Path("/anywhere.log"))

    assert captured == [{"level": "info", "event": "x"}]

"""Tests for the archive-backed scripted gateway (Spec 3 slice 2)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from kernel.provider_contracts import CompletionRequest, CompletionUsage, ProviderContinuation
from runtime.testing.archive import (
    ArchivedCall,
    ArchiveGatewayProvider,
    ArchiveOutcomeError,
)
from runtime.testing.scripted import ScriptError


def _request(messages: list[dict[str, Any]], **kwargs: Any) -> CompletionRequest:
    model = kwargs.pop("model_def", None) or SimpleNamespace(
        provider="ollama", model_id="qwen3:8b", display_name="Qwen3 8B"
    )
    return CompletionRequest(
        messages=messages, model_def=model, max_tokens=1024, temperature=0.0, **kwargs
    )


def _snapshot_store() -> dict[str, dict[str, Any]]:
    """Synthetic result-object store keyed by ref, matching the archive layout."""
    return {
        "r" + "a" * 64: {
            "messages": [{"role": "user", "content": "go"}],
            "tools": [{"type": "function", "function": {"name": "t"}}],
            "model_id": "qwen3:8b",
            "provider": "ollama",
        },
        "r" + "b" * 64: {
            "call_id": "call-1",
            "attempt": "primary",
            "iteration": 1,
            "outcome": "ok",
            "content": "first",
            "provider": "ollama",
            "model_id": "qwen3:8b",
            "provider_request_id": "req-1",
            "tool_calls": [
                {
                    "id": "call_recorded_1",
                    "type": "function",
                    "function": {"name": "t", "arguments": '{"x": 1}'},
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "reasoning_content": "thinking",
            "finish_reason": "tool_calls",
            "truncated": False,
            "duration_ms": 12.5,
        },
        "r" + "c" * 64: {
            "messages": [
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": "", "tool_calls": []},
                {"role": "tool", "tool_call_id": "call_recorded_1", "content": "ok"},
            ],
            "tools": [{"type": "function", "function": {"name": "t"}}],
            "model_id": "qwen3:8b",
            "provider": "ollama",
        },
        "r" + "d" * 64: {
            "call_id": "call-2",
            "attempt": "stream_tool_fallback",
            "iteration": 1,
            "outcome": "ok",
            "content": "",
            "provider": "ollama",
            "model_id": "qwen3:8b",
            "tool_calls": [
                {
                    "id": "call_recorded_2",
                    "type": "function",
                    "function": {"name": "t", "arguments": '{"x": 2}'},
                }
            ],
            "usage": None,
            "finish_reason": "tool_calls",
            "truncated": None,
            "continuation": {"provider": "ollama", "model_id": "qwen3:8b", "items": [{"k": "v"}]},
        },
        "r" + "e" * 64: {
            "messages": [{"role": "user", "content": "go"}],
            "tools": [],
            "model_id": "qwen3:8b",
            "provider": "ollama",
        },
        "r" + "f" * 64: {
            "call_id": "call-3",
            "attempt": "primary",
            "iteration": 2,
            "outcome": "ok",
            "content": "done",
            "provider": "ollama",
            "model_id": "qwen3:8b",
            "usage": {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
            },
            "finish_reason": "stop",
            "truncated": False,
        },
    }


def _turn_input() -> list[dict[str, Any]]:
    return [
        {
            "call_id": "call-1",
            "iteration": 1,
            "attempt": "primary",
            "ref": "r" + "a" * 64,
            "response_ref": "r" + "b" * 64,
        },
        {
            "call_id": "call-2",
            "iteration": 1,
            "attempt": "stream_tool_fallback",
            "ref": "r" + "c" * 64,
            "response_ref": "r" + "d" * 64,
        },
        {
            "call_id": "call-3",
            "iteration": 2,
            "attempt": "primary",
            "ref": "r" + "e" * 64,
            "response_ref": "r" + "f" * 64,
        },
    ]


def _load(store: dict[str, dict[str, Any]]) -> Any:
    def load(ref: str) -> dict[str, Any]:
        return store[ref]

    return load


def _provider(store: dict[str, dict[str, Any]] | None = None) -> ArchiveGatewayProvider:
    store = store if store is not None else _snapshot_store()
    calls = ArchivedCall.from_turn_input(_turn_input(), load=_load(store))
    return ArchiveGatewayProvider(calls=calls)


def test_ordered_consumption_across_three_calls_including_fallback_attempt() -> None:
    provider = _provider()
    messages = [{"role": "user", "content": "go"}]
    tools = [{"type": "function", "function": {"name": "t"}}]

    first = provider.answer(_request(messages, tools=tools))
    assert first.content == "first"
    assert provider.position == 1

    second = provider.answer(
        _request(
            [
                *messages,
                {"role": "assistant", "content": ""},
                {"role": "tool", "tool_call_id": "call_recorded_1", "content": "ok"},
            ],
            tools=tools,
        )
    )
    assert second.content == ""
    assert provider.position == 2

    third = provider.answer(_request(messages, tools=[]))
    assert third.content == "done"
    assert provider.position == 3


def test_tool_call_ids_preserved_verbatim() -> None:
    provider = _provider()
    result = provider.answer(
        _request(
            [{"role": "user", "content": "go"}],
            tools=[{"type": "function", "function": {"name": "t"}}],
        )
    )
    assert [call["id"] for call in result.tool_calls] == ["call_recorded_1"]
    assert result.tool_calls[0]["function"]["name"] == "t"


def test_result_fields_reproduce_snapshot_parity() -> None:
    provider = _provider()
    tools = [{"type": "function", "function": {"name": "t"}}]
    first = provider.answer(_request([{"role": "user", "content": "go"}], tools=tools))
    assert first.finish_reason == "tool_calls"
    assert first.usage == CompletionUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    assert first.reasoning_content == "thinking"
    assert first.truncated is False
    assert first.duration_ms == 12.5
    assert first.request_id == "req-1"

    second = provider.answer(
        _request(
            [
                {"role": "user", "content": "go"},
                {"role": "assistant", "content": "", "tool_calls": []},
                {"role": "tool", "tool_call_id": "call_recorded_1", "content": "ok"},
            ],
            tools=tools,
        )
    )
    assert second.usage is None
    assert second.continuation == ProviderContinuation(
        provider="ollama", model_id="qwen3:8b", items=[{"k": "v"}]
    )

    third = provider.answer(_request([{"role": "user", "content": "go"}], tools=[]))
    assert third.usage == CompletionUsage(
        prompt_tokens=None, completion_tokens=None, total_tokens=None
    )
    assert third.finish_reason == "stop"


def test_divergence_notes_collected_without_failing_the_call() -> None:
    provider = _provider()
    mismatched_model = SimpleNamespace(provider="ollama", model_id="other:1b", display_name="Other")
    result = provider.answer(
        _request(
            [{"role": "user", "content": "go"}, {"role": "user", "content": "extra"}],
            model_def=mismatched_model,
            tools=[{"type": "function", "function": {"name": "other_tool"}}],
        )
    )
    assert result.content == "first"
    fields = {note["field"] for note in provider.divergences}
    assert fields == {"model_id", "tools", "message_count"}
    model_note = next(note for note in provider.divergences if note["field"] == "model_id")
    assert model_note["recorded"] == "qwen3:8b" and model_note["replayed"] == "other:1b"


def test_archive_exhausted_raises_script_error() -> None:
    provider = _provider()
    for _ in range(3):
        provider.answer(_request([{"role": "user", "content": "go"}]))
    with pytest.raises(ScriptError, match="archive exhausted after 3 calls"):
        provider.answer(_request([{"role": "user", "content": "go"}]))


def test_non_ok_outcome_raises_archive_outcome_error() -> None:
    store = _snapshot_store()
    store["r" + "b" * 64] = {
        "call_id": "call-1",
        "attempt": "primary",
        "iteration": 1,
        "outcome": "wall_budget_expired",
        "error": {"class": "WallBudgetExpired", "message": "budget gone"},
    }
    provider = _provider(store)
    with pytest.raises(ArchiveOutcomeError) as exc_info:
        provider.answer(_request([{"role": "user", "content": "go"}]))
    assert exc_info.value.call_id == "call-1"
    assert exc_info.value.outcome == "wall_budget_expired"
    assert "budget gone" in str(exc_info.value)


def test_missing_response_ref_raises_script_error_naming_call() -> None:
    entries = [
        {
            "call_id": "call-gap",
            "iteration": 1,
            "attempt": "primary",
            "ref": "r" + "a" * 64,
            "response_ref": None,
        }
    ]
    with pytest.raises(ScriptError, match="call-gap"):
        ArchivedCall.from_turn_input(entries, load=_load(_snapshot_store()))


def test_unloadable_ref_raises_script_error_naming_call() -> None:
    entries = [
        {
            "call_id": "call-missing",
            "iteration": 1,
            "attempt": "primary",
            "ref": "r" + "z" * 64,
            "response_ref": "r" + "b" * 64,
        }
    ]
    with pytest.raises(ScriptError, match="call-missing"):
        ArchivedCall.from_turn_input(entries, load=_load(_snapshot_store()))


def test_gateway_for_returns_bound_gateway() -> None:
    provider = _provider()
    gateway = provider.gateway_for("ollama")
    assert gateway.owner is provider


@pytest.mark.asyncio
async def test_gateway_complete_answers_from_archive() -> None:
    provider = _provider()
    gateway = provider.gateway_for("ollama")
    result = await gateway.complete(
        _request(
            [{"role": "user", "content": "go"}],
            tools=[{"type": "function", "function": {"name": "t"}}],
        )
    )
    assert result.content == "first"

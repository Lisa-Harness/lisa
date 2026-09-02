"""Tests for streaming support (Part 2).

Covers ``StreamedChunk``/``StreamedCompletion`` types,
``_aggregate_streamed_chunks`` helper, the ``StreamedCompletionAdapter``
bridge, and the turn runner's streaming path (``TurnHooks.complete_stream``).

Gateway-specific streaming tests (``OpenAICompatibleGateway.complete_stream``,
``AnthropicGateway.complete_stream``) have moved to
``sdk/tests/test_gateway_streaming.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

from kernel.provider_contracts import (
    CompletionRequest,
    CompletionUsage,
    StreamedChunk,
    StreamedCompletion,
    _aggregate_streamed_chunks,
)
from kernel.turn_runner import StreamedCompletionAdapter, TurnHooks, run_turn

_CORE_ROOT: Path = (Path(__file__).resolve().parents[1] / ".." / "sdk" / "packs" / "core").resolve()

# The shared conftest fixture (_mcp_catalog_setup) handles all catalog wiring.


def _model_def(provider: str = "openai", model_id: str = "test-model") -> Any:
    return SimpleNamespace(
        provider=provider,
        model_id=model_id,
        display_name="Test Model",
        reasoning_model=False,
        thinking_model=False,
    )


def _request(messages: list[dict[str, Any]] | None = None) -> CompletionRequest:
    return CompletionRequest(
        messages=messages or [{"role": "user", "content": "hi"}],
        model_def=_model_def(),
        max_tokens=1024,
        temperature=0.0,
    )


# ── Type tests ──────────────────────────────────────────────────────────


def test_streamed_chunk_defaults() -> None:
    chunk = StreamedChunk()
    assert chunk.content == ""
    assert chunk.finish_reason is None
    assert chunk.usage is None


def test_streamed_completion_defaults() -> None:
    sc = StreamedCompletion()
    assert sc.content == ""
    assert sc.tool_calls == []
    assert sc.usage is None
    assert sc.finish_reason == "stop"
    assert sc.chunks == []


# ── Aggregation tests ────────────────────────────────────────────────────


def test_aggregate_basic_text() -> None:
    chunks = [
        StreamedChunk(content="Hello"),
        StreamedChunk(content=" "),
        StreamedChunk(content="world", finish_reason="stop"),
    ]
    result = _aggregate_streamed_chunks(chunks)
    assert result.content == "Hello world"
    assert result.finish_reason == "stop"
    assert result.tool_calls == []
    assert len(result.chunks) == 3


def test_aggregate_empty_chunks() -> None:
    result = _aggregate_streamed_chunks([])
    assert result.content == ""
    assert result.finish_reason == "stop"


def test_aggregate_usage_from_last_chunk() -> None:
    usage = CompletionUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    chunks = [
        StreamedChunk(content="a"),
        StreamedChunk(content="b", usage=usage),
    ]
    result = _aggregate_streamed_chunks(chunks)
    assert result.usage is not None
    assert result.usage.prompt_tokens == 10
    assert result.usage.completion_tokens == 20


def test_aggregate_finish_reason_last_wins() -> None:
    chunks = [
        StreamedChunk(content="a", finish_reason=None),
        StreamedChunk(content="b", finish_reason="length"),
        StreamedChunk(content="c", finish_reason="stop"),
    ]
    result = _aggregate_streamed_chunks(chunks)
    assert result.finish_reason == "stop"


# ── StreamedCompletionAdapter ───────────────────────────────────────────


def test_streamed_completion_adapter_fields() -> None:
    usage = CompletionUsage(prompt_tokens=5, completion_tokens=10, total_tokens=15)
    sc = StreamedCompletion(content="hello", usage=usage, finish_reason="stop")
    adapter = StreamedCompletionAdapter(sc)
    assert adapter.content == "hello"
    assert adapter.tool_calls == []
    assert adapter.usage is not None
    assert adapter.usage.prompt_tokens == 5


# ── Turn runner streaming path ───────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _StubCompletion:
    content: str
    tool_calls: list[dict[str, Any]] | None = None


def _make_streaming_complete_stream(chunks: list[StreamedChunk]):
    """Return an async generator that yields pre-defined chunks."""

    async def stream_fn(**kwargs: Any):
        for chunk in chunks:
            yield chunk

    return stream_fn


def _make_stub_complete(response: str):
    async def complete_fn(**kwargs: Any):
        return _StubCompletion(content=response)

    return complete_fn


def _build_minimal_hooks(complete_fn=None, stream_fn=None) -> TurnHooks:
    return TurnHooks(
        complete=complete_fn,
        complete_stream=stream_fn,
    )


@pytest.mark.asyncio
async def test_turn_runner_uses_stream_when_set() -> None:
    chunks = [
        StreamedChunk(content="Hello"),
        StreamedChunk(content=" world", finish_reason="stop"),
    ]
    stream_fn = _make_streaming_complete_stream(chunks)
    complete_fn = _make_stub_complete("should not be called")
    hooks = _build_minimal_hooks(complete_fn, stream_fn)
    result = await run_turn(mode_id="default", query="hi", hooks=hooks)
    assert "Hello world" in result.response_text


@pytest.mark.asyncio
async def test_turn_runner_falls_back_to_complete_when_stream_not_set() -> None:
    complete_fn = _make_stub_complete("non-streaming response")
    hooks = _build_minimal_hooks(complete_fn, stream_fn=None)
    result = await run_turn(mode_id="default", query="hi", hooks=hooks)
    assert "non-streaming response" in result.response_text


@pytest.mark.asyncio
async def test_turn_runner_stream_tool_calls_falls_back_to_complete() -> None:
    """When stream finish_reason='tool_calls', turn runner falls back to complete()."""

    stream_chunks = [StreamedChunk(content="", finish_reason="tool_calls")]
    stream_fn = _make_streaming_complete_stream(stream_chunks)

    async def complete_fn(**kwargs: Any):
        return _StubCompletion(content="final answer")

    hooks = _build_minimal_hooks(complete_fn, stream_fn)
    result = await run_turn(mode_id="default", query="hi", hooks=hooks)
    assert "final answer" in result.response_text


# ── Truncated-stream rescue (the 2026-09-02 write-announcement loop) ───


@pytest.mark.asyncio
async def test_turn_runner_rescues_truncated_stream_with_tools_offered() -> None:
    """A stream that stops at ``length`` while tools were offered is
    retried once with a doubled budget instead of being accepted as a
    narration-only final completion.

    Regression for the 2026-09-02 write-announcement loop: v1 streaming
    does not parse tool-call deltas, so when glm-5.2:cloud spent its
    4096-token budget narrating and starting a multi-KB ``write_file``
    payload, the stream ended ``finish_reason='length'`` with tool calls
    still pending. The kernel treated the narration as a natural
    completion, the announced write never dispatched, and the user had to
    re-ask "continue" every turn.
    """
    calls: list[int] = []

    def _stream_fn(chunks_by_call: list[list[StreamedChunk]]):
        call_idx = {"n": 0}

        async def stream_fn(**kwargs: Any):
            calls.append(kwargs.get("max_tokens"))
            chunks = chunks_by_call[call_idx["n"]]
            call_idx["n"] += 1
            for chunk in chunks:
                yield chunk

        return stream_fn

    chunks_by_call = [
        # First call: narration streams fine, then truncation mid payload.
        [StreamedChunk(content="Writing both files now."), StreamedChunk(finish_reason="length")],
        # Retry: the same narration plus a tool_calls finish — the payload
        # fits in the doubled budget.
        [StreamedChunk(content="Writing both files now."), StreamedChunk(finish_reason="tool_calls")],
    ]
    stream_fn = _stream_fn(chunks_by_call)

    async def complete_fn(**kwargs: Any):
        return _StubCompletion(content="wrote it")

    hooks = _build_minimal_hooks(complete_fn, stream_fn)
    result = await run_turn(mode_id="default", query="write the scripts", hooks=hooks)

    # The retry fired with a doubled budget…
    assert calls == [4096, 8192], f"expected one rescue retry at 2x, got {calls}"
    # …the retried stream ended tool_calls, so the fallback complete() ran
    # and produced the tool-call turn.
    assert "wrote it" in result.response_text


@pytest.mark.asyncio
async def test_turn_runner_truncated_stream_without_tools_is_not_retried() -> None:
    """``finish_reason='length'`` with NO tools offered is a natural
    completion, not a lost tool call — no rescue retry."""

    calls: list[int] = []

    async def stream_fn(**kwargs: Any):
        calls.append(kwargs.get("max_tokens"))
        yield StreamedChunk(content="A plain answer that ran long.")
        yield StreamedChunk(finish_reason="length")

    async def complete_fn(**kwargs: Any):
        raise AssertionError("complete() must not be called on the natural path")

    async def empty_tool_surface(_mode_config: Any, **_kwargs: Any):
        return ([], {"input_count": 0, "output_count": 0, "dropped_duplicate": 0,
                     "dropped_disallowed": 0, "dropped_invalid": 0})

    hooks = _build_minimal_hooks(complete_fn, stream_fn)
    # The default catalog loads ambient memory/yield tools; strip the tool
    # surface so this test exercises the genuinely tools-offered=False path.
    with patch("kernel.turn_runner.build_tool_surface", empty_tool_surface):
        result = await run_turn(mode_id="default", query="hello there", hooks=hooks)

    assert calls == [4096], "no retry should fire when tools were not offered"
    assert "plain answer" in result.response_text


@pytest.mark.asyncio
async def test_turn_runner_truncated_stream_retry_respects_context_cap() -> None:
    """The rescue retry budget is bounded by the model's context window."""

    calls: list[int] = []

    async def stream_fn(**kwargs: Any):
        calls.append(kwargs.get("max_tokens"))
        yield StreamedChunk(content="partial")
        yield StreamedChunk(finish_reason="length")

    async def complete_fn(**kwargs: Any):
        return _StubCompletion(content="wrote it")

    hooks = TurnHooks(
        complete=complete_fn,
        complete_stream=stream_fn,
        resolve_model=lambda _mid: SimpleNamespace(
            provider="test",
            model_id="small-model",
            display_name="Small Model",
            reasoning_model=False,
            thinking_model=False,
            context_window=6000,
        ),
    )
    # context_window 6000 caps the doubled budget at 6000 (not 8192).
    result = await run_turn(
        mode_id="default",
        query="write the scripts",
        hooks=hooks,
    )

    assert calls == [4096, 6000], f"retry budget must cap at context_window, got {calls}"
    assert "partial" in result.response_text

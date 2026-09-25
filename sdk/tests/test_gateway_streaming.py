"""Gateway streaming tests — OpenAI and Anthropic complete_stream.

Moved from ``kernel/tests/test_streaming.py`` when the gateways relocated to
``sdk/providers/``. The contract and turn-runner streaming tests remain in
``kernel/tests/test_streaming.py``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from kernel.provider_contracts import (
    CompletionRequest,
    _aggregate_streamed_chunks,
)
from sdk.providers.anthropic_gateway import AnthropicGateway
from sdk.providers.google_gateway import GoogleGeminiGateway
from sdk.providers.openai_gateway import OpenAICompatibleGateway


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


# ── OpenAI gateway streaming ────────────────────────────────────────────


class _StubStreamEvent:
    """Simulates one SSE event from OpenAI streaming."""

    def __init__(
        self, delta_content: str = "", finish_reason: str | None = None, usage_obj: Any = None
    ) -> None:
        self.choices = (
            [
                SimpleNamespace(
                    delta=SimpleNamespace(content=delta_content or None),
                    finish_reason=finish_reason,
                )
            ]
            if (delta_content or finish_reason)
            else []
        )
        self.usage = usage_obj


class _StubAsyncStream:
    """Async iterable that yields pre-defined events."""

    def __init__(self, events: list[_StubStreamEvent]) -> None:
        self._events = events
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._events):
            raise StopAsyncIteration
        event = self._events[self._index]
        self._index += 1
        return event


class _StubOpenAIChatCompletions:
    def __init__(self, events: list[_StubStreamEvent]) -> None:
        self._events = events
        self.captured_kwargs: dict[str, Any] = {}

    async def create(self, **kwargs: Any) -> _StubAsyncStream:
        self.captured_kwargs.update(kwargs)
        return _StubAsyncStream(self._events)


class _StubOpenAIClient:
    def __init__(self, events: list[_StubStreamEvent]) -> None:
        self.chat = SimpleNamespace(completions=_StubOpenAIChatCompletions(events))


@pytest.mark.asyncio
async def test_openai_gateway_stream_yields_chunks() -> None:
    events = [
        _StubStreamEvent(delta_content="Hello"),
        _StubStreamEvent(delta_content=" world"),
        _StubStreamEvent(finish_reason="stop"),
    ]
    client = _StubOpenAIClient(events)
    gateway = OpenAICompatibleGateway(client_factory=lambda: client)
    chunks = []
    async for chunk in gateway.complete_stream(_request()):
        chunks.append(chunk)
    assert len(chunks) == 3
    assert chunks[0].content == "Hello"
    assert chunks[1].content == " world"
    assert chunks[2].finish_reason == "stop"
    aggregated = _aggregate_streamed_chunks(chunks)
    assert aggregated.content == "Hello world"


@pytest.mark.asyncio
async def test_openai_gateway_stream_passes_stream_flag() -> None:
    events = [_StubStreamEvent(finish_reason="stop")]
    client = _StubOpenAIClient(events)
    gateway = OpenAICompatibleGateway(client_factory=lambda: client)
    async for _ in gateway.complete_stream(_request()):
        pass
    assert client.chat.completions.captured_kwargs.get("stream") is True


# ── Anthropic gateway streaming ─────────────────────────────────────────
#
# The gateway streams via raw httpx SSE (``client.stream("POST", url, ...)``)
# and parses ``data: {...}`` lines, so the stub serves pre-rendered SSE lines.


def _sse(event: dict[str, Any]) -> str:
    import json as _json

    return f"data: {_json.dumps(event)}"


class _StubAnthropicStreamResponse:
    """Async context manager yielding SSE lines like a raw httpx response."""

    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    async def __aenter__(self) -> _StubAnthropicStreamResponse:
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _StubAnthropicStreamClient:
    def __init__(self, lines: list[str], captured: dict[str, Any]) -> None:
        self._lines = lines
        self.captured = captured

    def stream(self, method: str, url: str, *, json: Any = None, headers: Any = None):
        self.captured["method"] = method
        self.captured["url"] = url
        self.captured["json"] = json
        self.captured["headers"] = headers
        return _StubAnthropicStreamResponse(self._lines)


@pytest.mark.asyncio
async def test_anthropic_gateway_stream_yields_text_deltas() -> None:
    lines = [
        _sse(
            {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "Hello"},
            }
        ),
        _sse(
            {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": " world"},
            }
        ),
        _sse(
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 10, "output_tokens": 20},
                },
            }
        ),
        _sse({"type": "message_stop"}),
    ]
    captured: dict[str, Any] = {}
    client = _StubAnthropicStreamClient(lines, captured)
    gateway = AnthropicGateway(
        client_factory=lambda: client, api_key="test-key", api_version="2023-06-01"
    )
    chunks = []
    async for chunk in gateway.complete_stream(_request()):
        chunks.append(chunk)
    # 2 text deltas + 1 message_stop (usage rides it; finish_reason from message_delta).
    assert len(chunks) == 3
    assert chunks[0].content == "Hello"
    assert chunks[1].content == " world"
    # Anthropic's native stop_reason passes through (tool_use → tool_calls).
    assert chunks[2].finish_reason == "end_turn"
    assert chunks[2].usage is not None
    assert chunks[2].usage.prompt_tokens == 10
    assert chunks[2].usage.completion_tokens == 20
    # Auth headers are injected per-request by the gateway.
    assert captured["headers"]["x-api-key"] == "test-key"
    assert captured["headers"]["anthropic-version"] == "2023-06-01"


@pytest.mark.asyncio
async def test_anthropic_gateway_stream_system_extracted() -> None:
    lines = [_sse({"type": "message_stop"})]
    captured: dict[str, Any] = {}
    client = _StubAnthropicStreamClient(lines, captured)
    gateway = AnthropicGateway(
        client_factory=lambda: client, api_key="test-key", api_version="2023-06-01"
    )
    async for _ in gateway.complete_stream(
        _request(
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hi"},
            ]
        )
    ):
        pass
    assert captured["json"].get("system") == "You are helpful."


# ── Google Gemini gateway streaming ────────────────────────────────────


class _StubGoogleStreamResponse:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines

    async def __aenter__(self) -> _StubGoogleStreamResponse:
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    async def aiter_lines(self):
        for line in self._lines:
            yield line


class _StubGoogleStreamClient:
    def __init__(self, lines: list[str], captured: dict[str, Any]) -> None:
        self._lines = lines
        self.captured = captured

    def stream(self, method: str, url: str, *, json: Any = None, headers: Any = None):
        self.captured.update(method=method, url=url, json=json, headers=headers)
        return _StubGoogleStreamResponse(self._lines)


@pytest.mark.asyncio
async def test_google_gateway_stream_yields_text_reasoning_and_usage() -> None:
    lines = [
        _sse({"candidates": [{"content": {"parts": [{"text": "Thinking", "thought": True}]}}]}),
        _sse({"candidates": [{"content": {"parts": [{"text": "Hello"}]}}]}),
        _sse(
            {
                "candidates": [{"finishReason": "STOP", "content": {"parts": []}}],
                "usageMetadata": {
                    "promptTokenCount": 10,
                    "candidatesTokenCount": 20,
                    "totalTokenCount": 30,
                    "thoughtsTokenCount": 4,
                },
            }
        ),
    ]
    captured: dict[str, Any] = {}
    gateway = GoogleGeminiGateway(
        api_key="test-key", client=_StubGoogleStreamClient(lines, captured)
    )
    chunks = [chunk async for chunk in gateway.complete_stream(_request())]
    assert [chunk.content for chunk in chunks] == ["", "Hello", ""]
    assert chunks[0].reasoning_content == "Thinking"
    assert chunks[2].finish_reason == "stop"
    assert chunks[2].usage is not None
    assert chunks[2].usage.reasoning_tokens == 4
    assert captured["url"].endswith(":streamGenerateContent?alt=sse")
    assert captured["headers"]["x-goog-api-key"] == "test-key"
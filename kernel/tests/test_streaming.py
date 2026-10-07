"""Tests for streaming support (Part 2).

Covers ``StreamedChunk``/``StreamedCompletion`` types,
``_aggregate_streamed_chunks`` helper, the ``StreamedCompletionAdapter``
bridge, and the turn runner's streaming path (``TurnHooks.complete_stream``).

Gateway-specific streaming tests (``OpenAICompatibleGateway.complete_stream``,
``AnthropicGateway.complete_stream``) have moved to
``sdk/tests/test_gateway_streaming.py``.
"""

from __future__ import annotations

import time
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
from kernel.turn_hooks import (
    ModelHooks,
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


def test_streamed_completion_adapter_keeps_reasoning_separate() -> None:
    """Reasoning-only stream surfaces EMPTY content + reasoning_content.

    The adapter must not fold reasoning into ``content`` — the kernel's
    ``reasoning_tail_excerpt`` display guard only bounds leaked thinking when
    ``content`` is empty (2026-10-04 non-streaming 119KB monologue incident).
    """
    sc = StreamedCompletion(
        content="",
        reasoning_content="thinking about it",
        finish_reason="length",
    )
    adapter = StreamedCompletionAdapter(sc)
    assert adapter.content == ""
    assert adapter.reasoning_content == "thinking about it"


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
        models=ModelHooks(
            complete=complete_fn,
            complete_stream=stream_fn,
        ),
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
        [
            StreamedChunk(content="Writing both files now."),
            StreamedChunk(finish_reason="tool_calls"),
        ],
    ]
    stream_fn = _stream_fn(chunks_by_call)

    async def complete_fn(**kwargs: Any):
        return _StubCompletion(content="wrote it")

    hooks = _build_minimal_hooks(complete_fn, stream_fn)
    # The query text must NOT trigger the routing-hint forced transition
    # (a "write"-intent gap arms a forced handoff that synthesizes a
    # transition dispatch and consumes a third model call the stubs do
    # not provide). This test pins the rescue contract, not routing —
    # suppress the hint through the documented kernel seam.
    with patch(
        "kernel.turn_runner._compute_capability_routing_hint",
        return_value="",
    ):
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
        return (
            [],
            {
                "input_count": 0,
                "output_count": 0,
                "dropped_duplicate": 0,
                "dropped_disallowed": 0,
                "dropped_invalid": 0,
            },
        )

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
        models=ModelHooks(
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
        ),
    )
    # context_window 6000 caps the doubled budget at 6000 (not 8192).
    # Suppress the routing hint (same seam as the rescue test above) so
    # the forced transition cannot consume a third stubbed stream call.
    with patch(
        "kernel.turn_runner._compute_capability_routing_hint",
        return_value="",
    ):
        result = await run_turn(
            mode_id="default",
            query="write the scripts",
            hooks=hooks,
        )

    assert calls == [4096, 6000], f"retry budget must cap at context_window, got {calls}"
    assert "partial" in result.response_text


# ── Truncated-completion rescue (non-streaming path) ──────────────────
# The 2026-09-19 narration-loop incident: `lisa once` defaults to
# streaming="none", so its turn-loop completions take the
# `_gateway.complete()` path. A `finish_reason='length'` truncation cut
# the response mid-tool-call and the narration half was accepted as the
# final answer. These tests pin the same rescue contract as the streaming
# tests above, but on the non-streaming path.


@dataclass(frozen=True, slots=True)
class _RescueStubCompletion:
    """Stub completion carrying the provider truncation signals.

    Mirrors ``CompletionResult``: ``finish_reason`` drives the rescue,
    ``tool_calls`` drives the dispatch path.
    """

    content: str
    tool_calls: list[dict[str, Any]] | None = None
    finish_reason: str | None = None


@pytest.mark.asyncio
async def test_turn_runner_rescues_truncated_completion_with_tools_offered() -> None:
    """A non-streaming completion that stops at ``length`` while tools
    were offered is retried once with a doubled budget instead of being
    accepted as a truncated narration-only final completion."""

    calls: list[int] = []
    completions = [
        # First call: narration, truncated mid-tool-call payload.
        _RescueStubCompletion(
            content="Let me write the config and the report script now.",
            finish_reason="length",
        ),
        # Retry (doubled budget): the payload fits — the model lands its
        # answer cleanly (same shape as the streaming twin: the retried
        # generation produces the full completion).
        _RescueStubCompletion(
            content="wrote it",
            finish_reason="stop",
        ),
    ]
    call_idx = {"n": 0}

    async def complete_fn(**kwargs: Any):
        calls.append(kwargs.get("max_tokens"))
        completion = completions[call_idx["n"]]
        call_idx["n"] += 1
        return completion

    hooks = _build_minimal_hooks(complete_fn, stream_fn=None)
    with patch(
        "kernel.turn_runner._compute_capability_routing_hint",
        return_value="",
    ):
        result = await run_turn(mode_id="default", query="write the scripts", hooks=hooks)

    # The retry fired with a doubled budget…
    assert calls == [4096, 8192], f"expected one rescue retry at 2x, got {calls}"
    # …and the retried completion's answer is surfaced, not the truncated
    # narration half.
    assert "wrote it" in result.response_text


@pytest.mark.asyncio
async def test_turn_runner_truncated_completion_stripped_tools_retries() -> None:
    """A completion truncated on a tools-stripped iteration still retries.

    Regression for the 2026-10-06 "unfinished turns" incident: turns ended
    blocked with partial reasoning-prose when the final no-tools completion
    (pending_yield = iteration cap, lifecycle _FinishAction, router terminal
    action, approval continuation) was truncated at the cap. The rescue's
    tools-offered predicate read the iteration-local effective_tools, which
    the loop strips on those iterations, so truncated stripped-tools finals
    never got the 2x retry and the turn aborted with a reasoning tail.

    The model had tools earlier in the turn; stripping them on one
    iteration is harness-internal (the loop's choice to force a summary).
    Retry on the doubled budget instead of surfacing the truncation abort.
    """

    calls: list[int] = []
    completions = [
        # First call (tools offered): the model makes no tool calls, runs
        # out of its budget narrating a summary, and truncates.
        _RescueStubCompletion(
            content="The work so far concludes with the change applied, but writing the summary…",
            finish_reason="length",
        ),
        # Retry (doubled budget): completes with a full final answer.
        _RescueStubCompletion(
            content="Conclusion: landed the change, ran the checks, and wrote the summary.",
            finish_reason="stop",
        ),
    ]
    call_idx = {"n": 0}

    async def complete_fn(**kwargs: Any):
        calls.append(kwargs.get("max_tokens"))
        completion = completions[call_idx["n"]]
        call_idx["n"] += 1
        return completion

    hooks = _build_minimal_hooks(complete_fn, stream_fn=None)
    with patch(
        "kernel.turn_runner._compute_capability_routing_hint",
        return_value="",
    ):
        result = await run_turn(mode_id="default", query="finish up the changes.", hooks=hooks)

    # The retry fired with a doubled budget — the tools-stripped final got
    # the same 2x rescue as a tools-offered call.
    assert calls == [4096, 8192], f"expected one rescue retry at 2x, got {calls}"
    # The turn completes with the retried completion's answer, not the
    # truncated "nothing was dispatched" abort message.
    assert "nothing was dispatched" not in result.response_text
    assert "Conclusion" in result.response_text


@pytest.mark.asyncio
async def test_turn_runner_truncated_completion_no_tools_no_retry() -> None:
    """A stripped-tools truncation is retried only when the turn began
    with a tool surface.

    The freeze-gated ``test_turn_runner_truncated_completion_without_tools_
    is_not_retried`` pins the no-tools-at-all contract: a plain chat turn
    that finishes at the cap is a natural completion, no retry. This case
    fills the other gap: a turn that *started* with tools (a
    ``memory.read`` catalog ambient tool) and then strips them for the
    final iteration because the cap fires must still retry.
    """

    calls: list[int] = []
    completions = [
        _RescueStubCompletion(content="partial", finish_reason="length"),
        _RescueStubCompletion(content="done", finish_reason="stop"),
    ]
    call_idx = {"n": 0}

    async def complete_fn(**kwargs: Any):
        calls.append(kwargs.get("max_tokens"))
        completion = completions[call_idx["n"]]
        call_idx["n"] += 1
        return completion

    hooks = _build_minimal_hooks(complete_fn, stream_fn=None)
    with patch(
        "kernel.turn_runner._compute_capability_routing_hint",
        return_value="",
    ):
        result = await run_turn(mode_id="default", query="summarize the findings", hooks=hooks)

    assert calls == [4096, 8192], f"expected one rescue retry at 2x, got {calls}"
    assert result.response_text

    """``finish_reason='length'`` with NO tools offered is a natural
    completion, not a lost tool call — no rescue retry on the
    non-streaming path either."""

    calls: list[int] = []

    async def complete_fn(**kwargs: Any):
        calls.append(kwargs.get("max_tokens"))
        return _RescueStubCompletion(
            content="A plain answer that ran long.",
            finish_reason="length",
        )

    async def empty_tool_surface(_mode_config: Any, **_kwargs: Any):
        return (
            [],
            {
                "input_count": 0,
                "output_count": 0,
                "dropped_duplicate": 0,
                "dropped_disallowed": 0,
                "dropped_invalid": 0,
            },
        )

    hooks = _build_minimal_hooks(complete_fn, stream_fn=None)
    # Strip the tool surface so this exercises the genuinely
    # tools-offered=False path (the ambient catalog loads memory/yield
    # tools otherwise — same note as the streaming twin above).
    with patch("kernel.turn_runner.build_tool_surface", empty_tool_surface):
        result = await run_turn(mode_id="default", query="hello there", hooks=hooks)

    assert calls == [4096], "no retry should fire when tools were not offered"
    assert "plain answer" in result.response_text


@pytest.mark.asyncio
async def test_turn_runner_truncated_completion_retry_respects_context_cap() -> None:
    """The non-streaming rescue retry budget is bounded by the model's
    context window."""

    calls: list[int] = []
    completions = [
        _RescueStubCompletion(content="partial", finish_reason="length"),
        _RescueStubCompletion(content="complete answer", finish_reason="stop"),
    ]
    call_idx = {"n": 0}

    async def complete_fn(**kwargs: Any):
        calls.append(kwargs.get("max_tokens"))
        completion = completions[call_idx["n"]]
        call_idx["n"] += 1
        return completion

    hooks = TurnHooks(
        models=ModelHooks(
            complete=complete_fn,
            resolve_model=lambda _mid: SimpleNamespace(
                provider="test",
                model_id="small-model",
                display_name="Small Model",
                reasoning_model=False,
                thinking_model=False,
                context_window=6000,
            ),
        ),
    )
    # context_window 6000 caps the doubled budget at 6000 (not 8192).
    with patch(
        "kernel.turn_runner._compute_capability_routing_hint",
        return_value="",
    ):
        result = await run_turn(
            mode_id="default",
            query="write the scripts",
            hooks=hooks,
        )

    assert calls == [4096, 6000], f"retry budget must cap at context_window, got {calls}"
    assert "complete answer" in result.response_text


# ── on_delta per-chunk forwarding (v1 streaming display) ──────────────


@pytest.mark.asyncio
async def test_turn_runner_forwards_chunks_to_on_delta() -> None:
    """Every streamed chunk's text/reasoning deltas reach ``on_delta``.

    The runtime's ``on_text_chunk`` transport display streams through this
    hook; without it the chunks were collected locally and only the
    aggregated completion reached the host — the 2026-10-04 "spinner with
    nothing streaming" CLI incident.
    """
    chunks = [
        StreamedChunk(content="Hel"),
        StreamedChunk(reasoning_content="thinking..."),
        StreamedChunk(content="lo world", finish_reason="stop"),
    ]
    stream_fn = _make_streaming_complete_stream(chunks)
    received: list[tuple[str, str]] = []

    async def on_delta(text: str, reasoning: str) -> None:
        received.append((text, reasoning))

    hooks = _build_minimal_hooks(_make_stub_complete("unused"), stream_fn)
    hooks.models.on_delta = on_delta
    result = await run_turn(mode_id="default", query="hi", hooks=hooks)

    assert "Hello world" in result.response_text
    assert received == [("Hel", ""), ("", "thinking..."), ("lo world", "")]


@pytest.mark.asyncio
async def test_turn_runner_skips_empty_chunks_in_on_delta() -> None:
    """Bare metadata chunks (finish_reason only) do not reach ``on_delta``."""
    chunks = [
        StreamedChunk(content="answer"),
        StreamedChunk(finish_reason="stop"),
    ]
    stream_fn = _make_streaming_complete_stream(chunks)
    received: list[tuple[str, str]] = []

    async def on_delta(text: str, reasoning: str) -> None:
        received.append((text, reasoning))

    hooks = _build_minimal_hooks(_make_stub_complete("unused"), stream_fn)
    hooks.models.on_delta = on_delta
    await run_turn(mode_id="default", query="hi", hooks=hooks)

    assert received == [("answer", "")]


@pytest.mark.asyncio
async def test_turn_runner_on_delta_absent_is_noop() -> None:
    """``on_delta=None`` (default) leaves the streaming path unchanged."""
    chunks = [StreamedChunk(content="plain stream")]
    stream_fn = _make_streaming_complete_stream(chunks)
    hooks = _build_minimal_hooks(_make_stub_complete("unused"), stream_fn)
    assert hooks.models.on_delta is None
    result = await run_turn(mode_id="default", query="hi", hooks=hooks)
    assert "plain stream" in result.response_text


# ── Stream idle watchdog (the 2026-10-06 mid-stream freeze) ───────────


@pytest.mark.asyncio
async def test_turn_runner_stalled_stream_aborts_after_idle_timeout() -> None:
    """A stream that goes silent mid-generation aborts after the idle
    timeout instead of hanging the turn forever.

    Regression for the 2026-10-06 glm-5.3:cloud marathon: the doubled
    truncation-rescue stream stalled mid-generation (socket byte counters
    frozen for 8+ minutes) and ``_collect_stream``'s bare ``async for``
    blocked indefinitely — no timeout existed between the POST and the
    next lifecycle event. The turn only recovered because the upstream
    happened to resume; a permanent upstream hang left the CLI dead.
    The stalled call raises ``StreamIdleTimeout``; the turn loop's
    provider-failure path turns that into a turn error the host renders.
    """
    import asyncio

    import kernel.turn_runner
    from kernel.turn_runner import _collect_stream

    chunks_yielded: list[StreamedChunk] = []

    def _stalled_stream_fn(idle_s: float):
        async def stream_fn(*_args: Any, **_kwargs: Any):
            # First chunk arrives promptly…
            chunk = StreamedChunk(content="thinking")
            chunks_yielded.append(chunk)
            yield chunk
            # …then the upstream freezes: nothing else ever arrives.
            await asyncio.sleep(idle_s * 10)

        return stream_fn

    request = _request()
    before = time.monotonic()
    with (
        patch(
            "kernel.turn_runner._kernel_settings",
            SimpleNamespace(stream_idle_timeout_s=0.05),
        ),
        pytest.raises(kernel.turn_runner.StreamIdleTimeout),
    ):
        await _collect_stream(_stalled_stream_fn(0.05), request, None)
    elapsed = time.monotonic() - before
    # The watchdog aborted the stalled stream within the timeout budget
    # (not the 0.5s the sleep would have taken).
    assert elapsed < 0.4, f"watchdog did not bound the stall: {elapsed:.2f}s"
    # The chunk that arrived before the stall was produced (the generator
    # ran at least once before freezing).
    assert [c.content for c in chunks_yielded] == ["thinking"]


@pytest.mark.asyncio
async def test_turn_runner_healthy_stream_unaffected_by_watchdog() -> None:
    """A stream that keeps producing within the idle window is consumed
    fully — the watchdog never fires on a live generation, even one
    slower than the timeout in aggregate."""
    import asyncio

    from kernel.turn_runner import _collect_stream

    def _slow_but_alive_stream_fn():
        async def stream_fn(*_args: Any, **_kwargs: Any):
            for _ in range(3):
                yield StreamedChunk(content="x")
                await asyncio.sleep(0.02)  # well under the idle timeout

        return stream_fn

    with patch(
        "kernel.turn_runner._kernel_settings",
        SimpleNamespace(stream_idle_timeout_s=0.5),
    ):
        chunks, _ = await _collect_stream(_slow_but_alive_stream_fn(), _request(), None)
    assert [c.content for c in chunks] == ["x", "x", "x"]


def test_stream_idle_timeout_setting_env_override() -> None:
    """``STREAM_IDLE_TIMEOUT_S`` tunes the watchdog; 0 disables it."""
    from kernel.config import Settings

    assert Settings.model_validate({"stream_idle_timeout_s": "120"}).stream_idle_timeout_s == 120
    assert Settings.model_validate({"stream_idle_timeout_s": "0"}).stream_idle_timeout_s == 0

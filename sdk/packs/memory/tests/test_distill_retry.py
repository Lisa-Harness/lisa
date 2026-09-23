"""Regression tests for the distillation retry loop in ``execute``.

Covers the 2026-07-18 incident where the local Ollama model returned HTTP
200 OK but empty content on the first ``response_format=DistilledLearning``
attempt, and the operation failed with "empty distillation output" — no
retry. The fix added a retry-on-failure loop that re-issues the call with a
JSON-only instruction before surfacing the failure. These tests lock in the
retry behavior for both the empty-output case and the JSON-truncation case.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from kernel.models import OperationExecutionContext, OperationResult
from kernel.seams.host_adapter import build_callable_host
from sdk.packs.memory.operations.memory.synthesize_learning import execute


def _completion(content: str) -> SimpleNamespace:
    return SimpleNamespace(content=content)


def _ctx(
    *,
    complete_fn: Any,
    raw_content: str,
    execute_tool_fn: Any | None = None,
) -> OperationExecutionContext:
    """Build a minimal OperationExecutionContext wired for distillation."""

    def _read_authored_text(_path: str) -> str:
        return "Distill the content into 1-3 durable facts as JSON."

    def _resolve_model(_model_id: str | None) -> SimpleNamespace:
        return SimpleNamespace(provider="ollama", model_id="llama-test")

    host = build_callable_host(
        complete_fn=complete_fn,
        resolve_model_fn=_resolve_model,
        read_authored_text_fn=_read_authored_text,
        execute_tool_fn=execute_tool_fn,
    )
    return OperationExecutionContext(
        query="remember the architectural rundown",
        response="",
        params={"content": raw_content},
        host=host,
        preferred_model_id="default",
    )


@pytest.mark.asyncio
async def test_execute_retries_then_succeeds_when_first_attempt_empty() -> None:
    """First completion returns empty content; retry returns valid JSON.

    The 2026-07-18 incident: Ollama ignored ``response_format`` and returned
    HTTP 200 OK with empty content. Without the retry the operation raised
    "empty distillation output" and every learn fork half silently dropped.
    """
    calls: list[dict[str, Any]] = []

    async def complete_fn(**kwargs: Any) -> SimpleNamespace:
        calls.append(dict(kwargs))
        # First attempt (with response_format) returns empty.
        if kwargs.get("response_format") is not None:
            return _completion("")
        # Retry (response_format=None, JSON-only instruction appended) returns valid JSON.
        return _completion('{"facts": ["fact one", "fact two"]}')

    # execute_tool_fn is required to persist; stub it so we isolate the distillation loop.
    persisted: dict[str, Any] = {}

    async def _execute_tool(_tool_id: str, tool_ctx: Any) -> OperationResult:
        persisted["content"] = tool_ctx.parameters.get("content")
        return OperationResult(output_text='{"ok": true}')

    ctx = _ctx(
        complete_fn=complete_fn,
        raw_content="the architectural rundown",
        execute_tool_fn=_execute_tool,
    )

    result = await execute(ctx)

    assert len(calls) == 2, "distillation must retry exactly once on empty output"
    assert calls[0]["response_format"] is not None, "first attempt uses response_format"
    assert calls[1]["response_format"] is None, "retry clears response_format"
    assert "Learning stored" in result.output_text
    assert persisted["content"] == "fact one | fact two"


@pytest.mark.asyncio
async def test_execute_retries_then_succeeds_when_first_attempt_truncated_json() -> None:
    """First completion returns fenced JSON truncated mid-array (the
    ``_looks_like_json`` guard refuses prose fallback). Retry returns valid
    JSON. This is the second failure mode seen on 2026-07-18: ``max_tokens``
    too small cut the facts array off mid-fact, producing invalid JSON.
    """
    truncated = '```json\n{\n  "facts": [\n    "Turn execution has three phases'

    async def complete_fn(**kwargs: Any) -> SimpleNamespace:
        if kwargs.get("response_format") is not None:
            return _completion(truncated)
        return _completion('{"facts": ["fact one"]}')

    async def _execute_tool(_tool_id: str, _tool_ctx: Any) -> OperationResult:
        return OperationResult(output_text='{"ok": true}')

    ctx = _ctx(
        complete_fn=complete_fn,
        raw_content="the rundown",
        execute_tool_fn=_execute_tool,
    )

    result = await execute(ctx)
    assert "Learning stored" in result.output_text


@pytest.mark.asyncio
async def test_execute_escalates_max_tokens_on_retry() -> None:
    """The retry re-issues the completion at 2x the first attempt's budget.

    The 2026-09-23 finances-profile incidents: on reasoning models served via
    Ollama, reasoning tokens bill against ``max_tokens``, so the first
    attempt's budget can truncate the JSON array mid-fact — and re-issuing
    the SAME budget with a JSON-only nudge just reproduces the truncation.
    The retry must escalate (mirroring the kernel's finish=length rescue),
    not merely rephrase.
    """
    calls: list[dict[str, Any]] = []

    async def complete_fn(**kwargs: Any) -> SimpleNamespace:
        calls.append(dict(kwargs))
        # First attempt truncates; retry (with the raised budget) succeeds.
        if kwargs.get("response_format") is not None:
            return _completion('{"facts": ["fact that got cut')
        return _completion('{"facts": ["fact one", "fact two"]}')

    async def _execute_tool(_tool_id: str, _tool_ctx: Any) -> OperationResult:
        return OperationResult(output_text='{"ok": true}')

    ctx = _ctx(
        complete_fn=complete_fn,
        raw_content="the rundown",
        execute_tool_fn=_execute_tool,
    )

    result = await execute(ctx)

    assert len(calls) == 2, "distillation must retry exactly once"
    assert calls[1]["max_tokens"] == calls[0]["max_tokens"] * 2, (
        "retry must escalate the token budget (2x), not reuse the truncated attempt's"
    )
    assert "Learning stored" in result.output_text


@pytest.mark.asyncio
async def test_execute_raises_when_both_attempts_fail() -> None:
    """When both the first attempt and the JSON-only retry fail, the
    operation surfaces the failure rather than silently persisting nothing
    or storing garbage."""

    async def complete_fn(**kwargs: Any) -> SimpleNamespace:
        # Both attempts return empty.
        return _completion("")

    async def _execute_tool(_tool_id: str, _tool_ctx: Any) -> OperationResult:
        return OperationResult(output_text='{"ok": true}')

    ctx = _ctx(
        complete_fn=complete_fn,
        raw_content="the rundown",
        execute_tool_fn=_execute_tool,
    )

    with pytest.raises(RuntimeError, match="distillation failed: empty distillation output"):
        await execute(ctx)

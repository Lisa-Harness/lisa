"""Tests for kernel/operations/declarative.py — declarative operation retry path.

Covers the 2026-07-25 incident: the JSON-only retry path called
``_parse_structured_payload(response_format_model, completion.content)`` with
no empty-content guard (unlike the first-attempt path). When the retry
completion returned empty content, ``model_validate_json("")`` leaked an
opaque Pydantic "Invalid JSON: EOF" ``ValidationError`` that the turn runner's
handoff entry-op handler swallowed, leaving the model in a data-less mode.
The fix raises a typed ``OperationExecutionError`` on empty retry content.
"""

from __future__ import annotations

from typing import Any

import pytest
from kernel.models import OperationExecutionContext
from kernel.operations.declarative import execute_declarative_operation
from kernel.operations.registry import OperationExecutionError
from pydantic import BaseModel


class _WebSearchStructuredOutput(BaseModel):
    """Mirrors the search pack's structured-output schema shape."""

    query: str
    results: list[dict[str, str]] = []


class _StubCompletion:
    """Minimal completion shape matching what ``execute_declarative_operation`` reads."""

    def __init__(
        self, *, content: str = "", tool_calls: list[dict[str, Any]] | None = None
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls


def _build_ctx(*, complete_fn, resolve_model_fn=None) -> OperationExecutionContext:
    """Build a minimal declarative-operation execution context."""
    from kernel.seams.host_adapter import build_callable_host

    host = build_callable_host(
        complete_fn=complete_fn,
        resolve_model_fn=resolve_model_fn or _default_resolve_model,
    )
    return OperationExecutionContext(
        operation_id="web_search",
        query="agent frameworks",
        host=host,
    )


def _default_resolve_model(model_id: str) -> Any:
    """Return a minimal model stub the declarative loop can resolve."""
    from types import SimpleNamespace

    return SimpleNamespace(model_id=model_id or "test-model")


def _make_complete_fn(completions: list[_StubCompletion]):
    """Return an async complete stub that returns each completion in turn."""
    call_count = {"n": 0}

    async def complete(**kwargs: Any) -> _StubCompletion:
        idx = min(call_count["n"], len(completions) - 1)
        call_count["n"] += 1
        return completions[idx]

    return complete


_INSTRUCTION_BODY = "Run web_search and return JSON."
_INSTRUCTION_PATH = "test/web_search.operation.md"


# ── Retry path: empty content raises OperationExecutionError ────────────


@pytest.mark.asyncio
async def test_declarative_empty_retry_content_raises_typed_error() -> None:
    """When the JSON-only retry returns empty content, the operation raises a
    typed ``OperationExecutionError`` (not a raw Pydantic ``ValidationError``).

    This is the exact 2026-07-25 incident: the first attempt returned
    non-JSON prose (triggering the retry), and the retry returned no text.
    Before the fix, ``model_validate_json("")`` leaked
    ``ValidationError: Invalid JSON: EOF while parsing a value``.
    """
    # First attempt returns prose (not valid JSON) -> triggers JSON-only retry.
    # Retry returns empty content -> must raise OperationExecutionError.
    completions = [
        _StubCompletion(content="I could not run the search."),  # non-JSON
        _StubCompletion(content=""),  # empty retry — the incident
    ]
    ctx = _build_ctx(complete_fn=_make_complete_fn(completions))

    with pytest.raises(OperationExecutionError) as exc_info:
        await execute_declarative_operation(
            ctx,
            operation_id="web_search",
            instruction_markdown_path=_INSTRUCTION_PATH,
            instruction_body=_INSTRUCTION_BODY,
            max_calls_per_turn=1,
            response_format_model=_WebSearchStructuredOutput,
            response_output_text_field="output_text",
            response_output_json_passthrough={},
            completion_max_tokens=1500,
            completion_temperature=0.2,
        )
    assert "returned no content" in str(exc_info.value)
    assert "web_search" in str(exc_info.value)


@pytest.mark.asyncio
async def test_declarative_empty_first_attempt_returns_done_no_leak() -> None:
    """When the FIRST attempt returns empty content, the first-attempt guard
    (``if response_format_model is not None and completion.content:``) is
    falsy, so no parse is attempted and no ``ValidationError`` leaks. The
    operation falls through to the ``not completion.tool_calls`` branch and
    returns ``"Done"``. This is the safe path — empty content never reaches
    ``model_validate_json`` on the first attempt. The incident was the
    *retry* path (``test_declarative_empty_retry_content_raises_typed_error``).
    """
    completions = [
        _StubCompletion(content=""),  # empty first attempt — no parse, no leak
    ]
    ctx = _build_ctx(complete_fn=_make_complete_fn(completions))

    result = await execute_declarative_operation(
        ctx,
        operation_id="web_search",
        instruction_markdown_path=_INSTRUCTION_PATH,
        instruction_body=_INSTRUCTION_BODY,
        max_calls_per_turn=1,
        response_format_model=_WebSearchStructuredOutput,
        response_output_text_field="output_text",
        response_output_json_passthrough={},
        completion_max_tokens=1500,
        completion_temperature=0.2,
    )
    # No crash, no leaked ValidationError — the empty-content guard held.
    assert result.output_text == "Done"


@pytest.mark.asyncio
async def test_declarative_valid_retry_succeeds() -> None:
    """A non-empty, valid-JSON retry still parses and returns a result —
    the empty-content guard does NOT break the happy retry path."""
    completions = [
        _StubCompletion(content="not json, retry me"),  # triggers retry
        _StubCompletion(content='{"query": "agent frameworks", "results": []}'),  # valid
    ]
    ctx = _build_ctx(complete_fn=_make_complete_fn(completions))

    result = await execute_declarative_operation(
        ctx,
        operation_id="web_search",
        instruction_markdown_path=_INSTRUCTION_PATH,
        instruction_body=_INSTRUCTION_BODY,
        max_calls_per_turn=1,
        response_format_model=_WebSearchStructuredOutput,
        response_output_text_field="output_text",
        response_output_json_passthrough={},
        completion_max_tokens=1500,
        completion_temperature=0.2,
    )
    # The structured payload's JSON is returned in output_json.
    assert result.output_json.get("query") == "agent frameworks"


def _make_tool_execute_fn(*, tool_output: str = "no results"):
    """Return an async tool-execution stub that records dispatched tools."""

    async def execute(tool_name: str, ctx: Any) -> str:
        return tool_output

    return execute


@pytest.mark.asyncio
async def test_declarative_retry_tool_call_turn_deferred_not_failed() -> None:
    """A JSON-only retry that returns a tool-call-only turn must NOT hard-fail.

    This is the 2026-08-01 ``web_search`` incident: the retry came back with
    ``finish_reason=tool_calls`` and empty visible content because the model
    wanted to dispatch the ``web_search`` tool before emitting its final JSON
    envelope. Before the fix the retry's empty-content guard raised
    ``OperationExecutionError`` ("returned no content after JSON-only retry").
    Now a tool-call turn defers parsing, executes the tool, and lets the loop
    continue to a real JSON answer.
    """
    completions = [
        _StubCompletion(content="I'll research that now."),  # prose -> retry
        _StubCompletion(  # retry: tool-call-only turn (empty content)
            content="",
            tool_calls=[
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": '{"query": "agent frameworks"}',
                    },
                }
            ],
        ),
        _StubCompletion(content='{"query": "agent frameworks", "results": []}'),
    ]
    ctx = _build_ctx(
        complete_fn=_make_complete_fn(completions),
    )
    from kernel.seams.host_adapter import build_callable_host as _bch

    # Preserve complete/resolve_model from the existing host, add execute_tool.
    _existing = ctx.host
    ctx.host = _bch(
        complete_fn=_existing._complete_fn,
        resolve_model_fn=_existing._resolve_model_fn,
        execute_tool_fn=_make_tool_execute_fn(),
    )

    result = await execute_declarative_operation(
        ctx,
        operation_id="web_search",
        instruction_markdown_path=_INSTRUCTION_PATH,
        instruction_body=_INSTRUCTION_BODY,
        max_calls_per_turn=2,
        response_format_model=_WebSearchStructuredOutput,
        response_output_text_field="output_text",
        response_output_json_passthrough={},
        completion_max_tokens=1500,
        completion_temperature=0.2,
    )
    assert result.output_json.get("query") == "agent frameworks"


@pytest.mark.asyncio
async def test_declarative_retry_prose_degrades_gracefully() -> None:
    """A JSON-only retry that returns prose again must not leak a
    ``ValidationError``; it degrades to the raw content.

    This is the other 2026-08-01 ``web_search`` signature: the retry response
    was a prose sentence (``input_value="I'll research the latest..."``) that
    the retry path fed to ``_parse_structured_payload`` unguarded. Before the
    fix the raw Pydantic ``ValidationError`` escaped to the operation registry.
    """
    completions = [
        _StubCompletion(content="Let me search for that."),  # prose -> retry
        _StubCompletion(content="I'll research the latest frameworks."),  # prose again
    ]
    ctx = _build_ctx(complete_fn=_make_complete_fn(completions))

    result = await execute_declarative_operation(
        ctx,
        operation_id="web_search",
        instruction_markdown_path=_INSTRUCTION_PATH,
        instruction_body=_INSTRUCTION_BODY,
        max_calls_per_turn=1,
        response_format_model=_WebSearchStructuredOutput,
        response_output_text_field="output_text",
        response_output_json_passthrough={},
        completion_max_tokens=1500,
        completion_temperature=0.2,
    )
    # No ValidationError raised; the raw prose is surfaced as output_text.
    assert "I'll research the latest frameworks." in result.output_text


# ── max_calls_per_turn=0 means unlimited (catalog convention) ──────────


@pytest.mark.asyncio
async def test_declarative_zero_max_calls_per_turn_is_unlimited() -> None:
    """``max_calls_per_turn=0`` is the catalog "unlimited" sentinel
    (``OperationDefinition.max_calls_per_turn``), not "zero calls allowed".

    Before the fix, ``remaining_tool_calls = max(0, 0) = 0`` made the very
    first tool call return ``"Tool call budget exhausted."`` instead of
    dispatching the tool — the 2026-08-07 incident: the ``web_search``
    operation (no authored cap, defaulting to 0) exhausted on its first
    invocation, surfacing "Tool call budget exhausted." as the turn's
    response text.
    """
    completions = [
        _StubCompletion(  # first attempt: dispatch the tool
            content="",
            tool_calls=[
                {
                    "id": "c1",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": '{"query": "agent frameworks"}',
                    },
                }
            ],
        ),
        _StubCompletion(content='{"query": "agent frameworks", "results": []}'),
    ]
    ctx = _build_ctx(complete_fn=_make_complete_fn(completions))
    from kernel.seams.host_adapter import build_callable_host as _bch

    dispatched: list[str] = []

    async def _execute_tool(tool_name: str, _ctx: Any) -> str:
        dispatched.append(tool_name)
        return "no results"

    _existing = ctx.host
    ctx.host = _bch(
        complete_fn=_existing._complete_fn,
        resolve_model_fn=_existing._resolve_model_fn,
        execute_tool_fn=_execute_tool,
    )

    result = await execute_declarative_operation(
        ctx,
        operation_id="web_search",
        instruction_markdown_path=_INSTRUCTION_PATH,
        instruction_body=_INSTRUCTION_BODY,
        max_calls_per_turn=0,
        response_format_model=_WebSearchStructuredOutput,
        response_output_text_field="output_text",
        response_output_json_passthrough={},
        completion_max_tokens=1500,
        completion_temperature=0.2,
    )
    # The tool was dispatched (not short-circuited as "budget exhausted").
    assert dispatched == ["web_search"]
    assert result.output_json.get("query") == "agent frameworks"

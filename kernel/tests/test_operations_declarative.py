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
from pydantic import BaseModel

from kernel.models import OperationExecutionContext
from kernel.operations.declarative import execute_declarative_operation
from kernel.operations.registry import OperationExecutionError


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
    return OperationExecutionContext(
        operation_id="web_search",
        query="agent frameworks",
        complete_fn=complete_fn,
        resolve_model_fn=resolve_model_fn or _default_resolve_model,
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

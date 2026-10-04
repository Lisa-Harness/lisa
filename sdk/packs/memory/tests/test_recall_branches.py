"""Branch-coverage tests for the recall_memory operation.

Complements ``test_memory_recall.py`` with the remaining uncovered branches:

- ``_parse_memory_records``: non-JSON output, dict envelope with a non-list
  ``results``, bare dict payload, non-dict list items, per-item validation
  failure
- ``_render_recall_output``: not-found path fields, uncertainty-note and
  evidence (with/without ``source_turn``) lines
- ``execute``: empty query no-op, missing ``ctx.params`` guard, TypeError
  "not wired" host, re-raised foreign TypeError, tool error propagation
- ``_resolve_recall_limit``: default-limit fallback, explicit limit,
  falsy limit clamped to 1
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from kernel.models import (
    MemoryRecord,
    OperationExecutionContext,
    ToolResult,
)
from kernel.seams.host_adapter import build_callable_host
from sdk.packs.memory.operations.memory import recall as op
from sdk.packs.memory.utils.memory_recall import build_memory_recall_answer


def _record(
    *,
    record_id: int,
    content: str,
    salience: float = 0.0,
    created_at: str = "2026-07-10T00:00:00Z",
) -> MemoryRecord:
    return MemoryRecord(
        id=record_id,
        content=content,
        content_json={},
        memory_scope="personal",
        memory_tier="long_term",
        source_user_slug="adam",
        operation_id="save_learning",
        surface_id="",
        created_at=created_at,
        salience=salience,
    )


# ── _parse_memory_records ─────────────────────────────────────────────


@pytest.mark.parametrize(
    ("payload", "expected_count"),
    [
        ("not json at all", 0),  # JSONDecodeError → []
        ("[]", 0),  # empty list
        ('{"results": "not-a-list"}', 0),  # envelope with non-list results → dict is not a list
        ('{"unexpected": true}', 0),  # bare dict without "results"
        ('["a", 42]', 0),  # non-dict list items skipped
        ('[{"id": "not-an-int"}]', 0),  # item fails MemoryRecord validation
    ],
)
def test_parse_memory_records_defensive_branches(payload: str, expected_count: int) -> None:
    result = ToolResult(tool_id="memory.search", output_text=payload)

    records = op._parse_memory_records(result)

    assert len(records) == expected_count


def test_parse_memory_records_skips_invalid_items_and_keeps_valid() -> None:
    payload = json.dumps(
        [
            "not-a-dict",
            {"id": "not-an-int"},  # ValidationError → dropped
            _record(record_id=1, content="valid").model_dump(mode="json"),
        ]
    )

    records = op._parse_memory_records(ToolResult(tool_id="memory.search", output_text=payload))

    assert [r.id for r in records] == [1]


# ── _render_recall_output ─────────────────────────────────────────────


def test_render_recall_output_not_found_returns_bare_answer() -> None:
    recall = build_memory_recall_answer(
        query="totally unrelated",
        records=[_record(record_id=1, content="We chose PostgreSQL.", salience=0.0)],
        limit=2,
    )
    assert recall.found is False

    result = op._render_recall_output(recall)

    assert result.output_text == recall.answer
    assert result.output_json is not None and result.output_json["found"] is False
    assert result.extra == {"confidence": recall.confidence, "auto_deep_used": False}


def test_render_recall_output_includes_turn_pointers_for_evidence_with_turns() -> None:
    """Evidence with a ``source_turn`` renders the thread-turn pointer so the
    model can verify the claim via thread_read_range."""
    evidence_record = _record(
        record_id=1, content="We chose PostgreSQL for analytics.", salience=0.9
    )
    recall = build_memory_recall_answer(
        query="postgres analytics",
        records=[evidence_record],
        limit=1,
    )
    assert recall.found is True
    # Stamp a source turn onto the evidence item.
    recall.evidence[0].source_turn = 17
    recall.uncertainty_note = "low signal"

    result = op._render_recall_output(recall)

    assert "Uncertainty: low signal" in result.output_text
    assert "Relevant memories:" in result.output_text
    assert "2026-07-10 (learned around turn 17)" in result.output_text
    assert "We chose PostgreSQL for analytics." in result.output_text


def test_render_recall_output_omits_turn_pointer_when_absent() -> None:
    recall = build_memory_recall_answer(
        query="postgres analytics",
        records=[_record(record_id=1, content="We chose PostgreSQL.", salience=0.9)],
        limit=1,
    )
    assert recall.found is True
    assert recall.evidence[0].source_turn is None

    result = op._render_recall_output(recall)

    assert "(learned around turn" not in result.output_text


# ── _resolve_recall_limit ─────────────────────────────────────────────


def test_resolve_recall_limit_defaults_and_clamps() -> None:
    ctx_default = OperationExecutionContext(
        operation_id="recall_memory",
        params={"query": "x"},
    )
    assert op._resolve_recall_limit(ctx_default) >= 1

    ctx_explicit = OperationExecutionContext(
        operation_id="recall_memory",
        params={"query": "x", "limit": 7},
    )
    assert op._resolve_recall_limit(ctx_explicit) == 7

    ctx_falsy = OperationExecutionContext(
        operation_id="recall_memory",
        params={"query": "x", "limit": ""},
    )
    # Falsy limit coerces to 1 via ``or 1``.
    assert op._resolve_recall_limit(ctx_falsy) == 1


# ── execute: guard branches ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_execute_empty_query_returns_no_op() -> None:
    ctx = OperationExecutionContext(
        operation_id="recall_memory",
        user_slug="adam",
        user_id="adam",
        workspace_id="default",
        query="   ",
        params={"query": "   "},
        host=build_callable_host(execute_tool_fn=None),
    )

    result = await op.execute(ctx)

    assert result.output_text == ""


@pytest.mark.asyncio
async def test_execute_empty_params_falls_back_to_ctx_query() -> None:
    """With empty params, the query is pulled from ``ctx.query`` instead."""

    async def execute_tool(tool_id: str, tool_ctx: Any) -> ToolResult:
        assert tool_id == "memory.search"
        assert tool_ctx.parameters["query"] == "fallback query"
        return ToolResult(tool_id=tool_id, output_text=json.dumps([]))

    ctx = OperationExecutionContext(
        operation_id="recall_memory",
        user_slug="adam",
        user_id="adam",
        workspace_id="default",
        query="fallback query",
        host=build_callable_host(execute_tool_fn=execute_tool),
    )
    object.__setattr__(ctx, "params", {})

    result = await op.execute(ctx)

    assert result.output_json is not None
    assert result.output_json["found"] is False


@pytest.mark.asyncio
async def test_execute_host_not_wired_typeerror_reports_unavailable() -> None:
    """A host whose execute_tool raises the canonical "not wired" TypeError
    surfaces the unavailable message instead of propagating."""

    async def execute_tool(_tool_id: str, _ctx: Any) -> ToolResult:
        raise TypeError("memory tool execution not wired")

    ctx = OperationExecutionContext(
        operation_id="recall_memory",
        user_slug="adam",
        user_id="adam",
        workspace_id="default",
        params={"query": "anything"},
        host=build_callable_host(execute_tool_fn=execute_tool),
    )

    result = await op.execute(ctx)

    assert result.output_text == "Memory recall unavailable: tool execution not injected."


@pytest.mark.asyncio
async def test_execute_re_raises_foreign_typeerror() -> None:
    """A TypeError that does NOT carry the "not wired" marker propagates."""

    async def execute_tool(_tool_id: str, _ctx: Any) -> ToolResult:
        raise TypeError("something else entirely")

    ctx = OperationExecutionContext(
        operation_id="recall_memory",
        user_slug="adam",
        user_id="adam",
        workspace_id="default",
        params={"query": "anything"},
        host=build_callable_host(execute_tool_fn=execute_tool),
    )

    with pytest.raises(TypeError, match="something else entirely"):
        await op.execute(ctx)


@pytest.mark.asyncio
async def test_execute_tool_error_reports_unavailable() -> None:
    async def execute_tool(tool_id: str, _ctx: Any) -> ToolResult:
        return ToolResult(tool_id=tool_id, error="search backend down")

    ctx = OperationExecutionContext(
        operation_id="recall_memory",
        user_slug="adam",
        user_id="adam",
        workspace_id="default",
        params={"query": "anything"},
        host=build_callable_host(execute_tool_fn=execute_tool),
    )

    result = await op.execute(ctx)

    assert result.output_text == "Memory recall unavailable: search backend down"


# ── execute: parse → build → render pipeline ──────────────────────────


@pytest.mark.asyncio
async def test_execute_found_with_evidence_without_turns_and_without_note() -> None:
    """The found branch with no ``uncertainty_note`` and evidence lacking a
    source turn renders neither the Uncertainty line nor the turn pointer."""

    async def execute_tool(tool_id: str, _ctx: Any) -> ToolResult:
        return ToolResult(
            tool_id=tool_id,
            output_text=json.dumps(
                [
                    _record(
                        record_id=1,
                        content="We chose PostgreSQL for analytics.",
                        salience=0.9,
                    ).model_dump(mode="json"),
                ]
            ),
        )

    ctx = OperationExecutionContext(
        operation_id="recall_memory",
        user_slug="adam",
        user_id="adam",
        workspace_id="default",
        params={"query": "postgres analytics", "limit": 3},
        host=build_callable_host(execute_tool_fn=execute_tool),
    )

    result = await op.execute(ctx)

    assert "Uncertainty:" not in result.output_text
    assert "(learned around turn" not in result.output_text
    assert "2026-07-10 — We chose PostgreSQL for analytics." in result.output_text

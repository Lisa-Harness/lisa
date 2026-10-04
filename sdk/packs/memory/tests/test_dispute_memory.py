"""Tests for the dispute_memory operation's deterministic lookup branches.

Covers the ``execute`` flow end-to-end against ``build_callable_host`` fakes:

- record found by id → superseding write with boosted salience
- ``memory.get`` error / "No memory found." / invalid JSON / wrong-shape
  payloads → lookup failure or not-found result
- search fallback: ``{"results": [...]}`` envelope, bare list, error
- memory.add error → failure result
- no correction provided, host not injected
- pure-lookup helpers ``_find_disputed_record_by_id/-_by_search``
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from kernel.models import MemoryRecord, OperationExecutionContext, ToolResult
from kernel.seams.host_adapter import build_callable_host
from sdk.packs.memory.operations.memory import dispute as op


def _memory_record(
    *,
    record_id: int = 5,
    content: str = "We chose Postgres for analytics.",
    salience: float = 0.5,
    memory_tier: str = "long_term",
) -> MemoryRecord:
    return MemoryRecord(
        id=record_id,
        content=content,
        content_json={},
        memory_scope="personal",
        memory_tier=memory_tier,
        source_user_slug="adam",
        operation_id="save_learning",
        surface_id="",
        created_at="2026-07-10T00:00:00Z",
        salience=salience,
    )


def _tool_result(tool_id: str, output_text: str = "", error: str = "") -> ToolResult:
    return ToolResult(tool_id=tool_id, output_text=output_text, error=error)


def _ctx(
    *,
    params: dict[str, Any],
    execute_tool_fn: Any = None,
    with_host: bool = True,
) -> OperationExecutionContext:
    return OperationExecutionContext(
        operation_id="dispute_memory",
        user_slug="adam",
        user_id="adam",
        workspace_id="default",
        surface_id="surface",
        memory_scope="personal",
        params=params,
        host=(
            build_callable_host(execute_tool_fn=execute_tool_fn)
            if with_host and execute_tool_fn is not None
            else None
        ),
    )


# ── execute: input validation ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_execute_without_correction_returns_no_op() -> None:
    async def execute_tool(tool_id: str, _ctx: object) -> ToolResult:
        return _tool_result(tool_id)

    ctx = _ctx(params={"memory_id": "5"}, execute_tool_fn=execute_tool)

    result = await op.execute(ctx)

    assert result.output_text == "⛔ No correction provided."


@pytest.mark.asyncio
async def test_execute_without_host_reports_unavailable() -> None:
    ctx = _ctx(params={"memory_id": "5", "correction": "fixed"}, with_host=False)

    result = await op.execute(ctx)

    assert result.output_text == "Memory dispute unavailable: tool execution not injected."


# ── execute: happy paths ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_execute_by_id_writes_superseding_record_with_boosted_salience() -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    async def execute_tool(tool_id: str, tool_ctx: Any) -> ToolResult:
        params = dict(getattr(tool_ctx, "parameters", {}) or {})
        calls.append((tool_id, params))
        if tool_id == "memory.get":
            return _tool_result(
                tool_id, output_text=json.dumps(_memory_record().model_dump(mode="json"))
            )
        if tool_id == "memory.add":
            return _tool_result(tool_id, output_text="Memory stored with ID 9")
        return _tool_result(tool_id, error=f"unknown tool {tool_id}")

    ctx = _ctx(
        params={
            "memory_id": "5",
            "correction": "We chose Postgres for both OLTP and analytics.",
            "reason": "outdated",
        },
        execute_tool_fn=execute_tool,
    )

    result = await op.execute(ctx)

    assert "✅ Memory record 5 disputed and superseded." in result.output_text
    assert "salience 0.8" in result.output_text  # floor binds: 0.5 + 0.2 = 0.7 < 0.8
    assert "Reason: outdated" in result.output_text
    assert result.output_json == {
        "superseded_id": "5",
        "correction": "We chose Postgres for both OLTP and analytics.",
        "reason": "outdated",
        "salience": 0.8,
    }
    assert calls[0][0] == "memory.get"
    assert calls[1][0] == "memory.add"
    add_params = calls[1][1]
    assert add_params["content_json"]["supersedes_id"] == "5"
    assert add_params["content_json"]["dispute_reason"] == "outdated"
    assert add_params["content_json"]["original_content"] == _memory_record().content


@pytest.mark.asyncio
async def test_execute_by_search_finds_first_hit_and_supersedes() -> None:
    async def execute_tool(tool_id: str, _ctx: Any) -> ToolResult:
        if tool_id == "memory.search":
            return _tool_result(
                tool_id,
                output_text=json.dumps([_memory_record(record_id=7).model_dump(mode="json")]),
            )
        if tool_id == "memory.add":
            return _tool_result(tool_id, output_text="Memory stored with ID 10")
        return _tool_result(tool_id, error=f"unknown tool {tool_id}")

    ctx = _ctx(
        params={"search_query": "postgres analytics", "correction": "corrected fact"},
        execute_tool_fn=execute_tool,
    )

    result = await op.execute(ctx)

    assert "✅ Memory record 7 disputed and superseded." in result.output_text
    assert result.output_json is not None
    assert result.output_json["superseded_id"] == "7"
    assert result.output_json["reason"] == "user_correction"


@pytest.mark.asyncio
async def test_execute_by_search_unwraps_results_envelope() -> None:
    """Mem0-style ``{"results": [...]}`` envelope is unwrapped to the list."""

    async def execute_tool(tool_id: str, _ctx: Any) -> ToolResult:
        if tool_id == "memory.search":
            return _tool_result(
                tool_id,
                output_text=json.dumps(
                    {"results": [_memory_record(record_id=11).model_dump(mode="json")]}
                ),
            )
        if tool_id == "memory.add":
            return _tool_result(tool_id, output_text="Memory stored with ID 12")
        return _tool_result(tool_id, error=f"unknown tool {tool_id}")

    ctx = _ctx(
        params={"search_query": "whatever", "correction": "corrected"},
        execute_tool_fn=execute_tool,
    )

    result = await op.execute(ctx)

    assert "✅ Memory record 11 disputed and superseded." in result.output_text


# ── execute: lookup misses and errors ─────────────────────────────────


@pytest.mark.asyncio
async def test_execute_no_get_hit_reports_record_not_found() -> None:
    """``memory.get`` answers the canonical "No memory found." sentinel."""

    async def execute_tool(tool_id: str, _ctx: Any) -> ToolResult:
        assert tool_id == "memory.get"
        return _tool_result(tool_id, output_text="No memory found.")

    ctx = _ctx(params={"memory_id": "42", "correction": "fix"}, execute_tool_fn=execute_tool)

    result = await op.execute(ctx)

    assert result.output_text.startswith("⛔ Could not find a memory record to dispute.")


@pytest.mark.asyncio
async def test_execute_get_error_surfaces_lookup_failure() -> None:
    async def execute_tool(tool_id: str, _ctx: Any) -> ToolResult:
        assert tool_id == "memory.get"
        return _tool_result(tool_id, error="store exploded")

    ctx = _ctx(params={"memory_id": "42", "correction": "fix"}, execute_tool_fn=execute_tool)

    result = await op.execute(ctx)

    assert result.output_text == "Memory dispute: could not fetch memory: store exploded"


@pytest.mark.asyncio
async def test_execute_search_error_surfaces_lookup_failure() -> None:
    async def execute_tool(tool_id: str, _ctx: Any) -> ToolResult:
        assert tool_id == "memory.search"
        return _tool_result(tool_id, error="backend 500")

    ctx = _ctx(
        params={"search_query": "anything", "correction": "fix"}, execute_tool_fn=execute_tool
    )

    result = await op.execute(ctx)

    assert result.output_text == "Memory dispute: search failed: backend 500"


@pytest.mark.asyncio
async def test_execute_no_identifier_reports_not_found() -> None:
    """Neither memory_id nor search_query → nothing looked up, not found."""

    async def execute_tool(tool_id: str, _ctx: Any) -> ToolResult:  # pragma: no cover
        raise AssertionError("no tool should be called")

    ctx = _ctx(params={"correction": "fix"}, execute_tool_fn=execute_tool)

    result = await op.execute(ctx)

    assert result.output_text.startswith("⛔ Could not find a memory record to dispute.")


@pytest.mark.asyncio
async def test_execute_add_error_reports_write_failure() -> None:
    async def execute_tool(tool_id: str, _ctx: Any) -> ToolResult:
        if tool_id == "memory.get":
            return _tool_result(
                tool_id, output_text=json.dumps(_memory_record().model_dump(mode="json"))
            )
        if tool_id == "memory.add":
            return _tool_result(tool_id, error="quota exhausted")
        return _tool_result(tool_id, error=f"unknown tool {tool_id}")

    ctx = _ctx(params={"memory_id": "5", "correction": "fix"}, execute_tool_fn=execute_tool)

    result = await op.execute(ctx)

    assert result.output_text == "Memory dispute: failed to write correction: quota exhausted"


# ── lookup helpers: defensive parse branches ──────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw",
    [
        "",  # empty output
        "   ",  # whitespace-only output
        "not json at all",  # JSONDecodeError → None
        '["a", "list"]',  # valid JSON but not a dict
        '{"unexpected": "shape"}',  # dict that fails MemoryRecord validation
    ],
)
async def test_find_by_id_defensive_parsing_returns_none(raw: str) -> None:
    async def execute_tool(tool_id: str, _ctx: Any) -> ToolResult:
        return _tool_result(tool_id, output_text=raw)

    ctx = _ctx(params={}, execute_tool_fn=execute_tool)

    assert await op._find_disputed_record_by_id(ctx, "5") is None


@pytest.mark.asyncio
async def test_find_by_id_returns_validated_record() -> None:
    async def execute_tool(tool_id: str, tool_ctx: Any) -> ToolResult:
        assert tool_id == "memory.get"
        params = getattr(tool_ctx, "parameters", {})
        assert params["memory_id"] == 5
        assert params["user_slug"] == "adam"
        return _tool_result(
            tool_id,
            output_text=json.dumps(_memory_record(record_id=5).model_dump(mode="json")),
        )

    ctx = _ctx(params={}, execute_tool_fn=execute_tool)

    record = await op._find_disputed_record_by_id(ctx, "5")

    assert record is not None
    assert record.id == 5


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw",
    [
        "not json",  # JSONDecodeError → []
        "[]",  # empty list → no first hit
        '{"no_results_key": true}',  # dict without "results" → not a list
        '[{"unexpected": "shape"}]',  # item fails MemoryRecord validation
    ],
)
async def test_find_by_search_defensive_parsing_returns_none(raw: str) -> None:
    async def execute_tool(tool_id: str, _ctx: Any) -> ToolResult:
        return _tool_result(tool_id, output_text=raw)

    ctx = _ctx(params={}, execute_tool_fn=execute_tool)

    assert await op._find_disputed_record_by_search(ctx, "query") is None

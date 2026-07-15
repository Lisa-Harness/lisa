from __future__ import annotations

import json

import pytest

from kernel.models import MemoryRecord, OperationExecutionContext, ToolResult
from sdk.packs.learning.operations.memory.recall import execute
from sdk.packs.learning.utils.memory_recall import build_memory_recall_answer


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


def test_build_memory_recall_answer_ranks_overlap_then_salience() -> None:
    recall = build_memory_recall_answer(
        query="postgres chose",
        records=[
            _record(record_id=2, content="We chose PostgreSQL for analytics.", salience=0.2),
            _record(
                record_id=1,
                content="We chose PostgreSQL for the primary database.",
                salience=0.8,
            ),
        ],
        limit=2,
    )

    assert recall.found is True
    assert recall.answer == "We chose PostgreSQL for the primary database."
    assert [item.memory_id for item in recall.evidence] == [1, 2]


@pytest.mark.asyncio
async def test_recall_operation_uses_memory_list_and_returns_ranked_evidence() -> None:
    calls: list[str] = []

    async def execute_tool(tool_id: str, _ctx: object) -> ToolResult:
        calls.append(tool_id)
        assert tool_id == "memory.search"
        return ToolResult(
            tool_id=tool_id,
            output_text=json.dumps(
                [
                    _record(
                            record_id=10,
                        content="We decided to use PostgreSQL for the control plane.",
                        salience=0.7,
                    ).model_dump(mode="json"),
                    _record(
                            record_id=20,
                        content="We postponed the Redis work.",
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
        params={"query": "postgres control plane", "limit": 3},
        execute_tool_fn=execute_tool,
    )

    result = await execute(ctx)

    assert calls == ["memory.search"]
    assert "Relevant memories:" in result.output_text
    assert "PostgreSQL for the control plane" in result.output_text
    assert result.output_json["found"] is True


@pytest.mark.asyncio
async def test_recall_operation_handles_missing_tool_executor() -> None:
    ctx = OperationExecutionContext(
        operation_id="recall_memory",
        params={"query": "postgres"},
    )

    result = await execute(ctx)

    assert result.output_text == "Memory recall unavailable: tool execution not injected."


@pytest.mark.asyncio
async def test_recall_operation_searches_and_unwraps_mem0_envelope() -> None:
    """recall_memory calls memory.search (not memory.list) and unwraps a
    Mem0-style paginated envelope dict into records for ranking."""
    calls: list[str] = []

    async def execute_tool(tool_id: str, _ctx: object) -> ToolResult:
        calls.append(tool_id)
        assert tool_id == "memory.search"
        # Return a Mem0-style paginated envelope (dict, not list).
        # The defensive unwrap in recall.py should extract .results.
        return ToolResult(
            tool_id=tool_id,
            output_text=json.dumps({
                "results": [
                    {
                        "id": "uuid-1",
                        "memory": "We decided to use PostgreSQL for analytics.",
                        "score": 0.9,
                        "created_at": "2026-07-10T12:00:00Z",
                    },
                    {
                        "id": "uuid-2",
                        "memory": "We postponed the Redis work.",
                        "score": 0.5,
                        "created_at": "2026-07-11T12:00:00Z",
                    },
                ],
            }),
        )

    ctx = OperationExecutionContext(
        operation_id="recall_memory",
        user_slug="adam",
        user_id="adam",
        workspace_id="default",
        params={"query": "postgres analytics", "limit": 3},
        execute_tool_fn=execute_tool,
    )

    result = await execute(ctx)

    # Verify it called memory.search, not memory.list
    assert calls == ["memory.search"]
    # Verify the Mem0 envelope was unwrapped and ranked
    assert result.output_json["found"] is True
    assert "PostgreSQL for analytics" in result.output_text
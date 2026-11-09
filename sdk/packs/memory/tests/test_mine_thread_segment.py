"""Tests for the mine_thread_segment operation.

Mocks the OperationHost seam (complete / resolve_model / read_authored_text /
execute_tool) to verify the segment distillation → semantic persistence flow
without a live model or SQLite store.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from kernel.models import OperationExecutionContext, ToolResult
from kernel.seams.host_adapter import build_callable_host
from sdk.packs.memory.operations.memory import mine_thread_segment as op


def _turns() -> list[dict[str, Any]]:
    return [
        {
            "turn": 10,
            "query": "how do we start the GCP bind VM?",
            "response_text": "needs project_id",
        },
        {"turn": 11, "query": "what model do we prefer?", "response_text": "user prefers Claude"},
        {"turn": 12, "query": "tools were down again", "response_text": "cs_list_vms failed"},
    ]


def _make_ctx(
    *,
    params: dict[str, Any],
    complete_fn: Any,
    execute_tool_fn: Any,
) -> OperationExecutionContext:
    def resolve_model_fn(model_id: str | None) -> dict[str, Any]:
        return {"model_id": "test-model"}

    def read_authored_text_fn(path: str) -> str:
        return "Mining system prompt (stubbed)."

    return OperationExecutionContext(
        operation_id="mine_thread_segment",
        user_slug="adam",
        user_id="adam",
        workspace_id="default",
        params=params,
        host=build_callable_host(
            complete_fn=complete_fn,
            resolve_model_fn=resolve_model_fn,
            read_authored_text_fn=read_authored_text_fn,
            execute_tool_fn=execute_tool_fn,
        ),
    )


@pytest.mark.asyncio
async def test_mine_stores_semantic_records_with_provenance() -> None:
    captured: list[tuple[str, dict[str, Any]]] = []

    async def complete_fn(messages: list[dict[str, Any]], **_: Any) -> str:
        return json.dumps(
            {
                "facts": [
                    {"action": "create", "fact": "GCP start-vm requires project_id"},
                    {"action": "create", "fact": "User prefers Claude"},
                ]
            }
        )

    async def execute_tool_fn(tool_id: str, ctx: Any) -> ToolResult:
        params = getattr(ctx, "parameters", {}) or {}
        captured.append((tool_id, dict(params)))
        if tool_id == "memory.search":
            return ToolResult(tool_id=tool_id, output_text="[]")
        if tool_id == "memory.add":
            return ToolResult(tool_id=tool_id, output_text="Memory stored with ID 42")
        return ToolResult(tool_id=tool_id, error=f"unknown tool {tool_id}")

    ctx = _make_ctx(
        params={"turns": _turns(), "user_slug": "adam"},
        complete_fn=complete_fn,
        execute_tool_fn=execute_tool_fn,
    )

    result = await op.execute(ctx)

    assert result.extra.get("mined") is True
    assert result.extra.get("memory_ids") == [42, 42]
    assert result.extra.get("thread_turn_lo") == 10
    assert result.extra.get("thread_turn_hi") == 12
    add_calls = [c for c in captured if c[0] == "memory.add"]
    assert len(add_calls) == 2
    add_params = add_calls[0][1]
    assert add_params["memory_kind"] == "semantic"
    assert add_params["operation_id"] == "mine_thread_segment"
    assert add_params["memory_tier"] == "long_term"
    assert add_params["content_json"]["thread_turn_lo"] == 10
    assert add_params["content_json"]["thread_turn_hi"] == 12


@pytest.mark.asyncio
async def test_mining_nothing_durable_stores_nothing() -> None:
    async def complete_fn(messages: list[dict[str, Any]], **_: Any) -> str:
        return json.dumps({"facts": []})

    async def execute_tool_fn(tool_id: str, ctx: Any) -> ToolResult:
        if tool_id == "memory.search":
            return ToolResult(tool_id=tool_id, output_text="[]")
        return ToolResult(tool_id=tool_id, error=f"unexpected call {tool_id}")

    ctx = _make_ctx(
        params={"turns": _turns()},
        complete_fn=complete_fn,
        execute_tool_fn=execute_tool_fn,
    )
    result = await op.execute(ctx)
    assert result.extra.get("mined") is False
    assert result.extra.get("memory_ids") == []


@pytest.mark.asyncio
async def test_invalid_params_reports_error() -> None:
    ctx = _make_ctx(
        params={"turns": []},
        complete_fn=None,
        execute_tool_fn=None,
    )
    result = await op.execute(ctx)
    assert "Invalid mining parameters" in result.output_text


@pytest.mark.asyncio
@pytest.mark.parametrize("candidate_id", [7, "memory-7"])
@pytest.mark.parametrize("provider_outcome", [True, False, None])
async def test_mine_update_reports_retirement_outcome(
    candidate_id: int | str, provider_outcome: bool | None, caplog: pytest.LogCaptureFixture
) -> None:
    """An update reports the provider's retirement outcome."""
    captured: list[tuple[str, dict[str, Any]]] = []

    async def complete_fn(messages: list[dict[str, Any]], **_: Any) -> str:
        return json.dumps(
            {
                "facts": [
                    {
                        "action": "update",
                        "fact": "User prefers Claude for distillation work",
                        "supersedes_id": candidate_id,
                        "reason": "refines existing preference",
                    }
                ]
            }
        )

    async def execute_tool_fn(tool_id: str, ctx: Any) -> ToolResult:
        params = getattr(ctx, "parameters", {}) or {}
        captured.append((tool_id, dict(params)))
        if tool_id == "memory.search":
            return ToolResult(
                tool_id=tool_id,
                output_text=json.dumps(
                    [
                        {"id": None, "content": "Record without an id"},
                        {"content": "Record missing an id"},
                        {
                            "id": candidate_id,
                            "content": "User prefers Claude",
                            "created_at": "2026-07-10T00:00:00Z",
                        },
                    ]
                ),
            )
        if tool_id == "memory.add":
            metadata = (
                {"supersession_retired": provider_outcome} if provider_outcome is not None else {}
            )
            return ToolResult(
                tool_id=tool_id, output_text="Memory stored with ID 9", output_json=metadata
            )
        return ToolResult(tool_id=tool_id, error=f"unknown tool {tool_id}")

    ctx = _make_ctx(
        params={"turns": _turns(), "user_slug": "adam"},
        complete_fn=complete_fn,
        execute_tool_fn=execute_tool_fn,
    )

    result = await op.execute(ctx)

    assert result.extra.get("mined") is True
    assert result.extra.get("updated_ids") == [9]
    assert result.extra.get("retired_ids") == ([candidate_id] if provider_outcome is True else [])
    assert result.extra.get("not_retired_by_write_ids") == (
        [candidate_id] if provider_outcome is False else []
    )
    assert result.extra.get("retirement_unknown_ids") == (
        [candidate_id] if provider_outcome is None else []
    )
    assert result.extra.get("supersession_outcomes") == [
        {"memory_id": 9, "supersedes_id": candidate_id, "retired": provider_outcome}
    ]
    assert ("provider did not report retirement outcome" in caplog.text) is (
        provider_outcome is None
    )
    add_calls = [c for c in captured if c[0] == "memory.add"]
    assert add_calls[0][1]["supersedes_id"] == candidate_id
    assert not [c for c in captured if c[0] == "memory.retire"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("receipt", "expected_ids", "stored_unparsed"),
    [
        ('{"memory_id":9,"supersession_retired":true}', [9], 0),
        ('{"supersession_retired":true}', [], 1),
    ],
)
async def test_mine_counts_stored_facts_with_structured_receipts(
    receipt: str, expected_ids: list[int], stored_unparsed: int
) -> None:
    async def complete_fn(messages: list[dict[str, Any]], **_: Any) -> str:
        return json.dumps(
            {"facts": [{"action": "update", "fact": "refined fact", "supersedes_id": 7}]}
        )

    async def execute_tool_fn(tool_id: str, _ctx: Any) -> ToolResult:
        if tool_id == "memory.search":
            return ToolResult(tool_id=tool_id, output_text='[{"id":7,"content":"existing"}]')
        if tool_id == "memory.add":
            return ToolResult(tool_id=tool_id, output_text=receipt)
        return ToolResult(tool_id=tool_id, error=f"unknown tool {tool_id}")

    result = await op.execute(
        _make_ctx(
            params={"turns": _turns()}, complete_fn=complete_fn, execute_tool_fn=execute_tool_fn
        )
    )

    assert result.extra["updated_ids"] == expected_ids
    assert result.extra["stored_unparsed"] == stored_unparsed
    assert result.extra["retired_ids"] == [7]
    assert result.extra["supersession_outcomes"] == [
        {
            "memory_id": expected_ids[0] if expected_ids else None,
            "supersedes_id": 7,
            "retired": True,
        }
    ]
    assert "Mined 1 durable fact(s)" in result.output_text


@pytest.mark.asyncio
async def test_mine_preserves_each_outcome_for_repeated_predecessors() -> None:
    async def complete_fn(messages: list[dict[str, Any]], **_: Any) -> str:
        return json.dumps(
            {
                "facts": [
                    {"action": "update", "fact": "first refinement", "supersedes_id": 7},
                    {"action": "update", "fact": "second refinement", "supersedes_id": 7},
                ]
            }
        )

    receipts = iter(
        [
            '{"memory_id":9,"supersession_retired":true}',
            '{"memory_id":10,"supersession_retired":false}',
        ]
    )

    async def execute_tool_fn(tool_id: str, _ctx: Any) -> ToolResult:
        if tool_id == "memory.search":
            return ToolResult(tool_id=tool_id, output_text='[{"id":7,"content":"existing"}]')
        if tool_id == "memory.add":
            return ToolResult(tool_id=tool_id, output_text=next(receipts))
        return ToolResult(tool_id=tool_id, error=f"unknown tool {tool_id}")

    result = await op.execute(
        _make_ctx(
            params={"turns": _turns()}, complete_fn=complete_fn, execute_tool_fn=execute_tool_fn
        )
    )

    assert result.extra["supersession_outcomes"] == [
        {"memory_id": 9, "supersedes_id": 7, "retired": True},
        {"memory_id": 10, "supersedes_id": 7, "retired": False},
    ]
    assert result.extra["retired_ids"] == [7]
    assert result.extra["not_retired_by_write_ids"] == [7]


@pytest.mark.asyncio
async def test_mine_skip_only_facts_stores_nothing() -> None:
    """Every fact marked skip → nothing stored, mined=False with the count."""

    async def complete_fn(messages: list[dict[str, Any]], **_: Any) -> str:
        return json.dumps({"facts": [{"action": "skip", "fact": "", "reason": "already stored"}]})

    async def execute_tool_fn(tool_id: str, ctx: Any) -> ToolResult:
        if tool_id == "memory.search":
            return ToolResult(
                tool_id=tool_id,
                output_text=json.dumps(
                    [
                        {
                            "id": 7,
                            "content": "User prefers Claude",
                            "created_at": "2026-07-10T00:00:00Z",
                        }
                    ]
                ),
            )
        return ToolResult(tool_id=tool_id, error=f"unexpected call {tool_id}")

    ctx = _make_ctx(
        params={"turns": _turns()},
        complete_fn=complete_fn,
        execute_tool_fn=execute_tool_fn,
    )

    result = await op.execute(ctx)

    assert result.extra.get("mined") is False
    assert result.extra.get("skipped") == 1


@pytest.mark.asyncio
async def test_mine_update_with_hallucinated_id_degrades_to_create() -> None:
    """A supersedes_id not in the candidate list must not link — the fact is
    stored as a new record instead of superseding an invented target."""

    async def complete_fn(messages: list[dict[str, Any]], **_: Any) -> str:
        return json.dumps(
            {"facts": [{"action": "update", "fact": "refined fact", "supersedes_id": 999}]}
        )

    async def execute_tool_fn(tool_id: str, ctx: Any) -> ToolResult:
        if tool_id == "memory.search":
            return ToolResult(
                tool_id=tool_id,
                output_text=json.dumps(
                    [
                        {
                            "id": 7,
                            "content": "an existing fact",
                            "created_at": "2026-07-10T00:00:00Z",
                        }
                    ]
                ),
            )
        if tool_id == "memory.add":
            return ToolResult(tool_id=tool_id, output_text="Memory stored with ID 11")
        return ToolResult(tool_id=tool_id, error=f"unexpected call {tool_id}")

    ctx = _make_ctx(
        params={"turns": _turns()},
        complete_fn=complete_fn,
        execute_tool_fn=execute_tool_fn,
    )

    result = await op.execute(ctx)

    assert result.extra.get("created_ids") == [11]
    assert result.extra.get("retired_ids") == []


@pytest.mark.asyncio
async def test_mine_all_adds_failing_returns_retryable_error() -> None:
    """Every memory.add erroring → typed status='error' result, never a
    success envelope: the mining slice keys its checkpoint on this."""

    async def complete_fn(messages: list[dict[str, Any]], **_: Any) -> str:
        return json.dumps({"facts": [{"action": "create", "fact": "a durable fact"}]})

    async def execute_tool_fn(tool_id: str, ctx: Any) -> ToolResult:
        if tool_id == "memory.search":
            return ToolResult(tool_id=tool_id, output_text="[]")
        if tool_id == "memory.add":
            return ToolResult(tool_id=tool_id, error="backend offline")
        return ToolResult(tool_id=tool_id, error=f"unexpected call {tool_id}")

    ctx = _make_ctx(
        params={"turns": _turns()},
        complete_fn=complete_fn,
        execute_tool_fn=execute_tool_fn,
    )

    result = await op.execute(ctx)

    assert result.status == "error"
    assert "backend offline" in (result.error_detail or "")
    assert result.extra.get("mined") is False
    assert result.extra.get("memory_ids") == []
    assert "✅" not in result.output_text


@pytest.mark.asyncio
async def test_mine_partial_storage_failure_stops_and_reports_error() -> None:
    """First fact stores, second errors → the operation stops (fail-fast)
    and reports a retryable error; dedup makes the re-mine idempotent."""

    async def complete_fn(messages: list[dict[str, Any]], **_: Any) -> str:
        return json.dumps(
            {
                "facts": [
                    {"action": "create", "fact": "fact one"},
                    {"action": "create", "fact": "fact two"},
                ]
            }
        )

    async def execute_tool_fn(tool_id: str, ctx: Any) -> ToolResult:
        if tool_id == "memory.search":
            return ToolResult(tool_id=tool_id, output_text="[]")
        if tool_id == "memory.add":
            if "fact two" in (getattr(ctx, "parameters", {}) or {}).get("content", ""):
                return ToolResult(tool_id=tool_id, error="backend flaked")
            return ToolResult(tool_id=tool_id, output_text="Memory stored with ID 31")
        return ToolResult(tool_id=tool_id, error=f"unexpected call {tool_id}")

    ctx = _make_ctx(
        params={"turns": _turns()},
        complete_fn=complete_fn,
        execute_tool_fn=execute_tool_fn,
    )

    result = await op.execute(ctx)

    assert result.status == "error"
    assert "backend flaked" in (result.error_detail or "")
    assert "✅" not in result.output_text


@pytest.mark.asyncio
async def test_mine_distill_escalates_max_tokens_on_retry() -> None:
    """The mining distiller's JSON-only retry re-issues the completion at 2x
    the first attempt's budget.

    The 2026-11-08..09 mining crash-loop: 21 consecutive segments failed with
    the model's reasoning text ('Let me analyze this conv…') failing JSON
    validation. On reasoning models the visible JSON shares the completion
    budget with reasoning tokens, so re-issuing the SAME budget with a
    JSON-only nudge reproduces the truncation — the escalation mirrors
    synthesize_learning's documented 2026-09-23 lesson.
    """
    calls: list[dict[str, Any]] = []

    async def complete_fn(messages: list[dict[str, Any]], **kwargs: Any) -> str:
        calls.append(dict(kwargs))
        if kwargs.get("response_format") is not None:
            return "Let me analyze this conversation segment. The user discussed"
        return json.dumps({"facts": [{"action": "create", "fact": "fact one"}]})

    async def execute_tool_fn(tool_id: str, ctx: Any) -> ToolResult:
        if tool_id == "memory.search":
            return ToolResult(tool_id=tool_id, output_text="[]")
        if tool_id == "memory.add":
            return ToolResult(tool_id=tool_id, output_text="Memory stored with ID 7")
        return ToolResult(tool_id=tool_id, error=f"unexpected call {tool_id}")

    ctx = _make_ctx(
        params={"turns": _turns(), "user_slug": "adam"},
        complete_fn=complete_fn,
        execute_tool_fn=execute_tool_fn,
    )

    result = await op.execute(ctx)

    assert len(calls) == 2, "mining distillation must retry exactly once"
    assert calls[1]["max_tokens"] == calls[0]["max_tokens"] * 2, (
        "retry must escalate the token budget (2x), not reuse the truncated attempt's"
    )
    assert result.extra.get("mined") is True
    assert result.extra.get("memory_ids") == [7]

"""Branch-coverage tests for the consolidate_semantic operation.

Complements ``test_consolidate_semantic.py`` with the remaining branches:

- ``_cfg_int`` fallbacks: None config, missing key, unparseable value
- ``_parse_consolidated``: empty output, JSON-shaped validation failure
  guard, non-JSON-lookalike ValidationError propagation
- ``_looks_like_json``: empty / fenced / bare braces / inner braces / plain
- ``_render_batch`` per-field str() coercion
- ``_distill_batch``: host None, model resolution fallback to ``default``,
  first-attempt failure retried with retry messages
- ``_persist_semantic``: salience floor/boost, add error, id-parse failures
- ``_retire_sources``: per-tool error skips the id
- ``execute``: invalid params, empty records, host None, distillation
  failure wrapping, hallucinated-id fallback, persist error, long-summary
  preview ellipsis, retired-ids "+N more" truncation
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from kernel.models import OperationExecutionContext, ToolResult
from kernel.seams.host_adapter import build_callable_host
from pydantic import ValidationError
from sdk.packs.memory.operations.memory import consolidate_semantic as op


def _model_def() -> dict[str, Any]:
    return {"model_id": "test-model"}


def _episodic_records(count: int = 3) -> list[dict[str, Any]]:
    return [
        {
            "id": idx,
            "content": f"Episodic fact {idx}.",
            "created_at": f"2026-07-0{idx}T00:00:00Z",
            "salience": 0.3 + idx * 0.1,
        }
        for idx in range(1, count + 1)
    ]


def _host(
    *,
    complete_fn: Any = None,
    resolve_model_fn: Any = None,
    read_authored_text_fn: Any = None,
    execute_tool_fn: Any = None,
) -> Any:
    return build_callable_host(
        complete_fn=complete_fn,
        resolve_model_fn=resolve_model_fn,
        read_authored_text_fn=read_authored_text_fn,
        execute_tool_fn=execute_tool_fn,
    )


def _ctx(
    *, params: dict[str, Any], host: Any, pack_config: Any = None
) -> OperationExecutionContext:
    return OperationExecutionContext(
        operation_id="consolidate_semantic",
        user_slug="adam",
        user_id="adam",
        workspace_id="default",
        params=params,
        pack_config=pack_config,
        host=host,
    )


def _params(records: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "episodic_records": records if records is not None else _episodic_records(),
        "user_slug": "adam",
        "memory_scope": "personal",
        "surface_id": "",
        "workspace_id": "default",
    }


# ── _cfg_int ──────────────────────────────────────────────────────────


def test_cfg_int_falls_back_for_none_config_missing_key_and_bad_value() -> None:
    assert op._cfg_int(None, "memory_consolidate_max_tokens", 768) == 768
    assert op._cfg_int({}, "memory_consolidate_max_tokens", 768) == 768
    assert (
        op._cfg_int({"memory_consolidate_max_tokens": ""}, "memory_consolidate_max_tokens", 768)
        == 768
    )
    assert (
        op._cfg_int(
            {"memory_consolidate_max_tokens": "not-an-int"},
            "memory_consolidate_max_tokens",
            768,
        )
        == 768
    )
    assert op._cfg_int({"memory_consolidate_max_tokens": "1024"}, "k", 768) == 768


def test_resolve_settings_honors_pack_config_override() -> None:
    ctx = _ctx(
        params={},
        host=None,
        pack_config={"memory_consolidate_max_tokens": "2048"},
    )

    settings = op._resolve_settings(ctx)

    assert settings.consolidate_max_tokens == 2048


# ── _looks_like_json ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", False),
        ("   ", False),
        ("```json\n{}\n```", True),
        ('{"any": "object"}', True),
        ("[1, 2]", True),
        ("prose with {inner} braces", True),
        ("plain prose, no braces", False),
    ],
)
def test_looks_like_json(text: str, expected: bool) -> None:
    assert op._looks_like_json(text) is expected


# ── _parse_consolidated ───────────────────────────────────────────────


def test_parse_consolidated_empty_output_raises_value_error() -> None:
    with pytest.raises(ValueError, match="empty consolidation output"):
        op._parse_consolidated("   ")


def test_parse_consolidated_rejects_json_shaped_invalid_output() -> None:
    """Fenced/JSON-shaped output that fails validation must NOT fall back —
    it raises the loud RuntimeError guard instead."""
    with pytest.raises(RuntimeError, match="refusing to fall back"):
        op._parse_consolidated('```json\n{"summary": 1, "item_ids": "x"}\n```')


def test_parse_consolidated_propagates_validation_error_for_plain_prose() -> None:
    """Plain prose that does not look like JSON propagates ValidationError."""
    with pytest.raises(ValidationError):
        op._parse_consolidated("just some plain prose, no braces")


def test_parse_consolidated_accepts_valid_payload() -> None:
    parsed = op._parse_consolidated(
        '{"summary": "durable fact", "confidence": 0.9, "item_ids": [1, 2]}'
    )

    assert parsed.summary == "durable fact"
    assert parsed.item_ids == [1, 2]


# ── _render_batch ─────────────────────────────────────────────────────


def test_render_batch_renders_numbered_lines_with_defaults() -> None:
    text = op._render_batch(
        [
            {"id": 5, "content": "  spaced content  ", "created_at": "2026-07-01"},
            {"content": None, "created_at": None},
        ]
    )

    lines = text.splitlines()
    assert lines[0] == "1. [id=5, created=2026-07-01] spaced content"
    assert lines[1] == "2. [id=?, created=None] None"


# ── _distill_batch ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_distill_batch_raises_when_host_is_none() -> None:
    ctx = _ctx(params={}, host=None)

    with pytest.raises(RuntimeError, match="could not resolve a model"):
        await op._distill_batch(ctx, "prompt", "batch", 100)


@pytest.mark.asyncio
async def test_distill_batch_falls_back_to_default_model() -> None:
    """Unknown preferred model id → resolve_model("default") is consulted."""
    seen: list[str | None] = []

    def resolve_model_fn(model_id: str | None) -> dict[str, Any] | None:
        seen.append(model_id)
        return _model_def() if model_id == "default" else None

    async def complete_fn(messages: list[dict[str, Any]], **_: Any) -> str:
        return json.dumps({"summary": "s", "confidence": 0.9, "item_ids": [1]})

    ctx = OperationExecutionContext(
        operation_id="consolidate_semantic",
        user_slug="adam",
        user_id="adam",
        workspace_id="default",
        params={},
        preferred_model_id="ghost-model",
        host=_host(resolve_model_fn=resolve_model_fn, complete_fn=complete_fn),
    )

    parsed = await op._distill_batch(ctx, "prompt", "batch", 100)

    assert parsed.summary == "s"
    assert seen == ["ghost-model", "default"]


@pytest.mark.asyncio
async def test_distill_batch_retries_once_on_first_attempt_failure() -> None:
    """First attempt raises → the JSON-only retry messages are issued."""
    calls: list[list[dict[str, Any]]] = []

    async def complete_fn(messages: list[dict[str, Any]], **_: Any) -> str:
        calls.append(messages)
        if len(calls) == 1:
            raise RuntimeError("provider blew up")
        return json.dumps({"summary": "recovered", "confidence": 0.5, "item_ids": [2]})

    def resolve_model_fn(model_id: str | None) -> dict[str, Any]:
        return _model_def()

    ctx = _ctx(
        params={},
        host=_host(complete_fn=complete_fn, resolve_model_fn=resolve_model_fn),
    )

    parsed = await op._distill_batch(ctx, "prompt", "batch", 100)

    assert parsed.summary == "recovered"
    assert len(calls) == 2
    # Retry payload carries the JSON-only nudge.
    assert any("JSON" in str(calls[1][-1].get("content", "")) for _ in (0,))


@pytest.mark.asyncio
async def test_distill_batch_raises_after_both_attempts_fail() -> None:
    async def complete_fn(messages: list[dict[str, Any]], **_: Any) -> str:
        raise RuntimeError("always broken")

    def resolve_model_fn(model_id: str | None) -> dict[str, Any]:
        return _model_def()

    ctx = _ctx(
        params={},
        host=_host(complete_fn=complete_fn, resolve_model_fn=resolve_model_fn),
    )

    with pytest.raises(RuntimeError, match="distillation failed: always broken"):
        await op._distill_batch(ctx, "prompt", "batch", 100)


# ── _persist_semantic ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_persist_semantic_raises_on_add_error() -> None:
    async def execute_tool_fn(tool_id: str, _ctx: Any) -> ToolResult:
        return ToolResult(tool_id=tool_id, error="store full")

    ctx = _ctx(params=_params(), host=_host(execute_tool_fn=execute_tool_fn))
    parsed = op.Parameters.model_validate(ctx.params)
    consolidated = op.ConsolidatedSemantic(summary="s", confidence=0.5, item_ids=[1])

    with pytest.raises(RuntimeError, match="semantic record storage failed: store full"):
        await op._persist_semantic(ctx, parsed, consolidated, [1], 1)


@pytest.mark.asyncio
async def test_persist_semantic_id_parse_failure_returns_none() -> None:
    """ "Memory stored with ID abc" → non-numeric id fragment → None."""

    async def execute_tool_fn(tool_id: str, _ctx: Any) -> ToolResult:
        return ToolResult(tool_id=tool_id, output_text="Memory stored with ID abc")

    ctx = _ctx(params=_params(), host=_host(execute_tool_fn=execute_tool_fn))
    parsed = op.Parameters.model_validate(ctx.params)
    consolidated = op.ConsolidatedSemantic(summary="s", confidence=0.5, item_ids=[1])

    assert await op._persist_semantic(ctx, parsed, consolidated, [1], 1) is None


@pytest.mark.asyncio
async def test_persist_semantic_salience_uses_floor_over_boost() -> None:
    """max_source_salience is below the 0.8 floor → the floor binds."""
    captured: list[dict[str, Any]] = []

    async def execute_tool_fn(tool_id: str, tool_ctx: Any) -> ToolResult:
        captured.append(dict(getattr(tool_ctx, "parameters", {}) or {}))
        return ToolResult(tool_id=tool_id, output_text="Memory stored with ID 55")

    low_salience = _params([{"id": 1, "content": "x", "created_at": "t", "salience": 0.1}])
    ctx = _ctx(params=low_salience, host=_host(execute_tool_fn=execute_tool_fn))
    parsed = op.Parameters.model_validate(ctx.params)
    consolidated = op.ConsolidatedSemantic(summary="s", confidence=0.5, item_ids=[1])

    semantic_id = await op._persist_semantic(ctx, parsed, consolidated, [1], 1)

    assert semantic_id == 55
    assert captured[0]["salience"] == 0.8


# ── _retire_sources ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_retire_sources_skips_ids_whose_retire_errors() -> None:
    async def execute_tool_fn(tool_id: str, tool_ctx: Any) -> ToolResult:
        params = dict(getattr(tool_ctx, "parameters", {}) or {})
        if params.get("memory_id") == 2:
            return ToolResult(tool_id=tool_id, error="locked")
        return ToolResult(tool_id=tool_id, output_text=f"Retired {params.get('memory_id')}")

    ctx = _ctx(params=_params(), host=_host(execute_tool_fn=execute_tool_fn))
    parsed = op.Parameters.model_validate(ctx.params)

    retired = await op._retire_sources(ctx, parsed, [1, 2, 3])

    assert retired == [1, 3]


# ── execute: guard branches ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_execute_invalid_params_returns_validation_message() -> None:
    ctx = _ctx(params={"episodic_records": "not-a-list"}, host=None)

    result = await op.execute(ctx)

    assert result.output_text.startswith("⛔ Invalid consolidation parameters:")


@pytest.mark.asyncio
async def test_execute_without_host_raises() -> None:
    ctx = _ctx(params=_params(), host=None)

    with pytest.raises(RuntimeError, match="requires the OperationHost seam"):
        await op.execute(ctx)


@pytest.mark.asyncio
async def test_execute_distillation_failure_is_wrapped() -> None:
    async def complete_fn(messages: list[dict[str, Any]], **_: Any) -> str:
        raise RuntimeError("model down")

    ctx = _ctx(
        params=_params(),
        host=_host(
            complete_fn=complete_fn,
            resolve_model_fn=lambda _mid: _model_def(),
            read_authored_text_fn=lambda _path: "stub prompt",
        ),
    )

    with pytest.raises(RuntimeError, match="consolidate_semantic distillation failed"):
        await op.execute(ctx)


@pytest.mark.asyncio
async def test_execute_persist_error_returns_failure_result() -> None:
    async def complete_fn(messages: list[dict[str, Any]], **_: Any) -> str:
        return json.dumps({"summary": "s", "confidence": 0.9, "item_ids": [1]})

    async def execute_tool_fn(tool_id: str, _ctx: Any) -> ToolResult:
        return ToolResult(tool_id=tool_id, error="disk full")

    ctx = _ctx(
        params=_params(),
        host=_host(
            complete_fn=complete_fn,
            resolve_model_fn=lambda _mid: _model_def(),
            read_authored_text_fn=lambda _path: "stub prompt",
            execute_tool_fn=execute_tool_fn,
        ),
    )

    result = await op.execute(ctx)

    assert result.output_text == "⛔ semantic record storage failed: disk full"


# ── execute: happy-path edge branches ─────────────────────────────────


@pytest.mark.asyncio
async def test_execute_hallucinated_item_ids_fall_back_to_full_input_set() -> None:
    """Model returns ids outside the batch → all input ids are used."""

    async def complete_fn(messages: list[dict[str, Any]], **_: Any) -> str:
        # 999 was never in the batch.
        return json.dumps({"summary": "s", "confidence": 0.9, "item_ids": [999]})

    async def execute_tool_fn(tool_id: str, _ctx: Any) -> ToolResult:
        return ToolResult(tool_id=tool_id, output_text="Memory stored with ID 77")

    ctx = _ctx(
        params=_params(_episodic_records(2)),
        host=_host(
            complete_fn=complete_fn,
            resolve_model_fn=lambda _mid: _model_def(),
            read_authored_text_fn=lambda _path: "stub prompt",
            execute_tool_fn=execute_tool_fn,
        ),
    )

    result = await op.execute(ctx)

    assert result.output_text.startswith("✅ Consolidated 2 episodic records")
    assert result.extra is not None
    assert result.extra["item_ids"] == [1, 2]


@pytest.mark.asyncio
async def test_execute_long_summary_preview_gets_ellipsis() -> None:
    long_summary = "x" * (op._PREVIEW_CHARS + 11)

    async def complete_fn(messages: list[dict[str, Any]], **_: Any) -> str:
        return json.dumps({"summary": long_summary, "confidence": 1.0, "item_ids": [1]})

    async def execute_tool_fn(tool_id: str, _ctx: Any) -> ToolResult:
        return ToolResult(tool_id=tool_id, output_text="Memory stored with ID 88")

    ctx = _ctx(
        params=_params(_episodic_records(1)),
        host=_host(
            complete_fn=complete_fn,
            resolve_model_fn=lambda _mid: _model_def(),
            read_authored_text_fn=lambda _path: "stub prompt",
            execute_tool_fn=execute_tool_fn,
        ),
    )

    result = await op.execute(ctx)

    assert "..." in result.output_text


@pytest.mark.asyncio
async def test_execute_retired_ids_preview_switches_to_plus_n_more() -> None:
    """More than ``_PREVIEW_IDS`` retired sources → "+N more" truncation."""
    records = _episodic_records(10)

    async def complete_fn(messages: list[dict[str, Any]], **_: Any) -> str:
        return json.dumps(
            {
                "summary": "cluster of ten",
                "confidence": 0.9,
                "item_ids": [r["id"] for r in records],
            }
        )

    async def execute_tool_fn(tool_id: str, _ctx: Any) -> ToolResult:
        return ToolResult(tool_id=tool_id, output_text="Memory stored with ID 99")

    ctx = _ctx(
        params=_params(records),
        host=_host(
            complete_fn=complete_fn,
            resolve_model_fn=lambda _mid: _model_def(),
            read_authored_text_fn=lambda _path: "stub prompt",
            execute_tool_fn=execute_tool_fn,
        ),
    )

    result = await op.execute(ctx)

    assert "+2 more" in result.output_text
    assert result.extra is not None
    assert len(result.extra["retired_ids"]) == 10


# ── _retired_ids_preview pure branches ────────────────────────────────


def test_retired_ids_preview_empty_returns_empty_string() -> None:
    assert op._retired_ids_preview([]) == ""


def test_retired_ids_preview_short_list_has_no_more_suffix() -> None:
    assert op._retired_ids_preview([1, 2]) == "[1, 2]. "

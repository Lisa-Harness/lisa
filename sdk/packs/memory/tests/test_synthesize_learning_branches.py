"""Branch-coverage tests for the synthesize_and_store_learning operation.

Complements ``test_synthesize_learning_tier.py`` with the remaining branches:

- ``_cfg_int``: None config and unparseable value fall back to the default
- ``_parse_bullet_facts`` / ``_parse_sentence_facts`` / ``_looks_like_json``
  pure branches (skip-empty, cap-at-three, strip quotes, fenced/bare/inner
  JSON signals)
- ``_parse_distilled_learning``: empty output, JSON-guard RuntimeError,
  bullet fallback, sentence fallback, unparseable prose re-raises
- ``_default_learning_content``: query+response join, query-only,
  response-only, empty
- ``_validation_summary``: one-line loc/type rendering
- ``_distill_facts``: host None, first-attempt failure retried at 2× the
  token budget with ``response_format=None``, total failure raises
- ``_build_store_content_json``: had_briefing + thread_turn provenance
- ``_parse_memory_id``: missing marker, non-numeric fragment
- ``execute``: content defaulting, unresolvable model, storage error,
  unparseable memory id → extras omit ``memory_ids``
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from kernel.models import OperationExecutionContext, ToolResult
from kernel.seams.host_adapter import build_callable_host
from pydantic import ValidationError
from sdk.packs.memory.operations.memory import synthesize_learning as op


def _model_def() -> dict[str, Any]:
    return {"model_id": "test-model"}


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


def _ctx(*, params: dict[str, Any], host: Any, **kwargs: Any) -> OperationExecutionContext:
    return OperationExecutionContext(
        operation_id="synthesize_and_store_learning",
        user_slug="adam",
        user_id="adam",
        workspace_id="default",
        params=params,
        host=host,
        **kwargs,
    )


# ── _cfg_int ──────────────────────────────────────────────────────────


def test_cfg_int_none_config_returns_default() -> None:
    assert op._cfg_int(None, "memory_distill_max_tokens", 512) == 512


def test_cfg_int_unparseable_value_returns_default() -> None:
    assert (
        op._cfg_int({"memory_distill_max_tokens": "abc"}, "memory_distill_max_tokens", 512) == 512
    )


def test_resolve_settings_honors_pack_config_override() -> None:
    ctx = _ctx(params={}, host=None, pack_config={"memory_distill_max_tokens": "4096"})

    assert op._resolve_settings(ctx).distill_max_tokens == 4096


# ── _parse_bullet_facts ───────────────────────────────────────────────


def test_parse_bullet_facts_skips_non_bullets_and_blank_facts() -> None:
    text = "- first fact\nplain line\n-   \n- second fact\n- third fact\n- fourth fact"

    facts = op._parse_bullet_facts(text)

    # The blank bullet is skipped; the cap keeps the first three real facts.
    assert facts == ["first fact", "second fact", "third fact"]


def test_parse_bullet_facts_supports_numbered_and_unicode_bullets() -> None:
    assert op._parse_bullet_facts("1. one\n• two\n* three") == ["one", "two", "three"]


# ── _parse_sentence_facts ─────────────────────────────────────────────


def test_parse_sentence_facts_splits_and_strips_quotes() -> None:
    # The quote strip only unquotes chunks that start AND end with a quote,
    # and ``(?<=[.!?])\s+`` does not split before a closing quote ("x." y).
    text = '"A quoted fact." Second fact!\n\nThird fact? Fourth fact.'

    facts = op._parse_sentence_facts(text)

    assert facts == ['A quoted fact." Second fact!', "Third fact?", "Fourth fact."]


def test_parse_sentence_facts_empty_returns_empty() -> None:
    assert op._parse_sentence_facts("   ") == []


# ── _looks_like_json ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("", False),
        ("   ", False),
        ("```\nfenced\n```", True),
        ('{"k": "v"}', True),
        ("[1, 2]", True),
        ("prose with {a} braces", True),
        ("no braces here", False),
    ],
)
def test_looks_like_json(text: str, expected: bool) -> None:
    assert op._looks_like_json(text) is expected


# ── _parse_distilled_learning ─────────────────────────────────────────


def test_parse_distilled_learning_empty_raises_value_error() -> None:
    with pytest.raises(ValueError, match="empty distillation output"):
        op._parse_distilled_learning("")


def test_parse_distilled_learning_rejects_json_shaped_invalid_output() -> None:
    with pytest.raises(RuntimeError, match="refusing to fall back"):
        op._parse_distilled_learning('{"facts": "not-a-list"}')


def test_parse_distilled_learning_falls_back_to_bullets() -> None:
    parsed = op._parse_distilled_learning("- we chose Postgres\n- analytics ships Friday")

    assert parsed.facts == ["we chose Postgres", "analytics ships Friday"]


def test_parse_distilled_learning_falls_back_to_sentences() -> None:
    parsed = op._parse_distilled_learning("We chose Postgres. Analytics ships Friday.")

    assert parsed.facts == ["We chose Postgres.", "Analytics ships Friday."]


def test_parse_distilled_learning_unshapeable_content_reraises() -> None:
    """No JSON, no bullets, no sentences → the original ValidationError."""
    with pytest.raises(ValidationError):
        # Quote-only chunks strip to nothing, so bullets and sentences both
        # come back empty and the original ValidationError propagates.
        op._parse_distilled_learning("'''")


# ── _validation_summary ───────────────────────────────────────────────


def test_validation_summary_renders_loc_and_type() -> None:
    try:
        op.DistilledLearning(facts="not-a-list")
    except ValidationError as exc:
        summary = op._validation_summary(exc)

    assert "facts" in summary
    assert ":" in summary


# ── _default_learning_content ─────────────────────────────────────────


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        (
            {"query": "  what is up?  ", "response": " the answer  "},
            "User asked:\nwhat is up?\n\nLisa answered:\nthe answer",
        ),
        ({"query": "only the query", "response": "  "}, "only the query"),
        ({"query": "  ", "response": "only the response"}, "only the response"),
        ({"query": "  ", "response": ""}, ""),
    ],
)
def test_default_learning_content(kwargs: dict[str, str], expected: str) -> None:
    ctx = _ctx(params={}, host=None, **kwargs)

    assert op._default_learning_content(ctx) == expected


# ── _distill_facts ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_distill_facts_raises_when_host_is_none() -> None:
    ctx = _ctx(params={}, host=None)

    with pytest.raises(RuntimeError, match="requires the OperationHost seam"):
        await op._distill_facts(
            ctx,
            model_def=_model_def(),
            system_prompt="prompt",
            raw_content="content",
            distill_max_tokens=100,
        )


@pytest.mark.asyncio
async def test_distill_facts_retry_doubles_budget_and_clears_response_format() -> None:
    """First attempt fails → retry at 2× max_tokens with response_format=None."""
    calls: list[dict[str, Any]] = []

    async def complete_fn(messages: list[dict[str, Any]], **kwargs: Any) -> str:
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError("empty content from provider")
        return "Facts are:\n- recovered durable fact"

    ctx = _ctx(params={}, host=_host(complete_fn=complete_fn))

    text = await op._distill_facts(
        ctx,
        model_def=_model_def(),
        system_prompt="prompt",
        raw_content="content",
        distill_max_tokens=111,
    )

    assert text == "recovered durable fact"
    assert len(calls) == 2
    assert calls[0]["max_tokens"] == 111
    assert calls[1]["max_tokens"] == 222
    assert calls[0]["response_format"] is op.DistilledLearning
    assert calls[1]["response_format"] is None


@pytest.mark.asyncio
async def test_distill_facts_raises_after_both_attempts_fail() -> None:
    async def complete_fn(messages: list[dict[str, Any]], **_: Any) -> str:
        raise RuntimeError("still broken")

    ctx = _ctx(params={}, host=_host(complete_fn=complete_fn))

    with pytest.raises(RuntimeError, match="distillation failed: still broken"):
        await op._distill_facts(
            ctx,
            model_def=_model_def(),
            system_prompt="prompt",
            raw_content="content",
            distill_max_tokens=100,
        )


# ── _build_store_content_json ─────────────────────────────────────────


def test_build_store_content_json_includes_provenance_when_present() -> None:
    ctx = _ctx(params={}, host=None, query="q", model_id="m")
    object.__setattr__(ctx, "extra", dict(ctx.extra, had_briefing=True, thread_turn=12))

    content_json = op._build_store_content_json(ctx)

    assert content_json == {
        "query": "q",
        "model": "m",
        "had_chase_briefing": True,
        "thread_turn": 12,
    }


def test_build_store_content_json_ignores_non_int_thread_turn() -> None:
    ctx = _ctx(params={}, host=None, query="q")
    object.__setattr__(ctx, "extra", dict(ctx.extra, thread_turn="not-an-int"))

    content_json = op._build_store_content_json(ctx)

    assert "thread_turn" not in content_json


# ── _parse_memory_id ──────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Memory stored with ID 42", 42),
        ("Memory stored with ID 42.", 42),
        ("no marker here", None),
        ("Memory stored with ID abc", None),
    ],
)
def test_parse_memory_id(text: str, expected: int | None) -> None:
    assert op._parse_memory_id(text) is expected


# ── execute guard branches ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_execute_defaults_content_from_query_and_response() -> None:
    """No ``content`` param → the completed exchange is distilled instead."""
    distilled: list[str] = []

    async def complete_fn(messages: list[dict[str, Any]], **_: Any) -> str:
        distilled.append(str(messages[-1].get("content", "")))
        return json.dumps({"facts": ["one durable fact"]})

    async def execute_tool_fn(tool_id: str, _ctx: Any) -> ToolResult:
        return ToolResult(tool_id=tool_id, output_text="Memory stored with ID 3")

    ctx = _ctx(
        params={},
        host=_host(
            complete_fn=complete_fn,
            resolve_model_fn=lambda _mid: _model_def(),
            read_authored_text_fn=lambda _path: "stub prompt",
            execute_tool_fn=execute_tool_fn,
        ),
        query="why postgres",
        response="because durability",
    )

    result = await op.execute(ctx)

    assert "✅ Learning stored" in result.output_text
    assert distilled == ["User asked:\nwhy postgres\n\nLisa answered:\nbecause durability"]
    assert result.extra is not None
    assert result.extra["memory_ids"] == [3]


@pytest.mark.asyncio
async def test_execute_defaults_content_with_query_only_and_silences_model_id() -> None:
    """Query-only content default still distills and stores; the id marker in
    the add reply parses the trailing-period form."""

    async def complete_fn(messages: list[dict[str, Any]], **_: Any) -> str:
        return json.dumps({"facts": ["solo fact"]})

    async def execute_tool_fn(tool_id: str, _ctx: Any) -> ToolResult:
        return ToolResult(tool_id=tool_id, output_text="Memory stored with ID 5.")

    ctx = _ctx(
        params={},
        host=_host(
            complete_fn=complete_fn,
            resolve_model_fn=lambda _mid: _model_def(),
            read_authored_text_fn=lambda _path: "stub prompt",
            execute_tool_fn=execute_tool_fn,
        ),
        query="standalone query",
    )

    result = await op.execute(ctx)

    assert "✅ Learning stored" in result.output_text
    assert result.extra is not None
    assert result.extra["memory_ids"] == [5]


@pytest.mark.asyncio
async def test_execute_raises_when_no_model_resolves() -> None:
    def resolve_model_fn(_model_id: str | None) -> dict[str, Any] | None:
        return None

    ctx = _ctx(
        params={"content": "anything"},
        host=_host(
            resolve_model_fn=resolve_model_fn,
            read_authored_text_fn=lambda _path: "stub prompt",
        ),
    )

    with pytest.raises(RuntimeError, match="could not resolve a model"):
        await op.execute(ctx)


@pytest.mark.asyncio
async def test_execute_storage_error_returns_failure_result() -> None:
    async def complete_fn(messages: list[dict[str, Any]], **_: Any) -> str:
        return json.dumps({"facts": ["a fact"]})

    async def execute_tool_fn(tool_id: str, _ctx: Any) -> ToolResult:
        return ToolResult(tool_id=tool_id, error="backend offline")

    ctx = _ctx(
        params={"content": "anything"},
        host=_host(
            complete_fn=complete_fn,
            resolve_model_fn=lambda _mid: _model_def(),
            read_authored_text_fn=lambda _path: "stub prompt",
            execute_tool_fn=execute_tool_fn,
        ),
    )

    result = await op.execute(ctx)

    assert result.output_text == "⛔ Learning storage failed: backend offline"


@pytest.mark.asyncio
async def test_execute_unparseable_memory_id_omits_memory_ids_extra() -> None:
    """An add reply without a parseable "ID <n>" leaves memory_ids absent."""

    async def complete_fn(messages: list[dict[str, Any]], **_: Any) -> str:
        return json.dumps({"facts": ["a fact"]})

    async def execute_tool_fn(tool_id: str, _ctx: Any) -> ToolResult:
        return ToolResult(tool_id=tool_id, output_text="stored, no id marker")

    ctx = _ctx(
        params={"content": "anything"},
        host=_host(
            complete_fn=complete_fn,
            resolve_model_fn=lambda _mid: _model_def(),
            read_authored_text_fn=lambda _path: "stub prompt",
            execute_tool_fn=execute_tool_fn,
        ),
    )

    result = await op.execute(ctx)

    assert "✅ Learning stored" in result.output_text
    assert result.extra is not None
    assert "memory_ids" not in result.extra
    assert result.extra["recorded"] is True

"""Tests for the checkpoint pack operations.

Covers:
- ``checkpoint_create`` with model enrichment (mocked complete_fn)
- ``checkpoint_create`` raises when complete_fn is not wired (no silent bare bundle)
- ``checkpoint_create`` with no tool executor
- ``checkpoint_restore`` successful load
- ``checkpoint_restore`` missing checkpoint
- ``checkpoint_restore`` no tool executor
- ``CheckpointBundle`` enrichment fields round-trip
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

from kernel.models import (
    CheckpointBundle,
    OperationExecutionContext,
    ToolResult,
)


# ── Test helpers ──────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _StubCompletion:
    """Fake model completion for tests."""

    content: str
    tool_calls: list[dict[str, Any]] | None = None


def _make_ctx(
    *,
    params: dict[str, Any] | None = None,
    complete_fn: Any = None,
    resolve_model_fn: Any = None,
    execute_tool_fn: Any = None,
    state_provider: Any = None,
    read_authored_text_fn: Any = None,
) -> OperationExecutionContext:
    """Build a minimal OperationExecutionContext for checkpoint operations."""
    authored_text = read_authored_text_fn or (lambda _path: "Test authored prompt.")
    return OperationExecutionContext(
        workspace_id="test",
        user_id="test-user",
        user_slug="test-user",
        surface_type="personal_dm",
        surface_id="test",
        memory_scope="personal",
        mode_id="default",
        query="test query",
        complete_fn=complete_fn,
        resolve_model_fn=resolve_model_fn,
        read_authored_text_fn=authored_text,
        execute_tool_fn=execute_tool_fn,
        state_provider=state_provider,
        params=params or {},
    )


def _mock_tool_executor(return_text: str, is_error: bool = False) -> Callable[[str, Any], Awaitable[ToolResult]]:
    """Build a mock execute_tool_fn that returns the given text."""

    async def _executor(tool_id: str, _ctx: Any) -> ToolResult:  # noqa: ARG001
        return ToolResult(tool_id=tool_id, output_text=return_text, error="" if not is_error else return_text)

    return _executor


def _mock_complete(enrichment_json: str) -> Callable[..., Awaitable[_StubCompletion]]:
    """Build a mock complete_fn that returns enrichment JSON."""

    async def _complete(**_kwargs: Any) -> _StubCompletion:
        return _StubCompletion(content=enrichment_json)

    return _complete


def _mock_resolve_model() -> Callable[[str], Any]:
    """Build a mock resolve_model_fn that returns a stub model def."""
    stub = MagicMock()
    stub.model_id = "test-model"

    def _resolve(_alias: str) -> Any:  # noqa: ARG001
        return stub

    return _resolve


# ── CheckpointBundle enrichment fields ────────────────────────────────


def test_checkpoint_bundle_enrichment_fields() -> None:
    """CheckpointBundle carries the enrichment fields with defaults."""
    bundle = CheckpointBundle(
        checkpoint_id="test",
        created_at="2026-07-12T00:00:00Z",
    )
    assert bundle.overview == ""
    assert bundle.work_done == ""
    assert bundle.important_files == []
    assert bundle.next_steps == []


def test_checkpoint_bundle_enrichment_roundtrip() -> None:
    """Enrichment fields survive serialization round-trip."""
    bundle = CheckpointBundle(
        checkpoint_id="test",
        created_at="2026-07-12T00:00:00Z",
        label="my-checkpoint",
        overview="Built a checkpoint pack",
        work_done="Created checkpoint_create and checkpoint_restore operations",
        important_files=["sdk/packs/checkpoint/operations/checkpoint_create.py"],
        next_steps=["Wire pack into host", "Write tests"],
        turn_history=[{"query": "hi", "response_text": "hello"}],
    )
    dumped = bundle.model_dump(mode="json")
    restored = CheckpointBundle.model_validate(dumped)
    assert restored.overview == "Built a checkpoint pack"
    assert restored.work_done == "Created checkpoint_create and checkpoint_restore operations"
    assert restored.important_files == ["sdk/packs/checkpoint/operations/checkpoint_create.py"]
    assert restored.next_steps == ["Wire pack into host", "Write tests"]
    assert len(restored.turn_history) == 1


# ── checkpoint_create ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_checkpoint_create_with_enrichment() -> None:
    """checkpoint_create calls the model for enrichment and saves to lisa-storage."""
    from sdk.packs.checkpoint.operations.checkpoint_create import execute

    enrichment_json = json.dumps({
        "overview": "User tested the checkpoint feature",
        "work_done": "Ran /checkpoint first_try successfully",
        "important_files": ["hosts/little_lisa/main.py"],
        "next_steps": ["Try /restore", "Test automatic checkpoints"],
    })
    save_response = json.dumps({"checkpoint_id": "abc123"})

    ctx = _make_ctx(
        params={
            "label": "test-label",
            "turn_history": [
                {"query": "hello", "response_text": "hi there"},
                {"query": "/checkpoint", "response_text": "saved"},
            ],
        },
        complete_fn=_mock_complete(enrichment_json),
        resolve_model_fn=_mock_resolve_model(),
        execute_tool_fn=_mock_tool_executor(save_response),
    )

    result = await execute(ctx)

    assert result.output_text
    assert "abc123" in result.output_text
    assert result.output_json.get("checkpoint_id") == "abc123"
    assert result.extra.get("enriched") is True
    assert result.extra.get("turn_count") == 2
    assert "hosts/little_lisa/main.py" in result.extra.get("important_files", [])


@pytest.mark.asyncio
async def test_checkpoint_create_without_model_raises() -> None:
    """checkpoint_create raises when complete_fn is not wired (no silent bare bundle)."""
    from sdk.packs.checkpoint.operations.checkpoint_create import execute

    save_response = json.dumps({"checkpoint_id": "def456"})

    ctx = _make_ctx(
        params={
            "label": "bare",
            "turn_history": [{"query": "hi", "response_text": "hello"}],
        },
        complete_fn=None,
        resolve_model_fn=None,
        execute_tool_fn=_mock_tool_executor(save_response),
    )

    with pytest.raises(RuntimeError, match="complete_fn and resolve_model_fn"):
        await execute(ctx)


@pytest.mark.asyncio
async def test_checkpoint_create_uses_prompt_override() -> None:
    """checkpoint_create prefers an explicit NL prompt over authored text."""
    from sdk.packs.checkpoint.operations.checkpoint_create import execute

    captured_system_prompts: list[str] = []

    async def _capture_complete(**kwargs: Any) -> _StubCompletion:
        messages = kwargs.get("messages") or []
        captured_system_prompts.append(str(messages[0]["content"]))
        return _StubCompletion(
            content=json.dumps(
                {
                    "overview": "override used",
                    "work_done": "done",
                    "important_files": [],
                    "next_steps": [],
                }
            )
        )

    ctx = _make_ctx(
        params={
            "turn_history": [{"query": "hello", "response_text": "hi"}],
            "prompt": "Use this custom checkpoint prompt.",
        },
        complete_fn=_capture_complete,
        resolve_model_fn=_mock_resolve_model(),
        execute_tool_fn=_mock_tool_executor(json.dumps({"checkpoint_id": "abc123"})),
        read_authored_text_fn=lambda _path: "ignored authored prompt",
    )

    await execute(ctx)

    assert captured_system_prompts == ["Use this custom checkpoint prompt."]


@pytest.mark.asyncio
async def test_checkpoint_create_no_tool_executor() -> None:
    """checkpoint_create returns an error when no tool executor is wired."""
    from sdk.packs.checkpoint.operations.checkpoint_create import execute

    enrichment_json = json.dumps({"overview": "x", "work_done": "y", "important_files": [], "next_steps": []})
    ctx = _make_ctx(
        params={"label": "no-tools", "turn_history": [{"query": "q", "response_text": "a"}]},
        complete_fn=_mock_complete(enrichment_json),
        resolve_model_fn=_mock_resolve_model(),
        execute_tool_fn=None,
    )

    result = await execute(ctx)
    assert "failed" in result.output_text.lower() or "not injected" in result.output_text.lower()


@pytest.mark.asyncio
async def test_checkpoint_create_tool_error() -> None:
    """checkpoint_create surfaces tool errors from lisa-storage."""
    from sdk.packs.checkpoint.operations.checkpoint_create import execute

    enrichment_json = json.dumps({"overview": "x", "work_done": "y", "important_files": [], "next_steps": []})
    ctx = _make_ctx(
        params={"label": "error-test", "turn_history": [{"query": "q", "response_text": "a"}]},
        complete_fn=_mock_complete(enrichment_json),
        resolve_model_fn=_mock_resolve_model(),
        execute_tool_fn=_mock_tool_executor("server is read-only", is_error=True),
    )

    result = await execute(ctx)
    assert "failed" in result.output_text.lower()


@pytest.mark.asyncio
async def test_checkpoint_history_summarize_with_model() -> None:
    """checkpoint_history_summarize condenses old turns and keeps recent turns."""
    from sdk.packs.checkpoint.operations.checkpoint_history_summarize import execute

    summary_json = json.dumps({
        "condensed_narrative": "Earlier turns selected a checkpoint strategy.",
        "key_points": ["Use checkpoint pack", "Keep recent turns verbatim"],
    })
    turn_history = [
        {"query": "q1", "response_text": "a1"},
        {"query": "q2", "response_text": "a2"},
        {"query": "q3", "response_text": "a3"},
        {"query": "q4", "response_text": "a4"},
    ]

    ctx = _make_ctx(
        params={"turn_history": turn_history, "retention_count": 2},
        complete_fn=_mock_complete(summary_json),
        resolve_model_fn=_mock_resolve_model(),
        read_authored_text_fn=lambda _path: "Authored summarize prompt.",
    )

    result = await execute(ctx)

    assert "Condensed 2 turn(s)" in result.output_text
    output_history = result.output_json.get("turn_history")
    assert isinstance(output_history, list)
    assert len(output_history) == 3
    assert output_history[0]["source"] == "checkpoint_history_summarize"
    assert output_history[1]["query"] == "q3"
    assert output_history[2]["query"] == "q4"
    assert result.output_json.get("was_summarized") is True


@pytest.mark.asyncio
async def test_checkpoint_history_summarize_reads_authored_prompt() -> None:
    """checkpoint_history_summarize uses the authored-text seam when no override is given."""
    from sdk.packs.checkpoint.operations.checkpoint_history_summarize import execute

    captured_system_prompts: list[str] = []

    async def _capture_complete(**kwargs: Any) -> _StubCompletion:
        messages = kwargs.get("messages") or []
        captured_system_prompts.append(str(messages[0]["content"]))
        return _StubCompletion(
            content=json.dumps(
                {
                    "condensed_narrative": "Earlier turns selected a checkpoint strategy.",
                    "key_points": ["Use checkpoint pack"],
                }
            )
        )

    ctx = _make_ctx(
        params={
            "turn_history": [
                {"query": "q1", "response_text": "a1"},
                {"query": "q2", "response_text": "a2"},
            ],
            "retention_count": 1,
        },
        complete_fn=_capture_complete,
        resolve_model_fn=_mock_resolve_model(),
        read_authored_text_fn=lambda _path: "Authored summarize prompt.",
    )

    await execute(ctx)

    assert captured_system_prompts == ["Authored summarize prompt."]


@pytest.mark.asyncio
async def test_checkpoint_history_summarize_noop_when_under_retention() -> None:
    """checkpoint_history_summarize no-ops when history length is within retention."""
    from sdk.packs.checkpoint.operations.checkpoint_history_summarize import execute

    turn_history = [{"query": "q1", "response_text": "a1"}]
    ctx = _make_ctx(params={"turn_history": turn_history, "retention_count": 3})

    result = await execute(ctx)

    assert "no summarization needed" in result.output_text.lower()
    assert result.output_json.get("was_summarized") is False
    assert result.output_json.get("turn_history") == turn_history


@pytest.mark.asyncio
async def test_checkpoint_history_summarize_raises_on_invalid_model_payload() -> None:
    """checkpoint_history_summarize raises (no deterministic fallback) on parse failure."""
    from sdk.packs.checkpoint.operations.checkpoint_history_summarize import execute

    turn_history = [
        {"query": "q1", "response_text": "a1"},
        {"query": "q2", "response_text": "a2"},
        {"query": "q3", "response_text": "a3"},
    ]

    ctx = _make_ctx(
        params={"turn_history": turn_history, "retention_count": 1},
        complete_fn=_mock_complete("not-json"),
        resolve_model_fn=_mock_resolve_model(),
    )

    with pytest.raises(RuntimeError, match="structured output parse failed"):
        await execute(ctx)


# ── checkpoint_restore ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_checkpoint_restore_success() -> None:
    """checkpoint_restore loads a bundle and returns the summary."""
    from sdk.packs.checkpoint.operations.checkpoint_restore import execute

    bundle = {
        "checkpoint_id": "abc123",
        "label": "test-label",
        "mode_id": "default",
        "stage_id": "gather",
        "turn_count": 5,
        "overview": "Test session",
        "work_done": "Built checkpoints",
        "important_files": ["main.py"],
        "next_steps": ["Run more tests"],
    }

    ctx = _make_ctx(
        params={"checkpoint_id": "abc123"},
        execute_tool_fn=_mock_tool_executor(json.dumps(bundle)),
    )

    result = await execute(ctx)

    assert "abc123" in result.output_text
    assert "test-label" in result.output_text
    assert "Test session" in result.output_text
    assert result.output_json.get("checkpoint_id") == "abc123"
    assert result.output_json.get("turn_count") == 5
    assert result.extra.get("has_overview") is True


@pytest.mark.asyncio
async def test_checkpoint_restore_missing() -> None:
    """checkpoint_restore surfaces error for missing checkpoint."""
    from sdk.packs.checkpoint.operations.checkpoint_restore import execute

    ctx = _make_ctx(
        params={"checkpoint_id": "nonexistent"},
        execute_tool_fn=_mock_tool_executor(
            json.dumps({"error": "checkpoint not found: nonexistent"}),
            is_error=True,
        ),
    )

    result = await execute(ctx)
    assert "failed" in result.output_text.lower()


@pytest.mark.asyncio
async def test_checkpoint_restore_no_id() -> None:
    """checkpoint_restore returns error when no checkpoint_id is provided."""
    from sdk.packs.checkpoint.operations.checkpoint_restore import execute

    ctx = _make_ctx(
        params={},
        execute_tool_fn=_mock_tool_executor("{}"),
    )

    result = await execute(ctx)
    assert "no checkpoint_id" in result.output_text.lower()
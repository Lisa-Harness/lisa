"""Tests for TurnResult.per_model_usage population in kernel.turn_runner.

These tests live in ``sdk/tests/`` because ``kernel/tests/conftest.py`` has an
autouse module-scope fixture that spawns the real ``lisa-catalog`` MCP server
over external packs (``lisa_packs/{search,markdown,compaction,letta}``). When
those submodules are not installed, every kernel test errors at setup.
``sdk/tests/`` has no such conftest, so we exercise ``run_turn`` with stubs
and patch ``get_workflow_catalog`` directly.

See ``/memories/repo/kernel-tests-catalog-fixture-requires-external-packs.md``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from kernel.models import WorkflowModeConfig
from kernel.provider_core import StreamedChunk
from kernel.turn_runner import TurnHooks, TurnResult, run_turn


# ── Stubs ─────────────────────────────────────────────────────────────


class _StubModelDef:
    """Minimal model def: the turn loop only reads ``model_id``."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.provider = "test"


class _StubUsage:
    def __init__(self, prompt: int, completion: int) -> None:
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = prompt + completion
        self.cached_tokens = 0
        self.reasoning_tokens = 0


class _StubCompletion:
    """Mimics the completion object the turn loop consumes."""

    def __init__(
        self,
        content: str,
        model_id: str,
        prompt: int,
        completion: int,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls or []
        self.model_id = model_id
        self.usage = _StubUsage(prompt, completion)


def _default_mode_config() -> WorkflowModeConfig:
    return WorkflowModeConfig(
        mode_id="default",
        start_stage_id="none",
        mode_kind="chat",
        routable=False,
        satisfies_intents=[],
        required_capabilities=[],
        routing_intent_tags=[],
        routing_intent_keywords={},
    )


def _patched_catalog():
    """Patch get_workflow_catalog in both turn_runner and ipsar_fsm.

    ``run_turn`` reads the catalog via ``kernel.turn_runner.get_workflow_catalog``
    (re-exported) and via ``kernel.ipsar_fsm.get_workflow_catalog`` inside
    ``run_mode_pre_turn_operations`` / ``run_mode_entry_operations``. Both
    must return the stubbed default-mode config or the lazy catalog loader
    raises because no content MCP config is registered.
    """
    catalog = {"default": _default_mode_config()}
    from contextlib import ExitStack

    stack = ExitStack()
    stack.enter_context(patch("kernel.turn_runner.get_workflow_catalog", return_value=catalog))
    stack.enter_context(patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=catalog))
    return stack


# ── Tests ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_turn_populates_per_model_usage_single_model() -> None:
    """A simple single-model turn produces one per_model_usage entry."""
    completion = _StubCompletion("Hello", "default-model", prompt=100, completion=50)

    async def complete(**kwargs: Any) -> _StubCompletion:
        return completion

    hooks = TurnHooks(
        complete=complete,
        resolve_model=lambda mid: _StubModelDef("default-model"),
    )

    with _patched_catalog():
        result = await run_turn(mode_id="default", query="hello", hooks=hooks)

    assert isinstance(result, TurnResult)
    assert result.prompt_tokens == 100
    assert result.completion_tokens == 50
    assert result.total_tokens == 150
    assert len(result.per_model_usage) == 1
    entry = result.per_model_usage[0]
    assert entry["model_id"] == "default-model"
    assert entry["prompt_tokens"] == 100
    assert entry["completion_tokens"] == 50
    assert entry["total_tokens"] == 150


@pytest.mark.asyncio
async def test_run_turn_records_per_model_usage_for_streaming_path() -> None:
    """The streaming completion path populates per_model_usage too."""

    async def complete_stream(**kwargs: Any):
        yield StreamedChunk(content="Hello", usage=None)
        yield StreamedChunk(
            content="",
            usage=_StubUsage(prompt=60, completion=30),
            finish_reason="stop",
        )

    hooks = TurnHooks(
        complete_stream=complete_stream,
        resolve_model=lambda mid: _StubModelDef("stream-model"),
    )

    with _patched_catalog():
        result = await run_turn(mode_id="default", query="hello", hooks=hooks)

    assert result.prompt_tokens == 60
    assert result.completion_tokens == 30
    assert len(result.per_model_usage) == 1
    entry = result.per_model_usage[0]
    assert entry["model_id"] == "stream-model"
    assert entry["total_tokens"] == 90


def test_turn_result_per_model_usage_defaults_empty() -> None:
    """TurnResult.per_model_usage defaults to an empty list."""
    result = TurnResult(response_text="ok")
    assert result.per_model_usage == []
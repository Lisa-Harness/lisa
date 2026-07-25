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


# ── Fork output_text visibility tests ─────────────────────────────────


@pytest.mark.asyncio
async def test_fork_parent_sees_child_output() -> None:
    """The parent model's post-fork tool message includes each child's
    output_text so the parent can produce an accurate summary.

    Regression: the success-case tool_output previously said only
    "Results are in stage_data.fork_results", giving the parent model
    zero visibility into what the children actually produced. The parent
    would hallucinate filenames — often the same wrong name for both.
    """
    call_count = 0
    parent_messages: list[list[dict[str, Any]]] = []

    async def two_step(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _StubCompletion(
                content="",
                model_id="parent-model",
                prompt=10,
                completion=0,
                tool_calls=[
                    {
                        "id": "call_1",
                        "function": {
                            "name": "transition_stage",
                            "arguments": '{"target_stage_id": "fork:default,default"}',
                        },
                    }
                ],
            )
        # Capture the messages the parent model sees on its second call
        # (after the fork tool result is appended).
        parent_messages.append(list(kwargs.get("messages", [])))
        return _StubCompletion(
            content="Synthesized from fork results.",
            model_id="parent-model",
            prompt=20,
            completion=10,
        )

    hooks = TurnHooks(
        complete=two_step,
        resolve_model=lambda mid: _StubModelDef("parent-model"),
    )

    with _patched_catalog():
        result = await run_turn(mode_id="default", query="research and write", hooks=hooks)

    assert result.response_text == "Synthesized from fork results."
    assert result.fork_results is not None
    assert len(result.fork_results) == 2

    # The parent model's second call must include a tool message
    # containing each child's output_text.  Note: fork children also
    # call complete (they run run_turn with the same hooks), so
    # parent_messages has entries for children + the parent's final
    # call.  The last entry is the parent's post-fork call.
    assert len(parent_messages) >= 1
    final_messages = parent_messages[-1]
    tool_msgs = [m for m in final_messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    tool_content = tool_msgs[0]["content"]
    assert isinstance(tool_content, str)
    # Must contain each child's mode_id and output_text
    for child in result.fork_results:
        if child.status == "ok":
            assert child.target.mode_id in tool_content
            assert child.output_text in tool_content
    # Must NOT contain the old opaque message
    assert "stage_data.fork_results" not in tool_content


@pytest.mark.asyncio
async def test_fork_child_output_text_populated() -> None:
    """Fork children that complete ok have non-empty output_text."""
    call_count = 0

    async def two_step(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _StubCompletion(
                content="",
                model_id="parent-model",
                prompt=10,
                completion=0,
                tool_calls=[
                    {
                        "id": "call_1",
                        "function": {
                            "name": "transition_stage",
                            "arguments": '{"target_stage_id": "fork:default,default"}',
                        },
                    }
                ],
            )
        return _StubCompletion(
            content="Synthesized from fork results.",
            model_id="parent-model",
            prompt=20,
            completion=10,
        )

    hooks = TurnHooks(
        complete=two_step,
        resolve_model=lambda mid: _StubModelDef("parent-model"),
    )

    with _patched_catalog():
        result = await run_turn(mode_id="default", query="research and write", hooks=hooks)

    assert result.fork_results is not None
    assert len(result.fork_results) == 2
    for child in result.fork_results:
        assert child.status in ("ok", "error", "timeout", "cancelled", "no_tools")
        if child.status == "ok":
            assert child.output_text != ""
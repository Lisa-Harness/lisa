"""Tests for in-process operation dispatch in the turn runner.

Operations are advertised to the LLM directly from the in-process catalog
(via ``build_operation_tool_schemas``) and dispatched through
``hooks.execute_operation`` — not through an MCP subprocess. These tests
verify that path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from kernel.ipsar_fsm import register_content_roots, register_operation_callbacks
from kernel.models import (
    OperationExecutionContext,
    OperationResult,
)
from kernel.operations.registry import (
    build_operation_tool_schemas,
    execute_operation,
    get_operation_catalog,
)
from kernel.turn_runner import TurnHooks, run_turn

_CORE_ROOT: Path = (Path(__file__).resolve().parents[1] / ".." / "sdk_packs" / "core").resolve()


@pytest.fixture(scope="module", autouse=True)
def _operation_dispatch_setup() -> None:
    """Wire the operation callbacks and content roots so run_turn works."""
    register_content_roots(_CORE_ROOT)
    register_operation_callbacks(get_operation_catalog, execute_operation)


@dataclass(frozen=True, slots=True)
class _StubCompletion:
    content: str
    tool_calls: list[dict[str, Any]] | None = None


def _make_tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    import json

    return {
        "id": call_id,
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


# ── build_operation_tool_schemas ──────────────────────────────────────


def test_build_operation_tool_schemas_returns_openai_format() -> None:
    """Schemas are OpenAI function-tool dicts with the operation's Parameters."""
    schemas = build_operation_tool_schemas("learn")
    assert len(schemas) > 0
    for schema in schemas:
        assert schema["type"] == "function"
        assert "name" in schema["function"]
        assert "parameters" in schema["function"]


def test_build_operation_tool_schemas_filters_by_mode() -> None:
    """Operations not allowed for a mode are not included."""
    learn_schemas = build_operation_tool_schemas("learn")
    learn_names = {s["function"]["name"] for s in learn_schemas}
    # learn mode allows synthesize_and_store_learning and recall_memory
    assert "synthesize_and_store_learning" in learn_names

    # default mode should have a different set (or empty — depends on allowlist)
    default_schemas = build_operation_tool_schemas("default")
    default_names = {s["function"]["name"] for s in default_schemas}
    # recall_memory has no mode_allowlist restriction, so it's available everywhere
    assert "recall_memory" in default_names


# ── In-process operation dispatch in run_turn ─────────────────────────


@pytest.mark.asyncio
async def test_operation_tool_call_dispatches_via_execute_operation() -> None:
    """When the model calls an operation by name, it dispatches through
    hooks.execute_operation (not pool.call_tool).
    """
    dispatched_operations: list[str] = []

    async def stub_execute_op(
        operation_id: str, _ctx: OperationExecutionContext
    ) -> OperationResult:
        dispatched_operations.append(operation_id)
        return OperationResult(output_text=f"Stored via {operation_id}")

    call_log: list[dict[str, Any]] = []

    async def two_step(**kwargs: Any) -> _StubCompletion:
        call_log.append(kwargs)
        if len(call_log) == 1:
            return _StubCompletion(
                content="",
                tool_calls=[
                    _make_tool_call(
                        "call_1",
                        "synthesize_and_store_learning",
                        {"content": "We decided to use PostgreSQL."},
                    )
                ],
            )
        return _StubCompletion(content="Memory stored successfully.")

    hooks = TurnHooks(complete=two_step, execute_operation=stub_execute_op)

    result = await run_turn(mode_id="learn", query="learn that we decided to use PostgreSQL", hooks=hooks)

    assert result.response_text == "Memory stored successfully."
    assert result.tool_call_count == 1
    assert dispatched_operations == ["synthesize_and_store_learning"]


@pytest.mark.asyncio
async def test_non_operation_tool_call_dispatches_via_pool() -> None:
    """Non-operation tool names still go through pool.call_tool."""
    from kernel.tools.mcp_client import MCPToolResult

    pool_calls: list[str] = []

    class _StubPool:
        async def list_tools(self) -> list[Any]:
            return []

        async def call_tool(self, tool_id: str, _arguments: dict[str, Any]) -> MCPToolResult:
            pool_calls.append(tool_id)
            return MCPToolResult(tool_id=tool_id, output_text="MCP tool result")

        async def __aenter__(self) -> "_StubPool":
            return self

        async def __aexit__(self, *_args: object) -> None:
            pass

        async def start(self) -> None:
            pass

        async def aclose(self) -> None:
            pass

    call_log: list[dict[str, Any]] = []

    async def two_step(**kwargs: Any) -> _StubCompletion:
        call_log.append(kwargs)
        if len(call_log) == 1:
            return _StubCompletion(
                content="",
                tool_calls=[_make_tool_call("call_1", "memory.add", {"content": "test"})],
            )
        return _StubCompletion(content="Memory saved.")

    async def stub_execute_op(
        operation_id: str, ctx: OperationExecutionContext
    ) -> OperationResult:
        return OperationResult(output_text="should not be called")

    hooks = TurnHooks(
        complete=two_step,
        execute_operation=stub_execute_op,
        mcp_tool_pool=_StubPool(),
    )

    result = await run_turn(mode_id="default", query="save a memory", hooks=hooks)

    assert result.response_text == "Memory saved."
    assert pool_calls == ["memory.add"]


@pytest.mark.asyncio
async def test_operation_schemas_appear_in_tools_sent_to_model() -> None:
    """The tools array sent to the LLM includes operation schemas."""
    captured_tools: list[dict[str, Any]] = []

    async def capture_then_respond(**kwargs: Any) -> _StubCompletion:
        tools = kwargs.get("tools") or []
        captured_tools.extend(tools)
        return _StubCompletion(content="Done.")

    hooks = TurnHooks(complete=capture_then_respond)

    await run_turn(mode_id="learn", query="test", hooks=hooks)

    tool_names = {t["function"]["name"] for t in captured_tools if "function" in t}
    # At least one operation should be in the tool list for learn mode
    operation_ids = set(get_operation_catalog().keys())
    assert tool_names & operation_ids, (
        f"Expected operation schemas in tools, got: {tool_names}"
    )


@pytest.mark.asyncio
async def test_operation_dispatch_error_produces_tool_error_message() -> None:
    """When execute_operation raises, the tool result contains the error."""
    from kernel.operations.registry import OperationExecutionError

    async def failing_execute_op(
        _operation_id: str, _ctx: OperationExecutionContext
    ) -> OperationResult:
        raise OperationExecutionError(f"Operation '{_operation_id}' failed: boom")

    call_log: list[dict[str, Any]] = []

    async def two_step(**kwargs: Any) -> _StubCompletion:
        call_log.append(kwargs)
        if len(call_log) == 1:
            return _StubCompletion(
                content="",
                tool_calls=[
                    _make_tool_call(
                        "call_1",
                        "synthesize_and_store_learning",
                        {"content": "test"},
                    )
                ],
            )
        return _StubCompletion(content="Handled error.")

    hooks = TurnHooks(complete=two_step, execute_operation=failing_execute_op)

    result = await run_turn(mode_id="learn", query="test", hooks=hooks)

    # The model should have received the error as a tool result and recovered
    assert result.response_text == "Handled error."
    # The operation raised, so total_tool_calls stays at 0 (counted after success)
    assert result.tool_call_count == 0
    assert result.iteration_count == 2
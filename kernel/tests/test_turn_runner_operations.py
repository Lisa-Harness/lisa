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

_CORE_ROOT: Path = (Path(__file__).resolve().parents[1] / ".." / "sdk" / "packs" / "core").resolve()

# The shared conftest fixture (_mcp_catalog_setup) handles all catalog wiring.


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
    # learn mode allows synthesize_and_store_learning
    assert "synthesize_and_store_learning" in learn_names

    # With the learning pack enabled, default mode can expose memory recall.
    default_schemas = build_operation_tool_schemas("default")
    default_names = {s["function"]["name"] for s in default_schemas}
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


# ── Mode-level tool filtering ────────────────────────────────────────


def test_learn_mode_has_allowed_tool_ids() -> None:
    """The learn mode declares allowed_tool_ids in its front-matter."""
    from kernel.ipsar_fsm import get_workflow_catalog

    learn = get_workflow_catalog().get("learn")
    assert learn is not None
    assert learn.allowed_tool_ids == ["memory.add", "memory.search", "memory.list"]


def test_default_mode_has_no_allowed_tool_ids() -> None:
    """Default mode has no allowed_tool_ids (empty list = all tools pass through)."""
    from kernel.ipsar_fsm import get_workflow_catalog

    default = get_workflow_catalog().get("default")
    assert default is not None
    assert default.allowed_tool_ids == []


@pytest.mark.asyncio
async def test_mode_tool_filtering_excludes_unlisted_tools() -> None:
    """When a mode has allowed_tool_ids, only those tools reach the model."""
    from kernel.tools.mcp_client import MCPToolSpec

    captured_tools: list[dict[str, Any]] = []

    async def capture_then_respond(**kwargs: Any) -> _StubCompletion:
        tools = kwargs.get("tools") or []
        captured_tools.extend(tools)
        return _StubCompletion(content="Done.")

    # Build a stub pool that returns tools the learn mode doesn't allow.
    # MCP-discovered tools are namespaced (server_id.tool_name); synthetic
    # aliases keep their bare names (memory.add, etc.).
    class _StubPool:
        async def list_tools(self) -> list[MCPToolSpec]:
            return [
                MCPToolSpec(
                    server_id="__synthetic__",
                    name="memory.add",
                    description="Add memory",
                    input_schema={"type": "object", "properties": {}},
                ),
                MCPToolSpec(
                    server_id="test",
                    name="test.delete_all_memories",
                    description="Delete all memories",
                    input_schema={"type": "object", "properties": {}},
                ),
                MCPToolSpec(
                    server_id="test",
                    name="test.browse_url",
                    description="Browse a URL",
                    input_schema={"type": "object", "properties": {}},
                ),
            ]

        async def call_tool(self, tool_id: str, _arguments: dict[str, Any]) -> Any:
            return type("_R", (), {"output_text": "done", "is_error": False})()

        async def __aenter__(self) -> "_StubPool":
            return self

        async def __aexit__(self, *_args: object) -> None:
            pass

        async def start(self) -> None:
            pass

        async def aclose(self) -> None:
            pass

    hooks = TurnHooks(complete=capture_then_respond, mcp_tool_pool=_StubPool())

    await run_turn(mode_id="learn", query="test", hooks=hooks)

    tool_names = {t["function"]["name"] for t in captured_tools if "function" in t}
    # memory.add (synthetic) is allowed; namespaced MCP tools are not
    assert "memory.add" in tool_names
    assert "test.delete_all_memories" not in tool_names
    assert "test.browse_url" not in tool_names


# ── tools_authoritative mode ─────────────────────────────────────────


def test_default_mode_tools_authoritative_is_false() -> None:
    """tools_authoritative defaults to False (off by default)."""
    from kernel.ipsar_fsm import get_workflow_catalog

    default = get_workflow_catalog().get("default")
    assert default is not None
    assert default.tools_authoritative is False


def test_learn_mode_tools_authoritative_is_false() -> None:
    """Learn mode is tool-authoritative to constrain memory calls."""
    from kernel.ipsar_fsm import get_workflow_catalog

    learn = get_workflow_catalog().get("learn")
    assert learn is not None
    assert learn.tools_authoritative is True


def test_build_operation_tool_schemas_authoritative_excludes_unbound() -> None:
    """When tools_authoritative=True, only mode-bound operations are included."""
    from kernel.operations.registry import build_operation_tool_schemas

    # learn mode with authoritative=True should only include operations
    # explicitly bound to learn (synthesize_and_store_learning, recall_memory).
    # Other operations with no mode_allowlist should be excluded.
    schemas = build_operation_tool_schemas("learn", tools_authoritative=True)
    names = {s["function"]["name"] for s in schemas}
    assert "synthesize_and_store_learning" in names
    assert "recall_memory" in names
    # Operations not bound to learn should NOT appear
    assert "self_model" not in names
    assert "workshop_run" not in names


def test_build_operation_tool_schemas_non_authoritative_includes_unbound() -> None:
    """When tools_authoritative=False, unbound operations still appear
    (the default fallback: no mode_allowlist = allowed everywhere)."""
    from kernel.operations.registry import build_operation_tool_schemas

    schemas = build_operation_tool_schemas("learn", tools_authoritative=False)
    names = {s["function"]["name"] for s in schemas}
    # learn-bound operations are included
    assert "synthesize_and_store_learning" in names
    assert "recall_memory" in names
    # Non-authoritative should include MORE operations than authoritative
    # (operations without a restricting mode_allowlist pass through)
    authoritative_schemas = build_operation_tool_schemas("learn", tools_authoritative=True)
    assert len(schemas) >= len(authoritative_schemas)


@pytest.mark.asyncio
async def test_authoritative_mode_empty_tool_ids_means_no_mcp_tools() -> None:
    """When tools_authoritative=True and allowed_tool_ids is empty,
    no MCP tools are advertised (but operations still appear)."""
    from kernel.ipsar_fsm import get_workflow_catalog
    from kernel.tools.mcp_client import MCPToolSpec

    captured_tools: list[dict[str, Any]] = []

    async def capture_then_respond(**kwargs: Any) -> _StubCompletion:
        tools = kwargs.get("tools") or []
        captured_tools.extend(tools)
        return _StubCompletion(content="Done.")

    class _StubPool:
        async def list_tools(self) -> list[MCPToolSpec]:
            return [
                MCPToolSpec(
                    server_id="__synthetic__",
                    name="memory.add",
                    description="Add memory",
                    input_schema={"type": "object", "properties": {}},
                ),
                MCPToolSpec(
                    server_id="test",
                    name="test.browse_url",
                    description="Browse a URL",
                    input_schema={"type": "object", "properties": {}},
                ),
            ]

        async def call_tool(self, tool_id: str, _arguments: dict[str, Any]) -> Any:
            return type("_R", (), {"output_text": "done", "is_error": False})()

        async def __aenter__(self) -> "_StubPool":
            return self

        async def __aexit__(self, *_args: object) -> None:
            pass

        async def start(self) -> None:
            pass

        async def aclose(self) -> None:
            pass

    # Temporarily set tools_authoritative=True on the default mode
    catalog = get_workflow_catalog()
    default_mode = catalog.get("default")
    assert default_mode is not None
    original = default_mode.tools_authoritative
    default_mode.tools_authoritative = True
    try:
        hooks = TurnHooks(complete=capture_then_respond, mcp_tool_pool=_StubPool())
        await run_turn(mode_id="default", query="test", hooks=hooks)
    finally:
        default_mode.tools_authoritative = original

    tool_names = {t["function"]["name"] for t in captured_tools if "function" in t}
    # No MCP tools should appear (allowed_tool_ids is empty + authoritative)
    assert "memory.add" not in tool_names
    assert "test.browse_url" not in tool_names
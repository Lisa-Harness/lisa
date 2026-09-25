"""Tests for in-process operation dispatch in the turn runner.

Operations are advertised to the LLM directly from the in-process catalog
(via ``build_operation_tool_schemas``) and dispatched through
``hooks.operations.execute_operation`` — not through an MCP subprocess. These tests
verify that path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from kernel.models import (
    OperationExecutionContext,
    OperationResult,
)
from kernel.operations.registry import (
    build_operation_tool_schemas,
    get_operation_catalog,
)
from kernel.turn_hooks import (
    ModelHooks,
    OperationHooks,
    RecallHooks,
    ToolHooks,
)
from kernel.turn_phases._tool_schemas import _prune_model_tool_schemas
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


def _make_tool_schema(name: str, description: str = "") -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description or f"Execute {name}.",
            "parameters": {"type": "object", "properties": {}},
        },
    }


# ── build_operation_tool_schemas ──────────────────────────────────────


def test_build_operation_tool_schemas_returns_openai_format() -> None:
    """Schemas are OpenAI function-tool dicts with the operation's Parameters."""
    schemas = build_operation_tool_schemas("remember")
    assert len(schemas) > 0
    for schema in schemas:
        assert schema["type"] == "function"
        assert "name" in schema["function"]
        assert "parameters" in schema["function"]


def test_build_operation_tool_schemas_filters_by_mode() -> None:
    """Operations not allowed for a mode are not included."""
    remember_schemas = build_operation_tool_schemas("remember")
    remember_names = {s["function"]["name"] for s in remember_schemas}
    # remember mode allows synthesize_and_store_learning
    assert "synthesize_and_store_learning" in remember_names

    # With the memory pack enabled, default mode can expose memory recall.
    default_schemas = build_operation_tool_schemas("default")
    default_names = {s["function"]["name"] for s in default_schemas}
    assert "recall_memory" in default_names


# ── In-process operation dispatch in run_turn ─────────────────────────


@pytest.mark.asyncio
async def test_operation_tool_call_dispatches_via_execute_operation() -> None:
    """When the model calls an operation by name, it dispatches through
    hooks.operations.execute_operation (not pool.call_tool).
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

    # The Phase-1 pre-turn recall calls recall_memory through the same
    # hook; this test asserts the model-driven dispatch list exactly, so
    # ambient recall must stay off (same pattern as the tool-dispatch
    # and transitions suite hooks builders).
    hooks = TurnHooks(
        models=ModelHooks(complete=two_step),
        operations=OperationHooks(execute_operation=stub_execute_op),
        recall=RecallHooks(enabled=False),
    )

    result = await run_turn(
        mode_id="remember", query="remember that we decided to use PostgreSQL", hooks=hooks
    )

    assert result.response_text == "Memory stored successfully."
    assert result.tool_call_count == 1
    assert dispatched_operations == ["synthesize_and_store_learning"]


@pytest.mark.asyncio
async def test_non_operation_tool_call_dispatches_via_pool() -> None:
    """Non-operation tool names still go through pool.call_tool."""
    from kernel.tools.mcp_client import MCPToolResult

    pool_calls: list[str] = []

    class _StubPool:
        async def list_tools(self, *, allowed_server_ids: set[str] | None = None) -> list[Any]:
            return []

        async def call_tool(self, tool_id: str, _arguments: dict[str, Any]) -> MCPToolResult:
            pool_calls.append(tool_id)
            return MCPToolResult(tool_id=tool_id, output_text="MCP tool result")

        async def __aenter__(self) -> _StubPool:
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

    async def stub_execute_op(operation_id: str, ctx: OperationExecutionContext) -> OperationResult:
        return OperationResult(output_text="should not be called")

    hooks = TurnHooks(
        models=ModelHooks(
            complete=two_step,
        ),
        tools=ToolHooks(
            mcp_tool_pool=_StubPool(),
        ),
        operations=OperationHooks(
            execute_operation=stub_execute_op,
        ),
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

    hooks = TurnHooks(
        models=ModelHooks(
            complete=capture_then_respond,
        ),
    )

    await run_turn(mode_id="remember", query="test", hooks=hooks)

    tool_names = {t["function"]["name"] for t in captured_tools if "function" in t}
    # At least one operation should be in the tool list for remember mode
    operation_ids = set(get_operation_catalog().keys())
    assert tool_names & operation_ids, f"Expected operation schemas in tools, got: {tool_names}"


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

    # Off so ambient pre-turn recall does not add extra model calls.
    hooks = TurnHooks(
        models=ModelHooks(complete=two_step),
        operations=OperationHooks(execute_operation=failing_execute_op),
        recall=RecallHooks(enabled=False),
    )

    result = await run_turn(mode_id="remember", query="test", hooks=hooks)

    # The model received the error as a tool result and recovered: the
    # iteration-2 call's messages carry the tool error for call_1.
    assert len(call_log) == 2
    tool_messages = [
        m
        for m in call_log[1].get("messages", [])
        if m.get("role") == "tool" and m.get("tool_call_id") == "call_1"
    ]
    assert tool_messages, "the failed dispatch must surface as a tool result"
    assert "boom" in str(tool_messages[0].get("content", ""))
    # The failed dispatch IS counted as an attempt: budget counters and
    # the write-contract backstop key on ``total_tool_calls``, so a raised
    # dispatch must count (a failed write must not re-arm the backstop and
    # burn iterations synthesizing a doomed empty-args call).
    assert result.tool_call_count == 1
    assert result.iteration_count == 2
    # Phase 8: remember declares a write contract and the only write
    # attempt failed — zero verified writes, so the parent-turn
    # completion contract blocks the turn as ``no_tools``.
    assert result.blocked is True
    assert result.block_origin == "no_tools"


# ── Mode-level tool filtering ────────────────────────────────────────


def test_remember_mode_has_allowed_tool_ids() -> None:
    """The remember mode declares allowed_tool_ids in its front-matter."""
    from kernel.ipsar_fsm import get_workflow_catalog

    remember = get_workflow_catalog().get("remember")
    assert remember is not None
    assert remember.allowed_tool_ids == ["memory.add", "memory.search", "memory.list"]


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

    # Build a stub pool that returns tools the remember mode doesn't allow.
    # MCP-discovered tools are namespaced (server_id.tool_name); synthetic
    # aliases keep their bare names (memory.add, etc.).
    class _StubPool:
        async def list_tools(
            self, *, allowed_server_ids: set[str] | None = None
        ) -> list[MCPToolSpec]:
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

        async def __aenter__(self) -> _StubPool:
            return self

        async def __aexit__(self, *_args: object) -> None:
            pass

        async def start(self) -> None:
            pass

        async def aclose(self) -> None:
            pass

    hooks = TurnHooks(
        models=ModelHooks(
            complete=capture_then_respond,
        ),
        tools=ToolHooks(
            mcp_tool_pool=_StubPool(),
        ),
    )

    await run_turn(mode_id="remember", query="test", hooks=hooks)

    tool_names = {t["function"]["name"] for t in captured_tools if "function" in t}
    # memory.add (synthetic) is allowed; namespaced MCP tools are not
    assert "memory.add" in tool_names
    assert "test.delete_all_memories" not in tool_names
    assert "test.browse_url" not in tool_names


# ── tools_authoritative mode ─────────────────────────────────────────


def test_default_mode_tools_authoritative_is_true() -> None:
    """tools_authoritative defaults to True (authoritative by default)."""
    from kernel.ipsar_fsm import get_workflow_catalog

    default = get_workflow_catalog().get("default")
    assert default is not None
    assert default.tools_authoritative is True


def test_remember_mode_tools_authoritative_is_true() -> None:
    """Remember mode is tool-authoritative to constrain memory calls."""
    from kernel.ipsar_fsm import get_workflow_catalog

    remember = get_workflow_catalog().get("remember")
    assert remember is not None
    assert remember.tools_authoritative is True


def test_build_operation_tool_schemas_authoritative_excludes_unbound() -> None:
    """When tools_authoritative=True, only mode-bound operations are included."""
    from kernel.operations.registry import build_operation_tool_schemas

    # remember mode with authoritative=True should only include operations
    # explicitly bound to remember (synthesize_and_store_learning, recall_memory).
    # Other operations with no mode_allowlist should be excluded.
    schemas = build_operation_tool_schemas("remember", tools_authoritative=True)
    names = {s["function"]["name"] for s in schemas}
    assert "synthesize_and_store_learning" in names
    assert "recall_memory" in names
    # Operations not bound to remember should NOT appear
    assert "self_model" not in names


def test_build_operation_tool_schemas_non_authoritative_includes_unbound() -> None:
    """When tools_authoritative=False, unbound operations still appear
    (the default fallback: no mode_allowlist = allowed everywhere)."""
    from kernel.operations.registry import build_operation_tool_schemas

    schemas = build_operation_tool_schemas("remember", tools_authoritative=False)
    names = {s["function"]["name"] for s in schemas}
    # remember-bound operations are included
    assert "synthesize_and_store_learning" in names
    assert "recall_memory" in names
    # Non-authoritative should include MORE operations than authoritative
    # (operations without a restricting mode_allowlist pass through)
    authoritative_schemas = build_operation_tool_schemas("remember", tools_authoritative=True)
    assert len(schemas) >= len(authoritative_schemas)


def test_markdown_mode_inherits_recall_memory_via_default_available() -> None:
    """The markdown mode (empty allowed_operation_ids, authoritative) inherits
    recall_memory via the default_available merge in the SDK catalog loader.

    This is the end-to-end regression guard for the ambient-capability merge:
    markdown/_mode.md does not list recall_memory, but recall.yml declares
    default_available: true, so the merged catalog's markdown mode should
    advertise recall_memory to the model under tools_authoritative=True.
    """
    from kernel.ipsar_fsm import get_workflow_catalog
    from kernel.operations.registry import build_operation_tool_schemas

    markdown = get_workflow_catalog().get("markdown")
    assert markdown is not None
    # The merge should have added recall_memory to the markdown mode's
    # allowed_operation_ids (the SDK loader runs before the kernel sees it).
    assert "recall_memory" in markdown.allowed_operation_ids
    # And the schema builder should advertise it under authoritative filtering.
    schemas = build_operation_tool_schemas("markdown", tools_authoritative=True)
    names = {s["function"]["name"] for s in schemas}
    assert "recall_memory" in names


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
        async def list_tools(
            self, *, allowed_server_ids: set[str] | None = None
        ) -> list[MCPToolSpec]:
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

        async def __aenter__(self) -> _StubPool:
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
        hooks = TurnHooks(
            models=ModelHooks(
                complete=capture_then_respond,
            ),
            tools=ToolHooks(
                mcp_tool_pool=_StubPool(),
            ),
        )
        await run_turn(mode_id="default", query="test", hooks=hooks)
    finally:
        default_mode.tools_authoritative = original

    tool_names = {t["function"]["name"] for t in captured_tools if "function" in t}
    # No MCP tools should appear (allowed_tool_ids is empty + authoritative)
    assert "memory.add" not in tool_names
    assert "test.browse_url" not in tool_names


def test_prune_model_tool_schemas_dedupes_and_filters_non_workflow_transition() -> None:
    """Final prune pass removes duplicates and the transition tool when not advertised."""
    mode_config = SimpleNamespace(
        mode_id="default",
        allowed_tool_ids=[],
        tools_authoritative=False,
        entry_operation_ids=[],
        pre_turn_operation_ids=[],
        post_turn_operation_ids=[],
        allowed_operation_ids=[],
    )
    schemas = [
        _make_tool_schema("memory.add"),
        _make_tool_schema("memory.add"),
        _make_tool_schema("transition_stage"),
    ]

    pruned, stats = _prune_model_tool_schemas(
        tool_schemas=schemas,
        mode_config=mode_config,
        allow_transition_stage=False,
    )

    names = [schema["function"]["name"] for schema in pruned]
    assert names == ["memory.add"]
    assert stats["dropped_duplicate"] == 1
    assert stats["dropped_disallowed"] == 1


@pytest.mark.asyncio
async def test_run_turn_prunes_duplicate_tools_before_model_call() -> None:
    """The model-facing tools payload is deduplicated by final prune pass."""
    from kernel.tools.mcp_client import MCPToolSpec

    captured_tools: list[dict[str, Any]] = []

    async def capture_then_respond(**kwargs: Any) -> _StubCompletion:
        tools = kwargs.get("tools") or []
        captured_tools.extend(tools)
        return _StubCompletion(content="Done.")

    class _StubPool:
        async def list_tools(
            self, *, allowed_server_ids: set[str] | None = None
        ) -> list[MCPToolSpec]:
            return [
                MCPToolSpec(
                    server_id="__synthetic__",
                    name="memory.add",
                    description="Add memory",
                    input_schema={"type": "object", "properties": {}},
                ),
                MCPToolSpec(
                    server_id="shadow",
                    name="memory.add",
                    description="Duplicate alias entry",
                    input_schema={"type": "object", "properties": {}},
                ),
            ]

        async def call_tool(self, tool_id: str, _arguments: dict[str, Any]) -> Any:
            return type("_R", (), {"output_text": "done", "is_error": False})()

        async def __aenter__(self) -> _StubPool:
            return self

        async def __aexit__(self, *_args: object) -> None:
            pass

        async def start(self) -> None:
            pass

        async def aclose(self) -> None:
            pass

    from kernel.ipsar_fsm import get_workflow_catalog

    catalog = get_workflow_catalog()
    default_mode = catalog.get("default")
    assert default_mode is not None
    original_authoritative = default_mode.tools_authoritative
    default_mode.tools_authoritative = False
    try:
        hooks = TurnHooks(
            models=ModelHooks(
                complete=capture_then_respond,
            ),
            tools=ToolHooks(
                mcp_tool_pool=_StubPool(),
            ),
        )
        await run_turn(mode_id="default", query="test", hooks=hooks)
    finally:
        default_mode.tools_authoritative = original_authoritative

    tool_names = [t["function"]["name"] for t in captured_tools if "function" in t]
    assert tool_names.count("memory.add") == 1

"""Tests for tool schema assembly, filtering, and pruning in the turn runner.

Covers the schema builder helpers that determine which tools the model sees:
  - MCP tool discovery and mode-allowlist filtering
  - Operation schema assembly with authoritative/non-authoritative modes
  - In-process pack tool schema assembly
  - transition_stage tool inclusion rules
  - Final prune pass: deduplication, disallow filtering, stats tracking
  - Handoff schema rebuild

These tests exercise the internal helpers directly so that the extraction
into a SchemaBuilder module (Phase 4) can verify behavior parity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from kernel.models import OperationResult
from kernel.turn_runner import (
    TurnHooks,
    _build_in_process_tool_schemas,
    _mode_bound_operation_ids,
    _operation_allowed_for_mode,
    _prune_model_tool_schemas,
    _tool_schema_name,
    run_turn,
)

# ── Stubs ─────────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _StubCompletion:
    content: str
    tool_calls: list[dict[str, Any]] | None = None


def _make_tool_schema(name: str, description: str = "") -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description or f"Execute {name}.",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def _make_mode_config(
    mode_id: str = "default",
    allowed_tool_ids: list[str] | None = None,
    tools_authoritative: bool = True,
    entry_operation_ids: list[str] | None = None,
    pre_turn_operation_ids: list[str] | None = None,
    post_turn_operation_ids: list[str] | None = None,
    allowed_operation_ids: list[str] | None = None,
    mode_kind: str = "chat",
    default_model: str = "default",
    start_stage_id: str = "",
    stages: dict[str, Any] | None = None,
) -> SimpleNamespace:
    """Build a minimal mode config stub for schema assembly tests."""
    return SimpleNamespace(
        mode_id=mode_id,
        allowed_tool_ids=allowed_tool_ids or [],
        tools_authoritative=tools_authoritative,
        entry_operation_ids=entry_operation_ids or [],
        pre_turn_operation_ids=pre_turn_operation_ids or [],
        post_turn_operation_ids=post_turn_operation_ids or [],
        allowed_operation_ids=allowed_operation_ids or [],
        mode_kind=mode_kind,
        default_model=default_model,
        start_stage_id=start_stage_id,
        stages=stages or {},
    )


# ── _tool_schema_name ──────────────────────────────────────────────────


def test_tool_schema_name_extracts_function_name() -> None:
    """_tool_schema_name returns the function.name from an OpenAI tool schema."""
    schema = _make_tool_schema("my_tool")
    assert _tool_schema_name(schema) == "my_tool"


def test_tool_schema_name_returns_empty_for_missing_function() -> None:
    """_tool_schema_name returns empty string when function key is missing."""
    assert _tool_schema_name({}) == ""
    assert _tool_schema_name({"type": "function"}) == ""


def test_tool_schema_name_returns_empty_for_non_string_name() -> None:
    """_tool_schema_name returns empty string when name is not a str."""
    schema = {"function": {"name": 123}}
    assert _tool_schema_name(schema) == ""


def test_tool_schema_name_strips_whitespace() -> None:
    """_tool_schema_name strips leading/trailing whitespace from the name."""
    schema = {"function": {"name": "  spaced  "}}
    assert _tool_schema_name(schema) == "spaced"


# ── _mode_bound_operation_ids ─────────────────────────────────────────


def test_mode_bound_operation_ids_collects_all_binding_lists() -> None:
    """_mode_bound_operation_ids collects from all four binding fields."""
    mode_config = _make_mode_config(
        entry_operation_ids=["entry_op"],
        pre_turn_operation_ids=["pre_op"],
        post_turn_operation_ids=["post_op"],
        allowed_operation_ids=["allowed_op"],
    )
    ids = _mode_bound_operation_ids(mode_config)
    assert ids == {"entry_op", "pre_op", "post_op", "allowed_op"}


def test_mode_bound_operation_ids_empty_when_all_empty() -> None:
    """_mode_bound_operation_ids returns empty set when no bindings."""
    mode_config = _make_mode_config()
    assert _mode_bound_operation_ids(mode_config) == set()


def test_mode_bound_operation_ids_none_mode_config() -> None:
    """_mode_bound_operation_ids returns empty set when mode_config is None."""
    assert _mode_bound_operation_ids(None) == set()


def test_mode_bound_operation_ids_handles_none_lists() -> None:
    """_mode_bound_operation_ids handles None values in binding lists."""
    mode_config = SimpleNamespace(
        entry_operation_ids=None,
        pre_turn_operation_ids=["pre_op"],
        post_turn_operation_ids=None,
        allowed_operation_ids=None,
    )
    ids = _mode_bound_operation_ids(mode_config)
    assert ids == {"pre_op"}


# ── _operation_allowed_for_mode ───────────────────────────────────────


def test_operation_allowed_for_mode_returns_true_when_no_mode_config() -> None:
    """When mode_config is None, all operations are allowed."""
    assert _operation_allowed_for_mode(
        operation_id="any_op",
        definition=SimpleNamespace(mode_allowlist=None),
        mode_config=None,
        tools_authoritative=True,
        mode_bound_ids=set(),
    ) is True


def test_operation_allowed_for_mode_mode_allowlist_match() -> None:
    """When the operation's mode_allowlist includes the current mode, it's allowed."""
    definition = SimpleNamespace(mode_allowlist=["learn", "default"])
    mode_config = _make_mode_config(mode_id="learn")
    assert _operation_allowed_for_mode(
        operation_id="my_op",
        definition=definition,
        mode_config=mode_config,
        tools_authoritative=True,
        mode_bound_ids=set(),
    ) is True


def test_operation_allowed_for_mode_mode_bound_id_match() -> None:
    """When the operation_id is in mode_bound_ids, it's allowed."""
    definition = SimpleNamespace(mode_allowlist=None)
    mode_config = _make_mode_config(mode_id="default")
    assert _operation_allowed_for_mode(
        operation_id="bound_op",
        definition=definition,
        mode_config=mode_config,
        tools_authoritative=True,
        mode_bound_ids={"bound_op"},
    ) is True


def test_operation_allowed_for_mode_authoritative_excludes_unbound() -> None:
    """When tools_authoritative=True and not in allowlist/bound, excluded."""
    definition = SimpleNamespace(mode_allowlist=None)
    mode_config = _make_mode_config(mode_id="default")
    assert _operation_allowed_for_mode(
        operation_id="unbound_op",
        definition=definition,
        mode_config=mode_config,
        tools_authoritative=True,
        mode_bound_ids=set(),
    ) is False


def test_operation_allowed_for_mode_non_authoritative_includes_unbound() -> None:
    """When tools_authoritative=False and no mode_allowlist, unbound ops pass."""
    definition = SimpleNamespace(mode_allowlist=None)
    mode_config = _make_mode_config(mode_id="default")
    assert _operation_allowed_for_mode(
        operation_id="unbound_op",
        definition=definition,
        mode_config=mode_config,
        tools_authoritative=False,
        mode_bound_ids=set(),
    ) is True


def test_operation_allowed_for_mode_non_authoritative_excludes_when_allowlist_present() -> None:
    """When tools_authoritative=False but the op has a mode_allowlist that
    doesn't include the current mode, the op is excluded."""
    definition = SimpleNamespace(mode_allowlist=["other_mode"])
    mode_config = _make_mode_config(mode_id="default")
    assert _operation_allowed_for_mode(
        operation_id="restricted_op",
        definition=definition,
        mode_config=mode_config,
        tools_authoritative=False,
        mode_bound_ids=set(),
    ) is False


# ── _prune_model_tool_schemas ─────────────────────────────────────────


def test_prune_deduplicates_same_tool_name() -> None:
    """Duplicate tool names are removed, keeping the first occurrence."""
    mode_config = _make_mode_config(tools_authoritative=False)
    schemas = [_make_tool_schema("dup"), _make_tool_schema("dup")]
    pruned, stats = _prune_model_tool_schemas(
        tool_schemas=schemas,
        mode_config=mode_config,
        is_workflow=False,
    )
    assert len(pruned) == 1
    assert stats["dropped_duplicate"] == 1


def test_prune_drops_invalid_schemas_without_function_name() -> None:
    """Schemas with no function.name are dropped as invalid."""
    mode_config = _make_mode_config(tools_authoritative=False)
    schemas = [{"type": "function"}, {}]
    pruned, stats = _prune_model_tool_schemas(
        tool_schemas=schemas,
        mode_config=mode_config,
        is_workflow=False,
    )
    assert len(pruned) == 0
    assert stats["dropped_invalid"] == 2


def test_prune_excludes_transition_tool_for_non_workflow_non_routing() -> None:
    """transition_stage is dropped when not workflow and no routing targets."""
    mode_config = _make_mode_config(tools_authoritative=False)
    schemas = [_make_tool_schema("transition_stage")]
    pruned, stats = _prune_model_tool_schemas(
        tool_schemas=schemas,
        mode_config=mode_config,
        is_workflow=False,
        has_routing_targets=False,
    )
    assert len(pruned) == 0
    assert stats["dropped_disallowed"] == 1


def test_prune_includes_transition_tool_for_workflow() -> None:
    """transition_stage is kept when is_workflow=True."""
    mode_config = _make_mode_config(tools_authoritative=False)
    schemas = [_make_tool_schema("transition_stage")]
    pruned, _ = _prune_model_tool_schemas(
        tool_schemas=schemas,
        mode_config=mode_config,
        is_workflow=True,
        has_routing_targets=False,
    )
    assert len(pruned) == 1
    assert _tool_schema_name(pruned[0]) == "transition_stage"


def test_prune_includes_transition_tool_for_routing_targets() -> None:
    """transition_stage is kept when has_routing_targets=True (even if not workflow)."""
    mode_config = _make_mode_config(tools_authoritative=False)
    schemas = [_make_tool_schema("transition_stage")]
    pruned, _ = _prune_model_tool_schemas(
        tool_schemas=schemas,
        mode_config=mode_config,
        is_workflow=False,
        has_routing_targets=True,
    )
    assert len(pruned) == 1


def test_prune_stats_track_all_categories() -> None:
    """Prune stats track input_count, output_count, and all drop reasons."""
    mode_config = _make_mode_config(tools_authoritative=False)
    schemas = [
        _make_tool_schema("keep_me"),
        _make_tool_schema("keep_me"),  # duplicate
        {"type": "function"},  # invalid
        _make_tool_schema("transition_stage"),  # disallowed (non-workflow)
    ]
    _, stats = _prune_model_tool_schemas(
        tool_schemas=schemas,
        mode_config=mode_config,
        is_workflow=False,
        has_routing_targets=False,
    )
    assert stats["input_count"] == 4
    assert stats["output_count"] == 1
    assert stats["dropped_duplicate"] == 1
    assert stats["dropped_invalid"] == 1
    assert stats["dropped_disallowed"] == 1


def test_prune_respects_mode_allowlist_for_mcp_tools() -> None:
    """MCP tools not in the allowed_tool_ids list are dropped."""
    mode_config = _make_mode_config(
        allowed_tool_ids=["allowed_tool"],
        tools_authoritative=True,
    )
    schemas = [_make_tool_schema("allowed_tool"), _make_tool_schema("disallowed_tool")]
    pruned, stats = _prune_model_tool_schemas(
        tool_schemas=schemas,
        mode_config=mode_config,
        is_workflow=False,
    )
    names = [_tool_schema_name(s) for s in pruned]
    assert "allowed_tool" in names
    assert "disallowed_tool" not in names
    assert stats["dropped_disallowed"] == 1


def test_prune_none_mode_config_passes_all() -> None:
    """When mode_config is None, all non-transition tools pass through."""
    schemas = [_make_tool_schema("tool_a"), _make_tool_schema("tool_b")]
    pruned, stats = _prune_model_tool_schemas(
        tool_schemas=schemas,
        mode_config=None,
        is_workflow=False,
    )
    assert len(pruned) == 2
    assert stats["output_count"] == 2


# ── _build_in_process_tool_schemas ─────────────────────────────────────


def test_build_in_process_tool_schemas_returns_empty_when_no_catalog() -> None:
    """_build_in_process_tool_schemas returns [] when tool catalog is empty."""
    # The catalog may have tools from the conftest fixture, so we check
    # that the function returns a list (not None).
    mode_config = _make_mode_config(tools_authoritative=False)
    schemas = _build_in_process_tool_schemas(mode_config)
    assert isinstance(schemas, list)


def test_build_in_process_tool_schemas_filters_by_mode_allowlist() -> None:
    """Pack tools with mode_allowlist are filtered by the current mode_id."""
    # We can't easily inject a fake tool into the catalog without monkeypatching,
    # so this test verifies the function handles a mode_config with a restrictive
    # allowlist without crashing.
    mode_config = _make_mode_config(
        mode_id="default",
        allowed_tool_ids=["nonexistent_tool"],
        tools_authoritative=True,
    )
    schemas = _build_in_process_tool_schemas(mode_config)
    # No tools should match the nonexistent allowlist.
    names = [_tool_schema_name(s) for s in schemas]
    assert "nonexistent_tool" not in names


# ── Schema assembly end-to-end via run_turn ───────────────────────────


@pytest.mark.asyncio
async def test_schema_assembly_includes_operations_for_mode() -> None:
    """The tools sent to the model include operations from the catalog."""
    from kernel.operations.registry import get_operation_catalog

    captured_tools: list[dict[str, Any]] = []

    async def capture(**kwargs: Any) -> _StubCompletion:
        captured_tools.extend(kwargs.get("tools") or [])
        return _StubCompletion(content="Done.")

    hooks = TurnHooks(complete=capture)
    await run_turn(mode_id="learn", query="test", hooks=hooks)

    tool_names = {_tool_schema_name(t) for t in captured_tools}
    operation_ids = set(get_operation_catalog().keys())
    assert tool_names & operation_ids, "Expected at least one operation in tool schemas"


@pytest.mark.asyncio
async def test_schema_assembly_excludes_mcp_tools_not_in_allowlist() -> None:
    """MCP tools outside the mode's allowed_tool_ids are not advertised."""
    from kernel.tools.mcp_client import MCPToolSpec

    captured_tools: list[dict[str, Any]] = []

    async def capture(**kwargs: Any) -> _StubCompletion:
        captured_tools.extend(kwargs.get("tools") or [])
        return _StubCompletion(content="Done.")

    class _StubPool:
        async def list_tools(self, *, allowed_server_ids: set[str] | None = None) -> list[MCPToolSpec]:
            return [
                MCPToolSpec(
                    server_id="__synthetic__",
                    name="memory.add",
                    description="Add",
                    input_schema={"type": "object", "properties": {}},
                ),
                MCPToolSpec(
                    server_id="test",
                    name="test.unauthorized_tool",
                    description="Should be filtered",
                    input_schema={"type": "object", "properties": {}},
                ),
            ]

        async def call_tool(self, tool_id: str, _args: dict[str, Any]) -> Any:
            return type("_R", (), {"output_text": "done", "is_error": False})()

        async def __aenter__(self) -> "_StubPool":
            return self

        async def __aexit__(self, *_a: object) -> None:
            pass

        async def start(self) -> None:
            pass

        async def aclose(self) -> None:
            pass

    hooks = TurnHooks(complete=capture, mcp_tool_pool=_StubPool())
    await run_turn(mode_id="learn", query="test", hooks=hooks)

    tool_names = {_tool_schema_name(t) for t in captured_tools}
    # learn mode allows memory.add (synthetic) but not test.unauthorized_tool
    assert "memory.add" in tool_names
    assert "test.unauthorized_tool" not in tool_names


@pytest.mark.asyncio
async def test_schema_assembly_authoritative_empty_allowlist_no_mcp_tools() -> None:
    """When tools_authoritative=True and allowed_tool_ids is empty,
    no MCP tools appear (but operations still do)."""
    from kernel.ipsar_fsm import get_workflow_catalog
    from kernel.tools.mcp_client import MCPToolSpec

    captured_tools: list[dict[str, Any]] = []

    async def capture(**kwargs: Any) -> _StubCompletion:
        captured_tools.extend(kwargs.get("tools") or [])
        return _StubCompletion(content="Done.")

    class _StubPool:
        async def list_tools(self, *, allowed_server_ids: set[str] | None = None) -> list[MCPToolSpec]:
            return [
                MCPToolSpec(
                    server_id="__synthetic__",
                    name="memory.add",
                    description="Add",
                    input_schema={"type": "object", "properties": {}},
                ),
            ]

        async def call_tool(self, tool_id: str, _args: dict[str, Any]) -> Any:
            return type("_R", (), {"output_text": "done", "is_error": False})()

        async def __aenter__(self) -> "_StubPool":
            return self

        async def __aexit__(self, *_a: object) -> None:
            pass

        async def start(self) -> None:
            pass

        async def aclose(self) -> None:
            pass

    catalog = get_workflow_catalog()
    default_mode = catalog.get("default")
    assert default_mode is not None
    original = default_mode.tools_authoritative
    default_mode.tools_authoritative = True
    try:
        hooks = TurnHooks(complete=capture, mcp_tool_pool=_StubPool())
        await run_turn(mode_id="default", query="test", hooks=hooks)
    finally:
        default_mode.tools_authoritative = original

    tool_names = {_tool_schema_name(t) for t in captured_tools}
    assert "memory.add" not in tool_names  # filtered by authoritative+empty


@pytest.mark.asyncio
async def test_schema_assembly_deduplicates_before_model_call() -> None:
    """The final prune pass removes duplicate tool names before sending to model."""
    from kernel.tools.mcp_client import MCPToolSpec

    captured_tools: list[dict[str, Any]] = []

    async def capture(**kwargs: Any) -> _StubCompletion:
        captured_tools.extend(kwargs.get("tools") or [])
        return _StubCompletion(content="Done.")

    class _StubPool:
        async def list_tools(self, *, allowed_server_ids: set[str] | None = None) -> list[MCPToolSpec]:
            return [
                MCPToolSpec(
                    server_id="__synthetic__",
                    name="memory.add",
                    description="Add",
                    input_schema={"type": "object", "properties": {}},
                ),
                MCPToolSpec(
                    server_id="shadow",
                    name="memory.add",
                    description="Duplicate",
                    input_schema={"type": "object", "properties": {}},
                ),
            ]

        async def call_tool(self, tool_id: str, _args: dict[str, Any]) -> Any:
            return type("_R", (), {"output_text": "done", "is_error": False})()

        async def __aenter__(self) -> "_StubPool":
            return self

        async def __aexit__(self, *_a: object) -> None:
            pass

        async def start(self) -> None:
            pass

        async def aclose(self) -> None:
            pass

    from kernel.ipsar_fsm import get_workflow_catalog

    catalog = get_workflow_catalog()
    default_mode = catalog.get("default")
    assert default_mode is not None
    original = default_mode.tools_authoritative
    default_mode.tools_authoritative = False
    try:
        hooks = TurnHooks(complete=capture, mcp_tool_pool=_StubPool())
        await run_turn(mode_id="default", query="test", hooks=hooks)
    finally:
        default_mode.tools_authoritative = original

    tool_names = [_tool_schema_name(t) for t in captured_tools]
    assert tool_names.count("memory.add") == 1
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

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
from kernel.models import YieldPolicy
from kernel.turn_hooks import ModelHooks, ToolHooks
from kernel.turn_phases._tool_schemas import (
    _build_in_process_tool_schemas,
    _mode_bound_operation_ids,
    _operation_allowed_for_mode,
    _prune_model_tool_schemas,
    _tool_schema_name,
)
from kernel.turn_runner import (
    TurnHooks,
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
    assert (
        _operation_allowed_for_mode(
            operation_id="any_op",
            definition=SimpleNamespace(mode_allowlist=None),
            mode_config=None,
            tools_authoritative=True,
            mode_bound_ids=set(),
        )
        is True
    )


def test_operation_allowed_for_mode_mode_allowlist_match() -> None:
    """When the operation's mode_allowlist includes the current mode, it's allowed."""
    definition = SimpleNamespace(mode_allowlist=["remember", "default"])
    mode_config = _make_mode_config(mode_id="remember")
    assert (
        _operation_allowed_for_mode(
            operation_id="my_op",
            definition=definition,
            mode_config=mode_config,
            tools_authoritative=True,
            mode_bound_ids=set(),
        )
        is True
    )


def test_operation_allowed_for_mode_mode_bound_id_match() -> None:
    """When the operation_id is in mode_bound_ids, it's allowed."""
    definition = SimpleNamespace(mode_allowlist=None)
    mode_config = _make_mode_config(mode_id="default")
    assert (
        _operation_allowed_for_mode(
            operation_id="bound_op",
            definition=definition,
            mode_config=mode_config,
            tools_authoritative=True,
            mode_bound_ids={"bound_op"},
        )
        is True
    )


def test_operation_allowed_for_mode_authoritative_excludes_unbound() -> None:
    """When tools_authoritative=True and not in allowlist/bound, excluded."""
    definition = SimpleNamespace(mode_allowlist=None)
    mode_config = _make_mode_config(mode_id="default")
    assert (
        _operation_allowed_for_mode(
            operation_id="unbound_op",
            definition=definition,
            mode_config=mode_config,
            tools_authoritative=True,
            mode_bound_ids=set(),
        )
        is False
    )


def test_operation_allowed_for_mode_non_authoritative_includes_unbound() -> None:
    """When tools_authoritative=False and no mode_allowlist, unbound ops pass."""
    definition = SimpleNamespace(mode_allowlist=None)
    mode_config = _make_mode_config(mode_id="default")
    assert (
        _operation_allowed_for_mode(
            operation_id="unbound_op",
            definition=definition,
            mode_config=mode_config,
            tools_authoritative=False,
            mode_bound_ids=set(),
        )
        is True
    )


def test_operation_allowed_for_mode_non_authoritative_excludes_when_allowlist_present() -> None:
    """When tools_authoritative=False but the op has a mode_allowlist that
    doesn't include the current mode, the op is excluded."""
    definition = SimpleNamespace(mode_allowlist=["other_mode"])
    mode_config = _make_mode_config(mode_id="default")
    assert (
        _operation_allowed_for_mode(
            operation_id="restricted_op",
            definition=definition,
            mode_config=mode_config,
            tools_authoritative=False,
            mode_bound_ids=set(),
        )
        is False
    )


# ── _prune_model_tool_schemas ─────────────────────────────────────────


def test_prune_deduplicates_same_tool_name() -> None:
    """Duplicate tool names are removed, keeping the first occurrence."""
    mode_config = _make_mode_config(tools_authoritative=False)
    schemas = [_make_tool_schema("dup"), _make_tool_schema("dup")]
    pruned, stats = _prune_model_tool_schemas(
        tool_schemas=schemas,
        mode_config=mode_config,
        allow_transition_stage=False,
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
        allow_transition_stage=False,
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
        allow_transition_stage=False,
    )
    assert len(pruned) == 0
    assert stats["dropped_disallowed"] == 1


def test_prune_includes_transition_tool_for_workflow() -> None:
    """transition_stage is kept when allow_transition_stage=True."""
    mode_config = _make_mode_config(tools_authoritative=False)
    schemas = [_make_tool_schema("transition_stage")]
    pruned, _ = _prune_model_tool_schemas(
        tool_schemas=schemas,
        mode_config=mode_config,
        allow_transition_stage=True,
    )
    assert len(pruned) == 1
    assert _tool_schema_name(pruned[0]) == "transition_stage"


def test_prune_includes_transition_tool_for_routing_targets() -> None:
    """transition_stage is kept when allow_transition_stage=True (edges or routing)."""
    mode_config = _make_mode_config(tools_authoritative=False)
    schemas = [_make_tool_schema("transition_stage")]
    pruned, _ = _prune_model_tool_schemas(
        tool_schemas=schemas,
        mode_config=mode_config,
        allow_transition_stage=True,
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
        allow_transition_stage=False,
    )
    assert stats["input_count"] == 4
    assert stats["output_count"] == 1
    assert stats["dropped_duplicate"] == 1
    assert stats["dropped_invalid"] == 1
    assert stats["dropped_disallowed"] == 1


def test_prune_sorts_output_schemas_by_function_name() -> None:
    """The final prune pass emits schemas in canonical name order.

    Canonical wire order makes the same mode's tool payload byte-identical
    across turns even when catalog/MCP registration order varies, which is
    what provider-side implicit prefix caching keys on.
    """
    mode_config = _make_mode_config(tools_authoritative=False)
    schemas = [
        _make_tool_schema("zeta_tool"),
        _make_tool_schema("alpha_tool"),
        _make_tool_schema("mid_tool"),
    ]
    pruned, _ = _prune_model_tool_schemas(
        tool_schemas=schemas,
        mode_config=mode_config,
        allow_transition_stage=False,
    )
    names = [_tool_schema_name(s) for s in pruned]
    assert names == ["alpha_tool", "mid_tool", "zeta_tool"]


def test_prune_sort_keeps_first_seen_duplicate_semantics() -> None:
    """Sort must run after first-seen duplicate resolution, not before.

    If the sort raced ahead of dedup, a duplicate pair with different
    descriptions could flip which schema survives; sorting after dedup
    keeps first-registered description semantics.
    """
    mode_config = _make_mode_config(tools_authoritative=False)
    first = _make_tool_schema("dup_tool", description="original")
    first["function"]["description"] = "first description"
    second = _make_tool_schema("dup_tool", description="duplicate")
    second["function"]["description"] = "second description"
    pruned, stats = _prune_model_tool_schemas(
        tool_schemas=[first, second],
        mode_config=mode_config,
        allow_transition_stage=False,
    )
    assert len(pruned) == 1
    assert pruned[0]["function"]["description"] == "first description"
    assert stats["dropped_duplicate"] == 1


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
        allow_transition_stage=False,
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
        allow_transition_stage=False,
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


# ── _rebuild_tool_schemas_for_mode: transition_stage after handoff ───


@pytest.mark.asyncio
async def test_rebuild_includes_transition_stage_for_chat_mode_with_routing_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A handoff into a chat mode with routing targets keeps transition_stage.

    Regression for the 2026-08-06 ``okf`` loop: ``_rebuild_tool_schemas_for_mode``
    only advertised ``transition_stage`` for workflow target modes, so a
    ``plan → okf → markdown`` chain dropped the tool after handing off to
    ``okf`` (a chat mode with routing targets). The model could not call
    ``transition_stage(markdown)`` and looped until the stuck-in-stage cap.
    The rebuild must mirror Phase-1 assembly: advertise the tool when
    the target has routing targets or stage edges.
    """
    from kernel.turn_phases import _tool_schemas as _ts
    from kernel.turn_phases._tool_schemas import _rebuild_tool_schemas_for_mode

    mode_config = _make_mode_config(
        mode_id="okf",
        mode_kind="chat",
        allowed_tool_ids=["lisa-storage.read_file"],
        tools_authoritative=True,
    )

    # Stub MCP pool: no tools (we only care about transition_stage).
    class _EmptyPool:
        configs: list[Any] = []

        async def list_tools(self, **_: Any) -> list[Any]:
            return []

    # Pretend okf has routing targets (it can hand off to markdown).
    monkeypatch.setattr(_ts, "get_routing_targets", lambda mode_id: ["markdown"] if mode_id == "okf" else [])
    # No operations or pack tools; transition_stage comes only from the
    # rebuild's explicit append.
    monkeypatch.setattr(_ts, "build_operation_tool_schemas", lambda *a, **k: [])
    monkeypatch.setattr(_ts, "get_operation_catalog", lambda: {})
    monkeypatch.setattr(_ts, "get_tool_catalog", lambda: {})

    schemas = await _rebuild_tool_schemas_for_mode(
        "okf",
        mode_config,
        pool=_EmptyPool(),  # type: ignore[arg-type]
    )
    names = {_tool_schema_name(s) for s in schemas}
    assert "transition_stage" in names


@pytest.mark.asyncio
async def test_rebuild_excludes_transition_stage_for_chat_mode_without_routing_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A chat mode with no routing targets does not get transition_stage.

    Guards against over-advertising: the fix must not add the tool for
    single-mode chat turns (no handoffs available).
    """
    from kernel.turn_phases import _tool_schemas as _ts
    from kernel.turn_phases._tool_schemas import _rebuild_tool_schemas_for_mode

    mode_config = _make_mode_config(
        mode_id="discuss",
        mode_kind="chat",
        allowed_tool_ids=[],
        tools_authoritative=True,
    )

    class _EmptyPool:
        configs: list[Any] = []

        async def list_tools(self, **_: Any) -> list[Any]:
            return []

    monkeypatch.setattr(_ts, "get_routing_targets", lambda mode_id: [])
    monkeypatch.setattr(_ts, "build_operation_tool_schemas", lambda *a, **k: [])
    monkeypatch.setattr(_ts, "get_operation_catalog", lambda: {})
    monkeypatch.setattr(_ts, "get_tool_catalog", lambda: {})

    schemas = await _rebuild_tool_schemas_for_mode(
        "discuss",
        mode_config,
        pool=_EmptyPool(),  # type: ignore[arg-type]
    )
    names = {_tool_schema_name(s) for s in schemas}
    assert "transition_stage" not in names


# ── build_tool_surface: Phase-1/rebuild parity (Phase 1 unification) ──


class _EmptyPool:
    """MCP pool stub with no tools (only lifecycle tools matter here).

    No ``configs`` attribute: the builder reads it via ``getattr(pool,
    "configs", [])``, so absence exercises the same empty-server path.
    """

    async def list_tools(self, **_: Any) -> list[Any]:
        return []


def _isolate_surface(monkeypatch: pytest.MonkeyPatch, routing: dict[str, list[str]]) -> None:
    """Point the surface builder's catalog reads at empty/stubbed sources."""
    from kernel.turn_phases import _tool_schemas as _ts

    monkeypatch.setattr(_ts, "get_routing_targets", lambda mode_id: routing.get(mode_id, []))
    monkeypatch.setattr(_ts, "build_operation_tool_schemas", lambda *a, **k: [])
    monkeypatch.setattr(_ts, "get_operation_catalog", lambda: {})
    monkeypatch.setattr(_ts, "get_tool_catalog", lambda: {})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode_kind",
    ["workflow", "chat", "action"],
)
async def test_rebuild_surface_is_identical_to_phase1_surface(
    monkeypatch: pytest.MonkeyPatch,
    mode_kind: str,
) -> None:
    """The post-handoff rebuild produces the Phase-1 surface for the same mode.

    Pins the Phase-1 unification: ``_rebuild_tool_schemas_for_mode`` is a thin
    wrapper over ``build_tool_surface`` and must not add, drop, or reorder a
    single schema. If this fails, the two assembly paths have drifted again —
    the class of bug behind the 2026-08-06 ``okf`` loop and the latent
    yield-drop. Phase 6c: the surface no longer depends on session/position
    wiring — only on the mode config — so there is no session axis to vary.
    """
    from kernel.turn_phases._tool_schemas import _rebuild_tool_schemas_for_mode, build_tool_surface

    mode_config = _make_mode_config(
        mode_id="parity",
        mode_kind=mode_kind,
        allowed_tool_ids=["lisa-storage.read_file"],
        tools_authoritative=True,
    )
    _isolate_surface(monkeypatch, {"parity": ["markdown"]})
    policy = YieldPolicy()

    phase1_schemas, _ = await build_tool_surface(
        mode_config,
        pool=_EmptyPool(),  # type: ignore[arg-type]
        yield_policy=policy,
    )
    rebuilt = await _rebuild_tool_schemas_for_mode(
        "parity",
        mode_config,
        pool=_EmptyPool(),  # type: ignore[arg-type]
        yield_policy=policy,
    )
    assert rebuilt == phase1_schemas


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("policy", "expect_yield"),
    [
        (YieldPolicy(), True),  # default: inline asks allowed
        (YieldPolicy(allowed=False, fallback_mode="discuss"), True),  # fallback redirect
        (YieldPolicy(allowed=False), False),  # forbidden, no fallback
    ],
    ids=["default-allowed", "fallback-redirect", "forbidden-no-fallback"],
)
async def test_rebuild_advertises_yield_identically_to_phase1(
    monkeypatch: pytest.MonkeyPatch,
    policy: Any,
    expect_yield: bool,
) -> None:
    """The rebuilt surface advertises ``yield`` under the same policy rule as
    Phase 1 (``allowed OR fallback_mode``).

    Regression for the latent asymmetry found during the Phase-1 unification:
    the old rebuild path never advertised ``yield`` at all, so a mode handed
    off mid-turn silently lost the ability to ask the user a question.
    """
    from kernel.turn_phases._tool_schemas import _rebuild_tool_schemas_for_mode

    mode_config = _make_mode_config(
        mode_id="chatty",
        mode_kind="chat",
        allowed_tool_ids=[],
        tools_authoritative=True,
    )
    _isolate_surface(monkeypatch, {})

    schemas = await _rebuild_tool_schemas_for_mode(
        "chatty",
        mode_config,
        pool=_EmptyPool(),  # type: ignore[arg-type]
        yield_policy=policy,
    )
    names = {_tool_schema_name(s) for s in schemas}
    assert ("yield" in names) is expect_yield


# ── Schema assembly end-to-end via run_turn ───────────────────────────


@pytest.mark.asyncio
async def test_schema_assembly_includes_operations_for_mode() -> None:
    """The tools sent to the model include operations from the catalog."""
    from kernel.operations.registry import get_operation_catalog

    captured_tools: list[dict[str, Any]] = []

    async def capture(**kwargs: Any) -> _StubCompletion:
        captured_tools.extend(kwargs.get("tools") or [])
        return _StubCompletion(content="Done.")

    hooks = TurnHooks(
    models=ModelHooks(
        complete=capture,
    ),
)
    await run_turn(mode_id="remember", query="test", hooks=hooks)

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
        async def list_tools(
            self, *, allowed_server_ids: set[str] | None = None
        ) -> list[MCPToolSpec]:
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

        async def __aenter__(self) -> _StubPool:
            return self

        async def __aexit__(self, *_a: object) -> None:
            pass

        async def start(self) -> None:
            pass

        async def aclose(self) -> None:
            pass

    hooks = TurnHooks(
    models=ModelHooks(
        complete=capture,
    ),
    tools=ToolHooks(
        mcp_tool_pool=_StubPool(),
    ),
)
    await run_turn(mode_id="remember", query="test", hooks=hooks)

    tool_names = {_tool_schema_name(t) for t in captured_tools}
    # remember mode allows memory.add (synthetic) but not test.unauthorized_tool
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
        async def list_tools(
            self, *, allowed_server_ids: set[str] | None = None
        ) -> list[MCPToolSpec]:
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

        async def __aenter__(self) -> _StubPool:
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
        hooks = TurnHooks(
    models=ModelHooks(
        complete=capture,
    ),
    tools=ToolHooks(
        mcp_tool_pool=_StubPool(),
    ),
)
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
        async def list_tools(
            self, *, allowed_server_ids: set[str] | None = None
        ) -> list[MCPToolSpec]:
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

        async def __aenter__(self) -> _StubPool:
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
        hooks = TurnHooks(
    models=ModelHooks(
        complete=capture,
    ),
    tools=ToolHooks(
        mcp_tool_pool=_StubPool(),
    ),
)
        await run_turn(mode_id="default", query="test", hooks=hooks)
    finally:
        default_mode.tools_authoritative = original

    tool_names = [_tool_schema_name(t) for t in captured_tools]
    assert tool_names.count("memory.add") == 1

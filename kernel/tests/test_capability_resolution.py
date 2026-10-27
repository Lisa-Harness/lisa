"""Tests for the shared capability-id resolution layer and provider-id invariants.

Covers the three concerns audited in the ``*_id`` resolution audit:

1. ``kernel.capability`` — ``resolve_capability`` precedence (operations before
   tools on id collision), per-catalog entry points, and None for unknown ids.
2. ``owner_id`` — the catalog-owned canonical identity on
    ``ToolDefinition`` / ``OperationDefinition``.
3. ``validate_export_boundaries`` — loud failure (``OperationValidationError``)
   when a mode with an empty ``mode_provider_id`` references any capability,
   instead of silently flagging every reference as cross-pack.

These tests exercise the kernel helpers directly; the SDK loader backfill of
``pack_id`` from ``tool_provider_id`` is covered by the SDK catalog loader
tests.
"""

from __future__ import annotations

import pytest
from kernel.capability import (
    resolve_capability,
    resolve_capability_for_turn_limit,
    resolve_operation_definition,
    resolve_tool_definition,
)
from kernel.models import OperationDefinition, ToolDefinition, WorkflowModeConfig

# ── Stubs ────────────────────────────────────────────────────────────────


def _tool(
    tool_id: str,
    *,
    provider_id: str = "",
    pack_id: str = "",
    description: str = "stub tool",
    exportable: bool = False,
    max_calls_per_turn: int = 3,
) -> ToolDefinition:
    return ToolDefinition(
        tool_id=tool_id,
        description=description,
        owner_id=provider_id or pack_id,
        pack_id=pack_id,
        provider_id=provider_id,
        exportable=exportable,
        max_calls_per_turn=max_calls_per_turn,
    )


def _operation(
    operation_id: str,
    *,
    provider_id: str = "",
    pack_id: str = "",
    description: str = "stub operation",
    exportable: bool = False,
    max_calls_per_turn: int = 3,
) -> OperationDefinition:
    return OperationDefinition(
        operation_id=operation_id,
        description=description,
        owner_id=provider_id or pack_id,
        pack_id=pack_id,
        provider_id=provider_id,
        exportable=exportable,
        max_calls_per_turn=max_calls_per_turn,
    )


def _mode(
    mode_id: str = "default",
    *,
    mode_provider_id: str = "lisa-pack-test",
    allowed_operation_ids: list[str] | None = None,
    allowed_tool_ids: list[str] | None = None,
    entry_operation_ids: list[str] | None = None,
    pre_turn_operation_ids: list[str] | None = None,
    post_turn_operation_ids: list[str] | None = None,
) -> WorkflowModeConfig:
    return WorkflowModeConfig(
        mode_id=mode_id,
        start_stage_id="",
        mode_provider_id=mode_provider_id,
        allowed_operation_ids=allowed_operation_ids or [],
        allowed_tool_ids=allowed_tool_ids or [],
        entry_operation_ids=entry_operation_ids or [],
        pre_turn_operation_ids=pre_turn_operation_ids or [],
        post_turn_operation_ids=post_turn_operation_ids or [],
    )


# ── owner_id ──────────────────────────────────────────────────────────────


def test_owner_id_is_canonical() -> None:
    """Definitions expose the catalog-owned identity directly."""
    tool = _tool("t", provider_id="lisa-pack-coding", pack_id="coding")
    assert tool.owner_id == "lisa-pack-coding"


def test_owner_id_falls_back_only_in_test_factory() -> None:
    """The test factory supplies an owner when authored metadata is absent."""
    tool = _tool("t", provider_id="", pack_id="coding")
    assert tool.owner_id == "coding"


def test_owner_id_empty_when_definition_is_unbound() -> None:
    """Definitions without catalog attribution remain visibly unbound."""
    tool = _tool("t")
    assert tool.owner_id == ""


def test_owner_id_works_for_operations() -> None:
    """Operations expose the same canonical owner field."""
    op = _operation("o", provider_id="lisa-pack-memory", pack_id="memory")
    assert op.owner_id == "lisa-pack-memory"
    op_bare = _operation("o", pack_id="memory")
    assert op_bare.owner_id == "memory"


# ── resolve_capability precedence ────────────────────────────────────────


def test_resolve_capability_finds_operation() -> None:
    """resolve_capability returns the operation when the id is in the op catalog."""
    ops = {"recall": _operation("recall")}
    tools = {"other": _tool("other")}
    result = resolve_capability("recall", operation_catalog=ops, tool_catalog=tools)
    assert result is not None
    assert isinstance(result, OperationDefinition)
    assert result.operation_id == "recall"


def test_resolve_capability_finds_tool() -> None:
    """resolve_capability returns the tool when the id is only in the tool catalog."""
    ops = {"recall": _operation("recall")}
    tools = {"web_search": _tool("web_search")}
    result = resolve_capability("web_search", operation_catalog=ops, tool_catalog=tools)
    assert result is not None
    assert isinstance(result, ToolDefinition)
    assert result.tool_id == "web_search"


def test_resolve_capability_operation_precedence_on_collision() -> None:
    """On id collision, operations take precedence over tools."""
    shared_id = "shared"
    ops = {shared_id: _operation(shared_id)}
    tools = {shared_id: _tool(shared_id)}
    result = resolve_capability(shared_id, operation_catalog=ops, tool_catalog=tools)
    assert isinstance(result, OperationDefinition)
    assert result.operation_id == shared_id


def test_resolve_capability_returns_none_for_unknown() -> None:
    """resolve_capability returns None when the id is in neither catalog."""
    result = resolve_capability(
        "nope", operation_catalog={"a": _operation("a")}, tool_catalog={"b": _tool("b")}
    )
    assert result is None


def test_resolve_operation_definition_only_checks_op_catalog() -> None:
    """resolve_operation_definition does not consult the tool catalog."""
    {"shared": _tool("shared")}
    result = resolve_operation_definition("shared", operation_catalog={})
    assert result is None
    # Passing the tool catalog explicitly must not change the result.
    result2 = resolve_operation_definition("shared", operation_catalog={})
    assert result2 is None


def test_resolve_tool_definition_only_checks_tool_catalog() -> None:
    """resolve_tool_definition does not consult the operation catalog."""
    {"shared": _operation("shared")}
    assert resolve_tool_definition("shared", tool_catalog={}) is None
    assert resolve_tool_definition("shared", tool_catalog={"other": _tool("other")}) is None


def test_resolve_capability_for_turn_limit_precedence_matches_resolve_capability() -> None:
    """resolve_capability_for_turn_limit uses the same op-before-tool precedence."""
    shared = "shared"
    ops = {shared: _operation(shared, max_calls_per_turn=5)}
    tools = {shared: _tool(shared, max_calls_per_turn=9)}
    result = resolve_capability_for_turn_limit(shared, operation_catalog=ops, tool_catalog=tools)
    assert isinstance(result, OperationDefinition)
    assert result.max_calls_per_turn == 5  # operation wins, not the tool's 9


def test_resolve_capability_for_turn_limit_none_for_unknown() -> None:
    """MCP-native tools (no kernel definition) resolve to None for turn limits."""
    result = resolve_capability_for_turn_limit(
        "lisa-storage.read_model_manifest",
        operation_catalog={},
        tool_catalog={},
    )
    assert result is None


# ── validate_export_boundaries: empty mode_provider_id loud failure ──────


def _patch_workflow_catalog(
    monkeypatch: pytest.MonkeyPatch, modes: dict[str, WorkflowModeConfig]
) -> None:
    """Patch kernel.ipsar_fsm.get_workflow_catalog to return the given modes."""
    from kernel import ipsar_fsm

    def _fake_get_workflow_catalog() -> dict[str, WorkflowModeConfig]:
        return modes

    monkeypatch.setattr(ipsar_fsm, "get_workflow_catalog", _fake_get_workflow_catalog)


def test_validate_export_boundaries_raises_on_empty_mode_provider_id_with_refs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mode with empty mode_provider_id + capability refs fails loud.

    Before the audit fix, an empty provider id silently flagged every
    referenced capability as cross-pack (any non-empty op_provider !=
    '' is True). Now it raises OperationValidationError at startup.
    """
    from kernel.operations.registry import validate_export_boundaries
    from kernel.tools.registry import ToolValidationError  # noqa: F401 (import side effect)

    # Mode with empty provider references one operation.
    _patch_workflow_catalog(
        monkeypatch,
        {"bad": _mode("bad", mode_provider_id="", allowed_operation_ids=["recall"])},
    )
    ops = {"recall": _operation("recall", provider_id="lisa-pack-memory")}
    with pytest.raises(Exception) as exc_info:  # OperationValidationError
        validate_export_boundaries(operation_catalog=ops)
    assert "mode_provider_id" in str(exc_info.value)
    assert "bad" in str(exc_info.value)


def test_validate_export_boundaries_passes_when_mode_provider_id_resolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same-pack reference with a resolved mode_provider_id passes."""
    from kernel.operations.registry import validate_export_boundaries

    _patch_workflow_catalog(
        monkeypatch,
        {"ok": _mode("ok", mode_provider_id="lisa-pack-memory", allowed_operation_ids=["recall"])},
    )
    ops = {"recall": _operation("recall", provider_id="lisa-pack-memory")}
    # Should not raise.
    validate_export_boundaries(operation_catalog=ops)


def test_validate_export_boundaries_empty_provider_with_no_refs_is_ok(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mode with empty provider id and no capability refs does not raise.

    The empty-provider-id check only triggers when the mode actually
    references capabilities; a provider-less mode that references nothing
    is not a misconfiguration.
    """
    from kernel.operations.registry import validate_export_boundaries

    _patch_workflow_catalog(
        monkeypatch,
        {
            "empty": _mode(
                "empty", mode_provider_id="", allowed_operation_ids=[], allowed_tool_ids=[]
            )
        },
    )
    # Should not raise.
    validate_export_boundaries(operation_catalog={})


def test_validate_export_boundaries_rejects_cross_pack_non_exported_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mode referencing a non-exported tool from another pack is rejected."""
    from kernel.operations.registry import validate_export_boundaries
    from kernel.tools import registry as tool_registry

    _patch_workflow_catalog(
        monkeypatch,
        {
            "consumer": _mode(
                "consumer", mode_provider_id="lisa-pack-coding", allowed_tool_ids=["search"]
            )
        },
    )
    tools = {"search": _tool("search", provider_id="lisa-pack-brave", exportable=False)}
    monkeypatch.setattr(tool_registry, "get_tool_catalog", lambda: tools)
    with pytest.raises(Exception) as exc_info:
        validate_export_boundaries(operation_catalog={})
    assert "exportable" in str(exc_info.value)


def test_validate_export_boundaries_allows_cross_pack_exported_tool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exported tool from another pack is allowed."""
    from kernel.operations.registry import validate_export_boundaries
    from kernel.tools import registry as tool_registry

    _patch_workflow_catalog(
        monkeypatch,
        {
            "consumer": _mode(
                "consumer", mode_provider_id="lisa-pack-coding", allowed_tool_ids=["search"]
            )
        },
    )
    tools = {"search": _tool("search", provider_id="lisa-pack-brave", exportable=True)}
    monkeypatch.setattr(tool_registry, "get_tool_catalog", lambda: tools)
    # Should not raise.
    validate_export_boundaries(operation_catalog={})

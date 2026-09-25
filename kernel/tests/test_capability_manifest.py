"""Tests for kernel/capability_manifest.py — dynamic capability manifest compiler.

Verifies that the manifest compiled from live tool schemas is deterministic,
matches exactly what the model sees, and carries mode-level constraints.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from kernel.capability_manifest import (
    compile_capability_manifest,
    render_capability_manifest_context,
)
from kernel.models import CapabilityManifest


def _make_tool_schema(name: str, description: str = "") -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description or f"Execute {name}.",
            "parameters": {"type": "object", "properties": {}},
        },
    }


def test_compile_manifest_empty_schemas() -> None:
    """Empty tool_schemas produces an empty manifest with valid structure."""
    manifest = compile_capability_manifest([])

    assert isinstance(manifest, CapabilityManifest)
    assert manifest.entries == []
    assert "0 available" in manifest.summary
    assert manifest.metadata["kind_counts"] == {}


def test_compile_manifest_classifies_operations() -> None:
    """Tool names matching the operation catalog are classified as operations."""
    from kernel.operations.registry import get_operation_catalog

    catalog = get_operation_catalog()
    if not catalog:
        pytest.skip("Operation catalog not populated in this test environment")
    op_name = next(iter(catalog))

    manifest = compile_capability_manifest([_make_tool_schema(op_name)])

    assert len(manifest.entries) == 1
    entry = manifest.entries[0]
    assert entry.kind == "operation"
    assert entry.source == "operation_catalog"
    assert entry.status == "available"
    assert entry.confidence == 1.0
    assert entry.last_checked_at != ""


def test_compile_manifest_classifies_transition_tool_as_mode() -> None:
    """The transition_stage FSM tool is classified as kind='mode'."""
    manifest = compile_capability_manifest([_make_tool_schema("transition_stage")])

    assert len(manifest.entries) == 1
    entry = manifest.entries[0]
    assert entry.kind == "mode"
    assert entry.source == "workflow_fsm"


def test_compile_manifest_classifies_unknown_as_mcp_tool() -> None:
    """Tool names not in any catalog are classified as MCP tools."""
    manifest = compile_capability_manifest([_make_tool_schema("some_mcp_tool")])

    assert len(manifest.entries) == 1
    entry = manifest.entries[0]
    assert entry.kind == "tool"
    assert entry.source == "mcp_pool"


def test_compile_manifest_reflects_exact_tool_count() -> None:
    """The manifest entry count matches the tool_schemas count exactly."""
    schemas = [
        _make_tool_schema("tool_a"),
        _make_tool_schema("tool_b"),
        _make_tool_schema("tool_c"),
    ]

    manifest = compile_capability_manifest(schemas)

    assert len(manifest.entries) == 3
    names = {e.capability_id for e in manifest.entries}
    assert names == {"tool_a", "tool_b", "tool_c"}


def test_compile_manifest_records_mode_constraints() -> None:
    """When mode_config has allowed_tool_ids, the manifest records the constraint."""
    mode_config = SimpleNamespace(
        mode_id="remember",
        allowed_tool_ids=["memory.add", "memory.search"],
        tools_authoritative=False,
    )

    manifest = compile_capability_manifest(
        [_make_tool_schema("memory.add")], mode_config=mode_config
    )

    assert manifest.metadata["mode_id"] == "remember"
    assert manifest.metadata["allowed_tool_ids"] == ["memory.add", "memory.search"]
    assert manifest.metadata["tools_authoritative"] is False
    # Non-authoritative mode does not add an authoritative constraint
    assert all(c.value != "tools_authoritative" for c in manifest.entries[0].constraints)


def test_compile_manifest_records_authoritative_constraint() -> None:
    """When tools_authoritative=True, the manifest records the constraint."""
    mode_config = SimpleNamespace(
        mode_id="strict_mode",
        allowed_tool_ids=["only_this"],
        tools_authoritative=True,
    )

    manifest = compile_capability_manifest(
        [_make_tool_schema("only_this")], mode_config=mode_config
    )

    assert manifest.metadata["tools_authoritative"] is True
    # The manifest summary should mention "authoritative"
    assert "authoritative" in manifest.summary


def test_compile_manifest_is_deterministic() -> None:
    """Same inputs produce same outputs — no randomness or stale state."""
    schemas = [_make_tool_schema("tool_x"), _make_tool_schema("transition_stage")]

    manifest_1 = compile_capability_manifest(schemas, captured_at="2026-07-11T00:00:00Z")
    manifest_2 = compile_capability_manifest(schemas, captured_at="2026-07-11T00:00:00Z")

    assert manifest_1.model_dump() == manifest_2.model_dump()


def test_render_capability_manifest_context_includes_counts() -> None:
    """The rendered context block includes live counts and tool names."""
    manifest = compile_capability_manifest(
        [_make_tool_schema("tool_a"), _make_tool_schema("tool_b")],
        captured_at="2026-07-11T00:00:00Z",
    )

    rendered = render_capability_manifest_context(manifest)

    assert "## Capability Context (live this turn)" in rendered
    assert "available_tools: 2" in rendered
    assert "tool_a" in rendered
    assert "tool_b" in rendered


def test_render_capability_manifest_context_shows_mode() -> None:
    """The rendered context block shows the active mode."""
    mode_config = SimpleNamespace(
        mode_id="remember",
        allowed_tool_ids=[],
        tools_authoritative=True,
    )
    manifest = compile_capability_manifest(
        [_make_tool_schema("memory.add")], mode_config=mode_config
    )

    rendered = render_capability_manifest_context(manifest)

    assert "mode: remember" in rendered
    assert "tools_authoritative: yes" in rendered

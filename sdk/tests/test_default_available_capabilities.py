"""Tests for the default-available capability merge and export boundary.

Covers the SDK-side merge of ``default_available`` ops/tools into each
mode's effective ``allowed_*_ids`` (``sdk.content._catalog.
_merge_default_capabilities``) and the kernel-side cross-pack export check
(``kernel.operations.registry.validate_export_boundaries``).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sdk.content import (
    build_mode_catalog_json,
    build_operation_catalog_json,
    build_tool_catalog_json,
    resolve_mode_roots,
    resolve_operation_roots,
    resolve_tool_roots,
)


# ── Fixtures ────────────────────────────────────────────────────────────────


def _write_mode(
    root: Path,
    *,
    mode_id: str,
    provider_id: str,
    allowed_operation_ids: list[str] | None = None,
    allowed_tool_ids: list[str] | None = None,
    accept_default_capabilities: bool | None = None,
    excluded_operation_ids: list[str] | None = None,
    excluded_tool_ids: list[str] | None = None,
    description: str = "test mode",
) -> None:
    mode_dir = root / "modes" / mode_id
    mode_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "---",
        f"mode_id: {mode_id}",
        f"mode_provider_id: {provider_id}",
        "start_stage_id: none",
        "mode_kind: chat",
        f'description: "{description}"',
    ]
    if allowed_operation_ids is not None:
        lines.append(f"allowed_operation_ids: [{', '.join(allowed_operation_ids)}]")
    if allowed_tool_ids is not None:
        lines.append(f"allowed_tool_ids: [{', '.join(allowed_tool_ids)}]")
    if accept_default_capabilities is not None:
        lines.append(f"accept_default_capabilities: {str(accept_default_capabilities).lower()}")
    if excluded_operation_ids is not None:
        lines.append(f"excluded_operation_ids: [{', '.join(excluded_operation_ids)}]")
    if excluded_tool_ids is not None:
        lines.append(f"excluded_tool_ids: [{', '.join(excluded_tool_ids)}]")
    lines += ["---", "", f"# {mode_id}", "", "Test mode.", ""]
    (mode_dir / "_mode.md").write_text("\n".join(lines), encoding="utf-8")


def _write_operation(
    root: Path,
    *,
    operation_id: str,
    provider_id: str,
    exportable: bool = False,
    default_available: bool = False,
) -> None:
    ops_dir = root / "operations" / "research"
    ops_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "---",
        f"operation_id: {operation_id}",
        f"operation_provider_id: {provider_id}",
        f'description: "{operation_id} op"',
        "tool_ids: []",
        "mode_allowlist: []",
    ]
    if exportable:
        lines.append("exportable: true")
    if default_available:
        lines.append("default_available: true")
    lines += ["---", "", f"# {operation_id}", "", "Test operation.", ""]
    (ops_dir / f"{operation_id}.operation.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def _write_tool(
    root: Path,
    *,
    tool_id: str,
    provider_id: str,
    exportable: bool = False,
    default_available: bool = False,
) -> None:
    tools_dir = root / "tools" / tool_id
    tools_dir.mkdir(parents=True, exist_ok=True)
    (tools_dir / f"{tool_id}.py").write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "",
                "from kernel.models import ToolExecutionContext, ToolResult",
                "",
                "",
                "async def execute(ctx: ToolExecutionContext) -> ToolResult:",
                f"    return ToolResult(tool_id=ctx.tool_id or '{tool_id}', output_text='ok')",
                "",
            ]
        ),
        encoding="utf-8",
    )
    lines = [
        f"tool_id: {tool_id}",
        f"tool_provider_id: {provider_id}",
        f'description: "{tool_id} tool"',
        "mode_allowlist: null",
        "max_calls_per_turn: 3",
        "timeout_seconds: 30",
    ]
    if exportable:
        lines.append("exportable: true")
    if default_available:
        lines.append("default_available: true")
    lines.append("")
    (tools_dir / f"{tool_id}.tool.yml").write_text("\n".join(lines), encoding="utf-8")


def _build_catalogs(*roots: Path) -> tuple[dict, dict, dict]:
    """Build the three catalogs and return them as dicts (modes, ops, tools)."""
    from sdk.content import invalidate_catalog_cache

    invalidate_catalog_cache()
    mode_roots = resolve_mode_roots(list(roots))
    op_roots = resolve_operation_roots(list(roots))
    tool_roots = resolve_tool_roots(list(roots))
    op_json = build_operation_catalog_json(op_roots)
    tool_json = build_tool_catalog_json(tool_roots)
    mode_json = build_mode_catalog_json(
        mode_roots,
        operation_catalog_json=op_json,
        tool_catalog_json=tool_json,
    )
    return json.loads(mode_json), json.loads(op_json), json.loads(tool_json)


# ── Default-available merge ─────────────────────────────────────────────────


def test_default_available_operation_merged_into_empty_allowlist(tmp_path: Path) -> None:
    """A default_available op appears in a mode with an empty allowlist."""
    pack_a = tmp_path / "pack_a"
    pack_b = tmp_path / "pack_b"
    _write_mode(pack_a, mode_id="chat", provider_id="lisa-pack-a")
    _write_operation(
        pack_b,
        operation_id="recall_memory",
        provider_id="lisa-pack-b",
        exportable=True,
        default_available=True,
    )
    modes, _ops, _tools = _build_catalogs(pack_a, pack_b)
    assert "recall_memory" in modes["chat"]["allowed_operation_ids"]


def test_default_available_operation_absent_when_opted_out(tmp_path: Path) -> None:
    """A default_available op is absent when accept_default_capabilities is False."""
    pack_a = tmp_path / "pack_a"
    pack_b = tmp_path / "pack_b"
    _write_mode(
        pack_a,
        mode_id="chat",
        provider_id="lisa-pack-a",
        accept_default_capabilities=False,
    )
    _write_operation(
        pack_b,
        operation_id="recall_memory",
        provider_id="lisa-pack-b",
        exportable=True,
        default_available=True,
    )
    modes, _ops, _tools = _build_catalogs(pack_a, pack_b)
    assert "recall_memory" not in modes["chat"]["allowed_operation_ids"]


def test_excluded_operation_ids_removes_default(tmp_path: Path) -> None:
    """excluded_operation_ids removes a default-available op from the merge."""
    pack_a = tmp_path / "pack_a"
    pack_b = tmp_path / "pack_b"
    _write_mode(
        pack_a,
        mode_id="chat",
        provider_id="lisa-pack-a",
        excluded_operation_ids=["recall_memory"],
    )
    _write_operation(
        pack_b,
        operation_id="recall_memory",
        provider_id="lisa-pack-b",
        exportable=True,
        default_available=True,
    )
    modes, _ops, _tools = _build_catalogs(pack_a, pack_b)
    assert "recall_memory" not in modes["chat"]["allowed_operation_ids"]


def test_default_available_tool_merged_into_empty_allowlist(tmp_path: Path) -> None:
    """A default_available tool appears in a mode with an empty tool allowlist."""
    pack_a = tmp_path / "pack_a"
    pack_b = tmp_path / "pack_b"
    _write_mode(pack_a, mode_id="chat", provider_id="lisa-pack-a")
    _write_tool(
        pack_b,
        tool_id="ambient_tool",
        provider_id="lisa-pack-b",
        exportable=True,
        default_available=True,
    )
    modes, _ops, _tools = _build_catalogs(pack_a, pack_b)
    assert "ambient_tool" in modes["chat"]["allowed_tool_ids"]


def test_explicit_allowlist_preserved_with_defaults_appended(tmp_path: Path) -> None:
    """The mode's authored order is preserved; defaults appended at the end."""
    pack_a = tmp_path / "pack_a"
    pack_b = tmp_path / "pack_b"
    _write_mode(
        pack_a,
        mode_id="chat",
        provider_id="lisa-pack-a",
        allowed_operation_ids=["native_op"],
    )
    _write_operation(
        pack_b,
        operation_id="recall_memory",
        provider_id="lisa-pack-b",
        exportable=True,
        default_available=True,
    )
    modes, _ops, _tools = _build_catalogs(pack_a, pack_b)
    assert modes["chat"]["allowed_operation_ids"] == ["native_op", "recall_memory"]


def test_non_default_available_operation_not_merged(tmp_path: Path) -> None:
    """An op without default_available is not merged into the allowlist."""
    pack_a = tmp_path / "pack_a"
    pack_b = tmp_path / "pack_b"
    _write_mode(pack_a, mode_id="chat", provider_id="lisa-pack-a")
    _write_operation(
        pack_b,
        operation_id="private_op",
        provider_id="lisa-pack-b",
        exportable=True,
        default_available=False,
    )
    modes, _ops, _tools = _build_catalogs(pack_a, pack_b)
    assert "private_op" not in modes["chat"]["allowed_operation_ids"]


def test_merge_idempotent_when_already_listed(tmp_path: Path) -> None:
    """A default_available op already in the allowlist is not duplicated."""
    pack_a = tmp_path / "pack_a"
    pack_b = tmp_path / "pack_b"
    _write_mode(
        pack_a,
        mode_id="chat",
        provider_id="lisa-pack-a",
        allowed_operation_ids=["recall_memory"],
    )
    _write_operation(
        pack_b,
        operation_id="recall_memory",
        provider_id="lisa-pack-b",
        exportable=True,
        default_available=True,
    )
    modes, _ops, _tools = _build_catalogs(pack_a, pack_b)
    assert modes["chat"]["allowed_operation_ids"] == ["recall_memory"]


# ── Export boundary (kernel) ────────────────────────────────────────────────


def test_validate_export_boundaries_rejects_non_exported_cross_pack_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mode referencing a non-exported op from another pack is rejected."""
    from kernel.operations.registry import validate_export_boundaries
    from kernel.operations.registry import OperationValidationError

    op = SimpleNamespace(
        operation_id="private_op",
        pack_id="",
        provider_id="lisa-pack-b",
        exportable=False,
        effective_provider_id=lambda: "lisa-pack-b",
    )
    mode = SimpleNamespace(
        mode_id="chat",
        mode_provider_id="lisa-pack-a",
        allowed_operation_ids=["private_op"],
        allowed_tool_ids=[],
        entry_operation_ids=[],
        pre_turn_operation_ids=[],
        post_turn_operation_ids=[],
    )
    monkeypatch.setattr(
        "kernel.operations.registry.get_operation_catalog",
        lambda: {"private_op": op},
    )
    monkeypatch.setattr(
        "kernel.tools.registry.get_tool_catalog",
        lambda: {},
    )
    monkeypatch.setattr(
        "kernel.ipsar_fsm.get_workflow_catalog",
        lambda: {"chat": mode},
    )
    with pytest.raises(OperationValidationError, match="not exported"):
        validate_export_boundaries({"private_op": op})


def test_validate_export_boundaries_allows_exported_cross_pack_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mode referencing an exported op from another pack is allowed."""
    from kernel.operations.registry import validate_export_boundaries

    op = SimpleNamespace(
        operation_id="recall_memory",
        pack_id="",
        provider_id="lisa-pack-memory",
        exportable=True,
        effective_provider_id=lambda: "lisa-pack-memory",
    )
    mode = SimpleNamespace(
        mode_id="chat",
        mode_provider_id="lisa-pack-core",
        allowed_operation_ids=["recall_memory"],
        allowed_tool_ids=[],
        entry_operation_ids=[],
        pre_turn_operation_ids=[],
        post_turn_operation_ids=[],
    )
    monkeypatch.setattr(
        "kernel.operations.registry.get_operation_catalog",
        lambda: {"recall_memory": op},
    )
    monkeypatch.setattr(
        "kernel.tools.registry.get_tool_catalog",
        lambda: {},
    )
    monkeypatch.setattr(
        "kernel.ipsar_fsm.get_workflow_catalog",
        lambda: {"chat": mode},
    )
    # Should not raise.
    validate_export_boundaries({"recall_memory": op})


def test_validate_export_boundaries_allows_same_pack_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mode referencing an op from its own pack is allowed regardless of exportable."""
    from kernel.operations.registry import validate_export_boundaries

    op = SimpleNamespace(
        operation_id="native_op",
        pack_id="",
        provider_id="lisa-pack-a",
        exportable=False,
        effective_provider_id=lambda: "lisa-pack-a",
    )
    mode = SimpleNamespace(
        mode_id="chat",
        mode_provider_id="lisa-pack-a",
        allowed_operation_ids=["native_op"],
        allowed_tool_ids=[],
        entry_operation_ids=[],
        pre_turn_operation_ids=[],
        post_turn_operation_ids=[],
    )
    monkeypatch.setattr(
        "kernel.operations.registry.get_operation_catalog",
        lambda: {"native_op": op},
    )
    monkeypatch.setattr(
        "kernel.tools.registry.get_tool_catalog",
        lambda: {},
    )
    monkeypatch.setattr(
        "kernel.ipsar_fsm.get_workflow_catalog",
        lambda: {"chat": mode},
    )
    # Should not raise.
    validate_export_boundaries({"native_op": op})


def test_validate_export_boundaries_allows_same_pack_with_prefix_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mode referencing an op from its own pack is allowed even when the
    capability's inferred provider id is the bare slug ('coding') and the
    mode's provider id is the full distribution name ('lisa-pack-coding').
    This is the real-world case that motivated the _provider_slug helper."""
    from kernel.operations.registry import validate_export_boundaries

    op = SimpleNamespace(
        operation_id="coding_list_directory",
        pack_id="",
        provider_id="coding",
        exportable=False,
        effective_provider_id=lambda: "coding",
    )
    mode = SimpleNamespace(
        mode_id="coding",
        mode_provider_id="lisa-pack-coding",
        allowed_operation_ids=["coding_list_directory"],
        allowed_tool_ids=[],
        entry_operation_ids=[],
        pre_turn_operation_ids=[],
        post_turn_operation_ids=[],
    )
    monkeypatch.setattr(
        "kernel.operations.registry.get_operation_catalog",
        lambda: {"coding_list_directory": op},
    )
    monkeypatch.setattr(
        "kernel.tools.registry.get_tool_catalog",
        lambda: {},
    )
    monkeypatch.setattr(
        "kernel.ipsar_fsm.get_workflow_catalog",
        lambda: {"coding": mode},
    )
    # Should not raise: 'coding' slug matches 'lisa-pack-coding' after stripping.
    validate_export_boundaries({"coding_list_directory": op})
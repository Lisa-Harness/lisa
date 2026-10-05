"""``timeout_seconds: 0`` must survive the catalog loaders.

Regression (2026-10-05 shell turn): all three catalog loaders used
``int(metadata.get("timeout_seconds", 30) or 30)`` — the falsy coercion
rewrote the documented "``0`` disables the wrapper timeout" declaration into
the 30s default, so a pack author had no expressible way to declare a
self-bounding tool. The kernel honors ``0`` (it skips ``asyncio.wait_for``);
the loaders silently destroyed the value before the kernel ever saw it.

Pins the loader side of the contract for the runtime tool, runtime operation,
and markdown-operation paths:
* an explicit ``timeout_seconds: 0`` round-trips as ``0``;
* a declared ``-5`` still round-trips verbatim (validation against
  negatives stays in the kernel's catalog validators, which raise);
* omitted keeps the documented default (30 tools / 60 operations).
"""

from __future__ import annotations

import json
from pathlib import Path

from sdk.content import (
    build_operation_catalog_json,
    build_tool_catalog_json,
    invalidate_catalog_cache,
    resolve_operation_roots,
    resolve_tool_roots,
)


def _write_runtime_tool(
    root: Path, *, timeout_seconds: int | None, null_value: bool = False
) -> None:
    tools_dir = root / "tools" / "probe"
    tools_dir.mkdir(parents=True, exist_ok=True)
    (tools_dir / "probe.py").write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "",
                "from kernel.models import ToolExecutionContext, ToolResult",
                "",
                "",
                "async def execute(ctx: ToolExecutionContext) -> ToolResult:",
                "    return ToolResult(tool_id='probe', output_text='ok')",
                "",
            ]
        ),
        encoding="utf-8",
    )
    lines = [
        "tool_id: probe",
        "tool_provider_id: test-provider",
        'description: "probe tool"',
        "mode_allowlist: null",
    ]
    if null_value:
        # `timeout_seconds:` with no value — YAML parses this as None. A
        # common authoring typo that must degrade to the default, not crash
        # the whole catalog build (2026-10-05 review, C2/C3).
        lines.append("timeout_seconds:")
    elif timeout_seconds is not None:
        lines.append(f"timeout_seconds: {timeout_seconds}")
    lines.append("")
    (tools_dir / "probe.tool.yml").write_text("\n".join(lines), encoding="utf-8")


def _write_runtime_operation(
    root: Path, *, timeout_seconds: int | None, null_value: bool = False
) -> None:
    ops_dir = root / "operations" / "probe"
    ops_dir.mkdir(parents=True, exist_ok=True)
    (ops_dir / "probe.py").write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "",
                "from kernel.models import OperationExecutionContext, OperationResult",
                "",
                "",
                "async def execute(ctx: OperationExecutionContext) -> OperationResult:",
                "    return OperationResult(output_text='ok')",
                "",
            ]
        ),
        encoding="utf-8",
    )
    lines = [
        "operation_id: probe_op",
        "operation_provider_id: test-provider",
        'description: "probe op"',
        "mode_allowlist: []",
    ]
    if null_value:
        lines.append("timeout_seconds:")
    elif timeout_seconds is not None:
        lines.append(f"timeout_seconds: {timeout_seconds}")
    lines.append("")
    (ops_dir / "probe.yml").write_text("\n".join(lines), encoding="utf-8")


def _write_markdown_operation(
    root: Path, *, timeout_seconds: int | None, null_value: bool = False
) -> None:
    ops_dir = root / "operations" / "probe"
    ops_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "---",
        "operation_id: probe_md",
        "operation_provider_id: test-provider",
        'description: "probe md op"',
        "tool_ids: []",
        "mode_allowlist: []",
    ]
    if null_value:
        lines.append("timeout_seconds:")
    elif timeout_seconds is not None:
        lines.append(f"timeout_seconds: {timeout_seconds}")
    lines += ["---", "", "# Probe", "", "Body.", ""]
    (ops_dir / "probe_md.operation.md").write_text("\n".join(lines), encoding="utf-8")


def _tool_catalog(*roots: Path) -> dict:
    invalidate_catalog_cache()
    return json.loads(build_tool_catalog_json(resolve_tool_roots(list(roots))))


def _operation_catalog(*roots: Path) -> dict:
    invalidate_catalog_cache()
    return json.loads(build_operation_catalog_json(resolve_operation_roots(list(roots))))


class TestRuntimeToolTimeoutZero:
    """``.tool.yml`` → tool catalog JSON."""

    def test_declared_zero_round_trips(self, tmp_path: Path) -> None:
        _write_runtime_tool(tmp_path, timeout_seconds=0)
        assert _tool_catalog(tmp_path)["probe"]["timeout_seconds"] == 0

    def test_declared_negative_round_trips(self, tmp_path: Path) -> None:
        _write_runtime_tool(tmp_path, timeout_seconds=-5)
        assert _tool_catalog(tmp_path)["probe"]["timeout_seconds"] == -5

    def test_omitted_keeps_default(self, tmp_path: Path) -> None:
        _write_runtime_tool(tmp_path, timeout_seconds=None)
        assert _tool_catalog(tmp_path)["probe"]["timeout_seconds"] == 30

    def test_null_value_degrades_to_default_not_crash(self, tmp_path: Path) -> None:
        """`timeout_seconds:` with no value (YAML None) keeps the default.

        Regression (2026-10-05 review C2): the verbatim-passthrough fix
        crashed the entire catalog build with a TypeError on this authoring
        typo; null must mean undeclared, not raise.
        """
        _write_runtime_tool(tmp_path, timeout_seconds=None, null_value=True)
        assert _tool_catalog(tmp_path)["probe"]["timeout_seconds"] == 30


class TestRuntimeOperationTimeoutZero:
    """``.yml`` → operation catalog JSON (runtime operations)."""

    def test_declared_zero_round_trips(self, tmp_path: Path) -> None:
        _write_runtime_operation(tmp_path, timeout_seconds=0)
        assert _operation_catalog(tmp_path)["probe_op"]["timeout_seconds"] == 0

    def test_declared_negative_round_trips(self, tmp_path: Path) -> None:
        _write_runtime_operation(tmp_path, timeout_seconds=-5)
        assert _operation_catalog(tmp_path)["probe_op"]["timeout_seconds"] == -5

    def test_omitted_keeps_default(self, tmp_path: Path) -> None:
        _write_runtime_operation(tmp_path, timeout_seconds=None)
        assert _operation_catalog(tmp_path)["probe_op"]["timeout_seconds"] == 60

    def test_null_value_degrades_to_default_not_crash(self, tmp_path: Path) -> None:
        """`timeout_seconds:` with no value (YAML None) keeps the default."""
        _write_runtime_operation(tmp_path, timeout_seconds=None, null_value=True)
        assert _operation_catalog(tmp_path)["probe_op"]["timeout_seconds"] == 60


class TestMarkdownOperationTimeoutZero:
    """``.operation.md`` front matter → operation catalog JSON."""

    def test_declared_zero_round_trips(self, tmp_path: Path) -> None:
        _write_markdown_operation(tmp_path, timeout_seconds=0)
        assert _operation_catalog(tmp_path)["probe_md"]["timeout_seconds"] == 0

    def test_declared_negative_round_trips(self, tmp_path: Path) -> None:
        _write_markdown_operation(tmp_path, timeout_seconds=-5)
        assert _operation_catalog(tmp_path)["probe_md"]["timeout_seconds"] == -5

    def test_omitted_keeps_default(self, tmp_path: Path) -> None:
        _write_markdown_operation(tmp_path, timeout_seconds=None)
        assert _operation_catalog(tmp_path)["probe_md"]["timeout_seconds"] == 60

    def test_null_value_degrades_to_default_not_crash(self, tmp_path: Path) -> None:
        """`timeout_seconds:` with no value (YAML None) keeps the default."""
        _write_markdown_operation(tmp_path, timeout_seconds=None, null_value=True)
        assert _operation_catalog(tmp_path)["probe_md"]["timeout_seconds"] == 60

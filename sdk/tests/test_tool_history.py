"""Pack-declared history semantics through the SDK catalog loaders.

Covers ``.tool.yml`` ``history:`` blocks (shape validation in the SDK,
parameter/hook validation on kernel import) and the ``working_set_tokens``
mode front-matter field.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from kernel.context.mcp_catalog import _deserialize_mode, _deserialize_tool
from kernel.history import HistoryPolicy, HistoryTemplateError
from sdk.content import (
    build_mode_catalog_json,
    build_tool_catalog_json,
    invalidate_catalog_cache,
    resolve_mode_roots,
    resolve_tool_roots,
)

_MODULE = """from __future__ import annotations

from kernel.models import ToolExecutionContext, ToolResult
from pydantic import BaseModel


class Parameters(BaseModel):
    artifact_id: str
    path: str


async def execute(ctx: ToolExecutionContext) -> ToolResult:
    return ToolResult(tool_id=ctx.tool_id, output_text="ok")
"""


@pytest.fixture(autouse=True)
def _importable_tools(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The kernel imports tool modules by their pack-relative dotted path.
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in [m for m in sys.modules if m == "tools" or m.startswith("tools.")]:
        monkeypatch.delitem(sys.modules, name)


def _write_tool(root: Path, tool_id: str, history_yaml: str) -> None:
    tools_dir = root / "tools" / tool_id
    tools_dir.mkdir(parents=True, exist_ok=True)
    (tools_dir / f"{tool_id}.py").write_text(_MODULE, encoding="utf-8")
    (tools_dir / f"{tool_id}.tool.yml").write_text(
        "\n".join(
            [
                f"tool_id: {tool_id}",
                "tool_provider_id: test-provider",
                f'description: "{tool_id} tool"',
                "mode_allowlist: null",
                history_yaml,
                "",
            ]
        ),
        encoding="utf-8",
    )


def _tool_catalog(root: Path) -> dict:
    invalidate_catalog_cache()
    return json.loads(build_tool_catalog_json(resolve_tool_roots([root])))


def test_history_block_round_trips_into_kernel_definition(tmp_path: Path) -> None:
    _write_tool(
        tmp_path,
        "reader",
        "\n".join(
            [
                "history:",
                '  key: "read:{artifact_id}:{path}@{result.revision}"',
                '  invalidates: ["status:{artifact_id}"]',
                '  anchor: "reader:{artifact_id}:{path}"',
            ]
        ),
    )
    entry = _tool_catalog(tmp_path)["reader"]
    assert entry["history"]["key"] == ["read:{artifact_id}:{path}@{result.revision}"]
    definition = _deserialize_tool(entry)
    assert definition.implementation_module is not None
    assert definition.history == HistoryPolicy(
        key=("read:{artifact_id}:{path}@{result.revision}",),
        invalidates=("status:{artifact_id}",),
        anchor="reader:{artifact_id}:{path}",
    )


def test_tool_without_history_block_has_none(tmp_path: Path) -> None:
    _write_tool(tmp_path, "plain", "")
    entry = _tool_catalog(tmp_path)["plain"]
    assert entry["history"] is None
    assert _deserialize_tool(entry).history is None


def test_malformed_history_block_fails_sdk_load(tmp_path: Path) -> None:
    _write_tool(tmp_path, "bad", "history:\n  scope: forever")
    with pytest.raises(ValueError, match="invalid history block"):
        _tool_catalog(tmp_path)


def test_unknown_parameter_fails_kernel_import(tmp_path: Path) -> None:
    _write_tool(tmp_path, "typo", 'history:\n  key: "read:{artifact}"')
    entry = _tool_catalog(tmp_path)["typo"]
    with pytest.raises(HistoryTemplateError, match="unknown parameters"):
        _deserialize_tool(entry)


def test_mode_working_set_tokens_round_trips(tmp_path: Path) -> None:
    mode_dir = tmp_path / "modes" / "worker"
    mode_dir.mkdir(parents=True)
    (mode_dir / "_mode.md").write_text(
        "\n".join(
            [
                "---",
                "mode_id: worker",
                "mode_provider_id: test-provider",
                "start_stage_id: none",
                "mode_kind: chat",
                'description: "worker"',
                "working_set_tokens: 48000",
                "---",
                "",
                "# worker",
                "",
            ]
        ),
        encoding="utf-8",
    )
    invalidate_catalog_cache()
    modes = json.loads(
        build_mode_catalog_json(
            resolve_mode_roots([tmp_path]),
            operation_catalog_json="{}",
            tool_catalog_json="{}",
        )
    )
    mode = modes["worker"] if isinstance(modes, dict) else modes[0]
    assert mode["working_set_tokens"] == 48000
    assert _deserialize_mode(mode).working_set_tokens == 48000

"""Pack-declared validation-recovery content through the SDK catalog loaders.

Covers ``.tool.yml`` ``validation_recovery:`` blocks: the content-path
declaration round-trips through the SDK scan into the kernel
``ToolDefinition``, a missing pack-relative resource fails catalog build,
and tools without the declaration stay ``None`` (coding-edit-schema-recovery).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from kernel.context.mcp_catalog import _deserialize_tool
from sdk.content import (
    build_tool_catalog_json,
    invalidate_catalog_cache,
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


def _write_tool(root: Path, tool_id: str, recovery_yaml: str, *, content: str = "") -> None:
    tools_dir = root / "tools" / tool_id
    tools_dir.mkdir(parents=True, exist_ok=True)
    (tools_dir / f"{tool_id}.py").write_text(_MODULE, encoding="utf-8")
    if content:
        (tools_dir / "recovery.md").write_text(content, encoding="utf-8")
    (tools_dir / f"{tool_id}.tool.yml").write_text(
        "\n".join(
            [
                f"tool_id: {tool_id}",
                "tool_provider_id: test-provider",
                f'description: "{tool_id} tool"',
                "mode_allowlist: null",
                recovery_yaml,
                "",
            ]
        ),
        encoding="utf-8",
    )


def _tool_catalog(root: Path) -> dict:
    invalidate_catalog_cache()
    return json.loads(build_tool_catalog_json(resolve_tool_roots([root])))


def test_validation_recovery_block_round_trips_into_kernel_definition(
    tmp_path: Path,
) -> None:
    _write_tool(
        tmp_path,
        "editor",
        "validation_recovery:\n  content_path: tools/editor/recovery.md",
        content="## digest_format\n\nRe-read the file.\n",
    )
    entry = _tool_catalog(tmp_path)["editor"]
    assert entry["validation_recovery"] == {"content_path": "tools/editor/recovery.md"}
    definition = _deserialize_tool(entry)
    assert definition.implementation_module is not None
    assert definition.validation_recovery is not None
    assert definition.validation_recovery.content_path == "tools/editor/recovery.md"


def test_tool_without_declaration_has_none(tmp_path: Path) -> None:
    _write_tool(tmp_path, "plain", "")
    entry = _tool_catalog(tmp_path)["plain"]
    assert entry["validation_recovery"] is None
    assert _deserialize_tool(entry).validation_recovery is None


def test_missing_content_fails_sdk_load(tmp_path: Path) -> None:
    _write_tool(
        tmp_path, "absent", "validation_recovery:\n  content_path: tools/absent/recovery.md"
    )
    with pytest.raises(ValueError, match="validation_recovery content not found in pack"):
        _tool_catalog(tmp_path)


def test_absolute_content_path_fails_sdk_load(tmp_path: Path) -> None:
    _write_tool(
        tmp_path,
        "escape",
        "validation_recovery:\n  content_path: /etc/passwd",
    )
    with pytest.raises(ValueError, match="pack-relative"):
        _tool_catalog(tmp_path)


def test_parent_traversal_content_path_fails_sdk_load(tmp_path: Path) -> None:
    _write_tool(
        tmp_path,
        "traversal",
        "validation_recovery:\n  content_path: ../outside/recovery.md",
    )
    with pytest.raises(ValueError, match="pack-relative"):
        _tool_catalog(tmp_path)


def test_empty_content_path_fails_sdk_load(tmp_path: Path) -> None:
    _write_tool(tmp_path, "empty", "validation_recovery:\n  content_path: ''")
    with pytest.raises(ValueError, match="must not be empty"):
        _tool_catalog(tmp_path)

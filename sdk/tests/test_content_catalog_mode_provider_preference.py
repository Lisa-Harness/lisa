from __future__ import annotations

import json
from pathlib import Path

import pytest

from sdk.content._catalog import (
    CatalogCollisionError,
    build_mode_catalog_json,
    build_operation_catalog_json,
    build_tool_catalog_json,
    resolve_mode_roots,
    resolve_operation_roots,
    resolve_tool_roots,
)


def _write_default_mode(root: Path, *, provider_id: str, description: str) -> None:
    mode_dir = root / "modes" / "default"
    mode_dir.mkdir(parents=True, exist_ok=True)
    (mode_dir / "_mode.md").write_text(
        "\n".join(
            [
                "---",
                "mode_id: default",
                f"mode_provider_id: {provider_id}",
                "start_stage_id: none",
                "mode_kind: chat",
                f'description: "{description}"',
                "---",
                "",
                "# Default",
                "",
                "Provider test mode.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_operation(root: Path, *, operation_id: str, description: str) -> None:
    operations_dir = root / "operations" / "research"
    operations_dir.mkdir(parents=True, exist_ok=True)
    (operations_dir / f"{operation_id}.operation.md").write_text(
        "\n".join(
            [
                "---",
                f"operation_id: {operation_id}",
                f'description: "{description}"',
                "tool_ids: [web_search]",
                "mode_allowlist: [search]",
                "---",
                "",
                "# Operation",
                "",
                "Provider test operation.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_tool(root: Path, *, tool_id: str, description: str) -> None:
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
                "    return ToolResult(tool_id=ctx.tool_id or 'web_search', output_text='ok')",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (tools_dir / f"{tool_id}.tool.yml").write_text(
        "\n".join(
            [
                f"tool_id: {tool_id}",
                f'description: "{description}"',
                "mode_allowlist: null",
                "max_calls_per_turn: 3",
                "timeout_seconds: 30",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_mode_collision_uses_session_provider_selection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    core = tmp_path / "core_pack"
    learning = tmp_path / "learning_pack"
    _write_default_mode(core, provider_id="lisa-pack-core", description="core default")
    _write_default_mode(learning, provider_id="lisa-pack-learning", description="learning default")

    monkeypatch.setenv(
        "LISA_SESSION_PROVIDER_SELECTIONS",
        json.dumps(
            {
                "modes": {
                    "default": "lisa-pack-learning",
                }
            }
        ),
    )

    mode_roots = resolve_mode_roots([core, learning])
    catalog = json.loads(build_mode_catalog_json(mode_roots))

    assert catalog["default"]["mode_provider_id"] == "lisa-pack-learning"
    assert catalog["default"]["description"] == "learning default"


def test_mode_collision_requires_explicit_session_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    core = tmp_path / "core_pack"
    learning = tmp_path / "learning_pack"
    _write_default_mode(core, provider_id="lisa-pack-core", description="core default")
    _write_default_mode(learning, provider_id="lisa-pack-learning", description="learning default")

    monkeypatch.delenv("LISA_SESSION_PROVIDER_SELECTIONS", raising=False)

    mode_roots = resolve_mode_roots([core, learning])
    with pytest.raises(CatalogCollisionError, match=r"providers\.modes"):
        build_mode_catalog_json(mode_roots)


def test_operation_collision_uses_provider_preferences(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    search = tmp_path / "search_pack"
    brave = tmp_path / "brave_pack"
    _write_operation(search, operation_id="web_search", description="search operation")
    _write_operation(brave, operation_id="web_search", description="brave operation")

    monkeypatch.setenv(
        "LISA_SESSION_PROVIDER_SELECTIONS",
        json.dumps(
            {
                "operations": {
                    "web_search": "brave-pack",
                }
            }
        ),
    )

    operation_roots = resolve_operation_roots([search, brave])
    catalog = json.loads(build_operation_catalog_json(operation_roots))

    assert catalog["web_search"]["description"] == "brave operation"
    assert catalog["web_search"]["operation_provider_id"] == "brave-pack"


def test_tool_collision_uses_provider_preferences(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    search = tmp_path / "search_pack"
    brave = tmp_path / "brave_pack"
    _write_tool(search, tool_id="web_search", description="search tool")
    _write_tool(brave, tool_id="web_search", description="brave tool")

    monkeypatch.setenv(
        "LISA_SESSION_PROVIDER_SELECTIONS",
        json.dumps(
            {
                "tools": {
                    "web_search": "brave-pack",
                }
            }
        ),
    )

    tool_roots = resolve_tool_roots([search, brave])
    catalog = json.loads(build_tool_catalog_json(tool_roots))

    assert catalog["web_search"]["description"] == "brave tool"
    assert catalog["web_search"]["tool_provider_id"] == "brave-pack"


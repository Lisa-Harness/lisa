"""Tests for the approval-gated core-pack storage move wrapper."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from kernel.models import ToolExecutionContext, ToolResult
from sdk.packs.core.tools.move_file import move_file


class _FakeStorage:
    def __init__(self, *, error: str = "", receipt: Mapping[str, Any] | None = None) -> None:
        self.error = error
        self.receipt = receipt or {
            "ok": True,
            "source": "draft.md",
            "destination": "archive/draft.md",
            "bytes": 8,
            "sha256": "a" * 64,
            "durable": True,
        }
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def __call__(self, tool_id: str, arguments: Mapping[str, Any]) -> ToolResult:
        self.calls.append((tool_id, dict(arguments)))
        if self.error:
            return ToolResult(tool_id=tool_id, error=self.error)
        return ToolResult(tool_id=tool_id, output_text=json.dumps(self.receipt))


def _ctx(parameters: dict[str, Any], capability: _FakeStorage | None) -> ToolExecutionContext:
    return ToolExecutionContext(
        tool_id="move_file",
        parameters=parameters,
        mcp_tool_call=capability,
    )


@pytest.mark.asyncio
async def test_move_file_delegates_to_optional_storage_capability() -> None:
    storage = _FakeStorage()

    result = await move_file.execute(
        _ctx(
            {
                "source": "draft.md",
                "destination": "archive/draft.md",
                "expected_sha256": "A" * 64,
                "create_parents": True,
            },
            storage,
        )
    )

    assert result.error == ""
    assert storage.calls == [
        (
            "storage.move_file",
            {
                "source": "draft.md",
                "destination": "archive/draft.md",
                "expected_sha256": "A" * 64,
                "create_parents": True,
            },
        )
    ]
    assert result.output_json["destination"] == "archive/draft.md"


@pytest.mark.asyncio
async def test_move_file_without_mcp_seam_fails_loudly() -> None:
    result = await move_file.execute(_ctx({"source": "a.txt", "destination": "b.txt"}, None))
    assert "storage.move_file" in result.error


@pytest.mark.asyncio
async def test_move_file_surfaces_storage_error() -> None:
    result = await move_file.execute(
        _ctx(
            {"source": "a.txt", "destination": "b.txt"},
            _FakeStorage(error="destination already exists"),
        )
    )
    assert result.error == "destination already exists"


@pytest.mark.asyncio
async def test_move_file_rejects_bad_hash_before_calling_storage() -> None:
    storage = _FakeStorage()
    result = await move_file.execute(
        _ctx(
            {
                "source": "a.txt",
                "destination": "b.txt",
                "expected_sha256": "not-a-hash",
            },
            storage,
        )
    )
    assert "Invalid move_file parameters" in result.error
    assert storage.calls == []


def test_move_file_catalog_requires_approval_and_default_mode_opts_in() -> None:
    import yaml

    core_root = Path(__file__).resolve().parents[1] / "src" / "sdk" / "packs" / "core"
    metadata = yaml.safe_load(
        (core_root / "tools" / "move_file" / "move_file.tool.yml").read_text(encoding="utf-8")
    )
    assert metadata["effect"] == "write"
    assert metadata["approval_required"] is True
    assert not metadata.get("default_available", False)
    default_mode = (core_root / "modes" / "default" / "_mode.md").read_text(encoding="utf-8")
    assert "allowed_tool_ids: [move_file]" in default_mode

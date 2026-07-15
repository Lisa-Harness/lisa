"""Security and bounded-behavior tests for local coding read tools."""

from __future__ import annotations

from pathlib import Path

import pytest

from kernel.models import ToolExecutionContext
from sdk.packs.coding.config import get_coding_config
from sdk.packs.coding.local_files import CodingPathError, resolve_relative_path
from sdk.packs.coding.tools.find_files.find_files import execute as find_files
from sdk.packs.coding.tools.read_file.read_file import execute as read_file
from sdk.packs.coding.tools.search_code.search_code import execute as search_code


def _context(root: Path, parameters: dict[str, object]) -> ToolExecutionContext:
    return ToolExecutionContext(
        tool_id="test",
        mode_id="coding",
        parameters=parameters,
        pack_config={"workspace_root": str(root), "max_file_bytes": 1024},
    )


def _workspace(tmp_path: Path) -> Path:
    (tmp_path / ".git").mkdir()
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("def hello():\n    return 'ok'\n", encoding="utf-8")
    return tmp_path


def test_resolve_relative_path_rejects_traversal(tmp_path: Path) -> None:
    """Workspace tools refuse paths outside their configured root."""

    root = _workspace(tmp_path)
    context = _context(root, {})

    with pytest.raises(CodingPathError):
        resolve_relative_path(get_coding_config(context), "../outside.txt")


@pytest.mark.asyncio
async def test_read_file_rejects_binary_and_bounds_text(tmp_path: Path) -> None:
    """Binary input is refused and textual input stops at the byte policy."""

    root = _workspace(tmp_path)
    (root / "binary.dat").write_bytes(b"\x00not source")
    binary = await read_file(_context(root, {"path": "binary.dat"}))
    assert "Binary files" in binary.error

    (root / "large.txt").write_text("a" * 2000, encoding="utf-8")
    large = await read_file(_context(root, {"path": "large.txt"}))
    assert large.output_json["truncated_by_bytes"] is True


@pytest.mark.asyncio
async def test_find_and_search_treat_injection_shaped_input_as_data(tmp_path: Path) -> None:
    """File matching and ripgrep never execute query text through a shell."""

    root = _workspace(tmp_path)
    found = await find_files(_context(root, {"pattern": "*.py"}))
    assert found.output_json["files"] == ["src/main.py"]

    marker = root / "injected.txt"
    searched = await search_code(_context(root, {"query": "$(New-Item injected.txt)"}))
    assert marker.exists() is False
    assert searched.error == "" or "ripgrep" in searched.error

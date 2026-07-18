"""Tests for the checkpoint pack slash-command write-back fix.

Covers the bug fix landed alongside the compaction pack:
``handle_summarize_history`` previously discarded
``result.output_json["turn_history"]`` and never persisted the compacted
history to the live thread JSONL. It now uses the host's
``write_turn_history_fn`` seam (when wired) or falls back to a direct
JSONL overwrite.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from sdk.packs.checkpoint.commands import (
    _persist_compacted_history,
    handle_summarize_history,
)

pytestmark = pytest.mark.asyncio


def _runtime_stub(
    *,
    thread_path: Path | None,
    write_turn_history_fn: Any = None,
) -> Any:
    """Build a minimal HostRuntime-shaped stub for command tests."""

    class _Cfg:
        user_slug = "tester"
        surface_type = "personal_dm"
        surface_id = "host-console"
        memory_scope = "personal"

    class _Rt:
        def __init__(self) -> None:
            self.config = _Cfg()
            self.thread_path = thread_path
            self.write_turn_history_fn = write_turn_history_fn
            # Unused but accessed by _build_context helpers in commands.py;
            # leave as None so they no-op rather than crash.
            self.manifest = None
            self.data_dir = None
            self.complete_fn = None
            self.state_snapshot = None
            self.mcp_tool_pool = None
            self.current_mode_id = "default"

    return _Rt()


def test_persist_compacted_history_uses_writer_seam_when_wired(tmp_path: Path) -> None:
    """When write_turn_history_fn is wired, the writer is called and no direct file write happens."""
    captured: list[list[dict[str, Any]]] = []

    def _writer(new_history: list[dict[str, Any]]) -> None:
        captured.append(new_history)

    runtime = _runtime_stub(
        thread_path=tmp_path / "thread.jsonl",
        write_turn_history_fn=_writer,
    )
    new_history = [{"summary": "x"}]
    ok = _persist_compacted_history(runtime, new_history)
    assert ok is True
    assert captured == [new_history]
    # The writer was used; the thread file was NOT written directly.
    assert not (tmp_path / "thread.jsonl").exists()


def test_persist_compacted_history_falls_back_to_direct_write(tmp_path: Path) -> None:
    """When write_turn_history_fn is not wired, fall back to a direct JSONL overwrite."""
    runtime = _runtime_stub(thread_path=tmp_path / "thread.jsonl", write_turn_history_fn=None)
    new_history = [{"summary": "x"}, {"q": "a"}]
    ok = _persist_compacted_history(runtime, new_history)
    assert ok is True
    written = (tmp_path / "thread.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in written] == new_history


def test_persist_compacted_history_no_thread_returns_false(tmp_path: Path) -> None:
    """When the host has no thread path and no writer, persistence returns False."""
    runtime = _runtime_stub(thread_path=None, write_turn_history_fn=None)
    ok = _persist_compacted_history(runtime, [{"x": 1}])
    assert ok is False


async def test_handle_summarize_history_writes_back_compacted_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: /summarize-history persists the compacted history via the writer seam."""
    thread_path = tmp_path / "thread.jsonl"
    thread_path.write_text(
        "\n".join(
            json.dumps({"query": f"q{i}", "response_text": f"r{i}"}) for i in range(8)
        )
        + "\n",
        encoding="utf-8",
    )

    captured: list[list[dict[str, Any]]] = []
    runtime = _runtime_stub(
        thread_path=thread_path,
        write_turn_history_fn=lambda h: captured.append(h),
    )

    # Stub the kernel operation registry to return a compacted history.
    from sdk.packs.checkpoint import commands as cp_cmds

    new_history = [{"summary": "compacted"}, {"q": "recent"}]

    async def _fake_exec(op_id: str, ctx: Any) -> Any:
        class _Result:
            output_text = "Condensed 7 turn(s)."
            output_json = {"turn_history": new_history}
        return _Result()

    monkeypatch.setattr(cp_cmds, "execute_operation", _fake_exec)

    result = await handle_summarize_history(runtime, "/summarize-history", "/summarize-history")

    assert "persisted" in result
    assert captured == [new_history]


async def test_handle_summarize_history_reports_when_no_write_back_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When no writer and no thread path, the command reports the result was not persisted."""
    runtime = _runtime_stub(thread_path=None, write_turn_history_fn=None)
    from sdk.packs.checkpoint import commands as cp_cmds

    async def _fake_exec(op_id: str, ctx: Any) -> Any:
        class _Result:
            output_text = "Condensed 7 turn(s)."
            output_json = {"turn_history": [{"summary": "x"}]}
        return _Result()

    monkeypatch.setattr(cp_cmds, "execute_operation", _fake_exec)

    result = await handle_summarize_history(runtime, "/summarize-history", "/summarize-history")
    assert "not persisted" in result
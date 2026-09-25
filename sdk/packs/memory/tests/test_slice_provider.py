"""Tests for the memory-consolidation slice provider.

Mocks the ``turn_hooks`` view (``execute_tool``, ``execute_operation``) to
verify the decision matrix, sidecar persistence, budget short-circuit, and
operation invocation — mirroring the slice provider test pattern. No live model
or SQLite store is involved.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from kernel.slice import SliceMode, SlicePolicy
from sdk.packs.memory.slice_provider import (
    _LAST_CONSOLIDATION_FILE_NAME,
    MemoryConsolidationSlice,
    MemoryConsolidationSliceConfig,
    get_slice_provider_factory,
)


def _config(
    *,
    pressure_threshold: float = 0.6,
    hard_threshold: float = 0.85,
    min_interval_turns: int = 0,
    batch_size: int = 10,
    min_episodic_count: int = 5,
    max_records: int = 200,
    memory_data_dir: Path | None = None,
    user_slug: str = "adam",
) -> MemoryConsolidationSliceConfig:
    return MemoryConsolidationSliceConfig(
        enabled=True,
        pressure_threshold=pressure_threshold,
        hard_threshold=hard_threshold,
        min_interval_turns=min_interval_turns,
        batch_size=batch_size,
        min_episodic_count=min_episodic_count,
        max_records=max_records,
        preferred_model_id="default",
        consolidate_operation="consolidate_semantic",
        memory_data_dir=memory_data_dir,
        user_slug=user_slug,
    )


def _policy(budget_ms: int = 500) -> SlicePolicy:
    return SlicePolicy(
        mode=SliceMode.POST_TURN,
        budget_ms=budget_ms,
        base_budget_ms=10,
        max_deferred_ms=200,
        hard_cap_ms=500,
    )


class _FakeHooks:
    """Minimal hooks view with execute_tool + execute_operation seams."""

    def __init__(
        self,
        *,
        episodic_records: list[dict[str, Any]] | None = None,
        operation_extra: dict[str, Any] | None = None,
    ) -> None:
        self._records = episodic_records or []
        self._operation_extra = operation_extra or {
            "consolidated": True,
            "retired_ids": [1, 2, 3],
        }
        self.tool_calls: list[tuple[str, dict[str, Any]]] = []
        self.op_calls: list[tuple[str, dict[str, Any]]] = []

    async def execute_tool(self, tool_id: str, params: dict[str, Any]) -> Any:
        self.tool_calls.append((tool_id, dict(params)))
        if tool_id == "memory.count":
            class _RCount:
                output_text = str(len(self._records))
            return _RCount()
        if tool_id == "memory.list":
            class _RList:
                output_text = json.dumps(self._records)
            return _RList()
        return type("_R", (), {"output_text": "", "error": "unknown"})()

    async def execute_operation(self, op_id: str, params: dict[str, Any]) -> Any:
        self.op_calls.append((op_id, dict(params)))
        return type(
            "_OpResult",
            (),
            {"output_text": "consolidated", "extra": self._operation_extra},
        )()


class _ErrorHooks(_FakeHooks):
    """Hooks view whose memory.count and memory.list return ToolResult-shaped errors.

    Mirrors what ``host.turn_hooks_builder._execute_tool`` now produces when
    the MCP pool returns an error: an object with a non-empty ``error`` and an
    empty ``output_text``. The slice must treat this as "no count" rather
    than attempting to parse an error string as JSON.
    """

    async def execute_tool(self, tool_id: str, params: dict[str, Any]) -> Any:
        self.tool_calls.append((tool_id, dict(params)))
        if tool_id in ("memory.count", "memory.list"):
            return type("_R", (), {"output_text": "", "error": "server down"})()
        return await super().execute_tool(tool_id, params)


@pytest.mark.asyncio
async def test_below_pressure_threshold_no_ops(tmp_path: Path) -> None:
    hooks = _FakeHooks(episodic_records=[{"id": 1, "content": "a"}])  # 1/200 = 0.005
    slc = MemoryConsolidationSlice(config=_config(memory_data_dir=tmp_path), turn_hooks=hooks)
    outcome = await slc.slice(None, _policy())
    assert outcome.work_items_completed == 0
    assert outcome.notes[0].startswith("below_pressure_threshold")
    # No operation called.
    assert hooks.op_calls == []


@pytest.mark.asyncio
async def test_below_min_episodic_count_defers(tmp_path: Path) -> None:
    # 4 records / 10 max = 0.4 pressure, above 0.3 threshold but below min_episodic_count=5.
    hooks = _FakeHooks(
        episodic_records=[{"id": i, "content": f"e{i}"} for i in range(4)]
    )
    slc = MemoryConsolidationSlice(
        config=_config(
            pressure_threshold=0.3,
            hard_threshold=0.9,
            min_episodic_count=5,
            max_records=10,
            memory_data_dir=tmp_path,
        ),
        turn_hooks=hooks,
    )
    outcome = await slc.slice(None, _policy())
    assert outcome.work_items_completed == 0
    assert outcome.notes[0].startswith("below_min_episodic_count")


@pytest.mark.asyncio
async def test_hard_threshold_override_fires(tmp_path: Path) -> None:
    # 9/10 = 0.9 >= hard_threshold 0.85 → fires regardless of min_interval.
    records = [{"id": i, "content": f"e{i}", "created_at": "2026-07-01T00:00:00Z"} for i in range(9)]
    hooks = _FakeHooks(episodic_records=records)
    slc = MemoryConsolidationSlice(
        config=_config(
            pressure_threshold=0.3,
            hard_threshold=0.85,
            min_interval_turns=100,  # would normally defer
            min_episodic_count=5,
            max_records=10,
            memory_data_dir=tmp_path,
        ),
        turn_hooks=hooks,
    )
    outcome = await slc.slice(None, _policy())
    assert outcome.work_items_completed == 1
    assert outcome.last_item.startswith("consolidated:9->1")
    assert hooks.op_calls[0][0] == "consolidate_semantic"
    # Sidecar persisted.
    sidecar = tmp_path / _LAST_CONSOLIDATION_FILE_NAME
    assert sidecar.exists()
    state = json.loads(sidecar.read_text())
    assert state["last_fired_turn"] >= 1


@pytest.mark.asyncio
async def test_low_slice_budget_defers_between_thresholds(tmp_path: Path) -> None:
    # 6/10 = 0.6, between 0.3 and 0.85, but budget < 150ms → defer.
    hooks = _FakeHooks(
        episodic_records=[{"id": i, "content": f"e{i}"} for i in range(6)]
    )
    slc = MemoryConsolidationSlice(
        config=_config(
            pressure_threshold=0.3,
            hard_threshold=0.85,
            min_episodic_count=5,
            max_records=10,
            memory_data_dir=tmp_path,
        ),
        turn_hooks=hooks,
    )
    outcome = await slc.slice(None, _policy(budget_ms=50))
    assert outcome.work_items_completed == 0
    assert outcome.notes[0].startswith("deferred_low_slice_budget")


@pytest.mark.asyncio
async def test_min_interval_defers_between_thresholds(tmp_path: Path) -> None:
    # 6/10 = 0.6, above 0.3 threshold, budget ok, but min_interval=100 and
    # we just fired → defer.
    records = [{"id": i, "content": f"e{i}"} for i in range(6)]
    hooks = _FakeHooks(episodic_records=records)
    slc = MemoryConsolidationSlice(
        config=_config(
            pressure_threshold=0.3,
            hard_threshold=0.85,
            min_interval_turns=100,
            min_episodic_count=5,
            max_records=10,
            memory_data_dir=tmp_path,
        ),
        turn_hooks=hooks,
    )
    # Prime the sidecar so last_fired_turn is recent.
    slc._state.last_fired_turn = 1
    slc._state.turn_count = 1
    outcome = await slc.slice(None, _policy())
    assert outcome.work_items_completed == 0
    assert outcome.notes[0].startswith("deferred_min_interval")


@pytest.mark.asyncio
async def test_no_execute_tool_seam_no_ops(tmp_path: Path) -> None:
    class _NoToolHooks:
        execute_operation = None  # no execute_tool attribute

    slc = MemoryConsolidationSlice(
        config=_config(memory_data_dir=tmp_path),
        turn_hooks=_NoToolHooks(),
    )
    outcome = await slc.slice(None, _policy())
    assert outcome.work_items_completed == 0


@pytest.mark.asyncio
async def test_tool_error_result_is_no_count(tmp_path: Path) -> None:
    """A ToolResult-shaped error (empty output_text, non-empty error) is
    treated as "no episodic count" instead of crashing json.loads.

    Regression for the host-adapter bug where ``str(MCPToolResult(...))``
    (the dataclass repr starting with "M") was stored as ``output_text``,
    causing ``json.loads`` to raise "Expecting value: line 1 column 1
    (char 0)" on every slice.
    """
    hooks = _ErrorHooks()
    slc = MemoryConsolidationSlice(
        config=_config(memory_data_dir=tmp_path),
        turn_hooks=hooks,
    )
    outcome = await slc.slice(None, _policy())
    assert outcome.work_items_completed == 0
    assert outcome.notes[0] == "no_episodic_count_seam"
    # The list call was attempted; the error short-circuited before firing.
    assert hooks.op_calls == []
    assert outcome.notes[0] == "no_episodic_count_seam"


@pytest.mark.asyncio
async def test_fallback_to_list_when_count_unavailable(tmp_path: Path) -> None:
    """When memory.count is unsupported/fails, the slice falls back to memory.list."""
    class _FallbackHooks(_FakeHooks):
        async def execute_tool(self, tool_id: str, params: dict[str, Any]) -> Any:
            self.tool_calls.append((tool_id, dict(params)))
            if tool_id == "memory.count":
                return type("_R", (), {"output_text": "", "error": "unknown tool memory.count"})()
            if tool_id == "memory.list":
                return type("_R", (), {"output_text": json.dumps(self._records), "error": None})()
            return type("_R", (), {"output_text": "", "error": "unknown"})()

    records = [{"id": i, "content": f"e{i}", "created_at": "2026-07-01T00:00:00Z"} for i in range(9)]
    hooks = _FallbackHooks(episodic_records=records)
    slc = MemoryConsolidationSlice(
        config=_config(
            pressure_threshold=0.3,
            hard_threshold=0.85,
            min_interval_turns=0,
            min_episodic_count=5,
            max_records=10,
            memory_data_dir=tmp_path,
        ),
        turn_hooks=hooks,
    )
    outcome = await slc.slice(None, _policy())
    assert outcome.work_items_completed == 1
    assert "memory.count" in [c[0] for c in hooks.tool_calls]
    assert "memory.list" in [c[0] for c in hooks.tool_calls]


@pytest.mark.asyncio
async def test_mines_new_sealed_segments(tmp_path: Path) -> None:
    """A newly sealed thread segment is mined into semantic memories exactly once."""
    # Build a real sealed thread segment.
    from host.thread import append_thread_entry

    threads_dir = tmp_path / "threads"
    threads_dir.mkdir(parents=True)
    thread_path = threads_dir / "adam"
    for i in range(5):
        append_thread_entry(
            thread_path=thread_path,
            user_slug="adam",
            mode_id="default",
            query=f"q{i}",
            response_text=f"response {i}",
            segment_turns=3,  # seal at 3 → one sealed segment + 2 active
        )

    hooks = _FakeHooks(episodic_records=[])
    slc = MemoryConsolidationSlice(
        config=_config(
            memory_data_dir=tmp_path,
            user_slug="adam",
        ),
        turn_hooks=hooks,
    )
    slc._mine_enabled = True
    outcome = await slc.slice(None, _policy())

    # One sealed segment was mined.
    mine_calls = [c for c in hooks.op_calls if c[0] == "mine_thread_segment"]
    assert len(mine_calls) == 1
    assert outcome.work_items_completed >= 1
    assert outcome.last_item.startswith("mined:")
    # The mine.marker records the mined segment id.
    marker = tmp_path / "mine.marker"
    assert marker.exists()
    assert marker.read_text(encoding="utf-8").strip() == "0001"

    # A second slice run must NOT re-mine the same segment (idempotent).
    outcome2 = await slc.slice(None, _policy())
    mine_calls_2 = [c for c in hooks.op_calls if c[0] == "mine_thread_segment"]
    assert len(mine_calls_2) == 1
    assert not outcome2.last_item.startswith("mined:")


@pytest.mark.asyncio
async def test_mining_disabled_when_mine_on_segment_seal_false(tmp_path: Path) -> None:
    """mine_on_segment_seal=False suppresses the mining pass entirely."""
    threads_dir = tmp_path / "threads"
    threads_dir.mkdir(parents=True)
    thread_path = threads_dir / "adam"
    from host.thread import append_thread_entry

    for i in range(5):
        append_thread_entry(
            thread_path=thread_path,
            user_slug="adam",
            mode_id="default",
            query=f"q{i}",
            response_text=f"response {i}",
            segment_turns=3,
        )

    hooks = _FakeHooks(episodic_records=[])
    slc = MemoryConsolidationSlice(config=_config(memory_data_dir=tmp_path, user_slug="adam"), turn_hooks=hooks)
    slc._mine_enabled = False
    outcome = await slc.slice(None, _policy())
    assert not [c for c in hooks.op_calls if c[0] == "mine_thread_segment"]
    assert not outcome.last_item.startswith("mined:")


def test_factory_returns_none_when_disabled() -> None:
    factory = get_slice_provider_factory()

    class _Plan:
        class manifest:
            class context:
                class memory:
                    class consolidation:
                        enabled = False

    assert factory(None, _Plan()) is None


def test_factory_returns_slice_when_enabled(tmp_path: Path) -> None:
    factory = get_slice_provider_factory()

    class _Tuning:
        enabled = True
        pressure_threshold = 0.5
        hard_threshold = 0.8
        min_interval_turns = 4
        batch_size = 8
        min_episodic_count = 3
        max_records = 100
        preferred_model_id = "default"
        consolidate_operation = "consolidate_semantic"

    class _Plan:
        class manifest:
            class context:
                class memory:
                    consolidation = _Tuning()

    slc = factory(_FakeHooks(), _Plan())
    assert slc is not None
    assert isinstance(slc, MemoryConsolidationSlice)
    assert slc._config.batch_size == 8
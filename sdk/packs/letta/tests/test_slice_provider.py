"""Tests for the Letta sync/compaction slice provider."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from sdk.packs.letta.slice_provider import LettaSyncSlice, get_slice_provider_factory


class _FakeHooks:
    def __init__(
        self,
        *,
        utilization: float | None,
        budget_ms: int,
        executed: list[str] | None = None,
    ) -> None:
        self.last_context_snapshot = (
            SimpleNamespace(
                model_context_window_tokens=1000,
                used_context_tokens=0,
                conversation_tokens=int(utilization * 1000) if utilization is not None else 0,
            )
            if utilization is not None
            else None
        )
        self._budget_ms = budget_ms
        self._executed = executed if executed is not None else []

    async def execute_operation(self, op_id: str, params: object) -> object:
        self._executed.append(op_id)
        return SimpleNamespace(output_text="ok")


def _policy(budget_ms: int) -> object:
    return SimpleNamespace(budget_ms=budget_ms)


@pytest.mark.asyncio
async def test_slice_disabled_by_default(tmp_path: Path) -> None:
    os.environ.pop("LETTA_SYNC_ENABLED", None)
    hooks = _FakeHooks(utilization=0.9, budget_ms=1000)
    slc = LettaSyncSlice(turn_hooks=hooks, thread_dir=tmp_path)
    out = await slc.slice(None, _policy(1000))
    assert out.work_items_completed == 0
    assert "sync_disabled" in out.notes


@pytest.mark.asyncio
async def test_slice_fires_compact_above_hard_threshold(tmp_path: Path) -> None:
    os.environ["LETTA_SYNC_ENABLED"] = "1"
    try:
        executed: list[str] = []
        hooks = _FakeHooks(utilization=0.9, budget_ms=1000, executed=executed)
        slc = LettaSyncSlice(turn_hooks=hooks, thread_dir=tmp_path)
        # Warm past the min-interval guard for compact
        slc._state.last_compact_turn = -100
        slc._state.turn_count = 100
        out = await slc.slice(None, _policy(1000))
        assert out.work_items_completed == 1
        assert out.last_item == "letta_compact_agent"
        assert executed == ["letta_compact_agent"]
    finally:
        os.environ.pop("LETTA_SYNC_ENABLED", None)


@pytest.mark.asyncio
async def test_slice_fires_sync_between_thresholds(tmp_path: Path) -> None:
    os.environ["LETTA_SYNC_ENABLED"] = "1"
    try:
        executed: list[str] = []
        hooks = _FakeHooks(utilization=0.7, budget_ms=1000, executed=executed)
        slc = LettaSyncSlice(turn_hooks=hooks, thread_dir=tmp_path)
        slc._state.last_sync_turn = -100
        slc._state.turn_count = 100
        out = await slc.slice(None, _policy(1000))
        assert out.work_items_completed == 1
        assert out.last_item == "letta_sync_memory"
    finally:
        os.environ.pop("LETTA_SYNC_ENABLED", None)


@pytest.mark.asyncio
async def test_slice_noops_below_pressure(tmp_path: Path) -> None:
    os.environ["LETTA_SYNC_ENABLED"] = "1"
    try:
        hooks = _FakeHooks(utilization=0.3, budget_ms=1000)
        slc = LettaSyncSlice(turn_hooks=hooks, thread_dir=tmp_path)
        out = await slc.slice(None, _policy(1000))
        assert out.work_items_completed == 0
        assert "below_pressure_threshold" in out.notes[0]
    finally:
        os.environ.pop("LETTA_SYNC_ENABLED", None)


@pytest.mark.asyncio
async def test_slice_defers_on_low_budget(tmp_path: Path) -> None:
    os.environ["LETTA_SYNC_ENABLED"] = "1"
    try:
        hooks = _FakeHooks(utilization=0.7, budget_ms=10)
        slc = LettaSyncSlice(turn_hooks=hooks, thread_dir=tmp_path)
        slc._state.last_sync_turn = -100
        slc._state.turn_count = 100
        out = await slc.slice(None, _policy(10))
        assert out.work_items_completed == 0
        assert "deferred_low_slice_budget" in out.notes[0]
    finally:
        os.environ.pop("LETTA_SYNC_ENABLED", None)


@pytest.mark.asyncio
async def test_slice_noops_without_snapshot(tmp_path: Path) -> None:
    os.environ["LETTA_SYNC_ENABLED"] = "1"
    try:
        hooks = _FakeHooks(utilization=None, budget_ms=1000)
        slc = LettaSyncSlice(turn_hooks=hooks, thread_dir=tmp_path)
        out = await slc.slice(None, _policy(1000))
        assert out.work_items_completed == 0
        assert "no_context_snapshot" in out.notes
    finally:
        os.environ.pop("LETTA_SYNC_ENABLED", None)


def test_factory_returns_provider() -> None:
    from types import SimpleNamespace
    provider = get_slice_provider_factory(SimpleNamespace(), SimpleNamespace())
    assert isinstance(provider, LettaSyncSlice)


@pytest.mark.asyncio
async def test_slice_swallows_operation_error(tmp_path: Path) -> None:
    os.environ["LETTA_SYNC_ENABLED"] = "1"
    try:
        class _ErrHooks(_FakeHooks):
            async def execute_operation(self, op_id: str, params: object) -> object:
                raise RuntimeError("boom")
        hooks = _ErrHooks(utilization=0.9, budget_ms=1000)
        slc = LettaSyncSlice(turn_hooks=hooks, thread_dir=tmp_path)
        slc._state.last_compact_turn = -100
        slc._state.turn_count = 100
        out = await slc.slice(None, _policy(1000))
        assert out.work_items_completed == 0
        assert "operation_error" in out.notes[0]
    finally:
        os.environ.pop("LETTA_SYNC_ENABLED", None)
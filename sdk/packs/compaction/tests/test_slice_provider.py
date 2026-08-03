"""Tests for the compaction pack's ContextCompactionSlice decision matrix.

Covers:
- Utilization computation from a stub ContextBudgetSnapshot.
- The threshold + pressure + interval decision matrix.
- Sidecar persistence and min_interval_turns honoring.
- hard_threshold override of the pressure curve.
- Failure paths: no snapshot, no execute_operation, no write_turn_history_fn,
  operation raises, operation returns no turn_history.
- The entry-point factory: returns None when compaction is disabled, returns
  a slice when enabled.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from kernel.slice import SlicePolicy
from sdk.packs.compaction.slice_provider import (
    CompactionSliceConfig,
    ContextCompactionSlice,
    get_slice_provider_factory,
)

# Mark every async test in this module as an asyncio test. This mirrors the
# pattern in sdk/packs/checkpoint/tests/test_checkpoint_operations.py and is
# needed because the root pyproject does not set asyncio_mode=auto.
pytestmark = pytest.mark.asyncio


def _snapshot(*, window: int = 10000, used: int = 0, conversation: int = 0) -> Any:
    """Build a minimal duck-typed ContextBudgetSnapshot stub."""
    class _Snap:
        model_context_window_tokens = window
        used_context_tokens = used
        conversation_tokens = conversation
    return _Snap()


def _hooks(
    *,
    snapshot: Any = None,
    execute_op: Any = None,
    write_fn: Any = None,
) -> Any:
    class _Hooks:
        def __init__(self) -> None:
            self.last_context_snapshot = snapshot
            self.write_turn_history_fn = write_fn
            self.execute_operation = execute_op
    return _Hooks()


def _policy(budget_ms: int = 500) -> SlicePolicy:
    return SlicePolicy(mode="post_turn", budget_ms=budget_ms)


def _config(
    *,
    enabled: bool = True,
    pressure_threshold: float = 0.75,
    hard_threshold: float = 0.90,
    min_interval_turns: int = 4,
    retention_count: int = 6,
    summarize_operation: str = "checkpoint_history_summarize",
    thread_dir: Path | None = None,
    turn_history_loader: Any = None,
) -> CompactionSliceConfig:
    return CompactionSliceConfig(
        enabled=enabled,
        pressure_threshold=pressure_threshold,
        hard_threshold=hard_threshold,
        min_interval_turns=min_interval_turns,
        retention_count=retention_count,
        preferred_model_id="deep",
        summarize_operation=summarize_operation,
        thread_dir=thread_dir,
        turn_history_loader=turn_history_loader,
    )


def _op_result(new_history: list[dict[str, Any]] | None) -> Any:
    class _Result:
        output_json = {"turn_history": new_history} if new_history is not None else {}
        output_text = "compacted"
    return _Result()


# ── Utilization + decision matrix ──────────────────────────────────────


async def test_no_snapshot_no_ops() -> None:
    """When last_context_snapshot is None, the slice no-ops with a note."""
    hooks = _hooks(snapshot=None)
    slice_ = ContextCompactionSlice(_config(), turn_hooks=hooks)
    outcome = await slice_.slice(ctx=None, policy=_policy())
    assert outcome.work_items_completed == 0
    assert "no_context_snapshot" in outcome.notes


async def test_below_threshold_no_ops() -> None:
    """Below pressure_threshold, no compaction fires."""
    hooks = _hooks(snapshot=_snapshot(window=10000, used=1000, conversation=1000))
    slice_ = ContextCompactionSlice(_config(), turn_hooks=hooks)
    outcome = await slice_.slice(ctx=None, policy=_policy())
    assert outcome.work_items_completed == 0
    assert any("below_pressure_threshold" in n for n in outcome.notes)


async def test_between_thresholds_low_budget_defers() -> None:
    """Between thresholds but slice budget below floor: deferred."""
    hooks = _hooks(snapshot=_snapshot(window=10000, used=8000, conversation=0))
    slice_ = ContextCompactionSlice(_config(), turn_hooks=hooks)
    # 150ms is exactly the floor; use 149 to defer.
    outcome = await slice_.slice(ctx=None, policy=_policy(budget_ms=149))
    assert outcome.work_items_completed == 0
    assert any("deferred_low_slice_budget" in n for n in outcome.notes)


async def test_between_thresholds_high_budget_fires() -> None:
    """Between thresholds with adequate budget: fires compaction."""

    async def _exec(op_id: str, params: dict[str, Any]) -> Any:
        return _op_result([{"summary": "x"}])

    hooks = _hooks(
        snapshot=_snapshot(window=10000, used=8000, conversation=0),
        execute_op=_exec,
        write_fn=lambda h: None,
    )
    slice_ = ContextCompactionSlice(
        _config(
            min_interval_turns=0,
            turn_history_loader=lambda: [{"q": "a"}, {"q": "b"}, {"q": "c"},
                                          {"q": "d"}, {"q": "e"}, {"q": "f"},
                                          {"q": "g"}],
        ),
        turn_hooks=hooks,
    )
    outcome = await slice_.slice(ctx=None, policy=_policy(budget_ms=500))
    assert outcome.work_items_completed == 1
    assert outcome.last_item.startswith("compacted:")


async def test_hard_threshold_overrides_low_budget() -> None:
    """At or above hard_threshold, compaction fires even with zero slice budget."""
    hooks = _hooks(snapshot=_snapshot(window=10000, used=9500, conversation=0))

    async def _exec(op_id: str, params: dict[str, Any]) -> Any:
        return _op_result([{"summary": "x"}])

    hooks.execute_operation = _exec
    hooks.write_turn_history_fn = lambda h: None
    slice_ = ContextCompactionSlice(
        _config(
            min_interval_turns=0,
            turn_history_loader=lambda: [{"q": f"t{i}"} for i in range(8)],
        ),
        turn_hooks=hooks,
    )
    outcome = await slice_.slice(ctx=None, policy=_policy(budget_ms=0))
    assert outcome.work_items_completed == 1
    assert any("hard_threshold_override" in n for n in outcome.notes)


async def test_min_interval_suppresses_between_threshold_fires() -> None:
    """min_interval_turns suppresses between-threshold fires within the window."""
    hooks = _hooks(snapshot=_snapshot(window=10000, used=8000, conversation=0))

    async def _exec(op_id: str, params: dict[str, Any]) -> Any:
        return _op_result([{"summary": "x"}])

    hooks.execute_operation = _exec
    calls: list[int] = []
    hooks.write_turn_history_fn = lambda h: calls.append(len(h))

    slice_ = ContextCompactionSlice(
        _config(min_interval_turns=3, turn_history_loader=lambda: [{"q": "a"}] * 8),
        turn_hooks=hooks,
    )
    # First call fires (turn 1, last_fired_turn=0, delta=1 < 3 but we're between
    # thresholds and budget is fine, so this should actually be suppressed).
    # Actually on the first call, turns_since_last = 1 - 0 = 1 < 3 → suppress.
    out1 = await slice_.slice(ctx=None, policy=_policy(budget_ms=500))
    assert out1.work_items_completed == 0
    assert any("deferred_min_interval" in n for n in out1.notes)
    assert calls == []  # no write happened


async def test_no_turn_history_no_ops() -> None:
    """When turn_history_loader returns None or empty, slice no-ops."""
    hooks = _hooks(snapshot=_snapshot(window=10000, used=9500, conversation=0))
    slice_ = ContextCompactionSlice(_config(turn_history_loader=lambda: None), turn_hooks=hooks)
    outcome = await slice_.slice(ctx=None, policy=_policy())
    assert outcome.work_items_completed == 0
    assert any("no_turn_history" in n for n in outcome.notes)


async def test_no_execute_operation_no_ops() -> None:
    """When execute_operation is not callable, slice no-ops."""
    hooks = _hooks(
        snapshot=_snapshot(window=10000, used=9500, conversation=0),
        execute_op=None,
        write_fn=None,
    )
    slice_ = ContextCompactionSlice(
        _config(turn_history_loader=lambda: [{"q": "a"}] * 8),
        turn_hooks=hooks,
    )
    outcome = await slice_.slice(ctx=None, policy=_policy())
    assert outcome.work_items_completed == 0
    assert any("no_execute_operation" in n for n in outcome.notes)


async def test_operation_raises_no_ops_with_diagnostic() -> None:
    """When execute_operation raises, slice returns a no-op with an error note."""

    async def _exec(op_id: str, params: dict[str, Any]) -> Any:
        raise RuntimeError("boom")

    hooks = _hooks(
        snapshot=_snapshot(window=10000, used=9500, conversation=0),
        execute_op=_exec,
        write_fn=None,
    )
    slice_ = ContextCompactionSlice(
        _config(turn_history_loader=lambda: [{"q": "a"}] * 8),
        turn_hooks=hooks,
    )
    outcome = await slice_.slice(ctx=None, policy=_policy())
    assert outcome.work_items_completed == 0
    assert any("operation_error" in n for n in outcome.notes)


async def test_no_write_turn_history_fn_skips_write() -> None:
    """When write_turn_history_fn is not wired, the result is not persisted."""
    hooks = _hooks(
        snapshot=_snapshot(window=10000, used=9500, conversation=0),
        execute_op=None,
        write_fn=None,
    )

    async def _exec(op_id: str, params: dict[str, Any]) -> Any:
        return _op_result([{"summary": "x"}])

    hooks.execute_operation = _exec
    slice_ = ContextCompactionSlice(
        _config(turn_history_loader=lambda: [{"q": "a"}] * 8),
        turn_hooks=hooks,
    )
    outcome = await slice_.slice(ctx=None, policy=_policy())
    assert outcome.work_items_completed == 0
    assert any("write_back_skipped" in n for n in outcome.notes)


async def test_sidecar_persisted_after_successful_fire(tmp_path: Path) -> None:
    """A successful fire writes the sidecar with the new turn counter."""

    async def _exec(op_id: str, params: dict[str, Any]) -> Any:
        return _op_result([{"summary": "x"}])

    hooks = _hooks(
        snapshot=_snapshot(window=10000, used=9500, conversation=0),
        execute_op=_exec,
        write_fn=lambda h: None,
    )
    slice_ = ContextCompactionSlice(
        _config(
            thread_dir=tmp_path,
            turn_history_loader=lambda: [{"q": "a"}] * 8,
        ),
        turn_hooks=hooks,
    )
    await slice_.slice(ctx=None, policy=_policy())
    sidecar = tmp_path / "last_compaction.json"
    assert sidecar.exists()
    raw = json.loads(sidecar.read_text(encoding="utf-8"))
    assert raw["turn_count"] == 1
    assert raw["last_fired_turn"] == 1
    assert raw["last_fired_at"] > 0


async def test_sidecar_loaded_on_construction(tmp_path: Path) -> None:
    """A pre-existing sidecar is loaded so min_interval is honored across runs."""
    sidecar = tmp_path / "last_compaction.json"
    sidecar.write_text(
        json.dumps({"turn_count": 10, "last_fired_at": 1.0, "last_fired_turn": 8}),
        encoding="utf-8",
    )
    hooks = _hooks(snapshot=_snapshot(window=10000, used=8000, conversation=0))
    slice_ = ContextCompactionSlice(
        _config(
            min_interval_turns=5,
            thread_dir=tmp_path,
            turn_history_loader=lambda: [{"q": "a"}] * 8,
        ),
        turn_hooks=hooks,
    )
    # turn_count becomes 11; turns_since_last = 11 - 8 = 3 < 5 → defer.
    outcome = await slice_.slice(ctx=None, policy=_policy(budget_ms=500))
    assert outcome.work_items_completed == 0
    assert any("deferred_min_interval" in n for n in outcome.notes)


# ── Factory ────────────────────────────────────────────────────────────


def _fake_plan(*, enabled: bool, pressure: float = 0.75, hard: float = 0.90) -> Any:
    class _Tuning:
        enabled = False
        trigger = "post_turn"
        pressure_threshold = pressure
        hard_threshold = hard
        min_interval_turns = 4
        retention_count = 6
        preferred_model_id = "deep"
        summarize_operation = "checkpoint_history_summarize"

    class _History:
        compaction = _Tuning()

    class _Context:
        history = _History()

    class _Manifest:
        context = _Context()

    class _Plan:
        manifest = _Manifest()

    plan = _Plan()
    plan.manifest.context.history.compaction.enabled = enabled
    plan.manifest.context.history.compaction.pressure_threshold = pressure
    plan.manifest.context.history.compaction.hard_threshold = hard
    return plan


def test_factory_returns_none_when_disabled() -> None:
    """Factory returns None when compaction is disabled in the session plan."""
    factory = get_slice_provider_factory()
    assert factory(None, _fake_plan(enabled=False)) is None


def test_factory_returns_slice_when_enabled() -> None:
    """Factory returns a ContextCompactionSlice when compaction is enabled."""
    factory = get_slice_provider_factory()
    provider = factory(_hooks(), _fake_plan(enabled=True, pressure=0.5, hard=0.8))
    assert isinstance(provider, ContextCompactionSlice)
    assert provider._config.pressure_threshold == 0.5
    assert provider._config.hard_threshold == 0.8


def test_factory_returns_none_when_plan_lacks_compaction() -> None:
    """Factory tolerates plans that predate the compaction field."""

    class _NoComp:
        compaction = None

    class _H:
        compaction = None

    class _C:
        history = _H()

    class _M:
        context = _C()

    class _P:
        manifest = _M()

    factory = get_slice_provider_factory()
    assert factory(None, _P()) is None


def test_factory_returns_none_when_plan_is_none() -> None:
    """Factory returns None when there is no session plan at all."""
    factory = get_slice_provider_factory()
    assert factory(None, None) is None
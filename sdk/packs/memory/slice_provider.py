"""Episodic-pressure-triggered, slice-driven memory consolidation slice.

The host wires the returned slice provider into ``TurnHooks.post_turn_slice``
(or ``pre_turn_slice`` / ``explicit_slice`` per
``MemoryConsolidationTuning.trigger``) via the
``lisa.extensions.slice_providers`` entry-point group. Each turn, after the
user has seen the answer, the provider:

1. Counts active episodic records for the scoped user via
   ``execute_tool("memory.list", {memory_kind="episodic", limit=1})`` —
   the list call returns recent records; for a precise count the slice
   uses a lightweight ``count_memories``-style signal when available, else
   falls back to the list length.
2. Computes episodic pressure: ``count / max_records``.
3. Applies the decision matrix from ``MemoryConsolidationTuning``:
   - below ``pressure_threshold`` → no-op (most turns)
   - between thresholds → fire only when the slice's pressure-derived
     ``budget_ms`` is at least ``_MIN_SLICE_BUDGET_MS``
   - at or above ``hard_threshold`` → fire regardless of slice budget
4. Honors ``min_interval_turns`` unless the hard threshold overrides.
5. Loads a batch of episodic records via ``execute_tool("memory.list")``,
   fires the configured ``consolidate_operation`` (default:
   ``consolidate_semantic``) via ``turn_hooks.execute_operation``, which
   distills the batch into one semantic record and retires the sources.
6. Updates a pack-local sidecar (``last_consolidation.json`` beside the
   memory DB) holding the turn counter and timestamp of the last fire.

All failures are caught and logged; they never break the turn. Every
branch returns a ``SliceOutcome`` with a diagnostic rationale note.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kernel.slice import SliceOutcome, SlicePolicy
from kernel.slice_provider_discovery import SliceProviderFactory

logger = logging.getLogger(__name__)

__all__ = [
    "MemoryConsolidationSlice",
    "MemoryConsolidationSliceConfig",
    "get_slice_provider_factory",
]

#: Minimum slice budget (ms) required to fire a between-thresholds
#: consolidation. The LLM distillation call needs runway to start; below
#: this the slice defers to a less-pressured turn.
_MIN_SLICE_BUDGET_MS = 150

#: Sidecar filename (beside the memory DB) holding the last-fire counter
#: and timestamp. Mirrors the compaction slice's ``last_compaction.json``.
_LAST_CONSOLIDATION_FILE_NAME = "last_consolidation.json"


@dataclass(slots=True)
class MemoryConsolidationSliceConfig:
    """Runtime-resolved configuration handed to a constructed slice.

    Built by the factory from the session plan's
    ``MemoryConsolidationTuning``. Kept separate from the pydantic model
    so the slice does not re-resolve the plan on every turn.
    """

    enabled: bool
    pressure_threshold: float
    hard_threshold: float
    min_interval_turns: int
    batch_size: int
    min_episodic_count: int
    max_records: int
    preferred_model_id: str
    consolidate_operation: str
    memory_data_dir: Path | None = None
    user_slug: str = "default"
    workspace_id: str = "default"
    memory_scope: str = "personal"
    surface_id: str = ""


@dataclass(slots=True)
class _LastConsolidationState:
    """Mutable in-memory mirror of the sidecar; persisted on each fire."""

    turn_count: int = 0
    last_fired_at: float = 0.0
    last_fired_turn: int = 0


class MemoryConsolidationSlice:
    """Slice provider that fires memory consolidation on episodic pressure.

    Constructed by the factory when ``MemoryConsolidationTuning.enabled``
    is true. The host wires ``slice`` into the trigger slot selected by
    ``MemoryConsolidationTuning.trigger``.
    """

    def __init__(
        self,
        config: MemoryConsolidationSliceConfig,
        *,
        turn_hooks: Any,
    ) -> None:
        self._config = config
        self._hooks = turn_hooks
        self._state = _LastConsolidationState()
        self._load_state()

    # ── Sidecar persistence ──────────────────────────────────────────

    def _sidecar_path(self) -> Path | None:
        data_dir = self._config.memory_data_dir
        if data_dir is None:
            return None
        return Path(data_dir) / _LAST_CONSOLIDATION_FILE_NAME

    def _load_state(self) -> None:
        path = self._sidecar_path()
        if path is None or not path.exists():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            self._state.turn_count = int(raw.get("turn_count", 0))
            self._state.last_fired_at = float(raw.get("last_fired_at", 0.0))
            self._state.last_fired_turn = int(raw.get("last_fired_turn", 0))
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("memory_consolidation: failed to load sidecar %s: %s", path, exc)

    def _persist_state(self) -> None:
        path = self._sidecar_path()
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "turn_count": self._state.turn_count,
                        "last_fired_at": self._state.last_fired_at,
                        "last_fired_turn": self._state.last_fired_turn,
                    }
                ),
                encoding="utf-8",
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("memory_consolidation: failed to persist sidecar %s: %s", path, exc)

    # ── Pressure + decision ──────────────────────────────────────────

    async def _episodic_count(self) -> int | None:
        """Count active episodic records via the memory.list tool.

        Uses ``execute_tool("memory.list", {memory_kind="episodic",
        status="active", limit=batch_size})`` and returns the list length.
        A precise ``count_memories`` tool is not model-facing, so the
        list length is the cheapest available proxy. Returns ``None`` when
        the tool seam is unavailable or the call fails.
        """
        exec_tool = getattr(self._hooks, "execute_tool", None)
        if not callable(exec_tool):
            return None
        try:
            result = await exec_tool(
                "memory.list",
                {
                    "user_slug": self._config.user_slug,
                    "workspace_id": self._config.workspace_id,
                    "memory_scope": self._config.memory_scope,
                    "surface_id": self._config.surface_id,
                    "memory_kind": "episodic",
                    "status": "active",
                    "limit": self._config.batch_size,
                },
            )
            text = getattr(result, "output_text", "") or "[]"
            records = json.loads(text) if text else []
            return len(records) if isinstance(records, list) else None
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("memory_consolidation: episodic count failed: %s", exc)
            return None

    async def _load_episodic_batch(self) -> list[dict[str, Any]]:
        """Load a batch of active episodic records via memory.list."""
        exec_tool = getattr(self._hooks, "execute_tool", None)
        if not callable(exec_tool):
            return []
        try:
            result = await exec_tool(
                "memory.list",
                {
                    "user_slug": self._config.user_slug,
                    "workspace_id": self._config.workspace_id,
                    "memory_scope": self._config.memory_scope,
                    "surface_id": self._config.surface_id,
                    "memory_kind": "episodic",
                    "status": "active",
                    "limit": self._config.batch_size,
                },
            )
            text = getattr(result, "output_text", "") or "[]"
            records = json.loads(text) if text else []
            return records if isinstance(records, list) else []
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("memory_consolidation: episodic batch load failed: %s", exc)
            return []

    def _should_fire(
        self,
        *,
        pressure: float,
        episodic_count: int,
        slice_budget_ms: int,
    ) -> tuple[bool, str]:
        """Decision matrix mirroring the compaction slice."""
        cfg = self._config
        self._state.turn_count += 1
        if pressure >= cfg.hard_threshold:
            return True, f"hard_threshold_override(pressure={pressure:.3f})"
        if pressure < cfg.pressure_threshold:
            return False, f"below_pressure_threshold(pressure={pressure:.3f})"
        if episodic_count < cfg.min_episodic_count:
            return False, f"below_min_episodic_count(count={episodic_count})"
        # Between thresholds: piggyback on the slice pressure curve.
        if slice_budget_ms < _MIN_SLICE_BUDGET_MS:
            return False, f"deferred_low_slice_budget(budget_ms={slice_budget_ms})"
        turns_since_last = self._state.turn_count - self._state.last_fired_turn
        if cfg.min_interval_turns > 0 and turns_since_last < cfg.min_interval_turns:
            return False, (
                f"deferred_min_interval(turns_since_last={turns_since_last},"
                f" min_interval={cfg.min_interval_turns})"
            )
        return True, f"pressure_threshold(pressure={pressure:.3f})"

    def _slice_budget_expired(self, policy: SlicePolicy, started: float) -> bool:
        """True when the elapsed wall-clock exceeds the slice budget."""
        if policy.budget_ms <= 0:
            return False
        return (time.monotonic() - started) * 1000.0 >= float(policy.budget_ms)

    # ── Slice entrypoint ─────────────────────────────────────────────

    async def slice(self, ctx: Any, policy: SlicePolicy) -> SliceOutcome:
        """Fire consolidation if episodic pressure warrants it.

        ``ctx`` is unused (matches the ``StateProvider.slice`` contract);
        all state is read from ``self._hooks`` and the sidecar.
        """
        del ctx  # explicit: the slice is stateless w.r.t. ctx
        started = time.monotonic()

        episodic_count = await self._episodic_count()
        if episodic_count is None:
            return SliceOutcome(
                work_items_completed=0,
                elapsed_ms=0.0,
                last_item="",
                notes=("no_episodic_count_seam",),
            )
        pressure = episodic_count / max(self._config.max_records, 1)

        fire, reason = self._should_fire(
            pressure=pressure,
            episodic_count=episodic_count,
            slice_budget_ms=int(getattr(policy, "budget_ms", 0) or 0),
        )
        if not fire:
            return SliceOutcome(
                work_items_completed=0,
                elapsed_ms=(time.monotonic() - started) * 1000.0,
                last_item="",
                notes=(reason,),
            )

        # Fire: load the episodic batch, invoke the consolidation operation.
        batch = await self._load_episodic_batch()
        if not batch:
            return SliceOutcome(
                work_items_completed=0,
                elapsed_ms=(time.monotonic() - started) * 1000.0,
                last_item="",
                notes=(f"{reason};no_episodic_batch",),
            )
        if self._slice_budget_expired(policy, started):
            return SliceOutcome(
                work_items_completed=0,
                elapsed_ms=(time.monotonic() - started) * 1000.0,
                last_item="",
                notes=(f"{reason};budget_expired_before_operation",),
            )

        exec_op = getattr(self._hooks, "execute_operation", None)
        if not callable(exec_op):
            return SliceOutcome(
                work_items_completed=0,
                elapsed_ms=(time.monotonic() - started) * 1000.0,
                last_item="",
                notes=(f"{reason};no_execute_operation",),
            )

        op_id = self._config.consolidate_operation
        params: dict[str, Any] = {
            "episodic_records": batch,
            "user_slug": self._config.user_slug,
            "workspace_id": self._config.workspace_id,
            "memory_scope": self._config.memory_scope,
            "surface_id": self._config.surface_id,
        }
        try:
            result = await exec_op(op_id, params)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("memory_consolidation: %s failed: %s", op_id, exc)
            return SliceOutcome(
                work_items_completed=0,
                elapsed_ms=(time.monotonic() - started) * 1000.0,
                last_item="",
                notes=(f"{reason};operation_error:{type(exc).__name__}",),
            )

        extra = getattr(result, "extra", {}) or {}
        consolidated = bool(extra.get("consolidated"))
        retired = len(extra.get("retired_ids", []) or [])

        self._state.last_fired_at = time.time()
        self._state.last_fired_turn = self._state.turn_count
        self._persist_state()

        return SliceOutcome(
            work_items_completed=1 if consolidated else 0,
            elapsed_ms=(time.monotonic() - started) * 1000.0,
            last_item=f"consolidated:{len(batch)}->1,retired:{retired}",
            notes=(reason,),
        )


def _resolve_consolidation_tuning(session_plan: Any) -> Any:
    """Pull the ``MemoryConsolidationTuning`` off a resolved ``SessionPlan``."""
    manifest = getattr(session_plan, "manifest", None)
    if manifest is None:
        return None
    context = getattr(manifest, "context", None)
    if context is None:
        return None
    memory = getattr(context, "memory", None)
    if memory is None:
        return None
    return getattr(memory, "consolidation", None)


def get_slice_provider_factory() -> SliceProviderFactory:
    """Entry-point factory for the ``lisa.extensions.slice_providers`` group.

    Returns a callable ``(turn_hooks, session_plan) -> provider | None``.
    The host invokes it during ``TurnHooks`` assembly. When the session
    plan does not enable consolidation, the factory returns ``None`` and
    the host skips wiring — no behavior change for that session.
    """

    def factory(
        turn_hooks: Any, session_plan: Any
    ) -> MemoryConsolidationSlice | None:
        tuning = _resolve_consolidation_tuning(session_plan)
        if tuning is None or not bool(getattr(tuning, "enabled", False)):
            return None
        config = MemoryConsolidationSliceConfig(
            enabled=bool(tuning.enabled),
            pressure_threshold=float(tuning.pressure_threshold),
            hard_threshold=float(tuning.hard_threshold),
            min_interval_turns=int(tuning.min_interval_turns),
            batch_size=int(tuning.batch_size),
            min_episodic_count=int(tuning.min_episodic_count),
            max_records=int(tuning.max_records),
            preferred_model_id=str(tuning.preferred_model_id),
            consolidate_operation=str(tuning.consolidate_operation),
        )
        return MemoryConsolidationSlice(config=config, turn_hooks=turn_hooks)

    return factory
"""Episodic-pressure-triggered, slice-driven memory consolidation + thread mining.

The host wires the returned slice provider into ``TurnHooks.post_turn_slice``
(or ``pre_turn_slice`` / ``explicit_slice`` per
``MemoryConsolidationTuning.trigger``) via the
``lisa.extensions.slice_providers`` entry-point group. Each turn, after the
user has seen the answer, the provider performs two jobs:

**Job 1 — Thread mining (new sealed segments).** The slice checks the
thread store's segment index for sealed segments not yet mined (tracked by
a pack-local ``mine.marker`` sidecar). Each newly sealed segment is
distilled via the ``mine_thread_segment`` operation into durable semantic
records carrying ``thread_turn_lo/hi`` provenance. Mining is idempotent
per segment and bounded (``_MAX_SEGMENTS_PER_SLICE`` per slice).

**Job 2 — Episodic consolidation (pressure relief).** When episodic
pressure (active episodic count / ``max_records``) crosses a threshold:

1. Counts active episodic records via ``memory.count`` (fallback:
   ``memory.list`` bounded by ``max_records``).
2. Applies the decision matrix from ``MemoryConsolidationTuning``:
   - below ``pressure_threshold`` → no-op (most turns)
   - between thresholds → fire only when the slice's pressure-derived
     ``budget_ms`` is at least ``_MIN_SLICE_BUDGET_MS``
   - at or above ``hard_threshold`` → fire regardless of slice budget
3. Honors ``min_interval_turns`` unless the hard threshold overrides.
4. Loads a batch of episodic records, fires ``consolidate_semantic``
   via ``turn_hooks.execute_operation``, which distills the batch into one
   semantic record and retires the sources.
5. Updates a pack-local sidecar (``last_consolidation.json`` beside the
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
#: and timestamp, mirroring the slice-provider sidecar pattern.
_LAST_CONSOLIDATION_FILE_NAME = "last_consolidation.json"

#: Sidecar filename holding the last-mined thread segment id. Mining is
#: idempotent per segment: a sealed segment is mined exactly once.
_LAST_MINED_SEGMENT_FILE_NAME = "mine.marker"

#: Default location of the thread store base path, relative to the memory
#: data dir. The thread store lives at ``<data_dir>/threads/<user_slug>``;
#: its segment index at ``<base>.index.jsonl``.
_THREADS_SUBDIR = "threads"

#: Max sealed segments mined per slice (budget-bounded, mirrors the old
#: digest batching).
_MAX_SEGMENTS_PER_SLICE = 2


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
        self._mine_enabled = True  # default on; overridden by the factory
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
        """Count active episodic records via the memory.count tool with list fallback.

        Uses ``execute_tool("memory.count", {memory_kind="episodic", status="active"})``.
        Falls back to ``memory.list`` (bounded by ``max_records``) when ``memory.count``
        is unavailable. Returns ``None`` when the tool seam is unavailable or fails.
        """
        exec_tool = getattr(self._hooks, "execute_tool", None)
        if not callable(exec_tool):
            return None
        try:
            result = await exec_tool(
                "memory.count",
                {
                    "user_slug": self._config.user_slug,
                    "workspace_id": self._config.workspace_id,
                    "memory_scope": self._config.memory_scope,
                    "surface_id": self._config.surface_id,
                    "memory_kind": "episodic",
                    "status": "active",
                },
            )
            if not getattr(result, "error", None):
                text = (getattr(result, "output_text", "") or "").strip()
                if text.isdigit():
                    return int(text)
        except Exception as exc:  # pylint: disable=broad-except
            logger.debug(
                "memory_consolidation: memory.count unavailable, trying list fallback: %s",
                exc,
            )

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
                    "limit": self._config.max_records,
                },
            )
            if getattr(result, "error", None):
                logger.warning(
                    "memory_consolidation: episodic count tool error: %s",
                    result.error,
                )
                return None
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
            if getattr(result, "error", None):
                logger.warning(
                    "memory_consolidation: episodic batch load tool error: %s",
                    result.error,
                )
                return []
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

    # ── Thread mining (sealed segments) ──────────────────────────────

    def _mine_marker_path(self) -> Path | None:
        """Path of the ``mine.marker`` sidecar, or ``None`` when unlocatable."""
        data_dir = self._config.memory_data_dir
        if data_dir is None:
            return None
        return Path(data_dir) / _LAST_MINED_SEGMENT_FILE_NAME

    def _thread_base(self) -> Path | None:
        """Thread store base path derived from the memory data dir + user slug."""
        data_dir = self._config.memory_data_dir
        if data_dir is None or not self._config.user_slug:
            return None
        return Path(data_dir) / _THREADS_SUBDIR / self._config.user_slug

    def _load_last_mined_seg_id(self) -> str:
        path = self._mine_marker_path()
        if path is None or not path.exists():
            return ""
        try:
            return path.read_text(encoding="utf-8").strip()
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("memory_consolidation: failed to read mine.marker: %s", exc)
            return ""

    def _persist_mine_marker(self, seg_id: str) -> None:
        path = self._mine_marker_path()
        if path is None or not seg_id:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(seg_id, encoding="utf-8")
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("memory_consolidation: failed to persist mine.marker: %s", exc)

    def _sealed_segments_to_mine(self) -> list[tuple[str, int, int]]:
        """List unmined sealed segments as ``(seg_id, turn_lo, turn_hi)``.

        Reads the thread store's sparse index directly (one small file) —
        no decompression. Returns segments after the last-mined seg_id,
        bounded by ``_MAX_SEGMENTS_PER_SLICE``.
        """
        base = self._thread_base()
        if base is None:
            return []
        index_path = Path(f"{base}.index.jsonl")
        if not index_path.exists():
            return []
        last_mined = self._load_last_mined_seg_id()
        try:
            raw_lines = index_path.read_text(encoding="utf-8").splitlines()
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("memory_consolidation: failed to read segment index: %s", exc)
            return []
        records: list[tuple[str, int, int]] = []
        seen_last = not last_mined  # no marker → mine from the start
        for raw_line in raw_lines:
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                decoded = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            seg_id = str(decoded.get("seg_id", ""))
            if not seg_id:
                continue
            if not seen_last:
                if seg_id == last_mined:
                    seen_last = True
                continue
            records.append(
                (
                    seg_id,
                    int(decoded.get("turn_lo", 0)),
                    int(decoded.get("turn_hi", 0)),
                )
            )
        return records[:_MAX_SEGMENTS_PER_SLICE]

    def _load_segment_turns(
        self, base: Path, seg_id: str, turn_lo: int, turn_hi: int
    ) -> list[dict[str, Any]]:
        """Load one sealed segment's turns by its global turn range.

        Uses the thread store's ``load_range`` with the segment's exact
        turn span from the index — only that segment is decompressed.
        """
        from runtime.thread import load_thread_range

        try:
            return load_thread_range(
                thread_path=base,
                start_turn=turn_lo,
                end_turn=turn_hi,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("memory_consolidation: failed to load segment %s: %s", seg_id, exc)
            return []

    async def _mine_new_segments(self, policy: SlicePolicy, started: float) -> tuple[int, str, str]:
        """Mine newly sealed thread segments into semantic memories.

        Returns ``(work_items, last_item, note)``.
        """
        exec_op = getattr(self._hooks, "execute_operation", None)
        if not callable(exec_op):
            return 0, "", ""
        segments = self._sealed_segments_to_mine()
        if not segments:
            return 0, "", ""
        base = self._thread_base()
        if base is None:
            return 0, "", ""

        work_items = 0
        last_item = ""
        for seg_id, turn_lo, turn_hi in segments:
            if self._slice_budget_expired(policy, started):
                break
            turns = self._load_segment_turns(base, seg_id, turn_lo, turn_hi)
            if not turns:
                # Segment unreadable — advance the marker so it is not retried.
                self._persist_mine_marker(seg_id)
                continue
            params: dict[str, Any] = {
                "turns": turns,
                "user_slug": self._config.user_slug,
                "workspace_id": self._config.workspace_id,
                "memory_scope": self._config.memory_scope,
                "surface_id": self._config.surface_id,
            }
            try:
                result = await exec_op("mine_thread_segment", params)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("memory_consolidation: mine_thread_segment failed: %s", exc)
                continue
            extra = getattr(result, "extra", {}) or {}
            stored_ids = extra.get("memory_ids") or []
            self._persist_mine_marker(seg_id)
            work_items += 1
            last_item = f"mined:{seg_id}({turn_lo}-{turn_hi})->{len(stored_ids)}"
        return (
            work_items,
            last_item,
            (f"mined {work_items} sealed segment(s)" if work_items else ""),
        )

    def _outcome(
        self,
        *,
        work_items: int,
        started: float,
        last_items: list[str],
        notes: list[str],
        extra_notes: tuple[str, ...] = (),
    ) -> SliceOutcome:
        """Build one SliceOutcome with elapsed time, merged notes, and last item.

        Args:
            work_items: Total work items completed so far this slice.
            started: The slice start time (``time.monotonic()`` baseline).
            last_items: Accumulated per-job last-item strings (may be empty).
            notes: Accumulated notes (may be empty).
            extra_notes: Trailing notes to append (e.g. ``"<reason>;no_batch"``).
        """
        return SliceOutcome(
            work_items_completed=work_items,
            elapsed_ms=(time.monotonic() - started) * 1000.0,
            last_item=last_items[-1] if last_items else "",
            notes=(*notes, *extra_notes),
        )

    async def _run_mining_pass(
        self,
        policy: SlicePolicy,
        started: float,
    ) -> tuple[int, list[str], list[str]]:
        """Run the mining job (job 1) and fold its results into the accumulators.

        Mines newly sealed thread segments when enabled. Returns the
        ``(total_work, last_items, notes)`` accumulators with the mining
        results folded in. A failed mining pass is logged and swallowed —
        the consolidation job still runs.
        """
        total_work = 0
        last_items: list[str] = []
        notes: list[str] = []
        if self._mine_enabled:
            try:
                mined, mined_last, mine_note = await self._mine_new_segments(policy, started)
                if mined > 0:
                    total_work += mined
                    if mined_last:
                        last_items.append(mined_last)
                    if mine_note:
                        notes.append(mine_note)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("memory_consolidation: mining pass failed: %s", exc)
        return total_work, last_items, notes

    def _no_episodic_count_outcome(
        self,
        total_work: int,
        started: float,
        last_items: list[str],
        notes: list[str],
    ) -> SliceOutcome:
        """Build the outcome for the missing episodic-count seam.

        The episodic-count seam returning ``None`` (backends without the
        count operation) ends the slice after the mining pass.
        """
        if total_work > 0:
            return self._outcome(
                work_items=total_work,
                started=started,
                last_items=last_items,
                notes=notes,
                extra_notes=() if notes else ("mined_only_no_episodic_count",),
            )
        return SliceOutcome(
            work_items_completed=0,
            elapsed_ms=(time.monotonic() - started) * 1000.0,
            last_item="",
            notes=("no_episodic_count_seam",),
        )

    def _no_fire_outcome(
        self,
        total_work: int,
        started: float,
        last_items: list[str],
        notes: list[str],
        reason: str,
    ) -> SliceOutcome:
        """Build the outcome for a consolidation pass that did not fire."""
        if total_work > 0:
            return self._outcome(
                work_items=total_work,
                started=started,
                last_items=last_items,
                notes=notes,
                extra_notes=(reason,),
            )
        return SliceOutcome(
            work_items_completed=0,
            elapsed_ms=(time.monotonic() - started) * 1000.0,
            last_item="",
            notes=(reason,),
        )

    async def _run_consolidation_pass(
        self,
        policy: SlicePolicy,
        started: float,
        total_work: int,
        last_items: list[str],
        *,
        notes: list[str],
        reason: str,
    ) -> SliceOutcome:
        """Run the consolidation job (job 2) once the fire decision is made.

        Loads the episodic batch, invokes the consolidation operation, and
        persists the fired state. Short-circuits with a note-suffixed
        outcome when the batch is empty, the slice budget expired, or the
        execute-operation seam is missing.
        """
        batch = await self._load_episodic_batch()
        if not batch:
            return self._outcome(
                work_items=total_work,
                started=started,
                last_items=last_items,
                notes=notes,
                extra_notes=(f"{reason};no_episodic_batch",),
            )
        if self._slice_budget_expired(policy, started):
            return self._outcome(
                work_items=total_work,
                started=started,
                last_items=last_items,
                notes=notes,
                extra_notes=(f"{reason};budget_expired_before_operation",),
            )

        exec_op = getattr(self._hooks, "execute_operation", None)
        if not callable(exec_op):
            return self._outcome(
                work_items=total_work,
                started=started,
                last_items=last_items,
                notes=notes,
                extra_notes=(f"{reason};no_execute_operation",),
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
            return self._outcome(
                work_items=total_work,
                started=started,
                last_items=last_items,
                notes=notes,
                extra_notes=(f"{reason};operation_error:{type(exc).__name__}",),
            )

        extra = getattr(result, "extra", {}) or {}
        consolidated = bool(extra.get("consolidated"))
        retired = len(extra.get("retired_ids", []) or [])

        self._state.last_fired_at = time.time()
        self._state.last_fired_turn = self._state.turn_count
        self._persist_state()

        return SliceOutcome(
            work_items_completed=total_work + (1 if consolidated else 0),
            elapsed_ms=(time.monotonic() - started) * 1000.0,
            last_item=(
                f"consolidated:{len(batch)}->1,retired:{retired}"
                if not last_items
                else last_items[-1]
            ),
            notes=(*notes, reason),
        )

    # ── Slice entrypoint ─────────────────────────────────────────────

    async def slice(self, ctx: Any, policy: SlicePolicy) -> SliceOutcome:
        """Mine new sealed segments, then fire consolidation if pressure warrants.

        ``ctx`` is unused (matches the ``StateProvider.slice`` contract);
        all state is read from ``self._hooks`` and the sidecars.
        """
        del ctx  # explicit: the slice is stateless w.r.t. ctx
        started = time.monotonic()

        # ── Job 1: mine newly sealed thread segments ────────────────
        total_work, last_items, notes = await self._run_mining_pass(policy, started)

        # ── Job 2: episodic-pressure consolidation ──────────────────
        episodic_count = await self._episodic_count()
        if episodic_count is None:
            return self._no_episodic_count_outcome(total_work, started, last_items, notes)
        pressure = episodic_count / max(self._config.max_records, 1)

        fire, reason = self._should_fire(
            pressure=pressure,
            episodic_count=episodic_count,
            slice_budget_ms=int(getattr(policy, "budget_ms", 0) or 0),
        )
        if not fire:
            return self._no_fire_outcome(total_work, started, last_items, notes, reason)

        # Fire: load the episodic batch, invoke the consolidation operation.
        return await self._run_consolidation_pass(
            policy,
            started,
            total_work,
            last_items,
            notes=notes,
            reason=reason,
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

    def factory(turn_hooks: Any, session_plan: Any) -> MemoryConsolidationSlice | None:
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
        slc = MemoryConsolidationSlice(config=config, turn_hooks=turn_hooks)
        slc._mine_enabled = bool(getattr(tuning, "mine_on_segment_seal", True))
        return slc

    return factory

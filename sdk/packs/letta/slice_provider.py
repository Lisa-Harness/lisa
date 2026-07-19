"""Pressure-driven Letta sync/compaction slice provider.

Mirrors the compaction pack's slice pattern: the host wires the returned
provider into ``TurnHooks.post_turn_slice`` via the
``lisa.extensions.slice_providers`` entry-point group. Each turn, after
the host's context-mapper pass, the provider decides whether to fire a
Letta sync or compaction based on:

1. Whether Letta sync is enabled in pack configuration (env-gated by
   ``LETTA_SYNC_ENABLED``; defaults to off so the slice is a no-op for
   hosts that have not opted in).
2. Context-window utilization pressure
   (``conversation_tokens + used_context_tokens`` / window) read from
   ``turn_hooks.last_context_snapshot``.
3. The slice's pressure-derived ``budget_ms`` — the sync/compact call
   needs runway to start; below a minimum the slice defers.
4. A ``min_interval_turns`` guard against firing every turn.

When it fires, it calls ``letta_sync_memory`` (between thresholds) or
``letta_compact_agent`` (at/above the hard threshold) via
``turn_hooks.execute_operation``.

All failures are caught and logged; they never break the turn.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kernel.slice import SliceOutcome, SlicePolicy
from kernel.slice_provider_discovery import SliceProviderFactory

logger = logging.getLogger(__name__)

__all__ = ["LettaSyncSlice", "get_slice_provider_factory"]

#: Minimum slice budget (ms) required to fire a between-thresholds sync.
_MIN_SLICE_BUDGET_MS = 200

#: Sidecar filename (beside the thread JSONL) holding the last-fire state.
_LETTA_SLICE_FILE = "last_letta_sync.json"

#: Pressure thresholds. Utilization is a 0..1 fraction of the model
#: context window. Below ``_PRESSURE_THRESHOLD`` → no-op. Between
#: thresholds → sync. At/above ``_HARD_THRESHOLD`` → compact.
_PRESSURE_THRESHOLD = 0.65
_HARD_THRESHOLD = 0.85
_MIN_INTERVAL_TURNS = 10


@dataclass(slots=True)
class _SliceState:
    """Mutable in-memory mirror of the sidecar."""

    turn_count: int = 0
    last_sync_turn: int = 0
    last_compact_turn: int = 0
    last_fired_at: float = 0.0


class LettaSyncSlice:
    """Slice provider that fires Letta sync/compaction on utilization pressure."""

    def __init__(self, *, turn_hooks: Any, thread_dir: Path | None = None) -> None:
        self._hooks = turn_hooks
        self._thread_dir = thread_dir
        self._state = _SliceState()
        self._load_state()

    # ── Sidecar persistence ──────────────────────────────────────────

    def _sidecar_path(self) -> Path | None:
        if self._thread_dir is None:
            return None
        return Path(self._thread_dir) / _LETTA_SLICE_FILE

    def _load_state(self) -> None:
        path = self._sidecar_path()
        if path is None or not path.exists():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            self._state.turn_count = int(raw.get("turn_count", 0))
            self._state.last_sync_turn = int(raw.get("last_sync_turn", 0))
            self._state.last_compact_turn = int(raw.get("last_compact_turn", 0))
            self._state.last_fired_at = float(raw.get("last_fired_at", 0.0))
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("letta slice: failed to load sidecar %s: %s", path, exc)

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
                        "last_sync_turn": self._state.last_sync_turn,
                        "last_compact_turn": self._state.last_compact_turn,
                        "last_fired_at": self._state.last_fired_at,
                    }
                ),
                encoding="utf-8",
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("letta slice: failed to persist sidecar %s: %s", path, exc)

    # ── Utilization + decision ──────────────────────────────────────

    def _utilization(self) -> float | None:
        snapshot = getattr(self._hooks, "last_context_snapshot", None)
        if snapshot is None:
            return None
        window = int(getattr(snapshot, "model_context_window_tokens", 0) or 0)
        if window <= 0:
            return None
        used = int(getattr(snapshot, "used_context_tokens", 0) or 0)
        conversation = int(getattr(snapshot, "conversation_tokens", 0) or 0)
        return (conversation + used) / window

    def _should_fire(
        self,
        *,
        utilization: float,
        slice_budget_ms: int,
    ) -> tuple[str, str]:
        """Decide which operation (if any) to fire.

        Returns ``(op_id_or_empty, reason)``. ``op_id`` is empty when the
        slice should no-op; otherwise it is ``letta_sync_memory`` or
        ``letta_compact_agent``.
        """
        self._state.turn_count += 1

        if utilization >= _HARD_THRESHOLD:
            if self._state.turn_count - self._state.last_compact_turn < _MIN_INTERVAL_TURNS:
                return "", (
                    f"deferred_compact_min_interval(turns_since_last="
                    f"{self._state.turn_count - self._state.last_compact_turn})"
                )
            return "letta_compact_agent", f"hard_threshold(utilization={utilization:.3f})"

        if utilization < _PRESSURE_THRESHOLD:
            return "", f"below_pressure_threshold(utilization={utilization:.3f})"

        if slice_budget_ms < _MIN_SLICE_BUDGET_MS:
            return "", (
                f"deferred_low_slice_budget(budget_ms={slice_budget_ms},"
                f"utilization={utilization:.3f})"
            )

        if self._state.turn_count - self._state.last_sync_turn < _MIN_INTERVAL_TURNS:
            return "", (
                f"deferred_sync_min_interval(turns_since_last="
                f"{self._state.turn_count - self._state.last_sync_turn})"
            )
        return "letta_sync_memory", f"pressure_threshold(utilization={utilization:.3f})"

    # ── Slice entrypoint ────────────────────────────────────────────

    async def slice(self, ctx: Any, policy: SlicePolicy) -> SliceOutcome:
        del ctx  # stateless w.r.t. ctx; all state from hooks + sidecar
        started = time.monotonic()

        if os.getenv("LETTA_SYNC_ENABLED", "0") not in ("1", "true", "True"):
            return SliceOutcome(
                work_items_completed=0,
                elapsed_ms=0.0,
                last_item="",
                notes=("sync_disabled",),
            )

        utilization = self._utilization()
        if utilization is None:
            return SliceOutcome(
                work_items_completed=0,
                elapsed_ms=0.0,
                last_item="",
                notes=("no_context_snapshot",),
            )

        op_id, reason = self._should_fire(
            utilization=utilization,
            slice_budget_ms=int(getattr(policy, "budget_ms", 0) or 0),
        )
        if not op_id:
            return SliceOutcome(
                work_items_completed=0,
                elapsed_ms=(time.monotonic() - started) * 1000.0,
                last_item="",
                notes=(reason,),
            )

        exec_op = getattr(self._hooks, "execute_operation", None)
        if not callable(exec_op):
            return SliceOutcome(
                work_items_completed=0,
                elapsed_ms=(time.monotonic() - started) * 1000.0,
                last_item="",
                notes=(f"{reason};no_execute_operation",),
            )

        try:
            await exec_op(op_id, {})
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("letta slice: %s failed: %s", op_id, exc)
            return SliceOutcome(
                work_items_completed=0,
                elapsed_ms=(time.monotonic() - started) * 1000.0,
                last_item="",
                notes=(f"{reason};operation_error:{type(exc).__name__}",),
            )

        if op_id == "letta_compact_agent":
            self._state.last_compact_turn = self._state.turn_count
        else:
            self._state.last_sync_turn = self._state.turn_count
        self._state.last_fired_at = time.time()
        self._persist_state()

        return SliceOutcome(
            work_items_completed=1,
            elapsed_ms=(time.monotonic() - started) * 1000.0,
            last_item=op_id,
            notes=(reason,),
        )


def get_slice_provider_factory(
    turn_hooks: Any,
    session_plan: Any,
) -> LettaSyncSlice | None:
    """Entry-point factory for the ``lisa.extensions.slice_providers`` group.

    The host calls this function directly with ``(turn_hooks, session_plan)``.
    The slice is disabled by default (env-gated), so it always returns a
    provider; the slice itself no-ops when ``LETTA_SYNC_ENABLED`` is unset.
    The host may set ``thread_dir`` on the returned provider for sidecar
    persistence.
    """
    return LettaSyncSlice(turn_hooks=turn_hooks)
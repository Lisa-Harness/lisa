"""Utilization-triggered, pressure-driven context compaction slice.

This module is the heart of the compaction pack. The host wires the
returned slice provider into ``TurnHooks.post_turn_slice`` (or
``pre_turn_slice`` / ``explicit_slice`` per ``CompactionTuning.trigger``)
via the ``lisa.extensions.slice_providers`` entry-point group. Each
turn, after the host's context-mapper pass, the provider:

1. Reads the live ``ContextBudgetSnapshot`` from
   ``turn_hooks.last_context_snapshot`` (stashed by the host's
   ``assemble_user_context`` implementation).
2. Computes context-window utilization:
   ``(conversation_tokens + used_context_tokens) /
   model_context_window_tokens``.
3. Applies the decision matrix from ``CompactionTuning``:
   - below ``pressure_threshold`` → no-op (most turns)
   - between thresholds → fire only when the slice's pressure-derived
     ``budget_ms`` is at least ``_MIN_SLICE_BUDGET_MS`` (piggyback on
     the pressure curve: idle user → big slice → compacts; hammering
     user → tiny slice → deferred)
   - at or above ``hard_threshold`` → fire regardless of slice budget
     (guardrail override; the session is about to overflow)
4. Honors ``min_interval_turns`` unless the hard threshold overrides.
5. Fires the configured ``summarize_operation`` (default:
   ``checkpoint_history_summarize``) via ``turn_hooks.execute_operation``,
   loads the live turn history, passes it through, and writes the
   compacted result back via ``turn_hooks.write_turn_history_fn``.
6. Updates a pack-local sidecar (``last_compaction.json`` beside the
   thread JSONL) holding the turn counter and timestamp of the last
   successful fire, so ``min_interval_turns`` is honored across turns.

All failures are caught and logged; they never break the turn (the
kernel's ``_fire_slice`` already catches slice exceptions, but we go
further and return a no-work ``SliceOutcome`` with a diagnostic note
rather than propagating).
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
    "CompactionSliceConfig",
    "ContextCompactionSlice",
    "get_slice_provider_factory",
]

#: Minimum slice budget (ms) required to fire a between-thresholds
#: compaction. The LLM summarize call needs runway to start; below this
#: the slice defers to a less-pressured turn. The operation's own
#: ``timeout_seconds`` governs the call once committed.
_MIN_SLICE_BUDGET_MS = 150

#: Sidecar filename (beside the thread JSONL) holding the last-fire
#: counter and timestamp. Mirrors the ``_THREAD_DIGEST_FILE_NAME``
#: pattern in ``host/state_provider.py``.
_LAST_COMPACTION_FILE_NAME = "last_compaction.json"


@dataclass(slots=True)
class CompactionSliceConfig:
    """Runtime-resolved configuration handed to a constructed slice.

    Built by the factory from the session plan's ``CompactionTuning`` and
    host-provided paths. Kept separate from the pydantic
    ``CompactionTuning`` so the slice does not re-resolve the plan on
    every turn.
    """

    enabled: bool
    pressure_threshold: float
    hard_threshold: float
    min_interval_turns: int
    retention_count: int
    preferred_model_id: str
    summarize_operation: str
    thread_dir: Path | None = None
    turn_history_loader: Any = None
    """Optional callable returning ``list[dict]`` of the live turn
    history. When ``None`` the slice attempts to read it from
    ``turn_hooks`` via the same seam ``checkpoint_history_summarize``
    uses; when that is also unavailable the slice no-ops with a note."""


@dataclass(slots=True)
class _LastCompactionState:
    """Mutable in-memory mirror of the sidecar; persisted on each fire."""

    turn_count: int = 0
    last_fired_at: float = 0.0
    last_fired_turn: int = 0


class ContextCompactionSlice:
    """Slice provider that fires context compaction on utilization pressure.

    Constructed by the factory when ``CompactionTuning.enabled`` is true.
    The host wires ``slice`` into the trigger slot selected by
    ``CompactionTuning.trigger``.
    """

    def __init__(
        self,
        config: CompactionSliceConfig,
        *,
        turn_hooks: Any,
    ) -> None:
        self._config = config
        self._hooks = turn_hooks
        self._state = _LastCompactionState()
        self._load_state()

    # ── Sidecar persistence ──────────────────────────────────────────

    def _sidecar_path(self) -> Path | None:
        thread_dir = self._config.thread_dir
        if thread_dir is None:
            return None
        return Path(thread_dir) / _LAST_COMPACTION_FILE_NAME

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
            logger.warning("compaction: failed to load sidecar %s: %s", path, exc)

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
            logger.warning("compaction: failed to persist sidecar %s: %s", path, exc)

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
    ) -> tuple[bool, str]:
        """Apply the threshold + pressure + interval decision matrix.

        Returns ``(fire, reason)``. ``reason`` is always set so the
        ``SliceOutcome`` notes carry the rationale for observability.
        """
        cfg = self._config
        self._state.turn_count += 1

        if utilization >= cfg.hard_threshold:
            # Guardrail override: fire regardless of pressure or interval.
            return True, f"hard_threshold_override(utilization={utilization:.3f})"

        if utilization < cfg.pressure_threshold:
            return False, f"below_pressure_threshold(utilization={utilization:.3f})"

        # Between thresholds: respect the pressure curve and interval.
        if slice_budget_ms < _MIN_SLICE_BUDGET_MS:
            return False, (
                f"deferred_low_slice_budget(budget_ms={slice_budget_ms},"
                f"utilization={utilization:.3f})"
            )

        turns_since_last = self._state.turn_count - self._state.last_fired_turn
        if cfg.min_interval_turns > 0 and turns_since_last < cfg.min_interval_turns:
            return False, (
                f"deferred_min_interval(turns_since_last={turns_since_last},"
                f"min_interval={cfg.min_interval_turns})"
            )

        return True, f"pressure_threshold(utilization={utilization:.3f})"

    # ── Turn-history access ─────────────────────────────────────────

    def _load_turn_history(self) -> list[dict[str, Any]] | None:
        loader = self._config.turn_history_loader
        if callable(loader):
            try:
                result = loader()
                if isinstance(result, list):
                    return result
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("compaction: turn_history_loader failed: %s", exc)
                return None
        # Fall back to the kernel's execute_operation path: some hosts
        # expose history via the state provider or the checkpoint seam.
        # We do not invent a new seam here; if no loader is wired, the
        # slice no-ops with a note so the host can wire one.
        return None

    def _write_turn_history(self, new_history: list[dict[str, Any]]) -> bool:
        writer = getattr(self._hooks, "write_turn_history_fn", None)
        if not callable(writer):
            logger.warning(
                "compaction: write_turn_history_fn not wired on TurnHooks; "
                "compacted history will not be persisted"
            )
            return False
        try:
            writer(new_history)
            return True
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("compaction: write_turn_history_fn failed: %s", exc)
            return False

    # ── Slice entrypoint ────────────────────────────────────────────

    async def slice(self, ctx: Any, policy: SlicePolicy) -> SliceOutcome:
        """Fire compaction if utilization pressure warrants it.

        ``ctx`` is unused (matches the ``StateProvider.slice`` contract);
        all state is read from ``self._hooks`` and the sidecar.
        """
        del ctx  # explicit: the slice is stateless w.r.t. ctx
        started = time.monotonic()

        utilization = self._utilization()
        if utilization is None:
            return SliceOutcome(
                work_items_completed=0,
                elapsed_ms=0.0,
                last_item="",
                notes=("no_context_snapshot",),
            )

        fire, reason = self._should_fire(
            utilization=utilization,
            slice_budget_ms=int(getattr(policy, "budget_ms", 0) or 0),
        )
        if not fire:
            return SliceOutcome(
                work_items_completed=0,
                elapsed_ms=(time.monotonic() - started) * 1000.0,
                last_item="",
                notes=(reason,),
            )

        # Fire: load history, invoke the summarize operation, write back.
        turn_history = self._load_turn_history()
        if not turn_history:
            return SliceOutcome(
                work_items_completed=0,
                elapsed_ms=(time.monotonic() - started) * 1000.0,
                last_item="",
                notes=(f"{reason};no_turn_history",),
            )

        exec_op = getattr(self._hooks, "execute_operation", None)
        if not callable(exec_op):
            return SliceOutcome(
                work_items_completed=0,
                elapsed_ms=(time.monotonic() - started) * 1000.0,
                last_item="",
                notes=(f"{reason};no_execute_operation",),
            )

        op_id = self._config.summarize_operation
        params: dict[str, Any] = {
            "turn_history": turn_history,
            "retention_count": self._config.retention_count,
        }
        try:
            result = await exec_op(op_id, params)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("compaction: %s failed: %s", op_id, exc)
            return SliceOutcome(
                work_items_completed=0,
                elapsed_ms=(time.monotonic() - started) * 1000.0,
                last_item="",
                notes=(f"{reason};operation_error:{type(exc).__name__}",),
            )

        new_history = None
        output_json = getattr(result, "output_json", None)
        if isinstance(output_json, dict):
            maybe = output_json.get("turn_history")
            if isinstance(maybe, list):
                new_history = maybe

        if not new_history:
            return SliceOutcome(
                work_items_completed=0,
                elapsed_ms=(time.monotonic() - started) * 1000.0,
                last_item="",
                notes=(f"{reason};no_output_turn_history",),
            )

        written = self._write_turn_history(new_history)
        if not written:
            return SliceOutcome(
                work_items_completed=0,
                elapsed_ms=(time.monotonic() - started) * 1000.0,
                last_item="",
                notes=(f"{reason};write_back_skipped",),
            )

        old_count = len(turn_history)
        retained = len(new_history)
        self._state.last_fired_at = time.time()
        self._state.last_fired_turn = self._state.turn_count
        self._persist_state()

        return SliceOutcome(
            work_items_completed=1,
            elapsed_ms=(time.monotonic() - started) * 1000.0,
            last_item=f"compacted:{old_count}->{retained}",
            notes=(reason,),
        )


def _resolve_compaction_tuning(session_plan: Any) -> Any:
    """Pull the ``CompactionTuning`` off a resolved ``SessionPlan``.

    Tolerates plans that predate the ``compaction`` field by returning
    ``None``; the factory then returns ``None`` and the host skips
    wiring the slice for that session.
    """
    manifest = getattr(session_plan, "manifest", None)
    if manifest is None:
        return None
    context = getattr(manifest, "context", None)
    if context is None:
        return None
    history = getattr(context, "history", None)
    if history is None:
        return None
    return getattr(history, "compaction", None)


def get_slice_provider_factory() -> SliceProviderFactory:
    """Entry-point factory for the ``lisa.extensions.slice_providers`` group.

    Returns a callable ``(turn_hooks, session_plan) -> provider | None``.
    The host invokes it during ``TurnHooks`` assembly. When the session
    plan does not enable compaction, the factory returns ``None`` and
    the host skips wiring — no behavior change for that session.

    The host is expected to set ``provider.config.thread_dir`` and
    ``provider.config.turn_history_loader`` after construction if it
    wants sidecar persistence and live history loading. Defaults
    (``None``) make the slice a no-op with diagnostic notes, which is
    the safe behavior for hosts that have not wired those seams.
    """

    def factory(turn_hooks: Any, session_plan: Any) -> ContextCompactionSlice | None:
        tuning = _resolve_compaction_tuning(session_plan)
        if tuning is None or not bool(getattr(tuning, "enabled", False)):
            return None
        config = CompactionSliceConfig(
            enabled=bool(tuning.enabled),
            pressure_threshold=float(tuning.pressure_threshold),
            hard_threshold=float(tuning.hard_threshold),
            min_interval_turns=int(tuning.min_interval_turns),
            retention_count=int(tuning.retention_count),
            preferred_model_id=str(tuning.preferred_model_id),
            summarize_operation=str(tuning.summarize_operation),
        )
        return ContextCompactionSlice(config=config, turn_hooks=turn_hooks)

    return factory

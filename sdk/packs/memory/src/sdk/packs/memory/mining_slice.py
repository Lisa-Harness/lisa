"""Slice-driven thread mining: sealed segments to durable semantic memories.

The host wires the returned slice provider into ``TurnHooks.post_turn_slice``
(or ``pre_turn_slice`` / ``explicit_slice`` per ``MemoryMiningTuning.trigger``)
via the ``lisa.extensions.slice_providers`` entry-point group. Each turn, after
the user has seen the answer, the provider:

1. Reads the thread store's sparse segment index (one small file — no
   decompression) for sealed segments not yet mined, tracked by a pack-local
   ``mine.marker`` sidecar.
2. Loads each newly sealed segment's turns via ``ThreadStore.load_range``
   (only that segment is decompressed).
3. Invokes the ``mine_thread_segment`` operation with a real
   ``OperationExecutionContext``, which distills the turns into durable
   semantic records with write-time dedup (create/update/skip per fact).

Mining is bounded (``max_segments_per_slice`` per pass) and idempotent: the
marker advances only after a confirmed success (or a segment proven
unreadable, or an intentional all-skip result), and the write-time dedup in
the operation makes even a re-mined segment non-destructive. A returned
``status="error"`` result — the operation's retryable storage-failure
outcome — leaves the segment pending and stops the pass: the next slice
retries it, and dedup keeps the re-mine safe even after a partial write.
The marker matches by *position* in the index, not exact id, so a rebuilt or
re-sealed store cannot strand the marker mid-scan — a lost marker re-mines
from the start and the dedup skips what is already stored.

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

from kernel.models import OperationExecutionContext
from kernel.seams.host_adapter import build_callable_host
from kernel.slice import SliceOutcome, SlicePolicy
from kernel.slice_provider_discovery import SliceProviderFactory
from sdk.thread import ThreadStore

logger = logging.getLogger(__name__)

__all__ = [
    "MemoryMiningSlice",
    "MemoryMiningSliceConfig",
    "get_slice_provider_factory",
]

#: Sidecar filename holding the last-mined thread segment id. Mining is
#: idempotent per segment: a sealed segment is mined exactly once.
_LAST_MINED_SEGMENT_FILE_NAME = "mine.marker"

#: Fallback when the view exposes no segments-per-slice bound.
_MAX_SEGMENTS_PER_SLICE = 2


@dataclass(slots=True)
class MemoryMiningSliceConfig:
    """Runtime-resolved configuration handed to a constructed slice.

    Built by the factory from the session plan's ``MemoryMiningTuning``.
    ``memory_data_dir`` is the host's durable data root and ``user_slug``
    the active user; the thread store lives at
    ``<data_dir>/threads/<user_slug>`` (its segment index at
    ``<base>.index.jsonl``).
    """

    enabled: bool
    preferred_model_id: str
    mine_operation: str
    max_segments_per_slice: int
    memory_data_dir: Path | None = None
    user_slug: str = "default"
    workspace_id: str = "default"
    memory_scope: str = "personal"
    surface_id: str = ""


class MemoryMiningSlice:
    """Slice provider that mines newly sealed thread segments."""

    def __init__(
        self,
        config: MemoryMiningSliceConfig,
        *,
        turn_hooks: Any,
    ) -> None:
        self._config = config
        self._hooks = turn_hooks

    # ── Sidecar persistence ──────────────────────────────────────────

    def _mine_marker_path(self) -> Path | None:
        """Path of the ``mine.marker`` sidecar, or ``None`` when unlocatable."""
        data_dir = self._config.memory_data_dir
        if data_dir is None:
            return None
        return Path(data_dir) / _LAST_MINED_SEGMENT_FILE_NAME

    def _load_last_mined_seg_id(self) -> str:
        path = self._mine_marker_path()
        if path is None or not path.exists():
            return ""
        try:
            return path.read_text(encoding="utf-8").strip()
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("memory_mining: failed to read mine.marker: %s", exc)
            return ""

    def _persist_mine_marker(self, seg_id: str) -> None:
        path = self._mine_marker_path()
        if path is None or not seg_id:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(seg_id, encoding="utf-8")
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("memory_mining: failed to persist mine.marker: %s", exc)

    # ── Thread store access ──────────────────────────────────────────

    def _thread_base(self) -> Path | None:
        data_dir = self._config.memory_data_dir
        if data_dir is None or not self._config.user_slug:
            return None
        return Path(data_dir) / "threads" / self._config.user_slug

    def _segment_index(self, base: Path) -> list[dict[str, Any]]:
        """Read the thread store's sparse segment index as decoded records."""
        index_path = Path(f"{base}.index.jsonl")
        if not index_path.exists():
            return []
        try:
            raw_lines = index_path.read_text(encoding="utf-8").splitlines()
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("memory_mining: failed to read segment index: %s", exc)
            return []
        records: list[dict[str, Any]] = []
        for raw_line in raw_lines:
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                decoded = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(decoded, dict) and decoded.get("seg_id"):
                records.append(decoded)
        return records

    def _sealed_segments_to_mine(self) -> list[tuple[str, int, int]]:
        """List unmined sealed segments as ``(seg_id, turn_lo, turn_hi)``.

        The marker is positional: everything after the marker's segment in
        the index is pending, whether or not the marker's id still exists.
        A marker id missing from the index (rebuilt store, index rewrite)
        therefore mines from the start instead of silently never mining
        again — safe because the operation's write-time dedup skips
        records that already exist.
        """
        base = self._thread_base()
        if base is None:
            return []
        index = self._segment_index(base)
        if not index:
            return []
        last_mined = self._load_last_mined_seg_id()
        marker_ids = {str(record.get("seg_id", "")) for record in index}
        # Positional recovery: a marker id absent from the index (rebuilt
        # store, index rewrite) means the position is unknowable — mine from
        # the start. Safe because the operation's write-time dedup skips
        # records that already exist.
        seen_marker = not last_mined or last_mined not in marker_ids
        pending: list[tuple[str, int, int]] = []
        for record in index:
            seg_id = str(record.get("seg_id", ""))
            if not seen_marker:
                if seg_id == last_mined:
                    seen_marker = True
                continue
            pending.append(
                (
                    seg_id,
                    int(record.get("turn_lo", 0)),
                    int(record.get("turn_hi", 0)),
                )
            )
        cap = max(1, self._config.max_segments_per_slice or _MAX_SEGMENTS_PER_SLICE)
        return pending[:cap]

    def _load_segment_turns(
        self, base: Path, seg_id: str, turn_lo: int, turn_hi: int
    ) -> list[dict[str, Any]]:
        """Load one sealed segment's turns by its global turn range."""
        try:
            return ThreadStore(base).load_range(
                start_turn=turn_lo,
                end_turn=turn_hi,
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("memory_mining: failed to load segment %s: %s", seg_id, exc)
            return []

    def _build_mining_context(self) -> OperationExecutionContext:
        """Build the mining operation's execution context.

        A real ``OperationExecutionContext`` — not a bare params dict — so
        the host's ``_execute_operation`` wrapper can stamp fields on it and
        the operation's ``ctx.host``/tool/model seams resolve through the
        injected dispatchers.
        """
        exec_tool = getattr(self._hooks, "execute_tool", None)
        exec_op = getattr(self._hooks, "execute_operation", None)
        resolve_model = getattr(self._hooks, "resolve_model", None)
        read_authored = getattr(self._hooks, "read_authored_text", None)
        complete = getattr(self._hooks, "complete", None)
        host = build_callable_host(
            execute_tool_fn=exec_tool,
            execute_operation_fn=exec_op,
            resolve_model_fn=resolve_model,
            read_authored_text_fn=read_authored,
            complete_fn=complete,
        )
        return OperationExecutionContext(
            workspace_id=self._config.workspace_id,
            user_id=self._config.user_slug,
            user_slug=self._config.user_slug,
            principal_key=self._config.user_slug,
            surface_id=self._config.surface_id,
            memory_scope=self._config.memory_scope,  # type: ignore[arg-type]
            host=host,
        )

    # ── Slice entrypoint ─────────────────────────────────────────────

    async def _mine_new_segments(self, policy: SlicePolicy, started: float) -> tuple[int, str]:
        """Mine newly sealed thread segments into semantic memories.

        Returns ``(work_items, last_item)``. The budget bounds the pass:
        each segment's operation call is checked against the remaining
        slice budget before it starts.

        The marker advances only on a confirmed success or an intentional
        all-skip result. An exception or ``status="error"`` result leaves the
        segment pending and stops the pass so a later success cannot advance
        the marker past it. Its retry cost is bounded by the per-pass budget,
        and write-time dedup makes each retry harmless.
        """
        exec_op = getattr(self._hooks, "execute_operation", None)
        if not callable(exec_op):
            return 0, ""
        segments = self._sealed_segments_to_mine()
        if not segments:
            return 0, ""
        base = self._thread_base()
        if base is None:
            return 0, ""

        op_ctx = self._build_mining_context()
        work_items = 0
        last_item = ""
        for seg_id, turn_lo, turn_hi in segments:
            if policy.budget_ms > 0 and (time.monotonic() - started) * 1000.0 >= float(
                policy.budget_ms
            ):
                break
            turns = self._load_segment_turns(base, seg_id, turn_lo, turn_hi)
            if not turns:
                # Segment unreadable — advance the marker so it is not retried.
                self._persist_mine_marker(seg_id)
                continue
            op_ctx.params = {
                "turns": turns,
                "user_slug": self._config.user_slug,
                "workspace_id": self._config.workspace_id,
                "memory_scope": self._config.memory_scope,
                "surface_id": self._config.surface_id,
            }
            try:
                result = await exec_op(self._config.mine_operation, op_ctx)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning(
                    "memory_mining: %s failed for segment %s; leaving it pending: %s",
                    self._config.mine_operation,
                    seg_id,
                    exc,
                )
                break
            if getattr(result, "status", "") == "error":
                # Retryable storage failure (the operation's documented
                # outcome contract): leave the segment pending and stop the
                # pass — a failing backend would fail the remaining segments
                # too, and the next slice retries from here. Write-time
                # dedup makes the re-mine safe, including after a partial
                # write.
                logger.warning(
                    "memory_mining: %s reported storage failure for segment %s; leaving it pending",
                    self._config.mine_operation,
                    seg_id,
                )
                break
            extra = getattr(result, "extra", {}) or {}
            stored_ids = extra.get("memory_ids") or []
            self._persist_mine_marker(seg_id)
            work_items += 1
            last_item = f"mined:{seg_id}({turn_lo}-{turn_hi})->{len(stored_ids)}"
        return work_items, last_item

    async def slice(self, ctx: Any, policy: SlicePolicy) -> SliceOutcome:
        """Mine new sealed segments; empty outcome when there is nothing to do."""
        del ctx  # explicit: the slice is stateless w.r.t. ctx
        started = time.monotonic()
        try:
            work_items, last_item = await self._mine_new_segments(policy, started)
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("memory_mining: mining pass failed: %s", exc)
            work_items, last_item = 0, ""
        notes: tuple[str, ...] = (f"mined {work_items} sealed segment(s)",) if work_items else ()
        return SliceOutcome(
            work_items_completed=work_items,
            elapsed_ms=(time.monotonic() - started) * 1000.0,
            last_item=last_item,
            notes=notes,
        )


def _resolve_mining_tuning(session_plan: Any) -> Any:
    """Pull the ``MemoryMiningTuning`` off a resolved ``SessionPlan``."""
    manifest = getattr(session_plan, "manifest", None)
    if manifest is None:
        return None
    context = getattr(manifest, "context", None)
    if context is None:
        return None
    memory = getattr(context, "memory", None)
    if memory is None:
        return None
    return getattr(memory, "mining", None)


def get_slice_provider_factory() -> SliceProviderFactory:
    """Entry-point factory for the ``lisa.extensions.slice_providers`` group.

    Returns a callable ``(turn_hooks, session_plan) -> provider | None``.
    The host invokes it during ``TurnHooks`` assembly. When the session
    plan does not enable mining, the factory returns ``None`` and the host
    skips wiring — no behavior change for that session.
    """

    def factory(turn_hooks: Any, session_plan: Any) -> MemoryMiningSlice | None:
        tuning = _resolve_mining_tuning(session_plan)
        if tuning is None or not bool(getattr(tuning, "enabled", False)):
            return None
        memory_data_dir = getattr(turn_hooks, "memory_data_dir", None)
        user_slug = str(getattr(turn_hooks, "user_slug", "") or "")
        config = MemoryMiningSliceConfig(
            enabled=bool(tuning.enabled),
            preferred_model_id=str(tuning.preferred_model_id),
            mine_operation=str(tuning.mine_operation),
            max_segments_per_slice=int(
                getattr(tuning, "max_segments_per_slice", _MAX_SEGMENTS_PER_SLICE)
            ),
            memory_data_dir=Path(memory_data_dir) if memory_data_dir else None,
            user_slug=user_slug,
            workspace_id=str(getattr(turn_hooks, "workspace_id", "") or "default"),
            memory_scope=str(getattr(turn_hooks, "memory_scope", "") or "personal"),
            surface_id=str(getattr(turn_hooks, "surface_id", "") or ""),
        )
        return MemoryMiningSlice(config=config, turn_hooks=turn_hooks)

    return factory

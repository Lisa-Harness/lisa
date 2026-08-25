"""Transport-neutral state provider and slice compaction.

This module owns:

- :func:`build_unified_state` — compiles a :class:`UnifiedState` from a
  dict-based state snapshot (loaded via ``--state-snapshot``) without
  touching global runtime state.
- :class:`FileBackedStateProvider` — a file-only digest provider that runs
  one deterministic, no-LLM slice operation: compacting overflowed thread
  JSONL entries into a ``thread_digest.md`` sidecar.
- :func:`_slice_budget_expired` — a private helper used by the provider to
  honour the slice time budget.

All functions are transport-neutral: they take explicit ``thread_path`` and
``state_snapshot`` parameters rather than reading global runtime state. A
transport host wires its own ``HostRuntime`` paths and snapshot into these
functions at startup.

Dependency direction
--------------------
``state_provider.py`` ← ``main.py`` / ``turn_hooks_builder.py``  (callers import
from state_provider; state_provider never imports main or any transport
code)

Public surface
--------------
- :func:`build_unified_state`
- :class:`FileBackedStateProvider`
"""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kernel.models import (
    ContextItem,
    UnifiedState,
)
from kernel.seams.protocols import ExtensionItemBuilder
from kernel.slice import SliceOutcome
from pydantic import ValidationError

from host.host_commands import is_host_command_query as _is_host_command_query
from host.lisa_config import _DEFAULT_THREAD_MAX_ENTRIES

logger = logging.getLogger(__name__)


def build_unified_state(
    *,
    state_snapshot: dict[str, Any] | None,
    item_builder: ExtensionItemBuilder | None = None,
    runtime_context: Any = None,
) -> UnifiedState:
    """Return a compiled :class:`UnifiedState` for operations and the context mapper.

    Builds the state from the dict-based ``state_snapshot`` (loaded via
    ``--state-snapshot``). Intake-produced context items (signals, intel,
    alerts) are merged with hook-produced extension items
    (``item_builder.build_extension_context_items(runtime_context, state)``)
    into ``context_items`` — the single feed the session-owned
    context mapper walks to produce ranked, lane-ordered context without a
    live MCP round-trip.

    Context admission and ordering policy is not carried on the state: it
    lives in the session plan's ``context`` block (``ContextPlan``), which the
    host passes to the mapper alongside this state.

    Returns an empty :class:`UnifiedState` when ``state_snapshot`` is
    ``None`` or falsy.
    """
    compiled_at = datetime.now(UTC).isoformat()
    snapshot = state_snapshot
    if not snapshot:
        state = UnifiedState(compiled_at=compiled_at)
    else:
        state = UnifiedState(
            compiled_at=str(
                snapshot.get("compiled_at") or snapshot.get("generated_at") or compiled_at
            ),
        )
        # Intake-produced context items (signals, intel, alerts,
        # self_model) is the single feed. No arbitrary named buckets.
        state.context_items = _coerce_items(
            snapshot.get("context_items")
        )

    # Invoke the extension context-item hook so installed packs contribute
    # ContextItem records (self_model, Cloudflare, GitHub, Stripe, etc.).
    # The hook is the only coupling between packs and the host context path.
    if item_builder is not None:
        try:
            contributed = item_builder.build_extension_context_items(
                runtime_context, state
            )
        except Exception:  # pylint: disable=broad-except
            logger.warning(
                "extension item builder raised; continuing with intake items only.",
                exc_info=True,
            )
            contributed = []
        if contributed:
            existing_ids = {e.item_id for e in state.context_items}
            for item in contributed:
                if not isinstance(item, ContextItem):
                    continue
                if item.item_id in existing_ids:
                    continue
                state.context_items.append(item)
                existing_ids.add(item.item_id)

    return state


def load_state_snapshot(
    *,
    explicit_path: str | None,
    data_dir: Path,
) -> dict[str, Any] | None:
    """Load the runtime state snapshot dict from disk.

    Resolution order:
    1. ``explicit_path`` (``--state-snapshot``) — must exist, else raise.
    2. ``<data_dir>/state.json`` — the default intake output. When the
       file does not exist, return ``None`` (intake has not run yet) so
       the host boots cleanly on a fresh data dir instead of crashing.

    Returns ``None`` when no snapshot is available. Raises ``ValueError``
    when an explicit path was given but the file is missing or not a JSON
    object.
    """
    configured = (explicit_path or "").strip()
    if configured:
        file_path = Path(configured).expanduser()
        source = "explicit"
    else:
        file_path = data_dir / "state.json"
        source = "default"

    if not file_path.exists():
        if source == "explicit":
            logger.error("state snapshot file not found: %s", file_path)
            raise ValueError(f"State snapshot file not found: {file_path}")
        return None

    try:
        decoded = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.error("state snapshot file %s is unreadable: %s", file_path, exc)
        raise ValueError(f"State snapshot file {file_path} is unreadable: {exc}") from exc

    if not isinstance(decoded, dict):
        logger.error("state snapshot file must contain a JSON object: %s", file_path)
        raise ValueError("State snapshot file must contain a JSON object.")

    logger.info("loaded state snapshot from %s (%d keys)", file_path, len(decoded))
    return decoded


def _coerce_items(raw_items: Any) -> list[ContextItem]:
    """Coerce a raw list of item dicts into a list of :class:`ContextItem`."""
    if not isinstance(raw_items, list):
        return []
    try:
        return [
            e if isinstance(e, ContextItem) else ContextItem.model_validate(e)
            for e in raw_items
        ]
    except (ValidationError, TypeError):
        return []


def _slice_budget_expired(policy: Any, started: float) -> bool:
    """Return ``True`` if the slice budget has been consumed.

    A budget of zero or negative means "no budget" (never expires).
    """
    budget_ms = getattr(policy, "budget_ms", 0) or 0
    if budget_ms <= 0:
        return False
    return (time.monotonic() - started) * 1000.0 >= budget_ms


class FileBackedStateProvider:
    """File-only digest provider for hosts with a segmented thread store.

    Runs one deterministic, no-LLM operation inside the existing slice
    budget:

    1. Thread digest: summarize sealed (compressed) thread segments into
       the per-user ``<base>.digest.md`` sidecar. The active segment is
       never truncated — sealing (in ``host.thread``) already moves old
       turns out of it. This slice digests sealed segments that haven't
       been digested yet (tracked by ``<base>.digest.marker``).

    Memory eviction is not host-side: ``lisa-memory``'s SQLite store
    enforces TTL via its ``expires_at`` column, so there is nothing for
    the slice to do.
    """

    def __init__(
        self,
        *,
        thread_path: Path | None,
        max_thread_entries: int = _DEFAULT_THREAD_MAX_ENTRIES,
        digest_max_lines: int | None = None,
        digest_summary_max_chars: int | None = None,
    ) -> None:
        self._thread_path = thread_path
        # ``max_thread_entries`` bounds how many sealed segments to digest
        # per slice (budget-bounded), not a truncation threshold.
        self._max_thread_entries = max(1, max_thread_entries)
        self._digest_max_lines = max(1, digest_max_lines or 200)
        self._digest_summary_max_chars = max(60, digest_summary_max_chars or 240)

    def __call__(self) -> Any:
        return None

    async def slice(self, ctx: Any, policy: Any) -> SliceOutcome:
        """Run bounded thread compaction."""
        del ctx
        started = time.monotonic()
        work_items = 0
        notes: list[str] = []
        last_item = ""

        thread_compacted = self._compact_thread(policy, started)
        if thread_compacted[0] > 0:
            work_items += thread_compacted[0]
            last_item = thread_compacted[1]
            notes.append(thread_compacted[2])

        elapsed_ms = (time.monotonic() - started) * 1000.0
        return SliceOutcome(
            work_items_completed=work_items,
            elapsed_ms=elapsed_ms,
            last_item=last_item,
            notes=tuple(notes),
        )

    def _build_digest_from_entry(self, entry: dict[str, Any]) -> str | None:
        """Build one digest line from a thread entry, or ``None`` to skip."""
        query = str(entry.get("query") or "").strip()
        response = str(entry.get("response_text") or "").strip()
        if not query and not response:
            return None
        # Strip host-only commands from the digest sidecar too.
        if _is_host_command_query(query):
            return None
        preview = response[:200] if response else query[:200]
        return f"- {query or 'unknown'}: {preview}"

    def _digest_segment(
        self,
        record: Any,
        paths_base: Path,
        policy: Any,
        started: float,
    ) -> tuple[list[str], int] | None:
        """Decompress and digest one sealed segment. Returns (lines, count) or None."""
        from host.thread import (  # noqa: PLC0415  # lazy: breaks cycle
            _decompress_bytes,
            _parse_entries,
            _sealed_segment_path,
        )

        if _slice_budget_expired(policy, started):
            return None
        sealed_path = _sealed_segment_path(paths_base, record.seg_id)
        if not sealed_path.exists():
            return [], 0
        try:
            decompressed = _decompress_bytes(
                sealed_path.read_bytes(), Path(f"{paths_base}.dict.zstd")
            )
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("digest: failed to decompress %s: %s", record.seg_id, exc)
            return [], 0
        entries = _parse_entries(decompressed.decode("utf-8").splitlines())
        lines: list[str] = []
        for entry in entries:
            digest_line = self._build_digest_from_entry(entry)
            if digest_line is not None:
                lines.append(digest_line)
        return lines, len(entries)

    def _compact_thread(
        self,
        policy: Any,
        started: float,
    ) -> tuple[int, str, str]:
        """Digest sealed thread segments into the per-user digest sidecar.

        Returns ``(entries_digested, last_item, note)``.
        """
        if self._thread_path is None or _slice_budget_expired(policy, started):
            return 0, "", ""

        from host.thread import _read_index  # noqa: PLC0415  # lazy: breaks cycle

        paths_base = self._thread_path
        index_path = Path(f"{paths_base}.index.jsonl")
        digest_path = Path(f"{paths_base}.digest.md")
        marker_path = Path(f"{paths_base}.digest.marker")

        records = _read_index(index_path)
        if not records:
            return 0, "", ""

        # Read the last-digested segment id.
        last_digested = ""
        if marker_path.exists():
            last_digested = marker_path.read_text(encoding="utf-8").strip()

        # Find undigested sealed segments, budget-bounded.
        digested_ids = {last_digested} if last_digested else set()
        to_digest = [r for r in records if r.seg_id not in digested_ids]
        if not to_digest:
            return 0, "", ""
        to_digest = to_digest[: self._max_thread_entries]

        digest_parts: list[str] = []
        entries_digested = 0
        last_seg_id = last_digested
        for record in to_digest:
            result = self._digest_segment(record, paths_base, policy, started)
            if result is None:
                break  # budget expired
            lines, count = result
            digest_parts.extend(lines)
            entries_digested += count
            last_seg_id = record.seg_id

        if not digest_parts:
            return 0, "", ""

        existing = ""
        if digest_path.is_file():
            existing = digest_path.read_text(encoding="utf-8").strip()
        new_block = "\n".join(digest_parts)
        digest_text = f"{existing}\n\n{new_block}" if existing else new_block
        digest_text = self._apply_digest_cap(digest_text)
        digest_path.parent.mkdir(parents=True, exist_ok=True)
        digest_path.write_text(digest_text, encoding="utf-8")

        # Update the marker.
        if last_seg_id:
            marker_path.parent.mkdir(parents=True, exist_ok=True)
            marker_path.write_text(last_seg_id, encoding="utf-8")

        note = f"digested {entries_digested} thread entries from {len(to_digest)} segment(s)"
        return entries_digested, f"thread:{entries_digested} entries", note

    def _apply_digest_cap(self, digest_text: str) -> str:
        """Bound digest growth and prepend one lossy summary when capped."""

        lines = [line for line in digest_text.splitlines() if line.strip()]
        if len(lines) <= self._digest_max_lines:
            return "\n".join(lines)

        dropped_count = len(lines) - self._digest_max_lines
        kept = lines[-self._digest_max_lines :]
        summary = (
            f"- [lossy_summary] Omitted {dropped_count} older digest lines "
            f"to enforce digest_max_lines={self._digest_max_lines}."
        )
        summary = summary[: self._digest_summary_max_chars].rstrip()

        if self._digest_max_lines == 1:
            return summary
        return "\n".join([summary, *kept[-(self._digest_max_lines - 1) :]])

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
``state_provider.py`` ← ``main.py`` / ``turn_hooks.py``  (callers import
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
    AnomalyAlert,
    ContextRoutingConfig,
    DomainKnowledgeArtifact,
    IntelCollection,
    UnifiedState,
)
from kernel.slice import SliceOutcome
from pydantic import ValidationError

from host.host_commands import is_host_command_query as _is_host_command_query
from host.lisa_config import _DEFAULT_THREAD_MAX_ENTRIES, _THREAD_DIGEST_FILE_NAME

logger = logging.getLogger(__name__)


def build_unified_state(
    *,
    state_snapshot: dict[str, Any] | None,
    context_routing: ContextRoutingConfig | None = None,
) -> UnifiedState:
    """Return a compiled :class:`UnifiedState` for operations and the context broker.

    Builds the state from the dict-based ``state_snapshot`` (loaded via
    ``--state-snapshot``), populating ``signal_snapshots``,
    ``intel_collections``, ``active_alerts``, and ``domain_knowledge`` from
    the snapshot's keys. The broker's evidence-catalog builder then walks
    these fields plus extension-contributed evidence to produce a ranked
    :class:`ContextPacket` without a live MCP round-trip.

    Returns an empty :class:`UnifiedState` when ``state_snapshot`` is
    ``None`` or falsy.
    """
    compiled_at = datetime.now(UTC).isoformat()
    snapshot = state_snapshot
    if not snapshot:
        return UnifiedState(compiled_at=compiled_at, context_routing=context_routing)

    state = UnifiedState(
        compiled_at=str(snapshot.get("compiled_at") or snapshot.get("generated_at") or compiled_at),
        context_routing=context_routing,
    )

    raw_signals = snapshot.get("signal_snapshots")
    if isinstance(raw_signals, dict):
        state.signal_snapshots = raw_signals

    state.active_alerts = _coerce_alerts(snapshot.get("active_alerts"))
    state.intel_collections = _coerce_intel(snapshot.get("intel_collections"))
    state.domain_knowledge = _coerce_domain(snapshot.get("domain_knowledge"))

    thinking = snapshot.get("thinking_vibe")
    if isinstance(thinking, str) and thinking.strip():
        state.thinking_vibe = thinking.strip()

    strategy = snapshot.get("strategy_vibe")
    if isinstance(strategy, str) and strategy.strip():
        state.strategy_vibe = strategy.strip()

    self_model_vibe = snapshot.get("self_model_vibe")
    if isinstance(self_model_vibe, str) and self_model_vibe.strip():
        state.self_model_vibe = self_model_vibe.strip()

    reflection_vibe = snapshot.get("reflection_vibe")
    if isinstance(reflection_vibe, str) and reflection_vibe.strip():
        state.reflection_vibe = reflection_vibe.strip()

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


def _coerce_alerts(raw_alerts: Any) -> list[AnomalyAlert]:
    """Coerce a raw list of alert dicts into a list of :class:`AnomalyAlert`."""
    if not isinstance(raw_alerts, list):
        return []
    try:
        return [
            a if isinstance(a, AnomalyAlert) else AnomalyAlert.model_validate(a) for a in raw_alerts
        ]
    except (ValidationError, TypeError):
        return []


def _coerce_intel(raw_intel: Any) -> dict[str, IntelCollection]:
    """Coerce a raw dict of intel entries into ``dict[str, IntelCollection]``."""
    if not isinstance(raw_intel, dict):
        return {}
    try:
        return {
            k: v if isinstance(v, IntelCollection) else IntelCollection.model_validate(v)
            for k, v in raw_intel.items()
        }
    except (ValidationError, TypeError):
        return {}


def _coerce_domain(raw_domain: Any) -> dict[str, DomainKnowledgeArtifact]:
    """Coerce a raw dict of domain-knowledge entries into ``dict[str, DomainKnowledgeArtifact]``."""
    if not isinstance(raw_domain, dict):
        return {}
    try:
        return {
            k: (
                v
                if isinstance(v, DomainKnowledgeArtifact)
                else DomainKnowledgeArtifact.model_validate(v)
            )
            for k, v in raw_domain.items()
        }
    except (ValidationError, TypeError):
        return {}


def _slice_budget_expired(policy: Any, started: float) -> bool:
    """Return ``True`` if the slice budget has been consumed.

    A budget of zero or negative means "no budget" (never expires).
    """
    budget_ms = getattr(policy, "budget_ms", 0) or 0
    if budget_ms <= 0:
        return False
    return (time.monotonic() - started) * 1000.0 >= budget_ms


class FileBackedStateProvider:
    """File-only digest provider for hosts with a thread JSONL.

    Runs one deterministic, no-LLM operation inside the existing slice
    budget:

    1. Thread digest: when the thread JSONL exceeds ``max_thread_entries``,
       compact the oldest overflow entries into a ``thread_digest.md``
       sidecar and truncate the JSONL to the most recent entries.

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

    def _compact_thread(
        self,
        policy: Any,
        started: float,
    ) -> tuple[int, str, str]:
        """Compact overflowed thread entries into a digest sidecar.

        Returns ``(entries_compacted, last_item, note)``.
        """
        if self._thread_path is None or not self._thread_path.exists():
            return 0, "", ""
        if _slice_budget_expired(policy, started):
            return 0, "", ""

        lines = self._thread_path.read_text(encoding="utf-8").splitlines()
        if len(lines) <= self._max_thread_entries:
            return 0, "", ""

        overflow_count = len(lines) - self._max_thread_entries
        overflow_lines = lines[:overflow_count]
        digest_parts: list[str] = []
        for raw_line in overflow_lines:
            try:
                entry = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if not isinstance(entry, dict):
                continue
            query = str(entry.get("query") or "").strip()
            response = str(entry.get("response_text") or "").strip()
            if not query and not response:
                continue
            # Strip host-only commands (``/pack``, ``/help``, ``/tokens`` ...)
            # from the digest sidecar too: they are operator UI gestures,
            # not dialogue, and would otherwise accumulate in the digest
            # forever (every ``/pack disable`` you ever ran would live on
            # in the model's "prior conversation digest" lane).
            if _is_host_command_query(query):
                continue
            preview = response[:200] if response else query[:200]
            digest_parts.append(f"- {query or 'unknown'}: {preview}")

        digest_path = self._thread_path.parent / _THREAD_DIGEST_FILE_NAME
        existing = ""
        if digest_path.is_file():
            existing = digest_path.read_text(encoding="utf-8").strip()
        new_block = "\n".join(digest_parts)
        digest_text = f"{existing}\n\n{new_block}" if existing else new_block
        digest_text = self._apply_digest_cap(digest_text)
        digest_path.write_text(digest_text, encoding="utf-8")

        remaining_lines = lines[overflow_count:]
        self._thread_path.write_text(
            "\n".join(remaining_lines) + "\n" if remaining_lines else "",
            encoding="utf-8",
        )

        note = f"compacted {overflow_count} thread entries into digest"
        return overflow_count, f"thread:{overflow_count} entries", note

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

"""Read one bounded forensic window from the host's data root.

``ForensicReader`` is the only component that touches the filesystem for
forensic reads. It composes the pieces so the semantics of "what evidence
exists for turns N..M" live in exactly one place:

- thread entries from :class:`sdk.thread.ThreadStore` (the durable turn record)
- ``turn.decision`` / ``tool.telemetry`` from ``logs/decisions.jsonl``
- model-call usage from ``logs/token-usage.jsonl``
- warning/error lines from ``logs/runtime.jsonl`` (rotation-aware)

Two contracts this reader holds, both learned from the introspection path:

**Degradation is reported, never absorbed.** A missing log yields an empty
list *and* an entry in ``sources.missing``. Presenting "the file was not there"
as "there were no errors" is the failure that makes a diagnostic surface
actively harmful.

**Every read is bounded.** Turns, records, log lines, and bytes all have caps,
and hitting one appends a note. A forensic call that silently truncates is
worse than one that says it did.

Paths are derived from the injected data root only — never from an argument —
so a caller cannot address another user's evidence by naming a path.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sdk.forensics._join import build_bundle, turn_from_entry
from sdk.forensics._logrec import (
    DEFAULT_MAX_BYTES,
    filter_window,
    iter_jsonl_tail,
)
from sdk.forensics._models import BundleSources, ForensicBundle, TurnEvidence
from sdk.layout import DATA_DIR, LOGS_DIR
from sdk.thread import ThreadRedactionPolicy, ThreadStore, resolve_redaction_policy

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_LOG_ERROR_LIMIT",
    "DEFAULT_MAX_TURNS",
    "DEFAULT_WINDOW_SECONDS",
    "ForensicReader",
]

#: Trailing window for log-discovered evidence (errors and costs).
DEFAULT_WINDOW_SECONDS = 24 * 60 * 60
#: Hard ceiling on turns per read. A bundle is a summary; wider ranges should
#: be a second call with a narrower question.
DEFAULT_MAX_TURNS = 20
#: Hard ceiling on log lines included in a bundle.
DEFAULT_LOG_ERROR_LIMIT = 50

_ERROR_LEVELS = frozenset({"error", "warning", "critical"})
_DECISIONS_FILENAME = "decisions.jsonl"
_TOKEN_USAGE_FILENAME = "token-usage.jsonl"
_RUNTIME_LOG_FILENAME = "runtime.jsonl"


class ForensicReader:
    """Compose one forensic bundle from a host data root and thread store.

    Args:
        data_dir: The host's resolved data root. Log paths are derived from it.
        thread_dir: The directory holding per-user thread stores.
        user_slug: The user whose thread is read. Selects the store; it is
            never taken from a tool argument, so a caller cannot request
            another user's history.
        redaction: Optional read-time redaction applied to every thread-derived
            string. Threads persist raw prompts and reasoning, so a forensic
            read is exactly the surface that must respect this policy.
        window_seconds: Trailing window for time-filtered log evidence.
        max_turns: Hard ceiling on turns returned by one read.
        max_bytes: Byte budget per telemetry file read.
    """

    def __init__(
        self,
        data_dir: Path | str,
        *,
        thread_dir: Path | str | None = None,
        user_slug: str = "",
        redaction: ThreadRedactionPolicy | None = None,
        resolve_redaction_from_env: bool = True,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        max_turns: int = DEFAULT_MAX_TURNS,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._thread_dir = Path(thread_dir) if thread_dir else self._data_dir / DATA_DIR / "threads"
        self._user_slug = user_slug
        # An explicit policy always wins. Otherwise resolve the shared
        # environment contract, so a host that configured redaction does not
        # silently lose it simply because the pack constructed the reader.
        # ``resolve_redaction_from_env=False`` opts a caller out (tests that
        # need raw thread text for assertions).
        self._redaction: ThreadRedactionPolicy | None
        if redaction is not None:
            self._redaction = redaction
        elif resolve_redaction_from_env:
            self._redaction = resolve_redaction_policy()
        else:
            self._redaction = None
        self._window_seconds = window_seconds
        self._max_turns = max(1, max_turns)
        self._max_bytes = max_bytes

    # ── Paths ──────────────────────────────────────────────────────────

    @property
    def decisions_path(self) -> Path:
        """Return the decision/dispatch ledger path."""
        return self._data_dir / LOGS_DIR / _DECISIONS_FILENAME

    @property
    def token_usage_path(self) -> Path:
        """Return the token usage log path."""
        return self._data_dir / LOGS_DIR / _TOKEN_USAGE_FILENAME

    @property
    def runtime_log_path(self) -> Path:
        """Return the structured runtime log path."""
        return self._data_dir / LOGS_DIR / _RUNTIME_LOG_FILENAME

    def thread_store_path(self) -> Path:
        """Return the per-user thread store base path."""
        return self._thread_dir / self._user_slug if self._user_slug else self._thread_dir

    # ── Public read ────────────────────────────────────────────────────

    def bundle(
        self,
        *,
        turns: str | int = "last",
        max_records: int = DEFAULT_LOG_ERROR_LIMIT,
        focus: str = "all",
    ) -> ForensicBundle:
        """Read the evidence for a turn selection.

        Args:
            turns: ``"last"`` (the most recent turn), a single index (``"12"``),
                or an inclusive range (``"3-7"``).
            max_records: Cap on log lines/dispatch rows included.
            focus: ``all``, ``blocked``, ``receipts``, ``cost``, or ``latency``.
                Selects which sections the renderer emphasizes; every value
                still returns the same evidence (filtering evidence would let a
                caller miss the finding by asking the wrong question).
        """
        notes: list[str] = []
        sources = BundleSources()
        now = datetime.now(UTC)

        store = ThreadStore(self.thread_store_path(), redaction=self._redaction)
        start, end = self._resolve_range(store, turns, notes)
        sources.thread_store = start is not None

        entries: list[TurnEvidence] = []
        if start is not None and end is not None:
            try:
                for turn, entry in store.iter_range(start_turn=start, end_turn=end):
                    entries.append(turn_from_entry(turn, self._redact_entry(entry)))
            except (OSError, ValueError) as exc:
                notes.append(f"thread read failed for {start}-{end}: {exc}")
                sources.thread_store = False

        if len(entries) > self._max_turns:
            notes.append(
                f"{len(entries)} turns matched but the read cap is {self._max_turns}; "
                f"showing the most recent {self._max_turns}. Narrow the range."
            )
            entries = entries[-self._max_turns :]

        decisions, dispatches = self._read_decisions(sources, notes, max_records)
        costs = self._read_costs(sources, now, notes)
        log_errors = self._read_log_errors(sources, now, notes, max_records)

        bundle = build_bundle(
            generated_at=now.isoformat(),
            turns=entries,
            decisions=decisions,
            dispatches=dispatches,
            costs=costs,
            log_errors=log_errors,
            sources=sources,
            notes=notes,
        )
        if focus not in {"all", "blocked", "receipts", "cost", "latency"}:
            notes.append(f"unknown focus {focus!r}; rendered everything")
        return bundle

    # ── Internals ──────────────────────────────────────────────────────

    def _redact_entry(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Apply the read-time redaction policy to a raw thread entry.

        ``ThreadStore.iter_range`` returns RAW entries by design — the store's
        lossless read contract, and the same reason the ``lisa-thread`` MCP
        server applies the policy itself when projecting entries for a model.
        A forensic read is a projection into a prompt, so it must do the same:
        threads persist raw prompts and reasoning, and this is precisely the
        surface that would otherwise hand them back unredacted.
        """
        if self._redaction is None:
            return entry
        redacted = dict(entry)
        for field in ("query", "response_text", "reasoning_text"):
            value = redacted.get(field)
            if isinstance(value, str):
                redacted[field] = self._redaction.redact(value)
        return redacted

    def _resolve_range(
        self,
        store: ThreadStore,
        turns: str | int,
        notes: list[str],
    ) -> tuple[int | None, int | None]:
        """Resolve a turn selection into an inclusive range, or ``(None, None)``."""
        if isinstance(turns, int) and not isinstance(turns, bool):
            return (turns, turns) if turns >= 0 else (None, None)

        text = str(turns or "").strip().lower()
        try:
            last = store.next_turn_index() - 1
        except (OSError, ValueError) as exc:
            notes.append(f"could not determine the latest turn: {exc}")
            return None, None

        if text in {"", "last"}:
            if last < 0:
                notes.append("no turns recorded yet")
                return None, None
            return last, last
        if "-" in text:
            return self._parse_range(text, turns, last, notes)
        return self._parse_single(text, turns, notes)

    @staticmethod
    def _parse_range(
        text: str,
        original: str | int,
        last: int,
        notes: list[str],
    ) -> tuple[int | None, int | None]:
        """Parse an inclusive ``N-M`` selection."""
        low, _, high = text.partition("-")
        try:
            start, end = int(low), int(high)
        except ValueError:
            notes.append(f"unparseable turn range {original!r}; expected 'N' or 'N-M'")
            return None, None
        if start < 0 or end < start:
            notes.append(f"invalid turn range {original!r}")
            return None, None
        return start, min(end, last)

    @staticmethod
    def _parse_single(
        text: str,
        original: str | int,
        notes: list[str],
    ) -> tuple[int | None, int | None]:
        """Parse a single ``N`` selection."""
        try:
            index = int(text)
        except ValueError:
            notes.append(f"unparseable turn selection {original!r}; expected 'last', 'N', or 'N-M'")
            return None, None
        if index < 0:
            notes.append(f"invalid turn index {original!r}")
            return None, None
        return index, index

    def _read_decisions(
        self,
        sources: BundleSources,
        notes: list[str],
        max_records: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Split the ledger into decision and dispatch payloads."""
        records = list(iter_jsonl_tail(self.decisions_path, max_bytes=self._max_bytes))
        if not records:
            notes.append(
                f"{_DECISIONS_FILENAME} has no records; decisions and dispatches "
                "are unknown (not absent)"
            )
            return [], []
        sources.decisions_log = True
        decisions = [r for r in records if r.get("topic") == "turn.decision"]
        dispatches = [r for r in records if r.get("topic") == "tool.telemetry"]
        if len(dispatches) > max_records:
            notes.append(
                f"{len(dispatches)} dispatch records in the ledger; "
                f"using the most recent {max_records}"
            )
            dispatches = dispatches[-max_records:]
        return decisions, dispatches

    def _read_costs(
        self,
        sources: BundleSources,
        now: datetime,
        notes: list[str],
    ) -> list[dict[str, Any]]:
        """Return token-usage records inside the window."""
        records = list(iter_jsonl_tail(self.token_usage_path, max_bytes=self._max_bytes))
        if not records:
            notes.append(f"{_TOKEN_USAGE_FILENAME} has no records; per-turn cost is unknown")
            return []
        sources.token_usage_log = True
        return filter_window(records, now=now, window_seconds=self._window_seconds)

    def _read_log_errors(
        self,
        sources: BundleSources,
        now: datetime,
        notes: list[str],
        max_records: int,
    ) -> list[dict[str, Any]]:
        """Return warning/error log lines inside the window, newest last."""
        records = list(iter_jsonl_tail(self.runtime_log_path, max_bytes=self._max_bytes))
        if not records:
            notes.append(f"{_RUNTIME_LOG_FILENAME} has no records; log diagnostics are unknown")
            return []
        sources.runtime_log = True
        windowed = filter_window(records, now=now, window_seconds=self._window_seconds)
        errors = [
            record for record in windowed if str(record.get("level") or "").lower() in _ERROR_LEVELS
        ]
        if len(errors) > max_records:
            notes.append(
                f"{len(errors)} warning/error lines in the window; "
                f"using the most recent {max_records}"
            )
            errors = errors[-max_records:]
        return errors

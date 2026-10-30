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
so a caller cannot address another user's evidence by naming a path. The
telemetry ledgers are shared by every user of a root, so a user-scoped reader
also keeps only ledger records stamped with its ``user_slug``; turn indices are
per-thread and would otherwise join another user's dispatches and spend onto
this user's turns. The runtime log carries no user and is read as host-wide
process diagnostics.
"""

from __future__ import annotations

import itertools
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sdk.forensics._join import build_bundle, turn_from_entry
from sdk.forensics._logrec import (
    DEFAULT_MAX_BYTES,
    JsonlTail,
    filter_owned,
    filter_window,
    parse_ts,
    read_jsonl_tail,
    record_ts,
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

#: Clock slack applied to the turn-derived lower bound for ledger reads, so a
#: record stamped fractionally before the previous turn's entry is still read.
_TURN_FLOOR_SLACK = timedelta(minutes=5)

_ERROR_LEVELS = frozenset({"error", "warning", "critical"})
_DECISIONS_FILENAME = "decisions.jsonl"
_TOKEN_USAGE_FILENAME = "token-usage.jsonl"
_RUNTIME_LOG_FILENAME = "runtime.jsonl"


class ForensicReader:
    """Compose one forensic bundle from a host data root and thread store.

    Args:
        data_dir: The host's resolved data root. Log paths are derived from it.
        thread_dir: The directory holding per-user thread stores.
        user_slug: The user whose thread is read. Selects the store and scopes
            the shared decision and token ledgers to this user's records; it
            is never taken from a tool argument, so a caller cannot request
            another user's history. Empty means an unscoped whole-root read.
        thread_store_slug: The thread store's filename stem when it differs
            from ``user_slug``. A host that isolates one store per
            conversation names the store ``<user>--<conversation>``. The same
            stem scopes the shared ledgers, because turn indices are
            per-thread: two conversations of one user both count from zero, so
            the owner alone cannot tell their rows apart. Passing the stem
            here reads the store the host wrote and only the rows that join
            onto it.
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
        thread_store_slug: str = "",
        redaction: ThreadRedactionPolicy | None = None,
        resolve_redaction_from_env: bool = True,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        max_turns: int = DEFAULT_MAX_TURNS,
        max_bytes: int = DEFAULT_MAX_BYTES,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._thread_dir = Path(thread_dir) if thread_dir else self._data_dir / DATA_DIR / "threads"
        self._user_slug = user_slug
        # The thread store stem and the ledger owner are different things.
        # A host that isolates one thread store per conversation names the
        # store ``<user>--<conversation>`` while ledger rows stay stamped
        # with the plain user slug, so the two cannot share one parameter:
        # scoping the store by the user slug reads a store the host never
        # wrote, and scoping the ledger by the store stem drops every row.
        self._thread_store_slug = thread_store_slug
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
        """Return the thread store base path this reader reads.

        The stem is ``thread_store_slug`` when given, else ``user_slug``.
        A host that isolates one store per conversation passes the resolved
        store stem here and keeps ``user_slug`` for ledger scoping.
        """
        stem = self._thread_store_slug or self._user_slug
        return self._thread_dir / stem if stem else self._thread_dir

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
        floor: datetime | None = None
        if start is not None and end is not None:
            start = self._clamp_to_max_turns(start, end, notes)
            entries = self._read_turns(store, start, end, sources, notes)
            floor = self._turn_floor(store, start)
        wanted = {turn.telemetry_turn_index for turn in entries}

        decisions, dispatches = self._read_decisions(sources, notes, max_records, wanted, floor)
        costs = self._read_costs(sources, notes, wanted, floor)
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

    def _clamp_to_max_turns(self, start: int, end: int, notes: list[str]) -> int:
        """Narrow ``start`` so the range spans at most ``max_turns`` turns.

        Applied before the thread read, so a wide range costs only the turns
        that will be returned rather than decoding and redacting all of them.
        """
        span = end - start + 1
        if span <= self._max_turns:
            return start
        notes.append(
            f"{span} turns requested but the read cap is {self._max_turns}; "
            f"showing the most recent {self._max_turns}. Narrow the range."
        )
        return end - self._max_turns + 1

    def _read_turns(
        self,
        store: ThreadStore,
        start: int,
        end: int,
        sources: BundleSources,
        notes: list[str],
    ) -> list[TurnEvidence]:
        """Read and redact the thread entries for ``start..end``."""
        entries: list[TurnEvidence] = []
        try:
            for turn, entry in store.iter_range(start_turn=start, end_turn=end):
                entries.append(turn_from_entry(turn, self._redact_entry(entry)))
        except (OSError, ValueError) as exc:
            notes.append(f"thread read failed for {start}-{end}: {exc}")
            sources.thread_store = False
        return entries

    @staticmethod
    def _turn_floor(store: ThreadStore, start: int) -> datetime | None:
        """Return the earliest time a record for turn ``start`` can carry.

        Telemetry for turn ``start`` is written after turn ``start - 1``'s
        entry, so that entry's timestamp (less clock slack) bounds how far
        back the append-only ledgers need to be read. ``None`` when there is no
        earlier turn or it carries no usable timestamp: the read is then
        bounded by bytes alone.
        """
        if start <= 0:
            return None
        try:
            for _, entry in store.iter_range(start_turn=start - 1, end_turn=start - 1):
                stamp = parse_ts(entry.get("timestamp_utc"))
                return stamp - _TURN_FLOOR_SLACK if stamp is not None else None
        except (OSError, ValueError):
            return None
        return None

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
        wanted: set[int],
        floor: datetime | None,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Return the ledger's decision and dispatch payloads for the selection.

        Every record for a requested turn is kept. Records naming another turn
        (or none) are surfaced as unattributed, capped to the most recent
        ``max_records`` per kind so a long ledger cannot crowd out the answer.
        """
        tail = read_jsonl_tail(
            self.decisions_path, max_bytes=self._max_bytes, backup_count=None, since=floor
        )
        if not tail.records:
            notes.append(
                f"{_DECISIONS_FILENAME} has no records; decisions and dispatches "
                "are unknown (not absent)"
            )
            return [], []
        sources.decisions_log = True
        records = self._owned(_since(tail, floor), _DECISIONS_FILENAME, notes)
        self._note_truncation(tail, _DECISIONS_FILENAME, "the selected turns", notes)
        self._note_sequence_gaps(records, _DECISIONS_FILENAME, notes)
        decisions = _cap_unattributed(
            [r for r in records if r.get("topic") == "turn.decision"],
            wanted,
            max_records,
            "decision",
            notes,
        )
        dispatches = _cap_unattributed(
            [r for r in records if r.get("topic") == "tool.telemetry"],
            wanted,
            max_records,
            "dispatch",
            notes,
        )
        return decisions, dispatches

    def _read_costs(
        self,
        sources: BundleSources,
        notes: list[str],
        wanted: set[int],
        floor: datetime | None,
    ) -> list[dict[str, Any]]:
        """Return the token-usage records for the requested turns."""
        tail = read_jsonl_tail(self.token_usage_path, max_bytes=self._max_bytes, since=floor)
        if not tail.records:
            notes.append(f"{_TOKEN_USAGE_FILENAME} has no records; per-turn cost is unknown")
            return []
        sources.token_usage_log = True
        self._note_truncation(tail, _TOKEN_USAGE_FILENAME, "the selected turns", notes)
        owned = self._owned(_since(tail, floor), _TOKEN_USAGE_FILENAME, notes)
        self._note_sequence_gaps(owned, _TOKEN_USAGE_FILENAME, notes)
        return [r for r in owned if r.get("turn_index") in wanted]

    def _owned(
        self, records: list[dict[str, Any]], name: str, notes: list[str]
    ) -> list[dict[str, Any]]:
        """Keep this user's ledger records, noting any that predate attribution."""
        owned = filter_owned(records, self._user_slug, self._thread_store_slug)
        if owned.unowned:
            notes.append(
                f"{owned.unowned} {name} records predate per-user attribution and were "
                "excluded; evidence for turns before the upgrade is incomplete"
            )
        if owned.unstemmed:
            notes.append(
                f"{owned.unstemmed} {name} records predate per-thread attribution and were "
                "excluded; evidence for turns before the upgrade is incomplete"
            )
        return owned.records

    def _read_log_errors(
        self,
        sources: BundleSources,
        now: datetime,
        notes: list[str],
        max_records: int,
    ) -> list[dict[str, Any]]:
        """Return warning/error log lines inside the window, newest last."""
        cutoff = now - timedelta(seconds=self._window_seconds)
        tail = read_jsonl_tail(self.runtime_log_path, max_bytes=self._max_bytes, since=cutoff)
        if not tail.records:
            notes.append(f"{_RUNTIME_LOG_FILENAME} has no records; log diagnostics are unknown")
            return []
        sources.runtime_log = True
        self._note_truncation(tail, _RUNTIME_LOG_FILENAME, "the log window", notes)
        windowed = filter_window(tail.records, now=now, window_seconds=self._window_seconds)
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

    def _note_truncation(self, tail: JsonlTail, name: str, span: str, notes: list[str]) -> None:
        """Record that a read hit its byte budget before covering ``span``."""
        if tail.truncated:
            notes.append(
                f"{name} read stopped at its {self._max_bytes // (1024 * 1024)} MiB budget "
                f"before covering {span}; older records are missing, so counts from it "
                "are a lower bound"
            )

    def _note_sequence_gaps(
        self, records: list[dict[str, Any]], name: str, notes: list[str]
    ) -> None:
        """Report dropped rows detectable in one writer's sequence numbering.

        Both durable ledgers stamp each row with a per-instance ``sequence``
        and the writer's ``ledger_instance`` (a restart or a second host
        process starts a new instance at 1, so sequences are compared only
        within one instance). A jump between consecutive rows of one
        instance means rows in between were dropped — by a failed publish, a
        lock timeout, or a crash between the stamp and the append. The count
        is a lower bound: it proves rows went missing, not which rows.

        Rows predating the instance stamps carry neither field; they are
        skipped rather than reported, the same honest posture as ledger rows
        written before per-user attribution.
        """
        missing_total = 0
        by_instance: dict[str, list[int]] = {}
        for record in records:
            instance = record.get("ledger_instance")
            sequence = record.get("sequence")
            if not instance or not isinstance(sequence, int):
                continue
            by_instance.setdefault(str(instance), []).append(sequence)
        for sequences in by_instance.values():
            ordered = sorted(sequences)
            # Only gaps within the observed span count: rows before the
            # first or after the last observed sequence of an instance may
            # simply live outside this read window.
            missing_total += sum(
                nxt - prev - 1 for prev, nxt in itertools.pairwise(ordered) if nxt - prev > 1
            )
        if missing_total:
            notes.append(
                f"{name} sequence numbering accounts for {missing_total} dropped row(s) "
                "across its stamped writers; the ledger is incomplete for that span"
            )


def _since(tail: JsonlTail, floor: datetime | None) -> list[dict[str, Any]]:
    """Drop records dated before ``floor``; undated records are kept."""
    if floor is None:
        return tail.records
    return [r for r in tail.records if (ts := record_ts(r)) is None or ts >= floor]


def _cap_unattributed(
    records: list[dict[str, Any]],
    wanted: set[int],
    limit: int,
    kind: str,
    notes: list[str],
) -> list[dict[str, Any]]:
    """Keep every record for a wanted turn and the newest ``limit`` of the rest."""
    others = [r for r in records if r.get("turn_index") not in wanted]
    if len(others) <= limit:
        return records
    keep = {id(r) for r in others[-limit:]}
    notes.append(
        f"{len(others)} {kind} records named no requested turn; listing the most recent {limit}"
    )
    return [r for r in records if r.get("turn_index") in wanted or id(r) in keep]

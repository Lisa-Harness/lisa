"""Stateless baseline replay over a token-usage ledger (SPEC \u00a77 criterion 1).

Replays a whole ledger \u2014 or any window of it \u2014 through the same
capture, line-classification, and partition-fold readers that back the
live ``/tokens`` cache section, so a baseline can be reproduced against
the aggregates the readers actually compute. The replay is read-only
and checkpoint-free: it never writes state and never touches the
measured layer.

The 2026-11-12 baseline report itself stays outside the repository
(user-specific aggregates); this module is the in-repo half of the
replay: given a ledger path, it reproduces the aggregation and exposes
coverage, exclusion, and diagnostic counts beside every figure so
incomplete historical rows are never silently compared against complete
post-change rows.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from kernel.cache_metrics import (
    MAX_WILSON_HALF_WIDTH,
    MIN_ELIGIBLE_SAMPLES,
)

from runtime.cache_report import (
    _MAX_INCREMENTAL_BYTES,
    _capture_stable_view,
    _fold_contributions,
    _ingest_lines,
    _Ingestion,
    _iso,
    _key_label,
    _parse_utc,
    _utc_now,
)

__all__ = [
    "ReplayOptions",
    "main",
    "render_replay_report",
    "replay_baseline",
    "replay_json",
]


@dataclass(frozen=True, slots=True)
class ReplayOptions:
    """Bounds for one baseline replay.

    ``as_of`` fixes the clock: rows stamped after it (beyond a small
    skew allowance) are excluded exactly as the live reader excludes
    future rows. ``window_days`` limits the observation window; ``None``
    replays the entire ledger, which is what distinguishes a full-scan
    baseline from the rolling 14-day live view.
    """

    as_of: datetime | None = None
    window_days: int | None = None
    max_bytes: int = _MAX_INCREMENTAL_BYTES


def replay_baseline(ledger_path: Path, *, options: ReplayOptions | None = None) -> _Ingestion:
    """Replay ``ledger_path`` through the live aggregation readers.

    Returns the same result shape as the incremental reader (partitions,
    keys, diagnostics, exclusion coverage, window bounds), computed over
    the requested window with no checkpoint state. Every failure mode
    degrades to an explicit ``unavailable``/``incomplete`` status \u2014 the
    replay never invents figures, and it never writes anything.
    """
    options = options or ReplayOptions()
    now = options.as_of or _utc_now()
    captured = _capture_stable_view(ledger_path)
    if isinstance(captured, _Ingestion):
        return captured
    data, _identity, _size = captured

    from runtime.cache_report import _Checkpoint

    state = _Checkpoint(last_read_utc=_iso(now))
    _offset, source_complete, earliest, latest, per_call = _ingest_lines(state, data, 0, now)

    if options.window_days is not None:
        cutoff = now - timedelta(days=options.window_days)
        kept = [
            item for item in state.contributions if (_parse_utc(item.get("t")) or now) >= cutoff
        ]
        state.contributions = kept

    partitions, keys, win_start, win_end = _fold_contributions(state)
    return _Ingestion(
        status="ok",
        source_complete=source_complete,
        partitions=partitions,
        keys=keys,
        diagnostics=per_call,
        excluded=dict(state.excluded),
        window_start=win_start or earliest,
        window_end=win_end or latest,
    )


def _partition_rows(ingestion: _Ingestion) -> list[str]:
    """Render the per-partition metric table shared by both reports."""
    lines: list[str] = []
    for tuple_key in sorted(ingestion.partitions, key=lambda k: _key_label(ingestion.keys[k])):
        key = ingestion.keys[tuple_key]
        metrics = ingestion.partitions[tuple_key]
        excluded = (
            ", ".join(
                f"{reason}:{count}" for reason, count in sorted(metrics.excluded_counts.items())
            )
            or "\u2014"
        )
        ambiguous = str(metrics.ambiguous_rows) if metrics.ambiguous_rows else "\u2014"
        hit = f"{metrics.request_hit_rate:.1%}" if metrics.request_hit_rate is not None else "n/a"
        share = f"{metrics.token_share:.1%}" if metrics.token_share is not None else "n/a"
        lines.append(
            f"| {_key_label(key)} | {metrics.hits}/{metrics.eligible_count} ({hit}) "
            f"| {metrics.cached_sum:,}/{metrics.prompt_sum:,} ({share}) "
            f"| {ambiguous} | {excluded} |"
        )
    return lines


def render_replay_report(replay: _Ingestion, *, title: str = "Baseline replay") -> list[str]:
    """Render the human replay report. Aggregate counts only, never rows."""
    lines = [f"### {title}"]
    if replay.status != "ok":
        lines.append(f"- Status: unavailable ({replay.reason}); no figures invented")
        return lines
    status = "complete" if replay.source_complete else "incomplete"
    start_label = _iso(replay.window_start) if replay.window_start else "n/a"
    end_label = _iso(replay.window_end) if replay.window_end else "n/a"
    lines.append(f"- Window: {start_label} \u2192 {end_label} ({status} source)")
    if replay.diagnostics:
        shown = ", ".join(
            f"{name}={count}" for name, count in sorted(replay.diagnostics.items()) if count
        )
        if shown:
            lines.append(f"- Diagnostics: {shown}")
    for label, buckets in sorted(replay.excluded.items()):
        shown = ", ".join(f"{reason}:{count}" for reason, count in sorted(buckets.items()))
        if shown:
            lines.append(f"- Excluded for {label}: {shown}")
    if not replay.partitions:
        lines.append("- No eligible cache observations in the replayed window")
        return lines
    lines.append("")
    lines.append("| Partition | Hit proxy | Token share | Ambiguous | Excluded |")
    lines.append("| --- | --- | --- | --- | --- |")
    lines.extend(_partition_rows(replay))
    lines.append("")
    lines.append("- Aggregate counts only; no prompt text or raw rows are emitted")
    return lines


def replay_json(replay: _Ingestion) -> dict[str, Any]:
    """Machine-readable replay payload for aggregate reproduction checks."""
    partitions: dict[str, Any] = {}
    for tuple_key, metrics in replay.partitions.items():
        label = _key_label(replay.keys[tuple_key])
        partitions[label] = {
            "eligible_count": metrics.eligible_count,
            "hits": metrics.hits,
            "observed_zero_count": metrics.observed_zero_count,
            "prompt_sum": metrics.prompt_sum,
            "cached_sum": metrics.cached_sum,
            "excluded_counts": dict(sorted(metrics.excluded_counts.items())),
            "ambiguous_rows": metrics.ambiguous_rows,
            "qualified": (
                metrics.eligible_count >= MIN_ELIGIBLE_SAMPLES
                and (metrics.wilson_half_width or 1.0) <= MAX_WILSON_HALF_WIDTH
                and replay.keys[tuple_key].identity_known
            ),
        }
    return {
        "status": replay.status,
        "reason": replay.reason,
        "source_complete": replay.source_complete,
        "window_start_utc": _iso(replay.window_start) if replay.window_start else None,
        "window_end_utc": _iso(replay.window_end) if replay.window_end else None,
        "diagnostics": dict(sorted(replay.diagnostics.items())),
        "excluded": {
            label: dict(sorted(b.items())) for label, b in sorted(replay.excluded.items())
        },
        "partitions": partitions,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI: replay one ledger and print the report (JSON with ``--json``)."""
    parser = argparse.ArgumentParser(
        prog="cache-replay",
        description="Replay a token-usage ledger through the cache aggregation readers.",
    )
    parser.add_argument("ledger", type=Path, help="path to token-usage.jsonl")
    parser.add_argument("--as-of", type=str, default=None, help="ISO-8601 as-of clock (UTC)")
    parser.add_argument("--window-days", type=int, default=None, help="limit to an N-day window")
    parser.add_argument("--json", action="store_true", help="emit the machine-readable payload")
    args = parser.parse_args(argv)

    as_of = _parse_utc(args.as_of) if args.as_of else None
    if args.as_of and as_of is None:
        print(f"invalid --as-of value: {args.as_of}", file=sys.stderr)
        return 2
    options = ReplayOptions(as_of=as_of, window_days=args.window_days)
    replay = replay_baseline(args.ledger, options=options)
    if args.json:
        print(json.dumps(replay_json(replay), indent=1))
    else:
        print("\n".join(render_replay_report(replay)))
    return 0 if replay.status == "ok" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

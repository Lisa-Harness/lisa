"""Analyze token-usage.jsonl and quantify the cached / reasoning / live split.

This is a diagnostic tool (non-destructive). It reads the durable token
ledger written by ``TokenUsageTelemetryHandler`` and answers the question:

    "Why does Lisa's token total look way bigger than the provider meter?"

The key insight is accounting *semantics*, not a units bug:

* Lisa sums the provider-reported ``total_tokens`` for every call.
* For OpenAI reasoning models, ``prompt_tokens`` (and thus ``total_tokens``)
  *includes* cached input tokens, billed at full weight on every call even
  though the platform bills cache hits at a much lower rate.
* ``reasoning_tokens`` are also folded into ``total_tokens`` while provider
  dashboards often report them separately.

Usage::

    python scripts/analyze_token_usage.py [path/to/token-usage.jsonl]

Prints a per-provider / per-model breakdown of total vs. billed
(``total - cached``) vs. live (``total - cached - reasoning``) tokens.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

#: Path relative to the repo root when the script is run from anywhere.
_DEFAULT_RELATIVE = "artifacts/token-usage.jsonl"

#: How many of the most recent records to analyze (mirrors introspect tail).
_TAIL_LIMIT = 2000


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_path(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    return _repo_root() / _DEFAULT_RELATIVE


def _load_records(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        print(f"error: ledger not found at {path}", file=sys.stderr)
        sys.exit(1)
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            try:
                decoded = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(decoded, dict):
                records.append(decoded)
    return records[-_TAIL_LIMIT:]


def _as_int(value: object) -> int:
    # Values originate from JSON, so they are int/float/str (or None).
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(0, value)
    if isinstance(value, float):
        return max(0, int(value))
    if isinstance(value, str):
        try:
            return max(0, int(value))
        except ValueError:
            return 0
    return 0


def _summarize(records: list[dict[str, object]]) -> None:
    # Per (provider, model) accumulators.
    totals: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {
            "calls": 0,
            "prompt": 0,
            "completion": 0,
            "total": 0,
            "cached": 0,
            "reasoning": 0,
        }
    )

    for record in records:
        provider = str(record.get("provider") or "-")
        model = str(record.get("model_id") or "-")
        prompt = _as_int(record.get("prompt_tokens"))
        completion = _as_int(record.get("completion_tokens"))
        total = _as_int(record.get("total_tokens"))
        cached = _as_int(record.get("cached_tokens"))
        reasoning = _as_int(record.get("reasoning_tokens"))
        if total <= 0:
            total = prompt + completion

        bucket = totals[(provider, model)]
        bucket["calls"] += 1
        bucket["prompt"] += prompt
        bucket["completion"] += completion
        bucket["total"] += total
        bucket["cached"] += cached
        bucket["reasoning"] += reasoning

    if not totals:
        print("No token-usage records in the analyzed window.")
        return

    print(f"Analyzed {len(records)} events (tail of {len(records)}).")
    print()
    print("Legend:  total = provider-reported total_tokens")
    print("         billed = total - cached   (cache hits removed, full-weight)")
    print("         live = total - cached - reasoning")
    print()

    grand_total = sum(b["total"] for b in totals.values())
    grand_billed = sum(b["total"] - b["cached"] for b in totals.values())
    grand_live = sum(b["total"] - b["cached"] - b["reasoning"] for b in totals.values())
    grand_cached = sum(b["cached"] for b in totals.values())
    grand_reasoning = sum(b["reasoning"] for b in totals.values())

    header = (
        f"{'provider':<12}{'model':<28}{'calls':>6}{'total':>10}"
        f"{'cached':>10}{'reasoning':>10}{'billed':>10}{'live':>10}"
    )
    print(header)
    print("-" * len(header))
    for (provider, model), b in sorted(totals.items()):
        billed = int(b["total"] - b["cached"])
        live = int(b["total"] - b["cached"] - b["reasoning"])
        print(
            f"{provider:<12}{model:<28}{int(b['calls']):>6}"
            f"{int(b['total']):>10}{int(b['cached']):>10}{int(b['reasoning']):>10}"
            f"{billed:>10}{live:>10}"
        )
    print("-" * len(header))
    print(
        f"{'TOTAL':<12}{'':<28}{'':>6}{grand_total:>10}"
        f"{grand_cached:>10}{grand_reasoning:>10}{grand_billed:>10}{grand_live:>10}"
    )
    print()

    if grand_total == 0:
        print("No nonzero token totals in the window.")
        return

    print("Share of provider-reported total_tokens:")
    if grand_total:
        print(f"  cached:     {grand_cached / grand_total:>6.1%}")
        print(f"  reasoning:  {grand_reasoning / grand_total:>6.1%}")
        print(f"  billed(no cache): {grand_billed / grand_total:>6.1%} of reported total")
        print(f"  live(no cache, no reasoning): {grand_live / grand_total:>6.1%} of reported total")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default=None, help="path to token-usage.jsonl")
    args = parser.parse_args()

    path = _resolve_path(args.path)
    records = _load_records(path)
    _summarize(records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Classify prompt-cache misses in Lisa's token-usage telemetry.

Reads ``token-usage.jsonl`` (default ``~/.lisa/logs/token-usage.jsonl``) and
labels each ``turn_loop`` call whose cached share falls short of the previous
call's prompt:

- ``first``: first call of a turn (gap from the previous call over ``--gap``).
- ``mode``: the mode changed from the previous call.
- ``trim``: the prompt shrank, so an in-turn reduction rewrote the prefix.
- ``node``: nothing cached, or the hit equals an older prompt in the turn. On
  breakpoint-cached models (Azure GPT-5.6+) this usually means no breakpoint
  ended the reusable prefix; otherwise the call reached a backend that held an
  older copy (no cache affinity).
- ``unexplained``: none of the above.

Usage: ``python scripts/cache_report.py [--since 2026-10-11T17:00] [--log PATH]``
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

_DEFAULT_LOG = Path.home() / ".lisa" / "logs" / "token-usage.jsonl"
# A hit within this many tokens of an earlier prompt counts as reusing it.
_MATCH_SLACK = 16
# Earlier calls of the turn kept for older-copy matching.
_HISTORY = 8


def _classify(prev: list[dict], row: dict, gap_seconds: float) -> str | None:
    last = prev[-1] if prev else None
    if (
        last is None
        or (
            datetime.fromisoformat(row["emitted_at"]) - datetime.fromisoformat(last["emitted_at"])
        ).total_seconds()
        > gap_seconds
    ):
        return "first"
    label: str | None = None
    if row["cached_tokens"] + _MATCH_SLACK < last["prompt_tokens"]:
        if row.get("mode_id") != last.get("mode_id"):
            label = "mode"
        elif row["prompt_tokens"] < last["prompt_tokens"]:
            label = "trim"
        elif row["cached_tokens"] == 0 or any(
            abs(row["cached_tokens"] - earlier["prompt_tokens"]) <= _MATCH_SLACK
            for earlier in prev[:-1]
        ):
            label = "node"
        else:
            label = "unexplained"
    return label


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--log", type=Path, default=_DEFAULT_LOG)
    parser.add_argument("--since", default="", help="ISO timestamp prefix lower bound (UTC)")
    parser.add_argument("--gap", type=float, default=120.0, help="seconds that start a new turn")
    parser.add_argument("-v", "--verbose", action="store_true", help="print every call")
    args = parser.parse_args()

    labels: Counter[str] = Counter()
    uncached: Counter[str] = Counter()
    prompt_total = cached_total = written_total = 0
    history: list[dict] = []
    with args.log.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("call_site") != "turn_loop" or not row.get("prompt_tokens"):
                continue
            if row["emitted_at"] < args.since:
                continue
            label = _classify(history, row, args.gap) or "hit"
            labels[label] += 1
            uncached[label] += row["prompt_tokens"] - row["cached_tokens"]
            prompt_total += row["prompt_tokens"]
            cached_total += row["cached_tokens"]
            written_total += row.get("cache_creation_tokens") or 0
            if args.verbose:
                print(
                    f"{row['emitted_at'][11:19]} {row.get('mode_id', ''):16} "
                    f"p={row['prompt_tokens']:>7} c={row['cached_tokens']:>7} "
                    f"w={row.get('cache_creation_tokens') or 0:>7} {label}"
                )
            history = [*history[-_HISTORY + 1 :], row]

    share = cached_total / prompt_total if prompt_total else 0.0
    print(
        f"calls={sum(labels.values())} prompt={prompt_total} "
        f"cached={cached_total} written={written_total} share={share:.0%}"
    )
    for label, count in labels.most_common():
        print(f"  {label:12} calls={count:>4} uncached_tokens={uncached[label]:>9}")


if __name__ == "__main__":
    main()

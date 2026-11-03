"""Token/cost roll-up over one run's durable ``token-usage.jsonl`` log."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

__all__ = ["usage_summary"]


def usage_summary(path: Path) -> dict[str, Any]:
    """Aggregate ``TokenUsageTelemetryHandler`` records into one summary.

    The runtime appends one event per model call (turn loop, forced-final,
    terminal-stage completions) with prompt/completion/cached/reasoning
    counters, ``cost_usd``, and the model/mode attribution. The summary reads
    that durable evidence — never the model's prose or client-side guesses —
    so a repeat-mode pass-rate can be paired with what the run actually cost.
    Zero-token rows (auxiliary/bookkeeping calls) are counted in ``calls``
    but contribute nothing to token totals.
    """
    calls = 0
    billed_calls = 0
    prompt = completion = cached = reasoning = 0
    total = 0
    cost = 0.0
    by_model: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            # Torn final line after a kill mid-append, or a hand edit: the
            # runtime's own reader tolerates unparsable lines, so a
            # completed run must not flip to a setup error over one bad row.
            continue
        calls += 1
        p = int(record.get("prompt_tokens") or 0)
        c = int(record.get("completion_tokens") or 0)
        if p or c:
            billed_calls += 1
        prompt += p
        completion += c
        cached += int(record.get("cached_tokens") or 0)
        reasoning += int(record.get("reasoning_tokens") or 0)
        total += int(record.get("total_tokens") or 0)
        cost += float(record.get("cost_usd") or 0.0)
        model_id = str(record.get("model_id") or "")
        if not model_id:
            continue
        entry = by_model.setdefault(
            model_id, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        )
        entry["calls"] += 1
        entry["prompt_tokens"] += p
        entry["completion_tokens"] += c
        entry["total_tokens"] += int(record.get("total_tokens") or 0)
    return {
        "calls": calls,
        "billed_calls": billed_calls,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "cached_tokens": cached,
        "reasoning_tokens": reasoning,
        "total_tokens": total,
        "cost_usd": round(cost, 6),
        "by_model": by_model,
    }

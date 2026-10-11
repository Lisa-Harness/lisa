"""Durable turn checkpoints, request archives, and incremental accounting."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from kernel.turn_continuation import TurnContinuation
from kernel.turn_runner import TurnResult
from sdk.thread._results import load_result_object

from runtime.thread._results import _persist_result


def persist_continuation(thread_path: Path | None, saved: TurnContinuation) -> str:
    if thread_path is None:
        raise ValueError("Turn continuation requires a thread storage path.")
    return _persist_result(thread_path, {"content": saved.model_dump_json()})


def load_continuation(
    thread_path: Path | None, provenance: dict[str, Any]
) -> TurnContinuation | None:
    reference = str(provenance.get("turn_continuation_ref") or "")
    if not reference:
        return None
    if thread_path is None:
        raise ValueError("Turn continuation requires a thread storage path.")
    payload = load_result_object(thread_path, reference)["content"]
    if not isinstance(payload, str):
        raise ValueError("The saved turn continuation is missing or invalid.")
    return TurnContinuation.model_validate_json(payload)


def persist_request(thread_path: Path | None, request: dict[str, Any]) -> str:
    if thread_path is None:
        raise ValueError("Model request recording requires a thread storage path.")
    return _persist_result(thread_path, {"content": json.dumps(request, ensure_ascii=False)})


def usage_since_resume(result: TurnResult, saved: TurnContinuation | None) -> TurnResult:
    """Account only new work while the kernel keeps cumulative turn budgets."""
    if saved is None:
        return result
    state = saved.restore_state()
    counters = (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cached_tokens",
        "reasoning_tokens",
        "cache_creation_tokens",
        "duration_ms",
        "latency_ms",
    )

    def breakdown(
        rows: list[dict[str, Any]],
        previous: dict[str, dict[str, Any]],
        identity_key: str,
    ) -> list[dict[str, Any]]:
        return [
            {
                **row,
                **{
                    key: row.get(key, 0) - previous.get(str(row[identity_key]), {}).get(key, 0)
                    for key in counters
                },
            }
            for row in rows
        ]

    return replace(
        result,
        prompt_tokens=result.prompt_tokens - state.total_prompt_tokens,
        completion_tokens=result.completion_tokens - state.total_completion_tokens,
        total_tokens=result.total_tokens - state.total_tokens,
        cached_tokens=result.cached_tokens - state.total_cached_tokens,
        reasoning_tokens=result.reasoning_tokens - state.total_reasoning_tokens,
        cache_creation_tokens=result.cache_creation_tokens - state.total_cache_creation_tokens,
        duration_ms=result.duration_ms - state.total_duration_ms,
        latency_ms=result.latency_ms - state.total_latency_ms,
        per_model_usage=breakdown(result.per_model_usage, state.per_model_usage, "model_id"),
        per_mode_usage=breakdown(result.per_mode_usage, state.per_mode_usage, "mode_id"),
    )

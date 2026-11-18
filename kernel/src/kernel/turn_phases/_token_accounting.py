"""Shared processed-token accounting for the turn envelope and budget rail.

Both the compaction governor's effective-window cap and the turn budget's
``budget_forces_final`` rail must agree on what a turn has "spent": cached
prompt prefixes are re-read rather than processed again, so the envelope
counts uncached input plus output. Keeping the formula in one cycle-free
module means the compaction cap and the budget rail can never drift apart.

This module must not import ``kernel.turn_phases._turn_state`` (which imports
compaction classes) or any other turn-phase module: it is the shared leaf.
"""

from __future__ import annotations


def processed_tokens(total_tokens: int, total_cached_tokens: int) -> int:
    """Return tokens newly processed by providers this turn.

    Cached prompt prefixes are re-read rather than processed again, so the
    whole-turn token envelope counts uncached input plus output. Counting the
    re-sent prefix would make useful context consumption, not work, the cap.
    """
    return max(0, int(total_tokens) - int(total_cached_tokens))


def remaining_envelope(envelope: int, processed: int) -> int | None:
    """Return the unspent portion of a finite turn token envelope.

    ``None`` means no finite envelope (envelope <= 0); zero means a finite
    envelope is exhausted. The distinction matters: ``None`` must leave the
    raw model window untouched, while zero must collapse the effective
    window so the existing finish path ends the turn.
    """
    if envelope <= 0:
        return None
    return max(0, int(envelope) - int(processed))


def cache_hit_rate(total_cached_tokens: int, total_prompt_tokens: int) -> float | None:
    """Return the provider-reported cache hit rate, or ``None`` when unknown.

    Diagnostic only: this value never feeds a budget decision. A missing or
    zero prompt-token count means the rate is unknown, not zero — returning
    ``None`` keeps the diagnostic honest instead of reporting a fabricated 0.0.
    """
    if total_prompt_tokens <= 0:
        return None
    return min(1.0, max(0.0, total_cached_tokens / total_prompt_tokens))

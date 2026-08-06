"""Deterministic auto-capture trigger detection for post-turn learning.

Extracted from ``host.turn_hooks_builder.build_turn_hooks`` to make the
trigger policy independently testable.  The function and its constants are
self-contained — they do not capture any runtime state.

Public surface
--------------
- :func:`should_auto_capture`
- :data:`COMMITMENT_PATTERNS`
- :data:`WRITE_EFFECT_TOOL_PREFIXES`
- :data:`WRITE_EFFECT_TOOL_SUFFIXES`
"""

from __future__ import annotations

import re

# ── Trigger constants ────────────────────────────────────────────────────

COMMITMENT_PATTERNS: list[tuple[str, float]] = [
    (r"\bremember\s+that\b", 0.7),
    (r"\bfrom\s+now\s+on\b", 0.7),
    (r"\blet'?s\s+go\s+with\b", 0.7),
    (r"\b(?:I\s+)?decided\b", 0.7),
    (r"\bI\s+prefer\b", 0.6),
    (r"\b(?:my|our)\s+default\b", 0.6),
    (r"\balways\b.{0,30}\b(?:use|do|run|call|check|set)\b", 0.6),
    (r"\bnever\b.{0,30}\b(?:use|do|run|call|check|set)\b", 0.6),
]

WRITE_EFFECT_TOOL_PREFIXES: tuple[str, ...] = (
    "coding_",
    "memory.add",
    "memory.dispute",
    "file_write",
    "storage_",
    "checkpoint_",
)

WRITE_EFFECT_TOOL_SUFFIXES: tuple[str, ...] = ("_create", "_update", "_delete", "_write", "_save")

# ── Salience → tier mapping ──────────────────────────────────────────────
#
# Auto-captured learnings are workflow state by default: a write-effect tool
# call or a soft commitment ("I prefer …") is task-specific context that
# rarely survives a week, let alone a year. Persisting it as ``long_term``
# (365-day TTL) pollutes long-term recall with ephemeral noise. Only the
# strongest signals — explicit approval granted (salience 0.8) or firm
# commitment language ("from now on", "decided") — earn a long-term slot.
# The thresholds are overridable via config (see ``classify_memory_tier``).
DEFAULT_TIER_THRESHOLD_DAILY: float = 0.65
DEFAULT_TIER_THRESHOLD_WEEKLY: float = 0.8

# Valid retention tiers for auto-captured memories. ``permanent`` is
# intentionally excluded — that tier is reserved for explicit, curated
# records (e.g. dispute resolutions), not opportunistic capture.
AUTO_CAPTURE_TIERS: tuple[str, ...] = ("daily", "weekly", "long_term")


# ── Trigger detection ────────────────────────────────────────────────────


def should_auto_capture(  # noqa: C901,PLR0912
    query: str,
    response_text: str,  # required by the calling signature
    messages: list[dict[str, object]],
) -> tuple[bool, float]:
    """Return ``(should_capture, salience)`` for deterministic trigger detection.

    Checks three trigger sources:
    1. Commitment language in the user query (regex patterns).
    2. Write-effect tool calls in assistant messages.
    3. Approval content in tool-role messages.

    Returns ``(False, 0.0)`` when no trigger fires or when the model already
    called ``synthesize_and_store_learning`` or ``dispute_memory`` during this
    turn (the learning was already persisted).
    """
    # Skip if synthesize_and_store_learning or dispute_memory was already
    # called during this turn — the model already persisted its learning.
    for msg in messages:
        role = str(msg.get("role") or "").strip().lower()
        if role != "assistant":
            continue
        tool_calls = msg.get("tool_calls") or []
        for tc in tool_calls:
            fn_name = str((tc.get("function") or {}).get("name", "")).strip()
            if fn_name in ("synthesize_and_store_learning", "dispute_memory"):
                return False, 0.0

    max_salience = 0.0

    # 1. Commitment language in the user query
    query_lower = query.lower()
    for pattern, salience in COMMITMENT_PATTERNS:
        if re.search(pattern, query_lower):
            max_salience = max(max_salience, salience)

    # 2. Write-effect tool calls
    for msg in messages:
        role = str(msg.get("role") or "").strip().lower()
        if role != "assistant":
            continue
        tool_calls = msg.get("tool_calls") or []
        for tc in tool_calls:
            fn_name = str((tc.get("function") or {}).get("name", "")).strip()
            if not fn_name:
                continue
            if fn_name.startswith(WRITE_EFFECT_TOOL_PREFIXES) or fn_name.endswith(
                WRITE_EFFECT_TOOL_SUFFIXES
            ):
                max_salience = max(max_salience, 0.6)

    # 3. Approval granted
    for msg in messages:
        role = str(msg.get("role") or "").strip().lower()
        if role != "tool":
            continue
        content = str(msg.get("content") or "")
        if "approved" in content.lower():
            max_salience = max(max_salience, 0.8)

    if max_salience <= 0.0:
        return False, 0.0
    return True, max_salience


def classify_memory_tier(
    salience: float,
    daily_threshold: float = DEFAULT_TIER_THRESHOLD_DAILY,
    weekly_threshold: float = DEFAULT_TIER_THRESHOLD_WEEKLY,
) -> str:
    """Map an auto-capture salience score onto a retention tier.

    Auto-captured learnings are workflow state by default; only the
    strongest signals earn a long-term slot. The mapping is:

    - ``salience < daily_threshold``  → ``daily``   (7-day TTL)
    - ``daily_threshold ≤ salience < weekly_threshold`` → ``weekly`` (30-day TTL)
    - ``salience ≥ weekly_threshold`` → ``long_term`` (365-day TTL)

    Thresholds default to 0.65 / 0.8 so that soft commitments and
    write-effect tool calls (salience 0.6–0.7) land in ``daily``/``weekly``,
    while explicit approvals (0.8) and firm commitments ("from now on",
    0.7) still reach ``long_term`` when they clear the weekly band. Callers
    override the thresholds via ``HostConfig.memory_auto_capture_*`` knobs
    rather than hard-coding new cut points here.
    """
    # Clamp negatives so a malformed salience never silently promotes a
    # record past the daily band.
    score = max(0.0, float(salience))
    if score < daily_threshold:
        return "daily"
    if score < weekly_threshold:
        return "weekly"
    return "long_term"

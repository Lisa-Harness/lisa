"""Shared helpers for model discovery probes.

Environment resolution and model-classification heuristics live here so
every probe reads credentials through the same path and applies the same
reasoning/low-latency tagging rules.
"""

from __future__ import annotations

import os

# Reasoning-model markers. Substring match against the lower-cased
# model id. Kept deliberately narrow to avoid false positives like
# matching "r1" inside "far1lab-..." — markers are now anchored to
# known product prefixes or explicit tokens.
_REASONING_MARKERS = (
    "reason",
    "thinking",
    "deepseek-r1",
    "-o1",
    "-o3",
    "-o4",
    "o1-",
    "o3-",
    "o4-",
    "o1:",
    "o3:",
    "o4:",
    "claude-opus-4",
    "gemini-2.5-pro",
)

# Low-latency / "small" model markers.
_LOW_LATENCY_MARKERS = (
    "mini",
    "nano",
    "flash",
    "small",
    "3b",
    "1b",
    "haiku",
)


def env_any(*names: str) -> str:
    """Return the first non-empty environment variable among ``names``."""

    for name in names:
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def is_reasoning_model(model_id: str) -> bool:
    """Heuristic: does ``model_id`` look like a reasoning-capable model?

    Anchored markers (``-o1``, ``o1-``, ``o1:``) avoid the previous
    false-positive where the bare substring ``"o1"`` matched anything
    containing those two characters.
    """

    value = model_id.strip().lower()
    return any(marker in value for marker in _REASONING_MARKERS)


def is_low_latency_model(model_id: str) -> bool:
    """Heuristic: does ``model_id`` look like a small/low-latency model?"""

    value = model_id.strip().lower()
    return any(marker in value for marker in _LOW_LATENCY_MARKERS)
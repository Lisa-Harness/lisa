"""Turn-hooks factory submodules for ``runtime``.

Each submodule owns one cohesive group of hook factories that were
previously nested closures inside ``build_turn_hooks``.  The factories
are module-level functions that accept ``runtime`` (and any other
dependencies) explicitly as parameters, returning the hook callable.
"""

from runtime.turn_hooks._auto_capture import (
    AUTO_CAPTURE_TIERS,
    COMMITMENT_PATTERNS,
    DEFAULT_TIER_THRESHOLD_DAILY,
    DEFAULT_TIER_THRESHOLD_WEEKLY,
    classify_memory_tier,
    should_auto_capture,
)

__all__ = [
    "AUTO_CAPTURE_TIERS",
    "COMMITMENT_PATTERNS",
    "DEFAULT_TIER_THRESHOLD_DAILY",
    "DEFAULT_TIER_THRESHOLD_WEEKLY",
    "classify_memory_tier",
    "should_auto_capture",
]

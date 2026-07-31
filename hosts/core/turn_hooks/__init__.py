"""Turn-hooks factory submodules for ``hosts.core``.

Each submodule owns one cohesive group of hook factories that were
previously nested closures inside ``build_turn_hooks``.  The factories
are module-level functions that accept ``runtime`` (and any other
dependencies) explicitly as parameters, returning the hook callable.
"""

from hosts.core.turn_hooks._auto_capture import (
    COMMITMENT_PATTERNS,
    WRITE_EFFECT_TOOL_PREFIXES,
    WRITE_EFFECT_TOOL_SUFFIXES,
    should_auto_capture,
)

__all__ = [
    "COMMITMENT_PATTERNS",
    "WRITE_EFFECT_TOOL_PREFIXES",
    "WRITE_EFFECT_TOOL_SUFFIXES",
    "should_auto_capture",
]

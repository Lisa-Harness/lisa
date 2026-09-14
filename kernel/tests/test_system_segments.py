"""Tests for ``kernel.system_segments.SystemPromptSegments``.

Locks the cache-aware segment registry's invariants:

- Byte-stable flattening matches the legacy single-string prompt.
- Stable segments precede dynamics regardless of set order.
- Dynamic segments replace in place (never accumulate).
- ``segments()`` returns one block per stable segment plus keyed dynamics.
"""

from __future__ import annotations

import pytest

from kernel.system_segments import (
    N_HANDOFF_CONTEXT,
    N_LIFECYCLE_BRIEFING,
    N_ROUTING_HINT,
    SystemPromptSegments,
)


def test_flatten_matches_single_string_base() -> None:
    """The flattened initial prompt equals the base text passed in."""
    segments = SystemPromptSegments.from_base("stable harness\n\nmode instructions")
    assert segments.flatten() == "stable harness\n\nmode instructions"


def test_dynamic_segments_append_after_stable() -> None:
    """Dynamics render after the stable prefix in the flattened prompt."""
    segments = SystemPromptSegments.from_base("BASE")
    segments.set_dynamic(N_ROUTING_HINT, "## Routing Hint\n\nuse search")
    assert segments.flatten() == "BASE\n\n## Routing Hint\n\nuse search"


def test_lifecycle_briefing_replaces_in_place() -> None:
    """Setting the same named dynamic twice replaces rather than stacks."""
    segments = SystemPromptSegments.from_base("BASE")
    segments.set_dynamic(N_LIFECYCLE_BRIEFING, "iter 1/50")
    segments.set_dynamic(N_LIFECYCLE_BRIEFING, "iter 2/50")
    assert segments.flatten() == "BASE\n\niter 2/50"
    flattened = segments.flatten()
    assert flattened.count("iter") == 1


def test_clear_dynamic_removes_segment() -> None:
    """Clearing a dynamic removes it entirely from the flattened prompt."""
    segments = SystemPromptSegments.from_base("BASE")
    segments.set_dynamic(N_ROUTING_HINT, "## Routing Hint")
    segments.clear_dynamic(N_ROUTING_HINT)
    assert segments.flatten() == "BASE"


def test_dynamic_order_is_canonical_not_insertion() -> None:
    """Ephemeral segments render in a stable, canonical name order.

    Two turns that set the same segments in different orders produce the
    same flattened prompt — the provider cache boundary (final stable
    block) then stays byte-identical across ordering jitter.
    """
    first = SystemPromptSegments.from_base("BASE")
    first.set_dynamic(N_ROUTING_HINT, "R")
    first.set_dynamic(N_LIFECYCLE_BRIEFING, "L")

    second = SystemPromptSegments.from_base("BASE")
    second.set_dynamic(N_LIFECYCLE_BRIEFING, "L")
    second.set_dynamic(N_ROUTING_HINT, "R")

    assert first.flatten() == second.flatten()
    # Order-independent flattening means the stable prefix boundary (end of
    # "BASE") is identical across turns even when segments were set in
    # different orders — the cache breakpoint lands on the same text.
    assert first.flatten().startswith("BASE")
    assert first.flatten().endswith("L")


def test_replace_base_swaps_stable_content() -> None:
    """``replace_base`` replaces the whole stable base (mode handoff)."""
    segments = SystemPromptSegments.from_base("OLD MODE")
    segments.replace_base("NEW MODE")
    assert segments.flatten() == "NEW MODE"


def test_replace_base_rejects_empty_stable_text() -> None:
    """The stable base must never empty out (empty harness fails loudly)."""
    segments = SystemPromptSegments.from_base("BASE")
    with pytest.raises(ValueError, match="cannot be empty"):
        segments.replace_base("   ")


def test_segments_carries_stability_marker() -> None:
    """Typed segments carry the stability flag the gateway keys on."""
    segments = SystemPromptSegments.from_base("BASE")
    segments.set_dynamic(N_ROUTING_HINT, "HINT")
    typed = segments.segments()
    stable = [s for s in typed if s.is_stable]
    ephemeral = [s for s in typed if not s.is_stable]
    assert len(stable) == 1 and stable[0].text == "BASE"
    assert len(ephemeral) == 1 and ephemeral[0].text == "HINT"


def test_handoff_context_segment_does_not_touch_stable_prefix() -> None:
    """Adding the post-handoff context segment leaves the stable base intact."""
    segments = SystemPromptSegments.from_base("BASE")
    segments.set_dynamic(N_HANDOFF_CONTEXT, "handoff chain info")
    # Stable prefix is independent of dynamics: it renders first, dynamics
    # follow. The boundary text (the stable block) is unchanged.
    assert segments.flatten().startswith("BASE")
    assert "handoff chain info" in segments.flatten()

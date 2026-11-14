"""Cache-aware system-prompt segments for the turn loop.

Owns a per-turn registry of the system prompt as ordered typed segments so
the kernel can rebuild per-iteration deltas (routing hints, lifecycle
briefings, handoff context) without rewriting the stable base. The rendered
``messages[0].content`` stays byte-identical to the historical single-string
prompt for providers that never read segment metadata; the typed segments are
carried alongside on :class:`~kernel.provider_contracts.CompletionRequest` for
provider gateways that can express a cache boundary (Anthropic's
``cache_control`` blocks).

Design rules:

- **One owner.** The kernel's turn runner manages segment lifecycle through
  :class:`SystemPromptSegments` — hosts do not mutate segments directly.
- **Byte-stable flattening.** ``segments.flatten()`` uses the historical
  ``\\n\\n`` join; tests assert equality against the pre-segment prompt so
  providers that do not consume segments get identical text.
- **Replaceable dynamics.** Dynamic segments are keyed by name; replacing
  one never touches the others or the stable base. Per-iteration updates
  (the lifecycle briefing) replace in place rather than accumulating.
- **Stable order.** Stable segments always precede ephemeral segments in the
  flattened output regardless of insertion order.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from kernel.provider_contracts import (
    SystemPromptSegment,
    flatten_system_prompt_segments,
)

__all__ = [
    "DYNAMIC_SEGMENT_NAMES",
    "SystemPromptSegments",
]

#: Canonical slot name for the optional mode-variable tail of the stable
#: base (the S1 of the safe-handoff prompt layout: mode + stage
#: instructions). Splitting the stable base into an invariant head and
#: this tail lets a mid-turn handoff swap only the tail, preserving the
#: cached invariant prefix; see docs/design-safe-handoff-prompt-layout.md.
N_MODE_TAIL = "mode_tail"

#: A factored stable base leads with exactly these two stable segments:
#: the invariant head (S0) and the mode tail (S1). See
#: :meth:`SystemPromptSegments.from_parts` and
#: docs/design-safe-handoff-prompt-layout.md.
FACTORED_BASE_SEGMENT_COUNT = 2

#: Canonical names for the per-turn dynamic (ephemeral) segments. Keeping the
#: names in one module keeps the turn loop's mutation points greppable and
#: gives tests a stable vocabulary to assert against.
N_LIFECYCLE_BRIEFING = "lifecycle_briefing"
N_ROUTING_HINT = "routing_hint"
N_HANDOFF_CONTEXT = "handoff_context"
N_STAGE_INSTRUCTION = "stage_instruction"
N_TERMINAL_DIRECTIVE = "terminal_directive"

DYNAMIC_SEGMENT_NAMES = (
    N_ROUTING_HINT,
    N_HANDOFF_CONTEXT,
    N_STAGE_INSTRUCTION,
    N_TERMINAL_DIRECTIVE,
    N_LIFECYCLE_BRIEFING,
)


@dataclass(slots=True)
class SystemPromptSegments:
    """Per-turn registry of the assembled system prompt as typed segments.

    Holds ``stable``/``ephemeral`` segments in insertion order. The host calls
    to update dynamic content (the routing hint, the per-iteration lifecycle
    briefing) are keyed by ``name`` so one segment can be rewritten without
    touching the rest — the pre-segments behavior that marker-splitting
    implemented indirectly.
    """

    _stable: list[SystemPromptSegment] = field(default_factory=list)
    _dynamic: dict[str, SystemPromptSegment] = field(default_factory=dict)

    @classmethod
    def from_parts(cls, invariant: str, mode_tail: str) -> SystemPromptSegments:
        """Seed the registry as a factored S0/S1 stable base.

        ``invariant`` is the cross-mode prefix (harness + session +
        copilot customizations) that never changes mid-conversation;
        ``mode_tail`` is the mode + stage instruction a handoff swaps
        via :meth:`replace_mode_tail`. The tail is stored as a *blank*
        placeholder when empty so the head/tail boundary — and the
        provider-visible cache boundary between them — stays stable
        whether or not the target mode has an intro (flatten() drops
        blanks, so the rendered bytes are identical either way).
        """
        head = invariant.strip()
        if not head:
            raise ValueError("Invariant prompt head cannot be empty")
        tail = mode_tail.strip()
        return cls(
            _stable=[
                SystemPromptSegment(text=head, stability="stable"),
                SystemPromptSegment(text=tail, stability="stable"),
            ]
        )

    def replace_mode_tail(self, mode_tail: str) -> None:
        """Swap only the mode-variable stable tail (S1), keeping S0 bytes.

        Used by the mid-turn handoff path when the registry was seeded
        through :meth:`from_parts`: the invariant head survives the
        handoff, so provider caches that cover the head stay valid. The
        tail is always the second stable segment; an empty ``mode_tail``
        stores a blank placeholder (dropped by flattening) rather than
        collapsing the boundary.
        """
        if not self.has_factored_base():
            raise ValueError("replace_mode_tail requires a factored base seeded via from_parts")
        self._stable[1] = SystemPromptSegment(text=mode_tail.strip(), stability="stable")

    def has_factored_base(self) -> bool:
        """Report whether the stable base leads with a factored S0/S1 pair.

        True exactly when the first two stable segments are the invariant
        head and the mode tail seeded via :meth:`from_parts`. Segments
        appended after them (mid-turn stage instructions) ride behind
        the tail and do not disqualify the base; a registry seeded via
        :meth:`from_base`, or collapsed back to a merged base by
        :meth:`replace_base`, is not factored.
        """
        return len(self._stable) >= FACTORED_BASE_SEGMENT_COUNT and all(
            s.is_stable for s in self._stable[:FACTORED_BASE_SEGMENT_COUNT]
        )

    @classmethod
    def from_base(cls, stable_text: str) -> SystemPromptSegments:
        """Seed the registry with the full pre-segmentation system prompt.

        Used at turn start when the host assembles one stable string through
        ``assemble_system_prompt``. The string is kept as one stable segment;
        later static-only segments (mid-turn mode switch) replace it via
        :meth:`replace_base`.
        """
        return cls(_stable=[SystemPromptSegment(text=stable_text, stability="stable")])

    def replace_base(self, stable_text: str) -> None:
        """Swap the stable base wholesale (mode/stage changes mid-turn)."""
        if not stable_text.strip():
            raise ValueError("System prompt base cannot be empty")
        self._stable = [SystemPromptSegment(text=stable_text, stability="stable")]

    def clear_base(self) -> None:
        """Drop all stable segments.

        Only used on historical code paths that intentionally allowed an
        empty system prompt (a mid-turn handoff to a bare mode with no
        instruction body). Turn-start assembly still routes through
        :meth:`from_base`, which keeps the non-empty invariant.
        """
        self._stable = []

    def append_stable(self, text: str) -> None:
        """Append a stable segment (e.g. a mid-turn stage instruction)."""
        if text.strip():
            self._stable.append(SystemPromptSegment(text=text, stability="stable"))

    def set_dynamic(self, name: str, text: str) -> None:
        """Set or replace a dynamic segment by name (non-empty text).

        An empty/whitespace ``text`` clears the segment — the turn loop uses
        this to drop the lifecycle briefing when the resolved briefing is
        empty (per-iteration replacement, matching the marker-split behavior).
        """
        if not text.strip():
            self._dynamic.pop(name, None)
            return
        self._dynamic[name] = SystemPromptSegment(text=text, stability="ephemeral")

    def clear_dynamic(self, name: str) -> None:
        """Remove a dynamic segment entirely (e.g. after a handoff clears guidance)."""
        self._dynamic.pop(name, None)

    def dynamic_text(self, name: str) -> str:
        """Return the text of a dynamic segment, or empty string when absent."""
        segment = self._dynamic.get(name)
        return segment.text if segment is not None else ""

    def flatten(self) -> str:
        """Render the full prompt as the legacy single string."""
        ordered = self.segments()
        return flatten_system_prompt_segments(ordered)

    def segments(self) -> tuple[SystemPromptSegment, ...]:
        """Return the ordered segments — stable first, then dynamics by
        canonical name order so the boundary to the dynamic tail is stable
        across turns regardless of insertion history."""
        ordered_dynamic = [
            self._dynamic[name] for name in DYNAMIC_SEGMENT_NAMES if name in self._dynamic
        ]
        # Preserve any dynamics set outside the canonical vocabulary at the end,
        # in insertion order, so extensions keep rendering predictably.
        ordered_dynamic.extend(
            segment for name, segment in self._dynamic.items() if name not in DYNAMIC_SEGMENT_NAMES
        )
        return (*self._stable, *ordered_dynamic)

    def as_segments_copy(self) -> tuple[SystemPromptSegment, ...]:
        """Return an immutable copy of the ordered segments for a request."""
        return tuple(replace(s) for s in self.segments())

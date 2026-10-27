"""Thread-read redaction: strikethrough of regex-matched substrings.

When thread entries are read into the model's context (via
``build_recent_excerpt``, ``build_manifest``, or ``search``), the
``response_text`` and ``query`` fields may contain substrings that
contaminate future turns — e.g. prior cap-fire messages ("I've hit my
iteration budget in explainer ...") that the model reads as ground truth
and then pattern-matches against the current turn's lifecycle briefing.

This module provides :class:`ThreadRedactionPolicy`, a compiled set of
regex patterns that are replaced with a strikethrough marker when thread
text is rendered into context. The raw thread log is never mutated;
redaction is a read-time transform only.

Patterns are authored as raw regex strings and compiled once at
construction. An empty pattern list is a no-op (the common case when
redaction is not configured).
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

__all__ = [
    "DEFAULT_REDACTION_PATTERNS",
    "ThreadRedactionPolicy",
    "resolve_redaction_policy",
]

#: The strikethrough marker substituted for matched substrings. Uses the
#: Unicode strikethrough character so the redaction is visually obvious
#: in rendered output without removing the structural context (the model
#: can see *that* something was said, just not *what*).
_REDACTION_MARKER = "̶"

#: Default patterns that strip prior cap-fire / budget-exhaustion
#: self-reports from thread context. These are the messages the harness
#: stamps into ``response_text`` when a stuck-in-stage cap fires — they
#: persist in the thread and poison subsequent turns because the model
#: reads them as evidence that the budget is always about to fire.
#:
#: Patterns are deliberately broad: they match the harness-authored
#: pause/yield text and the model's own self-reported "I've hit my
#: iteration budget" prose. The raw entry is never deleted; the matched
#: substring is replaced with the strikethrough marker.
DEFAULT_REDACTION_PATTERNS: tuple[str, ...] = (
    # Model-authored: "I've hit my iteration budget in <mode> (...)."
    r"I've hit my iteration budget[^\n]*",
    # Model-authored: "I've hit my iteration budget for this turn. ..."
    r"I've hit my iteration budget for this turn[^\n]*",
    # Harness-fabricated capability-cap yield: "I've reached a resource
    # budget for this turn. ..." — b79ba861's fallback question text. The
    # 2026-10-17 turns 873/874 incident showed the claim persisting
    # unredacted (none of the older phrasings match it), so the model read
    # its own false exhaustion report back as ground truth.
    r"I've reached a resource budget[^\n]*",
    # Harness pause diagnostic: "stuck in <mode> for N iterations (cap)"
    r"stuck in \w+ for \d+ iterations \(cap\)[^\n]*",
    # Harness yield: "iteration cap N reached"
    r"iteration cap \d+ reached[^\n]*",
    # Harness yield: "turn budget exhausted"
    r"turn budget exhausted[^\n]*",
)


@dataclass(frozen=True, slots=True)
class ThreadRedactionPolicy:
    """Compiled regex set for read-time thread text redaction.

    Constructed once at host startup from configured patterns (defaults
    + any host-supplied extras). Applied per-entry by ``ThreadStore``
    read methods. Immutable and safe to share across threads.
    """

    _compiled: tuple[re.Pattern[str], ...] = field(default_factory=tuple)

    @classmethod
    def from_patterns(cls, patterns: Sequence[str]) -> ThreadRedactionPolicy:
        """Compile a list of raw regex strings into a redaction policy.

        Invalid patterns are logged and skipped — a bad regex must not
        crash the thread read path.
        """
        import logging

        log = logging.getLogger(__name__)
        compiled: list[re.Pattern[str]] = []
        for raw in patterns:
            try:
                compiled.append(re.compile(raw))
            except re.error as exc:
                log.warning("thread-redaction: skipping invalid pattern %r: %s", raw, exc)
        return cls(_compiled=tuple(compiled))

    @classmethod
    def default(cls) -> ThreadRedactionPolicy:
        """The default policy: :data:`DEFAULT_REDACTION_PATTERNS`."""
        return cls.from_patterns(DEFAULT_REDACTION_PATTERNS)

    @property
    def is_empty(self) -> bool:
        """True when no patterns are compiled (redaction is a no-op)."""
        return not self._compiled

    def redact(self, text: str) -> str:
        """Replace all pattern matches in *text* with the strikethrough marker.

        Returns the original text unchanged when the policy is empty or
        no patterns match. The raw thread entry is never mutated.
        """
        if not self._compiled or not text:
            return text
        result = text
        for pattern in self._compiled:
            result = pattern.sub(_REDACTION_MARKER, result)
        return result


# ── Late import kept at module end for circular-import safety ─────────


#: Environment contract shared by every reader that projects thread text into
#: a prompt (the ``lisa-thread`` server and the forensic reader). Keeping the
#: names and precedence here means the two cannot disagree about whether a
#: given deployment redacts.
REDACTION_ENABLED_ENV = "LISA_THREAD_REDACTION_ENABLED"
REDACTION_PATTERNS_ENV = "LISA_THREAD_REDACTION_PATTERNS"


def resolve_redaction_policy(
    *,
    enabled_value: str | None = None,
    patterns_value: str | None = None,
) -> ThreadRedactionPolicy | None:
    """Resolve the redaction policy from the environment contract.

    ``ThreadStore``'s own redaction covers only ``build_recent_excerpt`` /
    ``build_manifest`` / ``search``; ``load_range`` / ``load_recent`` /
    ``iter_range`` return RAW entries by design (the lossless read contract).
    Any surface that projects entries into a prompt — the ``lisa-thread``
    server, the forensic reader — therefore has to apply the policy itself,
    and both resolve it through this function so they cannot drift.

    Precedence: ``LISA_THREAD_REDACTION_ENABLED=0`` or
    ``LISA_THREAD_REDACTION_PATTERNS=none`` disables redaction entirely. An
    unset pattern list means the framework defaults; a non-empty list means
    those patterns only (hard cutover, no merge — same contract as the host
    config).

    ``enabled_value`` / ``patterns_value`` carry values the host already
    resolved (so an in-process consumer does not read the ambient
    environment). ``None`` means "read the environment", which is the path a
    subprocess takes.
    """
    import os

    if enabled_value is None:
        enabled = os.getenv(REDACTION_ENABLED_ENV, "").strip().lower()
        raw = os.getenv(REDACTION_PATTERNS_ENV, "").strip()
    else:
        enabled = enabled_value.strip().lower()
        raw = (patterns_value or "").strip()
    if enabled in {"0", "false", "no", "off"}:
        return None
    if raw.lower() == "none":
        return None
    if raw:
        patterns = tuple(p for p in raw.splitlines() if p.strip())
    else:
        patterns = DEFAULT_REDACTION_PATTERNS
    if not patterns:
        return None
    return ThreadRedactionPolicy.from_patterns(patterns)

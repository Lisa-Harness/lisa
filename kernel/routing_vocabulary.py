"""Session-injected routing vocabulary for the kernel.

The routing-hint generator (``kernel.turn_phases._routing_hint``) and the
declarative planner (``sdk.packs.core.operations.planning.plan_workflow``)
both need a set of *sequential-connector* substrings ("then", "based on",
...) to decide whether a multi-intent query is a sequential chain (single
handoff, step-by-step) or a concurrent fan-out (fork). Hardcoding the list
in two places made them drift-prone and forced a code change for every
vocabulary tweak.

This module exposes a small process-global registry, mirroring the
``register_content_mcp`` / ``register_operation_callbacks`` injection
contract in ``kernel.ipsar_fsm``: the host calls
``register_routing_vocabulary(...)`` once at boot with the connectors
compiled from the session plan's ``routing.sequential_connectors`` list,
and the kernel reads them via ``get_sequential_connectors()``. When no
session plan has been wired (tests, standalone use), the historical
default list is returned so existing behavior is preserved without any
caller change.

Design constraints:

- The kernel stays tag-blind and prose-blind: it only matches opaque
  connector substrings; the vocabulary is session-authored.
- The default list is the *historical* five tokens, not a new curated
  set. Sessions opt into richer coverage by listing more connectors in
  ``session_context.md`` front matter.
- Connectors are matched case-insensitively as substrings surrounded by
  whitespace. The registry normalizes entries (strips, lowercases, pads
  with single spaces) so session authors can write ``then`` or ``then ``
  interchangeably.
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "DEFAULT_SEQUENTIAL_CONNECTORS",
    "get_sequential_connectors",
    "register_routing_vocabulary",
    "reset_routing_vocabulary",
]


# Historical default. Sessions override via
# ``routing.sequential_connectors`` in ``session_context.md``. Keep this
# conservative: common words like "after" or "from" are deliberately
# omitted to avoid false positives in sessions that have not opted in.
DEFAULT_SEQUENTIAL_CONNECTORS: Final[tuple[str, ...]] = (
    " then ",
    " and then ",
    " after that ",
    " next ",
    " followed by ",
)


# Process-global registry, injected by the host at boot. ``None`` means
# "no session plan wired — use the historical default".
_sequential_connectors: tuple[str, ...] | None = None


def _normalize_connectors(raw: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    """Normalize author-supplied connectors to lowercased, space-padded tokens.

    Each entry is stripped, lowercased, and padded with a single leading and
    trailing space so substring matching against the (space-padded) query is
    word-aware. Empty entries are dropped. Duplicates are removed, preserving
    first-seen order so test assertions stay stable.
    """
    if not raw:
        return ()
    seen: set[str] = set()
    out: list[str] = []
    for entry in raw:
        token = (entry or "").strip().lower()
        if not token:
            continue
        padded = f" {token} "
        if padded in seen:
            continue
        seen.add(padded)
        out.append(padded)
    return tuple(out)


def register_routing_vocabulary(
    *,
    sequential_connectors: list[str] | tuple[str, ...] | None = None,
) -> None:
    """Inject routing vocabulary compiled from the session plan.

    Called once at host boot after the session plan is compiled. Passing
    ``None`` (or an empty list) resets to the historical default, matching
    pre-vocabulary behavior. The host is responsible for compiling the
    session plan's ``routing.sequential_connectors`` list and passing it
    here; the kernel never reads the session plan directly.
    """
    global _sequential_connectors  # noqa: PLW0603
    normalized = _normalize_connectors(sequential_connectors)
    _sequential_connectors = normalized if normalized else None


def reset_routing_vocabulary() -> None:
    """Clear any session-injected vocabulary, returning to the historical default.

    Primarily for tests that construct a ``TurnHooks`` without a host boot
    path and need a clean slate. Production code never calls this.
    """
    global _sequential_connectors  # noqa: PLW0603
    _sequential_connectors = None


def get_sequential_connectors() -> tuple[str, ...]:
    """Return the active sequential-connector vocabulary.

    Returns the session-injected list when one has been registered, else
    the historical default. The returned tuple is immutable and safe to
    iterate without copying.
    """
    if _sequential_connectors is not None:
        return _sequential_connectors
    return DEFAULT_SEQUENTIAL_CONNECTORS
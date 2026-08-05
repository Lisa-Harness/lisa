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
    "DEFAULT_CONSUMER_INTENTS",
    "DEFAULT_PRODUCER_INTENTS",
    "DEFAULT_RESUME_PHRASES",
    "DEFAULT_SEQUENTIAL_CONNECTORS",
    "get_consumer_intents",
    "get_producer_intents",
    "get_resume_phrases",
    "get_sequential_connectors",
    "is_plan_reliable_for_model",
    "mark_plan_unreliable_for_model",
    "register_routing_vocabulary",
    "reset_plan_reliability",
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


# Producer/consumer intent roles for the fork-sequential guard.
#
# A fork runs children in parallel with the whole query; none can see
# another's output. That is correct when every child is *independent*
# (e.g. ``remember`` writes memory + ``markdown`` writes a doc from the same
# query text). It is wrong when one child *consumes* another's output —
# e.g. ``markdown`` writing a doc from research that ``ponder``/``search``
# gather in parallel. The consumer has no evidence to work from, so it
# re-does the research and runs out of iterations before reaching its
# action tool (the 2026-07-27 incident).
#
# Sessions override these via ``routing.producer_intents`` /
# ``routing.consumer_intents`` in ``session_context.md``. The kernel reads
# them as opaque intent-tag sets (it never names the intents itself in
# code — these defaults are the historical vocabulary, mirroring the
# sequential-connector pattern). A fork is refused when the gap intents
# include both a producer intent and a consumer intent; the hint then
# falls back to a single handoff to the first producer, and the
# post-handoff hint chains the consumer after it.
DEFAULT_PRODUCER_INTENTS: Final[tuple[str, ...]] = (
    "research",
    "reasoning",
)
DEFAULT_CONSUMER_INTENTS: Final[tuple[str, ...]] = (
    "write",
    "coding",
)


# Resume-phrase vocabulary for the affirmative-resume and continuation-query
# detectors. The host's ``handle_message`` treats a short message that
# starts with one of these phrases as an implicit workflow/approval resume
# rather than a fresh model turn; the kernel's ``_looks_like_continuation_query``
# treats a message containing one as a continuation prompt. Both sites read
# this registry so the vocabulary stays in one place.
#
# Sessions override via ``routing.resume_phrases`` in
# ``session_context.md``. Keep the list to unambiguous imperatives —
# ``"proceed"``, ``"continue"``, ``"go ahead"`` — and avoid common words
# that would false-positive on normal conversation (``"ok"``, ``"yes"``).
# The detectors apply these as **prefix** matches on the lowered message,
# so ``"proceed with the current step in the plan"`` matches ``"proceed"``.
DEFAULT_RESUME_PHRASES: Final[tuple[str, ...]] = (
    "proceed",
    "continue",
    "go ahead",
    "go for it",
    "do it",
    "carry on",
    "pick up",
    "resume",
    "next step",
    "keep going",
)


# Process-global registry, injected by the host at boot. ``None`` means
# "no session plan wired — use the historical default".
_sequential_connectors: tuple[str, ...] | None = None
_producer_intents: frozenset[str] | None = None
_consumer_intents: frozenset[str] | None = None
_resume_phrases: tuple[str, ...] | None = None

# Per-(model, op) entry-op reliability registry. When a routing-capable
# entry operation (``plan_workflow`` today; ``delegate`` in the future)
# fails to produce a parseable ``target_stage_id`` for a given model id,
# that model is marked unreliable *for that op* here so the routing-hint
# generator skips escalation to the op's mode for it. The hint generator
# calls ``is_entry_op_reliable_for_model(model_id, op_id)`` before
# suggesting a handoff to a routing-entry-op mode; the turn runner's
# entry-op-failure path calls
# ``mark_entry_op_unreliable_for_model(model_id, op_id)`` on the first
# failure. Entries persist for the process lifetime (a model that returns
# a ``StubCompletion`` once is empirically going to keep doing it on the
# same structured-output contract); ``reset_entry_op_reliability`` clears
# the registry for tests.
#
# Thin wrappers ``mark_plan_unreliable_for_model`` /
# ``is_plan_reliable_for_model`` are kept for the ``plan_workflow`` op so
# the routing-hint generator and the turn runner's call sites stay stable
# while the registry itself is generalized. When a second routing entry
# op (e.g. ``delegate``) is added, it gets its own wrapper pair with its
# op id, and the hint generator's plan-escalation branch generalizes to a
# routing-entry-op-escalation branch.
_unreliable_entry_op_models: set[tuple[str, str]] = set()

#: The op id for ``plan_workflow`` — the sole routing entry op today.
PLAN_WORKFLOW_OP_ID = "plan_workflow"


def mark_entry_op_unreliable_for_model(model_id: str, op_id: str) -> None:
    """Record that ``op_id`` could not produce structured output for
    ``model_id`` so the routing hint generator stops escalating to that
    op's mode for that model.

    Called from the turn runner's entry-op-failure path. Idempotent. The
    entry persists for the process lifetime; a model returning
    ``StubCompletion`` once is empirically going to keep doing it on the
    same structured-output contract, and re-escalating would just waste a
    model round-trip on a known-bad path.
    """
    model_key = (model_id or "").strip().lower()
    op_key = (op_id or "").strip().lower()
    if model_key and op_key:
        _unreliable_entry_op_models.add((model_key, op_key))


def is_entry_op_reliable_for_model(model_id: str, op_id: str) -> bool:
    """Return False when ``model_id`` has been marked unreliable for ``op_id``.

    The routing-hint generator calls this before suggesting a handoff to a
    routing-entry-op mode. Unknown models are assumed reliable — the
    registry only records *observed* failures, not speculation.
    """
    model_key = (model_id or "").strip().lower()
    op_key = (op_id or "").strip().lower()
    if not model_key or not op_key:
        return True
    return (model_key, op_key) not in _unreliable_entry_op_models


def reset_entry_op_reliability() -> None:
    """Clear the per-(model, op) entry-op reliability registry.

    Primarily for tests that need a clean slate; production code never
    calls this.
    """
    _unreliable_entry_op_models.clear()


# ── plan_workflow wrappers (sole routing entry op today) ──────────────


def mark_plan_unreliable_for_model(model_id: str) -> None:
    """Thin wrapper: mark ``model_id`` unreliable for ``plan_workflow``."""
    mark_entry_op_unreliable_for_model(model_id, PLAN_WORKFLOW_OP_ID)


def is_plan_reliable_for_model(model_id: str) -> bool:
    """Thin wrapper: check ``model_id`` reliability for ``plan_workflow``."""
    return is_entry_op_reliable_for_model(model_id, PLAN_WORKFLOW_OP_ID)


def reset_plan_reliability() -> None:
    """Clear the per-(model, op) entry-op reliability registry.

    Primarily for tests that need a clean slate; production code never calls
    this.
    """
    reset_entry_op_reliability()


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
    producer_intents: list[str] | tuple[str, ...] | None = None,
    consumer_intents: list[str] | tuple[str, ...] | None = None,
    resume_phrases: list[str] | tuple[str, ...] | None = None,
) -> None:
    """Inject routing vocabulary compiled from the session plan.

    Called once at host boot after the session plan is compiled. Passing
    ``None`` (or an empty list) for a field resets that field to the
    historical default, matching pre-vocabulary behavior. The host is
    responsible for compiling the session plan's
    ``routing.sequential_connectors`` / ``routing.producer_intents`` /
    ``routing.consumer_intents`` / ``routing.resume_phrases`` lists and
    passing them here; the kernel never reads the session plan directly.
    """
    global _sequential_connectors, _producer_intents, _consumer_intents, _resume_phrases  # noqa: PLW0603
    normalized = _normalize_connectors(sequential_connectors)
    _sequential_connectors = normalized if normalized else None
    _producer_intents = _normalize_intent_roles(producer_intents, DEFAULT_PRODUCER_INTENTS)
    _consumer_intents = _normalize_intent_roles(consumer_intents, DEFAULT_CONSUMER_INTENTS)
    normalized_resume = _normalize_resume_phrases(resume_phrases)
    _resume_phrases = normalized_resume if normalized_resume else None


def _normalize_intent_roles(
    raw: list[str] | tuple[str, ...] | None,
    default: tuple[str, ...],
) -> frozenset[str] | None:
    """Normalize an intent-role list to a frozenset, or ``None`` for default."""
    if not raw:
        return None
    roles = {(entry or "").strip().lower() for entry in raw if (entry or "").strip()}
    return frozenset(roles) if roles else None


def _normalize_resume_phrases(
    raw: list[str] | tuple[str, ...] | None,
) -> tuple[str, ...]:
    """Normalize author-supplied resume phrases to lowercased, stripped tokens.

    Unlike sequential connectors, resume phrases are NOT space-padded: the
    detectors apply them as **prefix** matches on the lowered message, so
    ``"proceed"`` must match ``"proceed with the current step"`` without
    requiring surrounding spaces. Empty entries are dropped; duplicates are
    removed, preserving first-seen order so test assertions stay stable.
    """
    if not raw:
        return ()
    seen: set[str] = set()
    out: list[str] = []
    for entry in raw:
        token = (entry or "").strip().lower()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return tuple(out)


def reset_routing_vocabulary() -> None:
    """Clear any session-injected vocabulary, returning to the historical default.

    Primarily for tests that construct a ``TurnHooks`` without a host boot
    path and need a clean slate. Production code never calls this.
    """
    global _sequential_connectors, _producer_intents, _consumer_intents, _resume_phrases  # noqa: PLW0603
    _sequential_connectors = None
    _producer_intents = None
    _consumer_intents = None
    _resume_phrases = None
    reset_plan_reliability()


def get_sequential_connectors() -> tuple[str, ...]:
    """Return the active sequential-connector vocabulary.

    Returns the session-injected list when one has been registered, else
    the historical default. The returned tuple is immutable and safe to
    iterate without copying.
    """
    if _sequential_connectors is not None:
        return _sequential_connectors
    return DEFAULT_SEQUENTIAL_CONNECTORS


def get_producer_intents() -> frozenset[str]:
    """Return the active producer-intent set (evidence-gathering intents).

    Intents whose satisfaction produces text/output that a downstream
    consumer intent (write/coding) needs as input. A fork mixing a
    producer-satisfying mode with a consumer-satisfying mode is sequential,
    not parallel. Returns the session-injected set or the historical
    default.
    """
    if _producer_intents is not None:
        return _producer_intents
    return frozenset(DEFAULT_PRODUCER_INTENTS)


def get_consumer_intents() -> frozenset[str]:
    """Return the active consumer-intent set (artifact-creating intents).

    Intents whose satisfaction consumes a producer's output to create an
    artifact (a doc, a code patch). See ``get_producer_intents``.
    """
    if _consumer_intents is not None:
        return _consumer_intents
    return frozenset(DEFAULT_CONSUMER_INTENTS)


def get_resume_phrases() -> tuple[str, ...]:
    """Return the active resume-phrase vocabulary.

    Returns the session-injected list when one has been registered, else
    the historical default. The detectors in ``host.handle_message``
    (affirmative resume) and ``kernel.turn_runner`` (continuation query)
    read this so the vocabulary stays in one place. Entries are lowercased,
    stripped, and applied as prefix matches on the lowered message.
    """
    if _resume_phrases is not None:
        return _resume_phrases
    return DEFAULT_RESUME_PHRASES

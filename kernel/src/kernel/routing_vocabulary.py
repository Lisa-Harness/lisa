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
and the kernel reads them via ``get_sequential_connectors()``.

Design constraints:

- The kernel stays tag-blind and prose-blind: it only matches opaque
  connector substrings; the vocabulary is session-authored.
- **No hardcoded fallbacks.** The session defines the vocabulary or boot
  fails. There are no ``DEFAULT_*`` constants — the kernel does not
  second-guess the session. A reader (``get_*``) raises ``RuntimeError``
  when no session vocabulary has been wired for that field. This is the
  2026-08-11 hard cutover: the prior ``DEFAULT_*`` constants coupled the
  kernel to pack data and let a broken session plan silently fall back
  to stale hardcoded vocabulary.
- Connectors are matched case-insensitively as substrings surrounded by
  whitespace. The registry normalizes entries (strips, lowercases, pads
  with single spaces) so session authors can write ``then`` or ``then ``
  interchangeably.
"""

from __future__ import annotations

__all__ = [
    "get_consumer_intents",
    "get_producer_intents",
    "get_resume_phrases",
    "get_sequential_connectors",
    "get_topic_abandonment_markers",
    "is_plan_reliable_for_model",
    "mark_plan_unreliable_for_model",
    "register_routing_vocabulary",
    "reset_plan_reliability",
    "reset_routing_vocabulary",
]


# Process-global registry, injected by the host at boot. ``None`` means
# "no session plan wired — reading raises". There are no fallbacks.
_sequential_connectors: tuple[str, ...] | None = None
_producer_intents: frozenset[str] | None = None
_consumer_intents: frozenset[str] | None = None
_resume_phrases: tuple[str, ...] | None = None
# Lowercased substrings signaling the user is leaving a topic, not
# requesting it (e.g. "side quest", "rewind", "never mind"). See
# ``SessionRouting.topic_abandonment_markers``. Empty tuple (not None)
# means "session explicitly wired an empty list — no abandonment
# detection"; None means "unwired — reader raises". The kernel has no
# hardcoded fallback.
_topic_abandonment_markers: tuple[str, ...] | None = None

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
    topic_abandonment_markers: list[str] | tuple[str, ...] | None = None,
) -> None:
    """Inject routing vocabulary compiled from the session plan.

    Called once at host boot after the session plan is compiled. Each
    field is set directly from the session plan's ``routing.*`` lists —
    there are no hardcoded fallbacks. Passing ``None`` (or an empty list)
    for a field leaves that vocabulary unset; a later reader raises
    ``RuntimeError``. The host is responsible for compiling the session
    plan's ``routing.sequential_connectors`` /
    ``routing.producer_intents`` / ``routing.consumer_intents`` /
    ``routing.resume_phrases`` / ``routing.topic_abandonment_markers``
    lists and passing them here; the kernel never reads the session plan
    directly.
    """
    global _sequential_connectors, _producer_intents, _consumer_intents  # noqa: PLW0603
    global _resume_phrases, _topic_abandonment_markers  # noqa: PLW0603
    _sequential_connectors = _normalize_connectors(sequential_connectors) or None
    _producer_intents = _normalize_intent_roles(producer_intents)
    _consumer_intents = _normalize_intent_roles(consumer_intents)
    _resume_phrases = _normalize_resume_phrases(resume_phrases) or None
    _topic_abandonment_markers = _normalize_abandonment_markers(topic_abandonment_markers)


def _normalize_intent_roles(
    raw: list[str] | tuple[str, ...] | None,
) -> frozenset[str] | None:
    """Normalize an intent-role list to a frozenset, or ``None`` when unset."""
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


def _normalize_abandonment_markers(
    raw: list[str] | tuple[str, ...] | None,
) -> tuple[str, ...] | None:
    """Normalize topic-abandonment markers to lowercased, stripped tokens.

    Unlike sequential connectors, abandonment markers are NOT space-padded:
    they are multi-word discourse phrases ("side quest", "never mind",
    "different topic") and must match anywhere in the lowered query, so
    padding would break multi-word phrases that abut punctuation. The
    detector applies them as plain case-insensitive substring matches.
    Empty entries are dropped; duplicates are removed, preserving
    first-seen order. Returns ``None`` when ``raw`` is falsy so the reader
    raises (no hardcoded fallback); an explicitly empty list also yields
    ``None`` — an empty abandonment vocabulary is "unwired", not "wire an
    empty detector", matching the other fields' contract.
    """
    if not raw:
        return None
    seen: set[str] = set()
    out: list[str] = []
    for entry in raw:
        token = (entry or "").strip().lower()
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return tuple(out) or None


def reset_routing_vocabulary() -> None:
    """Clear any session-injected vocabulary, returning to the historical default.

    Primarily for tests that construct a ``TurnHooks`` without a host boot
    path and need a clean slate. Production code never calls this.
    """
    global _sequential_connectors, _producer_intents, _consumer_intents  # noqa: PLW0603
    global _resume_phrases, _topic_abandonment_markers  # noqa: PLW0603
    _sequential_connectors = None
    _producer_intents = None
    _consumer_intents = None
    _resume_phrases = None
    _topic_abandonment_markers = None
    reset_plan_reliability()


def get_sequential_connectors() -> tuple[str, ...]:
    """Return the active sequential-connector vocabulary.

    Raises ``RuntimeError`` when no session vocabulary has been wired —
    the kernel does not fall back to hardcoded defaults. Call
    :func:`register_routing_vocabulary` at boot.
    """
    if _sequential_connectors is not None:
        return _sequential_connectors
    raise RuntimeError(
        "sequential connectors not wired — register_routing_vocabulary must "
        "be called at boot with routing.sequential_connectors from the session plan"
    )


def get_producer_intents() -> frozenset[str]:
    """Return the active producer-intent set (evidence-gathering intents).

    Intents whose satisfaction produces text/output that a downstream
    consumer intent (write/coding) needs as input. A fork mixing a
    producer-satisfying mode with a consumer-satisfying mode is sequential,
    not parallel. Raises ``RuntimeError`` when no session vocabulary has
    been wired.
    """
    if _producer_intents is not None:
        return _producer_intents
    raise RuntimeError(
        "producer intents not wired — register_routing_vocabulary must be "
        "called at boot with routing.producer_intents from the session plan"
    )


def get_consumer_intents() -> frozenset[str]:
    """Return the active consumer-intent set (artifact-creating intents).

    Intents whose satisfaction consumes a producer's output to create an
    artifact (a doc, a code patch). See ``get_producer_intents``. Raises
    ``RuntimeError`` when no session vocabulary has been wired.
    """
    if _consumer_intents is not None:
        return _consumer_intents
    raise RuntimeError(
        "consumer intents not wired — register_routing_vocabulary must be "
        "called at boot with routing.consumer_intents from the session plan"
    )


def get_resume_phrases() -> tuple[str, ...]:
    """Return the active resume-phrase vocabulary.

    Returns the session-injected list. The detectors in
    ``runtime.handle_message`` (affirmative resume) and ``kernel.turn_runner``
    (continuation query) read this so the vocabulary stays in one place.
    Entries are lowercased, stripped, and applied as prefix matches on the
    lowered message. Raises ``RuntimeError`` when no session vocabulary
    has been wired.
    """
    if _resume_phrases is not None:
        return _resume_phrases
    raise RuntimeError(
        "resume phrases not wired — register_routing_vocabulary must be "
        "called at boot with routing.resume_phrases from the session plan"
    )


def get_topic_abandonment_markers() -> tuple[str, ...]:
    """Return the active topic-abandonment marker vocabulary.

    Returns the session-injected list of lowercased discourse phrases
    (e.g. ``"side quest"``, ``"rewind"``, ``"never mind"``) that signal the
    user is *leaving* a topic, not requesting it. The routing-hint
    generator's :func:`kernel.turn_phases._routing_hint._detect_query_intents`
    reads this so that a query which *mentions* a topic to abandon it
    (e.g. "the whole cloudservers thing was a side quest, so lets
    rewind") does not route the user INTO the mode whose keywords matched
    the bare substring. Entries are applied as case-insensitive substring
    matches anywhere in the lowered query. Raises ``RuntimeError`` when
    no session vocabulary has been wired.
    """
    if _topic_abandonment_markers is not None:
        return _topic_abandonment_markers
    raise RuntimeError(
        "topic abandonment markers not wired — register_routing_vocabulary "
        "must be called at boot with routing.topic_abandonment_markers from "
        "the session plan"
    )

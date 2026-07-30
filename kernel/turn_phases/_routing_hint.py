"""Proactive capability-manifest routing-hint computation.

Extracted from ``kernel.turn_runner`` to reduce the file's complexity. The
hint is injected into the system prompt when the current mode lacks tools
the query seems to need. Tag-blind and prose-blind: the kernel works over
opaque pack-authored intent-tag vocabulary.

DESIGN — Re-exported by ``kernel.turn_runner``:
    Existing tests import ``_compute_capability_routing_hint``,
    ``_detect_query_intents``, ``_query_is_sequential`` and others directly
    from ``kernel.turn_runner``. That module re-exports every symbol below so
    no test file needs editing.
"""

from __future__ import annotations

import re
from typing import Any

from kernel.ipsar_fsm import get_workflow_catalog
from kernel.models import CapabilityManifest, WorkflowModeConfig

__all__ = [
    "_compute_capability_routing_hint",
    "_detect_query_intents",
    "_first_target_for_earliest_intent",
    "_format_remaining_steps",
    "_query_is_sequential",
    "_tool_effect",
]


def _detect_query_intents(
    *,
    query: str,
    routing_targets: list[str],
) -> set[str]:
    """Detect which intents the user's query expresses, using the routing
    targets' ``fast_path_patterns`` as the pattern source.

    Returns the set of intent tag strings (lowercased) that the query matches.
    Each routing target's ``routing_intent_tags`` are attributed to the query
    when any of that target's ``fast_path_patterns`` matches the query
    (case-insensitive). This is data-driven: the kernel does not hardcode
    which keywords map to which intents — the mode front-matter declares both
    the patterns and the tags.
    """
    if not routing_targets:
        return set()
    catalog = get_workflow_catalog()
    query_lower = query.lower()
    detected: set[str] = set()
    for target_id in routing_targets:
        mode_config = catalog.get(target_id)
        if mode_config is None:
            continue
        # Check fast_path_patterns (anchored regex) first.
        matched = False
        for pattern in mode_config.fast_path_patterns:
            try:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    detected.update(mode_config.routing_intent_tags)
                    matched = True
                    break
            except re.error:
                pass
        if matched:
            continue
        # Check routing_intent_keywords (per-tag substring match). Each
        # keyword only attributes its associated tag, so a keyword match for
        # one tag does not falsely trigger a different tag the same mode
        # declares. The tag vocabulary is pack-authored and opaque to the
        # kernel.
        for tag, keywords in mode_config.routing_intent_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    detected.add(tag)
                    break
        # Also check the command_pattern — a query that explicitly names the
        # mode (e.g. "learn that...") is a strong signal.
        if not mode_config.routing_intent_keywords and mode_config.command_pattern:
            try:
                if re.search(mode_config.command_pattern, query, re.IGNORECASE):
                    detected.update(mode_config.routing_intent_tags)
            except re.error:
                pass
    return detected


def _tool_effect(
    tool_name: str,
    *,
    operation_catalog: dict[str, Any],
    tool_catalog: dict[str, Any],
) -> str:
    """Return the ``EffectKind`` of a capability id, or ``"read"`` when
    unknown.

    Resolves against the operation catalog first, then the tool catalog
    (matching ``resolve_capability`` precedence). MCP-native tools have no
    catalog definition and default to ``"read"`` — the routing-hint write-tool
    guard only fires on declared write-effect tools, so MCP tools without a
    definition are left alone.
    """
    op_def = operation_catalog.get(tool_name)
    if op_def is not None:
        return str(getattr(op_def, "effect", "read") or "read")
    tool_def = tool_catalog.get(tool_name)
    if tool_def is not None:
        return str(getattr(tool_def, "effect", "read") or "read")
    return "read"


# Sequential-connector tokens that signal a request is a *chain* of steps
# ("search ... then code ... then write") rather than a set of concurrent
# goals. A language pattern, not an intent tag, so this stays tag-blind.
#
# This is the **fallback** list. The active list is supplied per-session
# via ``routing.sequential_connectors`` in ``session_context.md`` and
# injected at boot through ``kernel.routing_vocabulary``; when no session
# vocabulary is wired (tests, standalone use), this historical set keeps
# behavior unchanged. Sessions add entries like "based on" or "once" to
# catch dependency phrasings the fallback deliberately omits.
_FALLBACK_SEQUENTIAL_CONNECTORS: tuple[str, ...] = (
    " then ",
    " and then ",
    " after that ",
    " next ",
    " followed by ",
)


def _query_is_sequential(query: str) -> bool:
    """Return True when the query expresses a sequential chain of steps.

    Detects connectors like "then", "and then", "after that", "next",
    "followed by" (plus any session-supplied additions such as "based on"
    or "once"). A sequential multi-intent request ("search for X then
    write a markdown") must NOT fork — a fork runs children in parallel, so
    none can use another's output. Instead, hand off to the first step's
    target and let the post-handoff hint recomputation chain the rest.

    The connector vocabulary is read from
    :func:`kernel.routing_vocabulary.get_sequential_connectors`, which the
    host boot path populates from the session plan's
    ``routing.sequential_connectors`` list. When no session vocabulary is
    registered, the historical fallback list above is used so existing
    behavior is preserved.
    """
    from kernel.routing_vocabulary import get_sequential_connectors

    lowered = f" {query.lower()} "
    return any(conn in lowered for conn in get_sequential_connectors())


def _first_target_for_earliest_intent(
    *,
    query: str,
    routing_targets: list[str],
    catalog: dict[str, Any],
    gap_intents: set[str] | None = None,
) -> str | None:
    """Return the routing target whose intent keywords appear earliest in
    the query — the first step in a sequential chain.

    Each target's ``routing_intent_keywords`` are scanned for their earliest
    position in the (lowercased) query. The target with the smallest
    non-``-1`` position wins. Returns ``None`` when no target's keywords
    appear in the query (the detector should have found intents, but a
    keyword may have matched via a fast_path_pattern instead).

    When ``gap_intents`` is provided, only keywords mapped to a tag in the
    gap set are considered — already-satisfied intents are skipped so the
    function picks the *next* step, not a step whose intent the current
    mode already handled. Without this filter the function would scan all
    keywords including those for satisfied intents and could pick the wrong
    next target (e.g. after search satisfies "research", it might re-pick
    search or skip coding and pick markdown).
    """
    query_lower = query.lower()
    best_target: str | None = None
    best_pos = -1
    for target_id in routing_targets:
        mode_config = catalog.get(target_id)
        if mode_config is None:
            continue
        for tag, keywords in mode_config.routing_intent_keywords.items():
            # Skip keywords for intents the current mode already satisfies.
            if gap_intents is not None and tag not in gap_intents:
                continue
            for keyword in keywords:
                pos = query_lower.find(keyword)
                if pos != -1 and (best_pos == -1 or pos < best_pos):
                    best_pos = pos
                    best_target = target_id
    return best_target


def _single_handoff_target_for_gap(
    *,
    query: str,
    routing_targets: list[str],
    mode_config: WorkflowModeConfig | None = None,
    extra_satisfied: set[str] | None = None,
) -> str | None:
    """Return the single best handoff target for the residual intent gap.

    Pure resolution helper used by the write-effect correction nudge (see
    ``kernel.turn_runner``) and any other site that needs a concrete
    ``transition_stage`` target when the active routing hint is stale or
    has been suppressed by the duplicate-hint dedup. It does NOT consult
    the plan-escalation branch (the caller already knows planning did not
    resolve the gap — re-escalating would loop). Instead it picks:

    - the single target whose ``routing_intent_tags`` cover the whole
      residual gap (the section-5 single-handoff case), or
    - the first producer target when the gap mixes a producer intent with
      a consumer intent (the section-4b producer-first case), or
    - the target whose intent keywords appear earliest in the query for a
      sequential chain (the section-4 sequential case), or
    - ``None`` when no target covers any residual gap intent.

    Tag- and prose-blind: the kernel manipulates only opaque intent-tag
    sets and routing target ids; it names no intent vocabulary. Returns
    ``None`` (rather than the generic routing-target list) so the caller
    can fall back to its own message when no concrete target exists.
    """
    if not routing_targets:
        return None
    catalog = get_workflow_catalog()
    detected_intents = _detect_query_intents(query=query, routing_targets=routing_targets)
    current_satisfied: set[str] = set(
        mode_config.satisfies_intents if mode_config is not None else []
    )
    if extra_satisfied is not None:
        current_satisfied |= extra_satisfied
    gap_intents = detected_intents - current_satisfied
    if not gap_intents:
        return None
    # Single target covering the whole residual gap → that is the
    # canonical "route back to markdown for the write gap" case.
    for target_id in routing_targets:
        t_cfg = catalog.get(target_id)
        if t_cfg is None:
            continue
        if gap_intents.issubset(set(t_cfg.routing_intent_tags)):
            return target_id
    # Producer/consumer mix or sequential chain → first producer / first
    # remaining gap-intent target (mirror sections 4b and 4 sequential).
    chained = _chained_handoff_target(
        query=query,
        routing_targets=routing_targets,
        catalog=catalog,
        gap_intents=gap_intents,
    )
    if chained is not None:
        return chained
    # Partial coverage: pick the target covering the most residual gap
    # intents (deterministic tie-break by routing_targets order).
    return _best_partial_coverage_target(
        routing_targets=routing_targets,
        catalog=catalog,
        gap_intents=gap_intents,
    )


def _chained_handoff_target(
    *,
    query: str,
    routing_targets: list[str],
    catalog: Any,
    gap_intents: set[str],
) -> str | None:
    """Return the first producer (for a producer/consumer mix) or the first
    remaining gap-intent target (for a sequential chain).

    Mirrors sections 4b and 4-sequential of
    ``_compute_capability_routing_hint`` so the write-effect nudge lands
    on the same target the proactive hint would have named. Extracted to
    keep ``_single_handoff_target_for_gap`` under the McCabe complexity
    cap. Returns ``None`` when neither branch applies or no target's
    keywords match the query.
    """
    if _fork_has_producer_consumer_mix(gap_intents):
        producer_target = _first_target_for_earliest_intent(
            query=query,
            routing_targets=routing_targets,
            catalog=catalog,
            gap_intents=gap_intents & _producer_intent_set(),
        )
        if producer_target is not None:
            return producer_target
    if _query_is_sequential(query):
        first_target = _first_target_for_earliest_intent(
            query=query,
            routing_targets=routing_targets,
            catalog=catalog,
            gap_intents=gap_intents,
        )
        if first_target is not None:
            return first_target
    return None


def _best_partial_coverage_target(
    *,
    routing_targets: list[str],
    catalog: Any,
    gap_intents: set[str],
) -> str | None:
    """Return the target covering the most residual gap intents.

    Deterministic tie-break by ``routing_targets`` order so the helper is
    stable across calls. Returns ``None`` when no target covers any gap
    intent. Extracted from ``_single_handoff_target_for_gap`` to keep that
    function under the McCabe complexity / branch caps.
    """
    best_target: str | None = None
    best_overlap = 0
    for target_id in routing_targets:
        t_cfg = catalog.get(target_id)
        if t_cfg is None:
            continue
        overlap = len(set(t_cfg.routing_intent_tags) & gap_intents)
        if overlap > best_overlap:
            best_overlap = overlap
            best_target = target_id
    return best_target


def _format_remaining_steps(remaining_steps: list[str]) -> str:
    """Render the remaining chain steps for the handoff context block.

    Returns a comma-joined list, or a sentinel when this is the final step
    so the receiving mode knows it does not need to hand off.
    """
    if remaining_steps:
        return ", ".join(remaining_steps)
    return "(none — this is the final step)"


def _compute_fork_target_string(
    *,
    query: str,
    routing_targets: list[str],
    mode_config: WorkflowModeConfig | None = None,
    extra_satisfied: set[str] | None = None,
) -> str | None:
    """Return the deterministic ``fork:mode,mode,...`` target for this query.

    This is the pure, prose-free core of the routing-hint fork branch. It
    recomputes the same intent-gap / disjoint-coverage decision that
    ``_compute_capability_routing_hint`` narrates as a hint, and returns
    the bare ``fork:`` target string (or ``None`` when the routing decision
    is a single handoff, a sequential chain, or no gap at all).

    The kernel uses this in two places:

    1. ``_compute_capability_routing_hint`` wraps it in the advisory prose
       the model sees ("Hand off by calling ``transition_stage`` with
       target_stage_id ``fork:...``").
    2. The turn runner's **fork enforcer** uses it to detect that a fork
       hint is active and, when the model refuses to call
       ``transition_stage`` itself, force the fork deterministically (see
       the 2026-07-27 incident in ``docs/lessons/``).

    Tag-blind: the kernel only inspects opaque pack-authored intent-tag
    sets and routing target ids. It names no intent vocabulary here.
    """
    if not routing_targets:
        return None
    catalog = get_workflow_catalog()
    detected_intents = _detect_query_intents(query=query, routing_targets=routing_targets)
    current_satisfied: set[str] = set(
        mode_config.satisfies_intents if mode_config is not None else []
    )
    if extra_satisfied is not None:
        current_satisfied |= extra_satisfied
    gap_intents = detected_intents - current_satisfied
    if not gap_intents:
        return None
    # Plan-mode escalation owns the 3+ intent / sequential-2-intent case.
    # A fork is only suggested for a non-sequential 2+ gap intent query.
    if "plan" in routing_targets and (
        len(gap_intents) >= 3 or (len(gap_intents) >= 2 and _query_is_sequential(query))
    ):
        return None
    # Single target covering all gaps → single handoff, not a fork.
    for target_id in routing_targets:
        t_cfg = catalog.get(target_id)
        if t_cfg is None:
            continue
        if gap_intents.issubset(set(t_cfg.routing_intent_tags)):
            return None
    # Disjoint multi-target coverage → fork (subject to the
    # producer/consumer sequential guard below).
    if len(gap_intents) < 2 or _query_is_sequential(query):
        return None
    return _build_fork_target(
        routing_targets=routing_targets,
        catalog=catalog,
        gap_intents=gap_intents,
    )


def _fork_has_producer_consumer_mix(gap_intents: set[str]) -> bool:
    """Return True when the gap mixes a producer intent with a consumer intent.

    A fork runs children in parallel; none can see another's output. When
    the gap includes both a *producer* intent (research/reasoning —
    satisfied by producing text the next step consumes) and a *consumer*
    intent (write/coding — satisfied by consuming that text to create an
    artifact), a parallel fork is the wrong shape: the consumer has no
    evidence to work from, re-does the producer's research, and runs out
    of iterations before reaching its action tool (the 2026-07-27
    incident: ``markdown`` child did 7 ``fetch_page`` calls across all 5
    iterations and never called ``write_file``).

    The intent roles are read from the session-injected routing vocabulary
    (``get_producer_intents`` / ``get_consumer_intents``); the kernel
    reads them as opaque tags and never names them in code. Returns False
    when the gap has no such mix (independent forks are still allowed).
    """
    producers = _producer_intent_set()
    consumers = _consumer_intent_set()
    return bool(gap_intents & producers) and bool(gap_intents & consumers)


def _producer_intent_set() -> frozenset[str]:
    """Return the active producer-intent set (opaque tags)."""
    from kernel.routing_vocabulary import get_producer_intents  # noqa: PLC0415

    return get_producer_intents()


def _consumer_intent_set() -> frozenset[str]:
    """Return the active consumer-intent set (opaque tags)."""
    from kernel.routing_vocabulary import get_consumer_intents  # noqa: PLC0415

    return get_consumer_intents()


def _build_fork_target(
    *,
    routing_targets: list[str],
    catalog: Any,
    gap_intents: set[str],
) -> str | None:
    """Build the ``fork:`` target string from disjoint-coverage targets.

    Returns the ``fork:mode,mode,...`` string when at least two targets
    each cover a disjoint subset of ``gap_intents`` and together cover all
    of them; otherwise ``None``. Refuses the fork when the gap mixes a
    producer intent with a consumer intent (sequential dependency — see
    ``_fork_has_producer_consumer_mix``), so the caller falls back to a
    single handoff and chains the rest via post-handoff hints. Extracted
    from the fork branch to keep ``_compute_fork_target_string`` and
    ``_compute_capability_routing_hint`` under the McCabe complexity cap.
    """
    # Sequential-dependency guard: a producer/consumer intent mix means a
    # parallel fork is the wrong shape (the consumer can't see the
    # producer's output). Return None so the caller routes sequential.
    if _fork_has_producer_consumer_mix(gap_intents):
        return None
    partial_targets: list[str] = []
    covered_intents: set[str] = set()
    for target_id in routing_targets:
        t_cfg = catalog.get(target_id)
        if t_cfg is None:
            continue
        overlap = set(t_cfg.routing_intent_tags) & gap_intents
        if overlap:
            partial_targets.append(target_id)
            covered_intents.update(overlap)
    if len(partial_targets) >= 2 and covered_intents == gap_intents:
        return f"fork:{','.join(partial_targets)}"
    return None


def _producer_first_hint(
    *,
    query: str,
    routing_targets: list[str],
    catalog: Any,
    gap_intents: set[str],
) -> str | None:
    """Build a single-handoff hint to the first producer target.

    Used when the gap mixes a producer intent with a consumer intent: the
    consumer needs the producer's output, so route to the first producer
    and let the post-handoff hint chain the consumer after. Returns the
    hint string, or ``None`` when no producer target's keywords match the
    query (the caller then falls back to the generic list). Extracted from
    ``_compute_capability_routing_hint`` to keep that function under the
    McCabe complexity cap.
    """
    if not _fork_has_producer_consumer_mix(gap_intents):
        return None
    producer_target = _first_target_for_earliest_intent(
        query=query,
        routing_targets=routing_targets,
        catalog=catalog,
        gap_intents=gap_intents & _producer_intent_set(),
    )
    if not producer_target:
        return None
    return (
        "## Routing Hint\n\n"
        f"This request requires `{producer_target}`'s tools first (it "
        f"produces evidence a later step consumes). Call "
        f"`transition_stage` with target_stage_id `{producer_target}`. "
        f"The post-handoff hint will chain the consumer step after."
    )


def _compute_capability_routing_hint(
    *,
    mode_id: str,
    query: str,
    capability_manifest: CapabilityManifest,  # retained for API stability
    routing_targets: list[str],
    mode_config: WorkflowModeConfig | None = None,
    extra_satisfied: set[str] | None = None,
    planner_model_id: str = "",
) -> str:
    """Compute a proactive routing hint when the current mode lacks tools.

    Tag-blind and prose-blind: the kernel does not name, hardcode, or
    hard-author any intent tag (research, reasoning, write, learn, ...) or
    any steering prose. Intent tags are opaque pack-authored vocabulary;
    routing prose is pack-authored via each target's ``routing_hint``
    template. Each routing target declares its own ``fast_path_patterns``
    (what queries it handles) and ``routing_intent_tags`` (what intents it
    covers) in its _mode.md front matter. The *current* mode symmetrically
    declares ``satisfies_intents`` (what intents it handles in-place). This
    function:

    1. Detects query intents by matching against all routing targets' patterns.
       The returned set is opaque strings — the kernel never inspects its
       members.
    2. Computes the gap as the set difference of detected intents minus the
       current mode's ``satisfies_intents``. If the gap is empty, no hint
       (zero tokens). The live capability manifest is NOT consulted for the
       gap check: the coarse ``kind == "tool"`` heuristic it implied treated
       any tool-bearing mode as satisfying every intent, which suppressed
       handoffs from tool-rich modes that lacked the specific intent's tools
       (e.g. coding mode falling back to coding_create_worktree instead of
       handing off to markdown for a write-a-document request).
    3. Finds routing targets whose ``routing_intent_tags`` cover all detected
       intents — if exactly one does, suggest a single handoff.
    4. If multiple targets each cover a disjoint subset of intents, suggest a
       ``fork:`` fan-out so each child runs in its own mode with its own tools.
    5. If no target fully covers the detected intents but a gap exists,
       fall back to listing all routing targets.

    Returns an empty string when no hint is needed (zero tokens).
    """
    if not routing_targets:
        return ""
    catalog = get_workflow_catalog()

    # ── 1. Detect query intents from routing target patterns ──────────
    detected_intents = _detect_query_intents(query=query, routing_targets=routing_targets)

    # ── 2. Check if the current mode already satisfies detected intents ──
    # Declarative and tag-blind: the kernel works over opaque intent-tag
    # sets. It never names any specific intent (research, reasoning, write,
    # learn, ...) — those are pack-authored vocabulary. The current mode
    # authors ``satisfies_intents`` in its _mode.md front matter; the
    # detected intents it does not claim to satisfy are the gap. Default
    # empty (e.g. ``default`` mode) means every detected intent is a gap,
    # so a handoff hint is offered. The live capability manifest is NOT
    # consulted for the gap check: the coarse ``kind == "tool"`` heuristic
    # it implied treated any tool-bearing mode as satisfying every intent,
    # which suppressed handoffs from tool-rich modes that lacked the
    # specific intent's tools (e.g. coding mode falling back to
    # coding_create_worktree instead of handing off to markdown for a
    # write-a-document request).
    current_satisfied: set[str] = set(
        mode_config.satisfies_intents if mode_config is not None else []
    )
    # Include intents satisfied by prior modes in the chain (the turn's
    # cumulative memory). Without this a sequential chain re-detects
    # already-completed steps' intents from the original query text and
    # re-suggests them — e.g. after default → search → coding, "research"
    # is still detected and still a gap because coding only satisfies
    # "coding". The accumulator prevents the infinite re-routing loop.
    if extra_satisfied is not None:
        current_satisfied |= extra_satisfied
    gap_intents = detected_intents - current_satisfied
    has_gap = bool(gap_intents)
    if not detected_intents and not has_gap:
        return ""
    if detected_intents and not has_gap:
        # The current mode already satisfies all detected intents.
        return ""

    # ── 3. Find targets that cover all remaining gap intents (single handoff) ──
    # Use ``gap_intents`` (detected minus already-satisfied), NOT
    # ``detected_intents``. After a mid-turn handoff the current mode has
    # already satisfied some intents; the hint must name a target that
    # covers what is *still* missing, not the full original set. Using
    # ``detected_intents`` here meant that after search satisfied "research",
    # no single target covered {research, coding, write} so the single-
    # handoff branch was skipped and the wrong next step was picked.
    covering_targets: list[str] = []
    for target_id in routing_targets:
        target_cfg = catalog.get(target_id)
        if target_cfg is None:
            continue
        target_tags = set(target_cfg.routing_intent_tags)
        if gap_intents and gap_intents.issubset(target_tags):
            covering_targets.append(target_id)

    # ── 3b. Plan-mode escalation for complex multi-step queries ───────
    # When the request is complex (3+ remaining gap intents, or 2+ gap
    # intents in a sequential chain), hand off to the ``plan`` action mode
    # first instead of routing directly to the first target or forking.
    # ``plan`` runs ``plan_workflow`` (one cheap model call) to
    # disambiguate the routing target, then the model calls
    # ``transition_stage`` with the returned target. Simple queries (1-2
    # intents, non-sequential) still get direct hints below — zero extra
    # model calls for the common case. This separates planning from
    # execution without a separate mechanism.
    # Tag-blind: the kernel only checks whether ``plan`` is among the
    # routable targets and whether the gap is large/sequential; it does not
    # name any intent vocabulary. This runs BEFORE the sequential/fork
    # branches so a complex sequential chain escalates to the planner
    # rather than hand-rolling the first step.
    #
    # Re-escalation guard: ``plan`` is itself a routable target that
    # declares ``satisfies_intents: [plan]``. Once the turn has run
    # ``plan``, the host's chained-hint recomputation unions ``plan`` into
    # ``chain_satisfied`` and passes it back here as ``extra_satisfied``;
    # ``current_satisfied`` therefore contains ``plan``. Without this guard
    # the branch re-fires on every post-handoff recompute (the 2026-07-27
    # incident: after ``plan → coding`` the hint said "hand off to `plan`
    # again" instead of naming ``markdown`` for the residual ``write`` gap,
    # looping until the duplicate-hint dedup suppressed it and leaving the
    # model in ``coding`` with no path to ``markdown``). Gating on
    # ``"plan" not in current_satisfied`` trips exactly once per turn.
    #
    # Model-reliability guard: when the active model has been marked
    # unreliable for planning (``plan_workflow`` previously failed to
    # produce a parseable ``target_stage_id`` for it — e.g. a local
    # backend returning ``StubCompletion`` instead of JSON), escalating
    # to ``plan`` would waste a model round-trip on a known-bad path and
    # then hard-fail the turn (the 2026-07-30 incident). Skip plan
    # escalation for unreliable models and fall through to the
    # single-handoff / fork branches instead. See
    # ``kernel.routing_vocabulary.is_plan_reliable_for_model``.
    from kernel.routing_vocabulary import is_plan_reliable_for_model  # noqa: PLC0415

    if (
        "plan" in routing_targets
        and "plan" not in current_satisfied
        and is_plan_reliable_for_model(planner_model_id)
        and (len(gap_intents) >= 3 or (len(gap_intents) >= 2 and _query_is_sequential(query)))
    ):
        return (
            "## Routing Hint\n\n"
            "This is a complex multi-step request. Hand off to `plan` by "
            "calling `transition_stage` with target_stage_id `plan`. "
            "The plan mode will analyze the query and determine the best "
            "routing target."
        )

    # ── 4. Multi-intent fork detection ─────────────────────────────────
    # When no single target covers all remaining gap intents but multiple
    # targets each cover a disjoint subset, suggest a fork fan-out. This is
    # the generic form of the learn+write case: learn covers "learn", markdown
    # covers "write", neither covers both → fork:learn,markdown.
    #
    # Sequential requests ("search ... then code ... then write a markdown")
    # must NOT fork — a fork runs the children in parallel with the whole
    # query, so none can use another's output. For those, prefer a single
    # handoff to the first step's target and let the post-handoff hint
    # recomputation chain the rest one at a time. The kernel detects a
    # sequence by the presence of sequential connectors ("then", "and then",
    # "after that", "next", "followed by") — a language pattern, not an
    # intent tag, so this does not violate tag-blindness.
    if not covering_targets and len(gap_intents) >= 2:
        sequential = _query_is_sequential(query)
        if sequential:
            # Prefer a single handoff to the first-remaining-gap-intent's
            # target so the chain can proceed step by step. Pick the target
            # whose routing_intent_keywords (for a gap intent only) appear
            # earliest in the query.
            first_target = _first_target_for_earliest_intent(
                query=query,
                routing_targets=routing_targets,
                catalog=catalog,
                gap_intents=gap_intents,
            )
            if first_target:
                hint = (
                    "## Routing Hint\n\n"
                    f"This request requires `{first_target}`'s tools as the "
                    f"first step. Call `transition_stage` with "
                    f"target_stage_id `{first_target}`. Subsequent steps "
                    f"will be chained by post-handoff hints."
                )
                return hint
            # No partial target found — fall through (silence).
        else:
            # Delegate the disjoint-coverage decision to the pure helper so
            # the fork enforcer and the hint emitter share one truth. Pass
            # ``mode_config=None`` and the already-merged ``current_satisfied``
            # as ``extra_satisfied``: the helper would otherwise re-add the
            # entry mode's satisfies, and the covering-target loop above
            # shadowed ``mode_config`` with the last target's config.
            # ``current_satisfied`` is the authoritative gap baseline here.
            fork_target = _compute_fork_target_string(
                query=query,
                routing_targets=routing_targets,
                mode_config=None,
                extra_satisfied=current_satisfied,
            )
            if fork_target is not None:
                # Purely mechanical: name the tool and the fork target id. The
                # kernel does not narrate tool-policy semantics ("mode-scoped
                # tools", "rejected") or the fork's internal mechanics — the
                # model reads each child mode's own _mode.md body after the
                # fork. Same prose-blind contract as the single-handoff branch.
                hint = (
                    "## Routing Hint\n\n"
                    f"Hand off by calling `transition_stage` with "
                    f"target_stage_id `{fork_target}`."
                )
                return hint
            # Fork refused — when the gap mixes a producer intent with a
            # consumer intent, the fork was sequential-wrong. Suggest a
            # single handoff to the first producer target so the
            # post-handoff hint can chain the consumer after it.
            _producer_hint = _producer_first_hint(
                query=query,
                routing_targets=routing_targets,
                catalog=catalog,
                gap_intents=gap_intents,
            )
            if _producer_hint is not None:
                return _producer_hint

    # ── 4b. Producer-first guard for producer/consumer mixes ──────────
    # When the gap mixes a producer intent (research/reasoning) with a
    # consumer intent (write/coding), the consumer needs the producer's
    # output as input. Route to the first *producer* target so the
    # post-handoff hint can chain the consumer after the producer
    # produces its evidence. Without this, section 5 below would pick the
    # consumer (e.g. ``markdown`` covers ``write``) and the consumer would
    # run with no evidence to consume (the 2026-07-27 incident).
    _producer_hint = _producer_first_hint(
        query=query,
        routing_targets=routing_targets,
        catalog=catalog,
        gap_intents=gap_intents,
    )
    if _producer_hint is not None:
        return _producer_hint

    # ── 5. Single handoff to the covering target ──────────────────────
    if covering_targets:
        preferred = covering_targets[0]
        # Directive, not advisory: when exactly one target cleanly covers
        # every remaining gap intent, the routing decision is unambiguous.
        # The previous hedged prose ("Your current mode *may* lack the
        # tools...") invited the model to override the hint and improvise a
        # different target (the 2026-07-30 bounce: hint said ``discuss``,
        # model went to ``ponder``). When the kernel knows the single
        # covering target, it says so directly.
        hint = (
            "## Routing Hint\n\n"
            f"This request requires `{preferred}`'s tools. Call "
            f"`transition_stage` with target_stage_id `{preferred}`."
        )
        return hint

    # ── 6. Fallback: no clean cover → silence ──────────────────────────
    # When no single target cleanly covers the gap (the disjoint-coverage
    # and producer-first branches above did not fire), the hint layer does
    # NOT know the right answer. Previously it dumped the whole target
    # list and let the model guess — that guess is the hallucination-as-
    # progress pattern (the 2026-07-30 bounce: the fallback listed
    # ``discuss`` alongside others and the model picked one at random).
    # Return empty so the model stays in the current mode and either
    # asks the user for direction or proceeds with what it has. The hint
    # is a signal, not a default.
    if not has_gap:
        return ""
    return ""

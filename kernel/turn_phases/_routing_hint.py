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


def _format_remaining_steps(remaining_steps: list[str]) -> str:
    """Render the remaining chain steps for the handoff context block.

    Returns a comma-joined list, or a sentinel when this is the final step
    so the receiving mode knows it does not need to hand off.
    """
    if remaining_steps:
        return ", ".join(remaining_steps)
    return "(none — this is the final step)"


def _compute_capability_routing_hint(
    *,
    mode_id: str,
    query: str,
    capability_manifest: CapabilityManifest,  # retained for API stability
    routing_targets: list[str],
    mode_config: WorkflowModeConfig | None = None,
    extra_satisfied: set[str] | None = None,
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
        mode_config = catalog.get(target_id)
        if mode_config is None:
            continue
        target_tags = set(mode_config.routing_intent_tags)
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
    if "plan" in routing_targets and (
        len(gap_intents) >= 3 or (len(gap_intents) >= 2 and _query_is_sequential(query))
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
                    f"Your current mode may lack the tools needed for this "
                    f"request. Hand off to `{first_target}` by calling "
                    f"`transition_stage` with target_stage_id "
                    f"`{first_target}`."
                )
                return hint
            # No partial target found — fall through to the generic list.
        else:
            partial_targets: list[str] = []
            covered_intents: set[str] = set()
            for target_id in routing_targets:
                mode_config = catalog.get(target_id)
                if mode_config is None:
                    continue
                target_tags = set(mode_config.routing_intent_tags)
                overlap = target_tags & gap_intents
                if overlap:
                    partial_targets.append(target_id)
                    covered_intents.update(overlap)
            # Only suggest a fork when the partial targets collectively cover
            # all remaining gap intents and there are at least 2 of them.
            if len(partial_targets) >= 2 and covered_intents == gap_intents:
                fork_target = f"fork:{','.join(partial_targets)}"
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

    # ── 5. Single handoff to the covering target ──────────────────────
    if covering_targets:
        preferred = covering_targets[0]
        # The kernel owns no routing prose. The handoff instruction is purely
        # mechanical: name the tool (transition_stage) and the target id. Any
        # semantic steering ("researches then reasons", "do not use the
        # one-shot search action mode", ...) is pack-authored routing *input*
        # — the target's routing_intent_tags / fast_path_patterns that drove
        # the decision to pick this target — not kernel-authored routing
        # *output*. The model reads the chosen mode's own _mode.md body after
        # the handoff; the kernel does not narrate the mode's purpose here.
        hint = (
            "## Routing Hint\n\n"
            f"Your current mode may lack the tools needed for this request. "
            f"Hand off to `{preferred}` by calling `transition_stage` with "
            f"target_stage_id `{preferred}`."
        )
        return hint

    # ── 6. Fallback: list all routing targets ──────────────────────────
    if not has_gap:
        return ""
    targets_str = ", ".join(routing_targets)
    hint = (
        "## Routing Hint\n\n"
        "Your current mode may lack the tools needed for this request. "
        f"You can hand off to a better-suited mode by calling "
        f"`transition_stage` with one of these target_stage_id values: "
        f"{targets_str}."
    )
    return hint

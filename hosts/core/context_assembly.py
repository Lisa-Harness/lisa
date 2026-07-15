"""Transport-neutral system prompt and user context assembly.

Extracted from ``hosts/little_lisa/main.py``. All functions are pure /
parameterized — no ``_RUNTIME_STATE`` access, no ``console.print``, no Rich
imports. Transport hosts inject ``content_loader``, ``user_slug``,
``timezone``, ``state``, and thread/context loaders at call sites.

Public API (leading underscores removed)::

    load_system_prompt(content_loader)
    load_route_prompt(content_loader)
    load_fast_path_verifier_prompt(content_loader)
    assemble_system_prompt(mode_instruction, stage_instruction, content_loader)
    router_mode_supported(mode_id, mode, mode_support_gap)
    build_route_catalog_markdown(mode_support_gap)
    build_user_prompt(...)
    handoff_note(...)
    build_turn_context(...)
    build_brokered_context_markdown(mode_id, query, state)
    publish_context_usage_event(...)        # stub — no event bus yet
    assemble_dynamic_user_context(...)
"""

from __future__ import annotations

import platform
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from kernel.config import settings
from kernel import router as shared_router
from kernel.capability_manifest import render_capability_manifest_context
from kernel.context.context_broker import (
    build_ambient_facts,
    render_context_packet_markdown,
    resolve_host_name,
    select_ranked_context,
)
from kernel.context.context_mapper import (
    ContextMapper,
    ContextMappingError,
    context_candidates_from_unified_state,
)
from kernel.context.models import ContextMapperRequest
from kernel.session_plan import ContextPlan
from kernel.context.context_budget import ContextBudgetAllocator, ContextLane
from kernel.context.provenance import (
    normalize_markdown_to_tagged,
    wrap_section,
)
from kernel.ipsar_fsm import get_workflow_catalog, parse_front_matter
from kernel.models import (
    CapabilityManifest,
    SurfaceContext,
    UnifiedState,
    WorkflowModeConfig,
)

from hosts.core.model_runtime import StubModelDefinition
from hosts.core.session_context import ReplayControls

# ---------------------------------------------------------------------------
# Authored-text loaders (formerly lru_cache wrappers around _read_authored_text)
# ---------------------------------------------------------------------------


def load_system_prompt(content_loader: Callable[[str], str]) -> str:
    """Load the base system prompt from the authored system prompt asset.

    The system prompt is always-on authored content loaded into every turn
    and is not optional session context. If the file is missing, return an
    empty prompt so partial installs still produce a usable payload.
    """
    try:
        return content_loader("prompts/system-prompt.md").strip()
    except FileNotFoundError:
        return ""


def load_route_prompt(content_loader: Callable[[str], str]) -> str:
    """Load the route planner prompt from ``router/route.md``."""
    try:
        return content_loader("router/route.md").strip()
    except FileNotFoundError:
        return ""


def load_fast_path_verifier_prompt(
    content_loader: Callable[[str], str],
) -> str:
    """Load the fast-path verifier prompt from ``router/fast_path_verifier.md``."""
    try:
        return content_loader("router/fast_path_verifier.md").strip()
    except FileNotFoundError:
        return ""


# ---------------------------------------------------------------------------
# System-prompt assembly
# ---------------------------------------------------------------------------


def assemble_system_prompt(
    mode_instruction: str,
    stage_instruction: str | None,
    content_loader: Callable[[str], str],
    *,
    session_instruction: str,
) -> str:
    """Assemble the stable harness, session, mode, and stage instructions."""
    parts = [
        load_system_prompt(content_loader),
        session_instruction,
        mode_instruction,
    ]
    if stage_instruction:
        parts.append(stage_instruction)
    return "\n\n".join(part for part in parts if part.strip())


# ---------------------------------------------------------------------------
# Router helpers
# ---------------------------------------------------------------------------


def router_mode_supported(
    _mode_id: str,
    mode: WorkflowModeConfig,
    mode_support_gap: Callable[[WorkflowModeConfig], str | None],
) -> bool:
    """Return True when the host can run *mode* (no support gap)."""
    return mode_support_gap(mode) is None


def build_route_catalog_markdown(
    mode_support_gap: Callable[[WorkflowModeConfig], str | None],
) -> str:
    """Render the route catalog the router planner sees."""
    return shared_router.build_route_catalog_markdown(
        "rich",
        catalog=get_workflow_catalog(),
        mode_supported=lambda mid, mode: router_mode_supported(
            mid, mode, mode_support_gap
        ),
    )


def build_user_prompt(
    *,
    normalized_message: str,
    surface_context: SurfaceContext,
    preferred_mode_id: str,
    conversation_excerpt_markdown: str,
    user_slug: str,
    fast_path_candidate_mode_id: str = "",
    is_default_executor: bool = False,
) -> str:
    """Build the user prompt submitted to the route planner."""
    _ = surface_context
    _ = preferred_mode_id
    parts = [
        "## Router Input",
        f"- user_slug: {user_slug or 'none'}",
        f"- raw_message: {normalized_message}",
        f"- is_default_executor: {str(is_default_executor).lower()}",
    ]
    if is_default_executor:
        parts.append(
            "Optimization Hint: You are ALSO the default executor for this turn. "
            "If you route to 'default', you may provide the final user-facing answer "
            "directly in the `direct_response` field to save one LLM call."
        )
    if fast_path_candidate_mode_id:
        parts.append(f"- fast_path_candidate_mode: {fast_path_candidate_mode_id}")
    if conversation_excerpt_markdown.strip():
        parts.extend(["", "## Recent Conversation", conversation_excerpt_markdown.strip()])
    parts.extend(["", "## User Query", normalized_message.strip()])
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Turn-context / handoff
# ---------------------------------------------------------------------------


def handoff_note(
    *,
    raw_message: str,
    mode_id: str,
    deep_route: bool,
    preferred_model_id: str,
) -> str:
    """Produce a human-readable handoff reason for the turn context."""
    if deep_route:
        return "explicit /deep route handoff to deep-designated model"
    if mode_id == "default" and not raw_message.lstrip().startswith("/"):
        return "natural-language intent parsed by router, executing on default model"
    if raw_message.lstrip().startswith("/"):
        return f"explicit command route handoff to {preferred_model_id} model selection"
    return "default routing handoff"


def build_turn_context(
    *,
    raw_message: str,
    mode_id: str,
    query: str,
    deep_route: bool,
    preferred_model_id: str,
    resolved_model: StubModelDefinition,
    mode: WorkflowModeConfig,
) -> dict[str, Any]:
    """Build the serializable turn-context dict stored alongside the turn."""
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "raw_message": raw_message,
        "mode_id": mode_id,
        "mode_kind": mode.mode_kind,
        "query": query,
        "deep_route": deep_route,
        "preferred_model_id": preferred_model_id,
        "resolved_model": {
            "model_id": resolved_model.model_id,
            "provider": resolved_model.provider,
            "reasoning_model": resolved_model.reasoning_model,
        },
        "handoff": handoff_note(
            raw_message=raw_message,
            mode_id=mode_id,
            deep_route=deep_route,
            preferred_model_id=preferred_model_id,
        ),
    }


# ---------------------------------------------------------------------------
# Brokered context (ranked evidence from in-process UnifiedState)
# ---------------------------------------------------------------------------


def build_session_mapped_context_markdown(
    *,
    mode_id: str,
    query: str,
    state: UnifiedState | None,
    context_plan: ContextPlan | None,
    model_context_window_tokens: int,
    active_packs: frozenset[str] | None = None,
    conversation_tokens: int = 0,
    tool_schema_tokens: int = 0,
) -> str:
    """Select context via the session-owned ContextMapper and render as Markdown.

    Falls back to an empty string when *state* is ``None`` / uncompiled or
    when no ``context_plan`` is provided. This is the session-owned mapping
    path: candidates from ``UnifiedState`` are gated, ranked, budgeted, and
    rendered into lane-ordered markdown by :class:`ContextMapper`.
    """
    if state is None or not state.compiled_at or context_plan is None:
        return ""

    try:
        candidates = context_candidates_from_unified_state(state)
    except ContextMappingError:
        return ""
    if not candidates:
        return ""

    request = ContextMapperRequest(
        context_plan=context_plan,
        query=query,
        mode_id=mode_id,
        candidates=candidates,
        active_packs=active_packs or frozenset(),
        model_context_window_tokens=model_context_window_tokens,
        stable_prefix_tokens=0,
        conversation_tokens=conversation_tokens,
        tool_schema_tokens=tool_schema_tokens,
    )
    try:
        result = ContextMapper().map(request)
    except ContextMappingError:
        return ""
    return result.rendered_context


def build_brokered_context_markdown(
    *,
    mode_id: str,
    query: str,
    state: UnifiedState | None,
    context_plan: ContextPlan | None = None,
    model_context_window_tokens: int = 0,
    active_packs: frozenset[str] | None = None,
) -> str:
    """Select ranked context evidence from *state* and render it as Markdown.

    When a session-owned ``context_plan`` and ``model_context_window_tokens``
    are provided, routes through the session-owned :class:`ContextMapper`
    (lane-aware, budget-aware, MMR-diversified). Otherwise falls back to the
    legacy ranked-context broker.

    Returns an empty string when no evidence is selected or when *state*
    is ``None`` / uncompiled.
    """
    if state is None or not state.compiled_at:
        return ""

    if context_plan is not None and model_context_window_tokens > 0:
        return build_session_mapped_context_markdown(
            mode_id=mode_id,
            query=query,
            state=state,
            context_plan=context_plan,
            model_context_window_tokens=model_context_window_tokens,
            active_packs=active_packs,
        )

    # Resolve the active mode's pack-scoped evidence boundary. The mode's
    # own provider pack and any explicitly allowed packs form the allowed
    # set. Extension evidence from other packs is excluded so that, e.g.,
    # Cloudflare's bottom-half data only appears in Cloudflare-owned modes.
    mode_config = get_workflow_catalog().get(mode_id)
    mode_provider_id = ""
    context_evidence_packs: list[str] = []
    if mode_config is not None:
        mode_provider_id = mode_config.mode_provider_id
        context_evidence_packs = mode_config.context_evidence_packs

    packet = select_ranked_context(
        state=state,
        query=query,
        mode_id=mode_id,
        mode_provider_id=mode_provider_id,
        context_evidence_packs=context_evidence_packs or None,
        routing_config=state.context_routing if state else None,
    )
    if not packet.selected_evidence:
        return ""
    return render_context_packet_markdown(packet, routing_config=state.context_routing if state else None)


# ---------------------------------------------------------------------------
# Context-usage event (stub — event bus wiring deferred)
# ---------------------------------------------------------------------------


def publish_context_usage_event(
    *,
    allocator: ContextBudgetAllocator,
    mode_id: str,
    query: str,
) -> None:
    """Publish a ContextUsageEvent with the allocator summary.

    **Stub**: the event bus is not wired in this module yet. This is a
    no-op so callers can pass it as a hook without conditional logic.
    """
    _ = allocator
    _ = mode_id
    _ = query


# ---------------------------------------------------------------------------
# Dynamic user-context assembly
# ---------------------------------------------------------------------------


def assemble_dynamic_user_context(
    mode_id: str,
    query: str,
    pre_turn_briefing: str,
    *,
    content_loader: Callable[[str], str],
    user_slug: str,
    timezone: str,
    state: UnifiedState | None,
    session_context_loader: Callable[[], str],
    thread_digest_loader: Callable[[], str],
    recent_thread_excerpt_builder: Callable[[int], str],
    replay_controls_loader: Callable[[], ReplayControls] | None = None,
    host_name: str | None = None,
    host_os: str | None = None,
    capability_manifest: CapabilityManifest | None = None,
    capability_snapshot_builder: Callable[[], dict[str, Any]] | None = None,
    mode_support_gap: Callable[[WorkflowModeConfig], str | None] | None = None,
    context_plan: ContextPlan | None = None,
    model_context_window_tokens: int = 0,
) -> str:
    """Assemble the dynamic user-context block for the current turn.

    All runtime accessors are injected as parameters — no ``_RUNTIME_STATE``
    reads. The capability-manifest metadata injection (total horizon +
    mode routing map) is preserved as a mutation of the passed-in manifest
    object; the ``_set_last_capability_manifest`` side-effect is dropped.
    """
    gap_fn = mode_support_gap or (lambda _m: None)

    # --- Inject metadata into the live capability manifest ----------------
    if capability_manifest is not None:
        if capability_snapshot_builder is not None:
            snapshot = capability_snapshot_builder()
            capability_manifest.metadata["total_horizon"] = {
                "operations": len(snapshot["operations"]),
                "tools": len(snapshot["tools"]),
                "modes": len(snapshot["supported_modes"]),
                # Full name lists so the renderer can compute inactive caps
                # via set difference and group them by reason.
                "operation_names": list(snapshot["operations"]),
                "tool_names": list(snapshot["tools"]),
            }
        # Build a compact mode→capability routing map from the workflow
        # catalog. Each mode's allowed_operation_ids and allowed_tool_ids
        # are the capabilities the model could call if transitioned there.
        catalog = get_workflow_catalog()
        routing_map: dict[str, dict[str, list[str]]] = {}
        for route_mode_id, mode_config in sorted(catalog.items()):
            if gap_fn(mode_config) is not None:
                continue
            caps = sorted(
                set(
                    list(getattr(mode_config, "allowed_operation_ids", []))
                    + list(getattr(mode_config, "allowed_tool_ids", []))
                )
            )
            if caps:
                routing_map[route_mode_id] = {"capabilities": caps}
        capability_manifest.metadata["mode_routing_map"] = routing_map

    # --- Dynamic context header ------------------------------------------
    mode = get_workflow_catalog().get(mode_id)
    mode_kind = mode.mode_kind if mode is not None else "unknown"
    mode_description = (mode.description or "").strip() if mode is not None else ""
    command_pattern = (mode.command_pattern or "").strip() if mode is not None else ""
    lines = [
        f"- mode_id: {mode_id}",
        f"- mode_kind: {mode_kind}",
        f"- query_chars: {len(query)}",
        f"- pre_turn_briefing_present: {'true' if bool(pre_turn_briefing.strip()) else 'false'}",
        f"- host: {host_name or resolve_host_name()}",
    ]
    lines.insert(2, f"- user_slug: {user_slug or 'none'}")
    if mode_description:
        lines.append(f"- mode_description: {mode_description}")
    if command_pattern:
        lines.append(f"- command_pattern: {command_pattern}")
    ambient_facts = build_ambient_facts(
        timezone=timezone,
        host_os=host_os or platform.system(),
        host_name=host_name or resolve_host_name(),
    )

    # --- Raw sections, tagged by budget lane -----------------------------
    # Each section carries a provenance kind so the emitted context uses
    # XML tags as the canonical envelope (see kernel.context.provenance).
    # ``_SECTION_PROVENANCE`` maps the section key to its provenance kind.
    raw_sections: list[tuple[str, ContextLane, str]] = [
        ("ambient_facts", "core", "## Ambient Facts\n\n" + ambient_facts),
        ("dynamic_context", "core", "## Dynamic User Context\n\n" + "\n".join(lines)),
    ]

    replay_controls = (
        replay_controls_loader() if replay_controls_loader is not None else ReplayControls()
    )
    replay_policy = _resolve_replay_policy(
        replay_controls,
        query=query,
        thread_digest_loader=thread_digest_loader,
    )

    user_context = session_context_loader().strip()
    if user_context:
        # Normalize authored markdown (which may use provenance fences)
        # into the canonical tagged representation before emission.
        raw_sections.append((
            "selected_user_context",
            "memory",
            normalize_markdown_to_tagged(user_context),
        ))

    thread_digest = replay_policy.thread_digest_text
    if replay_policy.include_prior_conversation_digest and thread_digest:
        raw_sections.append(("prior_conversation_digest", "history", thread_digest))

    conversation_excerpt = recent_thread_excerpt_builder(6)
    if replay_policy.include_recent_conversation and conversation_excerpt:
        raw_sections.append(("recent_conversation", "history", conversation_excerpt))

    if replay_policy.include_diagnostic_reflection:
        diagnostic_reflection = _build_diagnostic_reflection_block(
            trigger_mode=replay_policy.trigger_mode,
            min_query_chars=replay_policy.min_query_chars,
            effective_history_max_tokens=replay_policy.effective_history_max_tokens,
            effective_reflection_max_tokens=replay_policy.effective_reflection_max_tokens,
            digest_max_lines=replay_policy.digest_max_lines,
            query=query,
            thread_digest_present=bool(thread_digest),
            recent_conversation_present=bool(conversation_excerpt),
        )
        raw_sections.append(("diagnostic_reflection", "reflection", diagnostic_reflection))

    # Ranked context evidence from the broker (no MCP round-trip — the
    # broker is a pure ranking function over the in-process UnifiedState).
    packet_markdown = build_brokered_context_markdown(
        mode_id=mode_id,
        query=query,
        state=state,
        context_plan=context_plan,
        model_context_window_tokens=model_context_window_tokens,
    )
    if packet_markdown:
        raw_sections.append(("ranked_context", "retrieval", packet_markdown))

    # Always inject the live capability manifest so the model's self-knowledge
    # is consistent with the function schema it can actually call this turn.
    # When the manifest is unavailable, fall back to the snapshot builder.
    if capability_manifest is not None:
        capability_block = render_capability_manifest_context(capability_manifest)
    elif capability_snapshot_builder is not None:
        capability = capability_snapshot_builder()
        capability_block = (
            "## Capability Context\n\n"
            f"- supported_modes: {len(capability['supported_modes'])}\n"
            f"- operations: {len(capability['operations'])}\n"
            f"- tools: {len(capability['tools'])}\n"
            f"- models: {len(capability['models'])}"
        )
    else:
        capability_block = ""
    if capability_block:
        raw_sections.append(("capability_context", "operational", capability_block))

    # --- Route through the budget allocator ------------------------------
    # Map each section key to its provenance kind. Sections not in the map
    # default to ``authoritative`` (the runtime-state default).
    section_provenance = _SECTION_PROVENANCE
    allocator = ContextBudgetAllocator.for_turn(reasoning_model=False)
    # Tighten-only override: session front matter may reduce lane budgets,
    # never increase kernel defaults.
    allocator.lane_limits["history"] = min(
        allocator.lane_limits.get("history", replay_policy.effective_history_max_tokens),
        replay_policy.effective_history_max_tokens,
    )
    allocator.lane_limits["reflection"] = min(
        allocator.lane_limits.get("reflection", replay_policy.effective_reflection_max_tokens),
        replay_policy.effective_reflection_max_tokens,
    )
    rendered_sections: list[str] = []
    for key, lane, text in raw_sections:
        elected = allocator.allocate(
            key=key,
            lane=lane,
            text=text,
            required=(key == "dynamic_context"),
            allow_truncate=True,
        )
        if elected:
            provenance = section_provenance.get(key, "authoritative")
            wrapped = wrap_section(provenance, elected, source=key)
            if wrapped:
                rendered_sections.append(wrapped)

    publish_context_usage_event(allocator=allocator, mode_id=mode_id, query=query)
    return "\n\n".join(rendered_sections)


# Provenance kind per section key. Keys absent from this map default to
# ``authoritative``. Keep this table co-located with
# ``assemble_dynamic_user_context`` so the mapping is auditable in one place.
_SECTION_PROVENANCE: dict[str, str] = {
    "ambient_facts": "system_observed",
    "dynamic_context": "authoritative",
    "selected_user_context": "authoritative",
    "prior_conversation_digest": "inferred",
    "recent_conversation": "authoritative",
    "diagnostic_reflection": "diagnostic",
    "ranked_context": "recalled_memory",
    "capability_context": "authoritative",
}


class _ReplayPolicy:
    """Resolved replay controls after tighten-only clamping."""

    __slots__ = (
        "trigger_mode",
        "include_prior_conversation_digest",
        "include_recent_conversation",
        "include_diagnostic_reflection",
        "effective_history_max_tokens",
        "effective_reflection_max_tokens",
        "min_query_chars",
        "digest_max_lines",
        "thread_digest_text",
    )

    def __init__(
        self,
        *,
        trigger_mode: str,
        include_prior_conversation_digest: bool,
        include_recent_conversation: bool,
        include_diagnostic_reflection: bool,
        effective_history_max_tokens: int,
        effective_reflection_max_tokens: int,
        min_query_chars: int,
        digest_max_lines: int,
        thread_digest_text: str,
    ) -> None:
        self.trigger_mode = trigger_mode
        self.include_prior_conversation_digest = include_prior_conversation_digest
        self.include_recent_conversation = include_recent_conversation
        self.include_diagnostic_reflection = include_diagnostic_reflection
        self.effective_history_max_tokens = effective_history_max_tokens
        self.effective_reflection_max_tokens = effective_reflection_max_tokens
        self.min_query_chars = min_query_chars
        self.digest_max_lines = digest_max_lines
        self.thread_digest_text = thread_digest_text


def _resolve_replay_policy(
    controls: ReplayControls,
    *,
    query: str,
    thread_digest_loader: Callable[[], str],
) -> _ReplayPolicy:
    """Resolve replay controls with tighten-only clamping and trigger gating."""

    trigger_mode = controls.trigger_mode or "hybrid"
    min_query_chars = max(0, controls.min_query_chars or 0)
    query_gate_passed = len(query.strip()) >= min_query_chars

    digest_text = thread_digest_loader() or ""
    digest_max_lines = max(1, controls.digest_max_lines or 200)
    digest_text = _truncate_digest_lines(digest_text, digest_max_lines)

    include_prior = True if controls.include_prior_conversation_digest is not False else False
    include_recent = True if controls.include_recent_conversation is not False else False
    include_diag = (
        True if controls.include_diagnostic_reflection is not False else False
    )

    if trigger_mode == "manual":
        include_prior = False
        include_recent = False
    elif trigger_mode in {"conditional", "hybrid"}:
        if not query_gate_passed:
            include_prior = False
            include_recent = False

    effective_history_max_tokens = _tighten_cap(
        settings.context_budget_history_max_tokens,
        controls.history_max_tokens,
    )
    effective_reflection_max_tokens = _tighten_cap(
        settings.context_budget_reflection_max_tokens,
        controls.reflection_max_tokens,
    )

    return _ReplayPolicy(
        trigger_mode=trigger_mode,
        include_prior_conversation_digest=include_prior,
        include_recent_conversation=include_recent,
        include_diagnostic_reflection=include_diag,
        effective_history_max_tokens=effective_history_max_tokens,
        effective_reflection_max_tokens=effective_reflection_max_tokens,
        min_query_chars=min_query_chars,
        digest_max_lines=digest_max_lines,
        thread_digest_text=digest_text,
    )


def _tighten_cap(default_value: int, candidate: int | None) -> int:
    """Return a tighten-only cap: front matter may reduce defaults, never raise."""

    if candidate is None:
        return default_value
    if candidate <= 0:
        return default_value
    return min(default_value, candidate)


def _truncate_digest_lines(text: str, max_lines: int) -> str:
    """Keep only the most recent digest lines under ``max_lines``."""

    if not text.strip():
        return ""
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[-max_lines:])


def _build_diagnostic_reflection_block(
    *,
    trigger_mode: str,
    min_query_chars: int,
    effective_history_max_tokens: int,
    effective_reflection_max_tokens: int,
    digest_max_lines: int,
    query: str,
    thread_digest_present: bool,
    recent_conversation_present: bool,
) -> str:
    """Render a compact diagnostic reflection section."""

    return (
        "## Diagnostic Reflection\n\n"
        f"- replay_trigger_mode: {trigger_mode}\n"
        f"- replay_min_query_chars: {min_query_chars}\n"
        f"- replay_effective_history_max_tokens: {effective_history_max_tokens}\n"
        f"- replay_effective_reflection_max_tokens: {effective_reflection_max_tokens}\n"
        f"- replay_digest_max_lines: {digest_max_lines}\n"
        f"- replay_query_chars: {len(query.strip())}\n"
        f"- replay_thread_digest_present: {'true' if thread_digest_present else 'false'}\n"
        f"- replay_recent_conversation_present: {'true' if recent_conversation_present else 'false'}"
    )

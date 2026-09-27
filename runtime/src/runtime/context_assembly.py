"""Transport-neutral system prompt and user context assembly.

Extracted from ``hosts/little_lisa/main.py``. All functions are pure /
parameterized — no ``_RUNTIME_STATE`` access, no ``console.print``, no Rich
imports. Transport hosts inject ``content_loader``, ``user_slug``,
``timezone``, ``state``, and thread/context loaders at call sites.

Public API (leading underscores removed)::

    load_system_prompt(content_loader)
    assemble_system_prompt(mode_instruction, stage_instruction, content_loader)
    handoff_note(...)
    build_turn_context(...)
    build_brokered_context_markdown(mode_id, query, state)
    publish_context_usage_event(...)        # stub — no event bus yet
    assemble_dynamic_user_context(...)
"""

from __future__ import annotations

import contextlib
import logging
import platform
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, TypedDict, cast

from kernel.capability_manifest import render_capability_manifest_context
from kernel.context.context_broker import (
    build_ambient_facts,
    resolve_host_name,
)
from kernel.context.context_budget import ContextBudgetAllocator, ContextLane
from kernel.context.context_mapper import (
    ContextMapper,
    context_candidates_from_unified_state,
)
from kernel.context.models import ContextBudgetSnapshot, ContextMapperRequest
from kernel.context.provenance import (
    normalize_markdown_to_tagged,
    wrap_section,
)
from kernel.ipsar_fsm import get_workflow_catalog
from kernel.models import (
    CapabilityManifest,
    UnifiedState,
    WorkflowModeConfig,
)
from kernel.session_plan import ContextPlan

# MissingAuthoredContentError is intentionally imported but not used in this
# module: tests patch it as runtime.context_assembly.MissingAuthoredContentError.
from sdk.content import MissingAuthoredContentError, require_authored_text  # noqa: F401

from runtime.model_runtime import StubModelDefinition
from runtime.session_context import ReplayControls

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Authored-text loaders (formerly lru_cache wrappers around _read_authored_text)
# ---------------------------------------------------------------------------

#: The base system prompt is required authored content loaded into every
#: turn. Missing it is a hard failure — partial installs are not supported.
_SYSTEM_PROMPT_PATH = "prompts/system-prompt.md"

#: Prose hint appended to the thread manifest section, telling the model
#: how to pull prior turns on demand. Authored Markdown — not inline — so a
#: missing or empty asset fails loudly instead of silently degrading.
_THREAD_MANIFEST_HINT_PATH = "prompts/thread_manifest_hint.md"


class ContextBudgetValues(TypedDict, total=False):
    """Host-owned per-lane context budgets passed into context assembly.

    Built via :func:`context_budget_values_from_config` from a
    ``HostConfig``; mirrors the ``context_budget_*`` fields there. Keys map
    to ``ContextBudgetAllocator`` constructor inputs.
    """

    enabled: bool
    token_char_ratio: int
    core_max_tokens: int
    retrieval_max_tokens: int
    memory_max_tokens: int
    reflection_max_tokens: int
    history_max_tokens: int
    operational_max_tokens: int


#: Defaults for pure-function callers (tests, tools) that construct
#: ``assemble_dynamic_user_context`` without a ``HostConfig``. Values match
#: the ``HostConfig`` context-budget defaults; a real host always supplies
#: its own.
_DEFAULT_CONTEXT_BUDGET_VALUES: ContextBudgetValues = {
    "enabled": True,
    "token_char_ratio": 4,
    "core_max_tokens": 2000,
    "retrieval_max_tokens": 1500,
    "memory_max_tokens": 1200,
    "reflection_max_tokens": 1000,
    "history_max_tokens": 1500,
    "operational_max_tokens": 800,
}


def context_budget_values_from_config(config: Any) -> ContextBudgetValues:
    """Build the assembly-time context-budget bundle from a ``HostConfig``."""

    return {
        "enabled": bool(getattr(config, "context_budget_enabled", True)),
        "token_char_ratio": int(getattr(config, "context_budget_token_char_ratio", 4) or 4),
        "core_max_tokens": int(getattr(config, "context_budget_core_max_tokens", 2000) or 2000),
        "retrieval_max_tokens": int(
            getattr(config, "context_budget_retrieval_max_tokens", 1500) or 1500
        ),
        "memory_max_tokens": int(getattr(config, "context_budget_memory_max_tokens", 1200) or 1200),
        "reflection_max_tokens": int(
            getattr(config, "context_budget_reflection_max_tokens", 1000) or 1000
        ),
        "history_max_tokens": int(
            getattr(config, "context_budget_history_max_tokens", 1500) or 1500
        ),
        "operational_max_tokens": int(
            getattr(config, "context_budget_operational_max_tokens", 800) or 800
        ),
    }


def load_system_prompt(content_loader: Callable[[str], str]) -> str:
    """Load the base system prompt from the authored system prompt asset.

    The system prompt is always-on authored content loaded into every turn
    and is not optional session context. If the file is missing, empty, or
    the loader is unwired, this raises :class:`MissingAuthoredContentError`
    so the host fails loudly rather than executing with an empty harness.
    """
    return require_authored_text(
        content_loader,
        _SYSTEM_PROMPT_PATH,
        context="load_system_prompt",
    )


def load_thread_manifest_hint(content_loader: Callable[[str], str]) -> str:
    """Load the prose hint appended to the thread manifest section.

    The hint tells the model how to pull prior turns on demand via
    ``thread_read_range``. It is authored Markdown shipped as package data;
    a missing, empty, or unloadable asset raises
    :class:`MissingAuthoredContentError` so a broken content install fails
    loudly instead of silently stripping the addressing instruction.
    """
    return require_authored_text(
        content_loader,
        _THREAD_MANIFEST_HINT_PATH,
        context="load_thread_manifest_hint",
    )


# ---------------------------------------------------------------------------
# System-prompt assembly
# ---------------------------------------------------------------------------


def assemble_system_prompt(
    mode_instruction: str,
    stage_instruction: str | None,
    content_loader: Callable[[str], str],
    *,
    session_instruction: str,
    copilot_customizations: str = "",
) -> str:
    """Assemble the stable harness, session, mode, and stage instructions."""
    parts = [
        load_system_prompt(content_loader),
        session_instruction,
    ]
    if copilot_customizations.strip():
        parts.append(copilot_customizations.strip())
    parts.append(mode_instruction)
    if stage_instruction:
        parts.append(stage_instruction)
    return "\n\n".join(part for part in parts if part.strip())


# ---------------------------------------------------------------------------
# Turn-context / handoff
# ---------------------------------------------------------------------------


def handoff_note(
    *,
    raw_message: str,
    mode_id: str,
    preferred_model_id: str,
) -> str:
    """Produce a human-readable handoff reason for the turn context."""
    if mode_id == "default" and not raw_message.lstrip().startswith("/"):
        return (
            "natural-language intent routed to default model "
            "(mid-turn handoff via transition_stage)"
        )
    if raw_message.lstrip().startswith("/"):
        return f"explicit command route handoff to {preferred_model_id} model selection"
    return "default routing handoff"


def build_turn_context(
    *,
    raw_message: str,
    mode_id: str,
    query: str,
    preferred_model_id: str,
    resolved_model: StubModelDefinition,
    mode: WorkflowModeConfig,
) -> dict[str, Any]:
    """Build the serializable turn-context dict stored alongside the turn."""
    logger.debug(
        "turn context built: mode=%s model=%s provider=%s",
        mode_id,
        resolved_model.model_id,
        resolved_model.provider,
    )
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "raw_message": raw_message,
        "mode_id": mode_id,
        "mode_kind": mode.mode_kind,
        "query": query,
        "preferred_model_id": preferred_model_id,
        "resolved_model": {
            "model_id": resolved_model.model_id,
            "provider": resolved_model.provider,
            "reasoning_model": resolved_model.reasoning_model,
        },
        "handoff": handoff_note(
            raw_message=raw_message,
            mode_id=mode_id,
            preferred_model_id=preferred_model_id,
        ),
    }


# ---------------------------------------------------------------------------
# Brokered context (ranked items from in-process UnifiedState)
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
    snapshot_sink: Callable[[ContextBudgetSnapshot], None] | None = None,
) -> str:
    """Select context via the session-owned ContextMapper and render as Markdown.

    Falls back to an empty string when *state* is ``None`` / uncompiled or
    when no ``context_plan`` is provided. This is the session-owned mapping
    path: candidates from ``UnifiedState`` are gated, ranked, budgeted, and
    rendered into lane-ordered markdown by :class:`ContextMapper`.

    When *snapshot_sink* is provided, the mapper's ``ContextBudgetSnapshot``
    (window size, conversation tokens, used tokens, lane totals) is handed
    to the callback before the rendered markdown is returned. This lets a
    host stash the latest snapshot on its ``TurnHooks.last_context_snapshot``
    field so utilization-aware slice providers can read
    actual context-window pressure without re-running the mapper. The sink
    is never called on the empty-state fallback paths.
    """
    if state is None or not state.compiled_at or context_plan is None:
        return ""

    # A ContextMappingError from candidate extraction means the state/plan is
    # broken — not that there is nothing to say. Let it propagate so the turn
    # fails loudly instead of silently executing with empty context.
    candidates = context_candidates_from_unified_state(state)
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
    # The mapper raises ContextMappingError on a broken plan/budget, not on
    # an empty result. Let it propagate so the turn fails loudly instead of
    # silently degrading to empty context.
    result = ContextMapper().map(request)
    if snapshot_sink is not None:
        # The snapshot sink is observational only; its failure must
        # never change the rendered context the model receives.
        with contextlib.suppress(Exception):
            snapshot_sink(result.trace.budget)
    logger.debug(
        "context map: candidates=%d window=%d rendered=%d chars",
        len(candidates),
        model_context_window_tokens,
        len(result.rendered_context),
    )
    return result.rendered_context


def build_brokered_context_markdown(
    *,
    mode_id: str,
    query: str,
    state: UnifiedState | None,
    context_plan: ContextPlan | None = None,
    model_context_window_tokens: int = 0,
    active_packs: frozenset[str] | None = None,
    snapshot_sink: Callable[[ContextBudgetSnapshot], None] | None = None,
) -> str:
    """Select ranked context items from *state* and render it as Markdown.

    Routes through the session-owned :class:`ContextMapper` (lane-aware,
    budget-aware, MMR-diversified). When no ``context_plan`` is supplied, a
    default :class:`ContextPlan` is used so the mapper is always the single
    selection engine.

    Returns an empty string when no evidence is selected or when *state*
    is ``None`` / uncompiled.

    *snapshot_sink* is forwarded to the mapper so a host can capture the
    latest ``ContextBudgetSnapshot`` on each turn.
    """
    if state is None or not state.compiled_at:
        return ""

    return build_session_mapped_context_markdown(
        mode_id=mode_id,
        query=query,
        state=state,
        context_plan=context_plan or ContextPlan(),
        model_context_window_tokens=model_context_window_tokens or 131072,
        active_packs=active_packs,
        snapshot_sink=snapshot_sink,
    )


# ---------------------------------------------------------------------------
# Context-usage event (stub — event bus wiring deferred)
# ---------------------------------------------------------------------------


def publish_context_usage_event(
    *,
    allocator: ContextBudgetAllocator,
    mode_id: str,
    query: str,
) -> None:
    """Emit a structured context-usage log line for the turn.

    The event bus is not wired in this module yet; this logs the lane
    summary via stdlib logging so per-turn context salience is observable
    in host.jsonl (which lanes were spent, which sections dropped/truncated).

    Never raises — telemetry must not break the turn.
    """
    try:
        summary = allocator.summary()
        logger.info(
            "context usage: mode=%s lanes=%s dropped=%s truncated=%s",
            mode_id,
            summary.get("lane_totals"),
            summary.get("dropped_keys"),
            summary.get("truncated_keys"),
        )
    except Exception:  # pylint: disable=broad-except
        logger.debug("context usage telemetry failed", exc_info=True)


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
    recent_thread_excerpt_builder: Callable[[int], str],
    replay_controls_loader: Callable[[], ReplayControls] | None = None,
    thread_manifest_builder: Callable[[], str] | None = None,
    host_name: str | None = None,
    host_os: str | None = None,
    capability_manifest: CapabilityManifest | None = None,
    capability_snapshot_builder: Callable[[], dict[str, Any]] | None = None,
    mode_support_gap: Callable[[WorkflowModeConfig], str | None] | None = None,
    context_plan: ContextPlan | None = None,
    model_context_window_tokens: int = 0,
    snapshot_sink: Callable[[ContextBudgetSnapshot], None] | None = None,
    context_budget_values: ContextBudgetValues | None = None,
) -> str:
    """Assemble the dynamic user-context block for the current turn.

    All runtime accessors are injected as parameters — no ``_RUNTIME_STATE``
    reads. The capability-manifest metadata injection (total horizon +
    mode routing map) is preserved as a mutation of the passed-in manifest
    object; the ``_set_last_capability_manifest`` side-effect is dropped.

    *snapshot_sink* is forwarded to the session-owned context mapper so a
    host can capture the latest ``ContextBudgetSnapshot`` on each turn.
    *context_budget_values* carries the host's per-lane context budgets
    (see ``context_budget_values_from_config``); when omitted, module
    defaults are used so pure-function tests do not need a HostConfig.
    """
    budget_values = context_budget_values or _DEFAULT_CONTEXT_BUDGET_VALUES
    gap_fn = mode_support_gap or (lambda _m: None)

    # --- Inject metadata into the live capability manifest ----------------
    if capability_manifest is not None:
        _inject_capability_manifest_metadata(
            capability_manifest=capability_manifest,
            capability_snapshot_builder=capability_snapshot_builder,
            gap_fn=gap_fn,
        )

    # --- Dynamic context header ------------------------------------------
    mode = get_workflow_catalog().get(mode_id)
    lines = _build_dynamic_context_header(
        mode_id=mode_id,
        query=query,
        pre_turn_briefing=pre_turn_briefing,
        user_slug=user_slug,
        host_name=host_name,
        mode=mode,
    )
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
        budget_values=budget_values,
    )

    _append_replay_gated_sections(
        raw_sections=raw_sections,
        content_loader=content_loader,
        query=query,
        session_context_loader=session_context_loader,
        recent_thread_excerpt_builder=recent_thread_excerpt_builder,
        thread_manifest_builder=thread_manifest_builder,
        replay_policy=replay_policy,
    )

    # Ranked context evidence from the session-owned mapper (no MCP
    # round-trip — the mapper is a pure ranking function over the
    # in-process UnifiedState). The mapper already fits ranked context to
    # the model-aware window, so it is NOT re-budgeted below; the snapshot
    # it produces also drives the allocator's total ceiling so both share
    # one window-derived budget.
    mapper_snapshot: ContextBudgetSnapshot | None = None

    def _capture_snapshot(snapshot: ContextBudgetSnapshot) -> None:
        nonlocal mapper_snapshot
        mapper_snapshot = snapshot
        if snapshot_sink is not None:
            snapshot_sink(snapshot)

    # Pack-scoped item boundary: a mode sees extension items from its own
    # provider pack plus any packs it explicitly allows via
    # ``context_item_packs`` in its ``_mode.md`` front-matter. Without this
    # the mapper gates every pack-tagged item as ``inactive_pack`` and the
    # ranked context silently drops all extension contributions.
    active_packs: frozenset[str] = frozenset()
    if mode is not None:
        active_packs = frozenset({mode.mode_provider_id, *mode.context_item_packs} - {""})
    packet_markdown = build_brokered_context_markdown(
        mode_id=mode_id,
        query=query,
        state=state,
        context_plan=context_plan,
        model_context_window_tokens=model_context_window_tokens,
        active_packs=active_packs,
        snapshot_sink=_capture_snapshot,
    )
    if packet_markdown:
        raw_sections.append(("ranked_context", "retrieval", packet_markdown))

    # Always inject the live capability manifest so the model's self-knowledge
    # is consistent with the function schema it can actually call this turn.
    # When the manifest is unavailable, fall back to the snapshot builder.
    capability_block = _build_capability_block(
        capability_manifest=capability_manifest,
        capability_snapshot_builder=capability_snapshot_builder,
    )
    if capability_block:
        raw_sections.append(("capability_context", "operational", capability_block))

    # --- Route through the budget allocator ------------------------------
    # Each section key's provenance kind is resolved inside
    # ``_render_budgeted_sections``; sections not in the map default to
    # ``authoritative`` (the runtime-state default). The allocator's total
    # ceiling is the mapper's model-aware available
    # context (window minus reserves, prefix, conversation, and tool
    # schema). When the mapper produced no snapshot (no ranked context),
    # fall back to the sum of the lane ceilings so the lane limits remain
    # the binding constraint. Lane budgets and the master switch live on
    # the host config (migrated from kernel.config).
    allocator = _build_context_allocator(
        budget_values=budget_values,
        replay_policy=replay_policy,
        total_tokens=(
            mapper_snapshot.available_context_tokens if mapper_snapshot is not None else None
        ),
    )
    rendered_sections = _render_budgeted_sections(
        raw_sections=raw_sections,
        allocator=allocator,
    )

    publish_context_usage_event(allocator=allocator, mode_id=mode_id, query=query)
    return "\n\n".join(rendered_sections)


def _render_budgeted_sections(
    *,
    raw_sections: list[tuple[str, ContextLane, str]],
    allocator: ContextBudgetAllocator,
) -> list[str]:
    """Route each raw section through the budget allocator and provenance wrap.

    ``ranked_context`` bypasses the allocator: the mapper already fit it to
    the model-aware window, and re-budgeting it here would double-trim
    with a fixed lane ceiling, so it is emitted as-is. Every other section
    is allocated (``dynamic_context`` is required and may not be dropped),
    then wrapped with its provenance tag. Empty wraps are skipped.
    """
    section_provenance = _SECTION_PROVENANCE
    rendered_sections: list[str] = []
    for key, lane, text in raw_sections:
        if key == "ranked_context":
            # The mapper already fit ranked context to the model-aware
            # window; re-budgeting it here would double-trim with a fixed
            # lane ceiling. Emit it as-is.
            provenance = section_provenance.get(key, "authoritative")
            wrapped = wrap_section(provenance, text, source=key)
            if wrapped:
                rendered_sections.append(wrapped)
            continue
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
    return rendered_sections


def _build_context_allocator(
    *,
    budget_values: ContextBudgetValues,
    replay_policy: _ReplayPolicy,
    total_tokens: int | None,
) -> ContextBudgetAllocator:
    """Build the per-turn lane budget allocator with tighten-only replay caps.

    Lane budgets and the master switch live on the host config
    (``budget_values``). The allocator's total ceiling is the mapper's
    model-aware available context (``total_tokens``); when the mapper
    produced no snapshot (no ranked context), the lanes' own limits remain
    the binding constraint. Session front matter may REDUCE lane budgets
    via the replay controls, never increase them.
    """
    allocator = ContextBudgetAllocator.for_turn(
        enabled=budget_values["enabled"],
        token_char_ratio=budget_values["token_char_ratio"],
        lane_limits=cast(
            "dict[ContextLane, int]",
            {
                "core": budget_values["core_max_tokens"],
                "retrieval": budget_values["retrieval_max_tokens"],
                "memory": budget_values["memory_max_tokens"],
                "reflection": budget_values["reflection_max_tokens"],
                "history": budget_values["history_max_tokens"],
                "operational": budget_values["operational_max_tokens"],
            },
        ),
        total_tokens=total_tokens,
    )
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
    return allocator


def _append_replay_gated_sections(
    *,
    raw_sections: list[tuple[str, ContextLane, str]],
    content_loader: Callable[[str], str],
    query: str,
    session_context_loader: Callable[[], str],
    recent_thread_excerpt_builder: Callable[[int], str],
    thread_manifest_builder: Callable[[], str] | None,
    replay_policy: _ReplayPolicy,
) -> None:
    """Append the memory/history/reflection sections gated by replay policy.

    The recent-conversation excerpt is allocated BEFORE the manifest so
    the most recent dialogue wins the history-lane budget. Both share the
    ``history`` lane; whichever is appended first consumes the lane and
    the other is dropped (``budget_exhausted``) or truncated. The recent
    excerpt is ``authoritative`` and time-sensitive — it carries the
    assistant's last answer (e.g. an Option A/B offer the user just saw
    and is replying to). The manifest is ``inferred``/structured and
    covers older turns the model can pull on demand via
    ``thread_read_range``. Dropping the recent excerpt while keeping
    the manifest amputates the turn the user is responding to, so recent
    conversation must be allocated first; the manifest takes the
    remainder.
    """
    user_context = session_context_loader().strip()
    if user_context:
        # Normalize authored markdown (which may use provenance fences)
        # into the canonical tagged representation before emission.
        raw_sections.append(
            (
                "selected_user_context",
                "memory",
                normalize_markdown_to_tagged(user_context),
            )
        )

    conversation_excerpt = recent_thread_excerpt_builder(3)
    if replay_policy.include_recent_conversation and conversation_excerpt:
        raw_sections.append(("recent_conversation", "history", conversation_excerpt))

    # Thread manifest: a structured "table of contents" of the recent
    # thread entries, replacing the lossy prose digest. The manifest gives
    # the model addressable entries it can pull on demand via
    # ``thread_read_range`` instead of a summarized narrative.
    thread_manifest = ""
    if thread_manifest_builder is not None:
        try:
            thread_manifest = thread_manifest_builder().strip()
        except Exception:  # pylint: disable=broad-except
            thread_manifest = ""
    if replay_policy.include_thread_manifest and thread_manifest:
        # The hint prose ("Use the `thread_read_range` tool...") is authored
        # Markdown shipped as package data. Load it through the strict loader
        # so a missing/empty asset fails loudly instead of silently degrading
        # the model's ability to address prior turns.
        hint = load_thread_manifest_hint(content_loader)
        raw_sections.append(
            (
                "thread_manifest",
                "history",
                thread_manifest + "\n\n" + hint,
            )
        )

    if replay_policy.include_diagnostic_reflection:
        diagnostic_reflection = _build_diagnostic_reflection_block(
            trigger_mode=replay_policy.trigger_mode,
            min_query_chars=replay_policy.min_query_chars,
            effective_history_max_tokens=replay_policy.effective_history_max_tokens,
            effective_reflection_max_tokens=replay_policy.effective_reflection_max_tokens,
            query=query,
            thread_manifest_present=bool(thread_manifest),
            recent_conversation_present=bool(conversation_excerpt),
        )
        raw_sections.append(("diagnostic_reflection", "reflection", diagnostic_reflection))


def _inject_capability_manifest_metadata(
    *,
    capability_manifest: CapabilityManifest,
    capability_snapshot_builder: Callable[[], dict[str, Any]] | None,
    gap_fn: Callable[[WorkflowModeConfig], str | None],
) -> None:
    """Inject total-horizon and mode-routing metadata into the live manifest.

    The total horizon (operation/tool/mode counts plus full name lists so
    the renderer can compute inactive caps via set difference) is written
    when a snapshot builder is wired. The compact mode→capability routing
    map is built from the workflow catalog: each mode's
    allowed_operation_ids and allowed_tool_ids are the capabilities the
    model could call if transitioned there. Modes with a support gap are
    skipped.
    """
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


def _build_dynamic_context_header(
    *,
    mode_id: str,
    query: str,
    pre_turn_briefing: str,
    user_slug: str,
    host_name: str | None,
    mode: WorkflowModeConfig | None,
) -> list[str]:
    """Build the key:value header lines for the dynamic user-context section."""
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
    return lines


def _build_capability_block(
    *,
    capability_manifest: CapabilityManifest | None,
    capability_snapshot_builder: Callable[[], dict[str, Any]] | None,
) -> str:
    """Build the capability context block (live manifest, else snapshot summary)."""
    if capability_manifest is not None:
        return render_capability_manifest_context(capability_manifest)
    if capability_snapshot_builder is not None:
        capability = capability_snapshot_builder()
        return (
            "## Capability Context\n\n"
            f"- supported_modes: {len(capability['supported_modes'])}\n"
            f"- operations: {len(capability['operations'])}\n"
            f"- tools: {len(capability['tools'])}\n"
            f"- models: {len(capability['models'])}"
        )
    return ""


# Provenance kind per section key. Keys absent from this map default to
# ``authoritative``. Keep this table co-located with
# ``assemble_dynamic_user_context`` so the mapping is auditable in one place.
# ``prior_conversation_digest`` was removed in the memory × thread
# realignment: the thread manifest is the only prior-history representation.
_SECTION_PROVENANCE: dict[str, str] = {
    "ambient_facts": "system_observed",
    "dynamic_context": "authoritative",
    "selected_user_context": "authoritative",
    "recent_conversation": "authoritative",
    "thread_manifest": "inferred",
    "diagnostic_reflection": "diagnostic",
    "ranked_context": "recalled_memory",
    "capability_context": "authoritative",
}


class _ReplayPolicy:
    """Resolved replay controls after tighten-only clamping."""

    __slots__ = (
        "effective_history_max_tokens",
        "effective_reflection_max_tokens",
        "include_diagnostic_reflection",
        "include_recent_conversation",
        "include_thread_manifest",
        "min_query_chars",
        "trigger_mode",
    )

    def __init__(
        self,
        *,
        trigger_mode: str,
        include_thread_manifest: bool,
        include_recent_conversation: bool,
        include_diagnostic_reflection: bool,
        effective_history_max_tokens: int,
        effective_reflection_max_tokens: int,
        min_query_chars: int,
    ) -> None:
        self.trigger_mode = trigger_mode
        self.include_thread_manifest = include_thread_manifest
        self.include_recent_conversation = include_recent_conversation
        self.include_diagnostic_reflection = include_diagnostic_reflection
        self.effective_history_max_tokens = effective_history_max_tokens
        self.effective_reflection_max_tokens = effective_reflection_max_tokens
        self.min_query_chars = min_query_chars


def _resolve_replay_policy(
    controls: ReplayControls,
    *,
    query: str,
    budget_values: ContextBudgetValues,
) -> _ReplayPolicy:
    """Resolve replay controls with tighten-only clamping and trigger gating.

    The lossy digest path was removed in the memory × thread realignment;
    the thread manifest (built by ``thread_manifest_builder``) is the only
    prior-history section, gated by ``include_thread_manifest``.
    """

    trigger_mode = controls.trigger_mode or "hybrid"
    min_query_chars = max(0, controls.min_query_chars or 0)
    query_gate_passed = len(query.strip()) >= min_query_chars

    include_manifest = controls.include_thread_manifest is not False
    include_recent = controls.include_recent_conversation is not False
    include_diag = controls.include_diagnostic_reflection is not False

    if trigger_mode == "manual":
        include_manifest = False
        include_recent = False
    elif trigger_mode in {"conditional", "hybrid"}:
        if not query_gate_passed:
            include_manifest = False
            include_recent = False

    effective_history_max_tokens = _tighten_cap(
        budget_values["history_max_tokens"],
        controls.history_max_tokens,
    )
    effective_reflection_max_tokens = _tighten_cap(
        budget_values["reflection_max_tokens"],
        controls.reflection_max_tokens,
    )

    return _ReplayPolicy(
        trigger_mode=trigger_mode,
        include_thread_manifest=include_manifest,
        include_recent_conversation=include_recent,
        include_diagnostic_reflection=include_diag,
        effective_history_max_tokens=effective_history_max_tokens,
        effective_reflection_max_tokens=effective_reflection_max_tokens,
        min_query_chars=min_query_chars,
    )


def _tighten_cap(default_value: int, candidate: int | None) -> int:
    """Return a tighten-only cap: front matter may reduce defaults, never raise."""

    if candidate is None:
        return default_value
    if candidate <= 0:
        return default_value
    return min(default_value, candidate)


def _build_diagnostic_reflection_block(
    *,
    trigger_mode: str,
    min_query_chars: int,
    effective_history_max_tokens: int,
    effective_reflection_max_tokens: int,
    query: str,
    thread_manifest_present: bool,
    recent_conversation_present: bool,
) -> str:
    """Render a compact diagnostic reflection section."""

    return (
        "## Diagnostic Reflection\n\n"
        f"- replay_trigger_mode: {trigger_mode}\n"
        f"- replay_min_query_chars: {min_query_chars}\n"
        f"- replay_effective_history_max_tokens: {effective_history_max_tokens}\n"
        f"- replay_effective_reflection_max_tokens: {effective_reflection_max_tokens}\n"
        f"- replay_query_chars: {len(query.strip())}\n"
        f"- replay_thread_manifest_present: {'true' if thread_manifest_present else 'false'}\n"
        f"- replay_recent_conversation_present: "
        f"{'true' if recent_conversation_present else 'false'}"
    )

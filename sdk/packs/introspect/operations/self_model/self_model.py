"""Deterministic self-model producer for Lisa's operational self-awareness.

Walks the wired catalogs (workflows, operations, tools, models) and emits a
``LisaSelfModelArtifact`` with entries across all seven ``SelfKnowledgeCategory``
buckets. This operation is invoked by the slice system, not by the LLM — it
requires no model call and runs entirely within the slice budget.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from kernel.ipsar_fsm import get_workflow_catalog
from kernel.model_manifest import ModelManifest
from kernel.models import (
    IntrospectSnapshot,
    LisaSelfModelArtifact,
    OperationExecutionContext,
    OperationResult,
    SelfKnowledgeEntry,
)

logger = logging.getLogger(__name__)

_VIBE_MAX_CHARS = 500


async def execute(ctx: OperationExecutionContext) -> OperationResult:
    """Build a fresh ``LisaSelfModelArtifact`` from the wired runtime catalogs."""

    # Lazy import to avoid circular import with kernel.operations.registry
    from kernel.operations.registry import get_operation_catalog  # noqa: PLC0415

    now_iso = datetime.now(UTC).isoformat()
    user_slug = ctx.user_slug or "default"

    # ── Gather source data ──────────────────────────────────────────────
    workflow_catalog = get_workflow_catalog()
    operation_catalog = get_operation_catalog()
    tool_catalog = _resolve_tool_catalog(ctx)
    model_manifest = _resolve_model_manifest(ctx)

    supported_modes: list[str] = []
    unsupported_modes: dict[str, str] = {}
    for mode_id, mode in sorted(workflow_catalog.items()):
        gap = _mode_support_gap(mode)
        if gap is None:
            supported_modes.append(mode_id)
        else:
            unsupported_modes[mode_id] = gap

    # ── Build entries by category ───────────────────────────────────────

    identity_entries = _build_identity_entries(ctx, user_slug, now_iso)
    capability_entries = _build_capability_entries(
        supported_modes, operation_catalog, tool_catalog, model_manifest, now_iso
    )
    limitation_entries = _build_limitation_entries(unsupported_modes, now_iso)
    architecture_entries = _build_architecture_entries(now_iso)
    memory_entries = _build_memory_entries(ctx, now_iso)
    grounding_entries = _build_grounding_entries(ctx, now_iso)
    known_unknown_entries = _build_known_unknown_entries(unsupported_modes, now_iso)

    identity_summary = (
        f"Lisa operating as {user_slug} with {len(supported_modes)} supported modes, "
        f"{len(operation_catalog)} operations, {len(tool_catalog)} tools, "
        f"and {len(model_manifest.models) if model_manifest else 0} models."
    )
    grounding_summary = (
        f"Host: little_lisa. Timezone: {_resolve_timezone(ctx)}. "
        f"Surface: {ctx.surface_type or 'personal_dm'}."
    )

    artifact = LisaSelfModelArtifact(
        generated_at=now_iso,
        workspace_id=ctx.workspace_id or "default",
        user_slug=user_slug,
        surface_type=ctx.surface_type or "personal_dm",
        surface_id=ctx.surface_id or "",
        memory_scope=ctx.memory_scope or "personal",
        identity_summary=identity_summary,
        grounding_summary=grounding_summary,
        identity=identity_entries,
        capabilities=capability_entries,
        limitations=limitation_entries,
        architecture=architecture_entries,
        memory_awareness=memory_entries,
        grounding=grounding_entries,
        known_unknowns=known_unknown_entries,
    )

    # ── Augment with live introspection telemetry (no LLM call) ──────────
    snapshot = await _gather_introspection(ctx)
    if snapshot is not None:
        artifact.uptime_seconds = snapshot.uptime_seconds
        artifact.asyncio_tasks = snapshot.asyncio_tasks
        artifact.python_threads = snapshot.python_threads
        artifact.process_rss_mb = snapshot.process_rss_mb
        artifact.mcp_pool_health = snapshot.mcp_pool_health
        artifact.token_usage_24h = snapshot.token_usage_24h
        artifact.recent_log_errors = snapshot.recent_log_errors
        artifact.operational_health = snapshot.operational_health
        artifact.operational_health_reasons = snapshot.operational_health_reasons

    vibe = build_self_model_vibe(artifact)

    return OperationResult(
        output_text=vibe,
        output_json=artifact.model_dump(mode="json"),
        extra={
            "entry_counts": {
                "identity": len(identity_entries),
                "capabilities": len(capability_entries),
                "limitations": len(limitation_entries),
                "architecture": len(architecture_entries),
                "memory_awareness": len(memory_entries),
                "grounding": len(grounding_entries),
                "known_unknowns": len(known_unknown_entries),
            },
        },
    )


def build_self_model_vibe(artifact: LisaSelfModelArtifact) -> str:
    """Extract a compact (~500 char) vibe string for prompt injection."""

    mode_list = (
        ", ".join(entry.title for entry in artifact.capabilities if entry.title)[:12] or "none"
    )
    if len(artifact.capabilities) > 12:
        mode_list += f" (+{len(artifact.capabilities) - 12} more)"

    limitation_preview = (
        ", ".join(entry.title for entry in artifact.limitations[:3] if entry.title) or "none"
    )

    known_unknown_preview = (
        ", ".join(entry.title for entry in artifact.known_unknowns[:3] if entry.title) or "none"
    )

    vibe = (
        f"## Self Model\n"
        f"{artifact.identity_summary}\n"
        f"Modes: {mode_list}\n"
        f"Limitations: {limitation_preview}\n"
        f"Known unknowns: {known_unknown_preview}\n"
        f"{artifact.grounding_summary}"
    )
    # Append live runtime telemetry when available
    if artifact.uptime_seconds > 0 or artifact.asyncio_tasks > 0:
        runtime_line = (
            f"\nRuntime: uptime={int(artifact.uptime_seconds)}s, "
            f"tasks={artifact.asyncio_tasks}, threads={artifact.python_threads}, "
            f"rss={artifact.process_rss_mb:.0f}MB"
        )
        failed_mcp = [sid for sid, h in artifact.mcp_pool_health.items() if h.status == "failed"]
        if failed_mcp:
            runtime_line += f", mcp_down={','.join(failed_mcp)}"
        if artifact.operational_health != "unknown":
            runtime_line += f", health={artifact.operational_health}"
        vibe += runtime_line
        if artifact.operational_health_reasons:
            vibe += f"\nHealth reasons: {', '.join(artifact.operational_health_reasons[:3])}"
    if len(vibe) > _VIBE_MAX_CHARS:
        vibe = vibe[: _VIBE_MAX_CHARS - 3] + "..."
    return vibe


# ── Private helpers ────────────────────────────────────────────────────────


async def _gather_introspection(ctx: OperationExecutionContext) -> IntrospectSnapshot | None:
    """Call the host's ``introspect`` callback when wired.

    Returns ``None`` when the callback is not set or raises — the self-model
    still works without live telemetry, it just lacks the runtime fields.
    """
    if ctx.host is None:
        return None
    try:
        result = ctx.host.introspect()
        # Support both sync and async callables
        if hasattr(result, "__await__"):
            result = await result
        if isinstance(result, IntrospectSnapshot):
            return result
    except Exception:
        logger.debug("introspect failed; self-model telemetry fields empty", exc_info=True)
    return None


def _resolve_tool_catalog(ctx: OperationExecutionContext) -> dict[str, Any]:
    """Runtime-wide tool catalog from the in-process registry.

    This deliberately does NOT read ``ctx.tool_schemas``: that field is
    populated by the operation registry as a per-operation allow-list (the
    tools the *current* operation is allowed to call). ``self_model`` declares
    no ``tool_ids`` because it reports on tools rather than calling them, so
    ``ctx.tool_schemas`` is always empty here — reading it produced a
    perpetual "0 tools" count. The runtime-wide inventory lives in
    ``kernel.tools.registry.get_tool_catalog``, the same source the operation
    registry itself uses internally.
    """
    try:
        from kernel.tools.registry import get_tool_catalog  # noqa: PLC0415

        return dict(get_tool_catalog())
    except Exception:
        logger.debug(
            "get_tool_catalog unavailable; self-model tool count will be 0",
            exc_info=True,
        )
        return {}


def _resolve_model_manifest(ctx: OperationExecutionContext) -> ModelManifest | None:
    """Best-effort model manifest from the host's resolve_model.

    The host adapter exposes ``resolve_model`` (single-model lookup) but
    not the manifest itself, so this helper cannot return a ``ModelManifest``
    today. It stays as a no-op seam so the self-model summary and capability
    entries degrade to "0 models" without raising. Do not probe role
    designations (e.g. ``fast_chat``) here — nothing maps them and the
    failed lookup logs an error on every slice run.
    """
    del ctx  # explicit: the manifest is not reachable via the host adapter
    return None


def _mode_support_gap(mode: Any) -> str | None:
    """Check whether a mode has a known support gap. Best-effort."""
    try:
        # Modes with a mode_kind that requires host wiring
        if hasattr(mode, "mode_kind") and str(mode.mode_kind) in (
            "workflow",
            "browser",
            "image",
        ):
            return f"requires host wiring for {mode.mode_kind}"
    except Exception:
        pass
    return None


def _resolve_timezone(ctx: OperationExecutionContext) -> str:
    """Best-effort timezone from context extra."""
    extra = ctx.extra or {}
    tz = extra.get("timezone") or extra.get("tz") or "UTC"
    return str(tz)


def _make_entry_id(category: str, index: int) -> str:
    return f"self_{category}_{index:03d}"


def _build_identity_entries(
    ctx: OperationExecutionContext, user_slug: str, now_iso: str
) -> list[SelfKnowledgeEntry]:
    return [
        SelfKnowledgeEntry(
            entry_id=_make_entry_id("identity", 0),
            category="identity",
            title="Host runtime",
            body=f"Lisa is running on the little_lisa host as user '{user_slug}'.",
            confidence=1.0,
            source_refs=["host:little_lisa"],
            updated_at=now_iso,
        ),
        SelfKnowledgeEntry(
            entry_id=_make_entry_id("identity", 1),
            category="identity",
            title="Surface context",
            body=(
                f"Surface type: {ctx.surface_type or 'personal_dm'}, "
                f"scope: {ctx.memory_scope or 'personal'}."
            ),
            confidence=1.0,
            source_refs=["ctx.surface_type", "ctx.memory_scope"],
            updated_at=now_iso,
        ),
    ]


def _build_capability_entries(
    supported_modes: list[str],
    operation_catalog: dict[str, Any],
    tool_catalog: dict[str, Any],
    model_manifest: ModelManifest | None,
    now_iso: str,
) -> list[SelfKnowledgeEntry]:
    entries: list[SelfKnowledgeEntry] = []
    idx = 0

    for mode_id in supported_modes:
        entries.append(
            SelfKnowledgeEntry(
                entry_id=_make_entry_id("capability", idx),
                category="capability",
                title=mode_id,
                body=f"Mode '{mode_id}' is supported and wired.",
                confidence=1.0,
                source_refs=["workflow_catalog"],
                tags=["mode"],
                updated_at=now_iso,
            )
        )
        idx += 1

    for op_id in sorted(operation_catalog.keys())[:20]:
        entries.append(
            SelfKnowledgeEntry(
                entry_id=_make_entry_id("capability", idx),
                category="capability",
                title=op_id,
                body=f"Operation '{op_id}' is registered.",
                confidence=1.0,
                source_refs=["operation_catalog"],
                tags=["operation"],
                updated_at=now_iso,
            )
        )
        idx += 1

    for tool_id in sorted(tool_catalog.keys())[:20]:
        entries.append(
            SelfKnowledgeEntry(
                entry_id=_make_entry_id("capability", idx),
                category="capability",
                title=tool_id,
                body=f"Tool '{tool_id}' is available.",
                confidence=1.0,
                source_refs=["tool_catalog"],
                tags=["tool"],
                updated_at=now_iso,
            )
        )
        idx += 1

    if model_manifest is not None:
        for model in model_manifest.models[:10]:
            entries.append(
                SelfKnowledgeEntry(
                    entry_id=_make_entry_id("capability", idx),
                    category="capability",
                    title=model.model_id,
                    body=f"Model '{model.model_id}' (provider: {model.provider}).",
                    confidence=1.0,
                    source_refs=["model_manifest"],
                    tags=["model"],
                    updated_at=now_iso,
                )
            )
            idx += 1

    return entries


def _build_limitation_entries(
    unsupported_modes: dict[str, str], now_iso: str
) -> list[SelfKnowledgeEntry]:
    entries: list[SelfKnowledgeEntry] = []
    for idx, (mode_id, gap) in enumerate(sorted(unsupported_modes.items())):
        entries.append(
            SelfKnowledgeEntry(
                entry_id=_make_entry_id("limitation", idx),
                category="limitation",
                title=mode_id,
                body=f"Mode '{mode_id}' is not supported: {gap}.",
                confidence=1.0,
                source_refs=["workflow_catalog"],
                tags=["unsupported_mode"],
                updated_at=now_iso,
            )
        )
    return entries


def _build_architecture_entries(now_iso: str) -> list[SelfKnowledgeEntry]:
    return [
        SelfKnowledgeEntry(
            entry_id=_make_entry_id("architecture", 0),
            category="architecture",
            title="Kernel layer",
            body=(
                "Lisa uses a kernel + SDK + packs architecture. "
                "The kernel owns the turn runner, slice system, provider gateways, "
                "and operation registry. The SDK adds MCP servers and context assembly. "
                "Packs ship modes, operations, and content."
            ),
            confidence=0.9,
            source_refs=["kernel/README.md"],
            tags=["layer_model"],
            updated_at=now_iso,
        ),
    ]


def _build_memory_entries(ctx: OperationExecutionContext, now_iso: str) -> list[SelfKnowledgeEntry]:
    has_memory = ctx.host is not None
    return [
        SelfKnowledgeEntry(
            entry_id=_make_entry_id("memory", 0),
            category="memory",
            title="Memory store",
            body=("Memory tools available via MCP." if has_memory else "Memory tools not wired."),
            confidence=1.0,
            source_refs=["ctx.host.execute_tool"],
            tags=["memory"],
            updated_at=now_iso,
        ),
    ]


def _build_grounding_entries(
    ctx: OperationExecutionContext, now_iso: str
) -> list[SelfKnowledgeEntry]:
    extra = ctx.extra or {}
    tz = _resolve_timezone(ctx)
    return [
        SelfKnowledgeEntry(
            entry_id=_make_entry_id("grounding", 0),
            category="grounding",
            title="Timezone",
            body=f"Operating timezone: {tz}.",
            confidence=1.0,
            source_refs=["ctx.extra.timezone"],
            tags=["timezone"],
            updated_at=now_iso,
        ),
    ]


def _build_known_unknown_entries(
    unsupported_modes: dict[str, str], now_iso: str
) -> list[SelfKnowledgeEntry]:
    entries: list[SelfKnowledgeEntry] = []
    idx = 0
    for mode_id, gap in sorted(unsupported_modes.items()):
        entries.append(
            SelfKnowledgeEntry(
                entry_id=_make_entry_id("known_unknown", idx),
                category="known_unknown",
                title=f"Unsupported: {mode_id}",
                body=f"Mode '{mode_id}' requires host capability not available: {gap}.",
                confidence=0.8,
                source_refs=["workflow_catalog"],
                tags=["gap", mode_id],
                updated_at=now_iso,
            )
        )
        idx += 1
    return entries

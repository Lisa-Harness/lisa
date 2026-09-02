"""Dynamic capability manifest compiler.

Compiles a :class:`CapabilityManifest` from the actual tool schemas that
will be sent to the model for the current turn. This closes the "Tooling
Paradox" — the capability context the model reads in its prompt now
reflects exactly what is injected into the current turn's function schema,
not a stale catalog walk.

The compiler cross-references the flat ``tool_schemas`` list (OpenAI
function-tool dicts) against the in-process operation and tool catalogs to
classify each entry by kind (operation, tool, mode-level FSM control, or
MCP tool). Every entry is marked ``status="available"`` with
``confidence=1.0`` because the manifest is deterministic — it is the ground
truth for this turn, not an educated guess.

Called by: ``kernel.turn_runner._run_turn_inner`` after tool schemas are
assembled and before ``assemble_user_context`` is invoked.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from kernel.models import (
    CapabilityConstraint,
    CapabilityEntry,
    CapabilityManifest,
)
from kernel.operations.registry import get_operation_catalog
from kernel.tools.registry import get_tool_catalog

_TRANSITION_TOOL_NAME = "transition_stage"


def compile_capability_manifest(
    tool_schemas: list[dict[str, Any]],
    mode_config: Any | None = None,
    *,
    captured_at: str = "",
    total_horizon: dict[str, int] | None = None,
) -> CapabilityManifest:
    """Compile a deterministic capability manifest from live tool schemas.

    Parameters
    ----------
    tool_schemas
        The OpenAI function-tool dicts that will be sent to the model this
        turn. Each entry is ``{"type": "function", "function": {"name": ...}}``.
    mode_config
        The active ``WorkflowModeConfig`` (or None). Used to record mode-level
        constraints (``allowed_tool_ids``, ``tools_authoritative``).
    captured_at
        ISO timestamp for ``last_checked_at`` on every entry. Defaults to now.
    total_horizon
        Optional dict with ``operations``, ``tools``, ``modes`` keys giving the
        total capability counts across all catalogs (the superset). When
        provided and different from the active counts, the renderer emits a
        ``total_capability_horizon`` line so the model can distinguish "what I
        can call right now" from "what I could do in another mode."
    """
    timestamp = captured_at or datetime.now(UTC).isoformat()
    operation_catalog = get_operation_catalog()
    tool_catalog = get_tool_catalog()

    entries: list[CapabilityEntry] = []
    kind_counts: dict[str, int] = {}

    for schema in tool_schemas:
        func = schema.get("function", {})
        name = func.get("name", "")
        if not name:
            continue
        description = func.get("description", "")

        if name in operation_catalog:
            kind: Literal["operation", "tool", "mode"] = "operation"
            source = "operation_catalog"
        elif name in tool_catalog:
            kind = "tool"
            source = "tool_catalog"
        elif name == _TRANSITION_TOOL_NAME:
            kind = "mode"
            source = "workflow_fsm"
        else:
            kind = "tool"
            source = "mcp_pool"

        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        entries.append(
            CapabilityEntry(
                capability_id=name,
                kind=kind,
                display_name=name,
                description=description,
                status="available",
                invocation=name,
                source=source,
                confidence=1.0,
                last_checked_at=timestamp,
            )
        )

    mode_id = getattr(mode_config, "mode_id", "") if mode_config else ""
    tools_authoritative = bool(mode_config and getattr(mode_config, "tools_authoritative", True))
    allowed_tool_ids = list(getattr(mode_config, "allowed_tool_ids", [])) if mode_config else []

    constraints: list[CapabilityConstraint] = []
    if tools_authoritative:
        constraints.append(
            CapabilityConstraint(
                constraint_type="mode",
                description=f"Mode '{mode_id}' is authoritative: only listed tools/operations are advertised.",
                value="tools_authoritative",
            )
        )
    elif allowed_tool_ids:
        constraints.append(
            CapabilityConstraint(
                constraint_type="mode",
                description=f"Mode '{mode_id}' restricts tools to: {', '.join(allowed_tool_ids)}",
                value="allowed_tool_ids",
            )
        )

    metadata: dict[str, Any] = {
        "mode_id": mode_id,
        "tools_authoritative": tools_authoritative,
        "allowed_tool_ids": allowed_tool_ids,
        "kind_counts": kind_counts,
    }
    if total_horizon:
        metadata["total_horizon"] = total_horizon

    summary = _build_summary(
        total=len(entries),
        kind_counts=kind_counts,
        mode_id=mode_id,
        tools_authoritative=tools_authoritative,
    )

    return CapabilityManifest(
        generated_at=timestamp,
        entries=entries,
        summary=summary,
        metadata=metadata,
    )


def _build_summary(
    *,
    total: int,
    kind_counts: dict[str, int],
    mode_id: str,
    tools_authoritative: bool,
) -> str:
    """Compact one-line summary for ``capability_manifest_summary`` fields."""
    parts = [f"{total} available"]
    detail_parts: list[str] = []
    for kind in ("operation", "tool", "mode"):
        count = kind_counts.get(kind, 0)
        if count:
            detail_parts.append(f"{count} {kind}s" if count != 1 else f"{count} {kind}")
    if detail_parts:
        parts.append(f"({', '.join(detail_parts)})")
    if mode_id:
        parts.append(f"mode={mode_id}")
    if tools_authoritative:
        parts.append("authoritative")
    return " ".join(parts)


def render_capability_manifest_context(manifest: CapabilityManifest) -> str:
    """Render a compact markdown block for context injection.

    This is the block that replaces the stale "Capability Context" section
    in the user context. It lists exact counts and the available capability
    names so the model can report its status with certainty.
    """
    meta = manifest.metadata
    mode_id = meta.get("mode_id", "")
    kind_counts: dict[str, int] = meta.get("kind_counts", {})
    tools_authoritative = meta.get("tools_authoritative", True)

    operations = kind_counts.get("operation", 0)
    workflow = kind_counts.get("mode", 0)
    mcp_count = sum(1 for e in manifest.entries if e.source == "mcp_pool")
    pack_tool_count = sum(1 for e in manifest.entries if e.source == "tool_catalog")

    lines = [
        "## Capability Context (live this turn)",
        "",
        f"- mode: {mode_id or 'default'}",
        f"- available_tools: {len(manifest.entries)} "
        f"(operations: {operations}, pack_tools: {pack_tool_count}, mcp: {mcp_count}, workflow: {workflow})",
        f"- tools_authoritative: {'yes' if tools_authoritative else 'no'}",
    ]

    # Mode routing map: intentionally NOT emitted into the model-facing
    # context. Naming capabilities owned by other modes leaks dormant
    # affordances and primes the model toward tools it cannot call this
    # turn. The routing map remains available in manifest.metadata for
    # developer-side introspection / debugging only.

    total_horizon = meta.get("total_horizon")
    if total_horizon and isinstance(total_horizon, dict):
        horizon_parts: list[str] = []
        for key, label in (("operations", "ops"), ("tools", "tools"), ("modes", "modes")):
            horizon_count = total_horizon.get(key, 0)
            if horizon_count:
                horizon_parts.append(f"{horizon_count} {label}")
        if horizon_parts:
            lines.append(
                f"- total_capability_horizon: {', '.join(horizon_parts)} "
                f"(available in other modes; only the {len(manifest.entries)} listed below are callable this turn)"
            )

    # Inactive capabilities are intentionally NOT listed in the model-facing
    # context. Emitting dormant capability names (even as "unavailable") primes
    # the model toward affordances it cannot exercise this turn. The inactive
    # set remains computable from manifest.metadata for developer-side tooling.

    available_names = sorted(e.capability_id for e in manifest.entries)
    if available_names:
        lines.append(f"- available: {', '.join(available_names)}")

    return "\n".join(lines)

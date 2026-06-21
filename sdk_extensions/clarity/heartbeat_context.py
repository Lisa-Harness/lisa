"""Extension-owned heartbeat prompt-context fragments for Clarity insights."""

from __future__ import annotations

from kernel.extension_runtime import (
    ExtensionHeartbeatContextFragment,
    ExtensionRuntimeContext,
)
from kernel.models import UnifiedState


def build_heartbeat_context_fragments(
    runtime_context: ExtensionRuntimeContext,
    state: UnifiedState,
) -> list[ExtensionHeartbeatContextFragment]:
    """Expose Clarity-owned heartbeat context blocks without host-owned rendering."""

    del runtime_context
    lines = ["## Clarity Insights"]
    clarity_collection = state.intel_collections.get("clarity")
    if clarity_collection and clarity_collection.summary_bullets:
        lines.extend(f"- {bullet}" for bullet in clarity_collection.summary_bullets[:6])
    else:
        lines.append("- No clarity data yet.")

    return [
        ExtensionHeartbeatContextFragment(
            order=30,
            prompt_context_blocks=("\n".join(lines),),
        )
    ]


__all__ = ["build_heartbeat_context_fragments"]

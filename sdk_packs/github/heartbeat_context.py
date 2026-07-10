"""Extension-owned heartbeat prompt-context fragments for GitHub."""

from __future__ import annotations

from kernel.extension_runtime import (
    ExtensionHeartbeatContextFragment,
    ExtensionRuntimeContext,
)
from kernel.models import UnifiedState
from sdk_packs.github.models import GitHubStatus


def build_heartbeat_context_fragments(
    runtime_context: ExtensionRuntimeContext,
    state: UnifiedState,
) -> list[ExtensionHeartbeatContextFragment]:
    """Expose GitHub context without app-owned rendering branches."""

    del runtime_context
    lines = ["## GitHub"]

    snapshot = state.signal_snapshots.get("github")
    if snapshot is not None:
        github = GitHubStatus.model_validate(snapshot)
        latest_commit = f"`{github.latest_commit_sha}` {github.latest_commit_message}".strip()
        lines.append(
            f"- Status: {github.status.upper()} | Branch: {(github.tracked_branch or 'unknown')} | "
            f"Workflow: {github.main_workflow_status.upper()} | "
            f"Latest: {latest_commit if latest_commit else 'n/a'}"
        )
    else:
        lines.append("- No GitHub snapshot yet.")

    return [
        ExtensionHeartbeatContextFragment(
            order=12,
            prompt_context_blocks=("\n".join(lines),),
        )
    ]


__all__ = ["build_heartbeat_context_fragments"]


"""Extension-owned digest sections for GitHub."""

from __future__ import annotations

from kernel.extension_runtime import ExtensionDigestSection, ExtensionRuntimeContext
from kernel.models import UnifiedState
from lisa_pack_github.models import GitHubStatus


def build_digest_sections(
    runtime_context: ExtensionRuntimeContext,
    state: UnifiedState,
) -> list[ExtensionDigestSection]:
    """Expose GitHub digest lines without host-owned rendering branches."""

    del runtime_context
    snapshot = state.signal_snapshots.get("github")
    if snapshot is None:
        return []

    github = GitHubStatus.model_validate(snapshot)
    return [
        ExtensionDigestSection(
            lines=(
                "- GitHub: "
                f"{github.status.upper()} | branch: {(github.tracked_branch or 'unknown')} | "
                f"workflow: {github.main_workflow_status.upper()} | "
                f"latest: {github.latest_commit_sha} {github.latest_commit_message}",
            ),
            order=12,
        )
    ]


__all__ = ["build_digest_sections"]


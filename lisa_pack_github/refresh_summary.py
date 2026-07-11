"""Extension-owned refresh-summary card details for GitHub snapshots."""

from __future__ import annotations

from kernel.models import UnifiedState
from lisa_pack_github.models import GitHubStatus
from sdk.rendering import RefreshSummarySection


def build_refresh_summary_sections(state: UnifiedState) -> list[RefreshSummarySection]:
    """Render GitHub-specific detail lines for the host refresh summary card."""

    snapshot = state.signal_snapshots.get("github")
    if snapshot is None:
        return []

    github = GitHubStatus.model_validate(snapshot)
    lines = [
        f"Workflow {github.main_workflow_status.upper()}",
        f"Open PRs {github.open_prs}",
    ]
    if github.latest_commit_sha or github.latest_commit_message:
        lines.append((f"Latest {github.latest_commit_sha} {github.latest_commit_message}").strip())

    return [
        RefreshSummarySection(
            connector_id="github",
            lines=tuple(lines),
            order=10,
        )
    ]


__all__ = ["build_refresh_summary_sections"]



"""GitHub status entry operation.

Reads the cached GitHub signal snapshot from the unified state and renders
a human-readable repository and CI/CD health summary.
"""

from __future__ import annotations

from kernel.models import OperationExecutionContext, OperationResult, UnifiedState
from lisa_pack_github.models import GitHubStatus


def _render_status(status: GitHubStatus) -> str:
    """Render a GitHubStatus snapshot as a concise health summary."""

    lines: list[str] = [f"## GitHub Status — {status.status.upper()}"]
    lines.append("")

    if status.status == "unconfigured":
        lines.append(
            "GitHub credentials are not configured. "
            "Set `GITHUB_TOKEN` and `GITHUB_REPO` to enable monitoring."
        )
        return "\n".join(lines)

    lines.append(f"- Tracked branch: {status.tracked_branch or 'unknown'}")
    lines.append(f"- Latest commit: "
                 f"{status.latest_commit_sha[:7] if status.latest_commit_sha else 'none'}"
                 f" — {status.latest_commit_message[:80]}")
    lines.append(f"- Commit author: {status.latest_commit_author or 'unknown'}")
    lines.append(f"- CI workflow: {status.main_workflow_status}")

    if status.main_workflow_status == "failure":
        lines.append("- ⚠️ Main branch CI is FAILING.")

    lines.append(f"- Open PRs: {status.open_prs}")

    if status.failing_checks:
        lines.append(f"- Failing checks: {', '.join(status.failing_checks[:5])}")

    if status.latest_release_tag:
        lines.append(f"- Latest release: {status.latest_release_tag}")

    return "\n".join(lines)


async def execute(ctx: OperationExecutionContext) -> OperationResult:
    """Run the GitHub status operation and return the summary text."""

    state: UnifiedState | None = None
    if ctx.state_provider is not None:
        state = ctx.state_provider()

    if state is None:
        return OperationResult(
            output_text=(
                "⚠️ No cached runtime state available. "
                "The GitHub signal snapshot has not been collected yet."
            )
        )

    snapshot = state.signal_snapshots.get("github")
    if snapshot is None:
        return OperationResult(
            output_text=(
                "⚠️ No GitHub snapshot in cached state. "
                "Ensure the GitHub signal connector is configured and has run at least once."
            )
        )

    try:
        github_status = GitHubStatus.model_validate(snapshot)
    except Exception as exc:  # pylint: disable=broad-except
        return OperationResult(
            output_text=f"⚠️ Failed to parse GitHub snapshot: {type(exc).__name__}: {exc}"
        )

    return OperationResult(output_text=_render_status(github_status))
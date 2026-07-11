"""GitHub repository and workflow signal connector.

Monitors GitHub API for repository health and CI/CD pipeline status.
Polled by a host scheduler.

What it monitors:
- Main branch commit history (latest commit SHA, message, author)
- CI workflow status on main branch (latest run conclusion)
- Open pull requests count on main branch

What signals it emits:
- connector status: 'degraded' if latest workflow conclusion is 'failure', 'ok' otherwise
- latest_commit_sha: short SHA of the latest main commit
- latest_commit_message: commit message (first 80 chars)
- latest_commit_author: commit author name
- main_workflow_status: workflow conclusion (success, failure, skipped, etc.)
- open_prs: count of open pull requests

Use case:
- Quick health check that the repo is active and CI is passing
- Alert on workflow failures to indicate deployment or test issues
- Track code velocity (commit frequency) and PR backlog

API endpoints:
- GET /repos/{repo}/commits/main (retrieve latest main branch commit)
- GET /repos/{repo}/actions/runs (retrieve latest workflow run)
- GET /search/issues (search for open PRs on main)

Timeout: all GitHub API calls use external_signal_timeout_seconds (typically 15s).
"""

from __future__ import annotations

import httpx

from kernel.intake.connectors import SignalCollector
from lisa_pack_github.config import get_github_extension_settings
from lisa_pack_github.models import GitHubStatus


class GitHubSignalCollector(SignalCollector[GitHubStatus]):
    """GitHub repository activity and CI status monitor.

    Singleton connector that polls GitHub API for repo health and workflow status.
    Injected as a dependency into the bottom-half tick, initialized once at app startup.

    Attributes:
        collector_id: 'github' (used to identify this collector in logs and results)
        display_name: 'GitHub' (human-readable name for UI/reports)

    Dependencies:
        - GITHUB_TOKEN (SecretStr env var): personal access token with repo read access
        - GITHUB_REPO (str env var): repository identifier (format: owner/repo)
    """

    collector_id = "github"
    display_name = "GitHub"

    def missing_configuration(self) -> list[str]:
        """Check if required GitHub API credentials are configured.

        Returns:
            List of missing env vars: ['GITHUB_TOKEN'] and/or ['GITHUB_REPO'].
            Empty list if both are set.
        """
        github_settings = get_github_extension_settings()
        missing: list[str] = []
        if not github_settings.token:
            missing.append("GITHUB_TOKEN")
        if not github_settings.repo:
            missing.append("GITHUB_REPO")
        return missing

    def build_unconfigured_snapshot(self, collected_at: str) -> GitHubStatus:
        """Build a snapshot when GitHub API credentials are missing.

        Args:
            collected_at: ISO 8601 timestamp (UTC).

        Returns:
            GitHubStatus with status='unconfigured' and minimal default data.
        """
        return GitHubStatus(
            collected_at=collected_at,
            status="unconfigured",
            main_workflow_status="unconfigured",
            tracked_branch=get_github_extension_settings().branch,
        )

    def build_error_snapshot(self, collected_at: str) -> GitHubStatus:
        """Build a snapshot when collection fails after retries.

        Args:
            collected_at: ISO 8601 timestamp (UTC).

        Returns:
            GitHubStatus with status='error' and minimal default data.
        """
        return GitHubStatus(
            collected_at=collected_at,
            status="error",
            main_workflow_status="error",
            tracked_branch=get_github_extension_settings().branch,
        )

    async def collect(self, client: httpx.AsyncClient, collected_at: str) -> GitHubStatus:
        """Collect GitHub repository and workflow health status.

        Fetches metrics from three GitHub API endpoints:
        1. GET /repos/{repo}/commits/main: latest commit SHA, message, author
        2. GET /repos/{repo}/actions/runs: latest workflow run conclusion and status
        3. GET /search/issues: count of open PRs on main branch

        Status classification:
        - 'degraded' if the latest workflow conclusion is 'failure'
        - 'ok' otherwise

        Args:
            client: httpx.AsyncClient for making API requests.
            collected_at: ISO 8601 UTC timestamp when collection started.

        Returns:
            GitHubStatus with workflow status, commit info, and PR count.

        Raises:
            httpx.HTTPStatusError: If any API call returns non-2xx status.
            httpx.TimeoutException: If any API call exceeds external_signal_timeout_seconds.
            httpx.NetworkError: If network connectivity fails.
        """
        github_settings = get_github_extension_settings()
        headers = {
            "Authorization": f"Bearer {github_settings.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        repo = github_settings.repo

        repo_response = await client.get(
            f"https://api.github.com/repos/{repo}",
            headers=headers,
            timeout=github_settings.timeout_seconds,
        )
        repo_response.raise_for_status()
        repo_data = repo_response.json()

        configured_branch = github_settings.branch
        default_branch = str(repo_data.get("default_branch") or "main")
        branch = configured_branch or default_branch

        commits_response = await client.get(
            f"https://api.github.com/repos/{repo}/commits/{branch}",
            headers=headers,
            timeout=github_settings.timeout_seconds,
        )
        commits_response.raise_for_status()
        commit_data = commits_response.json()
        latest_sha = commit_data.get("sha", "")[:8]
        latest_message = commit_data.get("commit", {}).get("message", "").split("\n")[0][:80]
        latest_author = commit_data.get("commit", {}).get("author", {}).get("name", "")

        runs_response = await client.get(
            f"https://api.github.com/repos/{repo}/actions/runs",
            headers=headers,
            timeout=github_settings.timeout_seconds,
            params={"branch": branch, "per_page": 1},
        )
        runs_response.raise_for_status()
        runs_data = runs_response.json()
        workflow_runs = runs_data.get("workflow_runs") or []
        if not workflow_runs:
            # Fallback to latest run across all branches so CI does not silently read
            # as healthy when the tracked branch has no executions.
            fallback_runs_response = await client.get(
                f"https://api.github.com/repos/{repo}/actions/runs",
                headers=headers,
                timeout=github_settings.timeout_seconds,
                params={"per_page": 1},
            )
            fallback_runs_response.raise_for_status()
            workflow_runs = fallback_runs_response.json().get("workflow_runs") or []

        run = workflow_runs[0] if workflow_runs else {}
        workflow_status = run.get("conclusion") or run.get("status") or "no_runs"

        prs_response = await client.get(
            "https://api.github.com/search/issues",
            headers=headers,
            timeout=github_settings.timeout_seconds,
            params={"q": f"repo:{repo} is:pr is:open", "per_page": 1},
        )
        prs_response.raise_for_status()
        open_prs = prs_response.json().get("total_count", 0)

        degraded_workflow_states = {
            "failure",
            "timed_out",
            "cancelled",
            "action_required",
            "stale",
            "startup_failure",
            "unknown",
            "no_runs",
        }
        status = "degraded" if workflow_status in degraded_workflow_states else "ok"
        return GitHubStatus(
            collected_at=collected_at,
            status=status,
            latest_commit_sha=latest_sha,
            latest_commit_message=latest_message,
            latest_commit_author=latest_author,
            main_workflow_status=workflow_status,
            tracked_branch=branch,
            open_prs=open_prs,
        )


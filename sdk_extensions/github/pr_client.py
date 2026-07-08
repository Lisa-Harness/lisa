"""Write-capable GitHub client for PR creation."""

from __future__ import annotations

import logging

import httpx
from pydantic import BaseModel

from sdk_extensions.github.config import get_github_extension_settings

logger = logging.getLogger(__name__)

PROTECTED_BRANCHES = frozenset({"main", "master", "develop", "prod", "production"})


class PullRequestCreationResult(BaseModel):
    """Result of opening a pull request."""

    pr_number: int
    pr_url: str
    status: str


class GitHubPRClient:
    """Client for performing PR actions against the configured GitHub repo."""

    async def create_pull_request(
        self,
        title: str,
        body: str,
        head_branch: str,
        base_branch: str = "main",
    ) -> PullRequestCreationResult:
        """Create a new pull request on GitHub via REST API."""
        if head_branch.lower() in PROTECTED_BRANCHES:
            raise ValueError(
                f"CRITICAL SAFETY ABORT: Refusing to create a PR where the source branch '{head_branch}' is a protected branch. The source must be a feature branch."
            )

        settings = get_github_extension_settings()

        if not settings.token or not settings.repo:
            raise RuntimeError("GITHUB_TOKEN or GITHUB_REPO not configured.")

        url = f"https://api.github.com/repos/{settings.repo}/pulls"
        headers = {
            "Authorization": f"Bearer {settings.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        payload = {
            "title": title,
            "body": body,
            "head": head_branch,
            "base": base_branch,
        }

        logger.info(
            "Creating PR against %s: '%s' (%s -> %s)",
            settings.repo,
            title,
            head_branch,
            base_branch,
        )

        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                json=payload,
                headers=headers,
                timeout=settings.timeout_seconds,
            )

            if response.status_code == 422:
                # Often means PR already exists, or no commits between head and base
                err_data = response.json()
                logger.error("PR Creation failed (422): %s", err_data)
                raise RuntimeError(
                    f"Failed to create PR: {err_data.get('message', 'Validation failed')}"
                )

            response.raise_for_status()
            data = response.json()

            return PullRequestCreationResult(
                pr_number=data["number"],
                pr_url=data["html_url"],
                status="created",
            )

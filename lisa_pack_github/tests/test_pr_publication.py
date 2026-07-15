"""Tests for artifact-gated GitHub pull request publication."""

from __future__ import annotations

import pytest

from kernel.code_artifacts import CodeChangeArtifact, CodeCheckRecord
from kernel.models import ToolExecutionContext
from lisa_pack_github.pr_client import PullRequestCreationResult
from lisa_pack_github.tools.github_open_pull_request.github_open_pull_request import (
    execute,
)

PR_NUMBER = 42


class ArtifactStore:
    """Small in-memory artifact store for publication-tool tests."""

    def __init__(self, artifact: CodeChangeArtifact) -> None:
        self.artifact = artifact

    async def get(self, artifact_id: str) -> CodeChangeArtifact | None:
        return self.artifact if artifact_id == self.artifact.artifact_id else None

    async def save(self, artifact: CodeChangeArtifact) -> CodeChangeArtifact:
        self.artifact = artifact
        return artifact


def _artifact() -> CodeChangeArtifact:
    return CodeChangeArtifact(
        artifact_id="artifact-1",
        workspace_id="workspace",
        user_id="user",
        user_slug="user",
        surface_type="personal_dm",
        surface_id="surface",
        session_plan_digest="session",
        source_root="source",
        worktree_path="worktree",
        branch="lisa/add-pr-tool",
        revision=2,
        required_check_ids=["tests"],
        checks={
            "tests": CodeCheckRecord(
                check_id="tests",
                revision=2,
                status="passed",
            )
        },
        commit_sha="commit-sha",
        head_sha="commit-sha",
        pushed_sha="commit-sha",
        status="pushed",
    )


def _context(store: ArtifactStore) -> ToolExecutionContext:
    return ToolExecutionContext(
        tool_id="github_open_pull_request",
        workspace_id="workspace",
        user_id="user",
        user_slug="user",
        surface_type="personal_dm",
        surface_id="surface",
        session_plan_digest="session",
        mode_id="coding",
        pack_config={
            "base_branch": "main",
            "allowed_base_branches": ["main"],
            "branch_prefix": "lisa/",
        },
        parameters={
            "artifact_id": "artifact-1",
            "title": "Add artifact-backed PR publication",
            "body": "Publishes the verified artifact.",
        },
        code_change_artifact_store=store,
    )


@pytest.mark.asyncio
async def test_open_pull_request_persists_result_on_checked_pushed_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The tool publishes only a current, owned, pushed artifact."""

    calls: list[dict[str, str]] = []

    class FakeClient:
        async def create_pull_request(
            self,
            title: str,
            body: str,
            head_branch: str,
            base_branch: str,
        ) -> PullRequestCreationResult:
            calls.append(
                {
                    "title": title,
                    "body": body,
                    "head_branch": head_branch,
                    "base_branch": base_branch,
                }
            )
            return PullRequestCreationResult(
                pr_number=PR_NUMBER,
                pr_url=f"https://github.example/owner/repo/pull/{PR_NUMBER}",
                status="created",
            )

    monkeypatch.setattr(
        "lisa_pack_github.tools.github_open_pull_request.github_open_pull_request.GitHubPRClient",
        FakeClient,
    )
    store = ArtifactStore(_artifact())

    result = await execute(_context(store))

    assert result.error == ""
    assert result.output_json["pr_number"] == PR_NUMBER
    assert calls == [
        {
            "title": "Add artifact-backed PR publication",
            "body": "Publishes the verified artifact.",
            "head_branch": "lisa/add-pr-tool",
            "base_branch": "main",
        }
    ]
    assert store.artifact.status == "pr_opened"
    assert store.artifact.pr_number == PR_NUMBER


@pytest.mark.asyncio
async def test_open_pull_request_rejects_artifact_before_push() -> None:
    """No provider call occurs until the local artifact was successfully pushed."""

    artifact = _artifact()
    artifact.status = "committed"
    artifact.pushed_sha = ""
    result = await execute(_context(ArtifactStore(artifact)))

    assert result.error == "Artifact must be pushed before opening a pull request."


@pytest.mark.asyncio
async def test_open_pull_request_leaves_artifact_pushed_when_provider_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A provider failure cannot advance or partially annotate the local artifact."""

    class FailingClient:
        async def create_pull_request(
            self,
            title: str,
            body: str,
            head_branch: str,
            base_branch: str,
        ) -> PullRequestCreationResult:
            _ = (title, body, head_branch, base_branch)
            raise RuntimeError("GitHub unavailable")

    monkeypatch.setattr(
        "lisa_pack_github.tools.github_open_pull_request.github_open_pull_request.GitHubPRClient",
        FailingClient,
    )
    artifact = _artifact()
    store = ArtifactStore(artifact)

    result = await execute(_context(store))

    assert result.error == "GitHub unavailable"
    assert store.artifact.status == "pushed"
    assert store.artifact.pr_number is None
    assert store.artifact.pr_url == ""
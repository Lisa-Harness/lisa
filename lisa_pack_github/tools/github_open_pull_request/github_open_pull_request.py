"""Open a GitHub pull request for one checked, pushed coding artifact."""

from __future__ import annotations

import httpx
from pydantic import BaseModel, Field

from kernel.code_artifacts import CodeChangeArtifact
from kernel.models import ToolExecutionContext, ToolResult
from lisa_pack_github.config import GitHubPackConfig, get_github_pack_config
from lisa_pack_github.pr_client import GitHubPRClient

TOOL_ID = "github_open_pull_request"


class Parameters(BaseModel):
    """Structured request for publishing one existing code-change artifact."""

    artifact_id: str = Field(min_length=1, max_length=100)
    title: str = Field(min_length=1, max_length=240)
    body: str = Field(default="", max_length=20_000)
    base_branch: str = Field(default="", max_length=255)


def _belongs_to_context(artifact: CodeChangeArtifact, ctx: ToolExecutionContext) -> bool:
    """Require the artifact's host identity and session boundary to match exactly."""

    return (
        artifact.workspace_id == ctx.workspace_id
        and artifact.user_id == ctx.user_id
        and artifact.user_slug == ctx.user_slug
        and artifact.surface_type == ctx.surface_type
        and artifact.surface_id == ctx.surface_id
        and artifact.session_plan_digest == ctx.session_plan_digest
    )


def _has_current_required_checks(artifact: CodeChangeArtifact) -> bool:
    """Require all mandatory checks to pass at the artifact's current revision."""

    return all(
        (record := artifact.checks.get(check_id)) is not None
        and record.status == "passed"
        and record.revision == artifact.revision
        for check_id in artifact.required_check_ids
    )


def _branch_is_allowed(artifact: CodeChangeArtifact, config: GitHubPackConfig) -> bool:
    """Accept only a policy-owned non-protected feature branch as PR source."""

    return (
        bool(artifact.branch)
        and artifact.branch.startswith(config.branch_prefix)
        and artifact.branch not in config.protected_branches
    )


def _validate_publication(
    artifact: CodeChangeArtifact,
    *,
    ctx: ToolExecutionContext,
    config: GitHubPackConfig,
    requested_base_branch: str,
) -> str:
    """Validate durable artifact state and return the approved PR base branch."""

    if not _belongs_to_context(artifact, ctx):
        raise ValueError("Artifact does not belong to this workspace, user, or session.")
    if artifact.status != "pushed" or not artifact.pushed_sha:
        raise ValueError("Artifact must be pushed before opening a pull request.")
    if not artifact.commit_sha or artifact.head_sha != artifact.commit_sha:
        raise ValueError("Artifact's recorded head does not match its committed revision.")
    if artifact.pushed_sha != artifact.commit_sha:
        raise ValueError("Artifact's pushed revision does not match its committed revision.")
    if not _has_current_required_checks(artifact):
        raise ValueError("Every required check must pass at the current artifact revision.")
    if not _branch_is_allowed(artifact, config):
        raise ValueError("Artifact branch is not an owned unprotected feature branch.")
    if artifact.pr_url:
        raise ValueError("Artifact already has an opened pull request.")

    base_branch = (requested_base_branch or config.base_branch).strip()
    if base_branch not in config.allowed_base_branches:
        raise ValueError("Requested base branch is not allowed by the GitHub pack policy.")
    return base_branch


async def execute(ctx: ToolExecutionContext) -> ToolResult:
    """Open a pull request after validating durable artifact publication state."""

    params = Parameters.model_validate(ctx.parameters)
    try:
        if "\n" in params.title or "\r" in params.title:
            raise ValueError("Pull request titles must be a single line.")
        config = get_github_pack_config(ctx)
        store = ctx.code_change_artifact_store
        if store is None:
            raise ValueError("No code change artifact store is available.")
        artifact = await store.get(params.artifact_id)
        if artifact is None:
            raise ValueError(f"Unknown code change artifact '{params.artifact_id}'.")
        base_branch = _validate_publication(
            artifact,
            ctx=ctx,
            config=config,
            requested_base_branch=params.base_branch,
        )
        result = await GitHubPRClient().create_pull_request(
            title=params.title,
            body=params.body,
            head_branch=artifact.branch,
            base_branch=base_branch,
        )
        artifact.pr_number = result.pr_number
        artifact.pr_url = result.pr_url
        artifact.status = "pr_opened"
        await store.save(artifact)
    except (httpx.HTTPError, RuntimeError, ValueError) as exc:
        return ToolResult(tool_id=TOOL_ID, error=str(exc))

    return ToolResult(
        tool_id=TOOL_ID,
        output_text=f"Opened pull request #{artifact.pr_number}: {artifact.pr_url}",
        output_json={
            "artifact_id": artifact.artifact_id,
            "pr_number": artifact.pr_number,
            "pr_url": artifact.pr_url,
            "branch": artifact.branch,
            "base_branch": base_branch,
        },
    )
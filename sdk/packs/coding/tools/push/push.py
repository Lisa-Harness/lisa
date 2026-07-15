"""Push an owned checked commit without any force option."""

from __future__ import annotations

from pydantic import BaseModel, Field

from kernel.models import ToolExecutionContext, ToolResult
from sdk.packs.coding.config import get_coding_config
from sdk.packs.coding.workflow import (
    CodingWorkflowError,
    artifact_branch_is_allowed,
    artifact_has_current_required_checks,
    artifact_worktree,
    load_owned_artifact,
    run_git,
)

TOOL_ID = "coding_push"


class Parameters(BaseModel):
    """Structured owned-branch push request."""

    artifact_id: str = Field(min_length=1, max_length=100)


async def execute(ctx: ToolExecutionContext) -> ToolResult:
    """Push the recorded feature branch with no force capability."""

    params = Parameters.model_validate(ctx.parameters)
    try:
        config = get_coding_config(ctx)
        artifact = await load_owned_artifact(
            ctx,
            artifact_id=params.artifact_id,
            config=config,
        )
        if not artifact.commit_sha:
            raise CodingWorkflowError("Artifact has no committed revision to push.")
        if not artifact_branch_is_allowed(artifact, config):
            raise CodingWorkflowError("Artifact branch is not an owned unprotected feature branch.")
        if not artifact_has_current_required_checks(artifact):
            raise CodingWorkflowError(
                "Every required check must pass at the current artifact revision."
            )
        worktree = artifact_worktree(artifact, config=config)
        current_head = await run_git(worktree, ("rev-parse", "HEAD"), config=config)
        if current_head.return_code != 0 or current_head.stdout.strip() != artifact.commit_sha:
            raise CodingWorkflowError("Owned worktree HEAD no longer matches the recorded commit.")
        pushed = await run_git(
            worktree,
            ("push", "--set-upstream", config.push_remote, artifact.branch),
            config=config,
        )
        if pushed.return_code != 0:
            raise CodingWorkflowError(pushed.stderr.strip() or "git push failed")
        artifact.pushed_sha = artifact.commit_sha
        artifact.status = "pushed"
        await ctx.code_change_artifact_store.save(artifact)
    except (CodingWorkflowError, OSError, ValueError) as exc:
        return ToolResult(tool_id=TOOL_ID, error=str(exc))

    return ToolResult(
        tool_id=TOOL_ID,
        output_text=f"Pushed {artifact.branch} at {artifact.pushed_sha}.",
        output_json={
            "artifact_id": artifact.artifact_id,
            "branch": artifact.branch,
            "pushed_sha": artifact.pushed_sha,
            "remote": config.push_remote,
        },
    )

"""Commit an owned worktree only after current required checks pass."""

from __future__ import annotations

from pydantic import BaseModel, Field

from kernel.models import ToolExecutionContext, ToolResult
from sdk.packs.coding.config import get_coding_config
from sdk.packs.coding.workflow import (
    CodingWorkflowError,
    artifact_branch_is_allowed,
    artifact_has_current_required_checks,
    artifact_worktree,
    changed_paths,
    load_owned_artifact,
    run_git,
)

TOOL_ID = "coding_commit"


class Parameters(BaseModel):
    """Structured commit request."""

    artifact_id: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=240)


async def execute(ctx: ToolExecutionContext) -> ToolResult:
    """Stage explicit changed paths and commit only a current checked revision."""

    params = Parameters.model_validate(ctx.parameters)
    try:
        if "\n" in params.message or "\r" in params.message:
            raise CodingWorkflowError("Commit messages must be a single line.")
        config = get_coding_config(ctx)
        artifact = await load_owned_artifact(
            ctx,
            artifact_id=params.artifact_id,
            config=config,
        )
        if not artifact_branch_is_allowed(artifact, config):
            raise CodingWorkflowError("Artifact branch is not an owned unprotected feature branch.")
        if not artifact_has_current_required_checks(artifact):
            raise CodingWorkflowError(
                "Every required check must pass at the current artifact revision."
            )
        worktree = artifact_worktree(artifact, config=config)
        current_branch = await run_git(worktree, ("branch", "--show-current"), config=config)
        if current_branch.return_code != 0 or current_branch.stdout.strip() != artifact.branch:
            raise CodingWorkflowError("Owned worktree is not on the recorded feature branch.")
        paths = await changed_paths(worktree, config=config)
        if not paths:
            raise CodingWorkflowError("Owned worktree has no changes to commit.")
        staged = await run_git(worktree, ("add", "--", *paths), config=config)
        if staged.return_code != 0:
            raise CodingWorkflowError(staged.stderr.strip() or "git add failed")
        committed = await run_git(worktree, ("commit", "-m", params.message), config=config)
        if committed.return_code != 0:
            raise CodingWorkflowError(committed.stderr.strip() or "git commit failed")
        head = await run_git(worktree, ("rev-parse", "HEAD"), config=config)
        if head.return_code != 0:
            raise CodingWorkflowError(head.stderr.strip() or "Cannot resolve committed HEAD.")
        artifact.commit_sha = head.stdout.strip()
        artifact.head_sha = artifact.commit_sha
        artifact.status = "committed"
        await ctx.code_change_artifact_store.save(artifact)
    except (CodingWorkflowError, OSError, ValueError) as exc:
        return ToolResult(tool_id=TOOL_ID, error=str(exc))

    return ToolResult(
        tool_id=TOOL_ID,
        output_text=f"Committed revision {artifact.revision} as {artifact.commit_sha}.",
        output_json={
            "artifact_id": artifact.artifact_id,
            "commit_sha": artifact.commit_sha,
            "branch": artifact.branch,
            "revision": artifact.revision,
        },
    )

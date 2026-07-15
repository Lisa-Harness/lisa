"""Inspect the durable and Git status of an owned coding worktree."""

from __future__ import annotations

from pydantic import BaseModel, Field

from kernel.models import ToolExecutionContext, ToolResult
from sdk.packs.coding.config import get_coding_config
from sdk.packs.coding.workflow import (
    CodingWorkflowError,
    artifact_worktree,
    load_owned_artifact,
    run_git,
)

TOOL_ID = "coding_status"


class Parameters(BaseModel):
    """Structured artifact status request."""

    artifact_id: str = Field(min_length=1, max_length=100)


async def execute(ctx: ToolExecutionContext) -> ToolResult:
    """Report Git and check state without changing the worktree."""

    params = Parameters.model_validate(ctx.parameters)
    try:
        config = get_coding_config(ctx)
        artifact = await load_owned_artifact(
            ctx,
            artifact_id=params.artifact_id,
            config=config,
        )
        worktree = artifact_worktree(artifact, config=config)
        status = await run_git(worktree, ("status", "--short"), config=config)
        diff_stat = await run_git(worktree, ("diff", "--stat"), config=config)
        if status.return_code != 0 or diff_stat.return_code != 0:
            raise CodingWorkflowError(
                status.stderr.strip() or diff_stat.stderr.strip() or "git status failed"
            )
    except (CodingWorkflowError, OSError, ValueError) as exc:
        return ToolResult(tool_id=TOOL_ID, error=str(exc))

    return ToolResult(
        tool_id=TOOL_ID,
        output_text=(
            f"Artifact {artifact.artifact_id} is {artifact.status} at revision "
            f"{artifact.revision}."
        ),
        output_json={
            "artifact_id": artifact.artifact_id,
            "status": artifact.status,
            "revision": artifact.revision,
            "branch": artifact.branch,
            "changed_paths": artifact.changed_paths,
            "checks": {
                key: value.model_dump(mode="json") for key, value in artifact.checks.items()
            },
            "git_status": status.stdout[: config.max_command_output_bytes],
            "git_diff_stat": diff_stat.stdout[: config.max_command_output_bytes],
        },
    )

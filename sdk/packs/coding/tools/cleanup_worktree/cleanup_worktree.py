"""Explicitly remove an owned clean coding worktree without force flags."""

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

TOOL_ID = "coding_cleanup_worktree"


class Parameters(BaseModel):
    """Structured worktree cleanup request."""

    artifact_id: str = Field(min_length=1, max_length=100)


async def execute(ctx: ToolExecutionContext) -> ToolResult:
    """Remove only the artifact's owned worktree, never a source checkout."""

    params = Parameters.model_validate(ctx.parameters)
    try:
        config = get_coding_config(ctx)
        artifact = await load_owned_artifact(
            ctx,
            artifact_id=params.artifact_id,
            config=config,
        )
        worktree = artifact_worktree(artifact, config=config)
        source_root = config.resolve_workspace_root()
        removed = await run_git(
            source_root,
            ("worktree", "remove", str(worktree)),
            config=config,
        )
        if removed.return_code != 0:
            raise CodingWorkflowError(
                removed.stderr.strip()
                or "git worktree remove failed; commit or revert changes first"
            )
        artifact.status = "cleaned"
        await ctx.code_change_artifact_store.save(artifact)
    except (CodingWorkflowError, OSError, ValueError) as exc:
        return ToolResult(tool_id=TOOL_ID, error=str(exc))

    return ToolResult(
        tool_id=TOOL_ID,
        output_text=f"Cleaned owned worktree for artifact {artifact.artifact_id}.",
        output_json={"artifact_id": artifact.artifact_id, "branch": artifact.branch},
    )

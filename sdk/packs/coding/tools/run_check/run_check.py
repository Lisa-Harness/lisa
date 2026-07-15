"""Run one configured finite check in an owned coding worktree."""

from __future__ import annotations

import time
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from kernel.code_artifacts import CodeCheckRecord, CodeCheckStatus
from kernel.models import ToolExecutionContext, ToolResult
from sdk.packs.coding.config import get_coding_config
from sdk.packs.coding.workflow import (
    CodingWorkflowError,
    artifact_worktree,
    load_owned_artifact,
    run_process,
)

TOOL_ID = "coding_run_check"


class Parameters(BaseModel):
    """Structured named-check invocation."""

    artifact_id: str = Field(min_length=1, max_length=100)
    check_id: str = Field(min_length=1, max_length=80)


async def execute(ctx: ToolExecutionContext) -> ToolResult:
    """Run only a session-configured command vector and persist its revision result."""

    params = Parameters.model_validate(ctx.parameters)
    try:
        config = get_coding_config(ctx)
        check = next((item for item in config.checks if item.id == params.check_id), None)
        if check is None:
            raise CodingWorkflowError(f"Unknown configured check '{params.check_id}'.")
        artifact = await load_owned_artifact(
            ctx,
            artifact_id=params.artifact_id,
            config=config,
        )
        worktree = artifact_worktree(artifact, config=config)
        started = time.perf_counter()
        result = await run_process(
            check.argv,
            cwd=worktree,
            timeout_seconds=check.timeout_seconds,
            max_output_bytes=config.max_command_output_bytes,
        )
        duration_ms = int((time.perf_counter() - started) * 1000)
        status: CodeCheckStatus = (
            "timed_out" if result.timed_out else "passed" if result.return_code == 0 else "failed"
        )
        artifact.checks[check.id] = CodeCheckRecord(
            check_id=check.id,
            revision=artifact.revision,
            status=status,
            argv=list(check.argv),
            duration_ms=duration_ms,
            stdout=result.stdout,
            stderr=result.stderr,
            completed_at=datetime.now(UTC),
        )
        artifact.status = "checked"
        await ctx.code_change_artifact_store.save(artifact)
    except (CodingWorkflowError, OSError, ValueError) as exc:
        return ToolResult(tool_id=TOOL_ID, error=str(exc))

    return ToolResult(
        tool_id=TOOL_ID,
        output_text=f"Check {check.id} {status} for revision {artifact.revision}.",
        output_json={
            "artifact_id": artifact.artifact_id,
            "check_id": check.id,
            "status": status,
            "revision": artifact.revision,
            "duration_ms": duration_ms,
            "output_truncated": result.output_truncated,
        },
        error=("Check timed out." if status == "timed_out" else ""),
    )

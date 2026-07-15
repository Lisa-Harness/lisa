"""Read one bounded excerpt from the configured GitHub repository."""

from __future__ import annotations

import httpx
from pydantic import BaseModel, Field

from kernel.models import ToolExecutionContext, ToolResult
from lisa_pack_github.client import GitHubRepoLookupError, is_configured, read_file_excerpt

TOOL_ID = "repo_read_file"


class Parameters(BaseModel):
    """Structured bounded file-excerpt request."""

    path: str = Field(min_length=1)
    ref: str = ""
    start_line: int = Field(default=1, ge=1)
    end_line: int = Field(default=80, ge=1)
    anchor_line: int = Field(default=0, ge=0)
    context_radius: int = Field(default=20, ge=0)


async def execute(ctx: ToolExecutionContext) -> ToolResult:
    """Read one text file excerpt by requested Git ref."""

    params = Parameters.model_validate(ctx.parameters)
    if not is_configured():
        return ToolResult(
            tool_id=TOOL_ID,
            error="GitHub repo file reads are not configured. Set GITHUB_TOKEN and GITHUB_REPO.",
        )
    try:
        response = await read_file_excerpt(
            params.path,
            ref=params.ref,
            start_line=params.start_line,
            end_line=params.end_line,
            anchor_line=params.anchor_line,
            context_radius=params.context_radius,
        )
    except (GitHubRepoLookupError, httpx.HTTPError) as exc:
        return ToolResult(tool_id=TOOL_ID, error=str(exc))

    return ToolResult(
        tool_id=TOOL_ID,
        output_text=(
            f"Read {response.path}:{response.start_line}-{response.end_line} "
            f"at {response.ref}."
        ),
        output_json=response.model_dump(mode="json"),
    )
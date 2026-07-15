"""Build exact repository evidence from verified GitHub content at one ref."""

from __future__ import annotations

import uuid

import httpx
from pydantic import BaseModel, Field

from kernel.models import ToolExecutionContext, ToolResult
from lisa_pack_github.client import GitHubRepoLookupError, is_configured, lookup_repo

TOOL_ID = "repo_lookup"


class Parameters(BaseModel):
    """Structured request for exact, target-ref-grounded repository evidence."""

    query: str = ""
    path_hints: list[str] = Field(default_factory=list)
    max_results: int = Field(default=4, ge=1, le=8)
    ref: str = ""


async def execute(ctx: ToolExecutionContext) -> ToolResult:
    """Search candidates and return only evidence confirmed at the target ref."""

    params = Parameters.model_validate(ctx.parameters)
    query = (params.query or ctx.query).strip()
    if not query:
        return ToolResult(tool_id=TOOL_ID, error="Repo lookup requires a query.")
    if not is_configured():
        return ToolResult(
            tool_id=TOOL_ID,
            error="GitHub repo lookup is not configured. Set GITHUB_TOKEN and GITHUB_REPO.",
        )
    try:
        response = await lookup_repo(
            query,
            path_hints=params.path_hints,
            max_results=params.max_results,
            ref=params.ref,
            request_id=f"tool-{uuid.uuid4()}",
        )
    except (GitHubRepoLookupError, httpx.HTTPError) as exc:
        return ToolResult(tool_id=TOOL_ID, error=str(exc))

    return ToolResult(
        tool_id=TOOL_ID,
        output_text=response.answer_summary,
        output_json=response.model_dump(mode="json"),
    )
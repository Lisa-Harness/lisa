"""Generic ``web_search`` contract stub for the bundled core pack.

The core pack owns the *contract* for ``web_search`` but ships no public web
provider implementation. A provider pack (``lisa-pack-brave``,
``lisa-pack-tavily``, ``lisa-pack-serper``) supplies the real implementation,
and the session plan selects it via ``providers.tools.web_search``.

The stub is load-bearing even when a provider is installed: it is the
catalog entry that keeps ``tool_ids: [web_search]`` resolvable in sessions
with no provider pack, where the kernel's operation validation would
otherwise raise on an unknown tool id.
"""

from __future__ import annotations

from kernel.models import ToolExecutionContext, ToolResult
from pydantic import BaseModel, Field

__all__ = ["Parameters", "execute", "is_configured", "search"]

_UNAVAILABLE_MESSAGE = (
    "Web search provider is unavailable. Install a provider pack "
    "(lisa-pack-brave, lisa-pack-tavily, or lisa-pack-serper) and select it with "
    "`providers.tools.web_search`."
)


class Parameters(BaseModel):
    """Structured input parameters for the web_search tool."""

    query: str = Field(..., description="The search query to execute.")
    count: int = Field(default=5, ge=1, le=20, description="Max results per query.")


def is_configured() -> bool:
    """The bundled contract stub is intentionally unavailable by default."""
    return False


async def search(query: str) -> dict[str, object]:
    """Fail with a deterministic provider-not-installed message."""
    _ = query
    raise RuntimeError(_UNAVAILABLE_MESSAGE)


async def execute(ctx: ToolExecutionContext) -> ToolResult:
    """Execute the web_search tool — called by the tool registry dispatcher."""
    from pydantic import ValidationError

    try:
        params = Parameters.model_validate(ctx.parameters or {})
        query = params.query.strip()
    except ValidationError:
        query = ""
    if not query:
        # Fall back to ctx.query if the model didn't supply a query argument
        query = (ctx.query or "").strip()
    if not query:
        return ToolResult(tool_id=ctx.tool_id or "web_search", error="web_search requires a query.")

    return ToolResult(
        tool_id=ctx.tool_id or "web_search",
        error=_UNAVAILABLE_MESSAGE,
    )

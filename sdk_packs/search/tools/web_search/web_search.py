"""Brave Web Search tool — pack-contributed internal tool.

Wraps the Brave Search API as an internal tool discovered from the search
pack's ``tools/`` directory. The kernel imports this module via
``module_path`` and dispatches calls through ``execute(ctx)``.

Configuration is read directly from environment variables (not
``sdk.config``) so the pack has zero SDK config linkage and is
self-contained for externalization.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime

import httpx
from pydantic import BaseModel, Field

from kernel.models import ToolExecutionContext, ToolResult
from sdk_packs.search.models import WebSearchResponse, WebSearchResult

__all__ = ["Parameters", "execute", "is_configured", "search"]

_DEFAULT_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
_DEFAULT_COUNT = 5
_DEFAULT_MARKET = "US"
_SEARCH_TIMEOUT_SECONDS: float = 10.0


class Parameters(BaseModel):
    """Structured input parameters for the web_search tool."""

    query: str = Field(..., description="The search query to execute.")
    count: int = Field(default=5, ge=1, le=20, description="Max results per query.")


def _get_api_key() -> str:
    return os.environ.get("WEB_SEARCH_API_KEY", "").strip()


def _get_endpoint() -> str:
    return os.environ.get("WEB_SEARCH_ENDPOINT", _DEFAULT_ENDPOINT).strip() or _DEFAULT_ENDPOINT


def _get_count() -> int:
    raw = os.environ.get("WEB_SEARCH_COUNT", "").strip()
    if not raw:
        return _DEFAULT_COUNT
    try:
        return max(1, min(20, int(raw)))
    except ValueError:
        return _DEFAULT_COUNT


def _get_market() -> str:
    return os.environ.get("WEB_SEARCH_MARKET", _DEFAULT_MARKET).strip() or _DEFAULT_MARKET


def is_configured() -> bool:
    """Return True when a web search API key is present in the environment."""
    return bool(_get_api_key())


async def search(query: str, client: httpx.AsyncClient | None = None) -> WebSearchResponse:
    """Execute one Brave Web Search query and return structured results.

    Raises:
        RuntimeError: when WEB_SEARCH_API_KEY is not configured.
        httpx.HTTPStatusError: when the search API returns a non-2xx response.
    """
    api_key = _get_api_key()
    if not api_key:
        raise RuntimeError("Web search is not configured. Set WEB_SEARCH_API_KEY to enable it.")

    params: dict[str, str] = {
        "q": query,
        "count": str(_get_count()),
        "country": _get_market(),
        "text_decorations": "false",
        "result_filter": "web",
    }
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key,
    }

    async def _do_request(c: httpx.AsyncClient) -> WebSearchResponse:
        response = await c.get(
            _get_endpoint(),
            params=params,
            headers=headers,
            timeout=_SEARCH_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return _parse_response(query, response.json())

    if client is not None:
        return await _do_request(client)

    async with httpx.AsyncClient() as managed_client:
        return await _do_request(managed_client)


def _parse_response(query: str, data: dict) -> WebSearchResponse:
    web = data.get("web", {})
    raw_values: list[dict] = web.get("results", [])

    results: list[WebSearchResult] = []
    for item in raw_values:
        url = item.get("url", "")
        if not url:
            continue
        profile = item.get("profile", {})
        results.append(
            WebSearchResult(
                name=item.get("title", ""),
                url=url,
                snippet=item.get("description", ""),
                display_url=profile.get("url", "") or url,
            )
        )

    return WebSearchResponse(
        query=query,
        generated_at=datetime.now(UTC).isoformat(),
        results=results,
    )


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

    if not is_configured():
        return ToolResult(
            tool_id=ctx.tool_id or "web_search",
            error="Web search is not configured. Set WEB_SEARCH_API_KEY to enable it.",
        )

    try:
        response = await search(query)
    except Exception as exc:  # pylint: disable=broad-except
        return ToolResult(
            tool_id=ctx.tool_id or "web_search",
            error=f"Web search failed: {type(exc).__name__}: {exc}",
        )

    return ToolResult(
        tool_id=ctx.tool_id or "web_search",
        output_text=render_search_results(response),
        output_json={"response": response.model_dump(mode="json")},
    )


def render_search_results(response: WebSearchResponse) -> str:
    """Render a ``WebSearchResponse`` as a text block for model consumption."""
    if not response.results:
        return f"No web results found for: {response.query}"

    lines: list[str] = []
    for i, result in enumerate(response.results, 1):
        lines.append(f"{i}. {result.name}")
        lines.append(f"   {result.display_url or result.url}")
        if result.snippet:
            lines.append(f"   {result.snippet}")
        lines.append("")
    lines.append(f"Brave Search • {len(response.results)} result(s)")
    return "\n".join(lines)

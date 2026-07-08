"""Brave Web Search integration for framework hosts.

Provides one async helper `search(query)` that calls the Brave Search API and
returns a typed `WebSearchResponse`.

Configuration
-------------
WEB_SEARCH_API_KEY   required; Brave Search subscription token
WEB_SEARCH_ENDPOINT  optional; defaults to Brave web search endpoint
WEB_SEARCH_COUNT     optional; max results per query (default: 5, max: 20)
WEB_SEARCH_MARKET    optional; country code for localization (default: US)
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx

from lisa_sdk.models import WebSearchResponse, WebSearchResult
from lisa_sdk.config import settings

__all__ = [
    "is_configured",
    "render_search_results",
    "search",
]

_SEARCH_TIMEOUT_SECONDS: float = 10.0


def is_configured() -> bool:
    """Return True when a web search API key is present in config."""
    return bool(settings.web_search_api_key.get_secret_value())


async def search(query: str, client: httpx.AsyncClient | None = None) -> WebSearchResponse:
    """Execute one Brave Web Search query and return structured results.

    Raises:
        RuntimeError: when WEB_SEARCH_API_KEY is not configured.
        httpx.HTTPStatusError: when the search API returns a non-2xx response.
    """
    api_key = settings.web_search_api_key.get_secret_value()
    if not api_key:
        raise RuntimeError(
            "Web search is not configured. "
            "Set WEB_SEARCH_API_KEY to enable it."
        )

    params: dict[str, str] = {
        "q": query,
        "count": str(settings.web_search_count),
        "country": settings.web_search_market,
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
            settings.web_search_endpoint,
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


def render_search_results(response: WebSearchResponse) -> str:
    """Render a `WebSearchResponse` as a Teams-friendly text block."""
    if not response.results:
        return f"No web results found for: {response.query}"

    lines = [
        f"Web search results for: **{response.query}**",
        "",
    ]
    for i, result in enumerate(response.results, 1):
        lines.append(f"{i}. **{result.name}**")
        lines.append(f"   {result.display_url or result.url}")
        if result.snippet:
            lines.append(f"   {result.snippet}")
        lines.append("")

    lines.append(f"Brave Search • {len(response.results)} result(s)")
    return "\n".join(lines)

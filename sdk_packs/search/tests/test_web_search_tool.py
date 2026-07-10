"""Tests for the pack-contributed web_search tool's execute(ctx) contract.

Mirrors the deleted Little Lisa host-side tests, but exercises the tool
through the pack's ``execute(ctx)`` function directly — the same entry
point the kernel's tool registry dispatcher calls.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from kernel.models import ToolExecutionContext


@pytest.mark.asyncio
async def test_web_search_returns_friendly_error_when_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sdk_packs.search.tools.web_search import web_search as ws

    monkeypatch.setattr(ws, "is_configured", lambda: False)

    ctx = ToolExecutionContext(
        tool_id="web_search",
        parameters={"query": "cloud cost"},
    )
    result = await ws.execute(ctx)

    assert result.error == "Web search is not configured. Set WEB_SEARCH_API_KEY to enable it."


@pytest.mark.asyncio
async def test_web_search_returns_error_on_empty_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sdk_packs.search.tools.web_search import web_search as ws

    monkeypatch.setattr(ws, "is_configured", lambda: True)

    ctx = ToolExecutionContext(tool_id="web_search", parameters={})
    result = await ws.execute(ctx)

    assert result.error == "web_search requires a query."


@pytest.mark.asyncio
async def test_web_search_returns_grounded_payload_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sdk_packs.search.models import WebSearchResponse, WebSearchResult
    from sdk_packs.search.tools.web_search import web_search as ws

    async def _fake_search(query: str):
        return WebSearchResponse(
            query=query,
            generated_at="2026-05-29T00:00:00+00:00",
            results=[
                WebSearchResult(
                    name="Cloud Cost Guardrails",
                    url="https://example.com/guardrails",
                    snippet="Use budgets and alerts for spend guardrails.",
                    display_url="example.com/guardrails",
                )
            ],
        )

    monkeypatch.setattr(ws, "is_configured", lambda: True)
    monkeypatch.setattr(ws, "search", _fake_search)

    ctx = ToolExecutionContext(
        tool_id="web_search",
        parameters={"query": "cloud cost"},
    )
    result = await ws.execute(ctx)

    assert result.error == ""
    assert result.output_json["response"]["query"] == "cloud cost"
    assert result.output_json["response"]["results"][0]["url"] == "https://example.com/guardrails"
"""Tests for the bundled web_search stub contract.

The bundled search pack keeps the generic ``web_search`` contract but should
return a deterministic provider-unavailable message until an external provider
pack is installed and preferred.
"""

from __future__ import annotations

import pytest

from kernel.models import ToolExecutionContext


@pytest.mark.asyncio
async def test_web_search_returns_friendly_error_when_unconfigured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sdk.packs.search.tools.web_search import web_search as ws

    monkeypatch.setattr(ws, "is_configured", lambda: False)

    ctx = ToolExecutionContext(
        tool_id="web_search",
        parameters={"query": "cloud cost"},
    )
    result = await ws.execute(ctx)

    assert "Install lisa-pack-brave-search" in result.error


@pytest.mark.asyncio
async def test_web_search_returns_error_on_empty_query(
) -> None:
    from sdk.packs.search.tools.web_search import web_search as ws

    ctx = ToolExecutionContext(tool_id="web_search", parameters={})
    result = await ws.execute(ctx)

    assert result.error == "web_search requires a query."

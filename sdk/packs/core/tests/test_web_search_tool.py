"""Tests for the core-pack web_search contract stub's execute(ctx) contract.

The core pack owns the ``web_search`` contract but ships no provider
implementation. These tests pin the stub's behavior: it must always fail
deterministically with a provider-not-installed message, because a silent
success would let a research turn synthesize from nothing.
"""

from __future__ import annotations

import pytest
from kernel.models import ToolExecutionContext


@pytest.mark.asyncio
async def test_web_search_requires_query() -> None:
    """An empty query returns a validation error, not a provider error."""
    from sdk.packs.core.tools.web_search import web_search as ws

    ctx = ToolExecutionContext(tool_id="web_search", parameters={"query": ""})
    result = await ws.execute(ctx)

    assert result.error == "web_search requires a query."


@pytest.mark.asyncio
async def test_web_search_reports_provider_unavailable() -> None:
    """A well-formed query still fails: the stub has no provider behind it."""
    from sdk.packs.core.tools.web_search import web_search as ws

    ctx = ToolExecutionContext(tool_id="web_search", parameters={"query": "lisa harness"})
    result = await ws.execute(ctx)

    assert result.error == ws._UNAVAILABLE_MESSAGE
    assert "provider" in result.error.lower()


@pytest.mark.asyncio
async def test_web_search_falls_back_to_ctx_query() -> None:
    """A model that omits the parameter still gets the provider error."""
    from sdk.packs.core.tools.web_search import web_search as ws

    ctx = ToolExecutionContext(tool_id="web_search", query="fallback query")
    result = await ws.execute(ctx)

    assert result.error == ws._UNAVAILABLE_MESSAGE


def test_web_search_stub_is_never_configured() -> None:
    """``is_configured`` stays False so the catalog never treats this as live."""
    from sdk.packs.core.tools.web_search import web_search as ws

    assert ws.is_configured() is False


@pytest.mark.asyncio
async def test_web_search_search_raises() -> None:
    """The convenience ``search`` helper raises rather than returning empty."""
    from sdk.packs.core.tools.web_search import web_search as ws

    with pytest.raises(RuntimeError, match="provider is unavailable"):
        await ws.search("anything")

"""Tests for the core-pack fetch_page tool's execute(ctx) contract.

Exercises the fetch_page tool through its ``execute(ctx)`` function
directly — the same entry point the kernel's tool registry dispatcher
calls. The fetch itself is the session's ``fetch`` capability, reached
through ``ToolExecutionContext.mcp_tool_call``; tests stand in for it with
a recording fake that returns the ``fetch_url`` JSON contract.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest
from kernel.models import ToolExecutionContext, ToolResult
from sdk.content import FetchStatusError
from sdk.packs.core.tools.fetch_page import fetch_page as fp


class _FakeFetchCapability:
    """Records ``mcp_tool_call`` invocations and replies with one canned result."""

    def __init__(self, *, output: Mapping[str, Any] | None = None, error: str = "") -> None:
        self.output = output
        self.error = error
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def __call__(self, tool_id: str, arguments: Mapping[str, Any]) -> ToolResult:
        self.calls.append((tool_id, dict(arguments)))
        if self.error:
            return ToolResult(tool_id=tool_id, error=self.error)
        return ToolResult(tool_id=tool_id, output_text=json.dumps(self.output))


def _page(url: str, text: str, *, content_type: str = "text/html") -> dict[str, Any]:
    return {
        "url": url,
        "status_code": 200,
        "text": text,
        "content_type": content_type,
        "etag": "",
        "last_modified": "",
        "fetched_at": 0.0,
        "from_cache": False,
    }


def _ctx(
    parameters: dict[str, Any], capability: _FakeFetchCapability | None
) -> ToolExecutionContext:
    return ToolExecutionContext(
        tool_id="fetch_page",
        parameters=parameters,
        mcp_tool_call=capability,
    )


@pytest.mark.asyncio
async def test_fetch_page_rejects_empty_url() -> None:
    """An empty URL parameter returns a friendly validation error."""
    capability = _FakeFetchCapability(output={})
    result = await fp.execute(_ctx({"url": ""}, capability))

    assert result.error == "URL is required."
    assert capability.calls == []


@pytest.mark.asyncio
async def test_fetch_page_rejects_non_http_scheme() -> None:
    """Only http(s) URLs are accepted, before the capability is called."""
    capability = _FakeFetchCapability(output={})
    result = await fp.execute(_ctx({"url": "file:///etc/passwd"}, capability))

    assert result.error.startswith("URL must be http or https")
    assert capability.calls == []


@pytest.mark.asyncio
async def test_fetch_page_delegates_to_fetch_capability(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fetch goes through ``fetch.fetch_url`` with the tool's cache policy."""
    monkeypatch.setenv("FETCH_PAGE_MAX_BYTES", "4096")
    url = "https://example.com/guide"
    html = (
        "<html><head><title>Guardrails Guide</title><style>x{}</style></head>"
        "<body><h1>Use budgets</h1><p>Set spend alerts for guardrails.</p></body></html>"
    )
    capability = _FakeFetchCapability(output=_page(url, html))

    result = await fp.execute(_ctx({"url": url}, capability))

    assert capability.calls == [
        ("fetch.fetch_url", {"url": url, "ttl_seconds": 86_400.0, "max_bytes": 4096})
    ]
    assert result.error == ""
    assert "Use budgets" in result.output_text
    assert "x{}" not in result.output_text
    assert "Source: https://example.com/guide" in result.output_text
    assert result.output_json["page"]["title"] == "Guardrails Guide"
    assert result.output_json["page"]["status_code"] == 200


@pytest.mark.asyncio
async def test_fetch_page_without_mcp_seam_fails_loudly() -> None:
    """A context with no MCP pool is an error, never an in-process fallback."""
    result = await fp.execute(_ctx({"url": "https://example.com/x"}, None))

    assert result.error.startswith("Fetch failed:")
    assert "fetch.fetch_url" in result.error


@pytest.mark.asyncio
async def test_fetch_page_surfaces_transport_error() -> None:
    """A capability error without an HTTP status is reported verbatim."""
    capability = _FakeFetchCapability(
        error="Failed to fetch 'https://example.com/down': ConnectError: boom"
    )
    result = await fp.execute(_ctx({"url": "https://example.com/down"}, capability))

    assert result.error == (
        "Fetch failed: Failed to fetch 'https://example.com/down': ConnectError: boom"
    )


@pytest.mark.asyncio
async def test_fetch_page_rejects_malformed_capability_payload() -> None:
    """A payload outside the fetch_url contract is an error, not empty text."""
    capability = _FakeFetchCapability(output={"unexpected": True})
    result = await fp.execute(_ctx({"url": "https://example.com/x"}, capability))

    assert result.error.startswith("Fetch failed: fetch.fetch_url returned an unexpected payload")


@pytest.mark.asyncio
async def test_fetch_page_honors_per_call_max_chars() -> None:
    """A per-call ``max_chars`` is applied to the extracted text.

    The tool advertises ``max_chars`` up to 20000, so a request above the
    old 12000 ceiling must actually return the requested amount rather than
    being silently truncated.
    """
    url = "https://example.com/long"
    capability = _FakeFetchCapability(output=_page(url, "word " * 6000))  # ~30k chars

    result = await fp.execute(_ctx({"url": url, "max_chars": 20_000}, capability))

    assert result.error == ""
    body = result.output_json["page"]["text"]
    assert 19_000 < len(body) <= 20_000


@pytest.mark.asyncio
async def test_fetch_page_reports_non_text_content() -> None:
    """A PDF (or other binary) is reported as not extractable, not as text."""
    url = "https://example.com/doc.pdf"
    capability = _FakeFetchCapability(
        output=_page(url, "%PDF-1.4\nbinary", content_type="application/pdf")
    )

    result = await fp.execute(_ctx({"url": url}, capability))

    assert result.error == ""
    assert "No extractable text" in result.output_text
    assert "application/pdf" in result.output_text


_BLOCKED = "https://example.com/blocked"


@pytest.mark.parametrize(
    "capability_error",
    [
        # Python lisa-fetch: FetchStatusError message (possibly MCP-prefixed).
        f"Error executing tool fetch_url: {FetchStatusError(_BLOCKED, 403, 'Forbidden')}",
        # Go lisa-fetch: StatusError.Error() format.
        "HTTP 403 (Forbidden) for url 'https://example.com/blocked'",
    ],
)
@pytest.mark.asyncio
async def test_fetch_page_reports_http_error_status(capability_error: str) -> None:
    """A refused fetch reports the HTTP status from either implementation."""
    capability = _FakeFetchCapability(error=capability_error)
    result = await fp.execute(_ctx({"url": "https://example.com/blocked"}, capability))

    assert result.error == (
        "Fetch failed: HTTP 403 for https://example.com/blocked. "
        "The site refused this request; try a different source."
    )

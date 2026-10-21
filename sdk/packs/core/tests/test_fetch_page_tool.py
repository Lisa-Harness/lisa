"""Tests for the core-pack fetch_page tool's execute(ctx) contract.

Exercises the fetch_page tool through its ``execute(ctx)`` function
directly — the same entry point the kernel's tool registry dispatcher
calls.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
from kernel.models import ToolExecutionContext
from sdk.packs.core.tools.fetch_page._models import FetchedPage


@pytest.mark.asyncio
async def test_fetch_page_rejects_empty_url() -> None:
    """An empty URL parameter returns a friendly validation error."""
    from sdk.packs.core.tools.fetch_page import fetch_page as fp

    ctx = ToolExecutionContext(tool_id="fetch_page", parameters={"url": ""})
    result = await fp.execute(ctx)

    assert result.error == "URL is required."


@pytest.mark.asyncio
async def test_fetch_page_rejects_non_http_scheme() -> None:
    """Only http(s) URLs are accepted."""
    from sdk.packs.core.tools.fetch_page import fetch_page as fp

    ctx = ToolExecutionContext(
        tool_id="fetch_page",
        parameters={"url": "file:///etc/passwd"},
    )
    result = await fp.execute(ctx)

    assert result.error.startswith("URL must be http or https")


@pytest.mark.asyncio
async def test_fetch_page_returns_extracted_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A successful fetch returns the stripped page text and structured payload."""
    from typing import Any

    from sdk.packs.core.tools.fetch_page import fetch_page as fp

    async def _fake_fetch(url: str, **_kwargs: Any) -> FetchedPage:
        return FetchedPage(
            url=url,
            status_code=200,
            title="Guardrails Guide",
            text="Use budgets Set spend alerts for guardrails.",
            content_type="text/html",
        )

    monkeypatch.setattr(fp, "fetch", _fake_fetch)

    ctx = ToolExecutionContext(
        tool_id="fetch_page",
        data_dir=str(tmp_path),
        parameters={"url": "https://example.com/guide"},
    )
    result = await fp.execute(ctx)

    assert result.error == ""
    assert "Use budgets" in result.output_text
    assert "Source: https://example.com/guide" in result.output_text
    assert result.output_json["page"]["title"] == "Guardrails Guide"
    assert result.output_json["page"]["status_code"] == 200


@pytest.mark.asyncio
async def test_fetch_page_requires_injected_data_dir() -> None:
    """A tool context without a data_dir fails loudly, never CWD-guesses."""
    from sdk.packs.core.tools.fetch_page import fetch_page as fp

    ctx = ToolExecutionContext(tool_id="fetch_page", parameters={"url": "https://example.com/x"})
    result = await fp.execute(ctx)

    assert result.error.startswith("Fetch failed:")
    assert "no data_dir" in result.error


@pytest.mark.asyncio
async def test_fetch_page_surfaces_transport_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A transport failure is reported as a tool error, not raised."""
    from typing import Any

    from sdk.packs.core.tools.fetch_page import fetch_page as fp

    async def _raise_fetch(url: str, **_kwargs: Any) -> FetchedPage:
        raise httpx.ConnectError("boom")

    monkeypatch.setattr(fp, "fetch", _raise_fetch)

    ctx = ToolExecutionContext(
        tool_id="fetch_page",
        data_dir=str(tmp_path),
        parameters={"url": "https://example.com/down"},
    )
    result = await fp.execute(ctx)

    assert result.error.startswith("Fetch failed:")


@pytest.mark.asyncio
async def test_fetch_page_honors_per_call_max_chars(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A per-call ``max_chars`` is applied to the extracted text.

    The tool advertises ``max_chars`` up to 20000, so a request above the
    old 12000 ceiling must actually return the requested amount rather than
    being silently truncated.
    """
    from typing import Any

    from sdk.packs.core.tools.fetch_page import fetch_page as fp

    long_text = "word " * 6000  # ~30k chars

    async def _fake_fetch(url: str, **kwargs: Any) -> FetchedPage:
        return FetchedPage(
            url=url,
            status_code=200,
            title="Long",
            text=long_text,
            content_type="text/html",
        )

    monkeypatch.setattr(fp, "fetch", _fake_fetch)

    ctx = ToolExecutionContext(
        tool_id="fetch_page",
        data_dir=str(tmp_path),
        parameters={"url": "https://example.com/long", "max_chars": 20_000},
    )
    result = await fp.execute(ctx)

    assert result.error == ""
    body = result.output_json["page"]["text"]
    assert 19_000 < len(body) <= 20_000


@pytest.mark.asyncio
async def test_fetch_page_reports_non_text_content(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A PDF (or other binary) is reported as not extractable, not as text."""
    from typing import Any

    from sdk.packs.core.tools.fetch_page import fetch_page as fp

    async def _fake_fetch(url: str, **kwargs: Any) -> FetchedPage:
        return FetchedPage(
            url=url,
            status_code=200,
            title="",
            text="%PDF-1.4\nbinary",
            content_type="application/pdf",
        )

    monkeypatch.setattr(fp, "fetch", _fake_fetch)

    ctx = ToolExecutionContext(
        tool_id="fetch_page",
        data_dir=str(tmp_path),
        parameters={"url": "https://example.com/doc.pdf"},
    )
    result = await fp.execute(ctx)

    assert result.error == ""
    assert "No extractable text" in result.output_text
    assert "application/pdf" in result.output_text


@pytest.mark.asyncio
async def test_fetch_page_reports_http_error_status(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A refused fetch reports the HTTP status instead of a generic failure."""
    from typing import Any

    from sdk.content import FetchStatusError
    from sdk.packs.core.tools.fetch_page import fetch_page as fp

    async def _raise_fetch(url: str, **kwargs: Any) -> FetchedPage:
        raise FetchStatusError(url, 403, "Forbidden")

    monkeypatch.setattr(fp, "fetch", _raise_fetch)

    ctx = ToolExecutionContext(
        tool_id="fetch_page",
        data_dir=str(tmp_path),
        parameters={"url": "https://example.com/blocked"},
    )
    result = await fp.execute(ctx)

    assert "403" in result.error
    assert "different source" in result.error

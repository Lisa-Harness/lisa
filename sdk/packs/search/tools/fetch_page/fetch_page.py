"""Fetch page tool — pack-contributed internal tool.

Retrieves a URL and extracts its main visible text so the ``web_search``
research loop can synthesize from full page content instead of search
snippets alone.

Configuration is read directly from environment variables (not
``sdk.config``) so the pack has zero SDK config linkage and is
self-contained for externalization, mirroring ``web_search``.

HTML-to-text extraction uses the standard library only — no new heavy
dependency. The extractor strips tags, scripts, styles, and collapses
whitespace, which is sufficient for grounded synthesis. A richer extractor
(trafilatura/readability) can be externalized later without changing the
tool contract.
"""

from __future__ import annotations

import os
import re
from datetime import UTC, datetime
from html import unescape
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, Field

from kernel.models import ToolExecutionContext, ToolResult
from sdk.packs.search.models import FetchedPage

__all__ = ["Parameters", "execute", "fetch", "is_configured"]

_DEFAULT_TIMEOUT_SECONDS: float = 15.0
_DEFAULT_MAX_BYTES: int = 200_000
_MAX_TEXT_CHARS: int = 12_000
_ALLOWED_SCHEMES: tuple[str, ...] = ("http", "https")
_BLOCK_TAGS = re.compile(
    r"<(script|style|noscript|template|svg|head)\b[\s\S]*?</\1\s*>",
    re.IGNORECASE,
)
_TAG_STRIP = re.compile(r"<[^>]+>")
_WS = re.compile(r"[ \t\r\f\v]+")
_BLANK_LINES = re.compile(r"\n{3,}")
_TITLE_TAG = re.compile(r"<title\b[^>]*>([\s\S]*?)</title\s*>", re.IGNORECASE)


class Parameters(BaseModel):
    """Structured input parameters for the fetch_page tool."""

    url: str = Field(..., description="The absolute http(s) URL to fetch.")
    max_chars: int = Field(
        default=_MAX_TEXT_CHARS,
        ge=500,
        le=20000,
        description="Max extracted text characters to return.",
    )


def _get_timeout() -> float:
    raw = os.environ.get("FETCH_PAGE_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return _DEFAULT_TIMEOUT_SECONDS
    try:
        return max(1.0, float(raw))
    except ValueError:
        return _DEFAULT_TIMEOUT_SECONDS


def _get_max_bytes() -> int:
    raw = os.environ.get("FETCH_PAGE_MAX_BYTES", "").strip()
    if not raw:
        return _DEFAULT_MAX_BYTES
    try:
        return max(1024, int(raw))
    except ValueError:
        return _DEFAULT_MAX_BYTES


def is_configured() -> bool:
    """Return True — fetch_page needs no API key, only network access.

    Kept for API symmetry with ``web_search.is_configured`` so the host can
    gate the tool uniformly.
    """
    return True


def _validate_url(url: str) -> str | None:
    """Return a normalized URL string or an error reason."""
    cleaned = (url or "").strip()
    if not cleaned:
        return "fetch_page requires a url."
    parsed = urlparse(cleaned)
    if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
        return f"fetch_page only supports http(s) URLs: {cleaned}"
    if not parsed.netloc:
        return f"fetch_page URL has no host: {cleaned}"
    return None


def _strip_html(html: str, *, max_chars: int) -> tuple[str, str]:
    """Extract visible text and a title from raw HTML.

    Returns ``(text, title)``. The text is collapsed-whitespace plain text
    capped at ``max_chars``.
    """
    title_match = _TITLE_TAG.search(html)
    title = unescape(title_match.group(1)).strip() if title_match else ""

    body = _BLOCK_TAGS.sub("", html)
    text = _TAG_STRIP.sub("", body)
    text = unescape(text)
    text = _WS.sub(" ", text)
    text = _BLANK_LINES.sub("\n\n", text)
    text = text.strip()
    if len(text) > max_chars:
        text = text[:max_chars].rsplit(" ", 1)[0] + "\u2026"
    return text, title


async def fetch(
    url: str,
    *,
    client: httpx.AsyncClient | None = None,
) -> FetchedPage:
    """Fetch one URL and return extracted text.

    Raises:
        httpx.HTTPError: on transport or status failure.
    """
    headers = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Encoding": "gzip",
    }

    async def _do_request(c: httpx.AsyncClient) -> FetchedPage:
        response = await c.get(url, headers=headers, timeout=_get_timeout())
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        raw = response.content[: _get_max_bytes()]
        body = raw.decode(response.encoding or "utf-8", errors="replace")
        text, title = _strip_html(body, max_chars=_MAX_TEXT_CHARS)
        return FetchedPage(
            url=url,
            status_code=response.status_code,
            title=title,
            text=text,
            content_type=content_type,
        )

    if client is not None:
        return await _do_request(client)

    async with httpx.AsyncClient() as managed_client:
        return await _do_request(managed_client)


async def execute(ctx: ToolExecutionContext) -> ToolResult:
    """Execute the fetch_page tool — called by the tool registry dispatcher."""
    from pydantic import ValidationError

    try:
        params = Parameters.model_validate(ctx.parameters or {})
        url = params.url
        max_chars = params.max_chars
    except ValidationError:
        url = (ctx.parameters or {}).get("url", "")
        max_chars = _MAX_TEXT_CHARS

    reason = _validate_url(url)
    if reason:
        return ToolResult(tool_id=ctx.tool_id or "fetch_page", error=reason)

    try:
        page = await fetch(url)
    except Exception as exc:  # pylint: disable=broad-except
        return ToolResult(
            tool_id=ctx.tool_id or "fetch_page",
            error=f"Fetch failed: {type(exc).__name__}: {exc}",
        )

    if not page.text:
        return ToolResult(
            tool_id=ctx.tool_id or "fetch_page",
            output_text=(
                f"No extractable text at {url} "
                f"(status {page.status_code}, type {page.content_type})."
            ),
            output_json={"page": page.model_dump(mode="json")},
        )

    # Honor the per-call max_chars even if the default constant was used.
    if len(page.text) > max_chars:
        page = page.model_copy(update={"text": page.text[:max_chars].rsplit(" ", 1)[0] + "\u2026"})

    timestamp = datetime.now(UTC).isoformat()
    header = f"# {page.title or url}\nSource: {url} \u2022 {page.status_code} \u2022 {timestamp}\n"
    return ToolResult(
        tool_id=ctx.tool_id or "fetch_page",
        output_text=header + page.text,
        output_json={"page": page.model_dump(mode="json")},
    )

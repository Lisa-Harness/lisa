"""Fetch page tool — pack-contributed internal tool.

Retrieves a URL and extracts its main visible text so research loops can
synthesize from full page content instead of search snippets alone.

Configuration is read directly from environment variables (not
``sdk.config``) so the tool has zero SDK config linkage and is
self-contained for externalization.

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
from pathlib import Path

from kernel.models import ToolExecutionContext, ToolResult
from pydantic import BaseModel, Field
from sdk.content import ContentFetcher, FetchPolicy, validate_url
from sdk.packs.core.tools.fetch_page._models import FetchedPage

__all__ = ["Parameters", "execute", "fetch", "get_fetcher", "is_configured"]

_DEFAULT_TIMEOUT_SECONDS: float = 15.0
_DEFAULT_MAX_BYTES: int = 200_000
_MAX_TEXT_CHARS: int = 12_000
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


def _get_data_dir(ctx: ToolExecutionContext) -> Path:
    """Return the data dir for the fetcher cache from the injected context.

    The host threads its resolved ``data_dir`` onto ``ToolExecutionContext``
    (via ``TurnHooks.data_dir``). There is deliberately NO fallback — a
    CWD-relative ``Path("artifacts")`` resolves differently depending on the
    process working directory and re-nests under ``data_dir/files`` (the
    2026-07-30 ``artifacts/files/artifacts/files`` incident). If the context
    carries no data dir, the tool fails loudly rather than silently writing
    to an ambiguous location.
    """
    raw = (getattr(ctx, "data_dir", "") or "").strip()
    if not raw:
        raise RuntimeError(
            "fetch_page: no data_dir injected on the tool context; the host "
            "must wire a resolved data dir before this tool can cache."
        )
    return Path(raw).expanduser().resolve()


def _build_policy(*, force_refresh: bool = False) -> FetchPolicy:
    """Translate env vars into a :class:`FetchPolicy` for this tool."""
    return FetchPolicy(
        ttl_seconds=86_400.0,  # fetch_page content is research-grade; 24h is fine
        force_refresh=force_refresh,
        max_bytes=_get_max_bytes(),
        cache=True,
        revalidate_etag=True,
        timeout_seconds=_get_timeout(),
    )


# ── Module-level lazy fetcher singleton ─────────────────────────────────────
# Mirrors the ``_reader_instance()`` pattern in ``mcp_server/content/server.py``:
# one ContentFetcher per process, shared across all fetch_page calls so the L1
# cache is reused within a turn and across turns. Tests call ``reset_fetcher``
# for isolation. The cache root is bound to the host's injected data dir at
# first use; a later data dir change rebuilds the singleton.

_fetcher: ContentFetcher | None = None


def get_fetcher(data_dir: Path | None = None) -> ContentFetcher:
    """Return the module-level :class:`ContentFetcher`, creating it lazily.

    ``data_dir`` must come from the caller (the host-injected context). There
    is no fallback — callers that cannot supply one must raise instead of
    guessing a CWD-relative path.
    """
    global _fetcher  # noqa: PLW0603 — module-level singleton seam
    if _fetcher is None:
        if data_dir is None:
            raise RuntimeError(
                "fetch_page: no data_dir provided to get_fetcher; the host must "
                "inject its resolved data dir before the fetcher cache is used."
            )
        _fetcher = ContentFetcher(
            data_dir,
            default_policy=_build_policy(),
        )
    return _fetcher


def reset_fetcher() -> None:
    """Reset the module-level fetcher (for test isolation)."""
    global _fetcher  # noqa: PLW0603 — module-level singleton seam
    _fetcher = None


def is_configured() -> bool:
    """Return True — fetch_page needs no API key, only network access.

    Kept for API symmetry with other tools so the host can gate the tool
    uniformly.
    """
    return True


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
    force_refresh: bool = False,
    fetcher: ContentFetcher | None = None,
    data_dir: Path | None = None,
) -> FetchedPage:
    """Fetch one URL via :class:`ContentFetcher` and return extracted text.

    Raises:
        ValueError: on SSRF rejection or network failure with no cache.
        RuntimeError: if no ``fetcher`` or ``data_dir`` is supplied and the
            module singleton was never initialized with a data dir.
    """
    f = fetcher if fetcher is not None else get_fetcher(data_dir=data_dir)
    result = await f.fetch(url, policy=_build_policy(force_refresh=force_refresh))
    text, title = _strip_html(result.text, max_chars=_MAX_TEXT_CHARS)
    return FetchedPage(
        url=result.url,
        status_code=result.status_code,
        title=title,
        text=text,
        content_type=result.content_type,
    )


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

    # Validate first so SSRF rejections surface as tool errors, not exceptions.
    _, reason = validate_url(url)
    if reason is not None:
        return ToolResult(tool_id=ctx.tool_id or "fetch_page", error=reason)

    try:
        page = await fetch(url, data_dir=_get_data_dir(ctx))
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

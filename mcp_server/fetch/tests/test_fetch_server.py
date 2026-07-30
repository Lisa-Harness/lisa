"""Tests for the lisa-fetch MCP server (thin wrapper over ContentFetcher).

Mirrors ``mcp_server/content/tests/`` — verifies the server delegates to the
SDK fetcher and that init/reset lifecycle works. Transport is not exercised
here; we call the tool functions directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from mcp_server.fetch.server import (
    fetch_url,
    init_fetch,
    invalidate_fetch_cache,
    reset_fetch_state,
)


def test_reset_fetch_state_clears_singleton() -> None:
    reset_fetch_state()
    from mcp_server.fetch.server import _fetcher

    assert _fetcher is None


def test_init_fetch_constructs_fetcher(tmp_path: Path) -> None:
    init_fetch(tmp_path)
    from mcp_server.fetch.server import _fetcher

    assert _fetcher is not None
    assert _fetcher._data_dir == tmp_path  # noqa: SLF001
    reset_fetch_state()


@pytest.mark.asyncio
async def test_fetch_url_delegates_to_fetcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    init_fetch(tmp_path)
    url = "https://github.com/foo.md"

    async def _fake_get(self: httpx.AsyncClient, req_url: str, **kw: Any) -> httpx.Response:
        return httpx.Response(
            200,
            request=httpx.Request("GET", req_url),
            content=b"Hello MCP",
            headers={"content-type": "text/markdown"},
        )

    monkeypatch.setattr(httpx.AsyncClient, "get", _fake_get)
    raw = await fetch_url(url)
    payload = json.loads(raw)
    assert payload["url"] == url
    assert payload["status_code"] == 200
    assert payload["text"] == "Hello MCP"
    assert payload["from_cache"] is False
    reset_fetch_state()


@pytest.mark.asyncio
async def test_fetch_url_ttl_override_hits_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    init_fetch(tmp_path)
    url = "https://github.com/foo.md"
    calls = {"n": 0}

    async def _fake_get(self: httpx.AsyncClient, req_url: str, **kw: Any) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, request=httpx.Request("GET", req_url), content=b"V1")

    monkeypatch.setattr(httpx.AsyncClient, "get", _fake_get)
    await fetch_url(url, ttl_seconds=3600)
    await fetch_url(url, ttl_seconds=3600)
    assert calls["n"] == 1  # second call served from L1
    reset_fetch_state()


@pytest.mark.asyncio
async def test_invalidate_fetch_cache_clears_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    init_fetch(tmp_path)
    url = "https://github.com/foo.md"

    async def _fake_get(self: httpx.AsyncClient, req_url: str, **kw: Any) -> httpx.Response:
        return httpx.Response(200, request=httpx.Request("GET", req_url), content=b"X")

    monkeypatch.setattr(httpx.AsyncClient, "get", _fake_get)
    await fetch_url(url)
    raw = await invalidate_fetch_cache(url)
    assert json.loads(raw)["ok"] is True
    reset_fetch_state()


@pytest.mark.asyncio
async def test_fetch_url_ssrf_rejected(tmp_path: Path) -> None:
    init_fetch(tmp_path)
    with pytest.raises(ValueError, match="private"):
        await fetch_url("https://169.254.169.254/meta")
    reset_fetch_state()
"""Tests for the lisa-fetch MCP server (thin wrapper over ContentFetcher).

Mirrors ``mcp_server/content/tests/`` — verifies the server delegates to the
SDK fetcher via the :class:`FetchDeps` test seam. Transport is not exercised
here; we call the tool functions directly.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
from mcp_server.fetch.server import (
    FetchDeps,
    fetch_url,
    invalidate_fetch_cache,
    set_deps_for_test,
)
from sdk.content import ContentFetcher


def _seed_deps(tmp_path: Path) -> None:
    """Install a fresh fetcher via the test seam (no lifespan needed)."""
    fetcher = ContentFetcher(tmp_path)
    set_deps_for_test(FetchDeps(fetcher=fetcher, default_policy=fetcher._default_policy))  # type: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_fetch_url_delegates_to_fetcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_deps(tmp_path)
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


@pytest.mark.asyncio
async def test_fetch_url_ttl_override_hits_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_deps(tmp_path)
    url = "https://github.com/foo.md"
    calls = {"n": 0}

    async def _fake_get(self: httpx.AsyncClient, req_url: str, **kw: Any) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, request=httpx.Request("GET", req_url), content=b"V1")

    monkeypatch.setattr(httpx.AsyncClient, "get", _fake_get)
    await fetch_url(url, ttl_seconds=3600)
    await fetch_url(url, ttl_seconds=3600)
    assert calls["n"] == 1  # second call served from L1


@pytest.mark.asyncio
async def test_invalidate_fetch_cache_clears_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_deps(tmp_path)
    url = "https://github.com/foo.md"

    async def _fake_get(self: httpx.AsyncClient, req_url: str, **kw: Any) -> httpx.Response:
        return httpx.Response(200, request=httpx.Request("GET", req_url), content=b"X")

    monkeypatch.setattr(httpx.AsyncClient, "get", _fake_get)
    await fetch_url(url)
    raw = await invalidate_fetch_cache(url)
    assert json.loads(raw)["ok"] is True


@pytest.mark.asyncio
async def test_fetch_url_ssrf_rejected(tmp_path: Path) -> None:
    _seed_deps(tmp_path)
    with pytest.raises(ValueError, match="private"):
        await fetch_url("https://169.254.169.254/meta")

"""Tests for the localhost web host server.

Migrated from ``hosts/little_lisa/test_little_lisa_desktop.py``.

The web host architecture changed in the Phase 5 decomposition:
- ``desktop.desktop_url`` → ``hosts.web.server.web_url``
- ``desktop.build_desktop_index_html`` → ``hosts.web.server.build_web_index_html``
- ``desktop._build_desktop_app(callbacks=..., token=...)`` →
  ``hosts.web.server._build_web_app(runtime=..., token=...)``
- Token header ``X-Little-Lisa-Token`` → ``X-Lisa-Token``
- Turn endpoint ``POST /api/chat`` → ``POST /api/turn``
- ``desktop.DesktopCallbacks`` is gone; the web server calls
  ``hosts.core.handle_message(runtime, message, transport)`` with an
  internal ``WebTransport``.

The route tests below construct a minimal ``HostRuntime`` and monkeypatch
``handle_message`` so they exercise the aiohttp wiring without starting
the MCP pool or hitting a real model.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from aiohttp.test_utils import TestClient, TestServer

from hosts.core.config import HostConfig
from hosts.core.runtime import HostRuntime, TurnContext
from hosts.web import server as web_server


def test_web_url_includes_host_port_and_token() -> None:
    url = web_server.web_url(host="127.0.0.1", port=8765, token="abc123")

    assert url == "http://127.0.0.1:8765/?token=abc123"


def test_build_web_index_html_embeds_api_token() -> None:
    html = web_server.build_web_index_html(token="token-xyz")

    assert "Lisa Web" in html
    assert "token-xyz" in html
    assert "/api/turn" in html


def test_build_web_index_html_escapes_token() -> None:
    html = web_server.build_web_index_html(token='token"<&')

    assert 'token: "token&quot;&lt;&amp;"' in html


async def _make_client(
    *,
    tmp_path,
    token: str,
    context: dict[str, Any] | None = None,
    error: Exception | None = None,
    response_text: str = "handled: hello",
) -> TestClient:
    """Build a TestClient with a minimal HostRuntime and mocked handle_message."""
    config = HostConfig.from_env(data_dir=str(tmp_path))
    runtime = HostRuntime(config)
    if context is not None:
        runtime.last_turn_context = TurnContext(**context)

    async def _fake_handle_message(rt, message, transport=None, **kwargs):
        if error is not None:
            raise error
        if transport is not None:
            await transport.on_response(response_text)
        from hosts.core.types import HostCommandResult

        return HostCommandResult(text=response_text)

    # monkeypatch handle_message at the server module's import site
    web_server.handle_message = _fake_handle_message  # type: ignore[assignment]

    app = web_server._build_web_app(runtime=runtime, token=token)
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


@pytest.mark.asyncio
async def test_web_routes_require_token_and_return_context(tmp_path) -> None:
    client = await _make_client(
        tmp_path=tmp_path,
        token="secret",
        context={"mode": "web"} if False else None,
    )
    try:
        unauthorized_index = await client.get("/")
        assert unauthorized_index.status == 401

        unauthorized_context = await client.get("/api/context")
        assert unauthorized_context.status == 401

        health = await client.get("/health")
        assert health.status == 200
        assert await health.json() == {"ok": True}

        authorized_context = await client.get("/api/context", params={"token": "secret"})
        assert authorized_context.status == 200

        authorized_index = await client.get("/", params={"token": "secret"})
        assert authorized_index.status == 200
        assert "Lisa Web" in await authorized_index.text()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_turn_endpoint_validates_and_returns_payload(tmp_path) -> None:
    client = await _make_client(tmp_path=tmp_path, token="secret", response_text="handled: hello")
    try:
        unauthorized = await client.post("/api/turn", json={"message": "hello"})
        assert unauthorized.status == 401

        invalid_json = await client.post(
            "/api/turn",
            data="{",
            headers={
                "X-Lisa-Token": "secret",
                "Content-Type": "application/json",
            },
        )
        assert invalid_json.status == 400
        assert (await invalid_json.json())["error"] == "invalid json body"

        missing_message = await client.post(
            "/api/turn",
            json={},
            headers={"X-Lisa-Token": "secret"},
        )
        assert missing_message.status == 400
        assert (await missing_message.json())["error"] == "message is required"

        success = await client.post(
            "/api/turn",
            json={"message": "hello"},
            headers={"X-Lisa-Token": "secret"},
        )
        assert success.status == 200
        payload = await success.json()
        assert payload["response_text"] == "handled: hello"
        assert "last_turn_context" in payload
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_turn_endpoint_surfaces_runtime_errors(tmp_path) -> None:
    client = await _make_client(
        tmp_path=tmp_path,
        token="secret",
        error=RuntimeError("boom"),
    )
    try:
        failure = await client.post(
            "/api/turn",
            json={"message": "hello"},
            headers={"X-Lisa-Token": "secret"},
        )
        assert failure.status == 500
        assert "RuntimeError: boom" in (await failure.json())["error"]
    finally:
        await client.close()


def test_run_web_server_wires_browser_startup_hook(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    config = HostConfig.from_env(data_dir=str(tmp_path))
    runtime = HostRuntime(config)

    captured: dict[str, Any] = {}
    opened: list[str] = []

    def _fake_run_app(app: object, **kwargs: object) -> None:
        captured["app"] = app
        captured["kwargs"] = kwargs

    def _fake_open(url: str) -> bool:
        opened.append(url)
        return True

    monkeypatch.setattr(web_server.web, "run_app", _fake_run_app)
    monkeypatch.setattr(web_server.webbrowser, "open", _fake_open)

    web_server.run_web_server(
        runtime=runtime,
        host="127.0.0.1",
        port=9000,
        token="secret",
        open_browser=True,
    )

    assert captured["kwargs"] == {"host": "127.0.0.1", "port": 9000, "print": None}

    app = captured["app"]
    on_startup = app.on_startup
    assert len(on_startup) >= 1

    import asyncio

    asyncio.run(on_startup[-1](app))
    assert opened == ["http://127.0.0.1:9000/?token=secret"]
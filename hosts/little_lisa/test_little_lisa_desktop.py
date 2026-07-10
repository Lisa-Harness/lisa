from __future__ import annotations

import asyncio
from typing import Any

import pytest
from aiohttp.test_utils import TestClient, TestServer

from . import desktop


def test_desktop_url_includes_host_port_and_token() -> None:
    url = desktop.desktop_url(host="127.0.0.1", port=8765, token="abc123")

    assert url == "http://127.0.0.1:8765/?token=abc123"


def test_build_desktop_index_html_embeds_api_token() -> None:
    html = desktop.build_desktop_index_html(token="token-xyz")

    assert "Little Lisa Desktop" in html
    assert "token-xyz" in html
    assert "/api/chat" in html


def test_build_desktop_index_html_escapes_token() -> None:
    html = desktop.build_desktop_index_html(token='token"<&')

    assert 'token: "token&quot;&lt;&amp;"' in html


async def _make_client(
    *,
    token: str,
    context: dict[str, Any] | None = None,
    error: Exception | None = None,
) -> tuple[TestClient, list[str]]:
    messages: list[str] = []

    async def _handle_turn(message: str) -> str:
        messages.append(message)
        if error is not None:
            raise error
        return f"handled: {message}"

    def _last_turn_context() -> dict[str, Any] | None:
        return context

    app = desktop._build_desktop_app(
        callbacks=desktop.DesktopCallbacks(
            handle_turn=_handle_turn,
            last_turn_context=_last_turn_context,
        ),
        token=token,
    )
    client = TestClient(TestServer(app))
    await client.start_server()
    return client, messages


@pytest.mark.asyncio
async def test_desktop_routes_require_token_and_return_context() -> None:
    client, _messages = await _make_client(token="secret", context={"mode": "desktop"})
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
        assert await authorized_context.json() == {"mode": "desktop"}

        authorized_index = await client.get("/", params={"token": "secret"})
        assert authorized_index.status == 200
        assert "Little Lisa Desktop" in await authorized_index.text()
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_chat_endpoint_validates_and_returns_turn_payload() -> None:
    client, messages = await _make_client(token="secret", context={"last": "ctx"})
    try:
        unauthorized = await client.post("/api/chat", json={"message": "hello"})
        assert unauthorized.status == 401

        invalid_json = await client.post(
            "/api/chat",
            data="{",
            headers={
                "X-Little-Lisa-Token": "secret",
                "Content-Type": "application/json",
            },
        )
        assert invalid_json.status == 400
        assert (await invalid_json.json())["error"] == "invalid json body"

        missing_message = await client.post(
            "/api/chat",
            json={},
            headers={"X-Little-Lisa-Token": "secret"},
        )
        assert missing_message.status == 400
        assert (await missing_message.json())["error"] == "message is required"

        success = await client.post(
            "/api/chat",
            json={"message": "hello"},
            headers={"X-Little-Lisa-Token": "secret"},
        )
        assert success.status == 200
        assert await success.json() == {
            "response_text": "handled: hello",
            "last_turn_context": {"last": "ctx"},
        }
        assert messages == ["hello"]
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_chat_endpoint_surfaces_runtime_errors() -> None:
    client, _messages = await _make_client(
        token="secret",
        error=RuntimeError("boom"),
    )
    try:
        failure = await client.post(
            "/api/chat",
            json={"message": "hello"},
            headers={"X-Little-Lisa-Token": "secret"},
        )
        assert failure.status == 500
        assert "RuntimeError: boom" in (await failure.json())["error"]
    finally:
        await client.close()


def test_run_desktop_server_wires_browser_startup_hook(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _handle_turn(_message: str) -> str:
        return "ok"

    def _last_turn_context() -> dict[str, Any] | None:
        return {"ok": True}

    callbacks = desktop.DesktopCallbacks(
        handle_turn=_handle_turn,
        last_turn_context=_last_turn_context,
    )

    captured: dict[str, Any] = {}
    opened: list[str] = []

    def _fake_run_app(app: object, **kwargs: object) -> None:
        captured["app"] = app
        captured["kwargs"] = kwargs

    def _fake_open(url: str) -> bool:
        opened.append(url)
        return True

    monkeypatch.setattr(desktop.web, "run_app", _fake_run_app)
    monkeypatch.setattr(desktop.webbrowser, "open", _fake_open)

    desktop.run_desktop_server(
        callbacks=callbacks,
        host="127.0.0.1",
        port=9000,
        token="secret",
        open_browser=True,
    )

    assert captured["kwargs"] == {"host": "127.0.0.1", "port": 9000, "print": None}

    app = captured["app"]
    on_startup = app.on_startup
    assert len(on_startup) >= 1
    asyncio.run(on_startup[-1](app))
    assert opened == ["http://127.0.0.1:9000/?token=secret"]

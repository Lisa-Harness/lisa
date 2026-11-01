"""Contract guards surfaced by source-wide typing and architecture checks."""

from __future__ import annotations

import asyncio
import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from kernel.model_router_discovery import PackModelRouterEntry
from runtime.callbacks import NullTransport
from runtime.gateway_provider import HostGatewayProvider
from runtime.testing import build_test_runtime
from runtime.types import HostCommandResult

from runtime import mcp_pool, turn_hooks_builder


class _Router:
    def choose(self, request: Any) -> None:
        return None


@pytest.mark.parametrize("indirect", [False, True])
def test_router_factory_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, indirect: bool
) -> None:
    runtime = build_test_runtime(tmp_path)
    router = _Router()

    def factory(manifest: Any, plan: Any) -> _Router:
        assert manifest is not None
        assert plan is runtime.session_plan
        return router

    entry = PackModelRouterEntry("test", (lambda: factory) if indirect else factory)
    monkeypatch.setattr(turn_hooks_builder, "discover_model_router_factories", lambda: [entry])
    assert turn_hooks_builder._resolve_router_for_runtime(runtime) is router


@pytest.mark.parametrize("indirect", [False, True])
def test_invalid_router_factory_is_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    indirect: bool,
) -> None:
    runtime = build_test_runtime(tmp_path)
    entry = PackModelRouterEntry(
        "invalid", object if indirect else (lambda manifest, plan: object())
    )
    monkeypatch.setattr(turn_hooks_builder, "discover_model_router_factories", lambda: [entry])
    assert turn_hooks_builder._resolve_router_for_runtime(runtime) is None
    assert "invalid" in caplog.text
    assert "must return" in caplog.text


@pytest.mark.parametrize("endpoint", [None, "https://example.com"])
def test_gateway_endpoint_contract(endpoint: str | None) -> None:
    discovered = SimpleNamespace(direct_base_url_for=lambda provider: endpoint)
    provider = HostGatewayProvider(runtime=object(), discovered_provider=discovered)
    assert provider.direct_base_url_for("test") == endpoint


def test_gateway_invalid_endpoint_raises() -> None:
    discovered = SimpleNamespace(direct_base_url_for=lambda provider: 42)
    provider = HostGatewayProvider(runtime=object(), discovered_provider=discovered)
    with pytest.raises(TypeError, match="Invalid direct base URL"):
        provider.direct_base_url_for("test")


@pytest.mark.parametrize("protocol", [None, "openai", 42])
def test_proxy_protocol_contract(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, protocol: Any
) -> None:
    discovered = SimpleNamespace(proxy_protocol_for=lambda provider: protocol)
    monkeypatch.setattr(mcp_pool, "_active_provider_gateway_provider", lambda runtime: discovered)
    resolve = mcp_pool._proxy_protocol_resolver(object())
    assert resolve("test") == (None if protocol == 42 else protocol)
    if protocol == 42:
        assert "Invalid proxy protocol" in caplog.text


def test_behavioral_slice_without_thread_retracts_old_signal(tmp_path: Path) -> None:
    runtime = build_test_runtime(tmp_path, no_thread=True)
    runtime.session.state_snapshot = {
        "context_items": [{"item_id": f"state:behavioural_signals:{runtime.config.user_slug}"}]
    }
    assert turn_hooks_builder._run_behavioral_signals_slice_impl(runtime) == (0, "", "")
    assert not runtime.session.state_snapshot["context_items"]


@pytest.mark.asyncio
async def test_deferred_start_failure_is_cached_and_reraised(tmp_path: Path) -> None:
    runtime = build_test_runtime(tmp_path)
    failure = RuntimeError("failed MCP startup")

    async def fail() -> None:
        raise failure

    resources = runtime.resources
    resources._mcp_started_event = asyncio.Event()
    resources._mcp_started_event.set()
    resources._mcp_start_task = asyncio.create_task(fail())
    await asyncio.sleep(0)
    for _ in range(2):
        with pytest.raises(RuntimeError) as raised:
            await runtime.ensure_pool_ready()
        assert raised.value is failure
    assert resources._mcp_start_exc is failure


@pytest.mark.asyncio
@pytest.mark.parametrize("structured", [False, True])
async def test_denial_dispatch_preserves_handler_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, structured: bool
) -> None:
    dispatch = importlib.import_module("runtime.handle_message")
    runtime = build_test_runtime(tmp_path)
    transport = NullTransport()
    result = HostCommandResult(text="denied") if structured else "denied"

    async def deny(*args: Any, **kwargs: Any) -> HostCommandResult | str:
        assert args == (runtime, "/deny request")
        assert kwargs["approve"] is False
        assert kwargs["transport"] is transport
        return result

    monkeypatch.setattr(dispatch, "handle_approval_command", deny)
    response, continuation = await dispatch._dispatch_approval_command(
        runtime, "/deny request", "/deny request", transport
    )
    if structured:
        assert response is None
        assert continuation is result
    else:
        assert response == result
        assert continuation is None

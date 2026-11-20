"""FIX-2: explicit call-origin routing metadata on ``CompletionRequest``.

``call_kind`` is vendor-neutral routing metadata stamped at the seam that
owns each model call: ``"turn"`` at the turn loop's gateway request,
``"auxiliary"`` at ``OperationHost`` (off-turn operations: planner, titles,
distillers) and the runtime off-turn/slice completion seam. It never reaches
a provider wire payload and never lands in an archive snapshot — the
request-snapshot builders enumerate their fields — so it exists purely so
a consumer that must distinguish call origins (the replay gateway routing
provider) can route on the recorded origin instead of guessing from the
request's shape (a tools-based heuristic misroutes a zero-tool turn onto
the auxiliary queue and desyncs both archives).
"""

from __future__ import annotations

from typing import Any, ClassVar
from unittest.mock import patch

import pytest
from kernel.models import WorkflowModeConfig, YieldPolicy
from kernel.provider_contracts import CompletionRequest
from kernel.seams.host_adapter import build_callable_host
from kernel.testing.gateway_stub import StubGatewayProvider
from kernel.tools.mcp_client import MCPToolPool
from kernel.turn_hooks import (
    ModelHooks,
    OperationHooks,
    ToolHooks,
    TurnHooks,
    TurnIdentity,
)
from kernel.turn_runner import ContextSeams, run_turn


class _StubCompletion:
    def __init__(self, content: str) -> None:
        self.content = content
        self.tool_calls: list[dict[str, Any]] = []


async def _stub_execute_operation(*_: Any, **__: Any) -> Any:
    class _R:
        output_text = ""
        output_json: ClassVar[dict[str, Any]] = {}

    return _R()


def _model_def() -> Any:
    class _ModelDef:
        model_id = "test-model"
        provider = "test"
        display_name = "Test Model"
        reasoning_model = False
        thinking_model = False

    return _ModelDef()


# ── The field itself ─────────────────────────────────────────────────────────


def test_completion_request_call_kind_defaults_to_unclassified() -> None:
    """A request built without classification stays explicitly unclassified."""
    request = CompletionRequest(
        messages=[{"role": "user", "content": "go"}],
        model_def=_model_def(),
        max_tokens=64,
        temperature=0.0,
    )
    assert request.call_kind == ""


def test_completion_request_accepts_turn_and_auxiliary_kinds() -> None:
    """The two production origins are the only classified values."""
    for kind in ("turn", "auxiliary"):
        request = CompletionRequest(
            messages=[{"role": "user", "content": "go"}],
            model_def=_model_def(),
            max_tokens=64,
            temperature=0.0,
            call_kind=kind,
        )
        assert request.call_kind == kind


# ── Turn-loop seam ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_turn_loop_stamps_turn_kind_on_gateway_requests() -> None:
    """The real turn loop's gateway request carries ``call_kind="turn"``."""

    async def complete_fn(**_kwargs: Any) -> _StubCompletion:
        return _StubCompletion(content="Done.")

    provider = StubGatewayProvider(complete_fn)
    hooks = TurnHooks(
        identity=TurnIdentity(task_id="t", turn_id="turn-1"),
        models=ModelHooks(complete=complete_fn),
        operations=OperationHooks(execute_operation=_stub_execute_operation),
        context_seams=ContextSeams(provider_gateways=provider),
    )
    result = await run_turn(mode_id="default", query="hello", hooks=hooks)
    assert result.response_text == "Done."

    last = provider.last_gateway
    assert last is not None
    request = last.last_request
    assert request is not None
    assert request.call_kind == "turn"


@pytest.mark.asyncio
async def test_turn_loop_zero_tool_request_still_carries_turn_kind() -> None:
    """A mode with an empty tool surface still classifies its calls as turns.

    This is the misroute the tools-based heuristic caused: a zero-tool
    in-turn call is a turn call, and its origin must say so. The minimal
    mode config is authoritative with an empty allowlist (no MCP,
    in-process, or operation tools) and not routable (no
    ``transition_stage``); the yield override keeps ``yield`` off the
    surface — a genuinely zero-tool turn.
    """

    async def complete_fn(**_kwargs: Any) -> _StubCompletion:
        return _StubCompletion(content="Done.")

    provider = StubGatewayProvider(complete_fn)
    # A host-owned (warm) pool routes ``run_turn`` through the path that
    # forwards ``yield_policy_override``; the per-turn-pool path drops it,
    # which would leave ``yield`` advertised and defeat the zero-tool setup.
    pool = MCPToolPool([])
    await pool.start()
    try:
        hooks = TurnHooks(
            identity=TurnIdentity(task_id="t", turn_id="turn-1"),
            models=ModelHooks(complete=complete_fn),
            operations=OperationHooks(execute_operation=_stub_execute_operation),
            context_seams=ContextSeams(provider_gateways=provider),
            tools=ToolHooks(mcp_tool_pool=pool),
        )
        patched_catalog = {"default": WorkflowModeConfig(mode_id="default")}
        with (
            patch("kernel.turn_runner.get_workflow_catalog", return_value=patched_catalog),
            patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=patched_catalog),
            patch("kernel.turn_runner.get_mode_system_intro", lambda mid, _c=patched_catalog: mid),
            patch("kernel.ipsar_fsm.get_mode_system_intro", lambda mid, _c=patched_catalog: mid),
        ):
            result = await run_turn(
                mode_id="default",
                query="hello",
                hooks=hooks,
                yield_policy_override=YieldPolicy(allowed=False),
            )
    finally:
        await pool.aclose()
    assert result.response_text == "Done."

    last = provider.last_gateway
    assert last is not None
    request = last.last_request
    assert request is not None
    assert request.call_kind == "turn"
    assert not request.tools


# ── OperationHost seam ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_operation_host_stamps_auxiliary_kind() -> None:
    """``OperationHost.complete`` classifies its gateway request as auxiliary."""

    async def complete_fn(**_kwargs: Any) -> _StubCompletion:
        return _StubCompletion(content="planned")

    # One provider instance for the whole test: ``gateway_for`` records the
    # last gateway on the provider it is called through, so the factory
    # must return the same instance the host used — a fresh provider per
    # call leaves ``last_gateway`` None on the one inspected here.
    provider = StubGatewayProvider(complete_fn)
    host = build_callable_host(gateway_provider_fn=lambda: provider)
    result = await host.complete(
        messages=[{"role": "user", "content": "plan the turn"}],
        model_def=_model_def(),
        max_tokens=256,
        temperature=0.1,
    )
    assert result.content == "planned"

    last = provider.last_gateway
    assert last is not None
    request = last.last_request
    assert request is not None
    assert request.call_kind == "auxiliary"


@pytest.mark.asyncio
async def test_operation_host_auxiliary_kind_survives_with_tools() -> None:
    """An off-turn call that passes tools is still auxiliary — origin, not shape."""

    async def complete_fn(**_kwargs: Any) -> _StubCompletion:
        return _StubCompletion(content="planned")

    provider = StubGatewayProvider(complete_fn)
    host = build_callable_host(gateway_provider_fn=lambda: provider)
    await host.complete(
        messages=[{"role": "user", "content": "plan the turn"}],
        model_def=_model_def(),
        max_tokens=256,
        temperature=0.1,
        tools=[{"type": "function", "function": {"name": "web_search"}}],
    )
    last = provider.last_gateway
    assert last is not None
    request = last.last_request
    assert request is not None
    assert request.call_kind == "auxiliary"
    assert request.tools


# ── Wire/snapshot isolation ──────────────────────────────────────────────────


def test_call_kind_is_not_in_the_turn_loop_request_snapshot() -> None:
    """The turn loop's archived request snapshot must not carry ``call_kind``.

    The snapshot builder enumerates its fields; routing metadata stays off
    the durable archive so recorded snapshots stay layout-stable.
    """
    import inspect

    from kernel.turn_runner import _record_request_snapshot

    source = inspect.getsource(_record_request_snapshot)
    assert "call_kind" not in source


def test_call_kind_is_not_in_the_host_request_snapshot() -> None:
    """The operation host's archived request snapshot must not carry ``call_kind``."""
    import inspect

    from kernel.seams import host_adapter

    source = inspect.getsource(host_adapter._CallableHost._record_request_snapshot)
    assert "call_kind" not in source

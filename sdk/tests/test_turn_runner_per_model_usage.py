"""Tests for TurnResult.per_model_usage population in kernel.turn_runner.

These tests live in ``sdk/tests/`` because ``kernel/tests/conftest.py`` has an
autouse module-scope fixture that spawns the real ``lisa-catalog`` MCP server
over external packs (``lisa_packs/{search,markdown,letta}``). When
those submodules are not installed, every kernel test errors at setup.
``sdk/tests/`` has no such conftest, so we exercise ``run_turn`` with stubs
and patch ``get_workflow_catalog`` directly.

See ``/memories/repo/kernel-tests-catalog-fixture-requires-external-packs.md``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from kernel.models import WorkflowModeConfig
from kernel.provider_contracts import StreamedChunk
from kernel.testing.gateway_stub import StubGatewayProvider
from kernel.testing.routing_vocabulary import representative_routing_vocabulary
from kernel.turn_hooks import ModelHooks
from kernel.turn_runner import ContextSeams, TurnHooks, TurnResult, run_turn


@pytest.fixture(autouse=True)
def _routing_vocabulary() -> Any:
    with representative_routing_vocabulary():
        yield


async def _no_complete(**_kwargs: Any) -> Any:
    raise AssertionError("non-streaming completion was not expected")


def _hooks(
    *,
    resolve_model: Any,
    complete: Any = None,
    complete_stream: Any = None,
    model_router: Any = None,
) -> TurnHooks:
    """Build hooks whose fakes back the provider-gateway completion path."""
    gateways = StubGatewayProvider(complete or _no_complete, complete_stream_fn=complete_stream)
    return TurnHooks(
        # ``complete_stream`` on ModelHooks is the runner's streaming opt-in.
        models=ModelHooks(resolve_model=resolve_model, complete_stream=complete_stream),
        context_seams=ContextSeams(provider_gateways=gateways, model_router=model_router),
    )


# ── Stubs ─────────────────────────────────────────────────────────────


class _StubModelDef:
    """Minimal model def: the turn loop only reads ``model_id``."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.provider = "test"


class _StubUsage:
    def __init__(self, prompt: int, completion: int) -> None:
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = prompt + completion
        self.cached_tokens = 0
        self.reasoning_tokens = 0


class _StubCompletion:
    """Mimics the completion object the turn loop consumes."""

    def __init__(
        self,
        content: str,
        model_id: str,
        prompt: int,
        completion: int,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls or []
        self.model_id = model_id
        self.usage = _StubUsage(prompt, completion)


def _default_mode_config() -> WorkflowModeConfig:
    return WorkflowModeConfig(
        mode_id="default",
        start_stage_id="none",
        mode_kind="chat",
        routable=False,
        satisfies_intents=[],
        required_capabilities=[],
        routing_intent_tags=[],
        routing_intent_keywords={},
    )


def _patched_catalog():
    """Patch get_workflow_catalog in both turn_runner and ipsar_fsm.

    ``run_turn`` reads the catalog via ``kernel.turn_runner.get_workflow_catalog``
    (re-exported) and via ``kernel.ipsar_fsm.get_workflow_catalog`` inside
    ``run_mode_pre_turn_operations`` / ``run_mode_entry_operations``. Both
    must return the stubbed default-mode config or the lazy catalog loader
    raises because no content MCP config is registered.
    """
    catalog = {"default": _default_mode_config()}
    from contextlib import ExitStack

    stack = ExitStack()
    stack.enter_context(patch("kernel.turn_runner.get_workflow_catalog", return_value=catalog))
    stack.enter_context(patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=catalog))
    return stack


# ── Tests ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_turn_populates_per_model_usage_single_model() -> None:
    """A simple single-model turn produces one per_model_usage entry."""
    completion = _StubCompletion("Hello", "default-model", prompt=100, completion=50)

    async def complete(**kwargs: Any) -> _StubCompletion:
        return completion

    hooks = _hooks(complete=complete, resolve_model=lambda mid: _StubModelDef("default-model"))

    with _patched_catalog():
        result = await run_turn(mode_id="default", query="hello", hooks=hooks)

    assert isinstance(result, TurnResult)
    assert result.prompt_tokens == 100
    assert result.completion_tokens == 50
    assert result.total_tokens == 150
    assert len(result.per_model_usage) == 1
    entry = result.per_model_usage[0]
    assert entry["model_id"] == "default-model"
    assert entry["prompt_tokens"] == 100
    assert entry["completion_tokens"] == 50
    assert entry["total_tokens"] == 150


@pytest.mark.asyncio
async def test_run_turn_records_per_model_usage_for_streaming_path() -> None:
    """The streaming completion path populates per_model_usage too."""

    async def complete_stream(**kwargs: Any):
        yield StreamedChunk(content="Hello", usage=None)
        yield StreamedChunk(
            content="",
            usage=_StubUsage(prompt=60, completion=30),
            finish_reason="stop",
        )

    hooks = _hooks(
        complete_stream=complete_stream,
        resolve_model=lambda mid: _StubModelDef("stream-model"),
    )

    with _patched_catalog():
        result = await run_turn(mode_id="default", query="hello", hooks=hooks)

    assert result.prompt_tokens == 60
    assert result.completion_tokens == 30
    assert len(result.per_model_usage) == 1
    entry = result.per_model_usage[0]
    assert entry["model_id"] == "stream-model"
    assert entry["total_tokens"] == 90


def test_turn_result_per_model_usage_defaults_empty() -> None:
    """TurnResult.per_model_usage defaults to an empty list."""
    result = TurnResult(response_text="ok")
    assert result.per_model_usage == []


def test_turn_result_per_mode_usage_defaults_empty() -> None:
    """TurnResult.per_mode_usage defaults to an empty list."""
    result = TurnResult(response_text="ok")
    assert result.per_mode_usage == []


@pytest.mark.asyncio
async def test_run_turn_populates_per_mode_usage_for_entry_mode() -> None:
    """A single-model turn attributes tokens to the entry mode_id.

    Without per_mode_usage the host's mode_token_totals would attribute the
    whole turn to the entry mode by default — but a kernel-driven recursive
    handoff (plan → search → coding) merges every inner turn's tokens into
    the outer result. per_mode_usage lets the host see the actual mode(s)
    that ran the completions.
    """
    completion = _StubCompletion("Hello", "default-model", prompt=100, completion=50)

    async def complete(**kwargs: Any) -> _StubCompletion:
        return completion

    hooks = _hooks(complete=complete, resolve_model=lambda mid: _StubModelDef("default-model"))

    with _patched_catalog():
        result = await run_turn(mode_id="search", query="hello", hooks=hooks)

    assert isinstance(result, TurnResult)
    assert len(result.per_mode_usage) == 1
    entry = result.per_mode_usage[0]
    assert entry["mode_id"] == "search"
    assert entry["prompt_tokens"] == 100
    assert entry["completion_tokens"] == 50
    assert entry["total_tokens"] == 150


# ── Capability gate vs. router ordering ───────────────────────────────


class _StubModelDefWithCaps:
    """Model def carrying capabilities for gate/router interaction tests."""

    def __init__(self, model_id: str, capabilities: tuple[str, ...]) -> None:
        self.model_id = model_id
        self.provider = "test"
        self.capabilities = capabilities
        self.cost_tier = "standard"
        self.latency_tier = "medium"
        self.reasoning_model = False
        self.thinking_model = False


class _StubRouter:
    """A ModelRouter that escalates to a code-capable model when the mode
    requires the ``code`` capability — mirrors the canonical
    ``"code" in route.required_capabilities → glm-5.2:cloud`` rule."""

    def __init__(self, escalate_to: str = "code-model") -> None:
        self.escalate_to = escalate_to
        self.calls = 0

    def choose(self, req: Any) -> Any:
        from kernel.seams.protocols import RouterDecision

        self.calls += 1
        if "code" in req.required_capabilities:
            return RouterDecision(model_id=self.escalate_to, routed_by="stub-router")
        return None


@pytest.mark.asyncio
async def test_router_escalates_past_capability_gate() -> None:
    """A mode requiring ``code`` with a default model lacking ``code`` must
    NOT be blocked when a router is wired and its rule escalates to a
    code-capable model.

    Regression for the gate-before-router ordering bug: the capability gate
    previously ran on the static preferred model before the router's first
    ``choose()``, hard-blocking turns the router would have rescued. With
    pre-gate router escalation (Option A), the gate validates the
    router-resolved model instead.
    """
    completion = _StubCompletion("done", "code-model", prompt=10, completion=5)

    async def complete(**kwargs: Any) -> _StubCompletion:
        return completion

    # Default model lacks the ``code`` capability the coding mode requires.
    models = {
        "default-model": _StubModelDefWithCaps("default-model", ("text", "tools")),
        "code-model": _StubModelDefWithCaps("code-model", ("text", "tools", "code")),
    }
    router = _StubRouter(escalate_to="code-model")

    def _mode_catalog():
        return {
            "default": _default_mode_config(),
            "coding": WorkflowModeConfig(
                mode_id="coding",
                start_stage_id="none",
                mode_kind="chat",
                routable=False,
                satisfies_intents=["coding"],
                required_capabilities=["tools", "code"],
                routing_intent_tags=[],
                routing_intent_keywords={},
            ),
        }

    from contextlib import ExitStack

    stack = ExitStack()
    stack.enter_context(
        patch("kernel.turn_runner.get_workflow_catalog", return_value=_mode_catalog())
    )
    stack.enter_context(
        patch("kernel.ipsar_fsm.get_workflow_catalog", return_value=_mode_catalog())
    )
    try:
        hooks = _hooks(
            complete=complete,
            resolve_model=lambda mid: models.get(mid, models["default-model"]),
            model_router=router,
        )
        result = await run_turn(
            mode_id="coding",
            query="implement a thing",
            hooks=hooks,
        )
    finally:
        stack.close()

    # The gate must NOT have blocked — the router escalated to code-model
    # before the gate ran, so the turn proceeded.
    assert not result.blocked, f"gate blocked: {result.response_text}"
    # The router was consulted for iteration 1 (pre-gate).
    assert router.calls >= 1
    # The completion ran on the router-escalated model.
    assert result.per_model_usage, "expected per-model usage"
    assert any(e["model_id"] == "code-model" for e in result.per_model_usage)

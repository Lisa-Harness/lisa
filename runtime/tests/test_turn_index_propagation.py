"""Turn-key propagation from the host turn context into kernel telemetry.

The forensic join is only possible if one value — the global thread turn
index — reaches the thread entry, the decision ledger, the dispatch ledger,
and the cost log. ``handle_message`` computes it once at turn start and puts
it on ``run_context``; these tests assert the rest of the chain does not
silently drop it.
"""

from __future__ import annotations

import pytest
from kernel.turn_hooks import TurnIdentity
from kernel.turn_phases._post_turn import _to_tool_context

from runtime import turn_hooks_builder


def test_turn_identity_defaults_turn_index_to_none() -> None:
    """A bare identity claims no turn — not turn 0, which is a real turn."""
    assert TurnIdentity().turn_index is None


@pytest.fixture(scope="module")
def hooks_runtime(tmp_path_factory: pytest.TempPathFactory):
    """A runtime whose content registry is wired, for ``build_turn_hooks``.

    ``build_turn_hooks`` requires the content registry, which
    ``mcp_pool.setup_framework`` installs. Built once per module because the
    setup scans the real catalogs.
    """
    from runtime.testing import build_test_runtime

    from runtime import mcp_pool

    models_yaml = "\n".join(
        [
            "version: 1",
            "mode_models: {default: gemma3:latest}",
            "models:",
            "  - model_id: gemma3:latest",
            "    provider: ollama",
        ]
    )
    runtime = build_test_runtime(tmp_path_factory.mktemp("hooks"), models_yaml=models_yaml)
    mcp_pool.setup_framework(runtime)
    return runtime


def test_build_turn_hooks_threads_turn_index_from_context(hooks_runtime) -> None:
    """``build_turn_hooks`` copies ``turn_index`` off the turn context."""
    hooks = turn_hooks_builder.build_turn_hooks(
        hooks_runtime, {"mode_id": "default", "task_id": "task-7", "turn_index": 31}
    )
    assert hooks.identity.turn_index == 31
    assert hooks.identity.task_id == "task-7"


def test_build_turn_hooks_threads_first_turn_zero(hooks_runtime) -> None:
    """Turn 0 is a real turn index and must not be confused with 'unset'."""
    hooks = turn_hooks_builder.build_turn_hooks(
        hooks_runtime, {"mode_id": "default", "turn_index": 0}
    )
    assert hooks.identity.turn_index == 0


def test_build_turn_hooks_coerces_missing_turn_index(hooks_runtime) -> None:
    """A context without ``turn_index`` is unattributed, not turn 0."""
    hooks = turn_hooks_builder.build_turn_hooks(hooks_runtime, {"mode_id": "default"})
    assert hooks.identity.turn_index is None


def test_build_turn_hooks_coerces_garbage_turn_index(hooks_runtime) -> None:
    """A non-numeric or negative turn index cannot be stamped on telemetry."""
    for raw in ("not-a-number", -3, None, {}):
        hooks = turn_hooks_builder.build_turn_hooks(
            hooks_runtime, {"mode_id": "default", "turn_index": raw}
        )
        assert hooks.identity.turn_index is None


def test_coerce_turn_index_accepts_string_ordinals() -> None:
    """A string ordinal from a JSON-ish context is coerced, not dropped."""
    assert turn_hooks_builder._coerce_turn_index("12") == 12
    assert turn_hooks_builder._coerce_turn_index(0) == 0
    assert turn_hooks_builder._coerce_turn_index("0") == 0
    assert turn_hooks_builder._coerce_turn_index(None) is None


def test_to_tool_context_carries_turn_index() -> None:
    """Nested tool dispatches keep the turn attribution of their parent."""
    from kernel.models import OperationExecutionContext

    ctx = OperationExecutionContext(
        mode_id="coding_execute",
        turn_index=57,
        data_dir="/tmp/data",
    )
    tool_ctx = _to_tool_context("fetch_page", ctx)
    assert tool_ctx.turn_index == 57


def test_to_tool_context_tolerates_missing_turn_index() -> None:
    """A context predating the field stays unattributed rather than raising."""

    class _LegacyCtx:
        data_dir = "/tmp/data"

    tool_ctx = _to_tool_context("fetch_page", _LegacyCtx())
    assert tool_ctx.turn_index is None


@pytest.mark.asyncio
async def test_decision_and_tool_payloads_include_turn_index() -> None:
    """Both published topics carry the turn and owner keys the forensic join needs.

    The ledger is shared by every user of a data root and turn indices are
    per-thread, so a record without its ``user_slug`` cannot be scoped.
    """
    from dataclasses import replace
    from typing import Any

    from kernel.testing.gateway_stub import StubGatewayProvider
    from kernel.tools.registry import _publish_tool_telemetry
    from kernel.turn_hooks import ModelHooks
    from kernel.turn_runner import TurnHooks, run_turn

    published: list[tuple[str, dict[str, Any]]] = []

    class _CaptureEventBus:
        async def publish(self, topic: str, payload: dict[str, Any]) -> None:
            published.append((topic, payload))

        def subscribe(self, topic: str, handler: Any) -> None:
            pass

    async def complete(**_kwargs: Any) -> Any:
        from types import SimpleNamespace

        return SimpleNamespace(content="Hello world.", tool_calls=[])

    bus = _CaptureEventBus()
    hooks = TurnHooks(models=ModelHooks(complete=complete))
    # runtime/tests has no gateway bridge conftest; wire the stub explicitly so
    # the kernel's provider-gateway call site resolves.
    hooks.context_seams = replace(
        hooks.context_seams, provider_gateways=StubGatewayProvider(complete)
    )
    hooks.context_seams.event_bus = bus
    hooks.identity.turn_index = 123
    hooks.identity.user_slug = "alice"

    result = await run_turn(mode_id="default", query="hi", hooks=hooks)
    assert result.response_text == "Hello world."

    decisions = [p for t, p in published if t == "turn.decision"]
    assert decisions, "turn.decision was never published"
    assert all(d["turn_index"] == 123 for d in decisions)
    assert all(d["user_slug"] == "alice" for d in decisions)

    # The dispatch ledger uses the same context field; assert the publisher
    # stamps it rather than reconstructing turn state on its own.
    from kernel.models import ToolExecutionContext

    tool_ctx = ToolExecutionContext(tool_id="fetch_page", turn_index=123, user_slug="alice")
    tool_ctx.event_bus = _CaptureEventBus()
    _publish_tool_telemetry(tool_ctx, "fetch_page", "success", 5, None)
    import asyncio

    await asyncio.sleep(0)
    telemetry = [p for t, p in published if t == "tool.telemetry"]
    assert telemetry, "tool.telemetry was never published"
    assert telemetry[-1]["turn_index"] == 123
    assert telemetry[-1]["user_slug"] == "alice"

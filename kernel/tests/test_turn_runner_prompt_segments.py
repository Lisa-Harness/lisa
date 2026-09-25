"""Turn-runner prompt-segment integration tests.

Locks the wire contract between the kernel's per-turn
``SystemPromptSegments`` registry and the ``CompletionRequest`` handed to
the provider gateway: the typed segments ride every turn-loop completion,
the flattened system message stays byte-identical to the legacy single
string, and the per-iteration lifecycle briefing replaces its own segment
rather than accumulating or rewriting the stable prefix.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest
from kernel.provider_contracts import flatten_system_prompt_segments
from kernel.testing.gateway_stub import StubGatewayProvider
from kernel.turn_hooks import ModelHooks, OperationHooks, TurnHooks
from kernel.turn_runner import ContextSeams, run_turn


class _StubCompletion:
    def __init__(
        self,
        content: str,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        self.content = content
        self.tool_calls = tool_calls or []


async def _stub_execute_operation(*_: Any, **__: Any) -> Any:
    class _R:
        output_text = ""
        output_json: ClassVar[dict[str, Any]] = {}

    return _R()


@pytest.mark.asyncio
async def test_completion_request_carries_ordered_prompt_segments() -> None:
    """The kernel sends typed system segments on the completion request."""
    captured: list[Any] = []

    async def complete_fn(**kwargs: Any) -> _StubCompletion:
        captured.append(kwargs)
        return _StubCompletion(content="Done.")

    hooks = TurnHooks(
        models=ModelHooks(complete=complete_fn),
        operations=OperationHooks(execute_operation=_stub_execute_operation),
        context_seams=ContextSeams(provider_gateways=StubGatewayProvider(complete_fn)),
    )
    result = await run_turn(mode_id="default", query="hello", hooks=hooks)
    assert result.response_text == "Done."

    # The provider gateway recorded the request object for the last gateway
    # it handed out; read segments off it directly.
    provider = hooks.context_seams.provider_gateways
    assert isinstance(provider, StubGatewayProvider)
    last = provider.last_gateway
    assert last is not None
    request = last.last_request
    assert request is not None

    segments = request.system_prompt_segments
    assert segments, "turn loop must send typed system segments"
    joined = flatten_system_prompt_segments(tuple(segments))
    assert joined == str(request.messages[0].get("content", ""))


@pytest.mark.asyncio
async def test_lifecycle_briefing_is_a_replaced_ephemeral_segment() -> None:
    """Per-iteration lifecycle briefing never accumulates in the segments."""
    call_count = 0

    async def two_step(**_kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _StubCompletion(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "function": {"name": "web_search", "arguments": '{"q": "x"}'},
                    }
                ],
            )
        return _StubCompletion(content="finished")

    provider = StubGatewayProvider(two_step)
    hooks = TurnHooks(
        models=ModelHooks(complete=two_step),
        operations=OperationHooks(execute_operation=_stub_execute_operation),
        context_seams=ContextSeams(provider_gateways=provider),
    )
    result = await run_turn(
        mode_id="default",
        query="hello",
        hooks=hooks,
        max_tool_iterations=5,
    )
    assert result.response_text == "finished"

    last = provider.last_gateway
    assert last is not None
    request = last.last_request
    assert request is not None
    segments = tuple(request.system_prompt_segments)

    lifecycle_blocks = [s for s in segments if not s.is_stable and "## Turn Lifecycle" in s.text]
    # Exactly one ephemeral lifecycle segment exists for the turn; iteration
    # 2's briefing replaced iteration 1's rather than appending a second.
    assert len(lifecycle_blocks) == 1
    # Stable segment content is untouched by the briefing.
    stable_texts = [s.text for s in segments if s.is_stable]
    assert stable_texts
    assert all("## Turn Lifecycle" not in text for text in stable_texts)


@pytest.mark.asyncio
async def test_flattened_prompt_is_byte_identical_to_messages_system() -> None:
    """Legacy message content equals the segment-flattened string."""

    async def complete_fn(**_kwargs: Any) -> _StubCompletion:
        return _StubCompletion(content="ok")

    provider = StubGatewayProvider(complete_fn)
    hooks = TurnHooks(
        models=ModelHooks(complete=complete_fn),
        operations=OperationHooks(execute_operation=_stub_execute_operation),
        context_seams=ContextSeams(provider_gateways=provider),
    )
    result = await run_turn(mode_id="default", query="hello", hooks=hooks)
    assert result.response_text == "ok"

    last = provider.last_gateway
    assert last is not None
    request = last.last_request
    segments = tuple(request.system_prompt_segments)
    assert flatten_system_prompt_segments(segments) == str(request.messages[0].get("content", ""))

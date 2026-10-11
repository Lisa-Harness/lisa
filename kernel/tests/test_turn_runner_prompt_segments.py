"""Turn-runner prompt-segment integration tests.

Locks the wire contract between the kernel's per-turn
``SystemPromptSegments`` registry and the ``CompletionRequest`` handed to
the provider gateway: the typed segments ride every turn-loop completion,
the flattened system message stays byte-identical to the legacy single
string, and lifecycle guidance does not rewrite the stable conversation prefix.
"""

from __future__ import annotations

import copy
from typing import Any, ClassVar

import pytest
from kernel.provider_contracts import flatten_system_prompt_segments
from kernel.testing.gateway_stub import StubGatewayProvider
from kernel.turn_hooks import ContextHooks, ModelHooks, OperationHooks, TurnHooks, TurnIdentity
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
async def test_lifecycle_briefing_preserves_provider_conversation_prefix() -> None:
    """Changing counters must not invalidate the already-sent conversation."""
    call_count = 0
    captured: list[dict[str, Any]] = []

    async def two_step(**kwargs: Any) -> _StubCompletion:
        nonlocal call_count
        captured.append(copy.deepcopy(kwargs))
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
    assert not lifecycle_blocks
    # Stable segment content is untouched by the briefing.
    stable_texts = [s.text for s in segments if s.is_stable]
    assert stable_texts
    assert all("## Turn Lifecycle" not in text for text in stable_texts)
    first, second = captured
    assert first["messages"] == second["messages"][: len(first["messages"])]
    assert any("## Turn Lifecycle" in str(m.get("content")) for m in first["messages"][1:])


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


@pytest.mark.asyncio
async def test_model_projection_preserves_raw_history_and_records_actual_request() -> None:
    raw = "full tool output " * 1000
    history = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "old",
                    "type": "function",
                    "function": {"name": "read", "arguments": "{}"},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "old", "content": raw},
    ]
    tasks: list[str] = []

    def project(
        messages: list[dict[str, Any]], task_id: str, *, budget_bytes: int
    ) -> list[dict[str, Any]]:
        tasks.append(task_id)
        return [
            {**message, "content": "recoverable preview"}
            if message.get("role") == "tool"
            else dict(message)
            for message in messages
        ]

    async def complete_fn(**_kwargs: Any) -> _StubCompletion:
        return _StubCompletion(content="ok")

    provider = StubGatewayProvider(complete_fn)
    hooks = TurnHooks(
        context=ContextHooks(prior_messages=lambda: history, project_messages=project),
        identity=TurnIdentity(task_id="task-1"),
        context_seams=ContextSeams(provider_gateways=provider),
    )
    result = await run_turn(mode_id="default", query="continue", hooks=hooks)
    assert tasks == ["task-1"]
    assert history[-1]["content"] == raw
    assert provider.last_gateway is not None
    request = provider.last_gateway.last_request
    assert request is not None
    assert (
        next(message for message in request.messages if message.get("role") == "tool")["content"]
        == "recoverable preview"
    )
    assert result.model_requests[0]["messages"] == request.messages


@pytest.mark.asyncio
@pytest.mark.parametrize(("context_window", "expected"), [(400_000, 400_000), (0, 131_072)])
async def test_projection_budget_is_sized_from_the_model_context_window(
    context_window: int, expected: int
) -> None:
    """The in-turn tool-result budget follows the effective model's window.

    A fixed 64KB budget with an 8KB per-result cap ignored the model: a
    400K-token coding model re-read the same 12-15KB files until the
    iteration cap (2026-10-09 turn-622). An undeclared window falls back to
    ``context_default_window_tokens``.
    """
    from types import SimpleNamespace

    budgets: list[int] = []

    def project(
        messages: list[dict[str, Any]], task_id: str, *, budget_bytes: int
    ) -> list[dict[str, Any]]:
        budgets.append(budget_bytes)
        return messages

    async def complete_fn(**_kwargs: Any) -> _StubCompletion:
        return _StubCompletion(content="ok")

    hooks = TurnHooks(
        models=ModelHooks(
            complete=complete_fn,
            resolve_model=lambda _mid: SimpleNamespace(
                provider="test",
                model_id="m",
                display_name="M",
                reasoning_model=False,
                thinking_model=False,
                context_window=context_window,
            ),
        ),
        context=ContextHooks(project_messages=project),
        context_seams=ContextSeams(provider_gateways=StubGatewayProvider(complete_fn)),
    )
    await run_turn(mode_id="default", query="hello", hooks=hooks)
    # 0.25 of the window at 4 chars per token.
    assert budgets == [expected]

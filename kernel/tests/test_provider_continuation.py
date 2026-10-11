"""Opaque provider context survives the same history used by approval/cap resume."""

import copy
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from kernel.provider_contracts import CompletionRequest, CompletionResult, ProviderContinuation
from kernel.turn_hooks import ModelHooks, TurnHooks
from kernel.turn_runner import ContextSeams, _record_iteration_completion, run_turn


@pytest.mark.asyncio
async def test_recording_completion_keeps_provider_context_separate_from_prose() -> None:
    continuation = ProviderContinuation(
        provider="azure_openai",
        model_id="reasoning-model",
        items=[{"type": "reasoning", "encrypted_content": "opaque", "summary": []}],
    )
    completion = CompletionResult(
        content="working",
        provider="azure_openai",
        model_id="reasoning-model",
        display_name="Reasoning Model",
        continuation=continuation,
    )
    state = SimpleNamespace(
        total_prompt_tokens=0,
        total_completion_tokens=0,
        total_tokens=0,
        total_cached_tokens=0,
        total_reasoning_tokens=0,
        total_cache_creation_tokens=0,
        total_duration_ms=0.0,
        total_latency_ms=0.0,
        iteration=1,
    )
    messages = []
    await _record_iteration_completion(
        state=state,
        hooks=TurnHooks(),
        mode_id="default",
        max_tool_iterations=5,
        messages=messages,
        completion=completion,
        effective_model_def=completion,
        record_model_usage=Mock(),
        emit_stage=AsyncMock(),
    )
    assert messages == [
        {
            "role": "assistant",
            "content": "working",
            "provider_continuation": asdict(continuation),
        }
    ]


@pytest.mark.asyncio
async def test_turn_loop_delivers_continuation_to_the_next_gateway_call() -> None:
    requests: list[CompletionRequest] = []

    class Gateway:
        async def complete(self, request: CompletionRequest) -> CompletionResult:
            requests.append(copy.deepcopy(request))
            return CompletionResult(
                provider=request.model_def.provider,
                model_id=request.model_def.model_id,
                display_name=request.model_def.display_name,
                content="" if len(requests) == 1 else "finished",
                tool_calls=[
                    {"id": "call_1", "function": {"name": "web_search", "arguments": '{"q":"x"}'}}
                ]
                if len(requests) == 1
                else [],
                continuation=ProviderContinuation(
                    provider=request.model_def.provider,
                    model_id=request.model_def.model_id,
                    items=[{"type": "reasoning", "encrypted_content": "opaque", "summary": []}],
                ),
                truncated=False,
            )

    class Gateways:
        def gateway_for(self, provider: str) -> Gateway:
            return Gateway()

    result = await run_turn(
        mode_id="default",
        query="hello",
        hooks=TurnHooks(
            models=ModelHooks(
                resolve_model=lambda _: SimpleNamespace(
                    provider="azure_openai",
                    model_id="reasoning-model",
                    display_name="Reasoning Model",
                    reasoning_model=True,
                    thinking_model=False,
                )
            ),
            context_seams=ContextSeams(provider_gateways=Gateways()),
        ),
        max_tool_iterations=5,
    )
    assert result.response_text == "finished"
    assistant = next(
        message for message in requests[1].messages if message.get("role") == "assistant"
    )
    assert assistant["provider_continuation"]["items"][0]["encrypted_content"] == "opaque"
    assert "opaque" not in assistant.get("content", "")

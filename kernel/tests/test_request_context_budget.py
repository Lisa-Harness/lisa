"""The whole request, not a fixed result share, controls residency."""

from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Any

import pytest
from kernel.context.context_budget import RequiredContextBudgetError
from kernel.context.request_budget import fit_request_context
from kernel.provider_contracts import CompletionRequest, RequestTokenCount


def _count(request: CompletionRequest) -> RequestTokenCount:
    size = sum(len(str(m.get("content") or "")) for m in request.messages)
    size += sum(len(str(m.get("tool_calls") or "")) for m in request.messages)
    size += len(str(request.tools or ""))
    return RequestTokenCount(tokens=size, method="test_characters")


def _request(content: str = "x" * 6000) -> CompletionRequest:
    return CompletionRequest(
        messages=[
            {"role": "system", "content": "system"},
            {"role": "assistant", "tool_calls": [{"id": "a"}]},
            {"role": "tool", "tool_call_id": "a", "content": content},
        ],
        tools=[{"schema": "arguments"}],
        model_def=SimpleNamespace(context_window=10_000),
        max_tokens=1000,
        temperature=0.4,
    )


def _project(messages: list[dict[str, Any]], *, budget_bytes: int) -> list[dict[str, Any]]:
    projected = copy.deepcopy(messages)
    projected[-1]["content"] = projected[-1]["content"][:budget_bytes]
    return projected


def test_full_results_use_more_than_old_quarter_share_without_reduction() -> None:
    request = _request()
    fitted = fit_request_context(request, _project, _count)
    assert fitted.messages == request.messages
    assert fitted.context_budget is not None
    assert fitted.context_budget.input_tokens == _count(request).tokens
    assert fitted.context_budget.completion_tokens == 1000
    assert fitted.context_budget.reduced is False


def test_pressure_accounts_for_schema_arguments_and_output_reserve() -> None:
    request = _request("x" * 12_000)
    fitted = fit_request_context(request, _project, _count)
    assert fitted.context_budget is not None
    budget = fitted.context_budget
    assert budget.reduced is True
    assert budget.pressure_headroom_tokens > 0
    assert budget.input_tokens + budget.completion_tokens + budget.margin_tokens <= 10_000
    assert budget.input_tokens == _count(fitted).tokens
    assert request.messages[-1]["content"] == "x" * 12_000
    larger_schema = copy.deepcopy(request)
    larger_schema.tools = [{"schema": "s" * 3000}]
    tighter = fit_request_context(larger_schema, _project, _count)
    assert len(tighter.messages[-1]["content"]) < len(fitted.messages[-1]["content"])


def test_required_context_overflow_is_explicit() -> None:
    request = _request()
    request.messages[0]["content"] = "s" * 10_000
    with pytest.raises(RequiredContextBudgetError, match="required"):
        fit_request_context(request, _project, _count)


def test_projection_that_cannot_fit_is_not_sent_to_provider() -> None:
    with pytest.raises(RequiredContextBudgetError, match="fit"):
        fit_request_context(_request("x" * 12_000), lambda messages, **_: messages, _count)


def test_no_projector_still_checks_request_bounds() -> None:
    with pytest.raises(RequiredContextBudgetError):
        fit_request_context(_request("x" * 12_000), None, _count)


def test_invalid_counter_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        fit_request_context(
            _request(), _project, lambda _: RequestTokenCount(tokens=-1, method="broken")
        )


def test_retry_cannot_take_space_already_allocated_to_input() -> None:
    from kernel.turn_runner import _truncation_retry_budget

    request = fit_request_context(_request("x" * 12_000), _project, _count)
    retry = _truncation_retry_budget(
        base_request=request,
        effective_model_def=request.model_def,
        effective_tools=request.tools,
        tools_offered_this_turn=True,
    )
    assert (
        retry is None
        or retry[0].max_tokens
        + request.context_budget.input_tokens
        + (request.context_budget.margin_tokens)
        <= request.model_def.context_window
    )
    if retry is not None:
        assert retry[0].context_budget == request.context_budget


def test_output_ceiling_tightens_to_fit_required_input_without_dropping_it() -> None:
    request = _request()
    request.messages[0]["content"] = "required " * 700
    request.max_tokens = 9000
    fitted = fit_request_context(request, _project, _count)
    budget = fitted.context_budget
    assert budget.requested_completion_tokens == 9000
    assert fitted.max_tokens == 10_000 - budget.margin_tokens - budget.required_tokens
    assert fitted.messages[0] == request.messages[0]
    assert budget.input_tokens + fitted.max_tokens + budget.margin_tokens <= 10_000


def test_observed_underestimate_tightens_later_requests_and_survives_checkpoint() -> None:
    import logging

    from kernel.provider_contracts import CompletionResult, CompletionUsage
    from kernel.turn_continuation import capture_state
    from kernel.turn_hooks import ModelHooks, TurnHooks
    from kernel.turn_phases._turn_state import _TurnState
    from kernel.turn_runner import _audit_request_context, _count_resident_request
    from pydantic import TypeAdapter

    request = fit_request_context(_request(), _project, _count)
    request.model_def.model_id = "test"
    state = _TurnState()
    observed = request.context_budget.input_tokens * 2
    completion = CompletionResult(
        content="done",
        provider="test",
        model_id="test",
        display_name="test",
        usage=CompletionUsage(prompt_tokens=observed),
    )
    _audit_request_context(
        completion, request=request, state=state, log=logging.getLogger(__name__)
    )
    restored = TypeAdapter(_TurnState).validate_json(capture_state(state))
    counted = _count_resident_request(
        request, state=restored, hooks=TurnHooks(models=ModelHooks(count_request_tokens=_count))
    )
    assert counted.tokens == observed
    assert "observed_scale=2.0000" in counted.method

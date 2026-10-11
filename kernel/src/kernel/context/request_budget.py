"""Fit a complete provider request; tool payloads use the remaining capacity."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from math import ceil
from typing import Any, Protocol

from kernel.config import settings
from kernel.context.context_budget import RequiredContextBudgetError, model_context_window_tokens
from kernel.provider_contracts import CompletionRequest, RequestContextBudget, RequestTokenCount

_PROJECTION_SEARCH_STEPS = 20


class ResultProjector(Protocol):
    def __call__(
        self, messages: list[dict[str, Any]], *, budget_bytes: int
    ) -> list[dict[str, Any]]: ...


def request_input_text(request: CompletionRequest) -> str:
    """Include framing, schemas, arguments, and structured-output instructions."""
    response_format = request.response_format
    if isinstance(response_format, type) and hasattr(response_format, "model_json_schema"):
        response_format = response_format.model_json_schema()
    return json.dumps(
        {
            "messages": request.messages,
            "tools": request.tools,
            "response_format": response_format,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def estimate_request_tokens(request: CompletionRequest) -> RequestTokenCount:
    """Conservative UTF-8 estimate, not a claim about a provider's tokenizer."""
    return RequestTokenCount(
        tokens=ceil(len(request_input_text(request).encode("utf-8")) / 3),
        method="utf8_bytes_div_3_estimate",
    )


def _count(
    request: CompletionRequest, counter: Callable[[CompletionRequest], RequestTokenCount]
) -> RequestTokenCount:
    counted = counter(request)
    if counted.tokens < 0:
        raise ValueError("Request token count must be non-negative.")
    return counted


def _fit_projection(
    request: CompletionRequest,
    projector: ResultProjector,
    counter: Callable[[CompletionRequest], RequestTokenCount],
    input_limit: int,
    payload_bytes: int,
) -> tuple[CompletionRequest, RequestTokenCount, int]:
    """Search the largest fitting byte budget and validate the resulting request."""
    low, high = 0, payload_bytes
    best: tuple[CompletionRequest, RequestTokenCount, int] | None = None
    for _ in range(_PROJECTION_SEARCH_STEPS):
        budget = (low + high) // 2
        candidate = replace(request, messages=projector(request.messages, budget_bytes=budget))
        counted = _count(candidate, counter)
        if counted.tokens <= input_limit:
            best = candidate, counted, budget
            low = budget + 1
        else:
            high = budget - 1
        if low > high:
            break
    if best is None:
        candidate = replace(request, messages=projector(request.messages, budget_bytes=0))
        counted = _count(candidate, counter)
        if counted.tokens > input_limit:
            raise RequiredContextBudgetError(
                f"Request cannot fit even with referenced tool results "
                f"({counted.tokens} input tokens, {input_limit} available)."
            )
        best = candidate, counted, 0
    return best


def fit_request_context(
    request: CompletionRequest,
    projector: ResultProjector | None,
    counter: Callable[[CompletionRequest], RequestTokenCount] = estimate_request_tokens,
) -> CompletionRequest:
    """Keep full permitted evidence until the entire request reaches its window."""
    window = model_context_window_tokens(request.model_def)
    margin = ceil(window * settings.context_request_margin_share)
    required_messages = [
        {**message, "content": ""} if message.get("role") == "tool" else message
        for message in request.messages
    ]
    required = _count(replace(request, messages=required_messages), counter)
    requested_completion = request.max_tokens
    output_room = window - margin - required.tokens
    if output_room <= 0:
        raise RequiredContextBudgetError(
            f"Request required content needs {required.tokens} input tokens, "
            f"but only {window - margin} remain before any completion."
        )
    request = replace(request, max_tokens=min(request.max_tokens, output_room))
    input_limit = window - request.max_tokens - margin
    payload_bytes = sum(
        len(str(message.get("content") or "").encode("utf-8"))
        for message in request.messages
        if message.get("role") == "tool"
    )
    # Redaction still applies when no reduction is needed.
    candidate = (
        replace(request, messages=projector(request.messages, budget_bytes=payload_bytes))
        if projector is not None
        else request
    )
    counted = _count(candidate, counter)
    budget = payload_bytes
    headroom = 0
    if counted.tokens > input_limit:
        if projector is None:
            raise RequiredContextBudgetError(
                "Request cannot fit its context window without a tool-result projector."
            )
        headroom = min(margin, max(0, input_limit - required.tokens))
        candidate, counted, budget = _fit_projection(
            request, projector, counter, input_limit - headroom, payload_bytes
        )
    return replace(
        candidate,
        context_budget=RequestContextBudget(
            window_tokens=window,
            requested_completion_tokens=requested_completion,
            completion_tokens=request.max_tokens,
            margin_tokens=margin,
            pressure_headroom_tokens=headroom,
            required_tokens=required.tokens,
            input_tokens=counted.tokens,
            tool_budget_bytes=budget,
            counting_method=counted.method,
            reduced=candidate.messages != request.messages,
        ),
    )

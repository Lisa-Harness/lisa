"""Routing logic for the host.

Module-level, testable functions that replace the nested closures in
``main.py``'s ``_plan_route``. The main module passes
``resolve_model`` and ``complete_live`` as callables so this module has
no dependency on ``_RUNTIME_STATE``.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from kernel import router as shared_router
from kernel.models import RouteDecision as RouterRouteDecision
from kernel.models import SurfaceContext
from kernel.operations.structured_output import json_only_retry_messages
from kernel.operations.structured_output import parse_structured_payload
from pydantic import ValidationError

from hosts.core.model_runtime import StubCompletion, StubModelDefinition

logger = logging.getLogger(__name__)

_last_debug: dict[str, Any] = {}


def last_debug_dict() -> dict[str, Any]:
    """Return the debug dict from the most recent plan_route call."""
    return dict(_last_debug)


def _build_router_debug_dict(
    raw_message: str,
    conversation_excerpt_markdown: str,
    router_alias: str,
) -> dict[str, Any]:
    """Initialize the router planner debug payload."""
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "model_alias": router_alias,
        "resolved_model": {},
        "raw_message": raw_message,
        "system_prompt_chars": 0,
        "user_prompt_chars": 0,
        "conversation_excerpt_chars": len(conversation_excerpt_markdown),
        "user_prompt_preview": "",
        "planner_prompt_tokens": 0,
        "planner_completion_tokens": 0,
        "planner_total_tokens": 0,
        "planner_completion_attempts": [],
        "structured_output_retry_used": False,
        "structured_output_retry_succeeded": False,
    }


def _resolve_planner_model(
    router_alias: str,
    resolve_model: Callable[[str], StubModelDefinition],
    router_debug: dict[str, Any],
    fallback: RouterRouteDecision,
) -> StubModelDefinition | RouterRouteDecision:
    """Resolve the router planner model or return a fallback decision."""
    try:
        model_def = resolve_model(router_alias)
    except Exception as exc:  # pylint: disable=broad-except — process boundary
        logger.warning("router planner model resolution failed: %s: %s", type(exc).__name__, exc)
        router_debug["planner_model_resolution_error"] = f"{type(exc).__name__}: {exc}"
        _apply_fallback(router_debug, fallback, "planner_model_resolution_error")
        return fallback

    router_debug["resolved_model"] = {
        "model_id": model_def.model_id,
        "provider": model_def.provider,
        "reasoning_model": model_def.reasoning_model,
    }
    return model_def


def _check_is_default_executor(
    model_def: StubModelDefinition,
    resolve_model: Callable[[str], StubModelDefinition],
    model_alias_for_mode: Callable[[str], str],
) -> bool:
    """Return True if the planner model is the same as the default model."""
    try:
        default_alias = model_alias_for_mode("default")
        default_model_def = resolve_model(default_alias)
        return model_def.model_id == default_model_def.model_id
    except Exception:  # pylint: disable=broad-except — best-effort check
        return False


def _apply_fallback(
    router_debug: dict[str, Any],
    fallback: RouterRouteDecision,
    reason: str,
) -> None:
    """Update the debug dict with fallback reason and decision fields."""
    router_debug.update(
        {
            "fallback_used": True,
            "fallback_reason": reason,
            "planner_reasoning": fallback.reasoning,
            "selected_mode_id": fallback.selected_mode_id,
            "normalized_query": fallback.normalized_query,
            "confidence_score": fallback.confidence_score,
        }
    )


def _record_router_completion(
    completion_result: StubCompletion,
    *,
    attempt: str,
    router_debug: dict[str, Any],
    coerce_int: Callable[[Any], int],
) -> None:
    """Record one router planner completion attempt into the debug dict."""
    prompt_tokens = coerce_int(completion_result.prompt_tokens)
    completion_tokens = coerce_int(completion_result.completion_tokens)
    total_tokens = coerce_int(completion_result.total_tokens)
    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens

    router_debug["planner_prompt_tokens"] = (
        coerce_int(router_debug.get("planner_prompt_tokens")) + prompt_tokens
    )
    router_debug["planner_completion_tokens"] = (
        coerce_int(router_debug.get("planner_completion_tokens")) + completion_tokens
    )
    router_debug["planner_total_tokens"] = (
        coerce_int(router_debug.get("planner_total_tokens")) + total_tokens
    )

    attempts = router_debug.get("planner_completion_attempts")
    if not isinstance(attempts, list):
        attempts = []
    attempts.append(
        {
            "attempt": attempt,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "content_chars": len(completion_result.content),
            "tool_call_count": len(completion_result.tool_calls),
        }
    )
    router_debug["planner_completion_attempts"] = attempts


def _capture_prompt_shape(
    messages: list[dict[str, Any]],
    *,
    router_debug: dict[str, Any],
    preview_fn: Callable[[str, int], str],
) -> None:
    """Capture system/user prompt shape into the router debug dict."""
    if not messages:
        return
    system_text = str(messages[0].get("content") or "")
    user_text = str(messages[1].get("content") or "") if len(messages) > 1 else ""
    router_debug["system_prompt_chars"] = len(system_text)
    router_debug["user_prompt_chars"] = len(user_text)
    router_debug["user_prompt_preview"] = preview_fn(user_text, limit=240)


def _extract_json_object(text: str) -> str:
    """Extract the first likely JSON object from a model response."""
    normalized = str(text or "").strip()
    if normalized.startswith("```json"):
        normalized = normalized[7:]
    elif normalized.startswith("```"):
        normalized = normalized[3:]
    if normalized.endswith("```"):
        normalized = normalized[:-3]
    normalized = normalized.strip()

    if normalized.startswith("{") and normalized.endswith("}"):
        return normalized

    start = normalized.find("{")
    end = normalized.rfind("}")
    if start >= 0 and end > start:
        return normalized[start : end + 1]
    return normalized


async def _retry_without_response_format(
    *,
    messages: list[dict[str, Any]] | None,
    model_def: StubModelDefinition,
    kwargs: dict[str, Any],
    response_format: type[Any],
    router_debug: dict[str, Any],
    complete_live: Callable[..., Awaitable[StubCompletion]],
    coerce_int: Callable[[Any], int],
    preview_fn: Callable[[str, int], str],
    attempt: str,
) -> StubCompletion:
    """Retry a router completion without structured output format.

    Uses the shared ``json_only_retry_messages`` helper to append a fresh
    user turn with a JSON-only instruction, which is more reliable with
    instruction-tuned models than mutating the system message. The retry
    uses a larger token budget than the original attempt so a truncated
    first completion (EOF mid-string) has room to finish on retry.
    """
    retry_messages = json_only_retry_messages(
        list(messages or []),
        response_format,
    )
    retry_max_tokens = int(kwargs.get("max_tokens") or 0)
    if retry_max_tokens:
        retry_max_tokens = min(retry_max_tokens * 2, 4096)
    retry_completion = await complete_live(
        messages=retry_messages,
        model_def=model_def,
        max_tokens=retry_max_tokens,
        temperature=float(kwargs.get("temperature") or 0.0),
        response_format=None,
    )
    _record_router_completion(
        retry_completion,
        attempt=attempt,
        router_debug=router_debug,
        coerce_int=coerce_int,
    )
    router_debug["structured_output_retry_used"] = True
    router_debug["structured_output_retry_succeeded"] = True
    retry_text = _extract_json_object(retry_completion.content)
    parsed = parse_structured_payload(response_format, retry_text)
    router_debug["raw_completion_preview"] = preview_fn(retry_completion.content, 320)
    return StubCompletion(
        content=parsed.model_dump_json(),
        tool_calls=retry_completion.tool_calls,
        prompt_tokens=retry_completion.prompt_tokens,
        completion_tokens=retry_completion.completion_tokens,
        total_tokens=retry_completion.total_tokens,
    )


async def _complete_router_planner(
    *,
    kwargs: dict[str, Any],
    model_def: StubModelDefinition,
    router_debug: dict[str, Any],
    complete_live: Callable[..., Awaitable[StubCompletion]],
    coerce_int: Callable[[Any], int],
    preview_fn: Callable[[str, int], str],
) -> StubCompletion:
    """Execute one router planner completion with retry logic."""
    response_format = kwargs.get("response_format")
    messages = kwargs.get("messages")
    if isinstance(messages, list):
        _capture_prompt_shape(
            messages,
            router_debug=router_debug,
            preview_fn=preview_fn,
        )

    try:
        completion = await complete_live(**kwargs)
        attempt = (
            "fast_path_structured"
            if response_format is not RouterRouteDecision
            else "primary_structured"
        )
        _record_router_completion(
            completion,
            attempt=attempt,
            router_debug=router_debug,
            coerce_int=coerce_int,
        )
    except Exception as exc:  # pylint: disable=broad-except — process boundary
        if response_format is None:
            raise
        logger.warning("router planner completion error: %s: %s", type(exc).__name__, exc)
        router_debug["planner_completion_error"] = f"{type(exc).__name__}: {exc}"
        try:
            return await _retry_without_response_format(
                messages=messages if isinstance(messages, list) else None,
                model_def=model_def,
                kwargs=kwargs,
                response_format=response_format,
                router_debug=router_debug,
                complete_live=complete_live,
                coerce_int=coerce_int,
                preview_fn=preview_fn,
                attempt="retry_after_completion_error",
            )
        except ValidationError as retry_exc:
            router_debug["planner_retry_validation_error"] = (
                f"{type(retry_exc).__name__}: {retry_exc}"
            )
        except Exception as retry_exc:  # pylint: disable=broad-except — process boundary
            router_debug["planner_retry_error"] = f"{type(retry_exc).__name__}: {retry_exc}"
        router_debug["structured_output_retry_succeeded"] = False
        router_debug["fallback_reason"] = "planner_completion_error"
        raise RuntimeError("router planner completion failed") from exc

    if response_format is None:
        return completion

    try:
        parsed = parse_structured_payload(
            response_format, _extract_json_object(completion.content)
        )
    except ValidationError as exc:
        logger.warning(
            "router planner output validation failed: %s: %s",
            type(exc).__name__,
            exc,
        )
        router_debug["planner_output_validation_error"] = f"{type(exc).__name__}: {exc}"
        router_debug["raw_completion_preview"] = preview_fn(completion.content, 320)
        try:
            return await _retry_without_response_format(
                messages=messages if isinstance(messages, list) else None,
                model_def=model_def,
                kwargs=kwargs,
                response_format=response_format,
                router_debug=router_debug,
                complete_live=complete_live,
                coerce_int=coerce_int,
                preview_fn=preview_fn,
                attempt="retry_after_validation_error",
            )
        except ValidationError as retry_exc:
            router_debug["planner_retry_validation_error"] = (
                f"{type(retry_exc).__name__}: {retry_exc}"
            )
        except Exception as retry_exc:  # pylint: disable=broad-except — process boundary
            router_debug["planner_retry_error"] = f"{type(retry_exc).__name__}: {retry_exc}"
        router_debug["structured_output_retry_succeeded"] = False
        router_debug["fallback_reason"] = "planner_output_validation_error"
        raise RuntimeError("router planner validation failed") from exc

    router_debug["raw_completion_preview"] = preview_fn(completion.content, 320)
    return StubCompletion(
        content=parsed.model_dump_json(),
        tool_calls=completion.tool_calls,
        prompt_tokens=completion.prompt_tokens,
        completion_tokens=completion.completion_tokens,
        total_tokens=completion.total_tokens,
    )


async def plan_route(
    *,
    raw_message: str,
    conversation_excerpt_markdown: str,
    router_alias: str,
    resolve_model: Callable[[str], StubModelDefinition],
    complete_live: Callable[..., Awaitable[StubCompletion]],
    model_alias_for_mode: Callable[[str], str],
    coerce_int: Callable[[Any], int],
    preview_fn: Callable[[str, int], str],
    load_route_prompt: Callable[[], str],
    load_fast_path_verifier_prompt: Callable[[], str],
    mode_supported: Callable[[str, Any], bool],
    build_user_prompt: Callable[..., str],
    get_workflow_catalog: Callable[[], dict[str, Any]],
    set_last_router_debug: Callable[[dict[str, Any]], None],
    current_mode_id: str = "default",
) -> RouterRouteDecision:
    """Plan the route for a natural-language turn.

    All dependencies are passed as callables so this function has no
    dependency on ``_RUNTIME_STATE`` or main.py globals.
    """
    global _last_debug  # noqa: PLW0603 — module-level cache for last_debug_dict()

    fallback = RouterRouteDecision(
        selected_mode_id="default",
        normalized_query=raw_message.strip(),
        reasoning="Router planner fallback to default route.",
        confidence_score=0.0,
    )
    router_debug = _build_router_debug_dict(
        raw_message, conversation_excerpt_markdown, router_alias
    )

    model_or_fallback = _resolve_planner_model(
        router_alias, resolve_model, router_debug, fallback
    )
    if isinstance(model_or_fallback, RouterRouteDecision):
        _last_debug = router_debug
        set_last_router_debug(router_debug)
        return model_or_fallback
    model_def = model_or_fallback

    is_default_executor = _check_is_default_executor(model_def, resolve_model, model_alias_for_mode)

    async def _complete_adapter(**kwargs: Any) -> StubCompletion:
        return await _complete_router_planner(
            kwargs=kwargs,
            model_def=model_def,
            router_debug=router_debug,
            complete_live=complete_live,
            coerce_int=coerce_int,
            preview_fn=preview_fn,
        )

    decision = await shared_router.plan_route(
        user_message=raw_message,
        current_route_mode=current_mode_id,
        surface_context=SurfaceContext(
            surface_type="personal_dm",
            surface_id="little-lisa-console",
            memory_scope="personal",
        ),
        preferred_mode_id=current_mode_id,
        conversation_excerpt_markdown=conversation_excerpt_markdown,
        catalog=get_workflow_catalog(),
        load_route_prompt=load_route_prompt,
        load_fast_path_verifier_prompt=load_fast_path_verifier_prompt,
        resolve_planner_model=lambda: model_def,
        complete=_complete_adapter,
        mode_supported=mode_supported,
        build_user_prompt=build_user_prompt,
        model_bounded_max_tokens=1024,
        is_default_executor=is_default_executor,
    )

    fallback_reason = str(router_debug.get("fallback_reason") or "")
    if (
        not fallback_reason
        and decision.planner_source == "deterministic"
        and decision.reasoning == fallback.reasoning
    ):
        fallback_reason = "planner_fallback"

    if fallback_reason:
        logger.info("router planner fell back to default (reason=%s)", fallback_reason)
    else:
        logger.debug(
            "router routed to mode=%s confidence=%.2f",
            decision.selected_mode_id,
            decision.confidence_score,
        )

    router_debug.update(
        {
            "fallback_used": bool(fallback_reason),
            "fallback_reason": fallback_reason,
            "selected_mode_id": decision.selected_mode_id,
            "normalized_query": decision.normalized_query,
            "planner_reasoning": decision.reasoning.strip(),
            "confidence_score": decision.confidence_score,
            "clarification_needed": decision.clarification_needed,
            "coerced_to_default_reason": "",
        }
    )
    _last_debug = router_debug
    set_last_router_debug(router_debug)
    return decision
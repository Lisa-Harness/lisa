"""Front-door routing logic for Little Lisa.

Module-level, testable functions that replace the nested closures in
``main.py``'s ``_plan_frontdoor_route``. The main module passes
``resolve_model`` and ``complete_live`` as callables so this module has
no dependency on ``_RUNTIME_STATE``.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from kernel import frontdoor as shared_frontdoor
from kernel.models import RouteDecision as FrontDoorRouteDecision
from kernel.models import SurfaceContext
from pydantic import ValidationError

from .model_runtime import StubCompletion, StubModelDefinition

logger = logging.getLogger(__name__)

_last_debug: dict[str, Any] = {}


def last_debug_dict() -> dict[str, Any]:
    """Return the debug dict from the most recent plan_frontdoor_route call."""
    return dict(_last_debug)


def _build_frontdoor_debug_dict(
    raw_message: str,
    conversation_excerpt_markdown: str,
    frontdoor_alias: str,
) -> dict[str, Any]:
    """Initialize the front-door planner debug payload."""
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "model_alias": frontdoor_alias,
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
    frontdoor_alias: str,
    resolve_model: Callable[[str], StubModelDefinition],
    frontdoor_debug: dict[str, Any],
    fallback: FrontDoorRouteDecision,
) -> StubModelDefinition | FrontDoorRouteDecision:
    """Resolve the front-door planner model or return a fallback decision."""
    try:
        model_def = resolve_model(frontdoor_alias)
    except Exception as exc:  # pylint: disable=broad-except — process boundary
        logger.warning("frontdoor planner model resolution failed: %s: %s", type(exc).__name__, exc)
        frontdoor_debug["planner_model_resolution_error"] = f"{type(exc).__name__}: {exc}"
        _apply_fallback(frontdoor_debug, fallback, "planner_model_resolution_error")
        return fallback

    frontdoor_debug["resolved_model"] = {
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
    frontdoor_debug: dict[str, Any],
    fallback: FrontDoorRouteDecision,
    reason: str,
) -> None:
    """Update the debug dict with fallback reason and decision fields."""
    frontdoor_debug.update(
        {
            "fallback_used": True,
            "fallback_reason": reason,
            "planner_reasoning": fallback.reasoning,
            "selected_mode_id": fallback.selected_mode_id,
            "normalized_query": fallback.normalized_query,
            "confidence_score": fallback.confidence_score,
        }
    )


def _record_frontdoor_completion(
    completion_result: StubCompletion,
    *,
    attempt: str,
    frontdoor_debug: dict[str, Any],
    coerce_int: Callable[[Any], int],
) -> None:
    """Record one front-door planner completion attempt into the debug dict."""
    prompt_tokens = coerce_int(completion_result.prompt_tokens)
    completion_tokens = coerce_int(completion_result.completion_tokens)
    total_tokens = coerce_int(completion_result.total_tokens)
    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens

    frontdoor_debug["planner_prompt_tokens"] = (
        coerce_int(frontdoor_debug.get("planner_prompt_tokens")) + prompt_tokens
    )
    frontdoor_debug["planner_completion_tokens"] = (
        coerce_int(frontdoor_debug.get("planner_completion_tokens")) + completion_tokens
    )
    frontdoor_debug["planner_total_tokens"] = (
        coerce_int(frontdoor_debug.get("planner_total_tokens")) + total_tokens
    )

    attempts = frontdoor_debug.get("planner_completion_attempts")
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
    frontdoor_debug["planner_completion_attempts"] = attempts


def _capture_prompt_shape(
    messages: list[dict[str, Any]],
    *,
    frontdoor_debug: dict[str, Any],
    preview_fn: Callable[[str, int], str],
) -> None:
    """Capture system/user prompt shape into the front-door debug dict."""
    if not messages:
        return
    system_text = str(messages[0].get("content") or "")
    user_text = str(messages[1].get("content") or "") if len(messages) > 1 else ""
    frontdoor_debug["system_prompt_chars"] = len(system_text)
    frontdoor_debug["user_prompt_chars"] = len(user_text)
    frontdoor_debug["user_prompt_preview"] = preview_fn(user_text, limit=240)


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
    frontdoor_debug: dict[str, Any],
    complete_live: Callable[..., Awaitable[StubCompletion]],
    coerce_int: Callable[[Any], int],
    preview_fn: Callable[[str, int], str],
    attempt: str,
) -> StubCompletion:
    """Retry a front-door completion without structured output format."""
    retry_messages = list(messages or [])
    if retry_messages:
        retry_system = dict(retry_messages[0])
        retry_system["content"] = (
            f"{retry_system.get('content') or ''!s}\n\n"
            f"Return only one valid JSON object matching {response_format.__name__}. "
            "Do not include Markdown fences, prose, or comments."
        )
        retry_messages[0] = retry_system
    retry_completion = await complete_live(
        messages=retry_messages,
        model_def=model_def,
        max_tokens=int(kwargs.get("max_tokens") or 0),
        temperature=float(kwargs.get("temperature") or 0.0),
        response_format=None,
    )
    _record_frontdoor_completion(
        retry_completion,
        attempt=attempt,
        frontdoor_debug=frontdoor_debug,
        coerce_int=coerce_int,
    )
    frontdoor_debug["structured_output_retry_used"] = True
    frontdoor_debug["structured_output_retry_succeeded"] = True
    retry_text = _extract_json_object(retry_completion.content)
    parsed = response_format.model_validate_json(retry_text)
    frontdoor_debug["raw_completion_preview"] = preview_fn(retry_completion.content, 320)
    return StubCompletion(
        content=parsed.model_dump_json(),
        tool_calls=retry_completion.tool_calls,
        prompt_tokens=retry_completion.prompt_tokens,
        completion_tokens=retry_completion.completion_tokens,
        total_tokens=retry_completion.total_tokens,
    )


async def _complete_frontdoor_planner(
    *,
    kwargs: dict[str, Any],
    model_def: StubModelDefinition,
    frontdoor_debug: dict[str, Any],
    complete_live: Callable[..., Awaitable[StubCompletion]],
    coerce_int: Callable[[Any], int],
    preview_fn: Callable[[str, int], str],
) -> StubCompletion:
    """Execute one front-door planner completion with retry logic."""
    response_format = kwargs.get("response_format")
    messages = kwargs.get("messages")
    if isinstance(messages, list):
        _capture_prompt_shape(
            messages,
            frontdoor_debug=frontdoor_debug,
            preview_fn=preview_fn,
        )

    try:
        completion = await complete_live(**kwargs)
        attempt = (
            "fast_path_structured"
            if response_format is not FrontDoorRouteDecision
            else "primary_structured"
        )
        _record_frontdoor_completion(
            completion,
            attempt=attempt,
            frontdoor_debug=frontdoor_debug,
            coerce_int=coerce_int,
        )
    except Exception as exc:  # pylint: disable=broad-except — process boundary
        if response_format is None:
            raise
        logger.warning("frontdoor planner completion error: %s: %s", type(exc).__name__, exc)
        frontdoor_debug["planner_completion_error"] = f"{type(exc).__name__}: {exc}"
        try:
            return await _retry_without_response_format(
                messages=messages if isinstance(messages, list) else None,
                model_def=model_def,
                kwargs=kwargs,
                response_format=response_format,
                frontdoor_debug=frontdoor_debug,
                complete_live=complete_live,
                coerce_int=coerce_int,
                preview_fn=preview_fn,
                attempt="retry_after_completion_error",
            )
        except ValidationError as retry_exc:
            frontdoor_debug["planner_retry_validation_error"] = (
                f"{type(retry_exc).__name__}: {retry_exc}"
            )
        except Exception as retry_exc:  # pylint: disable=broad-except — process boundary
            frontdoor_debug["planner_retry_error"] = f"{type(retry_exc).__name__}: {retry_exc}"
        frontdoor_debug["structured_output_retry_succeeded"] = False
        frontdoor_debug["fallback_reason"] = "planner_completion_error"
        raise RuntimeError("frontdoor planner completion failed") from exc

    if response_format is None:
        return completion

    try:
        parsed = response_format.model_validate_json(_extract_json_object(completion.content))
    except ValidationError as exc:
        logger.warning(
            "frontdoor planner output validation failed: %s: %s",
            type(exc).__name__,
            exc,
        )
        frontdoor_debug["planner_output_validation_error"] = f"{type(exc).__name__}: {exc}"
        frontdoor_debug["raw_completion_preview"] = preview_fn(completion.content, 320)
        try:
            return await _retry_without_response_format(
                messages=messages if isinstance(messages, list) else None,
                model_def=model_def,
                kwargs=kwargs,
                response_format=response_format,
                frontdoor_debug=frontdoor_debug,
                complete_live=complete_live,
                coerce_int=coerce_int,
                preview_fn=preview_fn,
                attempt="retry_after_validation_error",
            )
        except ValidationError as retry_exc:
            frontdoor_debug["planner_retry_validation_error"] = (
                f"{type(retry_exc).__name__}: {retry_exc}"
            )
        except Exception as retry_exc:  # pylint: disable=broad-except — process boundary
            frontdoor_debug["planner_retry_error"] = f"{type(retry_exc).__name__}: {retry_exc}"
        frontdoor_debug["structured_output_retry_succeeded"] = False
        frontdoor_debug["fallback_reason"] = "planner_output_validation_error"
        raise RuntimeError("frontdoor planner validation failed") from exc

    frontdoor_debug["raw_completion_preview"] = preview_fn(completion.content, 320)
    return StubCompletion(
        content=parsed.model_dump_json(),
        tool_calls=completion.tool_calls,
        prompt_tokens=completion.prompt_tokens,
        completion_tokens=completion.completion_tokens,
        total_tokens=completion.total_tokens,
    )


async def plan_frontdoor_route(
    *,
    raw_message: str,
    conversation_excerpt_markdown: str,
    frontdoor_alias: str,
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
    set_last_frontdoor_debug: Callable[[dict[str, Any]], None],
) -> FrontDoorRouteDecision:
    """Plan the front-door route for a natural-language turn.

    All dependencies are passed as callables so this function has no
    dependency on ``_RUNTIME_STATE`` or main.py globals.
    """
    global _last_debug  # noqa: PLW0603 — module-level cache for last_debug_dict()

    fallback = FrontDoorRouteDecision(
        selected_mode_id="default",
        normalized_query=raw_message.strip(),
        reasoning="Front-door planner fallback to default route.",
        confidence_score=0.0,
    )
    frontdoor_debug = _build_frontdoor_debug_dict(
        raw_message, conversation_excerpt_markdown, frontdoor_alias
    )

    model_or_fallback = _resolve_planner_model(
        frontdoor_alias, resolve_model, frontdoor_debug, fallback
    )
    if isinstance(model_or_fallback, FrontDoorRouteDecision):
        _last_debug = frontdoor_debug
        set_last_frontdoor_debug(frontdoor_debug)
        return model_or_fallback
    model_def = model_or_fallback

    is_default_executor = _check_is_default_executor(model_def, resolve_model, model_alias_for_mode)

    async def _complete_adapter(**kwargs: Any) -> StubCompletion:
        return await _complete_frontdoor_planner(
            kwargs=kwargs,
            model_def=model_def,
            frontdoor_debug=frontdoor_debug,
            complete_live=complete_live,
            coerce_int=coerce_int,
            preview_fn=preview_fn,
        )

    decision = await shared_frontdoor.plan_frontdoor_route(
        user_message=raw_message,
        current_route_mode="default",
        surface_context=SurfaceContext(
            surface_type="personal_dm",
            surface_id="little-lisa-console",
            memory_scope="personal",
        ),
        preferred_mode_id="default",
        conversation_excerpt_markdown=conversation_excerpt_markdown,
        catalog=get_workflow_catalog(),
        load_route_prompt=load_route_prompt,
        load_fast_path_verifier_prompt=load_fast_path_verifier_prompt,
        resolve_planner_model=lambda: model_def,
        complete=_complete_adapter,
        mode_supported=mode_supported,
        build_user_prompt=build_user_prompt,
        model_bounded_max_tokens=512,
        is_default_executor=is_default_executor,
    )

    fallback_reason = str(frontdoor_debug.get("fallback_reason") or "")
    if (
        not fallback_reason
        and decision.planner_source == "deterministic"
        and decision.reasoning == fallback.reasoning
    ):
        fallback_reason = "planner_fallback"

    if fallback_reason:
        logger.info("frontdoor planner fell back to default (reason=%s)", fallback_reason)
    else:
        logger.debug(
            "frontdoor routed to mode=%s confidence=%.2f",
            decision.selected_mode_id,
            decision.confidence_score,
        )

    frontdoor_debug.update(
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
    _last_debug = frontdoor_debug
    set_last_frontdoor_debug(frontdoor_debug)
    return decision

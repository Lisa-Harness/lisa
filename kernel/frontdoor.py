"""Shared front-door routing engine for Lisa framework hosts.

This module owns the common routing contract and bounded routing behavior for
ordinary natural-language turns. Hosts provide small hooks for authored prompt
loading, planner-model resolution, provider completion, and any host-specific
mode filtering or user-prompt shaping.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Literal, NamedTuple

from pydantic import BaseModel, Field, ValidationError

from kernel.models import (
    FrontDoorPromptProfile,
    OptionalContextLane,
    RouteDecision,
    RouteDecisionAlternative,
    RoutePlannerSource,
    SurfaceContext,
)
from kernel.operations.structured_output import parse_structured_payload

__all__ = [
    "build_deterministic_route_decision",
    "build_frontdoor_user_prompt",
    "build_route_catalog_markdown",
    "frontdoor_mode_ids",
    "looks_like_model_configuration_question",
    "plan_frontdoor_route",
    "sanitize_optional_context_policy",
    "select_prompt_profile",
    "shorten_description",
]

_MODEL_FAMILY_RE = re.compile(
    r"\b(?:gpt[\w.-]*|gemini[\w.-]*|claude[\w.-]*|o1|o3|o4)\b",
    re.IGNORECASE,
)
_MODEL_CONFIG_CUE_RE = re.compile(
    r"\b(?:"
    r"default\s+chat|"
    r"which\s+model|"
    r"what\s+model|"
    r"set\s+.*default|"
    r"switch\s+.*(?:to|from)|"
    r"should\s+i\s+switch|"
    r"when\s+(?:should|if).*switch|"
    r"how\s+about\s+for|"
    r"model\s+(?:choice|routing|selection)"
    r")\b",
    re.IGNORECASE,
)
_FAST_PATH_MULTI_INTENT_RE = re.compile(
    r"\b(?:and|also|plus|as\s+well\s+as|along\s+with|then|after\s+that|vs\.?|versus)\b",
    re.IGNORECASE,
)
_OPTIONAL_CONTEXT_LANE_ORDER: tuple[OptionalContextLane, ...] = (
    "reflection",
    "passive_intelligence",
    "memory_recall",
    "architecture_context",
    "capability_snapshot",
)
_FRONTDOOR_ROUTE_CATALOG_PLACEHOLDER = "{{FRONTDOOR_ROUTE_CATALOG}}"
_FRONTDOOR_ROUTE_HINTS_PLACEHOLDER = "{{FRONTDOOR_ROUTE_HINTS}}"
_FRONTDOOR_ROUTE_EXAMPLES_PLACEHOLDER = "{{FRONTDOOR_ROUTE_EXAMPLES}}"

_WorkflowCatalog = Mapping[str, Any]
_ModeSupported = Callable[[str, Any], bool]
_LoadRoutePrompt = Callable[[], str]
_LoadFastPathVerifierPrompt = Callable[[], str]
_ResolvePlannerModel = Callable[[], Any]
_CompleteFn = Callable[..., Awaitable[Any]]
_UserPromptBuilder = Callable[..., str]


class _FastPathVerification(BaseModel):
    """Bounded verifier output for one high-precision front-door candidate."""

    outcome: Literal["confirm", "abstain"] = "abstain"
    normalized_query: str = ""
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning: str = ""


class _FastPathCandidate(NamedTuple):
    mode_id: str
    reasoning: str


def looks_like_model_configuration_question(message: str) -> bool:
    """Return whether a turn asks about model selection or routing preferences."""

    normalized = " ".join(message.strip().split())
    if not normalized:
        return False
    if "default chat" in normalized.lower() and _MODEL_FAMILY_RE.search(normalized):
        return True
    if _MODEL_CONFIG_CUE_RE.search(normalized) and _MODEL_FAMILY_RE.search(normalized):
        return True
    return re.search(r"\b(?:which|what)\s+model\b", normalized, re.IGNORECASE) is not None


def shorten_description(description: str) -> str:
    """Return the first sentence of one mode description for prompt catalogs."""

    cleaned = " ".join(description.strip().split())
    if not cleaned:
        return ""
    sentences = [part.strip() for part in cleaned.split(".") if part.strip()]
    if not sentences:
        return cleaned
    first = sentences[0]
    return first if first.endswith(".") else f"{first}."


def select_prompt_profile(model_id: str) -> FrontDoorPromptProfile:
    """Return the prompt verbosity that best fits one planner model."""

    lowered = model_id.strip().lower()
    if "nano" in lowered:
        return "compact"
    if "mini" in lowered:
        return "standard"
    return "rich"


def _dedupe_optional_context_lanes(lanes: list[OptionalContextLane]) -> list[OptionalContextLane]:
    requested = set(lanes)
    return [lane for lane in _OPTIONAL_CONTEXT_LANE_ORDER if lane in requested]


def sanitize_optional_context_policy(
    *,
    include: list[OptionalContextLane],
    exclude: list[OptionalContextLane],
) -> tuple[list[OptionalContextLane], list[OptionalContextLane]]:
    """Normalize optional context hints while preferring include over exclude."""

    sanitized_include = _dedupe_optional_context_lanes(include)
    sanitized_exclude: list[OptionalContextLane] = [
        lane for lane in _dedupe_optional_context_lanes(exclude) if lane not in sanitized_include
    ]
    return sanitized_include, sanitized_exclude


def frontdoor_mode_ids(
    *,
    catalog: _WorkflowCatalog,
    mode_supported: _ModeSupported | None = None,
) -> list[str]:
    """Return front-door-routable chat and action modes in priority order."""

    ordered_modes = sorted(
        (
            (mode_id, mode_cfg)
            for mode_id, mode_cfg in catalog.items()
            if getattr(mode_cfg, "mode_kind", "") in {"chat", "action"}
            and bool(getattr(mode_cfg, "frontdoor_routable", False))
            and (mode_supported is None or mode_supported(mode_id, mode_cfg))
        ),
        key=lambda item: (-int(getattr(item[1], "frontdoor_priority", 0) or 0), item[0]),
    )
    return [mode_id for mode_id, _mode_cfg in ordered_modes]


def build_route_catalog_markdown(
    profile: FrontDoorPromptProfile,
    *,
    catalog: _WorkflowCatalog,
    mode_supported: _ModeSupported | None = None,
) -> str:
    """Render one compact markdown summary of routable modes."""

    lines = ["## Routable Modes"]
    for mode_id in frontdoor_mode_ids(catalog=catalog, mode_supported=mode_supported):
        if mode_id == "default":
            lines.append(
                "- default: ordinary conversation when no specialized one-shot "
                "mode clearly owns the request."
            )
            continue

        mode_cfg = catalog.get(mode_id)
        if mode_cfg is None:
            continue
        description = shorten_description(str(getattr(mode_cfg, "description", "") or ""))
        command_hint = f"/{mode_id}"
        if profile == "compact":
            lines.append(f"- {mode_id}: {description}")
            continue

        line = (
            f"- {mode_id}: {description} "
            f"Mode kind: {getattr(mode_cfg, 'mode_kind', '')}. "
            f"Query required: {'yes' if bool(getattr(mode_cfg, 'require_query', False)) else 'no'}."
        )
        if profile == "rich":
            line += f" Direct command: {command_hint}."
        lines.append(line)
    return "\n".join(lines)


def _mode_frontdoor_route_hints(mode_cfg: Any) -> list[str]:
    raw_hints = getattr(mode_cfg, "frontdoor_route_hints", [])
    if not isinstance(raw_hints, list):
        return []

    hints: list[str] = []
    for raw_hint in raw_hints:
        if not isinstance(raw_hint, str):
            continue
        hint = raw_hint.strip()
        if hint:
            hints.append(hint)
    return hints


def _mode_frontdoor_examples(mode_cfg: Any) -> list[str]:
    raw_examples = getattr(mode_cfg, "frontdoor_examples", [])
    if not isinstance(raw_examples, list):
        return []

    examples: list[str] = []
    for raw_example in raw_examples:
        if not isinstance(raw_example, str):
            continue
        example = raw_example.strip()
        if example:
            examples.append(example)
    return examples


def build_route_hints_markdown(
    *,
    catalog: _WorkflowCatalog,
    mode_supported: _ModeSupported | None = None,
) -> str:
    """Render route-hint bullets from mode-authored front-matter hints."""

    lines: list[str] = ["## Routing Hints"]
    for mode_id in frontdoor_mode_ids(catalog=catalog, mode_supported=mode_supported):
        mode_cfg = catalog.get(mode_id)
        if mode_cfg is None:
            continue
        for hint in _mode_frontdoor_route_hints(mode_cfg):
            lines.append(f"- {hint}")

    if len(lines) == 1:
        return ""
    return "\n".join(lines)


def build_route_examples_markdown(
    *,
    catalog: _WorkflowCatalog,
    mode_supported: _ModeSupported | None = None,
) -> str:
    """Render route examples sourced from mode front-matter."""

    examples: list[str] = []
    for mode_id in frontdoor_mode_ids(catalog=catalog, mode_supported=mode_supported):
        mode_cfg = catalog.get(mode_id)
        if mode_cfg is None:
            continue
        examples.extend(_mode_frontdoor_examples(mode_cfg))

    if not examples:
        return ""

    lines: list[str] = ["## Routing Examples"]
    for index, example in enumerate(examples, start=1):
        lines.append(f"Example {index}:")
        lines.append(example)
        if index < len(examples):
            lines.append("")
    return "\n".join(lines)


def build_frontdoor_system_prompt(
    *,
    base_prompt: str,
    prompt_profile: FrontDoorPromptProfile,
    catalog: _WorkflowCatalog,
    mode_supported: _ModeSupported | None = None,
) -> str:
    """Compose the router system prompt from boilerplate plus mode metadata."""

    prompt = base_prompt.strip()
    catalog_markdown = build_route_catalog_markdown(
        prompt_profile,
        catalog=catalog,
        mode_supported=mode_supported,
    )
    hints_markdown = build_route_hints_markdown(
        catalog=catalog,
        mode_supported=mode_supported,
    )
    examples_markdown = build_route_examples_markdown(
        catalog=catalog,
        mode_supported=mode_supported,
    )

    if _FRONTDOOR_ROUTE_CATALOG_PLACEHOLDER in prompt:
        prompt = prompt.replace(_FRONTDOOR_ROUTE_CATALOG_PLACEHOLDER, catalog_markdown)
    elif catalog_markdown:
        prompt = "\n\n".join(part for part in (prompt, catalog_markdown) if part)

    if _FRONTDOOR_ROUTE_HINTS_PLACEHOLDER in prompt:
        prompt = prompt.replace(_FRONTDOOR_ROUTE_HINTS_PLACEHOLDER, hints_markdown)
    elif hints_markdown:
        prompt = "\n\n".join(part for part in (prompt, hints_markdown) if part)

    if _FRONTDOOR_ROUTE_EXAMPLES_PLACEHOLDER in prompt:
        prompt = prompt.replace(_FRONTDOOR_ROUTE_EXAMPLES_PLACEHOLDER, examples_markdown)
    elif examples_markdown:
        prompt = "\n\n".join(part for part in (prompt, examples_markdown) if part)

    return prompt.strip()


def build_frontdoor_user_prompt(
    *,
    normalized_message: str,
    surface_context: SurfaceContext,
    preferred_mode_id: str,
    conversation_excerpt_markdown: str,
    fast_path_candidate_mode_id: str = "",
    is_default_executor: bool = False,
) -> str:
    """Build the default host-neutral front-door user prompt."""

    user_prompt_parts = [
        f"Surface type: {surface_context.surface_type}",
        f"Memory scope: {surface_context.memory_scope}",
        f"Preferred default mode: {preferred_mode_id or 'none'}",
        f"Is default executor: {str(is_default_executor).lower()}",
    ]
    if fast_path_candidate_mode_id:
        user_prompt_parts.append(f"Fast-path candidate mode: {fast_path_candidate_mode_id}")
    user_prompt_parts.append("")
    if conversation_excerpt_markdown:
        user_prompt_parts.extend(
            ["Recent conversation context:", conversation_excerpt_markdown, ""]
        )
    user_prompt_parts.extend(["User message:", normalized_message])
    return "\n".join(user_prompt_parts)


def _has_fast_path_disqualifier(message: str) -> bool:
    normalized = " ".join(message.strip().split())
    if not normalized:
        return True
    if _FAST_PATH_MULTI_INTENT_RE.search(normalized):
        return True
    if normalized.count("?") > 1:
        return True
    return ";" in normalized


def _compile_fast_path_patterns(mode_cfg: Any) -> list[re.Pattern[str]]:
    raw_patterns = getattr(mode_cfg, "frontdoor_fast_path_patterns", [])
    if not isinstance(raw_patterns, list):
        return []

    compiled: list[re.Pattern[str]] = []
    for raw_pattern in raw_patterns:
        if not isinstance(raw_pattern, str):
            continue
        pattern = raw_pattern.strip()
        if not pattern:
            continue
        try:
            compiled.append(re.compile(pattern, re.IGNORECASE))
        except re.error:
            continue
    return compiled


def _fast_path_reasoning(*, mode_id: str, mode_cfg: Any) -> str:
    description = shorten_description(str(getattr(mode_cfg, "description", "") or ""))
    if description:
        return f"Mode-authored fast-path pattern matched {mode_id}: {description}"
    return f"Mode-authored fast-path pattern matched {mode_id}."


def _select_fast_path_candidate(
    message: str,
    *,
    mode_order: list[str],
    catalog: _WorkflowCatalog,
) -> _FastPathCandidate | None:
    normalized = " ".join(message.strip().split())
    if _has_fast_path_disqualifier(normalized):
        return None

    for mode_id in mode_order:
        mode_cfg = catalog.get(mode_id)
        if mode_cfg is None:
            continue
        if any(pattern.search(normalized) for pattern in _compile_fast_path_patterns(mode_cfg)):
            return _FastPathCandidate(
                mode_id=mode_id,
                reasoning=_fast_path_reasoning(mode_id=mode_id, mode_cfg=mode_cfg),
            )
    return None


def _build_fast_path_verifier_system_prompt(
    base_prompt: str,
    candidate_mode_id: str,
    *,
    catalog: _WorkflowCatalog,
) -> str:
    mode_cfg = catalog.get(candidate_mode_id)
    description = shorten_description(str(getattr(mode_cfg, "description", "") or ""))
    query_required = "yes" if bool(getattr(mode_cfg, "require_query", False)) else "no"

    prompt = base_prompt.strip()
    if not prompt:
        raise ValueError("Fast-path verifier prompt was not supplied.")

    candidate_lines = [
        f"Candidate mode: {candidate_mode_id}.",
        f"Candidate requires query: {query_required}.",
    ]
    if description:
        candidate_lines.insert(1, f"Candidate description: {description}")

    return "\n\n".join((prompt, "\n".join(candidate_lines)))


async def _try_fast_path_route(
    *,
    normalized_message: str,
    ordered_mode_ids: list[str],
    model_def: Any,
    prompt_profile: FrontDoorPromptProfile,
    surface_context: SurfaceContext,
    preferred_mode_id: str,
    conversation_excerpt_markdown: str,
    fast_path_verifier_prompt: str,
    build_user_prompt: _UserPromptBuilder,
    complete: _CompleteFn,
    catalog: _WorkflowCatalog,
) -> RouteDecision | None:
    candidate = _select_fast_path_candidate(
        normalized_message,
        mode_order=ordered_mode_ids,
        catalog=catalog,
    )
    if candidate is None:
        return None

    system_prompt = _build_fast_path_verifier_system_prompt(
        fast_path_verifier_prompt,
        candidate.mode_id,
        catalog=catalog,
    )

    try:
        completion = await complete(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": build_user_prompt(
                        normalized_message=normalized_message,
                        surface_context=surface_context,
                        preferred_mode_id=preferred_mode_id,
                        conversation_excerpt_markdown=conversation_excerpt_markdown,
                        fast_path_candidate_mode_id=candidate.mode_id,
                    ),
                },
            ],
            model_def=model_def,
            max_tokens=220,
            temperature=0.0,
            response_format=_FastPathVerification,
        )
        verification = parse_structured_payload(
            _FastPathVerification, str(completion.content or "")
        )
        assert isinstance(verification, _FastPathVerification)
    except (ValidationError, ValueError, RuntimeError):
        return None

    if verification.outcome != "confirm":
        return None

    return RouteDecision(
        selected_mode_id=candidate.mode_id,
        normalized_query=(verification.normalized_query or normalized_message).strip(),
        confidence_score=max(verification.confidence_score, 0.9),
        reasoning=verification.reasoning.strip() or candidate.reasoning,
        planner_source="fast_path_verified",
        prompt_profile=prompt_profile,
    )


async def _run_model_bounded_route(
    *,
    normalized_message: str,
    current_route_mode: str,
    surface_context: SurfaceContext,
    preferred_mode_id: str,
    conversation_excerpt_markdown: str,
    model_def: Any,
    prompt_profile: FrontDoorPromptProfile,
    allowed_mode_ids: set[str],
    load_route_prompt: _LoadRoutePrompt,
    build_user_prompt: _UserPromptBuilder,
    complete: _CompleteFn,
    catalog: _WorkflowCatalog,
    completion_max_tokens: int,
    is_default_executor: bool = False,
    mode_supported: _ModeSupported | None = None,
) -> RouteDecision:
    fallback = build_deterministic_route_decision(
        selected_mode_id=current_route_mode or "default",
        normalized_query=normalized_message,
        reasoning="Fell back to the baseline route.",
    )
    system_prompt = build_frontdoor_system_prompt(
        base_prompt=load_route_prompt(),
        prompt_profile=prompt_profile,
        catalog=catalog,
        mode_supported=mode_supported,
    )
    user_prompt = build_user_prompt(
        normalized_message=normalized_message,
        surface_context=surface_context,
        preferred_mode_id=preferred_mode_id,
        conversation_excerpt_markdown=conversation_excerpt_markdown,
        is_default_executor=is_default_executor,
    )

    try:
        completion = await complete(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model_def=model_def,
            max_tokens=completion_max_tokens,
            temperature=0.1,
            response_format=RouteDecision,
        )
        decision_payload = parse_structured_payload(
            RouteDecision, str(completion.content or "")
        )
        decision = RouteDecision.model_validate(decision_payload.model_dump())
    except (ValidationError, ValueError, RuntimeError):
        return fallback

    selected_mode_id = decision.selected_mode_id.strip().lower() or "default"
    if selected_mode_id not in allowed_mode_ids:
        selected_mode_id = "default"

    alternatives = [
        RouteDecisionAlternative(
            mode_id=item.mode_id.strip().lower(),
            confidence_score=item.confidence_score,
        )
        for item in decision.alternatives
        if item.mode_id.strip().lower() in allowed_mode_ids
        and item.mode_id.strip().lower() != selected_mode_id
    ]

    clarification_question = decision.clarification_question.strip()
    if decision.clarification_needed and not clarification_question:
        clarification_question = (
            "I can route this, but I need one detail first: do you want status, "
            "investigation, message grounding, or a durable action?"
        )

    optional_context_include, optional_context_exclude = sanitize_optional_context_policy(
        include=decision.optional_context_include,
        exclude=decision.optional_context_exclude,
    )

    return RouteDecision(
        selected_mode_id=selected_mode_id,
        normalized_query=(decision.normalized_query or normalized_message).strip(),
        confidence_score=decision.confidence_score,
        reasoning=decision.reasoning.strip(),
        alternatives=alternatives,
        clarification_needed=decision.clarification_needed,
        clarification_question=clarification_question,
        planner_source="model_bounded",
        prompt_profile=prompt_profile,
        optional_context_include=optional_context_include,
        optional_context_exclude=optional_context_exclude,
    )


def build_deterministic_route_decision(
    *,
    selected_mode_id: str,
    normalized_query: str,
    planner_source: RoutePlannerSource = "deterministic",
    reasoning: str = "",
) -> RouteDecision:
    """Build one non-model route decision for explicit or preserved paths."""

    return RouteDecision(
        selected_mode_id=selected_mode_id or "default",
        normalized_query=normalized_query.strip(),
        confidence_score=1.0,
        reasoning=reasoning,
        planner_source=planner_source,
        prompt_profile="standard",
    )


async def plan_frontdoor_route(
    *,
    user_message: str,
    current_route_mode: str,
    surface_context: SurfaceContext,
    catalog: _WorkflowCatalog,
    load_route_prompt: _LoadRoutePrompt,
    resolve_planner_model: _ResolvePlannerModel,
    complete: _CompleteFn,
    preferred_mode_id: str = "",
    conversation_excerpt_markdown: str = "",
    mode_supported: _ModeSupported | None = None,
    build_user_prompt: _UserPromptBuilder = build_frontdoor_user_prompt,
    load_fast_path_verifier_prompt: _LoadFastPathVerifierPrompt | None = None,
    model_bounded_max_tokens: int = 500,
    is_default_executor: bool = False,
) -> RouteDecision:
    """Choose one owning mode for a natural-language turn using host hooks."""

    normalized_message = user_message.strip()
    fallback = build_deterministic_route_decision(
        selected_mode_id=current_route_mode or "default",
        normalized_query=normalized_message,
        reasoning="Fell back to the baseline route.",
    )
    if not normalized_message:
        return fallback

    if looks_like_model_configuration_question(normalized_message):
        return build_deterministic_route_decision(
            selected_mode_id="default",
            normalized_query=normalized_message,
            reasoning="Kept model selection/configuration request on default chat route.",
        )

    ordered_mode_ids = frontdoor_mode_ids(
        catalog=catalog,
        mode_supported=mode_supported,
    )
    allowed_mode_ids = set(ordered_mode_ids)
    if not allowed_mode_ids:
        return fallback

    try:
        model_def = resolve_planner_model()
    except (KeyError, RuntimeError, ValueError):
        return fallback

    fast_path_verifier_prompt = ""
    if load_fast_path_verifier_prompt is not None:
        try:
            fast_path_verifier_prompt = str(load_fast_path_verifier_prompt() or "").strip()
        except (FileNotFoundError, OSError, RuntimeError, ValueError):
            fast_path_verifier_prompt = ""

    prompt_profile = select_prompt_profile(str(getattr(model_def, "model_id", "") or ""))
    fast_path_decision = await _try_fast_path_route(
        normalized_message=normalized_message,
        ordered_mode_ids=ordered_mode_ids,
        model_def=model_def,
        prompt_profile=prompt_profile,
        surface_context=surface_context,
        preferred_mode_id=preferred_mode_id,
        conversation_excerpt_markdown=conversation_excerpt_markdown,
        fast_path_verifier_prompt=fast_path_verifier_prompt,
        build_user_prompt=build_user_prompt,
        complete=complete,
        catalog=catalog,
    )
    if fast_path_decision is not None:
        return fast_path_decision

    return await _run_model_bounded_route(
        normalized_message=normalized_message,
        current_route_mode=current_route_mode,
        surface_context=surface_context,
        preferred_mode_id=preferred_mode_id,
        conversation_excerpt_markdown=conversation_excerpt_markdown,
        model_def=model_def,
        prompt_profile=prompt_profile,
        allowed_mode_ids=allowed_mode_ids,
        load_route_prompt=load_route_prompt,
        build_user_prompt=build_user_prompt,
        complete=complete,
        catalog=catalog,
        completion_max_tokens=model_bounded_max_tokens,
        is_default_executor=is_default_executor,
        mode_supported=mode_supported,
    )

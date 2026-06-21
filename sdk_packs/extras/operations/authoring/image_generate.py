"""Generate one image on demand with requested format and size hints."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from pydantic import AliasChoices, BaseModel, Field

from kernel.models import OperationExecutionContext, OperationResult

_DEFAULT_IMAGE_SIZE = "1024x1024"
_LANDSCAPE_IMAGE_SIZE = "1536x1024"
_PORTRAIT_IMAGE_SIZE = "1024x1536"
_SUPPORTED_IMAGE_SIZES = {
    _DEFAULT_IMAGE_SIZE,
    _LANDSCAPE_IMAGE_SIZE,
    _PORTRAIT_IMAGE_SIZE,
}
_IMAGE_SIZE_ALIASES = {
    "1:1": _DEFAULT_IMAGE_SIZE,
    "square": _DEFAULT_IMAGE_SIZE,
    "16:9": _LANDSCAPE_IMAGE_SIZE,
    "4:3": _LANDSCAPE_IMAGE_SIZE,
    "3:2": _LANDSCAPE_IMAGE_SIZE,
    "21:9": _LANDSCAPE_IMAGE_SIZE,
    "landscape": _LANDSCAPE_IMAGE_SIZE,
    "wide": _LANDSCAPE_IMAGE_SIZE,
    "horizontal": _LANDSCAPE_IMAGE_SIZE,
    "banner": _LANDSCAPE_IMAGE_SIZE,
    "widescreen": _LANDSCAPE_IMAGE_SIZE,
    "9:16": _PORTRAIT_IMAGE_SIZE,
    "4:5": _PORTRAIT_IMAGE_SIZE,
    "3:4": _PORTRAIT_IMAGE_SIZE,
    "2:3": _PORTRAIT_IMAGE_SIZE,
    "portrait": _PORTRAIT_IMAGE_SIZE,
    "tall": _PORTRAIT_IMAGE_SIZE,
    "vertical": _PORTRAIT_IMAGE_SIZE,
}
_IMAGE_OUTPUT_FORMAT_ALIASES = {
    "png": "png",
    "jpg": "jpeg",
    "jpeg": "jpeg",
    "gif": "gif",
}
_SIZE_PATTERN = re.compile(
    r"\b(?P<width>\d{3,4})\s*(?:x|×|by)\s*(?P<height>\d{3,4})\b",
    re.IGNORECASE,
)
_RATIO_PATTERN = re.compile(r"\b(?P<width>\d{1,2})\s*:\s*(?P<height>\d{1,2})\b")
_ORIENTATION_PATTERNS = (
    (re.compile(r"\b(?:square)\b", re.IGNORECASE), _DEFAULT_IMAGE_SIZE),
    (
        re.compile(r"\b(?:portrait|tall|taller|vertical|story)\b", re.IGNORECASE),
        _PORTRAIT_IMAGE_SIZE,
    ),
    (
        re.compile(
            r"\b(?:landscape|wide|wider|horizontal|banner|widescreen|hero)\b",
            re.IGNORECASE,
        ),
        _LANDSCAPE_IMAGE_SIZE,
    ),
)
_OUTPUT_FORMAT_PATTERN = re.compile(r"\b(png|jpe?g|gif)\b", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class ResolvedImageRequest:
    output_format: str
    format_label: str
    size: str


class ImagePromptRewrite(BaseModel):
    """Structured prompt-normalization result for context-dependent image asks."""

    prompt: str


def _normalize_output_format(value: str) -> str:
    normalized = value.strip().lower().lstrip(".")
    return _IMAGE_OUTPUT_FORMAT_ALIASES.get(normalized, "")


def _extract_output_format(prompt_text: str) -> str:
    matches = list(_OUTPUT_FORMAT_PATTERN.finditer(prompt_text))
    if not matches:
        return ""
    return _normalize_output_format(matches[-1].group(1))


def _normalize_size_token(value: str) -> str:
    cleaned = value.strip().lower().replace(" ", "")
    if not cleaned:
        return ""
    if cleaned in _IMAGE_SIZE_ALIASES:
        return _IMAGE_SIZE_ALIASES[cleaned]

    ratio_match = _RATIO_PATTERN.search(cleaned)
    if ratio_match is not None:
        width = int(ratio_match.group("width"))
        height = int(ratio_match.group("height"))
        if width == height:
            return _DEFAULT_IMAGE_SIZE
        if width > height:
            return _LANDSCAPE_IMAGE_SIZE
        return _PORTRAIT_IMAGE_SIZE

    size_match = _SIZE_PATTERN.search(cleaned)
    if size_match is None:
        return ""
    width = int(size_match.group("width"))
    height = int(size_match.group("height"))
    normalized = f"{width}x{height}"
    if normalized in _SUPPORTED_IMAGE_SIZES:
        return normalized
    if width == height:
        return _DEFAULT_IMAGE_SIZE
    if width > height:
        return _LANDSCAPE_IMAGE_SIZE
    return _PORTRAIT_IMAGE_SIZE


def _extract_size(prompt_text: str) -> str:
    explicit_size = _normalize_size_token(prompt_text)
    if explicit_size:
        return explicit_size
    for pattern, mapped_size in _ORIENTATION_PATTERNS:
        if pattern.search(prompt_text):
            return mapped_size
    return ""


def _format_label(output_format: str) -> str:
    if output_format == "jpeg":
        return "JPEG"
    return output_format.upper()


def _previous_request_value(previous_request: dict[str, Any] | None, key: str) -> str:
    if not previous_request:
        return ""
    return str(previous_request.get(key) or "")


def _resolve_image_request(
    parameters: Parameters,
    prompt_text: str,
    previous_request: dict[str, Any] | None = None,
) -> ResolvedImageRequest:
    output_format = _normalize_output_format(parameters.output_format)
    if not output_format:
        output_format = _extract_output_format(prompt_text)
    if not output_format:
        output_format = _normalize_output_format(
            _previous_request_value(previous_request, "output_format")
        )
    if not output_format:
        output_format = "png"

    size = _normalize_size_token(parameters.size)
    if not size and parameters.aspect_ratio:
        size = _normalize_size_token(parameters.aspect_ratio)
    if not size:
        size = _extract_size(prompt_text)
    if not size:
        size = _normalize_size_token(_previous_request_value(previous_request, "size"))
    if not size:
        size = _DEFAULT_IMAGE_SIZE

    return ResolvedImageRequest(
        output_format=output_format,
        format_label=_format_label(output_format),
        size=size,
    )


_FOLLOW_UP_PATTERN = re.compile(
    r"\b(?:it|that|this|same|again|redo|regenerate|revise|change|adjust|make|turn)\b",
    re.IGNORECASE,
)
_CONTEXT_REFERENCE_PATTERN = re.compile(
    r"\b(?:context|earlier|above|previous|we\s+(?:said|discussed|joked)|joke|bit)\b",
    re.IGNORECASE,
)
_TEXT_ARTIFACT_REFERENCE_PATTERN = re.compile(
    r"\b(?:that|this|the|above|earlier|previous)\s+"
    r"(?:description|prompt|concept|idea|direction|version)\b",
    re.IGNORECASE,
)
_CONTEXTUAL_IMAGE_CONFIRMATION_PATTERN = re.compile(
    r"\b(?:generate|create|make|render|draw|illustrate)\s+"
    r"(?:it|that|this|an?\s+(?:image|picture|portrait|illustration|photo|logo|icon|avatar)|"
    r"the\s+(?:image|picture|portrait|illustration|photo|logo|icon|avatar))\b",
    re.IGNORECASE,
)
_JSONISH_PROMPT_OBJECT_PATTERN = re.compile(
    r"\{[^{}]*[\"']prompt[\"']\s*:\s*[\"'][^{}]*\}",
    re.IGNORECASE | re.DOTALL,
)
_PROMPT_FIELD_PATTERN = re.compile(
    r"[\"']prompt[\"']\s*:\s*[\"'](?P<prompt>.*?)(?<!\\\\)[\"']",
    re.IGNORECASE | re.DOTALL,
)
_PROMPT_LINE_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:[-*]\s*)?(?:image\s+|generation\s+)?prompt\s*:\s*"
    r"(?P<prompt>.+?)(?=\n\s*(?:###|[-*]\s+|[\"']?\w+[\"']?\s*:)|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def _recent_image_handoff(ctx: OperationExecutionContext) -> dict[str, Any]:
    payload = ctx.extra.get("recent_image_handoff")
    return payload if isinstance(payload, dict) else {}


def _previous_image_request(ctx: OperationExecutionContext) -> dict[str, Any]:
    handoff = _recent_image_handoff(ctx)
    artifact_data = handoff.get("artifact_data")
    if not isinstance(artifact_data, dict):
        return {}
    image_request = artifact_data.get("image_request")
    return image_request if isinstance(image_request, dict) else {}


def _clean_extracted_prompt(value: str) -> str:
    prompt = " ".join(value.strip().strip("\"'").split())
    return prompt.rstrip(",")


def _extract_recent_context_prompt(conversation_excerpt_markdown: str) -> str:
    """Return the newest explicit image prompt from recent assistant context."""

    if not conversation_excerpt_markdown.strip():
        return ""

    prompt_object_matches = list(
        _JSONISH_PROMPT_OBJECT_PATTERN.finditer(conversation_excerpt_markdown)
    )
    for object_match in reversed(prompt_object_matches):
        candidate = object_match.group(0)
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            prompt = payload.get("prompt")
            if isinstance(prompt, str) and prompt.strip():
                return _clean_extracted_prompt(prompt)

        field_match = _PROMPT_FIELD_PATTERN.search(candidate)
        if field_match is not None:
            prompt = _clean_extracted_prompt(field_match.group("prompt"))
            if prompt:
                return prompt

    prompt_line_matches = list(_PROMPT_LINE_PATTERN.finditer(conversation_excerpt_markdown))
    for line_match in reversed(prompt_line_matches):
        prompt = _clean_extracted_prompt(line_match.group("prompt"))
        if prompt:
            return prompt

    return ""


def _should_normalize_prompt(
    *,
    prompt_text: str,
    previous_request: dict[str, Any],
    conversation_excerpt_markdown: str,
) -> bool:
    if previous_request and _FOLLOW_UP_PATTERN.search(prompt_text):
        return True
    if not conversation_excerpt_markdown:
        return False
    return any(
        pattern.search(prompt_text)
        for pattern in (
            _CONTEXT_REFERENCE_PATTERN,
            _TEXT_ARTIFACT_REFERENCE_PATTERN,
            _CONTEXTUAL_IMAGE_CONFIRMATION_PATTERN,
        )
    )


def _fallback_contextual_prompt(
    *,
    prompt_text: str,
    previous_request: dict[str, Any],
    conversation_excerpt_markdown: str,
) -> str:
    previous_prompt = _previous_request_value(previous_request, "prompt")
    parts = ["Create one image from this request."]
    if previous_prompt:
        parts.append(f"Previous image prompt: {previous_prompt}")
    if conversation_excerpt_markdown.strip():
        excerpt = conversation_excerpt_markdown.strip()[:2400]
        parts.append(f"Recent conversation context:\n{excerpt}")
    parts.append(f"Current request: {prompt_text}")
    return "\n\n".join(parts)


async def _normalize_prompt_text(
    *,
    ctx: OperationExecutionContext,
    prompt_text: str,
    previous_request: dict[str, Any],
) -> str:
    conversation_excerpt_markdown = ctx.conversation_excerpt_markdown or ""
    if not _should_normalize_prompt(
        prompt_text=prompt_text,
        previous_request=previous_request,
        conversation_excerpt_markdown=conversation_excerpt_markdown,
    ):
        return prompt_text

    fallback_prompt = _fallback_contextual_prompt(
        prompt_text=prompt_text,
        previous_request=previous_request,
        conversation_excerpt_markdown=conversation_excerpt_markdown,
    )
    if not previous_request:
        extracted_prompt = _extract_recent_context_prompt(conversation_excerpt_markdown)
        if extracted_prompt:
            return extracted_prompt

    if not callable(ctx.complete_fn) or not callable(ctx.resolve_model_fn):
        return fallback_prompt

    model_def = ctx.resolve_model_fn(ctx.preferred_model_id or "gpt-5.4-mini")
    if model_def is None:
        return fallback_prompt

    try:
        completion = await ctx.complete_fn(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Rewrite conversational image requests into one concrete image prompt. "
                        "Use only the supplied recent context and previous image request. "
                        "Do not invent identity details for phrases like self portrait unless the "
                        "context explicitly defines them. Return only the normalized prompt."
                    ),
                },
                {
                    "role": "user",
                    "content": fallback_prompt,
                },
            ],
            model_def=model_def,
            max_tokens=700,
            temperature=0.1,
            response_format=ImagePromptRewrite,
        )
        rewritten = ImagePromptRewrite.model_validate_json(completion.content).prompt.strip()
    except Exception:  # pylint: disable=broad-except
        return fallback_prompt
    return rewritten or fallback_prompt


def _image_request_payload(
    *,
    prompt: str,
    source_query: str,
    request: ResolvedImageRequest,
) -> dict[str, str]:
    return {
        "prompt": prompt,
        "source_query": source_query,
        "output_format": request.output_format,
        "format_label": request.format_label,
        "size": request.size,
    }


class Parameters(BaseModel):
    """Optional structured parameters for image generation."""

    prompt: str = Field(
        default="",
        validation_alias=AliasChoices("prompt", "query", "request", "description"),
    )
    output_format: str = Field(
        default="",
        validation_alias=AliasChoices("output_format", "format", "file_format", "extension"),
    )
    size: str = Field(
        default="",
        validation_alias=AliasChoices("size", "dimensions", "resolution"),
    )
    aspect_ratio: str = Field(
        default="",
        validation_alias=AliasChoices("aspect_ratio", "ratio", "aspect"),
    )


async def execute(ctx: OperationExecutionContext) -> OperationResult:
    """Generate one image, persist the artifact, and return delivery metadata."""

    parameters = Parameters.model_validate(ctx.params or {})
    prompt_text = (parameters.prompt or "").strip() or (ctx.query or "").strip()
    if not prompt_text:
        return OperationResult(
            output_text=(
                "Please describe the image you want. "
                "Example: /image a neon skyline as a jpeg in portrait 9:16."
            )
        )

    previous_request = _previous_image_request(ctx)
    request = _resolve_image_request(parameters, prompt_text, previous_request)
    normalized_prompt = await _normalize_prompt_text(
        ctx=ctx,
        prompt_text=prompt_text,
        previous_request=previous_request,
    )

    if not callable(ctx.generate_image_fn):
        return OperationResult(
            output_text="Image generation unavailable: image generator not injected."
        )

    if ctx.blob_io is None:
        return OperationResult(
            output_text="Image generation unavailable: blob storage not injected."
        )

    try:
        generated = await ctx.generate_image_fn(
            normalized_prompt,
            user=ctx.user_id or ctx.user_slug,
            size=request.size,
            output_format=request.output_format,
        )
        artifact_ref, download_url = ctx.blob_io.store_generated_image(
            generated.image_bytes,
            workspace_id=ctx.workspace_id,
            memory_scope=ctx.memory_scope,
            surface_id=ctx.surface_id,
            user_slug=ctx.user_slug,
            user_id=ctx.user_id,
            prompt=normalized_prompt,
            content_type=generated.content_type,
            pixel_size=request.size,
        )
    except Exception as exc:  # pylint: disable=broad-except
        return OperationResult(output_text=f"Image generation failed: {type(exc).__name__}: {exc}")

    return OperationResult(
        output_text=f"Generated {request.format_label} ({request.size}) for: {normalized_prompt}",
        output_json={
            "generated_image": artifact_ref.model_dump(),
            "image_request": _image_request_payload(
                prompt=normalized_prompt,
                source_query=prompt_text,
                request=request,
            ),
        },
        extra={
            "generated_image_delivery": {
                "file_name": artifact_ref.file_name,
                "content_type": artifact_ref.content_type,
                "image_bytes": generated.image_bytes,
                "download_url": download_url,
                "pixel_size": request.size,
            }
        },
    )

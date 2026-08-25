"""Google Gemini provider gateway.

Implements the kernel's ``ProviderGateway`` and ``EmbeddingGateway`` protocols
for the Google Gemini REST API (generativelanguage.googleapis.com). Uses
``httpx`` directly; no Google SDK dependency.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from kernel.provider_contracts import (
    CompletionRequest,
    CompletionResult,
    CompletionUsage,
    EmbeddingRequest,
    EmbeddingResult,
    ProviderEventEmitter,
    ProviderStreamEvent,
    StreamedChunk,
    embedding_gateway_error_from_exception,
    extract_request_id,
    is_pydantic_model_class,
    provider_gateway_error_from_exception,
)
from kernel.token_usage import coerce_non_negative_int
from sdk.providers.openai_gateway import (
    _normalize_openai_tool_arguments,
    normalize_structured_content,
)

_JSON_ONLY_INSTRUCTION = "Return only a valid JSON object with no Markdown wrapper."


class GoogleGeminiGateway:
    """Google Gemini REST API gateway (generativelanguage.googleapis.com).

    Uses the v1beta ``generateContent`` endpoint with ``httpx``. Supports:
    - Text generation (gemini-2.5-flash, gemini-2.5-pro, etc.)
    - Tool calling (functionDeclarations → functionCall → functionResponse)
    - System instructions (top-level ``systemInstruction``)
    - Safety settings (configurable thresholds)
    - Thinking mode (gemini-2.5-pro thinking budget)

    No new dependencies — ``httpx`` is already in the SDK.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://generativelanguage.googleapis.com",
        api_version: str = "v1beta",
        client: httpx.AsyncClient | None = None,
        event_emitter: ProviderEventEmitter | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._api_version = api_version
        self._client = client
        self._event_emitter = event_emitter

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        client = self._get_client()
        body = _build_gemini_request_body(request)
        url = self._generate_content_url(request.model_def.model_id)
        await self._emit(
            {
                "type": "provider_call_started",
                "provider": request.model_def.provider,
                "model_id": request.model_def.model_id,
                "display_name": request.model_def.display_name,
            }
        )
        try:
            response = await client.post(url, json=body, headers=self._headers())
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            await self._emit(
                {
                    "type": "provider_call_failed",
                    "provider": request.model_def.provider,
                    "model_id": request.model_def.model_id,
                    "display_name": request.model_def.display_name,
                    "request_id": extract_request_id(exc),
                    "error": str(exc) or type(exc).__name__,
                }
            )
            raise provider_gateway_error_from_exception(request, exc) from exc
        text_parts, tool_calls, reasoning_parts = _extract_gemini_response(data)
        usage = _extract_gemini_usage(data)
        finish_reason, truncated = _extract_gemini_finish(data, has_tool_calls=bool(tool_calls))
        await self._emit(
            {
                "type": "provider_call_completed",
                "provider": request.model_def.provider,
                "model_id": request.model_def.model_id,
                "display_name": request.model_def.display_name,
                "request_id": "",
            }
        )
        return CompletionResult(
            content=normalize_structured_content("\n".join(text_parts), request.response_format),
            provider=request.model_def.provider,
            model_id=request.model_def.model_id,
            display_name=request.model_def.display_name,
            request_id="",
            tool_calls=tool_calls,
            usage=usage,
            reasoning_content="\n".join(reasoning_parts),
            finish_reason=finish_reason,
            truncated=truncated,
        )

    async def complete_stream(self, request: CompletionRequest) -> AsyncIterator[StreamedChunk]:
        """Stream Gemini generate-content deltas from its SSE endpoint."""

        client = self._get_client()
        body = _build_gemini_request_body(request)
        url = self._stream_generate_content_url(request.model_def.model_id)
        await self._emit(
            {
                "type": "provider_call_started",
                "provider": request.model_def.provider,
                "model_id": request.model_def.model_id,
                "display_name": request.model_def.display_name,
            }
        )
        try:
            async with client.stream("POST", url, json=body, headers=self._headers()) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    serialized = line.removeprefix("data:").strip()
                    if not serialized or serialized == "[DONE]":
                        continue
                    data = json.loads(serialized)
                    text_parts, tool_calls, reasoning_parts = _extract_gemini_response(data)
                    finish_reason, _ = _extract_gemini_finish(
                        data, has_tool_calls=bool(tool_calls)
                    )
                    usage = _extract_gemini_usage(data)
                    if text_parts or reasoning_parts or finish_reason or usage:
                        yield StreamedChunk(
                            content="".join(text_parts),
                            reasoning_content="".join(reasoning_parts),
                            finish_reason=finish_reason,
                            usage=usage,
                        )
        except Exception as exc:
            await self._emit(
                {
                    "type": "provider_call_failed",
                    "provider": request.model_def.provider,
                    "model_id": request.model_def.model_id,
                    "display_name": request.model_def.display_name,
                    "request_id": extract_request_id(exc),
                    "error": str(exc) or type(exc).__name__,
                }
            )
            raise provider_gateway_error_from_exception(request, exc) from exc
        await self._emit(
            {
                "type": "provider_call_completed",
                "provider": request.model_def.provider,
                "model_id": request.model_def.model_id,
                "display_name": request.model_def.display_name,
                "request_id": "",
            }
        )

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        client = self._get_client()
        url = self._embed_content_url(request.model_id)
        body: dict[str, Any] = {
            "requests": [
                {"model": f"models/{request.model_id}", "content": {"parts": [{"text": t}]}}
                for t in request.input_texts
            ],
        }
        try:
            response = await client.post(url, json=body, headers=self._headers())
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            raise embedding_gateway_error_from_exception(request, exc) from exc
        vectors = [e.get("values", []) for e in data.get("embeddings", [])]
        return EmbeddingResult(
            embeddings=vectors,
            provider=request.provider,
            model_id=request.model_id,
            request_id="",
            usage=None,
        )

    @staticmethod
    def supports(model_id: str) -> bool:
        return "gemini" in model_id.lower()

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=60.0)
        return self._client

    def _headers(self) -> dict[str, str]:
        """Return authentication headers without leaking the key into URLs."""

        return {"x-goog-api-key": self._api_key}

    def _generate_content_url(self, model_id: str) -> str:
        return f"{self._base_url}/{self._api_version}/models/{model_id}:generateContent"

    def _stream_generate_content_url(self, model_id: str) -> str:
        return (
            f"{self._base_url}/{self._api_version}/models/{model_id}:streamGenerateContent"
            "?alt=sse"
        )

    def _embed_content_url(self, model_id: str) -> str:
        return f"{self._base_url}/{self._api_version}/models/{model_id}:batchEmbedContents"

    async def _emit(self, event: ProviderStreamEvent) -> None:
        if self._event_emitter is None:
            return
        maybe_awaitable = self._event_emitter(event)
        if inspect.isawaitable(maybe_awaitable):
            await maybe_awaitable


def _build_gemini_request_body(request: CompletionRequest) -> dict[str, Any]:
    """Build one native Gemini generate-content request payload."""

    system_text = "\n\n".join(
        str(message.get("content") or "")
        for message in request.messages
        if message.get("role") == "system" and message.get("content")
    ).strip()
    if request.response_format is not None:
        system_text = "\n\n".join(part for part in (system_text, _JSON_ONLY_INSTRUCTION) if part)

    body: dict[str, Any] = {
        "contents": _build_gemini_contents(
            [message for message in request.messages if message.get("role") != "system"]
        ),
        "generationConfig": {"maxOutputTokens": request.max_tokens},
    }
    if request.temperature:
        body["generationConfig"]["temperature"] = request.temperature
    if system_text:
        body["systemInstruction"] = {"parts": [{"text": system_text}]}
    if request.response_format is not None:
        body["generationConfig"]["responseMimeType"] = "application/json"
        response_schema = _gemini_response_schema(request.response_format)
        if response_schema is not None:
            body["generationConfig"]["responseSchema"] = response_schema
    if request.tools:
        body["tools"] = [
            {
                "functionDeclarations": [
                    _openai_tool_to_gemini_function(tool) for tool in request.tools
                ]
            }
        ]
    return body


def _gemini_response_schema(response_format: Any) -> dict[str, Any] | None:
    """Translate a Pydantic schema to Gemini's supported OpenAPI subset."""

    if not is_pydantic_model_class(response_format):
        return None
    raw_schema = response_format.model_json_schema()
    definitions = raw_schema.get("$defs", {})
    translated = _translate_gemini_schema(raw_schema, definitions)
    return translated if isinstance(translated, dict) else None


def _translate_gemini_schema(
    schema: Any,
    definitions: dict[str, Any],
) -> dict[str, Any] | None:
    """Inline Pydantic definitions and remove JSON-Schema-only keywords."""

    if not isinstance(schema, dict):
        return None
    resolved = _resolve_gemini_reference(schema, definitions)
    if resolved is None:
        return None
    has_nullable_variant, nullable_schema = _translate_nullable_gemini_schema(
        resolved, definitions
    )
    if has_nullable_variant:
        return nullable_schema

    type_name = resolved.get("type")
    if not isinstance(type_name, str):
        return None
    translated = _copy_gemini_schema_metadata(resolved, type_name)
    if type_name in {"object", "array"}:
        return _translate_gemini_container_schema(
            type_name, resolved, definitions, translated
        )
    return translated


def _resolve_gemini_reference(
    schema: dict[str, Any], definitions: dict[str, Any]
) -> dict[str, Any] | None:
    """Resolve one local Pydantic ``$defs`` reference, when present."""

    reference = schema.get("$ref")
    if not isinstance(reference, str):
        return schema
    prefix = "#/$defs/"
    if not reference.startswith(prefix):
        return None
    target = definitions.get(reference.removeprefix(prefix))
    return target if isinstance(target, dict) else None


def _translate_nullable_gemini_schema(
    schema: dict[str, Any], definitions: dict[str, Any]
) -> tuple[bool, dict[str, Any] | None]:
    """Translate Pydantic's ``anyOf: [schema, null]`` optional form."""

    variants = schema.get("anyOf")
    if not isinstance(variants, list):
        return False, None
    non_null_variants = [
        variant
        for variant in variants
        if not (isinstance(variant, dict) and variant.get("type") == "null")
    ]
    if len(non_null_variants) != 1 or len(non_null_variants) == len(variants):
        return True, None
    translated = _translate_gemini_schema(non_null_variants[0], definitions)
    if translated is not None:
        translated["nullable"] = True
    return True, translated


def _copy_gemini_schema_metadata(schema: dict[str, Any], type_name: str) -> dict[str, Any]:
    """Copy the metadata Gemini accepts from an OpenAPI schema node."""

    translated: dict[str, Any] = {"type": type_name.upper()}
    for key in ("format", "description", "enum", "maxItems", "minItems", "propertyOrdering"):
        value = schema.get(key)
        if value is not None:
            translated[key] = value
    if schema.get("nullable") is True:
        translated["nullable"] = True
    return translated


def _translate_gemini_container_schema(
    type_name: str,
    schema: dict[str, Any],
    definitions: dict[str, Any],
    translated: dict[str, Any],
) -> dict[str, Any] | None:
    """Translate one object or array schema's recursive content."""

    if type_name == "object":
        return _translate_gemini_object_schema(schema, definitions, translated)
    return _translate_gemini_array_schema(schema, definitions, translated)


def _translate_gemini_object_schema(
    schema: dict[str, Any],
    definitions: dict[str, Any],
    translated: dict[str, Any],
) -> dict[str, Any] | None:
    """Translate object properties and their required names."""

    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        return None
    translated_properties: dict[str, Any] = {}
    for name, property_schema in properties.items():
        property_translation = _translate_gemini_schema(property_schema, definitions)
        if property_translation is None:
            return None
        translated_properties[name] = property_translation
    translated["properties"] = translated_properties
    required = schema.get("required")
    if isinstance(required, list):
        translated["required"] = required
    return translated


def _translate_gemini_array_schema(
    schema: dict[str, Any],
    definitions: dict[str, Any],
    translated: dict[str, Any],
) -> dict[str, Any] | None:
    """Translate an array's item schema."""

    items = _translate_gemini_schema(schema.get("items"), definitions)
    if items is None:
        return None
    translated["items"] = items
    return translated


def _build_gemini_contents(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate OpenAI-style messages to Gemini ``contents[]`` format.

    - ``user`` → ``{"role": "user", "parts": [{"text": ...}]}``
    - ``assistant`` → ``{"role": "model", "parts": [{"text": ...}, ...functionCall...]}``
    - ``tool`` → ``{"role": "user", "parts": [{"functionResponse": {...}}]}``
    """

    contents: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "user")
        if role == "tool":
            tool_name = msg.get("name", "")
            tool_content = str(msg.get("content", ""))
            contents.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": tool_name,
                                "response": {"result": tool_content},
                            }
                        }
                    ],
                }
            )
        elif role == "assistant" and msg.get("tool_calls"):
            parts: list[dict[str, Any]] = []
            text = msg.get("content")
            if text:
                parts.append({"text": str(text)})
            for tc in msg["tool_calls"]:
                function_payload = (
                    tc.get("function") if isinstance(tc.get("function"), dict) else tc
                )
                raw_args = function_payload.get("arguments", "{}")
                if isinstance(raw_args, str):
                    try:
                        parsed_args = json.loads(raw_args)
                    except json.JSONDecodeError:
                        parsed_args = {}
                elif isinstance(raw_args, dict):
                    parsed_args = raw_args
                else:
                    parsed_args = {}
                # Gemini thinking models require thoughtSignature to be echoed
                # back as a sibling of functionCall in the same part.  The
                # signature may be stored at the top level (REST gateway flat
                # shape) or nested under function (OpenAI-compatible shape).
                thought_sig = tc.get("thought_signature") or function_payload.get(
                    "thought_signature"
                )
                fc_part: dict[str, Any] = {
                    "functionCall": {
                        "name": str(function_payload.get("name", "")),
                        "args": parsed_args,
                    }
                }
                if thought_sig:
                    fc_part["thoughtSignature"] = thought_sig
                parts.append(fc_part)
            contents.append({"role": "model", "parts": parts})
        else:
            gemini_role = "model" if role == "assistant" else "user"
            contents.append(
                {
                    "role": gemini_role,
                    "parts": [{"text": str(msg.get("content", ""))}],
                }
            )
    return contents


def _openai_tool_to_gemini_function(tool: dict[str, Any]) -> dict[str, Any]:
    """Convert one OpenAI-format tool to Gemini ``functionDeclarations`` entry."""

    function_payload: dict[str, Any] = (
        tool.get("function") if isinstance(tool.get("function"), dict) else tool
    )
    return {
        "name": str(function_payload.get("name", "")),
        "description": str(function_payload.get("description", "")),
        "parameters": function_payload.get("parameters") or {"type": "object", "properties": {}},
    }


def _extract_gemini_response(
    data: dict[str, Any],
) -> tuple[list[str], list[dict[str, Any]], list[str]]:
    """Extract text and tool calls from a Gemini ``generateContent`` response.

    Returns ``(text_parts, tool_calls, reasoning_parts)`` where ``tool_calls`` is in the flat
    OpenAI shape: ``{"id":..., "type":"function", "name":..., "arguments":"<json>"}``.
    Gemini has no native tool-call IDs, so we synthesize ``gemini_call_N``.
    """

    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    reasoning_parts: list[str] = []
    candidates = data.get("candidates", [])
    call_counter = 0
    for candidate in candidates:
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            if "text" in part:
                target_parts = reasoning_parts if part.get("thought") is True else text_parts
                target_parts.append(part["text"])
            elif "functionCall" in part:
                fc = part["functionCall"]
                call_counter += 1
                tool_call: dict[str, Any] = {
                    "id": f"gemini_call_{call_counter}",
                    "type": "function",
                    "name": str(fc.get("name", "")),
                    "arguments": _normalize_openai_tool_arguments(fc.get("args", {})),
                }
                # Gemini thinking models attach thoughtSignature as a sibling
                # of functionCall in the same part.  It MUST be echoed verbatim
                # in the corresponding functionCall part on the next turn.
                thought_sig = part.get("thoughtSignature") or fc.get("thoughtSignature")
                if thought_sig:
                    tool_call["thought_signature"] = thought_sig
                tool_calls.append(tool_call)
    return text_parts, tool_calls, reasoning_parts


def _extract_gemini_finish(
    data: dict[str, Any],
    *,
    has_tool_calls: bool,
) -> tuple[str | None, bool | None]:
    """Map Gemini's native finish reason to the gateway completion contract."""

    candidates = data.get("candidates")
    candidate = candidates[0] if isinstance(candidates, list) and candidates else {}
    native_reason = candidate.get("finishReason") if isinstance(candidate, dict) else None
    if native_reason == "MAX_TOKENS":
        return "length", True
    if has_tool_calls:
        return "tool_calls", False
    if native_reason == "STOP":
        return "stop", False
    if isinstance(native_reason, str) and native_reason:
        return native_reason.lower(), None
    return None, None


def _extract_gemini_usage(data: dict[str, Any]) -> CompletionUsage | None:
    """Extract normalized usage from a Gemini ``generateContent`` response."""

    usage_meta = data.get("usageMetadata")
    if not usage_meta:
        return None
    prompt_tokens = coerce_non_negative_int(usage_meta.get("promptTokenCount"))
    completion_tokens = coerce_non_negative_int(usage_meta.get("candidatesTokenCount"))
    total_tokens = coerce_non_negative_int(usage_meta.get("totalTokenCount"))
    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens
    if prompt_tokens == 0 and completion_tokens == 0:
        return None
    cached_tokens = coerce_non_negative_int(usage_meta.get("cachedContentTokenCount"))
    reasoning_tokens = coerce_non_negative_int(usage_meta.get("thoughtsTokenCount"))
    return CompletionUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cached_tokens=cached_tokens,
        reasoning_tokens=reasoning_tokens,
    )
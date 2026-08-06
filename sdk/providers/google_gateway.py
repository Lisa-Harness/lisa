"""Google Gemini provider gateway.

Implements the kernel's ``ProviderGateway`` and ``EmbeddingGateway`` protocols
for the Google Gemini REST API (generativelanguage.googleapis.com). Uses
``httpx`` directly; no Google SDK dependency.
"""

from __future__ import annotations

import inspect
import json
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
    embedding_gateway_error_from_exception,
    extract_request_id,
    provider_gateway_error_from_exception,
)
from kernel.token_usage import coerce_non_negative_int

from sdk.providers.openai_gateway import _normalize_openai_tool_arguments


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
        system_text = next(
            (m["content"] for m in request.messages if m.get("role") == "system"),
            "",
        )
        contents = _build_gemini_contents(
            [m for m in request.messages if m.get("role") != "system"]
        )
        body: dict[str, Any] = {"contents": contents}
        if system_text:
            body["systemInstruction"] = {"parts": [{"text": system_text}]}
        if request.tools:
            body["tools"] = [
                {
                    "functionDeclarations": [
                        _openai_tool_to_gemini_function(t) for t in request.tools
                    ]
                }
            ]
        url = f"{self._base_url}/{self._api_version}/models/{request.model_def.model_id}:generateContent?key={self._api_key}"
        await self._emit(
            {
                "type": "provider_call_started",
                "provider": request.model_def.provider,
                "model_id": request.model_def.model_id,
                "display_name": request.model_def.display_name,
            }
        )
        try:
            response = await client.post(url, json=body)
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
        text_parts, tool_calls = _extract_gemini_response(data)
        usage = _extract_gemini_usage(data)
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
            content="\n".join(text_parts),
            provider=request.model_def.provider,
            model_id=request.model_def.model_id,
            display_name=request.model_def.display_name,
            request_id="",
            tool_calls=tool_calls,
            usage=usage,
        )

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        client = self._get_client()
        url = f"{self._base_url}/{self._api_version}/models/{request.model_id}:batchEmbedContents?key={self._api_key}"
        body: dict[str, Any] = {
            "requests": [
                {"model": f"models/{request.model_id}", "content": {"parts": [{"text": t}]}}
                for t in request.input_texts
            ],
        }
        try:
            response = await client.post(url, json=body)
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

    async def _emit(self, event: ProviderStreamEvent) -> None:
        if self._event_emitter is None:
            return
        maybe_awaitable = self._event_emitter(event)
        if inspect.isawaitable(maybe_awaitable):
            await maybe_awaitable


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


def _extract_gemini_response(data: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    """Extract text and tool calls from a Gemini ``generateContent`` response.

    Returns ``(text_parts, tool_calls)`` where ``tool_calls`` is in the flat
    OpenAI shape: ``{"id":..., "type":"function", "name":..., "arguments":"<json>"}``.
    Gemini has no native tool-call IDs, so we synthesize ``gemini_call_N``.
    """

    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    candidates = data.get("candidates", [])
    call_counter = 0
    for candidate in candidates:
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            if "text" in part:
                text_parts.append(part["text"])
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
    return text_parts, tool_calls


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
    return CompletionUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cached_tokens=cached_tokens,
        reasoning_tokens=0,
    )
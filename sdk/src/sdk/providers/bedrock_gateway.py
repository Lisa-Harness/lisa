"""AWS Bedrock provider gateway.

Implements the kernel's ``ProviderGateway`` and ``EmbeddingGateway`` protocols
for AWS Bedrock via boto3 (Converse API). ``boto3`` is an optional dependency;
``__init__`` raises ``ImportError`` with install instructions when it is not
installed.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
from typing import Any

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
from sdk.providers.openai_gateway import (
    _coerce_tool_arguments_object,
    _normalize_openai_tool_arguments,
    normalize_structured_content,
)

_JSON_ONLY_INSTRUCTION = "Return only a valid JSON object with no Markdown wrapper."


class BedrockGateway:
    """AWS Bedrock runtime gateway via boto3.

    Supports Claude (Anthropic), Llama (Meta), Nova (Amazon), and Titan
    (Amazon) models through the Bedrock **Converse API** which provides a
    normalized interface across all Bedrock models.

    ``boto3`` is an optional dependency. If it is not installed,
    ``__init__`` raises ``ImportError`` with install instructions.
    """

    def __init__(
        self,
        *,
        region_name: str = "us-east-1",
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_session_token: str | None = None,
        client: Any | None = None,
        event_emitter: ProviderEventEmitter | None = None,
    ) -> None:
        self._region_name = region_name
        self._aws_access_key_id = aws_access_key_id
        self._aws_secret_access_key = aws_secret_access_key
        self._aws_session_token = aws_session_token
        self._client = client
        self._event_emitter = event_emitter
        if client is None and importlib.util.find_spec("boto3") is None:
            raise ImportError(
                "boto3 is required for BedrockGateway. Install it with: pip install boto3"
            )

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        client = self._get_client()
        system_text = "\n\n".join(
            str(message.get("content") or "")
            for message in request.messages
            if message.get("role") == "system" and message.get("content")
        ).strip()
        if request.response_format is not None:
            system_text = "\n\n".join(
                part for part in (system_text, _JSON_ONLY_INSTRUCTION) if part
            )
        converse_messages = _build_bedrock_messages(
            [m for m in request.messages if m.get("role") != "system"]
        )
        converse_kwargs: dict[str, Any] = {
            "modelId": request.model_def.model_id,
            "messages": converse_messages,
        }
        if system_text:
            converse_kwargs["system"] = [{"text": system_text}]
        if request.tools:
            converse_kwargs["toolConfig"] = {
                "tools": [_openai_tool_to_bedrock_tool(t) for t in request.tools]
            }
        inference_config: dict[str, Any] = {"maxTokens": request.max_tokens}
        if request.temperature:
            inference_config["temperature"] = request.temperature
        converse_kwargs["inferenceConfig"] = inference_config
        await self._emit(
            {
                "type": "provider_call_started",
                "provider": request.model_def.provider,
                "model_id": request.model_def.model_id,
                "display_name": request.model_def.display_name,
            }
        )
        try:
            response = await asyncio.to_thread(client.converse, **converse_kwargs)
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
        text_parts, tool_calls = _extract_bedrock_response(response)
        usage = _extract_bedrock_usage(response)
        stop_reason = _extract_bedrock_stop_reason(response)
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
            finish_reason=stop_reason,
            truncated=_derive_bedrock_truncated(stop_reason),
        )

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        client = self._get_client()
        body = json.dumps({"inputText": request.input_texts[0] if request.input_texts else ""})
        try:
            response = await asyncio.to_thread(
                client.invoke_model,
                modelId=request.model_id,
                body=body,
                accept="application/json",
                contentType="application/json",
            )
        except Exception as exc:
            raise embedding_gateway_error_from_exception(request, exc) from exc
        response_body = json.loads(
            response.get("body", "{}").read()
            if hasattr(response.get("body", b""), "read")
            else response.get("body", b"{}")
        )
        embedding = response_body.get("embedding", [])
        return EmbeddingResult(
            embeddings=[embedding] if embedding else [],
            provider=request.provider,
            model_id=request.model_id,
            request_id="",
            usage=None,
        )

    @staticmethod
    def supports(model_id: str) -> bool:
        return any(prefix in model_id for prefix in ("anthropic.", "meta.", "amazon."))

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        boto3 = importlib.import_module("boto3")

        self._client = boto3.client(
            "bedrock-runtime",
            region_name=self._region_name,
            aws_access_key_id=self._aws_access_key_id,
            aws_secret_access_key=self._aws_secret_access_key,
            aws_session_token=self._aws_session_token,
        )
        return self._client

    async def _emit(self, event: ProviderStreamEvent) -> None:
        if self._event_emitter is None:
            return
        maybe_awaitable = self._event_emitter(event)
        if inspect.isawaitable(maybe_awaitable):
            await maybe_awaitable


def _build_bedrock_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate OpenAI-style messages to Bedrock Converse ``messages[]`` format.

    Converse expects ``[{"role": "user"|"assistant", "content": [{"text": ...}]}]``.
    Tool results are sent as ``user`` messages with ``toolResult`` content blocks.
    Tool calls from the assistant are in ``toolUse`` content blocks.
    """

    converse: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role", "user")
        if role == "tool":
            tool_call_id = str(msg.get("tool_call_id", ""))
            tool_content = str(msg.get("content", ""))
            converse.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "toolResult": {
                                "toolUseId": tool_call_id,
                                "content": [{"text": tool_content}],
                            }
                        }
                    ],
                }
            )
        elif role == "assistant" and msg.get("tool_calls"):
            content_blocks: list[dict[str, Any]] = []
            text = msg.get("content")
            if text:
                content_blocks.append({"text": str(text)})
            for tc in msg["tool_calls"]:
                function_payload = (
                    tc.get("function") if isinstance(tc.get("function"), dict) else tc
                )
                # A refused call (length-cut payload, or parallel calls
                # stitched together) stays in the history and is replayed
                # here; a payload that is not a JSON object would be a
                # malformed tool input, so it is coerced to the empty one.
                parsed_args = _coerce_tool_arguments_object(function_payload.get("arguments", "{}"))
                content_blocks.append(
                    {
                        "toolUse": {
                            "toolUseId": str(tc.get("id", "")),
                            "name": str(function_payload.get("name", "")),
                            "input": parsed_args,
                        }
                    }
                )
            converse.append({"role": "assistant", "content": content_blocks})
        else:
            converse.append(
                {
                    "role": "assistant" if role == "assistant" else "user",
                    "content": [{"text": str(msg.get("content", ""))}],
                }
            )
    return converse


def _derive_bedrock_truncated(stop_reason: str | None) -> bool | None:
    """Derive the normalized truncation flag from Bedrock Converse stopReason."""

    if stop_reason == "max_tokens":
        return True
    if stop_reason in {"end_turn", "stop_sequence", "tool_use"}:
        return False
    return None


def _openai_tool_to_bedrock_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Convert one OpenAI-format tool to Bedrock Converse ``toolConfig`` entry."""

    function_payload: dict[str, Any] = tool
    nested_payload = tool.get("function")
    if isinstance(nested_payload, dict):
        function_payload = nested_payload
    return {
        "toolSpec": {
            "name": str(function_payload.get("name", "")),
            "description": str(function_payload.get("description", "")),
            "inputSchema": {
                "json": function_payload.get("parameters") or {"type": "object", "properties": {}},
            },
        }
    }


def _extract_bedrock_response(response: Any) -> tuple[list[str], list[dict[str, Any]]]:
    """Extract text and tool calls from a Bedrock Converse response.

    Returns ``(text_parts, tool_calls)`` where ``tool_calls`` is in the flat
    OpenAI shape.
    """

    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    output = (
        response.get("output") if isinstance(response, dict) else getattr(response, "output", None)
    )
    if output is None:
        return text_parts, tool_calls
    message = (
        output.get("message") if isinstance(output, dict) else getattr(output, "message", None)
    )
    if message is None:
        return text_parts, tool_calls
    content = (
        message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
    )
    if not content:
        return text_parts, tool_calls
    for block in content:
        if isinstance(block, dict):
            if "text" in block:
                text_parts.append(block["text"])
            elif "toolUse" in block:
                tu = block["toolUse"]
                tool_calls.append(
                    {
                        "id": str(tu.get("toolUseId", "")),
                        "type": "function",
                        "name": str(tu.get("name", "")),
                        "arguments": _normalize_openai_tool_arguments(tu.get("input", {})),
                    }
                )
    return text_parts, tool_calls


def _extract_bedrock_usage(response: Any) -> CompletionUsage | None:
    """Extract normalized usage from a Bedrock Converse response."""

    raw_usage = (
        response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
    )
    if raw_usage is None:
        return None
    prompt_tokens = coerce_non_negative_int(
        raw_usage.get("inputTokens")
        if isinstance(raw_usage, dict)
        else getattr(raw_usage, "inputTokens", None)
    )
    completion_tokens = coerce_non_negative_int(
        raw_usage.get("outputTokens")
        if isinstance(raw_usage, dict)
        else getattr(raw_usage, "outputTokens", None)
    )
    total_tokens = coerce_non_negative_int(
        raw_usage.get("totalTokens")
        if isinstance(raw_usage, dict)
        else getattr(raw_usage, "totalTokens", None)
    )
    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens
    if prompt_tokens == 0 and completion_tokens == 0:
        return None
    return CompletionUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cached_tokens=0,
        reasoning_tokens=0,
    )


def _extract_bedrock_stop_reason(response: Any) -> str | None:
    """Return Bedrock Converse's native stop reason in the shared casing."""

    value = (
        response.get("stopReason")
        if isinstance(response, dict)
        else getattr(response, "stopReason", None)
    )
    return str(value).strip().lower() or None

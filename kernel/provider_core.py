from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

import httpx
from openai import AsyncAzureOpenAI, AsyncOpenAI
from pydantic import BaseModel as PydanticBaseModel

from kernel.token_usage import coerce_non_negative_int

ChatMessages = list[dict[str, Any]]
ProviderStreamEvent = dict[str, Any]
ProviderEventEmitter = Callable[[ProviderStreamEvent], Any]
_TRANSIENT_SERVER_ERROR_MIN_STATUS = 500
_TRANSIENT_STATUS_CODES = frozenset({0, 408, 429})


class ModelDescriptor(Protocol):
    provider: str
    model_id: str
    display_name: str
    reasoning_model: bool
    thinking_model: bool


@dataclass(slots=True)
class CompletionRequest:
    """One normalized completion request sent through a provider gateway."""

    messages: ChatMessages
    model_def: ModelDescriptor
    max_tokens: int
    temperature: float
    tools: list[dict[str, Any]] | None = None
    response_format: Any | None = None


@dataclass(slots=True)
class CompletionUsage:
    """Normalized per-call token usage extracted from one provider response.

    Fields are optional because not every provider surfaces every counter.
    ``prompt_tokens`` and ``completion_tokens`` are the canonical pair used
    for cost math; ``total_tokens`` falls out as the sum when the provider
    does not report it directly. ``cached_tokens`` and ``reasoning_tokens``
    cover the common sub-counters that vendors bill separately.

    ``duration_ms`` captures wall-clock time spent on the provider call.
    Some providers (notably Ollama when billing local GPU time) charge by
    duration rather than token count; hosts that need time-based cost
    math read this field. Defaults to 0 when the gateway did not measure.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    duration_ms: float = 0.0


@dataclass(slots=True)
class CompletionResult:
    """One normalized completion result returned from a provider gateway."""

    content: str
    provider: str
    model_id: str
    display_name: str
    request_id: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: CompletionUsage | None = None


@dataclass(slots=True)
class StreamedChunk:
    """One chunk from a streaming completion.

    v1 streams tokens only — tool-call streaming is deferred to v2.
    """

    content: str = ""
    """Delta text for this chunk."""

    finish_reason: str | None = None
    """``'stop'``, ``'length'``, ``'tool_calls'``, or ``None`` (mid-stream)."""

    usage: CompletionUsage | None = None
    """Final usage — only present on the last chunk for providers that
    include usage in the stream."""


@dataclass(slots=True)
class StreamedCompletion:
    """Aggregated result of a streaming completion."""

    content: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    usage: CompletionUsage | None = None
    finish_reason: str = "stop"
    chunks: list[StreamedChunk] = field(default_factory=list)


@dataclass(slots=True)
class EmbeddingRequest:
    """One normalized embeddings request sent through a provider gateway."""

    input_texts: list[str]
    provider: str
    model_id: str
    dimensions: int | None = None


@dataclass(slots=True)
class EmbeddingResult:
    """One normalized embeddings result returned from a provider gateway."""

    embeddings: list[list[float]]
    provider: str
    model_id: str
    request_id: str = ""
    usage: CompletionUsage | None = None


class ProviderGatewayError(RuntimeError):
    """Normalized model-provider failure surfaced to callers."""

    def __init__(
        self,
        provider: str,
        model_id: str,
        display_name: str,
        message: str,
        request_id: str = "",
        status_code: int = 0,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.model_id = model_id
        self.display_name = display_name
        self.request_id = request_id
        self.status_code = status_code

    @property
    def is_transient(self) -> bool:
        """True when the upstream provider returned a retryable status or timed out."""

        if (
            self.status_code >= _TRANSIENT_SERVER_ERROR_MIN_STATUS
            or self.status_code in _TRANSIENT_STATUS_CODES
        ):
            return True
        return "timeout" in self.message.lower()


class ProviderGateway(Protocol):
    async def complete(self, request: CompletionRequest) -> CompletionResult:
        """Execute a completion for the given provider request."""


class EmbeddingGateway(Protocol):
    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        """Execute one embeddings request for the given provider request."""


def normalize_openai_messages(messages: ChatMessages) -> ChatMessages:
    """Clone one chat history with OpenAI-compatible tool-call payloads."""

    normalized_messages: ChatMessages = []
    for message in messages:
        normalized_message = dict(message)
        raw_tool_calls = normalized_message.get("tool_calls")
        if isinstance(raw_tool_calls, list):
            normalized_message["tool_calls"] = [
                _normalize_openai_tool_call(tool_call) if isinstance(tool_call, dict) else tool_call
                for tool_call in raw_tool_calls
            ]
        normalized_messages.append(normalized_message)
    return normalized_messages


def build_openai_completion_kwargs(request: CompletionRequest) -> dict[str, Any]:
    """Build provider kwargs for one OpenAI-compatible completion request."""

    kwargs: dict[str, Any] = {
        "model": request.model_def.model_id,
        "messages": cast(Any, normalize_openai_messages(request.messages)),
    }
    if request.model_def.reasoning_model:
        kwargs["max_completion_tokens"] = request.max_tokens
    elif request.model_def.thinking_model:
        kwargs["max_tokens"] = request.max_tokens
    else:
        kwargs["max_tokens"] = request.max_tokens
        kwargs["temperature"] = request.temperature
    if request.tools:
        kwargs["tools"] = request.tools
    if request.response_format:
        kwargs["response_format"] = request.response_format
    return kwargs


def build_openai_responses_kwargs(request: CompletionRequest) -> dict[str, Any]:
    """Build provider kwargs for one OpenAI Responses API request."""

    kwargs: dict[str, Any] = {
        "model": request.model_def.model_id,
        "input": build_openai_responses_input(request.messages),
        "max_output_tokens": request.max_tokens,
    }
    if not request.model_def.reasoning_model and not request.model_def.thinking_model:
        kwargs["temperature"] = request.temperature
    if request.tools:
        kwargs["tools"] = [_normalize_openai_response_tool(tool) for tool in request.tools]
    if request.response_format:
        kwargs["text"] = {"format": _normalize_openai_response_format(request.response_format)}
    return kwargs


def build_openai_responses_input(messages: ChatMessages) -> list[dict[str, Any]]:
    """Translate chat-completions history into Responses API input items."""

    normalized_messages = normalize_openai_messages(messages)
    input_items: list[dict[str, Any]] = []
    for message in normalized_messages:
        role = str(message.get("role", "") or "")
        content = message.get("content", "")
        if role in {"system", "developer", "user"}:
            input_items.append(
                {
                    "type": "message",
                    "role": role,
                    "content": "" if content is None else content,
                }
            )
            continue
        if role == "assistant":
            if content not in (None, "", []):
                input_items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": content,
                    }
                )
            raw_tool_calls = message.get("tool_calls")
            if isinstance(raw_tool_calls, list):
                for raw_tool_call in raw_tool_calls:
                    if not isinstance(raw_tool_call, dict):
                        continue
                    function_payload = raw_tool_call.get("function")
                    if not isinstance(function_payload, dict):
                        continue
                    input_items.append(
                        {
                            "type": "function_call",
                            "call_id": str(raw_tool_call.get("id", "") or ""),
                            "name": str(function_payload.get("name", "") or ""),
                            "arguments": _normalize_openai_tool_arguments(
                                function_payload.get("arguments", "")
                            ),
                            "status": "completed",
                        }
                    )
            continue
        if role == "tool":
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": str(message.get("tool_call_id", "") or ""),
                    "output": "" if content is None else content,
                    "status": "completed",
                }
            )
    return input_items


def build_openai_embedding_kwargs(request: EmbeddingRequest) -> dict[str, Any]:
    """Build provider kwargs for one OpenAI-style embeddings request."""

    kwargs: dict[str, Any] = {
        "model": request.model_id,
        "input": request.input_texts,
    }
    if request.dimensions is not None:
        kwargs["dimensions"] = request.dimensions
    return kwargs


def extract_request_id(value: Any) -> str:
    """Return the best-effort provider request id from one response or exception."""

    for attribute in ("_request_id", "request_id", "id"):
        candidate = getattr(value, attribute, "")
        if isinstance(candidate, str) and candidate:
            return candidate
    return ""


def extract_openai_content(response: Any) -> str:
    """Extract text or parsed JSON content from one OpenAI-compatible response."""

    choices = getattr(response, "choices", [])
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", "")
    if content:
        return content
    parsed = getattr(message, "parsed", None)
    if parsed is not None:
        if isinstance(parsed, PydanticBaseModel):
            return parsed.model_dump_json()
        try:
            return json.dumps(parsed)
        except Exception:  # pylint: disable=broad-except
            return str(parsed)
    return ""


def extract_responses_content(response: Any) -> str:
    """Extract text content from one OpenAI Responses API response."""

    output_text = getattr(response, "output_text", "")
    if isinstance(output_text, str) and output_text:
        return output_text

    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", "") != "message":
            continue
        for part in getattr(item, "content", []) or []:
            text = getattr(part, "text", "")
            if isinstance(text, str) and text:
                chunks.append(text)
    return "\n".join(chunks)


def extract_openai_tool_calls(response: Any) -> list[dict[str, Any]]:
    """Extract OpenAI-style function tool calls from the first response choice."""

    choices = getattr(response, "choices", [])
    if not choices:
        return []
    message = getattr(choices[0], "message", None)
    raw_calls = getattr(message, "tool_calls", None) or []
    extracted: list[dict[str, Any]] = []
    for call in raw_calls:
        function_obj = getattr(call, "function", None)
        entry: dict[str, Any] = {
            "id": getattr(call, "id", ""),
            "type": getattr(call, "type", ""),
            "name": getattr(function_obj, "name", "") if function_obj else "",
            "arguments": getattr(function_obj, "arguments", "") if function_obj else "",
        }
        thought_sig = getattr(call, "thought_signature", None) or getattr(
            function_obj, "thought_signature", None
        )
        if thought_sig:
            entry["thought_signature"] = thought_sig
        extracted.append(entry)
    return extracted


def extract_responses_tool_calls(response: Any) -> list[dict[str, Any]]:
    """Extract Responses API function tool calls into Lisa's normalized shape."""

    extracted: list[dict[str, Any]] = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", "") != "function_call":
            continue
        extracted.append(
            {
                "id": str(getattr(item, "id", "") or getattr(item, "call_id", "") or ""),
                "type": "function",
                "name": str(getattr(item, "name", "") or ""),
                "arguments": _normalize_openai_tool_arguments(getattr(item, "arguments", "")),
            }
        )
    return extracted


def extract_openai_embeddings(response: Any) -> list[list[float]]:
    """Extract embedding vectors from one OpenAI-style embeddings response."""

    vectors: list[list[float]] = []
    for item in getattr(response, "data", []) or []:
        raw_embedding = getattr(item, "embedding", None)
        if raw_embedding is None:
            continue
        vectors.append([float(value) for value in raw_embedding])
    return vectors





def extract_openai_completion_usage(response: Any) -> CompletionUsage | None:
    """Extract normalized per-call usage from one OpenAI-style chat response.

    Returns None when the response has no usage object (e.g. some preview
    endpoints or explicitly disabled usage). All four known sub-counters
    are read defensively so a missing field never raises.
    """

    raw_usage = getattr(response, "usage", None)
    if raw_usage is None and isinstance(response, dict):
        raw_usage = response.get("usage")
    if raw_usage is None:
        return None
    prompt_tokens = coerce_non_negative_int(
        getattr(raw_usage, "prompt_tokens", None)
        if not isinstance(raw_usage, dict)
        else raw_usage.get("prompt_tokens")
    )
    completion_tokens = coerce_non_negative_int(
        getattr(raw_usage, "completion_tokens", None)
        if not isinstance(raw_usage, dict)
        else raw_usage.get("completion_tokens")
    )
    total_tokens = coerce_non_negative_int(
        getattr(raw_usage, "total_tokens", None)
        if not isinstance(raw_usage, dict)
        else raw_usage.get("total_tokens")
    )
    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens

    prompt_details = (
        getattr(raw_usage, "prompt_tokens_details", None)
        if not isinstance(raw_usage, dict)
        else raw_usage.get("prompt_tokens_details")
    )
    cached_tokens = 0
    if prompt_details is not None:
        cached_tokens = coerce_non_negative_int(
            getattr(prompt_details, "cached_tokens", None)
            if not isinstance(prompt_details, dict)
            else prompt_details.get("cached_tokens")
        )

    completion_details = (
        getattr(raw_usage, "completion_tokens_details", None)
        if not isinstance(raw_usage, dict)
        else raw_usage.get("completion_tokens_details")
    )
    reasoning_tokens = 0
    if completion_details is not None:
        reasoning_tokens = coerce_non_negative_int(
            getattr(completion_details, "reasoning_tokens", None)
            if not isinstance(completion_details, dict)
            else completion_details.get("reasoning_tokens")
        )

    if prompt_tokens == 0 and completion_tokens == 0 and total_tokens == 0:
        return None
    return CompletionUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cached_tokens=cached_tokens,
        reasoning_tokens=reasoning_tokens,
    )


def extract_responses_usage(response: Any) -> CompletionUsage | None:
    """Extract normalized usage from one OpenAI Responses API response."""

    raw_usage = getattr(response, "usage", None)
    if raw_usage is None and isinstance(response, dict):
        raw_usage = response.get("usage")
    if raw_usage is None:
        return None

    prompt_tokens = coerce_non_negative_int(
        getattr(raw_usage, "input_tokens", None)
        if not isinstance(raw_usage, dict)
        else raw_usage.get("input_tokens")
    )
    completion_tokens = coerce_non_negative_int(
        getattr(raw_usage, "output_tokens", None)
        if not isinstance(raw_usage, dict)
        else raw_usage.get("output_tokens")
    )
    total_tokens = coerce_non_negative_int(
        getattr(raw_usage, "total_tokens", None)
        if not isinstance(raw_usage, dict)
        else raw_usage.get("total_tokens")
    )
    if total_tokens <= 0:
        total_tokens = prompt_tokens + completion_tokens

    input_details = (
        getattr(raw_usage, "input_tokens_details", None)
        if not isinstance(raw_usage, dict)
        else raw_usage.get("input_tokens_details")
    )
    cached_tokens = 0
    if input_details is not None:
        cached_tokens = coerce_non_negative_int(
            getattr(input_details, "cached_tokens", None)
            if not isinstance(input_details, dict)
            else input_details.get("cached_tokens")
        )

    output_details = (
        getattr(raw_usage, "output_tokens_details", None)
        if not isinstance(raw_usage, dict)
        else raw_usage.get("output_tokens_details")
    )
    reasoning_tokens = 0
    if output_details is not None:
        reasoning_tokens = coerce_non_negative_int(
            getattr(output_details, "reasoning_tokens", None)
            if not isinstance(output_details, dict)
            else output_details.get("reasoning_tokens")
        )

    if prompt_tokens == 0 and completion_tokens == 0 and total_tokens == 0:
        return None
    return CompletionUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cached_tokens=cached_tokens,
        reasoning_tokens=reasoning_tokens,
    )


def extract_openai_embedding_usage(response: Any) -> CompletionUsage | None:
    """Extract normalized usage from one OpenAI-style embeddings response."""

    raw_usage = getattr(response, "usage", None)
    if raw_usage is None and isinstance(response, dict):
        raw_usage = response.get("usage")
    if raw_usage is None:
        return None
    prompt_tokens = coerce_non_negative_int(
        getattr(raw_usage, "prompt_tokens", None)
        if not isinstance(raw_usage, dict)
        else raw_usage.get("prompt_tokens")
    )
    total_tokens = coerce_non_negative_int(
        getattr(raw_usage, "total_tokens", None)
        if not isinstance(raw_usage, dict)
        else raw_usage.get("total_tokens")
    )
    if prompt_tokens == 0 and total_tokens == 0:
        return None
    return CompletionUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=0,
        total_tokens=total_tokens or prompt_tokens,
        cached_tokens=0,
        reasoning_tokens=0,
    )


def _aggregate_streamed_chunks(chunks: list[StreamedChunk]) -> StreamedCompletion:
    """Aggregate a list of ``StreamedChunk`` into one ``StreamedCompletion``.

    v1 streams tokens only — tool-call streaming is deferred to v2, so this
    helper does not attempt to reconstruct tool calls from deltas. The last
    non-``None`` ``finish_reason`` wins. The last non-``None`` ``usage`` wins.
    """

    parts: list[str] = []
    finish_reason = "stop"
    usage: CompletionUsage | None = None
    for chunk in chunks:
        if chunk.content:
            parts.append(chunk.content)
        if chunk.finish_reason is not None:
            finish_reason = chunk.finish_reason
        if chunk.usage is not None:
            usage = chunk.usage
    return StreamedCompletion(
        content="".join(parts),
        tool_calls=[],
        usage=usage,
        finish_reason=finish_reason,
        chunks=chunks,
    )


def extract_anthropic_usage(response: Any) -> CompletionUsage | None:
    """Extract normalized usage from one Anthropic Messages response.

    Anthropic uses ``input_tokens`` / ``output_tokens`` instead of
    ``prompt_tokens`` / ``completion_tokens``. Cache hits surface under
    ``cache_creation_input_tokens`` and ``cache_read_input_tokens``.
    """

    raw_usage = getattr(response, "usage", None)
    if raw_usage is None and isinstance(response, dict):
        raw_usage = response.get("usage")
    if raw_usage is None:
        return None
    prompt_tokens = coerce_non_negative_int(
        getattr(raw_usage, "input_tokens", None)
        if not isinstance(raw_usage, dict)
        else raw_usage.get("input_tokens")
    )
    completion_tokens = coerce_non_negative_int(
        getattr(raw_usage, "output_tokens", None)
        if not isinstance(raw_usage, dict)
        else raw_usage.get("output_tokens")
    )
    cache_creation = coerce_non_negative_int(
        getattr(raw_usage, "cache_creation_input_tokens", None)
        if not isinstance(raw_usage, dict)
        else raw_usage.get("cache_creation_input_tokens")
    )
    cache_read = coerce_non_negative_int(
        getattr(raw_usage, "cache_read_input_tokens", None)
        if not isinstance(raw_usage, dict)
        else raw_usage.get("cache_read_input_tokens")
    )
    if prompt_tokens == 0 and completion_tokens == 0:
        return None
    return CompletionUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        cached_tokens=cache_read,
        reasoning_tokens=cache_creation,
    )


def provider_gateway_error_from_exception(
    request: CompletionRequest,
    exc: Exception,
    request_id: str = "",
) -> ProviderGatewayError:
    """Normalize one completion exception into the shared provider error shape."""

    if isinstance(exc, ProviderGatewayError):
        return exc
    status_code = int(getattr(exc, "status_code", 0) or 0)
    return ProviderGatewayError(
        provider=request.model_def.provider,
        model_id=request.model_def.model_id,
        display_name=request.model_def.display_name,
        message=str(exc) or type(exc).__name__,
        request_id=request_id or extract_request_id(exc),
        status_code=status_code,
    )


def embedding_gateway_error_from_exception(
    request: EmbeddingRequest,
    exc: Exception,
    request_id: str = "",
) -> ProviderGatewayError:
    """Normalize one embeddings exception into the shared provider error shape."""

    if isinstance(exc, ProviderGatewayError):
        return exc
    status_code = int(getattr(exc, "status_code", 0) or 0)
    return ProviderGatewayError(
        provider=request.provider,
        model_id=request.model_id,
        display_name=f"{request.provider}:{request.model_id}",
        message=str(exc) or type(exc).__name__,
        request_id=request_id or extract_request_id(exc),
        status_code=status_code,
    )


def is_pydantic_model_class(obj: Any) -> bool:
    """Return True when obj is a Pydantic BaseModel class."""

    return isinstance(obj, type) and issubclass(obj, PydanticBaseModel)


class AzureOpenAIGateway:
    """Gateway for Azure OpenAI chat completions, responses, and embeddings."""

    def __init__(
        self,
        *,
        complete_client_factory: Callable[[], AsyncAzureOpenAI],
        event_emitter: ProviderEventEmitter | None = None,
    ) -> None:
        self._complete_client_factory = complete_client_factory
        self._event_emitter = event_emitter

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        client = self._complete_client_factory()
        await self._emit(
            {
                "type": "provider_call_started",
                "provider": request.model_def.provider,
                "model_id": request.model_def.model_id,
                "display_name": request.model_def.display_name,
            }
        )
        try:
            if _uses_openai_responses_api(request.model_def):
                response = await client.responses.create(**build_openai_responses_kwargs(request))
                content = extract_responses_content(response)
                tool_calls = extract_responses_tool_calls(response)
                usage = extract_responses_usage(response)
            else:
                kwargs = build_openai_completion_kwargs(request)
                if is_pydantic_model_class(request.response_format):
                    response = await client.beta.chat.completions.parse(**kwargs)
                else:
                    response = await client.chat.completions.create(**kwargs)
                content = extract_openai_content(response)
                tool_calls = extract_openai_tool_calls(response)
                usage = extract_openai_completion_usage(response)
        except Exception as exc:  # pylint: disable=broad-except
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

        request_id = extract_request_id(response)
        await self._emit(
            {
                "type": "provider_call_completed",
                "provider": request.model_def.provider,
                "model_id": request.model_def.model_id,
                "display_name": request.model_def.display_name,
                "request_id": request_id,
            }
        )
        return CompletionResult(
            content=content,
            provider=request.model_def.provider,
            model_id=request.model_def.model_id,
            display_name=request.model_def.display_name,
            request_id=request_id,
            tool_calls=tool_calls,
            usage=usage,
        )

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        """Execute one Azure OpenAI embeddings request."""

        client = self._complete_client_factory()
        kwargs = build_openai_embedding_kwargs(request)
        try:
            response = await client.embeddings.create(**kwargs)
        except Exception as exc:  # pylint: disable=broad-except
            raise embedding_gateway_error_from_exception(request, exc) from exc
        return EmbeddingResult(
            embeddings=extract_openai_embeddings(response),
            provider=request.provider,
            model_id=request.model_id,
            request_id=extract_request_id(response),
            usage=extract_openai_embedding_usage(response),
        )

    async def _emit(self, event: ProviderStreamEvent) -> None:
        if self._event_emitter is None:
            return
        maybe_awaitable = self._event_emitter(event)
        if inspect.isawaitable(maybe_awaitable):
            await maybe_awaitable


class OpenAICompatibleGateway:
    """Gateway for OpenAI-compatible chat completions and embeddings."""

    def __init__(
        self,
        client_factory: Callable[[], AsyncOpenAI],
        *,
        event_emitter: ProviderEventEmitter | None = None,
    ) -> None:
        self._client_factory = client_factory
        self._event_emitter = event_emitter

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        client = self._client_factory()
        kwargs = build_openai_completion_kwargs(request)
        await self._emit(
            {
                "type": "provider_call_started",
                "provider": request.model_def.provider,
                "model_id": request.model_def.model_id,
                "display_name": request.model_def.display_name,
            }
        )
        try:
            if is_pydantic_model_class(request.response_format):
                response = await client.beta.chat.completions.parse(**kwargs)
            else:
                response = await client.chat.completions.create(**kwargs)
        except Exception as exc:  # pylint: disable=broad-except
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

        request_id = extract_request_id(response)
        await self._emit(
            {
                "type": "provider_call_completed",
                "provider": request.model_def.provider,
                "model_id": request.model_def.model_id,
                "display_name": request.model_def.display_name,
                "request_id": request_id,
            }
        )
        return CompletionResult(
            content=extract_openai_content(response),
            provider=request.model_def.provider,
            model_id=request.model_def.model_id,
            display_name=request.model_def.display_name,
            request_id=request_id,
            tool_calls=extract_openai_tool_calls(response),
            usage=extract_openai_completion_usage(response),
        )

    async def embed(self, request: EmbeddingRequest) -> EmbeddingResult:
        """Execute one OpenAI-compatible embeddings request."""

        client = self._client_factory()
        kwargs = build_openai_embedding_kwargs(request)
        try:
            response = await client.embeddings.create(**kwargs)
        except Exception as exc:  # pylint: disable=broad-except
            raise embedding_gateway_error_from_exception(request, exc) from exc
        return EmbeddingResult(
            embeddings=extract_openai_embeddings(response),
            provider=request.provider,
            model_id=request.model_id,
            request_id=extract_request_id(response),
            usage=extract_openai_embedding_usage(response),
        )

    async def complete_stream(self, request: CompletionRequest) -> AsyncIterator[StreamedChunk]:
        """Stream completion chunks from an OpenAI-compatible endpoint.

        v1 streams tokens only. Tool-call deltas are not parsed; the host
        gets text deltas and a final ``finish_reason``. When the model wants
        to call tools, the stream's ``finish_reason`` will be ``'tool_calls'``
        and the host should fall back to ``complete()`` for that iteration.
        """

        client = self._client_factory()
        kwargs = build_openai_completion_kwargs(request)
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}
        await self._emit(
            {
                "type": "provider_call_started",
                "provider": request.model_def.provider,
                "model_id": request.model_def.model_id,
                "display_name": request.model_def.display_name,
            }
        )
        try:
            stream = await client.chat.completions.create(**kwargs)
        except Exception as exc:  # pylint: disable=broad-except
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
        async for event in stream:
            delta_content = ""
            finish_reason = None
            choices = getattr(event, "choices", None) or []
            if choices:
                choice = choices[0]
                delta = getattr(choice, "delta", None)
                if delta is not None:
                    delta_content = getattr(delta, "content", None) or ""
                finish_reason = getattr(choice, "finish_reason", None)
            usage = extract_openai_completion_usage(event)
            yield StreamedChunk(
                content=delta_content,
                finish_reason=finish_reason,
                usage=usage,
            )
        await self._emit(
            {
                "type": "provider_call_completed",
                "provider": request.model_def.provider,
                "model_id": request.model_def.model_id,
                "display_name": request.model_def.display_name,
                "request_id": "",
            }
        )

    async def _emit(self, event: ProviderStreamEvent) -> None:
        if self._event_emitter is None:
            return
        maybe_awaitable = self._event_emitter(event)
        if inspect.isawaitable(maybe_awaitable):
            await maybe_awaitable


class AnthropicGateway:
    """Gateway for Anthropic message completions."""

    def __init__(
        self,
        client_factory: Callable[[], Any],
        *,
        event_emitter: ProviderEventEmitter | None = None,
    ) -> None:
        self._client_factory = client_factory
        self._event_emitter = event_emitter

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        client = self._client_factory()
        system_text = next(
            (message["content"] for message in request.messages if message["role"] == "system"),
            "",
        )
        anthropic_messages = _build_anthropic_messages(
            [message for message in request.messages if message["role"] != "system"]
        )
        create_kwargs: dict[str, Any] = {
            "model": request.model_def.model_id,
            "max_tokens": request.max_tokens,
            "messages": anthropic_messages,
        }
        if system_text:
            create_kwargs["system"] = system_text
        if request.temperature:
            create_kwargs["temperature"] = request.temperature
        if request.tools:
            create_kwargs["tools"] = [
                _openai_tool_to_anthropic_tool(tool) for tool in request.tools
            ]
        if getattr(request.model_def, "thinking_model", False):
            create_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": request.max_tokens,
            }
        await self._emit(
            {
                "type": "provider_call_started",
                "provider": request.model_def.provider,
                "model_id": request.model_def.model_id,
                "display_name": request.model_def.display_name,
            }
        )
        try:
            response = await client.messages.create(**create_kwargs)
        except Exception as exc:  # pylint: disable=broad-except
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

        content_blocks = getattr(response, "content", [])
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for block in content_blocks:
            block_type = getattr(block, "type", "")
            if block_type == "text":
                text_parts.append(getattr(block, "text", "") or "")
            elif block_type == "tool_use":
                tool_calls.append(
                    {
                        "id": getattr(block, "id", ""),
                        "type": "function",
                        "name": getattr(block, "name", ""),
                        "arguments": _normalize_openai_tool_arguments(getattr(block, "input", {})),
                    }
                )
        request_id = extract_request_id(response)
        await self._emit(
            {
                "type": "provider_call_completed",
                "provider": request.model_def.provider,
                "model_id": request.model_def.model_id,
                "display_name": request.model_def.display_name,
                "request_id": request_id,
            }
        )
        return CompletionResult(
            content="\n".join(text_parts),
            provider=request.model_def.provider,
            model_id=request.model_def.model_id,
            display_name=request.model_def.display_name,
            request_id=request_id,
            tool_calls=tool_calls,
            usage=extract_anthropic_usage(response),
        )

    async def complete_stream(self, request: CompletionRequest) -> AsyncIterator[StreamedChunk]:
        """Stream completion chunks from the Anthropic Messages API.

        v1 streams tokens only. Uses ``client.messages.stream()`` which yields
        events; we extract ``text_delta`` events for content deltas and the
        ``message_stop`` event for ``finish_reason``. Usage is read from the
        final message object via ``extract_anthropic_usage``.
        """

        client = self._client_factory()
        system_text = next(
            (message["content"] for message in request.messages if message["role"] == "system"),
            "",
        )
        anthropic_messages = _build_anthropic_messages(
            [message for message in request.messages if message["role"] != "system"]
        )
        create_kwargs: dict[str, Any] = {
            "model": request.model_def.model_id,
            "max_tokens": request.max_tokens,
            "messages": anthropic_messages,
        }
        if system_text:
            create_kwargs["system"] = system_text
        if request.temperature:
            create_kwargs["temperature"] = request.temperature
        if request.tools:
            create_kwargs["tools"] = [
                _openai_tool_to_anthropic_tool(tool) for tool in request.tools
            ]
        if getattr(request.model_def, "thinking_model", False):
            create_kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": request.max_tokens,
            }
        await self._emit(
            {
                "type": "provider_call_started",
                "provider": request.model_def.provider,
                "model_id": request.model_def.model_id,
                "display_name": request.model_def.display_name,
            }
        )
        try:
            async with client.messages.stream(**create_kwargs) as stream:
                async for event in stream:
                    event_type = getattr(event, "type", "")
                    if event_type == "content_block_delta":
                        delta = getattr(event, "delta", None)
                        delta_text = getattr(delta, "text", None) if delta is not None else None
                        if delta_text:
                            yield StreamedChunk(content=delta_text)
                    elif event_type == "message_stop":
                        yield StreamedChunk(finish_reason="stop")
                final_message = await stream.get_final_message()
                usage = extract_anthropic_usage(final_message)
                if usage is not None:
                    yield StreamedChunk(usage=usage)
        except Exception as exc:  # pylint: disable=broad-except
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

    async def _emit(self, event: ProviderStreamEvent) -> None:
        if self._event_emitter is None:
            return
        maybe_awaitable = self._event_emitter(event)
        if inspect.isawaitable(maybe_awaitable):
            await maybe_awaitable


class GoogleGeminiGateway:
    """Google Gemini REST API gateway (generativelanguage.googleapis.com).

    Uses the v1beta ``generateContent`` endpoint with ``httpx``. Supports:
    - Text generation (gemini-2.5-flash, gemini-2.5-pro, etc.)
    - Tool calling (functionDeclarations → functionCall → functionResponse)
    - System instructions (top-level ``systemInstruction``)
    - Safety settings (configurable thresholds)
    - Thinking mode (gemini-2.5-pro thinking budget)

    No new dependencies — ``httpx`` is already in the kernel.
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
                thought_sig = (
                    tc.get("thought_signature")
                    or function_payload.get("thought_signature")
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
        if client is None:
            try:
                import boto3
            except ImportError as exc:
                raise ImportError(
                    "boto3 is required for BedrockGateway. Install it with: pip install boto3"
                ) from exc

    async def complete(self, request: CompletionRequest) -> CompletionResult:
        client = self._get_client()
        system_text = next(
            (m["content"] for m in request.messages if m.get("role") == "system"),
            "",
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
        if request.temperature:
            converse_kwargs["inferenceConfig"] = {"temperature": request.temperature}
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
        import boto3

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


def _openai_tool_to_bedrock_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Convert one OpenAI-format tool to Bedrock Converse ``toolConfig`` entry."""

    function_payload: dict[str, Any] = (
        tool.get("function") if isinstance(tool.get("function"), dict) else tool
    )
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


def _normalize_openai_tool_arguments(raw_arguments: Any) -> str:
    if isinstance(raw_arguments, str):
        return raw_arguments
    if raw_arguments is None:
        return ""
    try:
        return json.dumps(raw_arguments)
    except TypeError:
        return str(raw_arguments)


def _normalize_openai_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
    function_payload = tool_call.get("function")
    if isinstance(function_payload, dict):
        raw_name = function_payload.get("name", "")
        raw_arguments = function_payload.get("arguments", "")
    else:
        raw_name = tool_call.get("name", "")
        raw_arguments = tool_call.get("arguments", "")
    normalized: dict[str, Any] = {
        "id": tool_call.get("id", ""),
        "type": tool_call.get("type", "function") or "function",
        "function": {
            "name": str(raw_name or ""),
            "arguments": _normalize_openai_tool_arguments(raw_arguments),
        },
    }
    thought_sig = tool_call.get("thought_signature")
    if not thought_sig:
        fn = tool_call.get("function")
        if isinstance(fn, dict):
            thought_sig = fn.get("thought_signature")
    if thought_sig:
        normalized["thought_signature"] = thought_sig
    return normalized


def _openai_tool_to_anthropic_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Translate one OpenAI-format tool definition to Anthropic tool format.

    The turn runner produces tools via ``mcp_tool_to_openai_function``::

        {"type": "function", "function": {"name": ..., "description": ...,
        "parameters": <JSON-Schema>}}

    Anthropic expects::

        {"name": ..., "description": ..., "input_schema": <JSON-Schema>}
    """

    function_payload: dict[str, Any] = (
        tool.get("function") if isinstance(tool.get("function"), dict) else tool
    )
    return {
        "name": str(function_payload.get("name", "")),
        "description": str(function_payload.get("description", "")),
        "input_schema": function_payload.get("parameters") or {"type": "object", "properties": {}},
    }


def _build_anthropic_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate OpenAI-style chat messages to Anthropic messages format.

    Handles three conversions:
    - ``tool``-role messages → Anthropic ``user`` messages with ``tool_result``
      content blocks.
    - ``assistant`` messages with ``tool_calls`` → Anthropic ``assistant``
      messages with ``text`` + ``tool_use`` content blocks.
    - All other messages pass through with ``role`` and ``content`` preserved.
    """

    anthropic_messages: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role", "user")
        if role == "tool":
            tool_result_content: list[dict[str, Any]] = [
                {
                    "type": "tool_result",
                    "tool_use_id": str(message.get("tool_call_id", "")),
                    "content": str(message.get("content", "")),
                }
            ]
            anthropic_messages.append({"role": "user", "content": tool_result_content})
            continue
        if role == "assistant" and message.get("tool_calls"):
            content_blocks: list[dict[str, Any]] = []
            text = message.get("content")
            if text:
                content_blocks.append({"type": "text", "text": str(text)})
            for tool_call in message["tool_calls"]:
                function_payload = (
                    tool_call.get("function")
                    if isinstance(tool_call.get("function"), dict)
                    else tool_call
                )
                raw_input = function_payload.get("arguments", "{}")
                if isinstance(raw_input, str):
                    try:
                        parsed_input = json.loads(raw_input)
                    except json.JSONDecodeError:
                        parsed_input = {}
                elif isinstance(raw_input, dict):
                    parsed_input = raw_input
                else:
                    parsed_input = {}
                content_blocks.append(
                    {
                        "type": "tool_use",
                        "id": str(tool_call.get("id", "")),
                        "name": str(function_payload.get("name", "")),
                        "input": parsed_input,
                    }
                )
            anthropic_messages.append({"role": "assistant", "content": content_blocks})
            continue
        anthropic_messages.append({"role": role, "content": message.get("content", "")})
    return anthropic_messages


def _normalize_openai_response_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Translate one chat-completions tool definition into Responses API shape."""

    if tool.get("type") != "function":
        return dict(tool)

    function_payload = tool.get("function")
    if not isinstance(function_payload, dict):
        return dict(tool)

    normalized: dict[str, Any] = {
        "type": "function",
        "name": str(function_payload.get("name", "") or ""),
        "parameters": function_payload.get("parameters", {}),
    }
    description = function_payload.get("description")
    if description:
        normalized["description"] = description
    strict = function_payload.get("strict")
    if strict is not None:
        normalized["strict"] = strict
    defer_loading = function_payload.get("defer_loading")
    if defer_loading is not None:
        normalized["defer_loading"] = defer_loading
    return normalized


def _normalize_openai_response_format(response_format: Any) -> Any:
    """Translate one structured-output hint into Responses API text.format."""

    if is_pydantic_model_class(response_format):
        return {
            "type": "json_schema",
            "name": response_format.__name__,
            "schema": response_format.model_json_schema(),
            "strict": True,
        }
    if isinstance(response_format, dict):
        return dict(response_format)
    return response_format


def _uses_openai_responses_api(model_def: ModelDescriptor) -> bool:
    completion_api = str(getattr(model_def, "completion_api", "chat_completions") or "")
    return completion_api.strip().lower() == "responses"

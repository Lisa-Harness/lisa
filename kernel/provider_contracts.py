"""Vendor-neutral provider gateway contracts.

The kernel defines these dataclasses, protocols, and error types so the
turn runner and operations can produce/consume completion requests and
results without knowing which provider backend is wired. Concrete gateway
implementations live in the SDK (``sdk/providers/``) and are discovered
via the ``lisa.extensions.provider_gateways`` entry-point group.

This module has **no** vendor SDK imports (no ``openai``, no ``httpx``).
It is intentionally import-safe and dependency-light.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel as PydanticBaseModel

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
    #: Aggregated reasoning text for thinking/reasoning models that emit it
    #: separately from the visible answer. ``completion_content`` falls back
    #: to this when ``content`` is empty so reasoning models still record a
    #: visible assistant message. Matches ``StreamedChunk.reasoning_content``.
    reasoning_content: str = ""
    #: Wall-clock duration of the provider call in milliseconds. Surfaced flat
    #: (in addition to ``usage.duration_ms``) because the turn runner reads it
    #: as a flat completion attribute.
    duration_ms: float = 0.0


@dataclass(slots=True)
class StreamedChunk:
    """One chunk from a streaming completion.

    v1 streams tokens only — tool-call streaming is deferred to v2.
    """

    content: str = ""
    """Delta visible-answer text for this chunk."""

    reasoning_content: str = ""
    """Delta reasoning text for this chunk (gpt-5.x ``reasoning_content``).

    OpenAI reasoning models emit the model's "thinking" as
    ``delta.reasoning_content`` separately from ``delta.content``. When the
    model spends its budget reasoning and produces a terse/empty visible
    answer, ``content`` is empty but ``reasoning_content`` carries the
    substance. Aggregated reasoning content is surfaced as a fallback when
    the visible answer is empty (see ``_aggregate_streamed_chunks``).
    """

    finish_reason: str | None = None
    """``'stop'``, ``'length'``, ``'tool_calls'``, or ``None`` (mid-stream)."""

    usage: CompletionUsage | None = None
    """Final usage — only present on the last chunk for providers that
    include usage in the stream."""


@dataclass(slots=True)
class StreamedCompletion:
    """Aggregated result of a streaming completion."""

    content: str = ""
    reasoning_content: str = ""
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


def extract_request_id(value: Any) -> str:
    """Return the best-effort provider request id from one response or exception."""

    for attribute in ("_request_id", "request_id", "id"):
        candidate = getattr(value, attribute, "")
        if isinstance(candidate, str) and candidate:
            return candidate
    return ""


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


def _aggregate_streamed_chunks(chunks: list[StreamedChunk]) -> StreamedCompletion:
    """Aggregate a list of ``StreamedChunk`` into one ``StreamedCompletion``.

    v1 streams tokens only — tool-call streaming is deferred to v2, so this
    helper does not attempt to reconstruct tool calls from deltas. The last
    non-``None`` ``finish_reason`` wins. The last non-``None`` ``usage`` wins.

    ``reasoning_content`` (gpt-5.x reasoning models) is aggregated separately
    and kept on the ``StreamedCompletion`` so callers can surface it as a
    fallback when the visible ``content`` is empty. We do NOT fold reasoning
    into ``content`` here — that decision belongs to the caller (the turn
    runner), which knows whether to expose reasoning to the user.
    """

    parts: list[str] = []
    reasoning_parts: list[str] = []
    finish_reason = "stop"
    usage: CompletionUsage | None = None
    for chunk in chunks:
        if chunk.content:
            parts.append(chunk.content)
        if chunk.reasoning_content:
            reasoning_parts.append(chunk.reasoning_content)
        if chunk.finish_reason is not None:
            finish_reason = chunk.finish_reason
        if chunk.usage is not None:
            usage = chunk.usage
    return StreamedCompletion(
        content="".join(parts),
        reasoning_content="".join(reasoning_parts),
        tool_calls=[],
        usage=usage,
        finish_reason=finish_reason,
        chunks=chunks,
    )
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

import json
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


@dataclass(frozen=True, slots=True)
class SystemPromptSegment:
    """One ordered slice of the assembled system prompt, typed by cache intent.

    ``stable`` segments form the cross-turn prefix that providers can cache;
    ``ephemeral`` segments (routing hints, lifecycle briefings, handoff
    context) sit after the stable prefix and are expected to change turn to
    turn. The producer (kernel turn runner) owns ordering; the provider
    gateway owns whether/how to express the boundary on the wire.
    """

    text: str
    stability: str = "stable"  #: ``"stable"`` or ``"ephemeral"``

    @property
    def is_stable(self) -> bool:
        return self.stability == "stable"


def flatten_system_prompt_segments(
    segments: tuple[SystemPromptSegment, ...],
) -> str:
    """Render segments back to the legacy single system-prompt string.

    Joining with the historical ``\\n\\n`` separator preserves byte-for-byte
    equivalence with the pre-segmentation single-string prompt for providers
    that do not use the block boundary (OpenAI-family, Gemini).
    """
    return "\n\n".join(s.text for s in segments if s.text.strip())


@dataclass(slots=True)
class RequestTokenCount:
    """Input size plus the counting method; estimates must be labelled as such."""

    tokens: int
    method: str


@dataclass(slots=True)
class RequestContextBudget:
    """Auditable whole-request allocation, excluding no input surfaces."""

    window_tokens: int
    requested_completion_tokens: int
    completion_tokens: int
    margin_tokens: int
    pressure_headroom_tokens: int
    required_tokens: int
    input_tokens: int
    tool_budget_bytes: int
    counting_method: str
    reduced: bool


@dataclass(slots=True)
class CompletionRequest:
    """One normalized completion request sent through a provider gateway."""

    messages: ChatMessages
    model_def: ModelDescriptor
    max_tokens: int
    temperature: float
    tools: list[dict[str, Any]] | None = None
    response_format: Any | None = None
    #: Ordered system-prompt segments with cache-intent typing. When empty
    #: (the default), callers fall back to flattening the ``"system"`` role
    #: messages out of ``messages`` — historical behavior. When non-empty,
    #: the producer guarantees the segment texts, joined with ``\\n\\n``, are
    #: byte-identical to those system-role contents.
    system_prompt_segments: tuple[SystemPromptSegment, ...] = ()
    context_budget: RequestContextBudget | None = None
    #: Stable per-conversation routing key for providers with implicit
    #: prefix caching (OpenAI ``prompt_cache_key``). Empty means unset.
    prompt_cache_key: str = ""


@dataclass(slots=True)
class CompletionUsage:
    """Normalized per-call token usage extracted from one provider response.

    Fields are optional because not every provider surfaces every counter.
    ``prompt_tokens`` and ``completion_tokens`` are the canonical pair used
    for cost math; ``total_tokens`` falls out as the sum when the provider
    does not report it directly. ``cached_tokens`` is the provider-reported
    prompt-cache *read* count (tokens served from cache); ``reasoning_tokens``
    is reasoning output/effort.

    ``cache_creation_tokens`` is the provider-reported prompt-cache
    *creation/population* count — tokens paid at the write rate, not the
    read rate. Anthropic (``cache_creation_input_tokens``) and Azure
    GPT-5.6+ (``cache_write_tokens``) surface it; it defaults to 0 for
    providers that do not report it.

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
    cache_creation_tokens: int = 0
    duration_ms: float = 0.0


@dataclass(slots=True)
class ProviderContinuation:
    """Opaque provider-owned context, replayable only to its originating model."""

    provider: str
    model_id: str
    items: list[dict[str, Any]] = field(default_factory=list)


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
    #: Provider termination signal: ``'stop'``, ``'length'``, ``'tool_calls'``,
    #: or provider-specific text. Operations read this flat attribute for
    #: empty-completion diagnostics (distinguishing truncation from a genuine
    #: provider-empty response).
    finish_reason: str | None = None
    #: Authoritative truncation signal where the provider reports one:
    #: ``True`` = hit a token limit, ``False`` = clean stop, ``None`` =
    #: unknown. ``None`` is treated as a single safety-net continuation, not
    #: a multi-attempt loop.
    truncated: bool | None = None
    continuation: ProviderContinuation | None = None

    @property
    def prompt_tokens(self) -> int:
        """Flat prompt-token counter (operations read these as attributes)."""
        return self.usage.prompt_tokens if self.usage else 0

    @property
    def completion_tokens(self) -> int:
        """Flat completion-token counter."""
        return self.usage.completion_tokens if self.usage else 0

    @property
    def total_tokens(self) -> int:
        """Flat total-token counter."""
        return self.usage.total_tokens if self.usage else 0


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

    tool_call_deltas: list[dict[str, Any]] = field(default_factory=list)
    """Tool-call fragments streamed in this chunk (2026-10-12 delta stitching).

    Each entry mirrors one OpenAI ``delta.tool_calls`` element: ``index``
    (int, the stitch key), optional ``id``/``type`` on the first fragment of
    a call, and ``function.name``/``function.arguments`` fragments
    (arguments arrive in pieces and concatenate). Empty for text-only
    providers and backends that do not stream tool calls; the gateway
    contract stays "no deltas" for those, and the turn runner keeps the
    non-streaming fallback for a ``tool_calls`` finish with no deltas.
    """


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
        *,
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


#: Keyspace for slots the stitcher allocates itself. A backend's real
#: ``index`` values are integers, so a tuple key cannot collide with one
#: — the JSON streaming contract only defines an integer index, and
#: relying on its sign to keep the keyspaces apart left a negative index
#: able to land on a synthetic slot.
_SYNTHETIC_SLOT_KEY = ("synthetic",)

#: One staged tool call. ``_order`` is its arrival order and ``_index``
#: the backend index it was placed at; neither is part of the wire shape.
_StagedToolCall = dict[str, Any]


def _new_staged_tool_call(order: int, index: int | None = None) -> dict[str, Any]:
    """Return one empty staged tool call carrying its arrival order."""

    return {
        "id": "",
        "type": "",
        "name": "",
        "arguments": "",
        "_order": order,
        "_index": index,
    }


def _allocate_synthetic_slot(
    staged: dict[Any, dict[str, Any]],
) -> dict[str, Any]:
    """Return a new staged call in a slot no backend index can reach."""

    slot: Any = _SYNTHETIC_SLOT_KEY
    counter = 0
    while slot in staged:
        counter += 1
        slot = (_SYNTHETIC_SLOT_KEY[0], counter)
    return staged.setdefault(slot, _new_staged_tool_call(len(staged)))


def _find_staged_by_id(
    staged: dict[Any, dict[str, Any]],
    call_id: str,
) -> dict[str, Any] | None:
    """Return the staged call carrying ``call_id``, wherever it sits."""

    for candidate in staged.values():
        if candidate["id"] == call_id:
            return candidate
    return None


def _find_staged_by_index(
    staged: dict[Any, dict[str, Any]],
    index: int | None,
) -> dict[str, Any] | None:
    """Return the staged call at ``index``, or None when it is unknown.

    A concrete index that has not been seen yet identifies a call that
    has not started; it must open a new slot rather than continue
    whichever call started last.
    """

    if index is not None and index in staged:
        return staged[index]
    return None


def _start_new_staged_slot(
    staged: dict[Any, dict[str, Any]],
    index: int | None,
) -> dict[str, Any]:
    """Open a new staged call, preferring the backend's own index slot.

    A concrete index that has not been seen yet names a call that has not
    started, so it gets that slot. An occupied or unknown index gets a
    synthetic slot, which no backend index can collide with.
    """
    if index is not None and index not in staged:
        return staged.setdefault(index, _new_staged_tool_call(len(staged), index))
    return _allocate_synthetic_slot(staged)


def _arguments_are_complete(arguments: str) -> bool:
    """Return True when a call's arguments form one complete JSON object.

    The argument payload is the reliable signal for whether a call is
    finished. A call whose arguments already parse has delivered
    everything it will deliver, so the next fragment belongs to a
    DIFFERENT call. A call whose arguments are still incomplete (or are
    glued to a second object) is mid-payload, so the next fragment
    continues it.

    An empty object IS a finished payload: ``{}`` is what a
    zero-parameter call carries, and a backend that sends a real payload
    after it is starting a DIFFERENT call.
    """
    if not arguments:
        return False
    try:
        parsed = json.loads(arguments)
    except (json.JSONDecodeError, ValueError):
        return False
    return isinstance(parsed, dict)


def _call_is_complete(entry: dict[str, Any]) -> bool:
    """Return True when a staged call has delivered a complete payload."""

    return _arguments_are_complete(entry["arguments"])


def _name_is_compatible(entry_name: str, fragment_name: str) -> bool:
    """Return True when a fragment's name can belong to a staged call.

    An empty name on either side is compatible — the fragment simply did
    not carry one. Otherwise the names must match exactly, or one must
    be a prefix of the other because a backend may stream the name in
    pieces.
    """
    if not entry_name or not fragment_name:
        return True
    if entry_name == fragment_name:
        return True
    return entry_name.startswith(fragment_name) or fragment_name.startswith(entry_name)


def _calls_at_index(
    staged: dict[Any, dict[str, Any]],
    index: int | None,
) -> list[_StagedToolCall]:
    """Return the staged calls the backend placed at ``index``."""

    if index is None:
        return []
    return [c for c in staged.values() if c.get("_index") == index]


def _find_unfinished_at_index(
    staged: dict[Any, dict[str, Any]],
    index: int | None,
) -> dict[str, Any] | None:
    """Return the call at ``index`` that is still mid-payload.

    Only calls at that index are considered. A continuation fragment
    belongs to the call the backend placed at its index; routing it to a
    call at a DIFFERENT index merged two calls that never shared one.
    """
    for candidate in reversed(_calls_at_index(staged, index)):
        if not _call_is_complete(candidate):
            return candidate
    return None


def _find_unfinished_named(
    staged: dict[Any, dict[str, Any]],
    index: int | None,
    name: str,
) -> dict[str, Any] | None:
    """Return the call at ``index`` with a matching name that is mid-payload.

    A backend that streams a name in pieces sends the second piece with
    the rest of the payload, so the names are only prefix-compatible.

    Only the calls at the fragment's index are considered: a call the
    backend placed at another index is a different call.
    """
    for candidate in reversed(_calls_at_index(staged, index)):
        if _call_is_complete(candidate):
            continue
        if _name_is_compatible(candidate["name"], name):
            return candidate
    return None


def _find_unnamed_unfinished(
    staged: dict[Any, dict[str, Any]],
) -> dict[str, Any] | None:
    """Return the most recently started call that is still mid-payload."""

    candidates: list[_StagedToolCall] = list(staged.values())
    for candidate in reversed(candidates):
        if not _call_is_complete(candidate):
            return candidate
    return None


def _concatenation_is_one_object(staged: str, fragment: str) -> bool:
    """Return True when appending a fragment yields ONE JSON object.

    This is the decisive test for whether a fragment continues a call or
    starts a new one. A call's arguments are a single JSON object, so a
    fragment that completes one continues that call, and a fragment that
    would leave two objects behind starts a call of its own.

    The glued payload — two JSON objects end to end — is what the backend
    rejects with ``400 invalid tool call arguments``, and it is the
    signature of two calls merged into one.
    """
    try:
        parsed = json.loads(staged + fragment)
    except (json.JSONDecodeError, ValueError):
        return False
    return isinstance(parsed, dict)


def _names_are_incompatible(staged_name: str, fragment_name: str) -> bool:
    """Return True when two names plainly name DIFFERENT tools.

    The names alone cannot always tell a call from its continuation: a
    backend may repeat a name on every fragment or stream it in pieces.
    They are used only to REFUSE a merge, never to allow one. Two names
    are incompatible when neither is a prefix of the other, so a name
    streamed in pieces is never mistaken for a different tool.

    A name that is a prefix of the other is compatible — that is the
    shape of a name streamed in pieces. A repeated name is identical and
    compatible too.
    """
    if not staged_name or not fragment_name:
        return False
    if staged_name == fragment_name:
        return False
    return not (staged_name.startswith(fragment_name) or fragment_name.startswith(staged_name))


def _can_continue_call(
    staged: dict[str, Any],
    name: str,
    arguments: str,
) -> bool:
    """Return True when a fragment continues the staged call.

    The rules, in order:

    1. A fragment with no payload continues the staged call when it
       names the same tool, and starts a call of its own when it names a
       different one — otherwise the two names would be glued into a
       name no tool has.
    2. A fragment that completes the staged call's payload continues it.
    3. Two partial payloads are one call: neither can start a call.
    4. Two names that plainly name different tools are never one call.

    The name is checked only where the payload cannot decide. A name
    streamed in pieces is not a prefix of its own second half, so a name
    check placed first would split a call whose name arrived in pieces.
    The name refuses a merge the payload could not resolve.
    """
    staged_arguments = staged["arguments"]
    if not arguments or not staged_arguments:
        return not _names_are_incompatible(staged["name"], name)
    if _concatenation_is_one_object(staged_arguments, arguments):
        return True
    if not _arguments_are_complete(staged_arguments) and not _arguments_are_complete(arguments):
        return True
    return not _names_are_incompatible(staged["name"], name)


def _candidate_for_a_known_id(
    staged: dict[Any, dict[str, Any]],
    call_id: str,
    index: int | None,
    name: str,
) -> dict[str, Any] | None:
    """Return the staged call a fragment bearing this id continues.

    A call whose id has not arrived yet is waiting for it. The name
    picks out which one: a backend that interleaves parallel calls at
    one index sends each call's own name, and the late id belongs to
    the call with that name. A call that already carries an id is a
    different call.

    Only the calls the backend placed at this fragment's index are
    considered. A call at another index is a different call, and
    adopting it would merge two calls that never shared one.
    """
    waiting = [
        candidate
        for candidate in _calls_at_index(staged, index)
        if not candidate["id"] and not _call_is_complete(candidate)
    ]
    if not waiting and index is None:
        # Without an index there is nothing to narrow the search to, so
        # any call still waiting for an id may be the one.
        waiting = [
            candidate
            for candidate in list(staged.values())
            if not candidate["id"] and not _call_is_complete(candidate)
        ]
    if not waiting:
        return None
    for candidate in waiting:
        if candidate["name"] == name:
            return candidate
    for candidate in waiting:
        if _name_is_compatible(candidate["name"], name):
            return candidate
    if len(waiting) == 1:
        return waiting[0]
    # Several calls are waiting at this index and the fragment did not
    # name one of them. The most recently started is the current call
    # at the index: the backend interleaves the calls, and the id
    # belongs to the one it is currently filling.
    return waiting[-1]


def _candidate_for_a_fragment(
    staged: dict[Any, dict[str, Any]],
    index: int | None,
) -> dict[str, Any] | None:
    """Return the staged call a fragment bearing no id continues.

    The backend's index says where it placed the call, so only the calls
    at that index are considered. Without an index there is nothing to
    narrow the search to, so the most recently started call that is
    still mid-payload continues the fragment.
    """
    for candidate in reversed(_calls_at_index(staged, index)):
        if not _call_is_complete(candidate):
            return candidate
    if index is None:
        return _find_unnamed_unfinished(staged)
    return None


def _select_staged_slot(
    staged: dict[Any, dict[str, Any]],
    call_id: str,
    index: int | None,
    name: str,
    arguments: str,
) -> dict[str, Any] | None:
    """Return the staged call a fragment belongs to, starting one if needed.

    The rules:

    1. A fragment whose id is already staged resumes THAT call, wherever
       it sits. A backend that repeats the id on every fragment does not
       split its own call.
    2. A fragment whose id is new continues a staged call that is still
       waiting for an id — the id arrived late — and otherwise starts a
       new call.
    3. A fragment with no id continues the call at its index that is
       still mid-payload, and otherwise starts a new call.

    A call's identity comes from its ``id``; its completion comes from
    its argument payload. The tool name is used only to REFUSE a merge
    between two plainly different tools — a backend may repeat a name
    on every fragment or stream it in pieces, so it is never the sole
    discriminator.
    """
    if call_id:
        staged_call = _find_staged_by_id(staged, call_id)
        if staged_call is not None:
            # The id names the call, but the id alone is not enough: the
            # fragment must also not glue a second payload onto it. A
            # backend that repeats its id on every fragment sends a
            # PARTIAL payload each time, and that continues the call.
            #
            # A fragment that cannot continue the call must not be given
            # the same id either: two tool calls sharing an id are
            # rejected by the backend on their own. The fragment is
            # dropped instead — a call that cannot be dispatched cleanly
            # is better absent than duplicated.
            return staged_call if _can_continue_call(staged_call, name, arguments) else None
        candidate = _candidate_for_a_known_id(staged, call_id, index, name)
    else:
        candidate = _candidate_for_a_fragment(staged, index)
    if candidate is not None and _can_continue_call(candidate, name, arguments):
        return candidate
    return _start_new_staged_slot(staged, index)


def _merge_tool_call_name(entry: dict[str, Any], name: str) -> None:
    """Merge a fragment's tool name into a staged call.

    The name is normally a first-fragment field, but a backend may
    repeat it on every fragment or stream it in pieces. A repeat is
    ignored; a piece extends the name. The two are told apart by
    whether the new name starts with the name already recorded — a
    repetition does, and a piece does not.
    """
    current = entry["name"]
    if not current:
        entry["name"] = name
        return
    if name.startswith(current):
        # A repeated name: the call already has it.
        return
    if current.endswith(name):
        # The fragment repeated a suffix of the recorded name.
        return
    entry["name"] += name


def _stitch_tool_call_delta(
    staged: dict[Any, dict[str, Any]],
    delta: dict[str, Any],
) -> None:
    """Merge one streamed tool-call fragment into its aggregate entry.

    ``id`` is the call identity; ``index`` says where a backend placed
    the call. ``arguments`` concatenate across a call's fragments.
    ``id``/``type``/``name`` arrive on the first fragment that carries
    them.

    Some backends stream every parallel call with the SAME ``index`` (0)
    and rely on the call ``id`` to tell the calls apart — Ollama's
    MiniMax M3.1 Flash does exactly this. Stitching those by ``index``
    alone concatenated the calls into one garbage call:
    ``storage.list_filesstorage.read_file`` carrying three JSON
    objects glued end to end. The unparseable result was then correctly
    refused by the turn loop, but the broken call stayed in the message
    history and the next request replayed it — a strict backend answered
    ``400 invalid tool call arguments`` and killed the turn.
    """
    call_id = str(delta.get("id") or "")
    raw_index = delta.get("index")
    index = int(raw_index) if raw_index is not None else None
    function = delta.get("function") or {}
    name = str(function.get("name") or "")
    arguments = str(function.get("arguments") or "")

    entry = _select_staged_slot(staged, call_id, index, name, arguments)
    if entry is None:
        # The fragment cannot continue the call its id names and must
        # not be given that id again. It is dropped: a tool call that
        # cannot be dispatched cleanly is better absent than duplicated,
        # and the kernel refuses an unparseable payload anyway.
        return
    if entry.get("_index") is None and index is not None:
        # Remember where the backend placed this call: a later
        # continuation fragment carries the same index and must find it.
        entry["_index"] = index
    if call_id:
        entry["id"] = call_id
    if delta.get("type"):
        entry["type"] = str(delta["type"])
    if name:
        _merge_tool_call_name(entry, name)
    if arguments:
        entry["arguments"] += arguments


def _aggregate_streamed_chunks(chunks: list[StreamedChunk]) -> StreamedCompletion:
    """Aggregate a list of ``StreamedChunk`` into one ``StreamedCompletion``.

    v1 streamed tokens only; the 2026-10-12 change adds tool-call
    stitching (gateway yields ``tool_call_deltas`` fragments; this helper
    assembles them by ``index`` into the OpenAI tool-call shape so the turn
    runner can dispatch without the non-streaming regeneration pass). The
    last non-``None`` ``finish_reason`` wins. The last non-``None``
    ``usage`` wins.

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
    staged: dict[Any, dict[str, Any]] = {}
    for chunk in chunks:
        if chunk.content:
            parts.append(chunk.content)
        if chunk.reasoning_content:
            reasoning_parts.append(chunk.reasoning_content)
        if chunk.finish_reason is not None:
            finish_reason = chunk.finish_reason
        if chunk.usage is not None:
            usage = chunk.usage
        for delta in chunk.tool_call_deltas or []:
            _stitch_tool_call_delta(staged, delta)
    tool_calls = [
        {
            "id": entry["id"],
            "type": entry["type"] or "function",
            "function": {"name": entry["name"], "arguments": entry["arguments"]},
        }
        # Arrival order, not slot order: a synthetic slot is negative and
        # would otherwise sort the parallel calls ahead of the real ones.
        for _, entry in sorted(staged.items(), key=lambda item: item[1]["_order"])
    ]
    return StreamedCompletion(
        content="".join(parts),
        reasoning_content="".join(reasoning_parts),
        tool_calls=tool_calls,
        usage=usage,
        finish_reason=finish_reason,
        chunks=chunks,
    )

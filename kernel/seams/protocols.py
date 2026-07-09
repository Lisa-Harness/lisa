"""Kernel seams — host-injection protocols for blob, browser, state, and memory.

These Protocols define the dependency-inversion surface between the
runtime-neutral kernel and host applications. Hosts implement these
protocols and inject concrete instances into execution contexts; the
kernel never imports host code.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from kernel.models import (
    SurfaceContext,
    UnifiedState,
)


@runtime_checkable
class BlobIO(Protocol):
    """Protocol for host blob/object-storage I/O operations.

    Host apps implement this protocol and inject
    it into ``ToolExecutionContext.blob_io`` and ``OperationExecutionContext.blob_io``.
    """

    def read_unified_state(self) -> UnifiedState | None: ...
    def write_unified_state(self, state: UnifiedState) -> bool: ...

    def read_blob_bytes(self, container: str, blob_path: str) -> bytes | None: ...
    def write_blob_bytes(
        self,
        container: str,
        blob_path: str,
        payload: bytes,
        *,
        content_type: str | None = None,
    ) -> bool: ...
    def read_blob_text(self, container: str, blob_path: str) -> str | None: ...
    def write_blob_text(
        self,
        container: str,
        blob_path: str,
        payload: str,
        *,
        content_type: str | None = None,
    ) -> bool: ...

    def read_user_context_overlays(self) -> dict[str, Any] | None: ...
    def write_user_context_overlays(self, overlays: dict[str, Any]) -> bool: ...

    def read_surface_context_overlays(self) -> dict[str, Any] | None: ...
    def write_surface_context_overlays(self, overlays: dict[str, Any]) -> bool: ...

    def read_proactive_recipients(self) -> dict[str, Any] | None: ...
    def write_proactive_recipients(self, registry: dict[str, Any]) -> bool: ...

    def read_user_reminders(self) -> dict[str, Any] | None: ...
    def write_user_reminders(self, registry: dict[str, Any]) -> bool: ...

    def read_thread_state(self) -> dict[str, Any] | None: ...
    def write_thread_state(self, state: dict[str, Any]) -> bool: ...

    def read_memory_snapshot(self) -> dict[str, Any] | None: ...
    def write_memory_snapshot(self, snapshot: dict[str, Any]) -> bool: ...


@runtime_checkable
class OutboxReader(Protocol):
    """Protocol for reading parsed developer outbox entries."""

    def read_outbox_entries(self, max_bytes: int = 512_000) -> list[dict[str, Any]]: ...


@runtime_checkable
class StateProvider(Protocol):
    """Protocol for in-memory state access.

    Host apps implement this as a zero-argument callable returning the latest
    ``UnifiedState`` (or None).  Injected into execution contexts as
    ``ctx.state_provider``.
    """

    def __call__(self) -> UnifiedState | None: ...


@dataclass(slots=True)
class GuardResult:
    """The outcome of a guardrail check.

    ``allowed=False`` blocks the turn (input guard) or replaces the output
    (output guard). ``sanitized_text`` replaces the original text when the
    guard rewrites it. ``annotations`` carry provider-specific metadata
    (toxicity scores, PII findings, etc.) for audit logging.
    """

    allowed: bool
    blocked_reason: str | None = None
    sanitized_text: str | None = None
    annotations: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class InputGuard(Protocol):
    """Validate user input before it reaches the model.

    Called by the turn runner in Phase 1 (pre-turn), before system
    prompt assembly. If the guard returns ``allowed=False``, the turn
    is aborted and the ``blocked_reason`` is returned to the user.
    """

    async def validate(
        self,
        query: str,
        surface: SurfaceContext,
    ) -> GuardResult: ...


@runtime_checkable
class OutputGuard(Protocol):
    """Validate model output before it reaches the user.

    Called by the turn runner in Phase 3 (post-turn), after the tool
    loop produces final text and before ``store_memory()``. If the guard
    returns ``allowed=False``, the output is replaced with a safe
    fallback message and the original output is logged for audit.
    """

    async def validate(
        self,
        text: str,
        surface: SurfaceContext,
    ) -> GuardResult: ...


@runtime_checkable
class MemoryStore(Protocol):
    """Protocol for durable conversation memory.

    Host apps (e.g. ``app.memory_store``) implement this protocol and inject
    it into ``OperationExecutionContext.memory_store``.
    """

    async def record_memory(self, request: Any) -> Any: ...
    async def list_recent_memories(
        self,
        *,
        user_slug: str,
        operation_id: str = "",
        limit: int = 20,
        workspace_id: str = "default",
        memory_scope: str = "personal",
        surface_id: str = "",
    ) -> list[Any]: ...
    async def list_memories(
        self,
        *,
        user_slug: str,
        workspace_id: str = "default",
        memory_scope: str = "personal",
        surface_id: str = "",
        limit: int = 20,
    ) -> list[Any]: ...
    async def recall(self, query: str, **kwargs: Any) -> Any: ...


@runtime_checkable
class ThreadStore(Protocol):
    """Protocol for cached thread-state access used by data sources."""

    def iter_cached_states(self) -> list[Any]: ...


@runtime_checkable
class CompleteFn(Protocol):
    """Protocol for LLM completion calls.

    Injected into ``OperationExecutionContext.complete_fn``.
    """

    async def __call__(
        self,
        messages: list[dict[str, str]],
        *,
        model_id: str = "",
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: type[Any] | None = None,
        **kwargs: Any,
    ) -> Any: ...


@runtime_checkable
class ResolveModelFn(Protocol):
    """Protocol for model alias resolution.

    Injected into ``OperationExecutionContext.resolve_model_fn``.
    """

    def __call__(self, alias: str) -> str: ...


@runtime_checkable
class ExecuteToolFn(Protocol):
    """Protocol for invoking a tool by ID.

    Injected into ``OperationExecutionContext.execute_tool_fn``.
    """

    async def __call__(self, tool_id: str, ctx: Any) -> Any: ...


@runtime_checkable
class ExecuteOperationFn(Protocol):
    """Protocol for invoking an operation by ID.

    Injected into ``OperationExecutionContext.execute_operation_fn``.
    """

    async def __call__(self, operation_id: str, ctx: Any) -> Any: ...


@runtime_checkable
class DataSourceHooks(Protocol):
    """Protocol for host services injected into data-source execution.

    This mirrors the host-injected seam used by tool and operation execution
    contexts while keeping host-owned services behind explicit protocols.
    """

    blob_io: BlobIO | None
    outbox_reader: OutboxReader | None
    state_provider: StateProvider | None
    memory_store: MemoryStore
    thread_store: ThreadStore
    event_bus: Any


@dataclass(slots=True)
class DataSourceHookSet:
    """Concrete host-service bundle for data-source execution.

    Runtime owners build this typed bundle and pass it into data sources
    instead of relying on ``SimpleNamespace``-style dynamic objects.
    """

    blob_io: BlobIO | None
    state_provider: StateProvider | None
    memory_store: MemoryStore
    thread_store: ThreadStore
    event_bus: Any
    outbox_reader: OutboxReader | None = None


# ── Context seams ───────────────────────────────────────────────────────────
#
# The following Protocols invert the dependencies that ``kernel/context/``
# previously took on private ``app.*`` modules. Hosts inject concrete
# implementations through ``TurnHooks.context_seams``; the kernel ships public
# default implementations so third-party hosts get working context and memory
# without the private app.


@runtime_checkable
class ContentRegistry(Protocol):
    """Load authored text assets from merged content roots.

    Replaces ``app.content_registry.read_authored_text``. The default
    implementation reads from pack and extension content roots discovered
    through ``kernel.content_roots``.
    """

    def read_authored_text(self, relative_path: str) -> str: ...


@runtime_checkable
class ModelPolicy(Protocol):
    """Resolve model aliases and availability without coupling to a provider.

    Replaces ``app.model_policy``. The default implementation resolves
    aliases through ``kernel.model_manifest``.
    """

    def resolve_alias(self, alias: str) -> str: ...
    def is_available(self, model_id: str) -> bool: ...


@runtime_checkable
class ProviderGatewayProvider(Protocol):
    """Return a ``ProviderGateway`` for a given provider name.

    Replaces direct ``app.providers`` imports. The default implementation
    returns the gateways shipped in ``kernel.provider_core``.
    """

    def gateway_for(self, provider: str) -> Any | None: ...


@runtime_checkable
class EventBus(Protocol):
    """Lightweight pub/sub for salience events and consolidation triggers.

    Replaces ``app.event_bus.BusBackend``. The default implementation is an
    in-process ``asyncio`` queue bus with no external dependencies.
    """

    async def publish(self, topic: str, payload: dict[str, Any]) -> None: ...
    def subscribe(
        self, topic: str, handler: Callable[[dict[str, Any]], Awaitable[None]]
    ) -> None: ...


@runtime_checkable
class DataSanitizer(Protocol):
    """Apply redaction before memory writes or context assembly.

    Replaces ``app.data_sanitizer.apply_redaction_pipeline``. The default
    implementation is a pass-through that performs no redaction; production
    hosts override with a real sanitizer.
    """

    def redact(self, text: str, *, scope: str) -> str: ...


@runtime_checkable
class ExtensionEvidenceBuilder(Protocol):
    """Build context evidence records from installed extensions.

    Replaces ``app.extension_context_evidence.build_extension_context_evidence``.
    The default implementation returns an empty list; production hosts override
    with a builder that walks installed extensions.
    """

    def build_extension_context_evidence(
        self,
        runtime_context: Any,
        state: Any,
    ) -> list[Any]: ...

"""Kernel seams — host-injection protocols for blob, browser, state, and memory.

These Protocols define the dependency-inversion surface between the
runtime-neutral kernel and host applications. Hosts implement these
protocols and inject concrete instances into execution contexts; the
kernel never imports host code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from kernel.models import (
    SurfSessionState,
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

    def read_surf_workflow_session(
        self, workspace_id: str, surface_id: str
    ) -> SurfSessionState | None: ...
    def write_surf_workflow_session(self, session: SurfSessionState) -> bool: ...
    def delete_surf_workflow_session(self, workspace_id: str, surface_id: str) -> bool: ...

    def read_surf_screenshot(self, blob_path: str) -> bytes | None: ...
    def write_surf_trace(
        self,
        trace_json: str,
        screenshots: dict[str, bytes],
        artifact_prefix: str,
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
class BrowserAutomation(Protocol):
    """Protocol for Playwright-based browser automation.

    Host apps (e.g. ``app.browser_session``) implement this protocol and inject
    it into ``ToolExecutionContext.browser_automation``.
    """

    async def surf_url(self, url: str) -> Any: ...
    def flush(self) -> bool: ...


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

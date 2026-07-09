"""Runtime-neutral extension hooks for extension jobs and rendering seams."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

from kernel.models import (
    AnomalyAlert,
    ContextEvidence,
    IntelCollection,
    IntelMemoryArtifact,
    IntelSynthesis,
    UnifiedState,
)
from kernel.seams.protocols import StateProvider

ExtensionJobTrigger = Literal["interval", "cron"]


@runtime_checkable
class StateArtifactUpdater(Protocol):
    """Async callback that merges extension-produced artifacts into hot state."""

    async def __call__(
        self,
        *,
        collections: Mapping[str, IntelCollection] | None = None,
        analysis: Mapping[str, IntelSynthesis] | None = None,
        memory: Mapping[str, IntelMemoryArtifact] | None = None,
    ) -> UnifiedState | None: ...


@runtime_checkable
class StageChainExecutor(Protocol):
    """Async callback that runs one extension intelligence stage chain."""

    async def __call__(
        self,
        *,
        collection_scope: str,
        directional_query: str,
        payload: dict[str, Any],
        host_hooks: Any,
    ) -> dict[str, Any]: ...


@dataclass(slots=True)
class ExtensionRuntimeContext:
    """Host services injected into extension-owned jobs and consumers."""

    state_provider: StateProvider | None = None
    http_client: Any = None
    event_bus: Any = None
    openai_client: Any = None
    signal_connector_executor: Any = None
    data_source_executor: Any = None
    data_source_hooks: Any = None
    web_search: Callable[[str], Awaitable[Any]] | None = None
    render_search_results: Callable[[Any], str] | None = None
    complete_with_profile: Callable[..., Awaitable[str]] | None = None
    apply_state_updates: StateArtifactUpdater | None = None
    execute_stage_chain: StageChainExecutor | None = None


@dataclass(frozen=True, slots=True)
class ScheduledExtensionJob:
    """Declarative scheduled job owned by one extension package."""

    job_id: str
    runner: Callable[[ExtensionRuntimeContext], Awaitable[None]]
    trigger: ExtensionJobTrigger = "interval"
    cron_schedule: str = ""
    seconds: int = 0
    minutes: int = 0
    hour: int | str | None = None
    minute: int | str | None = None
    timezone: Any = None
    max_instances: int = 1
    coalesce: bool = True
    misfire_grace_time: int | None = None
    replace_existing: bool = True
    run_immediately: bool = False


@dataclass(frozen=True, slots=True)
class ExtensionDebugContextFragment:
    """Extension-owned `/debug context` lines rendered by the host generically."""

    poll_metadata_lines: tuple[str, ...] = ()
    artifact_lines: tuple[str, ...] = ()
    order: int = 100


@dataclass(frozen=True, slots=True)
class ExtensionHeartbeatContextFragment:
    """Extension-owned heartbeat prompt-context blocks rendered by the host."""

    prompt_context_blocks: tuple[str, ...] = ()
    order: int = 100


@dataclass(frozen=True, slots=True)
class ExtensionDigestSection:
    """Extension-owned digest section rendered by the host."""

    heading: str = ""
    lines: tuple[str, ...] = ()
    order: int = 100
    slot: Literal["infrastructure", "post_execution"] = "infrastructure"


@dataclass(frozen=True, slots=True)
class ExtensionStateHydrationPatch:
    """Extension-owned state fragments merged into heartbeat state."""

    collections: dict[str, IntelCollection] | None = None
    analysis: dict[str, IntelSynthesis] | None = None
    memory: dict[str, IntelMemoryArtifact] | None = None
    order: int = 100


ExtensionAnomalyDetector = Callable[[UnifiedState], Awaitable[list[AnomalyAlert]]]
ExtensionBusConsumer = Any
ExtensionAnomalyDetectorBuilder = Callable[
    [ExtensionRuntimeContext],
    Iterable[ExtensionAnomalyDetector],
]
ExtensionBusConsumerBuilder = Callable[[ExtensionRuntimeContext], Iterable[ExtensionBusConsumer]]
ExtensionDebugContextBuilder = Callable[
    [ExtensionRuntimeContext],
    Iterable[ExtensionDebugContextFragment],
]
ExtensionContextEvidenceBuilder = Callable[
    [ExtensionRuntimeContext, UnifiedState],
    Iterable[ContextEvidence],
]
ExtensionHeartbeatContextBuilder = Callable[
    [ExtensionRuntimeContext, UnifiedState],
    Iterable[ExtensionHeartbeatContextFragment],
]
ExtensionDigestSectionBuilder = Callable[
    [ExtensionRuntimeContext, UnifiedState],
    Iterable[ExtensionDigestSection],
]
ExtensionStateHydrationBuilder = Callable[
    [ExtensionRuntimeContext, UnifiedState],
    Iterable[ExtensionStateHydrationPatch],
]
ExtensionStateRefreshBuilder = Callable[
    [ExtensionRuntimeContext],
    Awaitable[Iterable[ExtensionStateHydrationPatch]],
]


__all__ = [
    "ExtensionAnomalyDetector",
    "ExtensionAnomalyDetectorBuilder",
    "ExtensionBusConsumer",
    "ExtensionBusConsumerBuilder",
    "ExtensionContextEvidenceBuilder",
    "ExtensionDebugContextBuilder",
    "ExtensionDebugContextFragment",
    "ExtensionDigestSection",
    "ExtensionDigestSectionBuilder",
    "ExtensionHeartbeatContextBuilder",
    "ExtensionHeartbeatContextFragment",
    "ExtensionJobTrigger",
    "ExtensionRuntimeContext",
    "ExtensionStateHydrationBuilder",
    "ExtensionStateHydrationPatch",
    "ExtensionStateRefreshBuilder",
    "ScheduledExtensionJob",
    "StageChainExecutor",
    "StateArtifactUpdater",
]

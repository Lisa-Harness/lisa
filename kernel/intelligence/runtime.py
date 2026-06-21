"""Runtime-neutral stage execution contracts for extension intelligence chains."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from kernel.intelligence.spec import (
    ExtensionIntelligenceSpec,
    IntelligenceStageKind,
    IntelligenceStageSpec,
)


@runtime_checkable
class StageExecutionHostHooks(Protocol):
    """Host-owned stage primitives exposed to extension stage modules."""

    async def run_local_analysis(
        self,
        *,
        collection_scope: str,
        raw_items: list[Any],
        directional_query: str,
    ) -> Any: ...

    async def run_synthesis(
        self,
        *,
        collection_scope: str,
        local_analysis: Any,
    ) -> Any: ...

    def build_memory_artifact(
        self,
        *,
        collection_scope: str,
        local_analysis: Any,
        synthesis: Any,
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class StageExecuteContext:
    """Context provided to one extension stage implementation."""

    extension_id: str
    collection_scope: str
    spec: ExtensionIntelligenceSpec
    stage: IntelligenceStageSpec
    directional_query: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    stage_outputs: dict[IntelligenceStageKind, Any] = field(default_factory=dict)
    host_hooks: StageExecutionHostHooks | None = None

    def require_stage_output(self, stage_kind: IntelligenceStageKind) -> Any:
        """Return one prior stage output or raise when missing."""

        if stage_kind not in self.stage_outputs:
            raise ValueError(
                f"stage output '{stage_kind}' is unavailable for stage '{self.stage.stage_id}'"
            )
        return self.stage_outputs[stage_kind]

    def require_host_hooks(self) -> StageExecutionHostHooks:
        """Return host hooks for this execution context."""

        if self.host_hooks is None:
            raise ValueError(f"host_hooks are required for stage '{self.stage.stage_id}'")
        return self.host_hooks


@dataclass(frozen=True, slots=True)
class StageResult:
    """Output envelope returned by one extension stage implementation."""

    stage_id: str
    stage_kind: IntelligenceStageKind
    output: Any
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "StageExecuteContext",
    "StageExecutionHostHooks",
    "StageResult",
]

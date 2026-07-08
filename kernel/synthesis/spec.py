"""Runtime-neutral intelligence specification contracts.

An extension intelligence spec declares how one extension participates in the
shared post-collection pipeline without replacing connector/data-source owners.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from kernel.models import ConversationSurface, MemoryScope

IntelligenceStageKind = Literal["local_analysis", "synthesis", "memory"]


@dataclass(frozen=True, slots=True)
class IntelligenceStageSpec:
    """One specialized downstream stage owned by an extension."""

    stage_id: str
    stage_kind: IntelligenceStageKind
    specialization: str
    implementation_id: str = ""
    description: str = ""


@dataclass(frozen=True, slots=True)
class ExtensionIntelligenceSpec:
    """Declarative extension contract for downstream intelligence stages.

    This spec is metadata-driven on purpose so extensions can opt into only the
    stages they need while keeping collection ownership in SignalConnector and
    DataSource implementations.
    """

    extension_id: str
    version: str = "1"
    collection_scopes: tuple[str, ...] = ()
    signal_connector_ids: tuple[str, ...] = ()
    data_source_ids: tuple[str, ...] = ()
    stages: tuple[IntelligenceStageSpec, ...] = ()
    ambient_projector_id: str = ""
    proactive_projector_id: str = ""
    query_tool_ids: tuple[str, ...] = ()
    upstream_dependencies: tuple[str, ...] = ()
    memory_scopes: tuple[MemoryScope, ...] = ("personal", "shared", "channel")
    surfaces: tuple[ConversationSurface, ...] = ("personal_dm", "group_chat", "channel")
    tags: tuple[str, ...] = field(default_factory=tuple)


def validate_intelligence_spec(spec: ExtensionIntelligenceSpec) -> tuple[str, ...]:
    """Return validation errors for one extension intelligence spec."""

    errors: list[str] = []

    if not spec.signal_connector_ids and not spec.data_source_ids:
        errors.append(
            "must declare at least one collector (signal_connector_ids or data_source_ids)"
        )

    normalized_scopes = [scope.strip().lower() for scope in spec.collection_scopes if scope.strip()]
    if len(normalized_scopes) != len(set(normalized_scopes)):
        errors.append("collection_scopes must not contain duplicates")

    seen_stage_ids: set[str] = set()
    stage_kinds: dict[IntelligenceStageKind, int] = {
        "local_analysis": -1,
        "synthesis": -1,
        "memory": -1,
    }
    for index, stage in enumerate(spec.stages):
        if stage.stage_id in seen_stage_ids:
            errors.append(f"duplicate stage_id '{stage.stage_id}'")
        seen_stage_ids.add(stage.stage_id)

        if not stage.stage_id.startswith(f"{spec.extension_id}."):
            errors.append(f"stage_id '{stage.stage_id}' must start with '{spec.extension_id}.'")
        if len(stage.specialization.strip()) < 8:
            errors.append(
                f"stage '{stage.stage_id}' must declare a specialization of at least 8 characters"
            )
        if stage_kinds[stage.stage_kind] != -1:
            errors.append(f"duplicate stage_kind '{stage.stage_kind}'")
        stage_kinds[stage.stage_kind] = index

    local_index = stage_kinds["local_analysis"]
    synth_index = stage_kinds["synthesis"]
    memory_index = stage_kinds["memory"]
    if synth_index != -1 and local_index == -1:
        errors.append("synthesis stage requires local_analysis stage")
    if memory_index != -1 and synth_index == -1:
        errors.append("memory stage requires synthesis stage")
    if local_index != -1 and synth_index != -1 and local_index > synth_index:
        errors.append("local_analysis stage must come before synthesis stage")
    if synth_index != -1 and memory_index != -1 and synth_index > memory_index:
        errors.append("synthesis stage must come before memory stage")

    return tuple(errors)


__all__ = [
    "ExtensionIntelligenceSpec",
    "IntelligenceStageKind",
    "IntelligenceStageSpec",
    "validate_intelligence_spec",
]

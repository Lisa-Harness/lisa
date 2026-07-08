"""Response-format model for the architecture_atlas declarative operation."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ArchitectureAtlasOutput(BaseModel):
    """Structured output envelope for the architecture atlas generator."""

    atlas_markdown: str = Field(
        default="",
        description="Markdown-formatted architecture overview with Mermaid diagrams where helpful.",
    )
    layer_model: list[str] = Field(
        default_factory=list,
        description="Named architectural layers (e.g. kernel, sdk, packs, host).",
    )
    module_ownership: list[str] = Field(
        default_factory=list,
        description="Key module-to-layer ownership mappings.",
    )
    runtime_flows: list[str] = Field(
        default_factory=list,
        description="Critical runtime flows (turn loop, slice, routing, context assembly).",
    )
    extension_points: list[str] = Field(
        default_factory=list,
        description="Documented extension points (hooks, seams, operations, MCP servers).",
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="Architectural constraints and invariants.",
    )
    known_gaps: list[str] = Field(
        default_factory=list,
        description="Known gaps or deferred capabilities.",
    )

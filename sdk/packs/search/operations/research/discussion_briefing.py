"""Structured output contract for declarative discussion_briefing."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


def _compact_json_schema(schema: dict[str, Any]) -> None:
    """Strip non-essential schema metadata for structured-output providers."""
    schema.pop("description", None)
    schema.pop("title", None)

    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return

    for property_schema in properties.values():
        if not isinstance(property_schema, dict):
            continue
        property_schema.pop("description", None)
        property_schema.pop("title", None)

        items_schema = property_schema.get("items")
        if isinstance(items_schema, dict):
            items_schema.pop("description", None)
            items_schema.pop("title", None)


class DiscussionBriefingStructuredOutput(BaseModel):
    """Declarative response envelope for discussion_briefing."""

    model_config = ConfigDict(json_schema_extra=_compact_json_schema)

    output_text: str = Field(
        default="",
        description="Briefing markdown, or empty when no enrichment is needed.",
    )
    source_kind: str = Field(
        default="",
        description="Chosen source kind: web, or empty when no briefing is emitted.",
    )
    briefing_points: list[str] = Field(
        default_factory=list,
        description="Compact grounded bullets used to assemble the briefing.",
    )
    source_urls: list[str] = Field(
        default_factory=list,
        description="Grounded source URLs copied from web-search results when used.",
    )
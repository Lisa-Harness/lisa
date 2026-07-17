"""Structured output contract for declarative web_search."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


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


class WebSearchStructuredOutput(BaseModel):
    """Declarative response envelope for the web_search operation."""

    model_config = ConfigDict(json_schema_extra=_compact_json_schema)

    output_text: str = Field(..., description="Full markdown research briefing.")
    core_consensus: str = Field(..., description="Main grounded consensus across results.")
    contradictory_points: list[str] = Field(
        default_factory=list,
        description="Grounded contradictory or alternative viewpoints.",
    )
    actionable_takeaways: list[str] = Field(
        default_factory=list,
        description="Actionable takeaways grounded in the search snippets.",
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Unique exact source URLs copied from tool output only.",
    )

    @field_validator("core_consensus", mode="before")
    @classmethod
    def _coerce_core_consensus(cls, value: Any) -> Any:
        """Tolerate models that return ``core_consensus`` as a list of strings."""
        if isinstance(value, list):
            return "\n".join(str(item) for item in value)
        return value

    @field_validator("output_text", mode="before")
    @classmethod
    def _coerce_output_text(cls, value: Any) -> Any:
        """Tolerate models that return ``output_text`` as a list of strings."""
        if isinstance(value, list):
            return "\n".join(str(item) for item in value)
        return value
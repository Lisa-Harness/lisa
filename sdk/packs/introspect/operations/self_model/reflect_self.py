"""Response-format model for the reflect_self declarative operation."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ReflectSelfOutput(BaseModel):
    """Structured output envelope for Lisa's reflective self-assessment."""

    overall_health: str = Field(
        default="unknown",
        description="One of: healthy, degraded, struggling, unknown.",
    )
    health_reasoning: str = Field(
        default="",
        description="One paragraph holistic assessment of current operational state.",
    )
    attention_items: list[str] = Field(
        default_factory=list,
        description="Specific things Lisa should be aware of right now.",
    )
    behavioral_patterns: list[str] = Field(
        default_factory=list,
        description="Observed patterns in recent behavior (e.g. 'last 10 turns search-heavy').",
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in this assessment (0.0 to 1.0).",
    )

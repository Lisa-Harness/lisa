"""Tests for the reflect_self declarative operation."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from sdk.packs.introspect.operations.self_model.reflect_self import ReflectSelfOutput


class _StubModelDef(BaseModel):
    """Minimal model definition stub for the declarative runtime."""

    model_id: str = "test-model"
    provider: str = "test"
    reasoning_model: bool = False


def _build_mock_complete_fn(output: ReflectSelfOutput):
    """Build a mock complete_fn that returns the given ReflectSelfOutput as JSON."""

    async def _complete_fn(
        *,
        messages: list[dict[str, Any]],
        model_def: Any = None,
        max_tokens: int = 1000,
        temperature: float = 0.2,
        tools: list[dict[str, Any]] | None = None,
        response_format: type[BaseModel] | None = None,
    ) -> Any:
        return _StubCompletion(content=output.model_dump_json())

    return _complete_fn


class _StubCompletion:
    """Minimal completion stub matching the shape the declarative runtime expects."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.tool_calls: list[dict[str, Any]] = []
        self.prompt_tokens = 10
        self.completion_tokens = 20
        self.total_tokens = 30
        self.cached_tokens = 0
        self.reasoning_tokens = 0
        self.duration_ms = 100.0
        self.latency_ms = 100


def _build_mock_resolve_model():
    """Build a mock resolve_model_fn."""

    def _resolve_model(model_id: str) -> Any:
        return _StubModelDef()

    return _resolve_model


def test_reflect_self_output_model_validates() -> None:
    """ReflectSelfOutput validates correctly with all fields."""
    output = ReflectSelfOutput(
        overall_health="healthy",
        health_reasoning="All systems nominal. RSS at 187MB. MCP pool warm.",
        attention_items=["mem0 backend unreachable"],
        behavioral_patterns=["last 10 turns were 70% search-heavy"],
        confidence=0.85,
    )
    assert output.overall_health == "healthy"
    assert "nominal" in output.health_reasoning
    assert len(output.attention_items) == 1
    assert len(output.behavioral_patterns) == 1
    assert output.confidence == 0.85


def test_reflect_self_output_defaults() -> None:
    """ReflectSelfOutput has sensible defaults."""
    output = ReflectSelfOutput()
    assert output.overall_health == "unknown"
    assert output.health_reasoning == ""
    assert output.attention_items == []
    assert output.behavioral_patterns == []
    assert output.confidence == 0.0


def test_reflect_self_output_json_roundtrip() -> None:
    """ReflectSelfOutput can be serialized and deserialized from JSON."""
    output = ReflectSelfOutput(
        overall_health="degraded",
        health_reasoning="mem0 backend is unreachable. Token costs elevated.",
        attention_items=["mem0 unreachable", "token costs elevated"],
        behavioral_patterns=["search-heavy in last 10 turns"],
        confidence=0.7,
    )
    json_str = output.model_dump_json()
    parsed = ReflectSelfOutput.model_validate_json(json_str)
    assert parsed.overall_health == "degraded"
    assert parsed.confidence == 0.7

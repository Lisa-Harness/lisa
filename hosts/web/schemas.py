"""Pydantic models for the Lisa web host API contract.

``pydantic>=2.7`` is already a dependency of the web host. These models
provide validation, serialization, and JSON Schema generation for the
REST and SSE endpoints.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TurnRequest(BaseModel):
    """Request body for ``POST /api/turn``."""

    message: str = Field(..., min_length=1, description="The user message to process.")


class TurnResponse(BaseModel):
    """Response body for ``POST /api/turn``."""

    response_text: str = Field(default="", description="The model's response text.")
    last_turn_context: dict[str, Any] | None = Field(
        default=None, description="Diagnostic context from the most recent turn."
    )
    is_error: bool = Field(default=False, description="Whether the turn ended in an error.")


class SSEStageEvent(BaseModel):
    """Payload for ``event: stage`` SSE frames."""

    label: str = Field(..., description="Stage label (e.g. model_call, tool_dispatch).")
    data: dict[str, Any] | None = Field(
        default=None, description="Stage metadata (mode_id, provider, iteration, etc.)."
    )


class SSETokenEvent(BaseModel):
    """Payload for ``event: token`` SSE frames."""

    text: str = Field(..., description="One streamed text chunk from the model.")


class SSEDoneEvent(BaseModel):
    """Payload for ``event: done`` SSE frames."""

    text: str = Field(default="", description="The full response text.")
    html: str | None = Field(default=None, description="Server-rendered HTML of the response.")
    data: dict[str, Any] | None = Field(
        default=None, description="Turn metadata (token counts, mode, etc.)."
    )


class SSEErrorEvent(BaseModel):
    """Payload for ``event: error`` SSE frames."""

    message: str = Field(..., description="Human-readable error message.")


class ModelsResponse(BaseModel):
    """Response body for ``GET /api/models``."""

    models: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = Field(default=None)


class TokensResponse(BaseModel):
    """Response body for ``GET /api/tokens``."""

    session: dict[str, Any] = Field(default_factory=dict)
    turn: dict[str, Any] = Field(default_factory=dict)
    recent_events: list[dict[str, Any]] = Field(default_factory=list)


class StatusResponse(BaseModel):
    """Response body for ``GET /api/status``."""

    mode: str = Field(default="")
    stage: str = Field(default="")
    models_loaded: int = Field(default=0)
    mcp_started: bool = Field(default=False)
    restart_required: bool = Field(default=False)

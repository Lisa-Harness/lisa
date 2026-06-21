"""Microsoft Clarity extension-owned Pydantic models."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from kernel.models import IntelCollection


class ClarityExportParams(BaseModel):
    num_of_days: int = 1
    dimension1: str | None = None
    dimension2: str | None = None
    dimension3: str | None = None


class ClarityMetricRow(BaseModel):
    metric_name: str
    information: list[dict[str, Any]] = Field(default_factory=list)


class ClarityLiveInsightsResponse(IntelCollection):
    generated_at: str = ""
    metrics: list[ClarityMetricRow] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)


class ClarityPollMetadata(BaseModel):
    updated_at: str = ""
    date_utc: str = ""
    request_count: int = 0
    last_polled_at: str = ""
    last_success_at: str = ""
    last_error: str = ""


class ClarityEventBase(BaseModel):
    type: str
    timestamp: datetime
    session_id: str | None = None
    page_url: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class ClarityClickEvent(ClarityEventBase):
    x: int | None = None
    y: int | None = None
    node_selector: str | None = None
    subtype: str | None = None


class ClarityJsErrorEvent(ClarityEventBase):
    message: str = ""
    stack: str | None = None
    file: str | None = None
    line: int | None = None
    column: int | None = None


class ClarityPageView(BaseModel):
    url: str
    title: str | None = None
    timestamp: datetime
    duration_ms: int | None = None
    scroll_depth_pct: int | None = None


class ClaritySessionSummary(BaseModel):
    session_id: str
    visitor_id: str | None = None
    start_time: datetime | None = None
    end_time: datetime | None = None
    duration_ms: int | None = None
    page_count: int | None = None
    click_count: int | None = None
    rage_click_count: int | None = None
    dead_click_count: int | None = None
    js_error_count: int | None = None
    entry_url: str | None = None
    exit_url: str | None = None
    user_agent: str | None = None
    device: str | None = None
    country: str | None = None
    pages: list[ClarityPageView] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)

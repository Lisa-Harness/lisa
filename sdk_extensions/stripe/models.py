"""Stripe signal connector snapshot model."""

from __future__ import annotations

from pydantic import BaseModel


class StripeMetrics(BaseModel):
    collected_at: str
    trials_active: int = 0
    trials_started_24h: int = 0
    subscriptions_active: int = 0
    mrr_cents: int = 0
    status: str = "ok"

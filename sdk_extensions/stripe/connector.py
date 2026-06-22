"""Stripe subscription and trial activity signal connector.

Monitors Stripe API for subscription and trial metrics.
Polled by a host scheduler.

What it monitors:
- Active subscriptions (status='active') count
- Active trials (status='trialing') count

What signals it emits:
- connector status: always 'ok' (no threshold-based degradation logic)
- subscriptions_active: count of active subscriptions
- trials_active: count of active trials

Use case:
- Quick health check on billing system and customer acquisition funnel
- Track MRR (monthly recurring revenue) through subscription count
- Understand trial-to-paid conversion by comparing trial vs active ratios

API endpoints:
- GET /v1/subscriptions (with status=active, limit=100)
- GET /v1/subscriptions (with status=trialing, limit=100)

Limit=100 and manual counting: The Stripe API removed the 'total_count' expand parameter,
so we use limit=100 per call and count returned objects. This assumes no more than 100
concurrent trials or active subscriptions per 3-minute collection window.

Timeout: all Stripe API calls use external_signal_timeout_seconds (typically 15s).
"""
from __future__ import annotations

import httpx

from kernel.intake.connectors import SignalConnector
from sdk_extensions.stripe.config import get_stripe_connector_settings
from sdk_extensions.stripe.models import StripeMetrics


class StripeSignalConnector(SignalConnector[StripeMetrics]):
    """Stripe billing and subscription health monitor.

    Singleton connector that polls Stripe API for subscription and trial metrics.
    Injected as a dependency into SignalConnectorExecutor, initialized once at app startup.

    Attributes:
        connector_id: 'stripe' (used to identify this connector in logs and results)
        display_name: 'Stripe' (human-readable name for UI/reports)

    Dependencies:
        - STRIPE_API_KEY (SecretStr env var): Stripe secret key with read:subscription scope

    Note:
        Stripe has removed total_count from API responses, so we limit=100 and count.
        This is safe as long as you have fewer than 100 active subscriptions or trials
        at any given time. Scale this threshold if needed.
    """
    connector_id = "stripe"
    display_name = "Stripe"

    def missing_configuration(self) -> list[str]:
        """Check if required Stripe API credentials are configured.

        Returns:
            List of missing env vars: ['STRIPE_API_KEY']. Empty list if set.
        """
        connector_settings = get_stripe_connector_settings()
        return ["STRIPE_API_KEY"] if not connector_settings.api_key else []

    def build_unconfigured_snapshot(self, collected_at: str) -> StripeMetrics:
        """Build a snapshot when Stripe API credentials are missing.

        Args:
            collected_at: ISO 8601 timestamp (UTC).

        Returns:
            StripeMetrics with status='unconfigured' and minimal default data.
        """
        return StripeMetrics(collected_at=collected_at, status="unconfigured")

    def build_error_snapshot(self, collected_at: str) -> StripeMetrics:
        """Build a snapshot when collection fails after retries.

        Args:
            collected_at: ISO 8601 timestamp (UTC).

        Returns:
            StripeMetrics with status='error' and minimal default data.
        """
        return StripeMetrics(collected_at=collected_at, status="error")

    async def collect(self, client: httpx.AsyncClient, collected_at: str) -> StripeMetrics:
        """Collect Stripe subscription and trial metrics.

        Fetches counts from two Stripe list endpoints:
        1. GET /v1/subscriptions?status=active (count of active subscriptions)
        2. GET /v1/subscriptions?status=trialing (count of active trials)

        Both calls use limit=100. Assumes no more than 100 concurrent active or trial
        subscriptions. If you scale beyond this, increase the limit parameter and
        implement pagination.

        Status: always 'ok' (no threshold-based degradation). Failures are handled
        by the executor and result in status='error'.

        Args:
            client: httpx.AsyncClient for making API requests.
            collected_at: ISO 8601 UTC timestamp when collection started.

        Returns:
            StripeMetrics with subscriptions_active and trials_active counts.

        Raises:
            httpx.HTTPStatusError: If any API call returns non-2xx status.
            httpx.TimeoutException: If any API call exceeds external_signal_timeout_seconds.
            httpx.NetworkError: If network connectivity fails.
        """
        connector_settings = get_stripe_connector_settings()
        headers = {"Authorization": f"Bearer {connector_settings.api_key}"}
        active_response = await client.get(
            "https://api.stripe.com/v1/subscriptions",
            headers=headers,
            timeout=connector_settings.timeout_seconds,
            # Use limit=100 and count data; total_count expand was removed from Stripe API.
            params={"status": "active", "limit": 100},
        )
        active_response.raise_for_status()
        active_data = active_response.json()

        trial_response = await client.get(
            "https://api.stripe.com/v1/subscriptions",
            headers=headers,
            timeout=connector_settings.timeout_seconds,
            params={"status": "trialing", "limit": 100},
        )
        trial_response.raise_for_status()
        trial_data = trial_response.json()

        return StripeMetrics(
            collected_at=collected_at,
            subscriptions_active=len(active_data.get("data", [])),
            trials_active=len(trial_data.get("data", [])),
            status="ok",
        )

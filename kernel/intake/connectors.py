"""Base contract for external signal connectors.

Signal connectors are singleton, stateless monitors for external systems.
Each connector:
- Defines what configuration is required (env vars or settings)
- Implements a collect() method to fetch fresh data from the external system
- Returns a typed snapshot (Pydantic model) that can be serialized to state cache or blob storage
- Provides error recovery paths for misconfiguration or collection failures

Signal connectors are invoked by a host scheduler or SignalConnectorExecutor,
which handles retries, timing, and error classification.

TypeVar SnapshotT is the Pydantic model returned by each connector's collect() method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Literal, TypeVar

import httpx
from pydantic import BaseModel

SignalRunMode = Literal["scheduled", "on_demand", "live_test"]
SnapshotT = TypeVar("SnapshotT", bound=BaseModel)


class SignalConnector(ABC, Generic[SnapshotT]):
    """Base class for stateless, repeatable external signal collection.

    Each signal connector monitors one external system (Cloudflare, GitHub, Stripe, etc.)
    and provides three key capabilities:

    1. Configuration validation: missing_configuration() returns required env vars not set.
    2. Collection: collect() fetches fresh data and returns a typed snapshot.
    3. Error recovery: build_unconfigured_snapshot() and build_error_snapshot() provide
       graceful fallbacks when collection cannot run.

    Subclasses must implement collect(), build_unconfigured_snapshot(), and
    build_error_snapshot(). The extract_status() method can be overridden to customize
    status classification logic beyond the default snapshot.status field.
    """

    connector_id: str
    display_name: str
    intel_collection_key: str | None = None
    """If set, the SignalConnectorExecutor caller will write the connector's snapshot
    into ``UnifiedState.intel_collections[intel_collection_key]`` after each successful
    or error collection run.  Connectors that do not need to populate intel_collections
    leave this as ``None``."""

    def missing_configuration(self) -> list[str]:
        """Return list of missing environment variables or settings required to run.

        Called by SignalConnectorExecutor before attempting collection. If any
        variables are missing, the executor will short-circuit to an unconfigured
        snapshot and skip the collect() call.

        Returns:
            List of missing env var names (e.g., ['API_TOKEN', 'ZONE_ID']).
            Empty list means configuration is complete.
        """
        return []

    def extract_status(self, snapshot: SnapshotT) -> str:
        """Extract normalized run status from a provider snapshot.

        Default implementation looks for a 'status' field in the snapshot model.
        Subclasses can override to provide custom classification logic
        (e.g., map multiple fields to a single 'ok' / 'degraded' / 'critical' status).

        Args:
            snapshot: Typed snapshot returned by collect().

        Returns:
            Normalized status string: 'ok', 'degraded', 'critical', 'error', or custom.
        """
        status = getattr(snapshot, "status", "ok")
        return status if isinstance(status, str) and status else "ok"

    @abstractmethod
    async def collect(self, client: httpx.AsyncClient, collected_at: str) -> SnapshotT:
        """Collect a fresh snapshot from the external system.

        Must make external API calls (HTTP, GraphQL, etc.) using the provided httpx client.
        Should use settings.external_signal_timeout_seconds for all requests.

        Args:
            client: httpx.AsyncClient to use for API calls.
            collected_at: ISO 8601 timestamp (UTC) for when collection started.

        Returns:
            Typed snapshot model (SnapshotT) with collected data and status.

        Raises:
            httpx.HTTPStatusError: HTTP call failed (non-2xx status).
            httpx.TimeoutException: Request timeout exceeded.
            httpx.NetworkError: Network or connection error.
            Any other exception: API client exception or parsing error.
        """

    @abstractmethod
    def build_unconfigured_snapshot(self, collected_at: str) -> SnapshotT:
        """Build a snapshot when required configuration is missing.

        Called when missing_configuration() returns non-empty list.
        Should return a snapshot with status='unconfigured' and minimal/default data.

        Args:
            collected_at: ISO 8601 timestamp (UTC) for the snapshot.

        Returns:
            Typed snapshot model with status='unconfigured'.
        """

    @abstractmethod
    def build_error_snapshot(self, collected_at: str) -> SnapshotT:
        """Build a snapshot when collection fails.

        Called after all retry attempts are exhausted or a non-retryable error occurs.
        Should return a snapshot with status='error' and minimal/fallback data.

        Args:
            collected_at: ISO 8601 timestamp (UTC) for the snapshot.

        Returns:
            Typed snapshot model with status='error'.
        """

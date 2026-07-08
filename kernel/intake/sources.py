"""Base contract for data sources.

Data sources are top-level types (alongside signal connectors and tools)
that fetch, aggregate, and enrich external data over time. Because they often
need to compare current state with historical state (e.g. rolling windows),
they are permitted to maintain stateful caches across execution cycles.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import httpx
from openai import AsyncAzureOpenAI
from pydantic import BaseModel

from kernel.seams.protocols import DataSourceHooks

DataSnapshotT = TypeVar("DataSnapshotT", bound=BaseModel)


class DataSource(ABC, Generic[DataSnapshotT]):
    """Base generic contract for external data sources.

    Subclasses must implement execute_cycle() and the fallback builders.
    """

    source_id: str
    display_name: str

    def missing_configuration(self) -> list[str]:
        """Return list of missing environment variables or settings required to run.

        Returns:
            List of missing env var names. Empty list means configuration is complete.
        """
        return []

    def extract_status(self, snapshot: DataSnapshotT) -> str:
        """Extract normalized run status from a data snapshot."""
        status = getattr(snapshot, "status", "ok")
        return status if isinstance(status, str) and status else "ok"

    @abstractmethod
    async def execute_cycle(
        self,
        http_client: httpx.AsyncClient,
        openai_client: AsyncAzureOpenAI | None,
        *,
        hooks: DataSourceHooks,
    ) -> DataSnapshotT:
        """Execute one full collection and enrichment cycle.

        Args:
            http_client: the shared AsyncClient for HTTP calls.
            openai_client: the AI client for enrichment, if configured.
            hooks: host-service hooks (blob/state/memory/thread/event bus). Required.

        Returns:
            A snapshot of the data source state.
        """

    @abstractmethod
    def build_unconfigured_snapshot(self) -> DataSnapshotT:
        """Build a snapshot when required configuration is missing."""

    @abstractmethod
    def build_error_snapshot(self, error: Exception) -> DataSnapshotT:
        """Build a snapshot when collection fundamentally fails."""

"""Base contract for pack-owned domain-knowledge providers.

Domain knowledge is distinct from ambient intel and self-knowledge:

- Ambient intel is broad operational or external signal data that can be ranked,
  merged, and reused across prompts.
- Self-knowledge is Lisa's deterministic description of her own architecture and
  capabilities.
- Domain knowledge is bounded, pack-owned orientation material such as repo
    atlases or operator guidance that should enter the host only through a
    generic provider seam.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import httpx
from openai import AsyncAzureOpenAI

from kernel.io.protocols import DataSourceHooks
from kernel.models import DomainKnowledgeArtifact

ArtifactT = TypeVar("ArtifactT", bound=DomainKnowledgeArtifact)


class DomainKnowledgeProvider(ABC, Generic[ArtifactT]):
    """Base generic contract for one domain-knowledge provider."""

    provider_id: str
    display_name: str

    def missing_configuration(self) -> list[str]:
        """Return missing settings that prevent the provider from running."""

        return []

    def extract_status(self, artifact: ArtifactT) -> str:
        """Extract normalized status from one provider artifact."""

        status = getattr(artifact, "status", "ok")
        return status if isinstance(status, str) and status else "ok"

    @abstractmethod
    async def collect_artifact(
        self,
        http_client: httpx.AsyncClient,
        openai_client: AsyncAzureOpenAI | None,
        *,
        hooks: DataSourceHooks,
    ) -> ArtifactT:
        """Collect one domain-knowledge artifact."""

    @abstractmethod
    def build_unconfigured_artifact(self) -> ArtifactT:
        """Build one artifact when the provider is disabled or unconfigured."""

    @abstractmethod
    def build_error_artifact(self, error: Exception) -> ArtifactT:
        """Build one artifact when collection fails."""

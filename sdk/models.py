"""SDK-owned domain models for MCP servers and extensions.

These models are MCP-domain contracts — they live in lisa-sdk so both the
kernel (MCP adapters) and MCP servers can import them without circular deps.

Web search models (``WebSearchResponse``/``WebSearchResult``) have been moved
to ``sdk.packs.search.models`` — the search pack owns its own response models.
"""

from __future__ import annotations

__all__: list[str] = []

"""Default ``ModelPolicy`` backed by a ``kernel.model_manifest`` manifest.

Replaces ``app.model_policy`` for public hosts. Alias resolution and
availability checks are answered from the loaded manifest; hosts that need
runtime availability probing (provider health, deployment status) override
this seam.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kernel.model_manifest import ModelManifest, load_model_manifest


class ManifestModelPolicy:
    """Resolve model aliases and availability from a loaded model manifest."""

    def __init__(
        self, manifest: ModelManifest | None = None, *, manifest_path: Path | None = None
    ) -> None:
        self._manifest = manifest or (load_model_manifest(manifest_path) if manifest_path else None)

    def _resolve_entry(self, reference: str) -> Any | None:
        if self._manifest is None:
            return None
        return self._manifest.find_model(reference)

    def resolve_alias(self, alias: str) -> str:
        """Return the canonical ``model_id`` for one alias, or the alias itself."""

        entry = self._resolve_entry(alias)
        return entry.model_id if entry is not None else alias

    def is_available(self, model_id: str) -> bool:
        """Return True when the model_id is present in the manifest.

        Without a provider health probe, presence is the best availability
        signal a public host can give. Production hosts override with real
        provider probing.
        """

        return self._resolve_entry(model_id) is not None

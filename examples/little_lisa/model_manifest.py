from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from kernel.model_manifest import ModelManifest
from kernel.model_manifest import load_model_manifest as load_shared_model_manifest

logger = logging.getLogger(__name__)

SimpleModelDesignation = Literal["front-door", "default", "deep", "reasoning"]
_SUPPORTED_DESIGNATIONS: set[str] = {"front-door", "default", "deep", "reasoning"}


def default_manifest_path() -> Path:
    """Return the colocated default manifest path for Little Lisa."""

    return Path(__file__).resolve().with_name("models.yaml")


def load_model_manifest(path: Path | None = None) -> ModelManifest:
    """Load and validate a Little Lisa model manifest via the shared schema."""

    resolved = path or default_manifest_path()
    logger.debug("loading model manifest from %s", resolved)
    manifest = load_shared_model_manifest(
        resolved,
        require_mode_models=True,
        supported_designations=_SUPPORTED_DESIGNATIONS,
    )
    logger.info("loaded model manifest: %d model(s)", len(manifest.models))
    return manifest

from __future__ import annotations

from pathlib import Path
from typing import Literal

from kernel.model_manifest import ModelManifest
from kernel.model_manifest import load_model_manifest as load_shared_model_manifest

SimpleModelDesignation = Literal["front-door", "default", "deep", "reasoning"]
_SUPPORTED_DESIGNATIONS: set[str] = {"front-door", "default", "deep", "reasoning"}


def default_manifest_path() -> Path:
    """Return the colocated default manifest path for Little Lisa."""

    return Path(__file__).resolve().with_name("models.yaml")


def load_model_manifest(path: Path | None = None) -> ModelManifest:
    """Load and validate a Little Lisa model manifest via the shared schema."""

    return load_shared_model_manifest(
        path or default_manifest_path(),
        require_mode_models=True,
        supported_designations=_SUPPORTED_DESIGNATIONS,
    )

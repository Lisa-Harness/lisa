from __future__ import annotations

import logging
from pathlib import Path

from kernel.model_manifest import ModelManifest
from kernel.model_manifest import load_model_manifest as load_shared_model_manifest

logger = logging.getLogger(__name__)


def default_manifest_path() -> Path:
    """Return the default manifest path rooted under the runtime data-dir."""

    from hosts.core.config import default_data_dir  # noqa: PLC0415

    return (default_data_dir() / "models.yaml").resolve()


def load_model_manifest(path: Path | None = None) -> ModelManifest:
    """Load and validate a host model manifest via the shared schema."""

    resolved = path or default_manifest_path()
    logger.debug("loading model manifest from %s", resolved)
    manifest = load_shared_model_manifest(resolved, require_mode_models=True)
    logger.info("loaded model manifest: %d model(s)", len(manifest.models))
    return manifest

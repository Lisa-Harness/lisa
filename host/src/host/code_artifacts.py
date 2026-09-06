"""Atomic file-backed storage for pack-neutral WorkProducts.

The typed contract (``WorkProduct`` / ``WorkProductStore`` Protocol) lives in
``sdk.workproducts``; the host only owns this concrete file store and injects
it into the opaque ``code_change_artifact_store`` execution-context field."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from sdk.workproducts import WorkProduct

__all__ = ["FileCodeChangeArtifactStore"]


class FileCodeChangeArtifactStore:
    """Persist WorkProducts under a host-controlled data directory."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = asyncio.Lock()

    async def create(self, artifact: WorkProduct) -> WorkProduct:
        async with self._lock:
            artifacts = self._load()
            if artifact.artifact_id in artifacts:
                raise ValueError(f"Artifact '{artifact.artifact_id}' already exists.")
            artifact.updated_at = datetime.now(UTC)
            artifacts[artifact.artifact_id] = artifact
            self._store(artifacts)
            return artifact

    async def get(self, artifact_id: str) -> WorkProduct | None:
        async with self._lock:
            return self._load().get(artifact_id)

    async def save(self, artifact: WorkProduct) -> WorkProduct:
        async with self._lock:
            artifacts = self._load()
            if artifact.artifact_id not in artifacts:
                raise KeyError(f"Unknown artifact '{artifact.artifact_id}'.")
            artifact.updated_at = datetime.now(UTC)
            artifacts[artifact.artifact_id] = artifact
            self._store(artifacts)
            return artifact

    async def delete(self, artifact_id: str) -> bool:
        async with self._lock:
            artifacts = self._load()
            if artifact_id not in artifacts:
                return False
            del artifacts[artifact_id]
            self._store(artifacts)
            return True

    def _load(self) -> dict[str, WorkProduct]:
        if not self._path.is_file():
            return {}
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(raw, dict):
            return {}
        artifacts: dict[str, WorkProduct] = {}
        for artifact_id, value in raw.items():
            try:
                artifacts[str(artifact_id)] = WorkProduct.model_validate(value)
            except Exception:  # pylint: disable=broad-except
                continue
        return artifacts

    def _store(self, artifacts: dict[str, WorkProduct]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            artifact_id: artifact.model_dump(mode="json")
            for artifact_id, artifact in artifacts.items()
        }
        temporary_path = self._path.with_suffix(self._path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary_path, self._path)

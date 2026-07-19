"""Persistent Lisa-to-Letta identity mapping.

Maps a Lisa identity tuple ``(workspace_id, user_slug, surface_id,
conversation_id)`` to a Letta ``agent_id`` (plus optional core block ids,
source ids, and sync metadata) so the adapter server and operations can
resolve the right Letta agent without re-creating one each turn.

The map is persisted as a JSON sidecar under the owning process's data
directory (``<data_dir>/letta/identity-map.json``). Reads tolerate a
missing file as an empty map. Writes are atomic via a temp-file rename.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["IdentityEntry", "IdentityMap", "identity_key"]

_IDENTITY_MAP_SUBDIR = "letta"
_IDENTITY_MAP_FILE = "identity-map.json"


def identity_key(
    *,
    workspace_id: str,
    user_slug: str,
    surface_id: str = "",
    conversation_id: str = "",
) -> str:
    """Build a stable composite key for one Lisa identity.

    The key is filesystem- and JSON-safe so it can be used as a dict key
    in the sidecar without escaping concerns.
    """
    return "|".join(
        (
            workspace_id or "default",
            user_slug or "default",
            surface_id or "",
            conversation_id or "",
        )
    )


@dataclass(slots=True)
class IdentityEntry:
    """One resolved Letta binding for one Lisa identity."""

    agent_id: str
    agent_name: str = ""
    core_block_ids: dict[str, str] = field(default_factory=dict)
    source_ids: list[str] = field(default_factory=list)
    last_sync_turn: int = 0
    last_sync_at: str = ""
    last_compact_turn: int = 0
    last_compact_at: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class IdentityMap:
    """Load/save the Lisa↔Letta identity sidecar.

    The map is deliberately a single JSON file under the owning data dir.
    It is NOT a kernel seam and holds no host runtime state.
    """

    def __init__(self, data_dir: Path) -> None:
        self._path = Path(data_dir) / _IDENTITY_MAP_SUBDIR / _IDENTITY_MAP_FILE

    @property
    def path(self) -> Path:
        return self._path

    def load(self) -> dict[str, dict[str, Any]]:
        """Return the raw mapping ``{identity_key: entry_dict}``.

        Tolerates a missing file (returns ``{}``) and a corrupt file
        (logs a warning, returns ``{}``).
        """
        if not self._path.exists():
            return {}
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("letta: identity map %s unreadable: %s", self._path, exc)
            return {}
        if not isinstance(raw, dict):
            return {}
        return {
            str(k): (v if isinstance(v, dict) else {})
            for k, v in raw.items()
        }

    def get(self, key: str) -> IdentityEntry | None:
        """Return one entry by composite key, or ``None``."""
        raw = self.load().get(key)
        if raw is None:
            return None
        try:
            return IdentityEntry(**raw)
        except TypeError:
            return None

    def upsert(self, key: str, entry: IdentityEntry) -> None:
        """Insert or replace one entry and persist atomically."""
        data = self.load()
        data[key] = asdict(entry)
        self._write(data)

    def update(self, key: str, **fields: Any) -> IdentityEntry | None:
        """Patch one entry's fields in place; returns the new entry or None."""
        current = self.get(key)
        if current is None:
            return None
        for name, value in fields.items():
            if hasattr(current, name):
                setattr(current, name, value)
        self.upsert(key, current)
        return current

    def all_entries(self) -> dict[str, IdentityEntry]:
        """Return all parsed entries keyed by composite identity key."""
        out: dict[str, IdentityEntry] = {}
        for key, raw in self.load().items():
            try:
                out[key] = IdentityEntry(**raw)
            except TypeError:
                continue
        return out

    def _write(self, data: dict[str, dict[str, Any]]) -> None:
        """Atomic write: temp file in the same dir, then ``os.replace``."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(
            prefix=_IDENTITY_MAP_FILE + ".",
            suffix=".tmp",
            dir=str(self._path.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(data, indent=2, sort_keys=True))
            os.replace(tmp, self._path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
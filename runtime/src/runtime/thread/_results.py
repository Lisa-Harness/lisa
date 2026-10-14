"""Durable thread-owned result objects for provider-only live/replay previews."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from sdk.thread import result_object_path

_COMPARE_BLOCK_BYTES = 65536


def _matches_payload(path: Path, payload: bytes) -> bool:
    """Compare an existing object to ``payload`` in blocks, without a second full copy."""
    if path.stat().st_size != len(payload):
        return False
    view = memoryview(payload)
    with path.open("rb") as handle:
        for start in range(0, len(payload), _COMPARE_BLOCK_BYTES):
            block = view[start : start + _COMPARE_BLOCK_BYTES]
            if handle.read(len(block)) != block:
                return False
    return True


def _persist_result(base: Path, result: dict[str, Any]) -> str:
    payload = json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ref = "r" + hashlib.sha256(payload).hexdigest()
    path = result_object_path(base, ref)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.exists():
        if not _matches_payload(path, payload):
            raise ValueError("Existing result object failed integrity verification")
        return ref
    with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary.write(payload)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    if os.name == "nt":
        # POSIX NamedTemporaryFile is owner-only by construction; on Windows
        # the temp file inherits the umask (typically 0666), so tighten it
        # explicitly to keep persisted (redacted) tool outputs owner-only.
        os.chmod(temporary_path, 0o600)
    try:
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return ref

"""Durable thread-owned result objects for provider-only live/replay previews."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from sdk.thread import result_object_path


def _persist_result(base: Path, result: dict[str, Any]) -> str:
    payload = json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ref = "r" + hashlib.sha256(payload).hexdigest()
    path = result_object_path(base, ref)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError("Existing result object failed integrity verification")
        return ref
    with NamedTemporaryFile(dir=path.parent, delete=False) as temporary:
        temporary.write(payload)
        temporary.flush()
        os.fsync(temporary.fileno())
        temporary_path = Path(temporary.name)
    try:
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return ref

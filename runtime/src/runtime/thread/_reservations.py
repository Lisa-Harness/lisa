"""In-flight turn reservations: distinct provisional indices for overlapping turns.

A thread entry's turn number is its log position, assigned at commit under
the store lock — the log stays contiguous, so a turn's committed index is
only known when it finishes. Overlapping turns on one thread (several
``lisa`` processes against the same data dir) each need a provisional index
at turn start for the briefing ("Turn N"), the banner and telemetry. Reading
the entry count alone hands two overlapping turns the same number (the
2026-10-28 incident: both concurrent turns claimed 1095).

``<base>.reservations.json`` records each in-flight turn as
``{"token", "pid", "turn"}``. A new turn takes
``max(entry_count, max(live reserved turn) + 1)``; the reservation is
released at commit (atomically, under the same lock) and again when the turn
exits by any other path. Reservations whose owning process is gone are
pruned on every read. Every function here must be called with the store lock
(``<base>.lock``) held.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from sdk.filelock import atomic_write_text, is_process_alive


def reservations_path(base: Path) -> Path:
    """Return the reservation registry path for a thread store base path."""
    return Path(f"{base}.reservations.json")


def _load_live(path: Path) -> list[dict[str, Any]]:
    """Return the reservations whose owning process is still alive."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (OSError, ValueError):
        # A torn or corrupt registry carries no trustworthy claim; the
        # entry count alone still yields a valid (if possibly shared) index.
        return []
    if not isinstance(raw, list):
        return []
    live: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        pid, turn = item.get("pid"), item.get("turn")
        if not isinstance(pid, int) or not isinstance(turn, int) or not item.get("token"):
            continue
        if is_process_alive(pid):
            live.append(item)
    return live


def _save(path: Path, reservations: list[dict[str, Any]]) -> None:
    """Replace the registry atomically; remove it when no reservation is live."""
    if not reservations:
        path.unlink(missing_ok=True)
        return
    atomic_write_text(path, json.dumps(reservations))


def reserve(path: Path, *, entry_count: int, token: str) -> int:
    """Record a reservation for ``token`` and return its provisional turn index."""
    live = _load_live(path)
    turn = max([entry_count, *(int(item["turn"]) + 1 for item in live)])
    live.append({"token": token, "pid": os.getpid(), "turn": turn})
    _save(path, live)
    return turn


def release(path: Path, *, token: str) -> None:
    """Drop the reservation for ``token`` (pruning dead ones); a no-op when absent."""
    if not token or not path.exists():
        return
    _save(path, [item for item in _load_live(path) if item.get("token") != token])

"""Provider-neutral local coding workflow pack."""

from __future__ import annotations

from pathlib import Path


def get_content_root() -> Path:
    """Return this pack's catalog content root."""

    return Path(__file__).parent

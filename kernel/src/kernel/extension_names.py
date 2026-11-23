"""Normalize distribution names used to attribute extension entry points."""

from __future__ import annotations

import re


def normalize_pack_name(raw: str) -> str:
    """Normalize a pack distribution name for comparison."""
    value = raw.strip().lower()
    return re.sub(r"[-_.\s]+", "-", value).strip("-")

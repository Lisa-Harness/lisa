"""Normalize distribution names used to attribute extension entry points."""

from __future__ import annotations

import re


def normalize_pack_name(raw: str) -> str:
    """Normalize a pack distribution name for comparison."""
    value = raw.strip().lower().replace("_", "-")
    return re.sub(r"[^a-z0-9-]+", "-", value).strip("-")

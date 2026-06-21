"""Extras content pack for Lisa.

Contains optional general-purpose modes and operations that are not part of the
core launch-stable baseline pack.
"""

from __future__ import annotations

from pathlib import Path


def get_content_root() -> Path:
    """Return the content root directory for this pack."""
    return Path(__file__).parent

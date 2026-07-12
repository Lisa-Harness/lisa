"""Core baseline pack for Lisa.

Contains the shipped baseline modes, authored markdown assets, and reusable operations
that the host runtime loads through pack discovery.
"""

from __future__ import annotations

from pathlib import Path


def get_content_root() -> Path:
    """Return the content root directory for this pack."""
    return Path(__file__).parent

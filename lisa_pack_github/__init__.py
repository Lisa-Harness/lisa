"""GitHub pack — repository and CI/CD health signal connector plus status mode."""

from __future__ import annotations

from pathlib import Path


def get_content_root() -> Path:
    """Return the content root directory for this pack."""
    return Path(__file__).parent

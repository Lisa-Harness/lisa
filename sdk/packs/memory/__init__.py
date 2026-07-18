"""Memory pack for Lisa.

Adds memory-backed conversational behavior and explicit learn-mode workflows
without requiring changes to core-pack content.
"""

from __future__ import annotations

from pathlib import Path


def get_content_root() -> Path:
    """Return the content root directory for this pack."""
    return Path(__file__).parent

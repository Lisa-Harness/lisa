"""Search pack for Lisa.

A self-contained, externalizable pack providing the web search capability:
modes (search, thinkabout), operations (web_search research loop,
discussion_briefing), and the ``web_search`` internal tool. The pack is
discovered via the ``lisa.extensions.content_roots`` entry point.
"""

from __future__ import annotations

from pathlib import Path


def get_content_root() -> Path:
    """Return the content root directory for this pack."""
    return Path(__file__).parent
"""Core baseline pack for Lisa.

Contains the shipped baseline modes, authored markdown assets, and reusable operations
that the host runtime loads through pack discovery.

It also owns the two ambient web capabilities every mode inherits: the
``fetch_page`` tool and the ``web_search`` *contract* stub. A provider pack
(Brave, Tavily, Serper) supplies the live ``web_search`` implementation and
the session plan selects it with ``providers.tools.web_search``; the stub is
what keeps ``tool_ids: [web_search]`` resolvable in a session with no
provider installed.
"""

from __future__ import annotations

from pathlib import Path


def get_content_root() -> Path:
    """Return the content root directory for this pack."""
    return Path(__file__).parent

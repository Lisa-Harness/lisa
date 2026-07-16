"""Markdown pack — memory-grounded markdown document forging.

Provides the ``markdown`` action mode which combines user input with recalled
memory context, calls an LLM to synthesize a markdown document, validates the
document's structure through the ``markdown_generate`` tool (backed by
``markdown-it-py``), and persists the result via the ``lisa-storage`` MCP
server's ``write_file`` tool.
"""

from __future__ import annotations

from pathlib import Path


def get_content_root() -> Path:
    """Return the content root directory for this pack."""
    return Path(__file__).parent

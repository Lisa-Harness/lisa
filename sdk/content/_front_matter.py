"""Shared YAML front-matter parsing for content-pack assets.

Used by the ``lisa-content`` MCP server to parse mode/stage/operation
markdown files without importing kernel FSM internals. The kernel keeps
its own copy of this pure string function; the server has this one so it
can parse front-matter before returning structured JSON to the kernel.
"""

from __future__ import annotations

from typing import Any

import yaml

__all__ = ["FrontMatterError", "parse_front_matter"]

_FRONT_MATTER_DELIMITER = "---"


class FrontMatterError(ValueError):
    """Raised when a front-matter block is malformed."""


def parse_front_matter(text: str) -> tuple[dict[str, Any], str]:
    """Parse a YAML front-matter block from the top of a markdown string.

    Expects the file to start with a ``---`` delimiter line, followed by YAML
    content, followed by another ``---`` delimiter line. The body is
    everything after the closing delimiter, stripped of leading and trailing
    whitespace.

    Always uses ``yaml.safe_load()`` — never ``yaml.load()`` — to avoid
    arbitrary code execution vulnerabilities in untrusted files.

    Args:
        text: Full file contents including the front-matter block.

    Returns:
        A tuple of ``(front_matter_dict, body_text)``. If no valid
        front-matter block is found, returns ``({}, text)`` so callers
        degrade gracefully.

    Raises:
        FrontMatterError: If the YAML in the front-matter is invalid.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != _FRONT_MATTER_DELIMITER:
        # No front-matter block: treat entire content as body.
        return {}, text

    # Find the closing --- line (must be after at least one content line).
    try:
        close_idx = next(
            i
            for i, line in enumerate(lines[1:], start=1)
            if line.strip() == _FRONT_MATTER_DELIMITER
        )
    except StopIteration:
        # Opening --- found but no closing ---: treat entire content as body.
        return {}, text

    yaml_text = "\n".join(lines[1:close_idx])
    body_text = "\n".join(lines[close_idx + 1 :]).strip()

    try:
        front_matter = yaml.safe_load(yaml_text) or {}
    except yaml.YAMLError as exc:
        raise FrontMatterError(f"Invalid YAML front-matter: {exc}") from exc

    if not isinstance(front_matter, dict):
        raise FrontMatterError(
            f"Front-matter must be a YAML mapping, got {type(front_matter).__name__}."
        )

    return front_matter, body_text

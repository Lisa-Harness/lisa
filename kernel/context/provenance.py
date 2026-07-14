"""Provenance-tagged context envelope.

The canonical runtime representation of injected context uses XML tags so the
model can distinguish *why* a block is present (authoritative runtime state
vs. recalled memory vs. speculative reasoning) without relying on prose
formatting alone. Markdown/text authored with fenced provenance blocks is
normalized into the same tagged structure before emission.

Tag grammar
-----------
Every section is wrapped in an XML element whose local name is the
provenance kind::

    <authoritative source="runtime">
    ...
    </authoritative>

Supported provenance kinds (see :data:`Provenance`):

- ``authoritative``       — verified runtime state the model can rely on.
- ``inferred``            — previously inferred by the assistant; not verified.
- ``speculative``         — explicitly speculative architectural claims.
- ``user_asserted``       — the user stated this; treat as their claim, not fact.
- ``system_observed``     — ambient machine facts the host measured.
- ``recalled_memory``     — fetched from a memory backend (mem0, lisa_memory, …).
- `diagnostic`            — introspection / debugging context; not authoritative.

Markdown authoring
-------------------
Authors may write fenced code blocks whose info string is a provenance kind.
The parser lifts the fence body into a tagged section so authored text and
runtime-generated state share one representation::

    ```authoritative
    mode_id: default
    tools_authoritative: yes
    ```

    ```recalled_memory
    User prefers concise answers.
    ```

Inline single-line tags are also supported for short assertions::

    ```user_asserted
    I want strict security.
    ```

Any fenced block whose info string is NOT a known provenance kind is left
untouched (ordinary code fence).
"""

from __future__ import annotations

import re
from typing import Final

Provenance = str
"""Provenance kind string. See :data:`PROVENANCE_KINDS` for the closed set."""

PROVENANCE_KINDS: Final[tuple[str, ...]] = (
    "authoritative",
    "inferred",
    "speculative",
    "user_asserted",
    "system_observed",
    "recalled_memory",
    "diagnostic",
)
"""Closed set of provenance kinds. Order is stable for rendering."""

_PROVENANCE_SET: Final[frozenset[str]] = frozenset(PROVENANCE_KINDS)

# --- XML emission ----------------------------------------------------------

_TAG_PATTERN = re.compile(r"</?[A-Za-z_][A-Za-z0-9_\-]*\b")


def wrap_section(
    provenance: str,
    content: str,
    *,
    source: str = "",
) -> str:
    """Wrap *content* in an XML tag named after *provenance*.

    ``source`` is rendered as an XML attribute when non-empty. The content is
    stripped of leading/trailing whitespace; empty content returns ``""``
    (no empty tags) so callers can blindly append.

    Raises :class:`ValueError` when *provenance* is not a known kind.
    """
    if provenance not in _PROVENANCE_SET:
        raise ValueError(
            f"Unknown provenance kind {provenance!r}; "
            f"expected one of {', '.join(PROVENANCE_KINDS)}"
        )
    body = content.strip()
    if not body:
        return ""
    attr = f' source="{_xml_escape_attr(source)}"' if source else ""
    return f"<{provenance}{attr}>\n{body}\n</{provenance}>"


def wrap_sections(sections: list[tuple[str, str, str]]) -> str:
    """Wrap and join multiple ``(provenance, source, content)`` tuples.

    Empty bodies are skipped. Returns ``""`` when nothing survives.
    """
    parts: list[str] = []
    for provenance, source, content in sections:
        rendered = wrap_section(provenance, content, source=source)
        if rendered:
            parts.append(rendered)
    return "\n\n".join(parts)


def _xml_escape_attr(value: str) -> str:
    """Escape a value for use inside an XML attribute."""
    return (
        value.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


# --- Markdown → tagged-section parsing -------------------------------------

# Fenced block: ```provenance\nbody\n```  (3+ backticks, info string = kind)
_FENCE_RE = re.compile(
    r"```(" + "|".join(PROVENANCE_KINDS) + r")\s*\n(.*?)\n```",
    re.DOTALL,
)


def parse_markdown_tagged(text: str) -> list[tuple[str, str, str]]:
    """Lift provenance-fenced blocks out of *text* into structured sections.

    Returns a list of ``(provenance, source, body)`` tuples in document order.
    Text *between* tagged fences is returned as ``("authoritative", "", chunk)``
    when non-empty — author prose defaults to authoritative unless tagged
    otherwise, matching the existing runtime contract.

    Ordinary code fences (info string not a provenance kind) are left inline
    inside the surrounding authoritative chunk.
    """
    sections: list[tuple[str, str, str]] = []
    cursor = 0

    for match in _FENCE_RE.finditer(text):
        # Emit the prose between the last fence and this one as authoritative.
        if match.start() > cursor:
            prose = text[cursor: match.start()].strip()
            if prose:
                sections.append(("authoritative", "", prose))
        kind = match.group(1)
        body = match.group(2).strip()
        if body:
            sections.append((kind, "", body))
        cursor = match.end()

    # Trailing prose
    if cursor < len(text):
        tail = text[cursor:].strip()
        if tail:
            sections.append(("authoritative", "", tail))

    return sections


def normalize_markdown_to_tagged(text: str) -> str:
    """Convert markdown-authored text into provenance-tagged XML sections.

    Fenced provenance blocks become XML tags; surrounding prose becomes
    ``<authoritative>`` blocks. Non-provenance fences (e.g. ```python) are
    preserved verbatim inside the surrounding authoritative block.
    """
    sections = parse_markdown_tagged(text)
    return wrap_sections(sections)


def is_provenance_kind(value: str) -> bool:
    """Return True when *value* is a recognized provenance kind."""
    return value in _PROVENANCE_SET
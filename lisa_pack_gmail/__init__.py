"""Gmail pack — draft email creation via the Gmail API.

A self-contained, externalizable Lisa content pack providing one capability
(tool) and one action mode for composing and saving Gmail drafts:

- Mode: ``gmail`` (action) — draft an email from a natural-language request
  and save it as a Gmail draft via the Gmail REST API.
- Operation: ``gmail_draft`` — orchestrates model-driven drafting plus the
  ``gmail_create_draft`` tool call.
- Tool: ``gmail_create_draft`` — calls the Gmail REST API ``drafts.create``
  endpoint with a base64url-encoded RFC 2822 message.

The pack is discovered via the ``lisa.extensions.content_roots`` entry point
and reads configuration from environment variables only (no ``sdk`` or
``kernel`` config dependency), so it stays self-contained for
externalization.
"""

from __future__ import annotations

from pathlib import Path


def get_content_root() -> Path:
    """Return the content root directory for this pack."""
    return Path(__file__).parent
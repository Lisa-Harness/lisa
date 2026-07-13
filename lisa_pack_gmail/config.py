"""Environment-backed settings for the Gmail pack.

The pack reads configuration from environment variables only and never
imports ``sdk`` or ``kernel`` config, so it stays self-contained for
externalization. Secrets are kept out of any cached state; only boolean
configured status and non-secret API base URLs are exposed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

__all__ = ["GmailSettings", "gmail_settings"]


@dataclass(frozen=True)
class GmailSettings:
    """Typed, read-only settings used by the Gmail pack."""

    token: str
    api_base: str

    @property
    def is_configured(self) -> bool:
        """Return ``True`` when a Gmail OAuth2 token is present."""
        return bool(self.token)


def gmail_settings() -> GmailSettings:
    """Build a :class:`GmailSettings` snapshot from the environment.

    Environment variables:

    - ``GMAIL_TOKEN`` — OAuth2 access token with ``gmail.compose`` scope.
      Required for the pack to function.
    - ``GMAIL_API_BASE`` — Gmail REST API base URL. Optional; defaults to
      ``https://gmail.googleapis.com/gmail/v1``.
    """

    return GmailSettings(
        token=os.environ.get("GMAIL_TOKEN", ""),
        api_base=os.environ.get(
            "GMAIL_API_BASE",
            "https://gmail.googleapis.com/gmail/v1",
        ),
    )
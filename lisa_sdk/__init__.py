"""Lisa SDK — author-facing helpers for framework extensions and packs.

This package provides optional extension content for Lisa framework hosts:
- Product- and domain-specific extension surfaces that build on the core pack
- Extension operations used by those surfaces
- Signal connector and data-source base contracts

Design contract:
- A host app remains a useful standalone foundation (chat, memory, search, markdown).
- The SDK layer adds capabilities without replacing host-owned mode manifests.
- SDK modules may depend on kernel contracts but must not import app modules.
- Extensions declare themselves via ``lisa.extensions.*`` entry-point groups.
  Host apps discover and wire them at startup with zero code changes here.
"""

from __future__ import annotations

from pathlib import Path


def get_content_root() -> Path:
    """Return the filesystem root of this SDK package.

    Registered as the ``lisa_sdk`` entry-point in the
    ``lisa.extensions.content_roots`` group so the app layer can discover
    mode and operation YAML definitions without hard-coded paths.
    """
    return Path(__file__).parent

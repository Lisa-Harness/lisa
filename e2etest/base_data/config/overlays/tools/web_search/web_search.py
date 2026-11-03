"""Overlay anchor re-exporting the core contract stub.

The overlay exists only to flag ``web_search`` as ambient
(``default_available: true``) for the bundled base-data dir without
editing the core pack. The ``.tool.yml`` pins ``module_path`` to the core
stub module so import and dispatch are the production code; this file is
the scannable anchor the tool-catalog walker requires next to the
``.tool.yml``.
"""

from __future__ import annotations

from sdk.packs.core.tools.web_search.web_search import (
    Parameters,
    execute,
    is_configured,
    search,
)

__all__ = ["Parameters", "execute", "is_configured", "search"]

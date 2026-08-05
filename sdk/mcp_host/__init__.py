"""Reusable helpers for hosts that spawn first-party Lisa MCP servers.

This package owns the *pure config construction* for the framework's
first-party MCP servers (``lisa-content``, ``lisa-catalog``): turning
content roots, disabled-pack sets, and provider selections into
:class:`kernel.tools.mcp_client.MCPClientConfig` entries without any I/O
or runtime state.

It does NOT own the MCP pool lifecycle (start / deferred-start / close).
Pool lifecycle is host-owned because the anyio cancel scopes entered
during ``MCPToolPool.start()`` bind to the host's event loop and task —
they cannot be entered in one loop and exited in another (see
``debugging.md``). External hosts bring their own lifecycle wiring and
call these builders to produce the configs they hand to
``MCPToolPool``.

Public surface::

    from sdk.mcp_host import build_content_mcp_config, build_catalog_mcp_config

    content_cfg = build_content_mcp_config(
        data_dir=Path("~/.lisa"),
        content_roots=[Path("../lisa-packs/search/lisa_packs/search")],
        disabled_packs=frozenset(),
        provider_selections=selections,
        packs_venv_site_packages=Path("~/.lisa/packs-venv/lib/python3.11/site-packages"),
    )
    catalog_cfg = build_catalog_mcp_config(content_config=content_cfg, data_dir=Path("~/.lisa"))
"""

from __future__ import annotations

from sdk.mcp_host.content_servers import (
    build_catalog_mcp_config,
    build_content_mcp_config,
    session_provider_selection_payload,
)

__all__ = [
    "build_catalog_mcp_config",
    "build_content_mcp_config",
    "session_provider_selection_payload",
]

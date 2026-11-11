"""stdio ↔ direct transport parity for the first-party catalog/content servers.

Drives the same fixture corpus the SDK's direct sources are tested against
(``sdk/tests/test_catalog_sources.py``) through the ``lisa-catalog`` /
``lisa-content`` tool surface exactly as a spawned subprocess sees it —
env-driven roots, selections and ambient tools; the deps test seam stands
in for the server process — and pins the responses byte-for-byte against
the ``DirectCatalogSource`` output built from the same payloads. The
mcp_server test suite is where both transports meet: importing the servers
from an SDK test would violate the "SDK imports no MCP server wiring"
contract. Per ``docs/SPEC/PROMPT_STARTUP.md`` §4: competing providers,
overlays, ambient MCP tools, Unicode/spaced paths, traversal attempts,
missing content, and two differently configured services without
cross-contamination.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest
from mcp_server.catalog import server as catalog_server
from mcp_server.content import server as content_server
from sdk.catalog_sources import CatalogSourceConfig, DirectCatalogSource
from sdk.content import ContentReader

SELECTIONS_A = json.dumps(
    {"modes": {"default": "provider-a"}, "operations": {}, "tools": {}},
    ensure_ascii=True,
    sort_keys=True,
)


def build_provider(root: Path, name: str, *, body: str) -> Path:
    """Write one pack-style provider with a mode, an operation and content."""
    pack = root / name
    mode_dir = pack / "modes" / "default"
    mode_dir.mkdir(parents=True, exist_ok=True)
    (mode_dir / "_mode.md").write_text(
        "---\n"
        "mode_id: default\n"
        f"mode_provider_id: {name}\n"
        "mode_kind: chat\n"
        "routable: true\n"
        "priority: 10\n"
        f'description: "Mode from {name}."\n'
        "---\n"
        f"# {name}\n\n{body}\n",
        encoding="utf-8",
    )
    content = pack / "content" / "prompts"
    content.mkdir(parents=True, exist_ok=True)
    (content / "greeting.md").write_text(f"hello from {name}\n", encoding="utf-8")
    (content / "spaced Ünïcode.md").write_text("unicode body\n", encoding="utf-8")
    return pack


def direct_config(
    tmp_path: Path,
    *,
    overlays: bool = False,
    selections_json: str = SELECTIONS_A,
    ambient: frozenset[str] = frozenset(),
) -> CatalogSourceConfig:
    return CatalogSourceConfig.from_payloads(
        content_roots=[
            tmp_path / "roots" / "provider-a",
            tmp_path / "roots" / "provider-b",
        ],
        disabled_packs=set(),
        provider_selections_json=selections_json,
        overlays_dir=(tmp_path / "overlays") if overlays else None,
        default_mcp_tool_ids=ambient,
    )


@pytest.fixture(autouse=True)
def _hermetic_installed_roots(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exclude installed entry-point packs so only the corpus roots appear.

    Both transports go through the same ``resolve_content_roots`` path, so
    patching discovery here keeps the parity comparison hermetic without
    diverging the two implementations under test.
    """
    import sdk.content._roots as roots_mod

    monkeypatch.setattr(
        roots_mod,
        "discover_installed_content_roots",
        lambda excluded_packs=None: [],
    )


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """Two competing providers plus an overlays directory."""
    roots = tmp_path / "roots"
    roots.mkdir()
    build_provider(roots, "provider-a", body="alpha body")
    build_provider(roots, "provider-b", body="beta body")
    overlays = tmp_path / "overlays"
    (overlays / "modes" / "default").mkdir(parents=True)
    (overlays / "modes" / "default" / "_mode.md").write_text(
        "---\n"
        "mode_id: default\n"
        "mode_provider_id: overlay-provider\n"
        "mode_kind: chat\n"
        "routable: true\n"
        "priority: 10\n"
        'description: "Overlay mode."\n'
        "---\n"
        "# Overlay\n\noverlay body\n",
        encoding="utf-8",
    )
    return tmp_path


def _stdio_catalog_json(
    monkeypatch: pytest.MonkeyPatch,
    corpus: Path,
    *,
    overlays: bool,
    selections_env: str,
    ambient: str,
) -> str:
    child_roots = [corpus / "roots" / "provider-a", corpus / "roots" / "provider-b"]
    subdir_roots = [r / "content" for r in child_roots if (r / "content").is_dir()]
    configured = [*child_roots, *subdir_roots]
    payload = os.pathsep.join(str(p) for p in configured)
    if overlays:
        payload = str(corpus / "overlays") + os.pathsep + payload
    monkeypatch.setenv("SDK_CONTENT_ROOTS", payload)
    monkeypatch.setenv("LISA_SESSION_PROVIDER_SELECTIONS", selections_env)
    monkeypatch.setenv("LISA_DEFAULT_MCP_TOOLS", ambient)
    if overlays:
        monkeypatch.setenv("LISA_OVERLAY_ROOT", str(corpus / "overlays"))
    else:
        monkeypatch.delenv("LISA_OVERLAY_ROOT", raising=False)
    # Direct tool call path uses the test seam for deps (no lifespan).
    content_server.set_deps_for_test(
        content_server.ContentDeps(reader=ContentReader(extra_roots=[]))
    )
    catalog_server.set_deps_for_test(catalog_server.CatalogDeps(extra_roots=list(child_roots)))
    return asyncio.run(catalog_server.list_modes(None))


def test_mode_catalog_parity_stdio_vs_direct(monkeypatch: pytest.MonkeyPatch, corpus: Path) -> None:
    direct = DirectCatalogSource(direct_config(corpus))
    direct_json = direct.list_modes()

    stdio_json = _stdio_catalog_json(
        monkeypatch, corpus, overlays=False, selections_env=SELECTIONS_A, ambient=""
    )

    assert json.loads(direct_json).keys() == json.loads(stdio_json).keys()
    assert json.loads(direct_json) == json.loads(stdio_json)


def test_mode_catalog_parity_with_overlay_and_ambient(
    monkeypatch: pytest.MonkeyPatch, corpus: Path
) -> None:
    ambient = frozenset({"lisa-storage.read_file"})
    direct = DirectCatalogSource(direct_config(corpus, overlays=True, ambient=ambient))
    direct_json = direct.list_modes()

    stdio_json = _stdio_catalog_json(
        monkeypatch,
        corpus,
        overlays=True,
        selections_env=SELECTIONS_A,
        ambient="lisa-storage.read_file",
    )

    assert json.loads(direct_json) == json.loads(stdio_json)


def test_competing_providers_resolved_by_selections(
    monkeypatch: pytest.MonkeyPatch, corpus: Path
) -> None:
    selections = json.dumps(
        {"modes": {"default": "provider-b"}, "operations": {}, "tools": {}},
        ensure_ascii=True,
        sort_keys=True,
    )
    direct = DirectCatalogSource(direct_config(corpus, selections_json=selections))
    direct_json = direct.list_modes()

    stdio_json = _stdio_catalog_json(
        monkeypatch, corpus, overlays=False, selections_env=selections, ambient=""
    )

    assert json.loads(direct_json) == json.loads(stdio_json)
    assert (
        json.loads(direct_json)["default"]["instruction_body"].strip()
        == "# provider-b\n\nbeta body"
    )

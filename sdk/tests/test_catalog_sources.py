"""Slice B: in-process catalog/content source contracts and stdio parity.

Drives the same fixture corpus through the ``lisa-catalog`` /
``lisa-content`` tool surface (env-driven, as the subprocesses run) and the
runtime-injected direct sources (explicit inputs), per
``docs/SPEC/PROMPT_STARTUP.md`` §4: competing providers, overlays, ambient
MCP tools, Unicode/spaced paths, traversal attempts, missing content,
hit-only read caching, explicit invalidation, and two differently
configured services without cross-contamination.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest
from sdk.catalog_sources import CatalogSourceConfig, DirectCatalogSource, DirectContentSource
from sdk.content import ContentReader

SELECTIONS_A = json.dumps(
    {"modes": {"default": "provider-a"}, "operations": {}, "tools": {}},
    ensure_ascii=True,
    sort_keys=True,
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


def build_provider(root: Path, name: str, *, body: str) -> Path:
    """Write one pack-style provider with a mode, an operation and content."""
    pack = root / name
    mode_dir = pack / "modes" / "default"
    mode_dir.mkdir(parents=True)
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
    content.mkdir(parents=True)
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


def test_content_source_reads_spaced_and_unicode_paths(corpus: Path) -> None:
    source = DirectContentSource(direct_config(corpus))

    assert source.read_authored_text("prompts/greeting.md") == "hello from provider-a\n"
    assert source.read_authored_text("prompts/spaced Ünïcode.md") == "unicode body\n"


def test_content_source_rejects_traversal_and_missing(corpus: Path) -> None:
    source = DirectContentSource(direct_config(corpus))

    assert source.read_authored_text("../../etc/passwd") == ""
    assert source.read_authored_text("prompts/missing.md") == ""


def test_content_source_invalidate_forces_reread(corpus: Path) -> None:
    source = DirectContentSource(direct_config(corpus))
    path = corpus / "roots" / "provider-a" / "content" / "prompts" / "greeting.md"
    assert source.read_authored_text("prompts/greeting.md") == "hello from provider-a\n"
    path.write_text("edited\n", encoding="utf-8")

    source.invalidate()

    assert source.read_authored_text("prompts/greeting.md") == "edited\n"


def test_content_source_two_configs_no_cross_contamination(corpus: Path) -> None:
    first = DirectContentSource(direct_config(corpus))
    second_roots = corpus / "other-roots"
    second_roots.mkdir()
    build_provider(second_roots, "provider-c", body="other")
    second = DirectContentSource(
        CatalogSourceConfig.from_payloads(
            content_roots=[second_roots / "provider-c"],
            disabled_packs=set(),
            provider_selections_json="",
            overlays_dir=None,
        )
    )

    assert first.read_authored_text("prompts/greeting.md") == "hello from provider-a\n"
    assert second.read_authored_text("prompts/greeting.md") == "hello from provider-c\n"


# ── stdio ↔ direct parity ───────────────────────────────────────────


def _stdio_catalog_json(
    monkeypatch: pytest.MonkeyPatch,
    corpus: Path,
    *,
    overlays: bool,
    selections_env: str,
    ambient: str,
) -> str:
    from mcp_server.catalog import server as catalog_server
    from mcp_server.content import server as content_server

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


def test_direct_catalog_source_invalidate_reflects_edit(corpus: Path) -> None:
    direct = DirectCatalogSource(direct_config(corpus))
    before = json.loads(direct.list_operations())
    op_dir = corpus / "roots" / "provider-a" / "operations"
    op_dir.mkdir(parents=True)
    (op_dir / "ping.operation.md").write_text(
        "---\n"
        "operation_id: ping\n"
        "operation_provider_id: provider-a\n"
        'description: "Ping."\n'
        "---\n"
        "Ping.\n",
        encoding="utf-8",
    )
    assert "ping" not in before

    direct.invalidate()

    assert "ping" in json.loads(direct.list_operations())


def test_provider_selection_cache_key_variants() -> None:
    from sdk.content._catalog import (
        _selections_cache_key,
        load_session_provider_selections,
    )

    assert _selections_cache_key(None) != _selections_cache_key(SELECTIONS_A)
    explicit = load_session_provider_selections(SELECTIONS_A)
    assert explicit["modes"] == {"default": "provider-a"}
    assert explicit["operations"] == {}
    assert explicit["tools"] == {}
    assert load_session_provider_selections("") == {
        "modes": {},
        "operations": {},
        "tools": {},
    }


def test_normalize_disabled_pack_names() -> None:
    from sdk.content._roots import normalize_disabled_pack_names

    assert normalize_disabled_pack_names(f"alpha{os.pathsep} beta") == {"alpha", "beta"}
    assert normalize_disabled_pack_names("") == set()


def test_direct_sources_are_synchronous(corpus: Path) -> None:
    source = DirectContentSource(direct_config(corpus))
    catalog = DirectCatalogSource(direct_config(corpus))

    assert isinstance(source.read_authored_text("prompts/greeting.md"), str)
    assert isinstance(asyncio.run(asyncio.to_thread(catalog.list_modes)), str)
    assert sys.modules.get("asyncio") is not None

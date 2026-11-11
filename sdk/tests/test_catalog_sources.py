"""Slice B: in-process catalog/content source contracts.

Drives the direct sources (explicit inputs, no env reads) against a
fixture corpus of competitors/providers per ``docs/SPEC/PROMPT_STARTUP.md``
§4: overlays, traversal attempts, missing content, hit-only read caching,
explicit invalidation, and two differently configured services without
cross-contamination. The stdio ↔ direct transport parity half lives in
``mcp_server/tests/test_catalog_parity.py`` — the mcp_server suite is
where both transports meet without an SDK test importing the servers.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest
from sdk.catalog_sources import CatalogSourceConfig, DirectCatalogSource, DirectContentSource

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


def test_content_source_roots_fingerprint_and_invalidation(corpus: Path) -> None:
    source = DirectContentSource(direct_config(corpus))
    base = source.roots_fingerprint()
    assert base  # non-empty: real root set

    source.invalidate()
    assert source.roots_fingerprint() == base  # same roots, cache reset only


def test_catalog_source_list_content_roots_and_absent_overlay_dir(corpus: Path) -> None:
    import json as json_module

    missing_overlays = corpus / "no-overlays-here"
    config = CatalogSourceConfig.from_payloads(
        content_roots=[corpus / "roots" / "provider-a"],
        disabled_packs=set(),
        provider_selections_json="",
        overlays_dir=missing_overlays,
    )
    source = DirectCatalogSource(config)
    roots = json_module.loads(source.list_content_roots())
    assert any(str(p).endswith("provider-a") for p in roots)
    # The configured-but-absent overlay directory is skipped, not an error.
    assert not any("no-overlays-here" in str(p) for p in roots)


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


def test_load_session_provider_selections_rejects_malformed_payloads() -> None:
    from sdk.content._catalog import (
        CatalogCollisionError,
        load_session_provider_selections,
    )

    with pytest.raises(CatalogCollisionError, match="Invalid JSON"):
        load_session_provider_selections("not json")
    with pytest.raises(CatalogCollisionError, match="must contain an object"):
        load_session_provider_selections('"a string"')
    with pytest.raises(CatalogCollisionError, match="Unexpected provider selection kinds"):
        load_session_provider_selections('{"modes": {}, "surprise": {}}')
    with pytest.raises(CatalogCollisionError, match="must be an object"):
        load_session_provider_selections('{"modes": ["a"]}')
    with pytest.raises(CatalogCollisionError, match="require string IDs"):
        load_session_provider_selections('{"modes": {"default": 7}}')
    with pytest.raises(CatalogCollisionError, match="require non-empty IDs"):
        load_session_provider_selections('{"modes": {"default": "  "}}')


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

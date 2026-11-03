"""Catalog JSON must have the object-of-objects shape its consumers require."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from packstest import catalog


@pytest.mark.parametrize("payload", [[], {"bad": []}, {"bad": None}])
def test_invalid_catalog_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: Any
) -> None:
    monkeypatch.setattr(catalog, "load_pack_content_root", lambda path: object())
    monkeypatch.setattr(catalog, "resolve_tool_roots", lambda roots: [tmp_path])
    monkeypatch.setattr(catalog, "build_tool_catalog_json", lambda roots: json.dumps(payload))
    with pytest.raises(ValueError, match="mapping tool IDs to objects"):
        catalog.build_tool_catalog(tmp_path, invalidate=False, ensure_importable=False)


def test_valid_catalog_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {"tool": {"description": "test"}}
    monkeypatch.setattr(catalog, "load_pack_content_root", lambda path: object())
    monkeypatch.setattr(catalog, "resolve_tool_roots", lambda roots: [tmp_path])
    monkeypatch.setattr(catalog, "build_tool_catalog_json", lambda roots: json.dumps(payload))
    assert (
        catalog.build_tool_catalog(tmp_path, invalidate=False, ensure_importable=False) == payload
    )


def test_resolve_namespace_child_prefers_slug_match(tmp_path: Path) -> None:
    """Multi-child namespaces resolve by distribution slug, not by accident."""
    children = [tmp_path / "other", tmp_path / "widgets"]
    assert catalog._resolve_namespace_child(children, "widgets") == tmp_path / "widgets"
    assert catalog._resolve_namespace_child(children, "unknown-slug") == tmp_path / "other"
    assert catalog._resolve_namespace_child([tmp_path / "solo"], "anything") == tmp_path / "solo"


def test_src_fallback_root_walks_nested_and_flat_layouts(tmp_path: Path) -> None:
    """The bounded src walk prefers nested lisa_packs/<slug>, then content dirs."""
    nested = tmp_path / "nested" / "src" / "lisa_packs" / "lisa_packs" / "widgets" / "tools"
    nested.mkdir(parents=True)
    assert catalog._src_fallback_root(tmp_path / "nested", "widgets") == nested.parent

    flat = tmp_path / "flat" / "src" / "toolkit" / "modes"
    flat.mkdir(parents=True)
    assert catalog._src_fallback_root(tmp_path / "flat", "toolkit") == flat.parent

    bare = tmp_path / "bare" / "src" / "empty-namespace"
    bare.mkdir(parents=True)
    (tmp_path / "bare" / "src" / "__pycache__").mkdir()
    assert catalog._src_fallback_root(tmp_path / "bare", "x") is None

    assert catalog._src_fallback_root(tmp_path / "absent", "x") is None


def test_read_content_root_entries_skips_undecodable_group(tmp_path: Path) -> None:
    """A non-dict entry-points group yields no content roots (no crash)."""
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "probe"\n\n[project.entry-points."lisa.extensions.content_roots"]\n',
        encoding="utf-8",
    )
    assert catalog._read_content_root_entries(tmp_path) == []


def test_read_content_root_entries_skips_invalid_group_type(tmp_path: Path) -> None:
    """A group value that is not a mapping yields no content roots quietly."""
    (tmp_path / "pyproject.toml").write_text(
        "[project]\n"
        'name = "probe"\n'
        'entry-points = { "lisa.extensions.content_roots" = "module:attr" }\n',
        encoding="utf-8",
    )
    assert catalog._read_content_root_entries(tmp_path) == []


def test_content_root_from_entry_point_skips_empty_module(tmp_path: Path) -> None:
    """An entry target without a module part is skipped, not resolved."""
    pack = tmp_path / "pack"
    pack.mkdir()
    (pack / "pyproject.toml").write_text(
        "[project]\n"
        'name = "lisa-pack-probe"\n\n'
        '[project.entry-points."lisa.extensions.content_roots"]\n'
        'probe = ":attr"\n',
        encoding="utf-8",
    )
    assert catalog._content_root_from_entry_point(pack) is None

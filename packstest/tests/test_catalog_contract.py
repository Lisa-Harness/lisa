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

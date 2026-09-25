"""Tests for overlay mode catalog precedence and shortcut resolution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from sdk.content import (
    CatalogCollisionError,
    build_mode_catalog_json,
    resolve_mode_roots,
)


def _write_mode(
    root: Path,
    *,
    mode_id: str,
    provider_id: str,
    description: str,
    instruction_body: str = "Test mode.",
) -> None:
    """Write a mode directory with _mode.md under a content root."""
    mode_dir = root / "modes" / mode_id
    mode_dir.mkdir(parents=True, exist_ok=True)
    (mode_dir / "_mode.md").write_text(
        "\n".join(
            [
                "---",
                f"mode_id: {mode_id}",
                f"mode_provider_id: {provider_id}",
                "start_stage_id: none",
                "mode_kind: chat",
                f'description: "{description}"',
                "---",
                "",
                f"# {mode_id}",
                "",
                instruction_body,
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_mode_in_provider_folder(
    root: Path,
    *,
    mode_id: str,
    provider_id: str,
    description: str,
    instruction_body: str = "Test mode.",
) -> None:
    """Write a mode under a provider folder (e.g. modes/lisa-pack-core/default/)."""
    mode_dir = root / "modes" / provider_id / mode_id
    mode_dir.mkdir(parents=True, exist_ok=True)
    (mode_dir / "_mode.md").write_text(
        "\n".join(
            [
                "---",
                f"mode_id: {mode_id}",
                f"mode_provider_id: {provider_id}",
                "start_stage_id: none",
                "mode_kind: chat",
                f'description: "{description}"',
                "---",
                "",
                f"# {mode_id}",
                "",
                instruction_body,
                "",
            ]
        ),
        encoding="utf-8",
    )


# ── Overlay precedence tests ──────────────────────────────────────────────


def test_overlay_auto_wins_over_pack(
    tmp_path: Path,
) -> None:
    """An overlay entry with _overlay tag takes precedence over a pack entry."""
    pack_root = tmp_path / "pack"
    overlay_root = tmp_path / "overlay"

    _write_mode(
        pack_root, mode_id="default", provider_id="lisa-pack-core", description="pack default"
    )
    _write_mode(
        overlay_root, mode_id="default", provider_id="lisa-pack-core", description="overlay default"
    )

    mode_roots = resolve_mode_roots([overlay_root, pack_root])
    catalog = json.loads(build_mode_catalog_json(mode_roots, overlay_root=overlay_root))

    assert catalog["default"]["description"] == "overlay default"
    assert catalog["default"]["mode_provider_id"] == "lisa-pack-core"


def test_overlay_beats_explicit_provider_selection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Overlay wins even when session explicitly selects a different provider."""
    pack_a = tmp_path / "pack_a"
    pack_b = tmp_path / "pack_b"
    overlay_root = tmp_path / "overlay"

    _write_mode(pack_a, mode_id="default", provider_id="lisa-pack-core", description="core default")
    _write_mode(
        pack_b, mode_id="default", provider_id="lisa-pack-memory", description="memory default"
    )
    _write_mode(
        overlay_root, mode_id="default", provider_id="lisa-pack-core", description="overlay default"
    )

    monkeypatch.setenv(
        "LISA_SESSION_PROVIDER_SELECTIONS",
        json.dumps({"modes": {"default": "lisa-pack-memory"}}),
    )

    mode_roots = resolve_mode_roots([overlay_root, pack_a, pack_b])
    catalog = json.loads(build_mode_catalog_json(mode_roots, overlay_root=overlay_root))

    # Overlay wins despite session selecting lisa-pack-memory
    assert catalog["default"]["description"] == "overlay default"


def test_multiple_overlay_entries_raises_collision(
    tmp_path: Path,
) -> None:
    """Two overlay entries for the same mode_id raise CatalogCollisionError."""
    pack_root = tmp_path / "pack"
    overlay_root = tmp_path / "overlay"

    _write_mode(
        pack_root, mode_id="default", provider_id="lisa-pack-core", description="pack default"
    )
    # Two overlay entries under different provider folders — both get _overlay tag
    _write_mode_in_provider_folder(
        overlay_root, mode_id="default", provider_id="lisa-pack-core", description="overlay a"
    )
    _write_mode_in_provider_folder(
        overlay_root, mode_id="default", provider_id="lisa-pack-memory", description="overlay b"
    )

    mode_roots = resolve_mode_roots([overlay_root, pack_root])
    with pytest.raises(CatalogCollisionError, match="Multiple overlay entries"):
        build_mode_catalog_json(mode_roots, overlay_root=overlay_root)


def test_no_overlay_dir_unchanged_behavior(
    tmp_path: Path,
) -> None:
    """Without an overlay root, behavior is unchanged."""
    pack_root = tmp_path / "pack"

    _write_mode(
        pack_root, mode_id="default", provider_id="lisa-pack-core", description="pack default"
    )

    mode_roots = resolve_mode_roots([pack_root])
    catalog = json.loads(build_mode_catalog_json(mode_roots))

    assert catalog["default"]["description"] == "pack default"


# ── Shortcut resolution tests ─────────────────────────────────────────────


def test_shortcut_shadows_installed_mode(
    tmp_path: Path,
) -> None:
    """A shortcut _mode.md in the overlay modes root shadows the matching installed mode."""
    pack_root = tmp_path / "pack"
    overlay_root = tmp_path / "overlay"

    _write_mode(
        pack_root, mode_id="default", provider_id="lisa-pack-core", description="pack default"
    )
    # Shortcut: _mode.md directly under overlay/modes/default/
    _write_mode(
        overlay_root, mode_id="default", provider_id="lisa-pack-core", description="overlay default"
    )

    mode_roots = resolve_mode_roots([overlay_root, pack_root])
    catalog = json.loads(build_mode_catalog_json(mode_roots, overlay_root=overlay_root))

    assert catalog["default"]["description"] == "overlay default"


def test_shortcut_with_no_matching_base_raises_in_strict_mode(
    tmp_path: Path,
) -> None:
    """A shortcut that doesn't match any installed mode raises in strict mode."""
    pack_root = tmp_path / "pack"
    overlay_root = tmp_path / "overlay"

    _write_mode(
        pack_root, mode_id="default", provider_id="lisa-pack-core", description="pack default"
    )
    # Shortcut targeting a non-existent provider
    _write_mode(
        overlay_root,
        mode_id="default",
        provider_id="nonexistent-pack",
        description="orphan overlay",
    )

    mode_roots = resolve_mode_roots([overlay_root, pack_root])
    with pytest.raises(CatalogCollisionError, match="does not match any installed mode"):
        build_mode_catalog_json(mode_roots, overlay_root=overlay_root, strict=True)


def test_shortcut_with_no_matching_base_skips_in_non_strict(
    tmp_path: Path,
) -> None:
    """A shortcut that doesn't match any installed mode is skipped in non-strict mode."""
    pack_root = tmp_path / "pack"
    overlay_root = tmp_path / "overlay"

    _write_mode(
        pack_root, mode_id="default", provider_id="lisa-pack-core", description="pack default"
    )
    _write_mode(
        overlay_root,
        mode_id="default",
        provider_id="nonexistent-pack",
        description="orphan overlay",
    )

    mode_roots = resolve_mode_roots([overlay_root, pack_root])
    catalog = json.loads(
        build_mode_catalog_json(mode_roots, overlay_root=overlay_root, strict=False)
    )

    # The orphan shortcut is skipped; the pack entry remains
    assert catalog["default"]["description"] == "pack default"


# ── Provider-folder overlay tests ─────────────────────────────────────────


def test_provider_folder_overlay_shadows_pack(
    tmp_path: Path,
) -> None:
    """A mode under overlays/modes/<provider>/<mode>/ shadows the pack's mode."""
    pack_root = tmp_path / "pack"
    overlay_root = tmp_path / "overlay"

    _write_mode(
        pack_root, mode_id="default", provider_id="lisa-pack-core", description="pack default"
    )
    _write_mode_in_provider_folder(
        overlay_root,
        mode_id="default",
        provider_id="lisa-pack-core",
        description="overlay default",
    )

    mode_roots = resolve_mode_roots([overlay_root, pack_root])
    catalog = json.loads(build_mode_catalog_json(mode_roots, overlay_root=overlay_root))

    assert catalog["default"]["description"] == "overlay default"


def test_provider_folder_mismatch_warns(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A provider folder whose name doesn't match frontmatter mode_provider_id warns."""
    pack_root = tmp_path / "pack"
    overlay_root = tmp_path / "overlay"

    _write_mode(
        pack_root, mode_id="default", provider_id="lisa-pack-core", description="pack default"
    )
    # Folder is named lisa-pack-core but frontmatter says lisa-pack-memory
    mode_dir = overlay_root / "modes" / "lisa-pack-core" / "default"
    mode_dir.mkdir(parents=True, exist_ok=True)
    (mode_dir / "_mode.md").write_text(
        "\n".join(
            [
                "---",
                "mode_id: default",
                "mode_provider_id: lisa-pack-memory",  # mismatched with folder name
                "start_stage_id: none",
                "mode_kind: chat",
                'description: "overlay default"',
                "---",
                "",
                "# default",
                "",
                "Overlay mode.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    mode_roots = resolve_mode_roots([overlay_root, pack_root])
    catalog = json.loads(build_mode_catalog_json(mode_roots, overlay_root=overlay_root))

    # The overlay still wins (frontmatter is authoritative)
    assert catalog["default"]["description"] == "overlay default"
    # But a warning was logged
    assert any(
        "folder name and frontmatter disagree" in record.message for record in caplog.records
    )


# ── _overlay tag stripping test ──────────────────────────────────────────


def test_overlay_tag_stripped_from_output(
    tmp_path: Path,
) -> None:
    """The internal _overlay tag is stripped from the serialized catalog."""
    pack_root = tmp_path / "pack"
    overlay_root = tmp_path / "overlay"

    _write_mode(
        pack_root, mode_id="default", provider_id="lisa-pack-core", description="pack default"
    )
    _write_mode(
        overlay_root, mode_id="default", provider_id="lisa-pack-core", description="overlay default"
    )

    mode_roots = resolve_mode_roots([overlay_root, pack_root])
    catalog = json.loads(build_mode_catalog_json(mode_roots, overlay_root=overlay_root))

    assert "_overlay" not in catalog["default"]

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sdk.lisa_content._catalog import build_mode_catalog_json, resolve_mode_roots


def _write_default_mode(root: Path, *, provider_id: str, description: str) -> None:
    mode_dir = root / "modes" / "default"
    mode_dir.mkdir(parents=True, exist_ok=True)
    (mode_dir / "_mode.md").write_text(
        "\n".join(
            [
                "---",
                "mode_id: default",
                f"mode_provider_id: {provider_id}",
                "start_stage_id: none",
                "mode_kind: chat",
                f'description: "{description}"',
                "---",
                "",
                "# Default",
                "",
                "Provider test mode.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_mode_collision_uses_preference_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    core = tmp_path / "core_pack"
    learning = tmp_path / "learning_pack"
    _write_default_mode(core, provider_id="lisa-pack-core", description="core default")
    _write_default_mode(learning, provider_id="lisa-pack-learning", description="learning default")

    monkeypatch.setenv(
        "LISA_MODE_PROVIDER_PREFERENCES",
        json.dumps({"default": ["lisa-pack-learning", "lisa-pack-core"]}),
    )

    mode_roots = resolve_mode_roots([core, learning])
    catalog = json.loads(build_mode_catalog_json(mode_roots))

    assert catalog["default"]["mode_provider_id"] == "lisa-pack-learning"
    assert catalog["default"]["description"] == "learning default"


def test_mode_collision_falls_back_to_last_discovered_when_no_preference(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    core = tmp_path / "core_pack"
    learning = tmp_path / "learning_pack"
    _write_default_mode(core, provider_id="lisa-pack-core", description="core default")
    _write_default_mode(learning, provider_id="lisa-pack-learning", description="learning default")

    monkeypatch.delenv("LISA_MODE_PROVIDER_PREFERENCES", raising=False)

    mode_roots = resolve_mode_roots([core, learning])
    catalog = json.loads(build_mode_catalog_json(mode_roots))

    assert catalog["default"]["mode_provider_id"] == "lisa-pack-learning"
    assert catalog["default"]["description"] == "learning default"


"""Tests for disabled-pack exclusion helpers in kernel.pack_activation."""

from __future__ import annotations

import json
import os
from pathlib import Path

from kernel.pack_activation import (
    load_disabled_pack_names,
    load_disabled_pack_names_from_env,
)


def test_load_disabled_pack_names_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_disabled_pack_names(tmp_path) == set()


def test_load_disabled_pack_names_corrupt_json_returns_empty(tmp_path: Path) -> None:
    lock = tmp_path / "installed-packs.json"
    lock.write_text("{not valid json", encoding="utf-8")
    assert load_disabled_pack_names(tmp_path) == set()


def test_load_disabled_pack_names_all_enabled_returns_empty(tmp_path: Path) -> None:
    lock = tmp_path / "installed-packs.json"
    lock.write_text(
        json.dumps(
            {
                "packs": [
                    {"package_name": "lisa-pack-cloudflare", "enabled": True},
                    {"package_name": "lisa-pack-github", "enabled": True},
                ],
            },
        ),
        encoding="utf-8",
    )
    assert load_disabled_pack_names(tmp_path) == set()


def test_load_disabled_pack_names_returns_disabled_only(tmp_path: Path) -> None:
    lock = tmp_path / "installed-packs.json"
    lock.write_text(
        json.dumps(
            {
                "packs": [
                    {"package_name": "lisa-pack-cloudflare", "enabled": False},
                    {"package_name": "lisa-pack-github", "enabled": True},
                    {"package_name": "lisa-pack-stripe", "enabled": False},
                ],
            },
        ),
        encoding="utf-8",
    )
    result = load_disabled_pack_names(tmp_path)
    assert result == {"lisa-pack-cloudflare", "lisa-pack-stripe"}


def test_load_disabled_pack_names_defaults_enabled_true(tmp_path: Path) -> None:
    """Packs without an explicit 'enabled' field default to enabled."""
    lock = tmp_path / "installed-packs.json"
    lock.write_text(
        json.dumps(
            {
                "packs": [
                    {"package_name": "lisa-pack-cloudflare"},
                ],
            },
        ),
        encoding="utf-8",
    )
    assert load_disabled_pack_names(tmp_path) == set()


def test_load_disabled_pack_names_normalizes_names(tmp_path: Path) -> None:
    lock = tmp_path / "installed-packs.json"
    lock.write_text(
        json.dumps(
            {
                "packs": [
                    {"package_name": "Lisa_Pack_Cloudflare", "enabled": False},
                ],
            },
        ),
        encoding="utf-8",
    )
    result = load_disabled_pack_names(tmp_path)
    assert result == {"lisa-pack-cloudflare"}


def test_load_disabled_pack_names_empty_packs_list(tmp_path: Path) -> None:
    lock = tmp_path / "installed-packs.json"
    lock.write_text(json.dumps({"packs": []}), encoding="utf-8")
    assert load_disabled_pack_names(tmp_path) == set()


def test_load_disabled_pack_names_non_dict_top_level(tmp_path: Path) -> None:
    lock = tmp_path / "installed-packs.json"
    lock.write_text(json.dumps([]), encoding="utf-8")
    assert load_disabled_pack_names(tmp_path) == set()


def test_load_disabled_pack_names_from_env_unset(monkeypatch) -> None:
    monkeypatch.delenv("LISA_DISABLED_PACKS", raising=False)
    assert load_disabled_pack_names_from_env() == set()


def test_load_disabled_pack_names_from_env_empty(monkeypatch) -> None:
    monkeypatch.setenv("LISA_DISABLED_PACKS", "")
    assert load_disabled_pack_names_from_env() == set()


def test_load_disabled_pack_names_from_env_single(monkeypatch) -> None:
    monkeypatch.setenv("LISA_DISABLED_PACKS", "lisa-pack-cloudflare")
    assert load_disabled_pack_names_from_env() == {"lisa-pack-cloudflare"}


def test_load_disabled_pack_names_from_env_multiple(monkeypatch) -> None:
    sep = os.pathsep
    monkeypatch.setenv(
        "LISA_DISABLED_PACKS",
        sep.join(["lisa-pack-cloudflare", "lisa-pack-stripe"]),
    )
    assert load_disabled_pack_names_from_env() == {
        "lisa-pack-cloudflare",
        "lisa-pack-stripe",
    }


def test_load_disabled_pack_names_from_env_normalizes(monkeypatch) -> None:
    sep = os.pathsep
    monkeypatch.setenv(
        "LISA_DISABLED_PACKS",
        sep.join(["Lisa_Pack_Cloudflare", " lisa-pack-stripe "]),
    )
    assert load_disabled_pack_names_from_env() == {
        "lisa-pack-cloudflare",
        "lisa-pack-stripe",
    }

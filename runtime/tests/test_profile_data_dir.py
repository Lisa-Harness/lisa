"""Tests for LISA_PROFILE — named data-dir sub-workspaces.

A profile (``LISA_PROFILE=<name>``) resolves ``HostConfig.data_dir`` to
``<profiles_root>/<name>`` and shares the base data dir's packs venv.
Precedence: explicit data-dir arg/``LISA_DATA_DIR`` wins over the
profile; the profile wins over the platform default. Names are validated to
a single path segment so a profile can never escape the profiles root.

The profiles root itself defaults to ``<data-dir>/profiles`` and is
relocatable with ``LISA_PROFILES_DIR`` (e.g. onto a synced OneDrive
folder). A relocated root still shares the base data dir's packs venv.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from runtime.config import (
    HostConfig,
    base_data_dir,
    default_data_dir,
    profile_data_dir,
    profiles_dir,
)
from runtime.paths import packs_venv_for, profile_base_data_dir

# ── profile_data_dir ─────────────────────────────────────────────────


def test_profile_resolves_under_profiles_root() -> None:
    """LISA_PROFILE=name → <base>/profiles/<name>."""
    base = default_data_dir()
    assert profile_data_dir("taxes") == (base / "profiles" / "taxes").resolve()


def test_profile_accepts_path_safe_names() -> None:
    """Valid names: alphanumerics plus '-', '_', '.', not starting with '.'."""
    assert profile_data_dir("project-x_2026").name == "project-x_2026"
    assert profile_data_dir("a.b").name == "a.b"


@pytest.mark.parametrize(
    "bad",
    ["..", "../escape", "a/b", "a\\b", ".hidden"],
)
def test_profile_rejects_unsafe_names(bad: str) -> None:
    """Traversal / separator / leading-dot names are rejected."""
    with pytest.raises(ValueError, match=r"[Ii]nvalid profile name"):
        profile_data_dir(bad)


def test_profile_rejects_blank_name() -> None:
    """A whitespace-only profile name is rejected (distinct message)."""
    with pytest.raises(ValueError, match=r"must not be empty"):
        profile_data_dir("  ")


def test_profile_base_dir_override(tmp_path: pytest.TempPathFactory) -> None:
    """An explicit base anchors the profile instead of ~/.lisa."""
    got = profile_data_dir("survey", base_dir=tmp_path)
    assert got == (tmp_path / "profiles" / "survey").resolve()
    assert got.parent.parent == tmp_path.resolve()


def test_profile_dir_not_created_by_resolver(tmp_path) -> None:
    """Resolution is pure — the directory is created lazily by the host."""
    got = profile_data_dir("never-created", base_dir=tmp_path)
    assert not got.exists()


# ── HostConfig.from_env precedence ────────────────────────────────────


def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "LISA_DATA_DIR",
        "LISA_PROFILE",
        "LISA_PROFILES_DIR",
        "LISA_THREAD_DIR",
        "LISA_TIMEZONE",
    ):
        monkeypatch.delenv(var, raising=False)


def test_from_env_profile_sets_data_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    """LISA_PROFILE routes data_dir to <default>/profiles/<name>."""
    _clean_env(monkeypatch)
    monkeypatch.setenv("LISA_PROFILE", "taxes")
    config = HostConfig.from_env()
    assert config.data_dir == (default_data_dir() / "profiles" / "taxes").resolve()


def test_from_env_explicit_arg_beats_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit data_dir kwarg (CLI --data-dir) wins over LISA_PROFILE."""
    _clean_env(monkeypatch)
    monkeypatch.setenv("LISA_PROFILE", "taxes")
    config = HostConfig.from_env(data_dir="/tmp/explicit-lisa")
    assert config.data_dir == HostConfig.from_env(data_dir="/tmp/explicit-lisa").data_dir
    assert "profiles" not in str(config.data_dir)


def test_from_env_data_dir_env_beats_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """LISA_DATA_DIR (machine pin) wins over LISA_PROFILE."""
    _clean_env(monkeypatch)
    monkeypatch.setenv("LISA_PROFILE", "taxes")
    monkeypatch.setenv("LISA_DATA_DIR", "/tmp/env-lisa")
    config = HostConfig.from_env()
    assert config.data_dir == HostConfig.from_env(data_dir="/tmp/env-lisa").data_dir


def test_from_env_invalid_profile_raises_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unsafe profile name surfaces as a startup ValueError."""
    _clean_env(monkeypatch)
    monkeypatch.setenv("LISA_PROFILE", "../escape")
    with pytest.raises(ValueError, match=r"[Ii]nvalid profile name"):
        HostConfig.from_env()


def test_from_env_without_profile_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """No profile → the platform default (~/.lisa), exactly as before."""
    _clean_env(monkeypatch)
    config = HostConfig.from_env()
    assert config.data_dir == default_data_dir().resolve()


# ── packs venv sharing ───────────────────────────────────────────────


def test_packs_venv_for_profile_shares_base_venv(tmp_path) -> None:
    """<base>/profiles/<name> → the venv is <base>/venv, not <profile>/venv."""
    profile = tmp_path / "profiles" / "taxes"
    assert packs_venv_for(profile) == tmp_path / "venv"


def test_packs_venv_for_non_profile_keeps_data_dir_venv(tmp_path) -> None:
    """A plain data dir keeps its own venv (existing behavior)."""
    assert packs_venv_for(tmp_path) == tmp_path / "venv"


def test_packs_venv_override_beats_profile_sharing(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """LISA_PACKS_VENV (explicit) still wins over profile base sharing."""
    monkeypatch.setenv("LISA_PACKS_VENV", str(tmp_path / "custom-venv"))
    profile = tmp_path / "profiles" / "taxes"
    assert packs_venv_for(profile) == tmp_path / "custom-venv"


def test_profile_base_data_dir_detects_and_rejects(tmp_path) -> None:
    """Structural detection: parent named profiles/ → base, else None."""
    profile = tmp_path / "profiles" / "survey"
    assert profile_base_data_dir(profile) == tmp_path
    assert profile_base_data_dir(tmp_path) is None
    # A dir literally named profiles/ at the root: parent of profiles/<x>
    # is <x>'s parent = profiles/, whose parent is tmp's parent — still
    # structural, no special-casing of ~.
    assert profile_base_data_dir(profile / "profiles" / "nested") == profile


def test_default_profile_root_shares_default_base(monkeypatch: pytest.MonkeyPatch) -> None:
    _clean_env(monkeypatch)
    base = default_data_dir().resolve()

    assert profile_base_data_dir(base / "profiles" / "survey") == base


# ── LISA_PROFILES_DIR — relocatable profiles root ───────────────────


def test_profiles_dir_defaults_under_data_dir(tmp_path) -> None:
    """Without the override the root is <base>/profiles."""
    assert profiles_dir(tmp_path) == (tmp_path / "profiles").resolve()


def test_profiles_dir_override_relocates_root(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """LISA_PROFILES_DIR moves the root; base_dir is then ignored."""
    monkeypatch.setenv("LISA_PROFILES_DIR", str(tmp_path / "Lisa.Profiles"))
    assert profiles_dir(tmp_path / "ignored") == (tmp_path / "Lisa.Profiles").resolve()


def test_profiles_dir_override_expands_user(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """A leading ~ expands to the home directory."""
    monkeypatch.setenv("LISA_PROFILES_DIR", "~/Lisa.Profiles")
    assert profiles_dir() == (Path.home() / "Lisa.Profiles").resolve()


def test_profile_resolves_under_relocated_root(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """A profile lives under the relocated root: <root>/<name>."""
    monkeypatch.setenv("LISA_PROFILES_DIR", str(tmp_path / "Lisa.Profiles"))
    assert profile_data_dir("taxes") == (tmp_path / "Lisa.Profiles" / "taxes").resolve()


def test_relocated_profile_shares_base_packs_venv(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """A relocated profile still uses the BASE data dir's venv.

    The relocated root may live in a synced folder with no venv of its
    own, so the venv stays with the data dir.
    """
    monkeypatch.setenv("LISA_PROFILES_DIR", str(tmp_path / "OneDrive" / "Lisa.Profiles"))
    monkeypatch.setenv("LISA_DATA_DIR", str(tmp_path / "lisa-data"))
    profile = tmp_path / "OneDrive" / "Lisa.Profiles" / "taxes"
    assert profile_base_data_dir(profile) == (tmp_path / "lisa-data").resolve()
    assert packs_venv_for(profile) == (tmp_path / "lisa-data" / "venv")


def test_relocated_profile_defaults_to_platform_data_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Without LISA_DATA_DIR the base is the platform default (~/.lisa)."""
    monkeypatch.delenv("LISA_DATA_DIR", raising=False)
    monkeypatch.setenv("LISA_PROFILES_DIR", str(tmp_path / "Lisa.Profiles"))
    profile = tmp_path / "Lisa.Profiles" / "taxes"
    assert profile_base_data_dir(profile) == default_data_dir().resolve()


def test_from_env_profile_uses_relocated_root(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """LISA_PROFILE resolves under the relocated root, not <base>/profiles."""
    _clean_env(monkeypatch)
    monkeypatch.setenv("LISA_PROFILES_DIR", str(tmp_path / "Lisa.Profiles"))
    monkeypatch.setenv("LISA_PROFILE", "taxes")
    config = HostConfig.from_env()
    assert config.data_dir == (tmp_path / "Lisa.Profiles" / "taxes").resolve()


def test_base_data_dir_follows_data_dir_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """base_data_dir honors LISA_DATA_DIR, else the platform default."""
    monkeypatch.setenv("LISA_DATA_DIR", str(tmp_path / "custom"))
    assert base_data_dir() == (tmp_path / "custom").resolve()
    monkeypatch.delenv("LISA_DATA_DIR")
    assert base_data_dir() == default_data_dir().resolve()

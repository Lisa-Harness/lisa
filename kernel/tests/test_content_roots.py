from __future__ import annotations

import os
from pathlib import Path

from kernel import content_roots


class _FakeEntryPoint:
    def __init__(
        self,
        *,
        name: str,
        root: Path | None = None,
        error: Exception | None = None,
        dist_name: str | None = None,
    ) -> None:
        self.name = name
        self._root = root
        self._error = error

        class _FakeDist:
            class metadata:  # type: ignore[no-redef]
                pass

            def __init__(inner, dname: str) -> None:
                inner.metadata = {"Name": dname}  # type: ignore[method-assign]

        self.dist = _FakeDist(dist_name) if dist_name else None

    def load(self):
        if self._error is not None:
            raise self._error
        return lambda: self._root


def test_discover_configured_content_roots_uses_os_pathsep(
    monkeypatch,
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "pack-one"
    second_root = tmp_path / "pack-two"
    first_root.mkdir()
    second_root.mkdir()

    monkeypatch.setattr(
        content_roots.settings,
        "sdk_content_roots",
        os.pathsep.join([str(first_root), str(second_root)]),
    )

    assert content_roots.discover_configured_content_roots() == [first_root, second_root]


def test_discover_installed_content_roots_skips_broken_and_missing_entries(
    monkeypatch,
    tmp_path: Path,
) -> None:
    installed_root = tmp_path / "installed-pack"
    installed_root.mkdir()

    monkeypatch.setattr(
        content_roots.importlib.metadata,
        "entry_points",
        lambda group: [
            _FakeEntryPoint(name="installed", root=installed_root),
            _FakeEntryPoint(name="duplicate", root=installed_root),
            _FakeEntryPoint(name="missing", root=tmp_path / "missing-pack"),
            _FakeEntryPoint(name="broken", error=RuntimeError("boom")),
        ],
    )

    assert content_roots.discover_installed_content_roots() == [installed_root]


def test_discover_installed_content_roots_excludes_disabled_packs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Entry points whose distribution name is in excluded_packs are skipped."""
    active_root = tmp_path / "active-pack"
    disabled_root = tmp_path / "disabled-pack"
    active_root.mkdir()
    disabled_root.mkdir()

    monkeypatch.setattr(
        content_roots.importlib.metadata,
        "entry_points",
        lambda group: [
            _FakeEntryPoint(name="active", root=active_root, dist_name="lisa-pack-active"),
            _FakeEntryPoint(name="disabled", root=disabled_root, dist_name="lisa-pack-disabled"),
        ],
    )

    result = content_roots.discover_installed_content_roots(
        excluded_packs={"lisa-pack-disabled"},
    )
    assert result == [active_root]


def test_discover_installed_content_roots_excluded_packs_none_returns_all(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """excluded_packs=None preserves backward-compatible behavior (no filtering)."""
    first_root = tmp_path / "pack-one"
    second_root = tmp_path / "pack-two"
    first_root.mkdir()
    second_root.mkdir()

    monkeypatch.setattr(
        content_roots.importlib.metadata,
        "entry_points",
        lambda group: [
            _FakeEntryPoint(name="one", root=first_root, dist_name="lisa-pack-one"),
            _FakeEntryPoint(name="two", root=second_root, dist_name="lisa-pack-two"),
        ],
    )

    result = content_roots.discover_installed_content_roots(excluded_packs=None)
    assert result == [first_root, second_root]


def test_discover_packs_venv_content_roots_missing_venv_returns_empty(
    tmp_path: Path,
) -> None:
    """Regression: first launch with no packs venv must not crash boot.

    On Linux with no ``.env``, the packs venv does not exist yet. The
    site-packages path computed by ``packs_venv_site_packages`` points at
    a non-existent ``lib/pythonX.Y/site-packages``. ``_scan_editable_packs``
    previously called ``site_packages.iterdir()`` unguarded, raising
    ``FileNotFoundError`` that propagated up through
    ``discover_packs_venv_content_roots`` and crashed boot.
    """
    nonexistent_venv = tmp_path / "packs venv"  # deliberately not created

    def venv_resolver(_data_dir: Path | None) -> Path:
        return nonexistent_venv

    def site_packages_resolver(venv_path: Path) -> Path:
        # Mirror the real resolver: a path that does not exist on disk.
        return venv_path / "lib" / "python3.13" / "site-packages"

    result = content_roots.discover_packs_venv_content_roots(
        data_dir=tmp_path,
        venv_path_resolver=venv_resolver,
        site_packages_resolver=site_packages_resolver,
    )
    assert result == []


def test_discover_packs_venv_content_roots_venv_exists_no_packs_returns_empty(
    tmp_path: Path,
) -> None:
    """Empty packs venv (no lisa_packs/, no editable finders) returns empty."""
    venv = tmp_path / "packs venv"
    site = venv / "lib" / "python3.11" / "site-packages"
    site.mkdir(parents=True)

    def venv_resolver(_data_dir: Path | None) -> Path:
        return venv

    def site_packages_resolver(venv_path: Path) -> Path:
        return venv_path / "lib" / "python3.11" / "site-packages"

    result = content_roots.discover_packs_venv_content_roots(
        data_dir=tmp_path,
        venv_path_resolver=venv_resolver,
        site_packages_resolver=site_packages_resolver,
    )
    assert result == []

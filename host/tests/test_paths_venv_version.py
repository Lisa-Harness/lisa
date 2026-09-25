"""Regression tests for ``host.paths`` venv site-packages resolution.

Covers the Linux-first-launch bug where the packs venv is created by
``uv venv`` with a Python version (e.g. 3.11) that differs from the host
interpreter (e.g. 3.13). ``packs_venv_site_packages`` must read the
venv's own ``pyvenv.cfg`` (or scan ``lib/``) rather than assume the host
interpreter's ``sys.version_info`` matches the venv's.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from host.paths import _venv_python_dir, packs_venv_site_packages


def _write_pyvenv_cfg(venv: Path, version: str) -> None:
    """Write a minimal ``pyvenv.cfg`` with a ``version_info`` line."""
    (venv / "pyvenv.cfg").write_text(
        f"home = /usr/bin\nversion_info = {version}\n",
        encoding="utf-8",
    )


def test_venv_python_dir_reads_version_info(tmp_path: Path) -> None:
    """``pyvenv.cfg`` ``version_info`` wins over ``sys.version_info``."""
    venv = tmp_path / "packs venv"
    venv.mkdir()
    _write_pyvenv_cfg(venv, "3.11.7")
    assert _venv_python_dir(venv) == "python3.11"


def test_venv_python_dir_reads_version_field(tmp_path: Path) -> None:
    """Older ``pyvenv.cfg`` uses ``version`` instead of ``version_info``."""
    venv = tmp_path / "packs venv"
    venv.mkdir()
    (venv / "pyvenv.cfg").write_text(
        "home = /usr/bin\nversion = 3.12.1\n",
        encoding="utf-8",
    )
    assert _venv_python_dir(venv) == "python3.12"


def test_venv_python_dir_scans_lib_when_no_cfg(tmp_path: Path) -> None:
    """Without ``pyvenv.cfg``, fall back to scanning ``lib/python*/``."""
    venv = tmp_path / "packs venv"
    lib = venv / "lib"
    (lib / "python3.10").mkdir(parents=True)
    (lib / "python3.11").mkdir()
    assert _venv_python_dir(venv) == "python3.10"


def test_venv_python_dir_falls_back_to_host_version(tmp_path: Path) -> None:
    """No cfg, no ``lib/`` dir → use the running interpreter's version."""
    venv = tmp_path / "empty-venv"
    venv.mkdir()
    expected = f"python{sys.version_info.major}.{sys.version_info.minor}"
    assert _venv_python_dir(venv) == expected


@pytest.mark.skipif(
    os.name == "nt",
    reason="POSIX venv layout (lib/pythonX.Y) only; Windows uses Lib/site-packages",
)
def test_packs_venv_site_packages_uses_venv_version_not_host(tmp_path: Path) -> None:
    """End-to-end regression: site-packages path follows the venv's Python.

    Simulates the reported bug: host runs 3.13 but ``uv venv`` created the
    packs venv with 3.11. The returned path must point at
    ``lib/python3.11/site-packages``, not ``lib/python3.13/site-packages``.
    """
    venv = tmp_path / "packs venv"
    site = venv / "lib" / "python3.11" / "site-packages"
    site.mkdir(parents=True)
    _write_pyvenv_cfg(venv, "3.11.4")

    result = packs_venv_site_packages(venv)
    assert result == site

    # The bug would have produced python3.13 (host version) instead.
    host_dir = (
        venv / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    )
    assert result != host_dir or sys.version_info[:2] == (3, 11)


def test_packs_venv_site_packages_windows_layout(tmp_path: Path) -> None:
    """On Windows the layout is ``Lib/site-packages`` regardless of version."""
    venv = tmp_path / "packs venv"
    venv.mkdir()
    # The Windows branch does not consult pyvenv.cfg, so this test just
    # asserts the shape when os.name == "nt". On non-Windows runners the
    # assertion is skipped to avoid a false positive.
    if os.name != "nt":
        return
    result = packs_venv_site_packages(venv)
    assert result == venv / "Lib" / "site-packages"

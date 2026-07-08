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
    ) -> None:
        self.name = name
        self._root = root
        self._error = error

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

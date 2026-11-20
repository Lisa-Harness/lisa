"""Tests for the release metadata gate."""

from __future__ import annotations

import importlib.util
import runpy
import sys
from pathlib import Path
from types import ModuleType

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "check_release_metadata.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("release_metadata_under_test", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


release_metadata = _load_module()


def _write_project(path: Path, *, name: str, version: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'[project]\nname = "{name}"\nversion = "{version}"\n',
        encoding="utf-8",
    )


def _minimal_repo(root: Path) -> None:
    _write_project(root / "pyproject.toml", name="lisa-workspace", version="1.2.3")
    (root / "CHANGELOG.md").write_text(
        "# Changelog\n\n## [1.2.3] - 2026-11-20\n",
        encoding="utf-8",
    )
    for directory, name in release_metadata._PACKAGES.items():
        _write_project(root / directory / "pyproject.toml", name=name, version="1.2.3")


def test_repository_release_metadata_is_valid() -> None:
    assert release_metadata.validate(_ROOT)


def test_validate_reports_all_mismatches(tmp_path: Path) -> None:
    _minimal_repo(tmp_path)
    _write_project(
        tmp_path / "kernel" / "pyproject.toml",
        name="wrong-name",
        version="9.9.9",
    )

    with pytest.raises(release_metadata.ReleaseMetadataError) as exc_info:
        release_metadata.validate(tmp_path, tag="v2.0.0")

    message = str(exc_info.value)
    assert "expected distribution 'lisa-harness-kernel'" in message
    assert "workspace version '1.2.3'" in message
    assert "release tag 'v2.0.0'" in message


def test_validate_requires_matching_tag(tmp_path: Path) -> None:
    _minimal_repo(tmp_path)
    assert release_metadata.validate(tmp_path, tag="v1.2.3") == "1.2.3"
    with pytest.raises(release_metadata.ReleaseMetadataError, match="release tag"):
        release_metadata.validate(tmp_path, tag="v1.2.4")


@pytest.mark.parametrize("date_text", ["TBD", "2026-02-30", "2026-11-20 trailing"])
def test_validate_requires_dated_changelog_heading(tmp_path: Path, date_text: str) -> None:
    _minimal_repo(tmp_path)
    (tmp_path / "CHANGELOG.md").write_text(
        f"# Changelog\n\n## [1.2.3] - {date_text}\n",
        encoding="utf-8",
    )

    with pytest.raises(
        release_metadata.ReleaseMetadataError,
        match=r"no dated release section for 1\.2\.3",
    ):
        release_metadata.validate(tmp_path)


def test_script_entrypoint_validates_repository(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", [str(_SCRIPT)])
    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(_SCRIPT), run_name="__main__")

    assert exc_info.value.code == 0
    assert "release metadata valid for" in capsys.readouterr().out


def test_main_reports_invalid_repository(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        release_metadata.main(["--root", str(tmp_path)])

    assert exc_info.value.code == 1
    assert capsys.readouterr().err.startswith("release metadata:")

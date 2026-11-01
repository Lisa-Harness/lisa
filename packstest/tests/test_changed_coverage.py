"""The changed-line gate must fail closed, including for unmeasured new files."""

from __future__ import annotations

import importlib
import runpy
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
from coverage import Coverage
from coverage.exceptions import CoverageException


@pytest.fixture
def gate(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    monkeypatch.syspath_prepend(str(root))
    return importlib.import_module("scripts.check_changed_coverage")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q", "--initial-branch=main", str(tmp_path)], check=True)
    for key, value in (
        ("user.name", "Test"),
        ("user.email", "test@example.com"),
        ("commit.gpgsign", "false"),
        ("core.autocrlf", "false"),
    ):
        subprocess.run(["git", "-C", str(tmp_path), "config", key, value], check=True)
    (tmp_path / "seed.py").write_text("VALUE = 1\n", encoding="utf-8")
    _commit(tmp_path)
    return tmp_path


def _commit(repo: Path) -> None:
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fixture"], check=True)


def _measure(repo: Path) -> Coverage:
    cov = Coverage(
        data_file=str(repo / ".coverage"), config_file=False, branch=True, source=[str(repo)]
    )
    cov.start()
    runpy.run_path(str(repo / "seed.py"))
    cov.stop()
    cov.save()
    return cov


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("sdk/src/sdk/file.py", True),
        ("scripts/check.py", True),
        ("tests/helper.py", False),
        ("sdk/tests/file.py", False),
        ("sdk/test_file.py", False),
        ("conftest.py", False),
        ("vulture_allowlist.py", False),
        ("README.md", False),
    ],
)
def test_source_selection(gate: ModuleType, path: str, expected: bool) -> None:
    assert gate._source(path) is expected


def test_git_delta_includes_all_states_and_unusual_names(gate: ModuleType, repo: Path) -> None:
    (repo / "seed.py").write_text("VALUE = 2\n", encoding="utf-8")
    _commit(repo)
    (repo / "seed.py").write_text("VALUE = 2\nMORE = 3\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "seed.py"], check=True)
    (repo / "seed.py").write_text("VALUE = 2\nMORE = 3\nLAST = 4\n", encoding="utf-8")
    unusual = "new name\twith space.py" if sys.platform != "win32" else "new name with space.py"
    (repo / unusual).write_text("# comment\nVALUE = 1\n", encoding="utf-8")
    assert gate.changed_lines(repo, "HEAD~1") == {
        unusual: {1, 2},
        "seed.py": {1, 2, 3},
    }


def test_rename_and_deletion_hunks(gate: ModuleType, repo: Path) -> None:
    (repo / "seed.py").write_text("ONE = 1\nTWO = 2\nTHREE = 3\n", encoding="utf-8")
    _commit(repo)
    (repo / "seed.py").write_text("ONE = 1\nTHREE = 3\n", encoding="utf-8")
    assert gate.changed_lines(repo, "HEAD") == {"seed.py": set()}
    (repo / "seed.py").rename(repo / "renamed.py")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    changes = gate.changed_lines(repo, "HEAD")
    assert set(changes) == {"renamed.py"}
    (repo / "renamed.py").unlink()
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    assert gate.changed_lines(repo, "HEAD") == {}


def test_invalid_git_ref_fails(gate: ModuleType, repo: Path) -> None:
    with pytest.raises(RuntimeError):
        gate.changed_lines(repo, "missing-ref")
    assert gate.main(["--cwd", str(repo), "--base", "missing-ref"]) == 2


def test_absent_source_measurement_does_not_pass(gate: ModuleType, repo: Path) -> None:
    cov = _measure(repo)
    (repo / "new.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert gate.check_coverage(cov, {"new.py": {1}}, repo) == 1


def test_changed_function_body_and_comments(gate: ModuleType, repo: Path) -> None:
    (repo / "seed.py").write_text(
        "# comment\nVALUE = 1\ndef unused():\n    return 2\n", encoding="utf-8"
    )
    cov = _measure(repo)
    assert gate.check_coverage(cov, {"seed.py": {1}}, repo) == 0
    assert gate.check_coverage(cov, {"seed.py": {2}}, repo) == 0
    assert gate.check_coverage(cov, {"seed.py": {3, 4}}, repo) == 1


@pytest.mark.parametrize(
    ("total", "missing", "expected"),
    [(100, 4, 0), (100, 5, 0), (100, 6, 1), (200, 11, 1), (100000, 5001, 1)],
)
def test_exact_95_percent_boundary(
    gate: ModuleType,
    repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    total: int,
    missing: int,
    expected: int,
) -> None:
    cov = _measure(repo)
    monkeypatch.setattr(
        cov,
        "analysis2",
        lambda path: (path, list(range(1, total + 1)), [], list(range(1, missing + 1)), ""),
    )
    assert gate.check_coverage(cov, {"seed.py": set(range(1, total + 1))}, repo) == expected


def test_missing_and_line_only_data_fail(
    gate: ModuleType, repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert gate.main(["--cwd", str(repo), "--base", "HEAD"]) == 2
    assert "Missing branch-coverage data" in capsys.readouterr().err
    cov = Coverage(
        data_file=str(repo / ".coverage"), config_file=False, branch=False, source=[str(repo)]
    )
    cov.start()
    runpy.run_path(str(repo / "seed.py"))
    cov.stop()
    cov.save()
    assert gate.main(["--cwd", str(repo), "--base", "HEAD"]) == 2
    assert "Missing branch-coverage data" in capsys.readouterr().err


def test_cli_success_and_coverage_failure(
    gate: ModuleType, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (repo / "seed.py").write_text("VALUE = 2\n", encoding="utf-8")
    _measure(repo)
    assert gate.main(["--cwd", str(repo), "--base", "HEAD"]) == 0
    monkeypatch.setattr(
        gate.Coverage,
        "analysis2",
        lambda *args: (_ for _ in ()).throw(CoverageException("bad data")),
    )
    assert gate.main(["--cwd", str(repo), "--base", "HEAD"]) == 2


def test_unresolved_base_fails(
    gate: ModuleType, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gate, "resolve_base", lambda **kwargs: None)
    assert gate.main(["--cwd", str(repo)]) == 2


@pytest.mark.skipif(not hasattr(Path, "symlink_to"), reason="symlinks unavailable")
def test_source_cannot_escape_repo(gate: ModuleType, repo: Path, tmp_path: Path) -> None:
    target = tmp_path.parent / f"{tmp_path.name}-outside.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    try:
        try:
            (repo / "escape.py").symlink_to(target)
        except (OSError, NotImplementedError) as exc:
            pytest.skip(f"symlink creation unavailable: {exc}")
        with pytest.raises(RuntimeError, match="escapes"):
            gate.changed_lines(repo, "HEAD")
    finally:
        target.unlink()


def test_explicit_config(gate: ModuleType, repo: Path) -> None:
    (repo / "pyproject.toml").write_text("[tool.coverage.run]\nbranch = true\n", encoding="utf-8")
    (repo / "seed.py").write_text("VALUE = 2\n", encoding="utf-8")
    _measure(repo)
    assert gate.main(["--cwd", str(repo), "--base", "HEAD"]) == 0


@pytest.mark.parametrize(("first_missing", "second_missing"), [(0, 6), (6, 0), (6, 6)])
def test_multiple_files_accumulate_coverage_and_report_missing_lines(
    gate: ModuleType,
    repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    capsys: pytest.CaptureFixture[str],
    first_missing: int,
    second_missing: int,
) -> None:
    cov = _measure(repo)

    def analysis(path: str) -> tuple[str, list[int], list[int], list[int], str]:
        missing = first_missing if path.endswith("first.py") else second_missing
        return path, list(range(1, 101)), [], list(range(1, missing + 1)), ""

    monkeypatch.setattr(cov, "analysis2", analysis)
    changed = {name: set(range(1, 101)) for name in ("first.py", "second.py")}
    missing = first_missing + second_missing
    assert gate.check_coverage(cov, changed, repo) == int(missing > 10)
    output = capsys.readouterr().out
    assert f"{200 - missing}/200 executable lines" in output
    for name, count in (("first.py", first_missing), ("second.py", second_missing)):
        if count:
            assert f"{name}: uncovered changed lines 1, 2, 3, 4, 5, 6" in output


def test_single_line_hunk_without_count(
    gate: ModuleType, repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def git(cwd: Path, *args: str) -> str:
        if "--name-only" in args:
            return "seed.py\0"
        if "ls-files" in args:
            return ""
        assert "--no-ext-diff" in args and "--no-textconv" in args
        return "@@ -1 +2 @@\n-VALUE = 1\n+VALUE = 2\n"

    monkeypatch.setattr(gate, "_git", git)
    assert gate.changed_lines(repo, "HEAD") == {"seed.py": {2}}


def test_ignored_untracked_source_is_not_selected(gate: ModuleType, repo: Path) -> None:
    (repo / ".gitignore").write_text("ignored.py\n", encoding="utf-8")
    (repo / "ignored.py").write_text("VALUE = 2\n", encoding="utf-8")
    assert gate.changed_lines(repo, "HEAD") == {}


@pytest.mark.parametrize("stderr", ["details\n", ""])
def test_git_failure_preserves_diagnostic(
    gate: ModuleType, repo: Path, monkeypatch: pytest.MonkeyPatch, stderr: str
) -> None:
    monkeypatch.setattr(
        gate.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 1, "", stderr),
    )
    with pytest.raises(RuntimeError, match="details" if stderr else "git command failed"):
        gate._git(repo, "status")

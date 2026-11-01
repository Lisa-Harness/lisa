"""The combined coverage check must stop on every failed prerequisite."""

from __future__ import annotations

import importlib
import runpy
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


@pytest.fixture
def runner(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.delenv("COVERAGE_BASE", raising=False)
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    return importlib.import_module("check_coverage")


@pytest.mark.parametrize("failed", [None, "global", "package", "tests", "changed"])
def test_combined_runner_fails_at_correct_step(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch, failed: str | None
) -> None:
    commands: list[list[str]] = []

    def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        commands.append(argv)
        is_test = "pytest" in argv
        is_changed = any("check_changed_coverage.py" in part for part in argv)
        code = int((is_test and failed == "tests") or (is_changed and failed == "changed"))
        return subprocess.CompletedProcess(argv, code)

    def coverage(*args: str) -> subprocess.CompletedProcess[str]:
        global_report = args == ("report",)
        code = int(failed == ("global" if global_report else "package"))
        return subprocess.CompletedProcess(
            args, code, "TOTAL 10 0 100%\n", "report failed" if code else ""
        )

    monkeypatch.setattr(runner.subprocess, "run", run)
    monkeypatch.setattr(runner, "_coverage", coverage)
    assert runner.main(["--run", "--base", "main"]) == int(failed is not None)
    if failed is None or failed == "changed":
        assert commands[-1][-2:] == ["--base", "main"]
    else:
        assert not any("check_changed_coverage.py" in part for cmd in commands for part in cmd)


def test_floor_only_runner_and_missing_total(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        runner, "_coverage", lambda *args: subprocess.CompletedProcess(args, 1, "", "No data")
    )
    assert runner.main([]) == 1
    monkeypatch.setattr(
        runner, "_coverage", lambda *args: subprocess.CompletedProcess(args, 0, "", "")
    )
    assert runner.main([]) == 1
    with pytest.raises(SystemExit) as raised:
        runner.main(["--base", "main"])
    assert raised.value.code == 2


def test_runner_without_explicit_base(runner: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    commands: list[list[str]] = []

    def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        commands.append(argv)
        return subprocess.CompletedProcess(argv, 0, "TOTAL 10 0 100%\n", "")

    monkeypatch.setattr(runner.subprocess, "run", run)
    assert runner.main(["--run"]) == 0
    assert commands[-1][-1].endswith("check_changed_coverage.py")


def test_runner_reads_ci_base_without_shell_expansion(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []

    def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        commands.append(argv)
        return subprocess.CompletedProcess(argv, 0, "TOTAL 10 0 100%\n", "")

    monkeypatch.setattr(runner.subprocess, "run", run)
    monkeypatch.setenv("COVERAGE_BASE", "main")
    assert runner.main(["--run"]) == 0
    assert commands[-1][-2:] == ["--base", "main"]


def test_coverage_subprocess_uses_current_interpreter(
    runner: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []

    def run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        commands.append(argv)
        assert kwargs == {"capture_output": True, "text": True, "check": False}
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(runner.subprocess, "run", run)
    assert runner._coverage("report").returncode == 0
    assert commands == [[runner.sys.executable, "-m", "coverage", "report"]]


def test_changed_gate_entry_point(runner: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    # Resolve-base failure exercises the CLI wrapper without rewriting this repository.
    monkeypatch.setattr("check_tests.resolve_base", lambda **kwargs: None)
    script = str(Path(runner.__file__).with_name("check_changed_coverage.py"))
    monkeypatch.setattr("sys.argv", [script])
    with pytest.raises(SystemExit) as raised:
        runpy.run_path(script, run_name="__main__")
    assert raised.value.code == 2


def test_coverage_runner_entry_point(runner: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 0, "TOTAL 10 0 100%\n", ""),
    )
    monkeypatch.setattr("sys.argv", [runner.__file__])
    with pytest.raises(SystemExit) as raised:
        runpy.run_path(runner.__file__, run_name="__main__")
    assert raised.value.code == 0

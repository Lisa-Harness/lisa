"""Tests for the offline coverage gate driver."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def coverage_gate(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    root = Path(__file__).resolve().parents[2]
    monkeypatch.syspath_prepend(str(root / "scripts"))
    return importlib.import_module("check_coverage")


def test_skip_changed_lines_keeps_suite_and_package_gates(
    coverage_gate: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    test_commands: list[list[str]] = []
    floor_checks: list[bool] = []
    monkeypatch.setattr(
        coverage_gate.subprocess,
        "run",
        lambda command, check=False: test_commands.append(command) or SimpleNamespace(returncode=0),
    )
    monkeypatch.setattr(
        coverage_gate,
        "_coverage",
        lambda *args: SimpleNamespace(returncode=0, stdout="TOTAL 100%\n", stderr=""),
    )
    monkeypatch.setattr(
        coverage_gate,
        "_check_package_floors",
        lambda: floor_checks.append(True) or 0,
    )

    assert coverage_gate.main(["--run", "--skip-changed-lines"]) == 0
    assert len(test_commands) == 1
    assert "pytest" in test_commands[0]
    assert floor_checks == [True]

"""Tests for the CLI entry ``scripts/startup_benchmark.py``.

The script wires the library's probe/summarize/format pieces into one
report run; these tests patch the child-probe boundary so the CLI flow
(cold + warm aggregation, the ``--boot`` variant, JSON publication and the
failed-probe exit) is exercised without spawning real interpreters.
The library itself is covered by ``test_startup_benchmark.py``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from runtime.startup_benchmark import BenchmarkSample

_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "startup_benchmark.py"


@pytest.fixture
def script(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import ``scripts/startup_benchmark.py`` as a module."""
    spec = importlib.util.spec_from_file_location("startup_benchmark_under_test", _SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _ok(variant: str, wall: float = 1.0) -> BenchmarkSample:
    return BenchmarkSample(variant=variant, wall_seconds=wall, exit_code=0, setup_invocations=1)


def test_main_runs_cold_then_warm_and_prints_report(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen_variants: list[str] = []

    def fake_probe(command: list[str], *, variant: str, timeout_seconds: float) -> BenchmarkSample:
        seen_variants.append(variant)
        return _ok(variant)

    monkeypatch.setattr(script, "run_child_probe", fake_probe)
    assert script.main(["--samples", "3", "--python", "python3"]) == 0
    assert seen_variants == ["cold", "warm", "warm", "warm"]
    out = capsys.readouterr().out
    assert "cold" in out and "warm" in out
    assert "p50 (s)" in out


def test_main_boot_variant_appends_boot_run(
    script: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    variants: list[str] = []

    def fake_probe(command: list[str], *, variant: str, timeout_seconds: float) -> BenchmarkSample:
        variants.append(variant)
        return _ok(variant)

    monkeypatch.setattr(script, "run_child_probe", fake_probe)
    assert script.main(["--samples", "1", "--boot"]) == 0
    assert variants == ["cold", "warm", "boot"]


def test_main_writes_json_and_fails_on_failed_probe(
    script: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    state = {"calls": 0}

    def fake_probe(command: list[str], *, variant: str, timeout_seconds: float) -> BenchmarkSample:
        state["calls"] += 1
        if state["calls"] == 2:
            return BenchmarkSample(variant=variant, wall_seconds=1.0, exit_code=1)
        return _ok(variant)

    monkeypatch.setattr(script, "run_child_probe", fake_probe)
    out_path = tmp_path / "report.json"
    ret = script.main(["--samples", "1", "--json", str(out_path)])
    assert ret == 1
    assert "1 probe(s) failed" in capsys.readouterr().err
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    samples: list[dict[str, Any]] = payload["samples"]
    assert len(samples) == 2
    assert samples[1]["exit_code"] == 1
    assert [run["variant"] for run in payload["runs"]] == ["cold", "warm"]

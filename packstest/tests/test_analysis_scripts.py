"""Tests for the analysis scripts with no other coverage.

``cache_report.py`` runs over a generated telemetry log and
``check_direct_io_policy`` over a temp tree with clean and banned
callsites — synthetically, without touching the developer's real
``~/.lisa``. The scripts are loaded by path: ``scripts/`` is not an
importable package in this workspace.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str) -> ModuleType:
    path = _REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ── cache_report ─────────────────────────────────────────────────────


def _telemetry_row(emitted_at: str, *, prompt: int, cached: int, mode: str = "m") -> str:
    return json.dumps(
        {
            "call_site": "turn_loop",
            "emitted_at": emitted_at,
            "prompt_tokens": prompt,
            "cached_tokens": cached,
            "mode_id": mode,
        }
    )


def test_cache_report_classifies_and_prints_summary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    log = tmp_path / "token-usage.jsonl"
    rows = [
        _telemetry_row("2026-11-11T10:00:00", prompt=1000, cached=0),
        # 30s later: inside the gap, prompt shrank, nothing cached -> trim.
        _telemetry_row("2026-11-11T10:00:30", prompt=500, cached=0),
        # Far later: beyond the gap -> first.
        _telemetry_row("2026-11-11T10:10:00", prompt=2000, cached=2000),
    ]
    log.write_text("\n".join(rows) + "\n", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["cache_report.py", "--log", str(log)])
    module = _load_script("cache_report")
    module.main()
    out = capsys.readouterr().out
    assert "calls=3" in out
    assert "first" in out and "trim" in out


# ── check_direct_io_policy ───────────────────────────────────────────


def test_direct_io_policy_flags_unallowlisted_write_and_passes_clean(
    tmp_path: Path,
) -> None:
    policy = _load_script("check_direct_io_policy")
    clean = tmp_path / "clean.py"
    clean.write_text("value = configure(payload)\n", encoding="utf-8")
    assert policy._check_file(clean, "clean.py", scan_reads=False, allowlist_enabled=True) == []

    dirty = tmp_path / "dirty.py"
    dirty.write_text("path.write_text(payload)\n", encoding="utf-8")
    violations = policy._check_file(dirty, "dirty.py", scan_reads=False, allowlist_enabled=True)
    assert len(violations) == 1
    rel_path, line_no, snippet = violations[0]
    assert rel_path == "dirty.py"
    assert line_no == 1
    assert "write_text" in snippet


def test_direct_io_policy_allowlist_carve_out(tmp_path: Path) -> None:
    policy = _load_script("check_direct_io_policy")
    rel_path, snippets = next(iter(policy._ALLOWLIST.items()))
    carve_out = tmp_path / "carve.py"
    carve_out.write_text(snippets[0] + "\n", encoding="utf-8")
    assert policy._check_file(carve_out, rel_path, scan_reads=False, allowlist_enabled=True) == []

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from sdk.logging_setup import install_log_excepthook


@pytest.mark.parametrize("install_count", [1, 2])
def test_uncaught_exception_is_printed_and_logged(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    install_count: int,
) -> None:
    monkeypatch.setattr(sys, "excepthook", sys.__excepthook__)
    for _ in range(install_count):
        install_log_excepthook()
    hook = sys.excepthook

    with caplog.at_level(logging.CRITICAL, logger="lisa.excepthook"):
        try:
            raise RuntimeError("startup failed")
        except RuntimeError as error:
            hook(type(error), error, error.__traceback__)

    stderr = capsys.readouterr().err
    assert "Traceback (most recent call last):" in stderr
    assert stderr.count("RuntimeError: startup failed") == 1
    assert "excepthook: uncaught RuntimeError: startup failed" in caplog.text
    assert len(caplog.records) == 1
    assert caplog.records[0].exc_info is not None


def test_keyboard_interrupt_preserves_previous_hook(
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous = Mock()
    monkeypatch.setattr(sys, "excepthook", previous)
    install_log_excepthook()
    error = KeyboardInterrupt()

    sys.excepthook(type(error), error, None)

    previous.assert_called_once_with(KeyboardInterrupt, error, None)
    assert capsys.readouterr().err == ""
    assert caplog.records == []


def test_cli_crash_is_visible_with_stderr_logging_disabled(tmp_path: Path) -> None:
    log_path = tmp_path / "runtime.jsonl"
    script = """
import sys
from pathlib import Path
from sdk.logging_setup import configure_logging

configure_logging(log_path=Path(sys.argv[1]), level="info", host_kind="cli")
raise RuntimeError("binary connector startup failed")
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(log_path)],
        env={
            **os.environ,
            "LISA_LOG_STDERR": "0",
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert "Traceback (most recent call last):" in result.stderr
    assert "RuntimeError: binary connector startup failed" in result.stderr
    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert any(
        record["level"] == "critical"
        and "RuntimeError: binary connector startup failed" in record["msg"]
        for record in records
    )

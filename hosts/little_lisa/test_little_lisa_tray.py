from __future__ import annotations

import builtins
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Self

import pytest

from . import tray


class _FakeProcess:
    def __init__(self, *, wait_times_out: bool = False) -> None:
        self.wait_times_out = wait_times_out
        self.terminate_calls = 0
        self.wait_calls = 0
        self.kill_calls = 0
        self._running = True

    def poll(self) -> int | None:
        return None if self._running else 0

    def terminate(self) -> None:
        self.terminate_calls += 1
        self._running = False

    def wait(self, timeout: float) -> None:
        self.wait_calls += 1
        if self.wait_times_out:
            raise subprocess.TimeoutExpired(cmd="little_lisa", timeout=timeout)

    def kill(self) -> None:
        self.kill_calls += 1
        self._running = False


def _tray_config(tmp_path: Path, *, open_on_start: bool = False) -> tray.TrayConfig:
    return tray.TrayConfig(
        host="127.0.0.1",
        port=8765,
        token="secret",
        desktop_args=["desktop", "--port", "8765"],
        open_on_start=open_on_start,
        logs_dir=tmp_path / "logs",
    )


def test_desktop_subprocess_command_for_module_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.setattr(sys, "executable", "C:/Python312/python.exe")

    command = tray._desktop_subprocess_command(["desktop", "--port", "8765"])

    assert command == [
        "C:/Python312/python.exe",
        "-m",
        "little_lisa.main",
        "desktop",
        "--port",
        "8765",
    ]


def test_desktop_subprocess_command_for_frozen_executable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", "C:/dist/little_lisa.exe")

    command = tray._desktop_subprocess_command(["desktop", "--port", "9000"])

    assert command == ["C:/dist/little_lisa.exe", "desktop", "--port", "9000"]


def test_desktop_process_start_and_stop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _tray_config(tmp_path)
    fake_process = _FakeProcess()
    popen_calls: list[list[str]] = []
    wait_calls: list[tuple[str, int]] = []

    def _fake_popen(command: list[str]) -> _FakeProcess:
        popen_calls.append(command)
        return fake_process

    def _fake_wait_for_health(host: str, port: int, _timeout_seconds: float = 10.0) -> None:
        wait_calls.append((host, port))

    monkeypatch.setattr(tray.subprocess, "Popen", _fake_popen)
    monkeypatch.setattr(tray, "_wait_for_health", _fake_wait_for_health)

    manager = tray._DesktopProcess(config)
    manager.start()
    manager.start()

    assert len(popen_calls) == 1
    assert wait_calls == [("127.0.0.1", 8765)]
    assert manager.launch_url == "http://127.0.0.1:8765/?token=secret"

    manager.stop()
    assert fake_process.terminate_calls == 1
    assert fake_process.wait_calls == 1


def test_desktop_process_stop_kills_after_timeout(tmp_path: Path) -> None:
    config = _tray_config(tmp_path)
    manager = tray._DesktopProcess(config)
    fake_process = _FakeProcess(wait_times_out=True)
    manager._process = fake_process

    manager.stop()

    assert fake_process.terminate_calls == 1
    assert fake_process.wait_calls == 1
    assert fake_process.kill_calls == 1


def test_desktop_process_stop_noops_when_process_is_missing(tmp_path: Path) -> None:
    manager = tray._DesktopProcess(_tray_config(tmp_path))

    manager.stop()


def test_desktop_process_stop_clears_finished_process(tmp_path: Path) -> None:
    manager = tray._DesktopProcess(_tray_config(tmp_path))
    fake_process = _FakeProcess()
    fake_process._running = False
    manager._process = fake_process

    manager.stop()

    assert manager._process is None
    assert fake_process.terminate_calls == 0


def test_desktop_process_restart_invokes_stop_then_start(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manager = tray._DesktopProcess(_tray_config(tmp_path))
    events: list[str] = []

    monkeypatch.setattr(manager, "stop", lambda: events.append("stop"))
    monkeypatch.setattr(manager, "start", lambda: events.append("start"))

    manager.restart()

    assert events == ["stop", "start"]


def test_wait_for_health_returns_after_success(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class _FakeResponse:
        status_code = 200

    class _FakeClient:
        def __init__(self, timeout: float) -> None:
            self.timeout = timeout

        def __enter__(self) -> Self:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            return False

        def get(self, url: str) -> _FakeResponse:
            calls.append(url)
            return _FakeResponse()

    monkeypatch.setattr(tray.httpx, "Client", _FakeClient)

    tray._wait_for_health("127.0.0.1", 8765, timeout_seconds=0.5)

    assert calls == ["http://127.0.0.1:8765/health"]


def test_wait_for_health_retries_after_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    sleep_calls: list[float] = []

    class _FakeResponse:
        status_code = 200

    class _FakeClient:
        def __init__(self, timeout: float) -> None:
            self.timeout = timeout

        def __enter__(self) -> Self:
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
            return False

        def get(self, url: str) -> _FakeResponse:
            calls.append(url)
            if len(calls) == 1:
                msg = "transient"
                raise RuntimeError(msg)
            return _FakeResponse()

    monkeypatch.setattr(tray.httpx, "Client", _FakeClient)
    monkeypatch.setattr(tray.time, "sleep", sleep_calls.append)  # type: ignore[arg-type]

    tray._wait_for_health("127.0.0.1", 8765, timeout_seconds=0.5)

    assert calls == [
        "http://127.0.0.1:8765/health",
        "http://127.0.0.1:8765/health",
    ]
    assert sleep_calls == [0.2]


def test_open_logs_dir_uses_startfile_on_windows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    opened: list[Path] = []

    def _fake_startfile(path: Path) -> None:
        opened.append(path)

    monkeypatch.setattr(tray.os, "name", "nt", raising=False)
    monkeypatch.setattr(tray.os, "startfile", _fake_startfile, raising=False)

    logs_dir = tmp_path / "logs"
    tray._open_logs_dir(logs_dir)

    assert logs_dir.exists()
    assert opened == [logs_dir]


def test_open_logs_dir_uses_browser_on_non_windows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    opened: list[str] = []

    def _fake_open(url: str) -> bool:
        opened.append(url)
        return True

    monkeypatch.setattr(tray.os, "name", "posix", raising=False)
    monkeypatch.setattr(tray.webbrowser, "open", _fake_open)

    logs_dir = tmp_path / "logs"
    tray._open_logs_dir(logs_dir)

    assert logs_dir.exists()
    assert opened == [logs_dir.as_uri()]


def test_build_tray_icon_image_returns_expected_canvas_size() -> None:
    image = tray._build_tray_icon_image()

    assert getattr(image, "size", None) == (64, 64)


def test_run_tray_raises_when_pystray_import_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _tray_config(tmp_path)
    import_builtin = builtins.__import__

    def _fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "pystray":
            msg = "missing"
            raise ImportError(msg)
        return import_builtin(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "pystray", raising=False)
    monkeypatch.setattr(builtins, "__import__", _fake_import)

    with pytest.raises(RuntimeError, match="pystray is required for tray mode"):
        tray.run_tray(config)


def test_run_tray_starts_desktop_and_runs_icon(  # noqa: C901 — integration test with many setup branches
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _tray_config(tmp_path, open_on_start=True)
    desktop_events: list[str] = []
    opened_urls: list[str] = []
    opened_log_paths: list[Path] = []

    class _FakeDesktopProcess:
        def __init__(self, _config: tray.TrayConfig) -> None:
            self.launch_url = "http://127.0.0.1:8765/?token=secret"

        def start(self) -> None:
            desktop_events.append("start")

        def stop(self) -> None:
            desktop_events.append("stop")

        def restart(self) -> None:
            desktop_events.append("restart")

    fake_pystray = ModuleType("pystray")
    created_icons: list[object] = []

    class _FakeMenuItem:
        def __init__(self, label: str, callback: object) -> None:
            self.label = label
            self.callback = callback

    class _FakeMenu:
        def __init__(self, *items: object) -> None:
            self.items = items

    class _FakeIcon:
        def __init__(self, name: str, image: object, title: str, menu: object) -> None:
            self.name = name
            self.image = image
            self.title = title
            self.menu = menu
            self.ran = False
            self.stopped = False
            created_icons.append(self)

        def run(self) -> None:
            self.ran = True
            for item in self.menu.items:
                item.callback(self, None)

        def stop(self) -> None:
            self.stopped = True

    fake_pystray.MenuItem = _FakeMenuItem
    fake_pystray.Menu = _FakeMenu
    fake_pystray.Icon = _FakeIcon

    def _fake_open(url: str) -> bool:
        opened_urls.append(url)
        return True

    monkeypatch.setattr(tray, "_DesktopProcess", _FakeDesktopProcess)
    monkeypatch.setattr(tray, "_build_tray_icon_image", lambda: object())  # noqa: PLW0108
    monkeypatch.setattr(tray, "_open_logs_dir", opened_log_paths.append)  # type: ignore[arg-type]
    monkeypatch.setattr(tray.webbrowser, "open", _fake_open)
    monkeypatch.setitem(sys.modules, "pystray", fake_pystray)

    tray.run_tray(config)

    assert desktop_events == ["start", "start", "restart", "stop", "stop"]
    assert opened_urls == [
        "http://127.0.0.1:8765/?token=secret",
        "http://127.0.0.1:8765/?token=secret",
    ]
    assert opened_log_paths == [config.logs_dir]
    assert len(created_icons) == 1
    assert created_icons[0].ran is True
    assert created_icons[0].stopped is True

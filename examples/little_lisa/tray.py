from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
import webbrowser
from dataclasses import dataclass
from pathlib import Path

import httpx

try:
    from .desktop import desktop_url
except ImportError:  # pragma: no cover - direct script execution fallback
    from desktop import desktop_url  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TrayConfig:
    """Configuration for the Little Lisa system tray launcher."""

    host: str
    port: int
    token: str
    desktop_args: list[str]
    open_on_start: bool
    logs_dir: Path


class _DesktopProcess:
    """Own the desktop subprocess lifecycle for tray controls."""

    def __init__(self, config: TrayConfig) -> None:
        self._config = config
        self._process: subprocess.Popen[str] | None = None

    @property
    def launch_url(self) -> str:
        return desktop_url(host=self._config.host, port=self._config.port, token=self._config.token)

    def start(self) -> None:
        if self._process is not None and self._process.poll() is None:
            logger.debug("desktop process already running; skip start")
            return
        command = _desktop_subprocess_command(self._config.desktop_args)
        logger.info("starting desktop subprocess: %s", " ".join(command))
        self._process = subprocess.Popen(command)
        _wait_for_health(self._config.host, self._config.port)

    def stop(self) -> None:
        if self._process is None:
            logger.debug("desktop process already stopped; skip stop")
            return
        if self._process.poll() is not None:
            logger.debug("desktop process already exited; clearing handle")
            self._process = None
            return
        logger.info("stopping desktop subprocess")
        self._process.terminate()
        try:
            self._process.wait(timeout=4)
        except subprocess.TimeoutExpired:
            logger.warning("desktop subprocess did not exit within 4s; killing")
            self._process.kill()
        finally:
            self._process = None

    def restart(self) -> None:
        logger.info("restarting desktop subprocess")
        self.stop()
        self.start()


def _desktop_subprocess_command(desktop_args: list[str]) -> list[str]:
    if getattr(sys, "frozen", False):
        # PyInstaller one-file executable: relaunch the same binary with subcommand args.
        return [sys.executable, *desktop_args]
    return [sys.executable, "-m", "little_lisa.main", *desktop_args]


def _wait_for_health(host: str, port: int, timeout_seconds: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    url = f"http://{host}:{port}/health"
    while time.monotonic() < deadline:
        try:
            with httpx.Client(timeout=0.5) as client:
                response = client.get(url)
                if response.status_code == 200:
                    logger.debug("desktop health check passed at %s", url)
                    return
        except Exception:
            pass
        time.sleep(0.2)
    logger.error("desktop health check timed out after %.1fs at %s", timeout_seconds, url)


def _open_logs_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        os.startfile(path)  # type: ignore[attr-defined]
        return
    webbrowser.open(path.as_uri())


def _build_tray_icon_image() -> object:
    from PIL import Image, ImageDraw

    image = Image.new("RGBA", (64, 64), (20, 34, 52, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, 56, 56), outline=(84, 163, 255, 255), width=3)
    draw.rectangle((18, 20, 46, 44), fill=(84, 163, 255, 255))
    return image


def run_tray(config: TrayConfig) -> None:
    """Run the Little Lisa tray process and manage the desktop subprocess."""

    try:
        import pystray
    except Exception as exc:
        logger.error("pystray import failed: %s: %s", type(exc).__name__, exc)
        raise RuntimeError(
            "pystray is required for tray mode. Install 'pystray' and 'pillow'."
        ) from exc

    logger.info(
        "starting tray (host=%s port=%d open_on_start=%s)",
        config.host,
        config.port,
        config.open_on_start,
    )
    desktop = _DesktopProcess(config)
    desktop.start()
    if config.open_on_start:
        webbrowser.open(desktop.launch_url)

    def _open_now(_icon: object, _item: object) -> None:
        desktop.start()
        webbrowser.open(desktop.launch_url)

    def _restart(_icon: object, _item: object) -> None:
        desktop.restart()

    def _open_logs(_icon: object, _item: object) -> None:
        _open_logs_dir(config.logs_dir)

    def _quit(icon: object, _item: object) -> None:
        desktop.stop()
        icon.stop()

    icon = pystray.Icon(
        "little-lisa",
        _build_tray_icon_image(),
        "Little Lisa",
        menu=pystray.Menu(
            pystray.MenuItem("Open Little Lisa", _open_now),
            pystray.MenuItem("Restart", _restart),
            pystray.MenuItem("Open Logs", _open_logs),
            pystray.MenuItem("Quit", _quit),
        ),
    )

    try:
        icon.run()
    finally:
        desktop.stop()
        logger.info("tray stopped; desktop subprocess cleaned up")

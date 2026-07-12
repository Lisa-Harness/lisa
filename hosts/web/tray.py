"""System tray launcher for the localhost web host.

This is the web-host adaptation of ``hosts/little_lisa/tray.py``. It manages
the ``lisa-web desktop`` subprocess lifecycle and renders a system-tray
icon with Open / Restart / Open Logs / Quit controls.

``pystray`` and ``pillow`` are optional dependencies. Import this module
only when the user explicitly requests tray mode; the web host itself does
not require them.
"""

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
    from .server import web_url
except ImportError:  # pragma: no cover — direct script execution fallback
    from server import web_url  # type: ignore[no-redef]

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TrayConfig:
    """Configuration for the Lisa web host system tray launcher."""

    host: str
    port: int
    token: str
    web_args: list[str]
    open_on_start: bool
    logs_dir: Path


class _WebHostProcess:
    """Own the web host subprocess lifecycle for tray controls."""

    def __init__(self, config: TrayConfig) -> None:
        self._config = config
        self._process: subprocess.Popen[str] | None = None

    @property
    def launch_url(self) -> str:
        return web_url(host=self._config.host, port=self._config.port, token=self._config.token)

    def start(self) -> None:
        if self._process is not None and self._process.poll() is None:
            logger.debug("web host process already running; skip start")
            return
        command = _web_subprocess_command(self._config.web_args)
        logger.info("starting web host subprocess: %s", " ".join(command))
        self._process = subprocess.Popen(command)
        _wait_for_health(self._config.host, self._config.port)

    def stop(self) -> None:
        if self._process is None:
            logger.debug("web host process already stopped; skip stop")
            return
        if self._process.poll() is not None:
            logger.debug("web host process already exited; clearing handle")
            self._process = None
            return
        logger.info("stopping web host subprocess")
        self._process.terminate()
        try:
            self._process.wait(timeout=4)
        except subprocess.TimeoutExpired:
            logger.warning("web host subprocess did not exit within 4s; killing")
            self._process.kill()
        finally:
            self._process = None

    def restart(self) -> None:
        logger.info("restarting web host subprocess")
        self.stop()
        self.start()


def _web_subprocess_command(web_args: list[str]) -> list[str]:
    if getattr(sys, "frozen", False):
        # PyInstaller one-file executable: relaunch the same binary with subcommand args.
        return [sys.executable, *web_args]
    return [sys.executable, "-m", "hosts.web.main", *web_args]


def _wait_for_health(host: str, port: int, timeout_seconds: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    url = f"http://{host}:{port}/health"
    while time.monotonic() < deadline:
        try:
            with httpx.Client(timeout=0.5) as client:
                response = client.get(url)
                if response.status_code == httpx.codes.OK:
                    logger.debug("web host health check passed at %s", url)
                    return
        except Exception:
            pass
        time.sleep(0.2)
    logger.error("web host health check timed out after %.1fs at %s", timeout_seconds, url)


def _open_logs_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        os.startfile(path)  # type: ignore[attr-defined]
        return
    webbrowser.open(path.as_uri())


def _build_tray_icon_image() -> object:
    from PIL import Image, ImageDraw  # noqa: PLC0415 — optional dep, deferred import

    image = Image.new("RGBA", (64, 64), (20, 34, 52, 255))
    draw = ImageDraw.Draw(image)
    draw.rectangle((8, 8, 56, 56), outline=(84, 163, 255, 255), width=3)
    draw.rectangle((18, 20, 46, 44), fill=(84, 163, 255, 255))
    return image


def run_tray(config: TrayConfig) -> None:
    """Run the tray process and manage the web host subprocess."""

    try:
        import pystray  # noqa: PLC0415 — optional dep, deferred import
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
    web_proc = _WebHostProcess(config)
    web_proc.start()
    if config.open_on_start:
        webbrowser.open(web_proc.launch_url)

    def _open_now(_icon: object, _item: object) -> None:
        web_proc.start()
        webbrowser.open(web_proc.launch_url)

    def _restart(_icon: object, _item: object) -> None:
        web_proc.restart()

    def _open_logs(_icon: object, _item: object) -> None:
        _open_logs_dir(config.logs_dir)

    def _quit(icon: object, _item: object) -> None:
        web_proc.stop()
        icon.stop()

    icon = pystray.Icon(
        "lisa-web",
        _build_tray_icon_image(),
        "Lisa Web",
        menu=pystray.Menu(
            pystray.MenuItem("Open Lisa Web", _open_now),
            pystray.MenuItem("Restart", _restart),
            pystray.MenuItem("Open Logs", _open_logs),
            pystray.MenuItem("Quit", _quit),
        ),
    )

    try:
        icon.run()
    finally:
        web_proc.stop()
        logger.info("tray stopped; web host subprocess cleaned up")


__all__ = ["TrayConfig", "run_tray"]
"""hosts.vscode.bridge — stdio JSON-RPC bridge for the VS Code extension host.

This is a standalone subprocess that the VS Code extension spawns. It:

1. Reads JSON-RPC messages from stdin (one per line).
2. Builds a ``HostRuntime`` from ``HostConfig.from_env(...)`` on the first
   message.
3. Calls ``setup_framework(runtime)`` and ``compile_mcp_configs(runtime)``.
4. Starts the MCP pool with ``await runtime.start()``.
5. For each ``"turn"`` message, calls
   ``hosts.core.handle_message(runtime, message, transport)``.
6. Writes the response as JSON to stdout.
7. On shutdown (EOF or ``"shutdown"`` method), calls
   ``await runtime.aclose()``.

Protocol
--------
All messages are single-line JSON objects, one per line.

Request (stdin)::

    {"id": 1, "method": "turn", "params": {"message": "hello"}}

Response (stdout, matches request ``id``)::

    {"id": 1, "result": {"text": "...", "data": null, "is_error": false}}

Stage notification (stdout, during a turn, no ``id``)::

    {"method": "notify", "params": {"stage": "frontdoor", "data": {...}}}

Error notification (stdout, non-fatal, no ``id``)::

    {"method": "error", "params": {"message": "..."}}

Error response (stdout, for a failed request)::

    {"id": 1, "error": {"code": -32601, "message": "Method not found"}}

Logging goes to ``<data-dir>/vscode-bridge.log`` — never to stderr.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

from hosts.core.config import HostConfig
from hosts.core.handle_message import handle_message
from hosts.core.mcp_pool import compile_mcp_configs, configure_runtime, setup_framework
from hosts.core.runtime import HostRuntime

logger = logging.getLogger(__name__)

# ── JSON-RPC error codes ──────────────────────────────────────────────

_ERR_PARSE_ERROR = -32700
_ERR_INVALID_REQUEST = -32600
_ERR_METHOD_NOT_FOUND = -32601
_ERR_INTERNAL = -32603


# ── Transport ─────────────────────────────────────────────────────────


class VSCodeTransport:
    """``TransportCallbacks`` implementation for the VS Code stdio bridge.

    - ``on_response`` stores text + data for the JSON-RPC response.
    - ``on_stage`` writes a notification JSON line to stdout immediately.
    - ``on_error`` writes an error notification JSON line to stdout
      immediately.
    - Other callbacks (``on_text_chunk``, ``on_debug_context``) are
      no-ops — v1 does not stream text or render debug context.

    The ``write_output`` callable must be thread-safe with respect to
    the event loop. In practice all callbacks fire inside the same
    event loop task that is awaiting ``handle_message``, so there is no
    concurrent access.
    """

    def __init__(self, write_output: Any) -> None:
        self._write_output = write_output
        self.response_text: str = ""
        self.response_data: dict[str, Any] | None = None

    async def on_stage(self, label: str, data: dict[str, Any] | None = None) -> None:
        self._write_output(
            {"method": "notify", "params": {"stage": label, "data": data}}
        )

    async def on_text_chunk(self, text: str) -> None:
        pass

    async def on_response(self, text: str, data: dict[str, Any] | None = None) -> None:
        self.response_text = text
        self.response_data = data

    async def on_debug_context(self, context: dict[str, Any]) -> None:
        pass

    async def on_error(self, message: str, exc: Exception | None = None) -> None:
        self._write_output({"method": "error", "params": {"message": message}})


# ── Runtime initialization ────────────────────────────────────────────


def _initialize_runtime(data_dir: str) -> HostRuntime:
    """Build and wire a ``HostRuntime`` for the VS Code bridge.

    Mirrors ``hosts.web.main._initialize_runtime`` but without Typer —
    the bridge is a headless subprocess. The MCP pool is NOT started
    here; ``runtime.start()`` runs inside the bridge's event loop (see
    :func:`_run_bridge`).
    """

    config = HostConfig.from_env(
        data_dir=data_dir,
        user_slug="vscode",
        session_context="vscode",
        surface_type="vscode",
        surface_id="vscode-chat",
    )
    _configure_logging(config)
    runtime = HostRuntime(config)
    setup_framework(runtime)
    configure_runtime(runtime)
    warm, cold = compile_mcp_configs(runtime)
    runtime.set_mcp_configs(warm, cold)
    return runtime


def _configure_logging(config: HostConfig) -> None:
    """Configure stdlib root logging to a file (never stderr).

    stderr is reserved for the VS Code extension's diagnostic capture;
    the JSON-RPC protocol flows over stdin/stdout. Logging goes to
    ``<data-dir>/vscode-bridge.log``.
    """

    log_path = config.data_dir / "vscode-bridge.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    level_name = config.effective_log_level.upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    )
    root.addHandler(handler)


# ── Stdio I/O helpers ──────────────────────────────────────────────────


def _make_writer() -> Any:
    """Return a callable that writes one JSON object as a line to stdout."""

    def write_output(obj: dict[str, Any]) -> None:
        sys.stdout.write(json.dumps(obj, default=str) + "\n")
        sys.stdout.flush()

    return write_output


async def _read_stdin_line() -> str | None:
    """Read one line from stdin asynchronously.

    Uses a thread executor because Windows stdin does not support
    ``loop.connect_read_pipe``. Returns ``None`` on EOF.
    """

    loop = asyncio.get_running_loop()
    line = await loop.run_in_executor(None, sys.stdin.readline)
    if not line:
        return None
    return line.strip()


# ── Main bridge loop ───────────────────────────────────────────────────


async def _run_bridge(data_dir: str) -> None:
    """Run the stdio JSON-RPC bridge until EOF or shutdown.

    Lifecycle:
    - The ``HostRuntime`` is lazily initialized on the first message.
    - ``runtime.start()`` (MCP pool) runs in this event loop.
    - ``runtime.aclose()`` runs in the ``finally`` block (same loop).
    - This satisfies the anyio cross-loop rule: start and close in the
      same loop that executes turns.
    """

    runtime: HostRuntime | None = None
    write_output = _make_writer()

    try:
        while True:
            line = await _read_stdin_line()
            if line is None:
                logger.info("bridge: stdin EOF; shutting down")
                break

            if not line:
                continue

            # ── Parse JSON-RPC message ───────────────────────────────
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                write_output(
                    {
                        "error": {
                            "code": _ERR_PARSE_ERROR,
                            "message": "Parse error",
                        }
                    }
                )
                continue

            if not isinstance(msg, dict):
                write_output(
                    {
                        "error": {
                            "code": _ERR_INVALID_REQUEST,
                            "message": "Invalid request: expected JSON object",
                        }
                    }
                )
                continue

            msg_id = msg.get("id")
            method = str(msg.get("method", "")).strip()
            params = msg.get("params", {})
            if not isinstance(params, dict):
                params = {}

            # ── Lazy runtime init on first message ───────────────────
            if runtime is None:
                try:
                    logger.info("bridge: initializing runtime (data_dir=%s)", data_dir)
                    runtime = _initialize_runtime(data_dir)
                    await runtime.start()
                    logger.info("bridge: runtime started")
                except Exception as exc:  # pylint: disable=broad-except
                    logger.exception("bridge: startup failed")
                    write_output(
                        {
                            "id": msg_id,
                            "error": {
                                "code": _ERR_INTERNAL,
                                "message": f"Startup failed: {type(exc).__name__}: {exc}",
                            },
                        }
                    )
                    runtime = None
                    continue

            # ── Dispatch ──────────────────────────────────────────────
            if method == "shutdown":
                logger.info("bridge: shutdown requested")
                write_output({"id": msg_id, "result": {"ok": True}})
                break

            if method == "turn":
                message = str(params.get("message", "")).strip()
                if not message:
                    write_output(
                        {
                            "id": msg_id,
                            "error": {
                                "code": _ERR_INVALID_REQUEST,
                                "message": "params.message is required",
                            },
                        }
                    )
                    continue

                transport = VSCodeTransport(write_output)
                try:
                    result = await handle_message(runtime, message, transport)
                except Exception as exc:  # pylint: disable=broad-except
                    logger.exception("bridge: turn failed")
                    write_output(
                        {
                            "id": msg_id,
                            "error": {
                                "code": _ERR_INTERNAL,
                                "message": f"{type(exc).__name__}: {exc}",
                            },
                        }
                    )
                    continue

                write_output(
                    {
                        "id": msg_id,
                        "result": {
                            "text": transport.response_text or result.text,
                            "data": transport.response_data,
                            "is_error": result.is_error,
                        },
                    }
                )
            else:
                write_output(
                    {
                        "id": msg_id,
                        "error": {
                            "code": _ERR_METHOD_NOT_FOUND,
                            "message": f"Method not found: {method!r}",
                        },
                    }
                )
    finally:
        if runtime is not None:
            logger.info("bridge: closing runtime")
            await runtime.aclose()
            logger.info("bridge: runtime closed")


# ── Entry point ────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point for the VS Code extension stdio bridge.

    Usage::

        python -m hosts.vscode.bridge --data-dir <path>

    The bridge reads JSON-RPC messages from stdin and writes responses
    to stdout, one JSON object per line. Logging goes to
    ``<data-dir>/vscode-bridge.log``.
    """

    parser = argparse.ArgumentParser(
        description="VS Code extension stdio bridge for the Lisa framework.",
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Data root for artifacts, memory, and logs.",
    )
    args = parser.parse_args()

    data_dir = str(Path(args.data_dir).expanduser().resolve())

    try:
        asyncio.run(_run_bridge(data_dir))
    except KeyboardInterrupt:
        logger.info("bridge: interrupted by user")


if __name__ == "__main__":
    main()
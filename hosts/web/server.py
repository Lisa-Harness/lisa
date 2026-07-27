"""aiohttp web server for the localhost Lisa web host.

This is the web-transport adaptation of the former
``hosts/little_lisa/desktop.py``. It wraps the transport-neutral
:mod:`hosts.core` runtime with an aiohttp web server that exposes:

- ``GET /`` — the desktop chat page (inline HTML)
- ``POST /api/turn`` — execute one turn via ``hosts.core.handle_message``
- ``GET /api/context`` — last turn diagnostic context (JSON)
- ``GET /health`` — liveness probe
- ``POST /api/heartbeat`` — client keepalive ping (drives the watchdog)
- ``POST /api/quit`` — client signals tab/window close (beacon)

The runtime lifecycle is bound to the server's event loop:
``runtime.start()`` runs in ``on_startup`` and ``runtime.aclose()`` runs
in ``on_cleanup`` — both inside the aiohttp event loop, satisfying the
anyio cross-loop rule.

Single-tab lifecycle: the chat page pings ``/api/heartbeat`` every few
seconds and sends a ``/api/quit`` beacon on ``pagehide``/``beforeunload``.
A server-side watchdog stops the server (and closes the runtime) when no
heartbeat or turn activity arrives within ``_HEARTBEAT_TIMEOUT_SECONDS``.
This makes ``lisa web`` exit cleanly when the user closes the browser tab.
"""

from __future__ import annotations

import asyncio
import html
import json
import logging
import secrets
import webbrowser
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from aiohttp import web

from hosts.core.handle_message import handle_message
from hosts.core.runtime import HostRuntime, TurnContext

logger = logging.getLogger(__name__)

DEFAULT_WEB_HOST = "127.0.0.1"
DEFAULT_WEB_PORT = 8765
_DEFAULT_TOKEN_BYTES = 16

# Heartbeat watchdog. The chat page pings /api/heartbeat every
# _HEARTBEAT_INTERVAL_SECONDS; if the server sees no activity (heartbeat,
# turn, or context fetch) for this many seconds, it shuts down cleanly.
# Generous enough to survive a brief network stall, tight enough that
# closing the tab exits the process within ~15s.
_HEARTBEAT_INTERVAL_SECONDS = 5.0
_HEARTBEAT_TIMEOUT_SECONDS = 15.0


def generate_web_token() -> str:
    """Return a URL-safe localhost token for web API requests."""

    return secrets.token_urlsafe(_DEFAULT_TOKEN_BYTES)


def web_url(*, host: str, port: int, token: str) -> str:
    """Return the browser URL for one web host session."""

    return f"http://{host}:{port}/?token={token}"


def build_web_index_html(*, token: str) -> str:
    """Build the lightweight local web chat page.

    CSS and JS are served from ``/static/`` so they are editable and
    cacheable. The token is injected as ``window.LISA_TOKEN`` so the
    JS can read it without templating.
    """

    safe_token = html.escape(token, quote=True)
    return f"""<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Lisa Web</title>
    <link rel=\"stylesheet\" href=\"/static/style.css\" />
  </head>
  <body>
    <main class=\"shell\">
      <section class=\"header\">
        <div>
          <h1 class=\"title\">Lisa Web</h1>
          <p class=\"subtitle\">Localhost web host for the Lisa framework.</p>
        </div>
        <div class=\"badge\" id=\"status\">ready</div>
      </section>

      <div class=\"stage-bar ready\" id=\"stage-bar\"></div>

      <section class=\"panel\">
        <div class=\"messages\" id=\"messages\"></div>
        <div class=\"composer\">
          <textarea id=\"prompt\" placeholder=\"Ask Lisa...\"></textarea>
          <div class=\"actions\">
            <span class=\"subtitle\">Ctrl+Enter to send</span>
            <button id=\"send\">Send</button>
          </div>
        </div>
      </section>

      <div class=\"token-hud\" id=\"token-hud\"></div>

      <details>
        <summary>Last turn context</summary>
        <section class=\"context\"><pre id=\"context\">(none)</pre></section>
      </details>
    </main>

    <script>window.LISA_TOKEN = \"{safe_token}\";</script>
    <script src=\"/static/app.js\"></script>
  </body>
</html>
"""


# ── Token helpers ──────────────────────────────────────────────────────


def _token_from_request(request: web.Request) -> str:
    header_token = str(request.headers.get("X-Lisa-Token") or "").strip()
    if header_token:
        return header_token
    return str(request.query.get("token") or "").strip()


def _is_token_valid(request: web.Request, expected_token: str) -> bool:
    incoming = _token_from_request(request)
    return bool(incoming) and secrets.compare_digest(incoming, expected_token)


def _unauthorized_response(
    request: web.Request, endpoint: str, response: web.Response
) -> web.Response:
    """Log and return an unauthorized response for a web endpoint."""

    logger.warning("web %s unauthorized request from %s", endpoint, request.remote)
    return response


# ── TurnContext serialization ──────────────────────────────────────────


def _turn_context_to_dict(ctx: TurnContext | None) -> dict[str, Any]:
    """Serialize a ``TurnContext`` dataclass to a plain dict for JSON."""

    if ctx is None:
        return {}
    if is_dataclass(ctx):
        return asdict(ctx)
    # Fallback for non-dataclass shapes (defensive).
    return getattr(ctx, "__dict__", {})


# ── WebTransport ───────────────────────────────────────────────────────


class WebTransport:
    """Minimal ``TransportCallbacks`` implementation for the web host.

    The web host (v1) does not stream. It collects the final response
    text from ``on_response`` and ignores stage updates, text chunks,
    and debug context. Errors are logged; the HTTP handler reads
    ``error`` to surface failures.

    Kept for backward compatibility with ``POST /api/turn``.
    """

    def __init__(self) -> None:
        self.response_text: str = ""
        self.error: str | None = None

    async def on_stage(self, label: str, data: dict[str, Any] | None = None) -> None:
        pass

    async def on_text_chunk(self, text: str) -> None:
        pass

    async def on_response(self, text: str, data: dict[str, Any] | None = None) -> None:
        self.response_text = text

    async def on_debug_context(self, context: dict[str, Any]) -> None:
        pass

    async def on_error(self, message: str, exc: Exception | None = None) -> None:
        self.error = message
        logger.error("web transport error: %s", message)


# ── SSETransport ────────────────────────────────────────────────────────


def render_markdown_to_html(text: str) -> str:
    """Render Markdown text to self-contained HTML using Rich.

    ``rich>=13.7`` is already a dependency of the web host. This produces
    inline-styled HTML with syntax-highlighted code blocks, tables, and
    formatted text — no CDN, no client-side markdown library, no build step.
    """

    import io

    from rich.console import Console
    from rich.markdown import Markdown

    console = Console(record=True, file=io.StringIO(), width=120)
    console.print(Markdown(text))
    return console.export_html(inline_styles=True)


class SSETransport:
    """``TransportCallbacks`` implementation that pushes events to an SSE queue.

    Each callback pushes a dict ``{"event": <type>, "data": {...}}`` onto
    an ``asyncio.Queue``. The SSE endpoint reads from the queue and writes
    ``text/event-stream`` frames to the client.

    This mirrors the VS Code host's ``VSCodeTransport`` pattern (the
    ``lisa_vscode.host.bridge`` module in the sibling ``lisa_vscode``
    repo) — forward every kernel event to the transport surface — but
    uses SSE frames instead of JSON-RPC notifications.
    """

    def __init__(self, queue: asyncio.Queue[dict[str, Any]], *, debug: bool = False) -> None:
        self._queue = queue
        self._debug = debug

    async def on_stage(self, label: str, data: dict[str, Any] | None = None) -> None:
        payload: dict[str, Any] = {"label": label}
        if data:
            payload.update(data)
        await self._queue.put({"event": "stage", "data": payload})

    async def on_text_chunk(self, text: str) -> None:
        await self._queue.put({"event": "token", "data": {"text": text}})

    async def on_response(self, text: str, data: dict[str, Any] | None = None) -> None:
        html_content = render_markdown_to_html(text)
        await self._queue.put(
            {
                "event": "done",
                "data": {"text": text, "html": html_content, "data": data},
            }
        )

    async def on_debug_context(self, context: dict[str, Any]) -> None:
        if self._debug:
            await self._queue.put({"event": "debug", "data": context})

    async def on_error(self, message: str, exc: Exception | None = None) -> None:
        logger.error("web transport error: %s", message)
        await self._queue.put({"event": "error", "data": {"message": message}})


# ── aiohttp app factory ─────────────────────────────────────────────────


def _build_web_app(*, runtime: HostRuntime, token: str) -> web.Application:
    """Build the aiohttp application with routes wired to the runtime."""

    app = web.Application()
    app["runtime"] = runtime
    app["web_token"] = token
    app["turn_lock"] = asyncio.Lock()
    app["last_activity"] = 0.0
    app["watchdog_trigger"] = asyncio.Event()
    app["watchdog_task"] = None

    # ── Lifecycle hooks ─────────────────────────────────────────────
    async def _on_startup(_app: web.Application) -> None:
        rt: HostRuntime = _app["runtime"]
        logger.info("web on_startup: starting runtime")
        await rt.start()
        # Start the heartbeat watchdog now that the loop is live. It stops
        # the server when the browser tab closes (no heartbeat within the
        # timeout window).
        _app["last_activity"] = _app.loop.time() if _app.loop else 0.0
        _app["watchdog_task"] = asyncio.create_task(_heartbeat_watchdog(_app))

    async def _on_cleanup(_app: web.Application) -> None:
        # Cancel the watchdog first so it does not race the runtime teardown.
        task = _app["watchdog_task"]
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        rt: HostRuntime = _app["runtime"]
        logger.info("web on_cleanup: closing runtime")
        await rt.aclose()

    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)

    # ── Token auth middleware ────────────────────────────────────────
    @web.middleware
    async def _token_middleware(
        request: web.Request, handler: Any
    ) -> web.StreamResponse:
        # Skip auth for /health and static files.
        if request.path == "/health" or request.path.startswith("/static/"):
            return await handler(request)
        expected = str(request.app["web_token"])
        if not _is_token_valid(request, expected):
            logger.warning(
                "web %s unauthorized request from %s", request.path, request.remote
            )
            raise web.HTTPUnauthorized(text="Unauthorized")
        return await handler(request)

    app.middlewares.append(_token_middleware)

    # ── Route handlers ───────────────────────────────────────────────
    async def _index(request: web.Request) -> web.Response:
        expected_token = str(request.app["web_token"])
        html_page = build_web_index_html(token=expected_token)
        return web.Response(text=html_page, content_type="text/html")

    async def _health(_request: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    async def _context(request: web.Request) -> web.Response:
        _mark_activity(request.app)
        rt: HostRuntime = request.app["runtime"]
        context = _turn_context_to_dict(rt.last_turn_context)
        return web.json_response(context)

    async def _heartbeat(request: web.Request) -> web.Response:
        _mark_activity(request.app)
        return web.json_response({"ok": True})

    async def _quit(request: web.Request) -> web.Response:
        logger.info("web quit beacon received, shutting down")
        # Set the trigger; _run_server_until_quit is awaiting it and will
        # run runner.cleanup() (which fires on_cleanup → runtime.aclose).
        # We do NOT call app.shutdown() here — that would race the runner
        # teardown. Respond 200 so the beacon completes first.
        request.app["watchdog_trigger"].set()
        return web.json_response({"ok": True})

    async def _turn(request: web.Request) -> web.Response:
        return await _handle_turn_request(request)

    async def _turn_stream(request: web.Request) -> web.StreamResponse:
        return await _handle_turn_stream_request(request)

    # ── Runtime API endpoints ─────────────────────────────────────
    async def _models(request: web.Request) -> web.Response:
        rt: HostRuntime = request.app["runtime"]
        manifest = rt.manifest
        if manifest is None:
            return web.json_response({"models": [], "error": "no manifest loaded"})
        return web.json_response(
            {
                "models": [
                    {
                        "id": m.id,
                        "provider": getattr(m, "provider", ""),
                        "designation": getattr(m, "designation", ""),
                    }
                    for m in manifest.models
                ]
            }
        )

    async def _tokens(request: web.Request) -> web.Response:
        rt: HostRuntime = request.app["runtime"]
        handler = rt.token_usage_handler
        recent = handler.read_recent(limit=200) if handler else []
        counters = rt.tokens
        return web.json_response(
            {
                "session": {
                    "prompt_tokens": counters.session_prompt_tokens,
                    "completion_tokens": counters.session_completion_tokens,
                    "total_tokens": counters.session_total_tokens,
                    "turn_count": counters.session_turn_count,
                    "success_count": counters.session_success_count,
                    "error_count": counters.session_error_count,
                    "model_token_totals": counters.model_token_totals,
                    "mode_token_totals": counters.mode_token_totals,
                },
                "turn": {
                    "prompt_tokens": counters.turn_prompt_tokens,
                    "completion_tokens": counters.turn_completion_tokens,
                    "total_tokens": counters.turn_total_tokens,
                },
                "recent_events": recent,
            }
        )

    async def _status(request: web.Request) -> web.Response:
        rt: HostRuntime = request.app["runtime"]
        manifest = rt.manifest
        return web.json_response(
            {
                "mode": rt.current_mode_id,
                "stage": rt.current_stage_id,
                "models_loaded": len(manifest.models) if manifest else 0,
                "mcp_started": rt.mcp_started,
                "restart_required": rt.restart_required,
            }
        )

    async def _schema(_request: web.Request) -> web.Response:
        from hosts.web.schemas import TurnRequest, TurnResponse

        return web.json_response(
            {
                "turn_request": TurnRequest.model_json_schema(),
                "turn_response": TurnResponse.model_json_schema(),
            }
        )

    app.router.add_get("/", _index)
    app.router.add_get("/health", _health)
    app.router.add_get("/api/context", _context)
    app.router.add_post("/api/turn", _turn)
    app.router.add_get("/api/turn/stream", _turn_stream)
    app.router.add_post("/api/heartbeat", _heartbeat)
    app.router.add_post("/api/quit", _quit)
    app.router.add_get("/api/models", _models)
    app.router.add_get("/api/tokens", _tokens)
    app.router.add_get("/api/status", _status)
    app.router.add_get("/api/schema", _schema)

    # ── Static files ────────────────────────────────────────────────
    static_dir = Path(__file__).parent / "static"
    if static_dir.is_dir():
        app.router.add_static("/static/", path=static_dir, name="static")
    return app


def _mark_activity(app: web.Application) -> None:
    """Record the current event-loop time as the last client activity."""

    app["last_activity"] = app.loop.time() if app.loop else 0.0


async def _heartbeat_watchdog(app: web.Application) -> None:
    """Stop the server when the browser tab goes away.

    Polls ``last_activity`` every ``_HEARTBEAT_INTERVAL_SECONDS``. If no
    heartbeat, turn, or context fetch has arrived within
    ``_HEARTBEAT_TIMEOUT_SECONDS``, sets ``watchdog_trigger`` so
    ``_run_server_until_quit`` unblocks and tears the runner down (which
    fires ``on_cleanup`` → ``runtime.aclose()``). We do NOT call
    ``app.shutdown()`` here — the runner owns the teardown.
    """

    loop = app.loop
    if loop is None:  # pragma: no cover — defensive
        return
    app["last_activity"] = loop.time()
    while True:
        await asyncio.sleep(_HEARTBEAT_INTERVAL_SECONDS)
        last = app["last_activity"]
        if loop.time() - last >= _HEARTBEAT_TIMEOUT_SECONDS:
            logger.info(
                "web watchdog: no heartbeat for %.0fs, shutting down", _HEARTBEAT_TIMEOUT_SECONDS
            )
            app["watchdog_trigger"].set()
            return


async def _handle_turn_request(request: web.Request) -> web.Response:
    """Process a turn request after token validation passes."""

    _mark_activity(request.app)

    try:
        payload = await request.json()
    except Exception:
        logger.warning("web turn received invalid JSON body from %s", request.remote)
        return web.json_response({"error": "invalid json body"}, status=400)

    message = str(payload.get("message") or "").strip() if isinstance(payload, dict) else ""
    if not message:
        logger.debug("web turn rejected empty message from %s", request.remote)
        return web.json_response({"error": "message is required"}, status=400)

    rt: HostRuntime = request.app["runtime"]
    turn_lock: asyncio.Lock = request.app["turn_lock"]

    async with turn_lock:
        transport = WebTransport()
        try:
            result = await handle_message(rt, message, transport)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("web turn failed: %s: %s", type(exc).__name__, exc)
            return web.json_response(
                {"error": f"{type(exc).__name__}: {exc}"},
                status=500,
            )
        response_text = transport.response_text or result.text
        context = _turn_context_to_dict(rt.last_turn_context)
        return web.json_response(
            {
                "response_text": response_text,
                "last_turn_context": context,
                "is_error": result.is_error,
            }
        )


async def _handle_turn_stream_request(request: web.Request) -> web.StreamResponse:
    """Process a turn request via SSE streaming.

    Creates an ``SSETransport`` backed by an ``asyncio.Queue``, launches
    ``handle_message`` as a background task, and streams events as
    ``text/event-stream`` frames to the client.

    The turn lock serializes concurrent turns — only one SSE stream is
    active at a time.
    """

    _mark_activity(request.app)

    message = str(request.query.get("message") or "").strip()
    if not message:
        raise web.HTTPBadRequest(text="message query parameter is required")

    rt: HostRuntime = request.app["runtime"]
    turn_lock: asyncio.Lock = request.app["turn_lock"]
    debug = rt.config.debug

    # ── Prepare SSE response ───────────────────────────────────────
    rsp = web.StreamResponse(
        status=200,
        reason="OK",
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
    await rsp.prepare(request)

    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    transport = SSETransport(queue, debug=debug)

    async def _run_turn() -> None:
        """Background task: execute the turn and push a sentinel when done."""
        try:
            await handle_message(rt, message, transport)
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("web turn stream failed: %s: %s", type(exc).__name__, exc)
            await queue.put(
                {"event": "error", "data": {"message": f"{type(exc).__name__}: {exc}"}}
            )
        finally:
            await queue.put({"event": "__sentinel__", "data": {}})

    async with turn_lock:
        turn_task = asyncio.create_task(_run_turn())

        try:
            while True:
                item = await queue.get()
                event_type = item["event"]
                if event_type == "__sentinel__":
                    break
                data = item["data"]
                # Write SSE frame: "event: <type>\ndata: <json>\n\n"
                frame = f"event: {event_type}\ndata: {json.dumps(data, default=str)}\n\n"
                await rsp.write(frame.encode("utf-8"))
        finally:
            # Ensure the turn task is done before releasing the lock.
            if not turn_task.done():
                turn_task.cancel()
                try:
                    await turn_task
                except asyncio.CancelledError:
                    pass
            # Drain any remaining items so the queue doesn't leak memory.
            while not queue.empty():
                queue.get_nowait()

    # SSE sentinel comment to signal stream end.
    await rsp.write(b":[DONE]\n\n")
    await rsp.write_eof()
    return rsp


# ── Server entry point ──────────────────────────────────────────────────


async def _cleanup_runner_on_interrupt(app: web.Application) -> None:
    """Best-effort runner cleanup after a KeyboardInterrupt.

    On Ctrl-C, ``asyncio.run`` cancels the coroutine mid-``await``; the
    normal ``runner.cleanup()`` path in the ``finally`` may not run
    because the ``CancelledError`` unwinds first. This helper is run in
    a fresh loop to fire ``on_cleanup`` (watchdog cancel +
    ``runtime.aclose()``) so the MCP pool does not leak.
    """

    for handler in list(app.on_cleanup):
        try:
            await handler(app)
        except Exception:  # pylint: disable=broad-except
            logger.debug("web on_cleanup handler failed during interrupt", exc_info=True)


async def _run_server_until_quit(
    *,
    app: web.Application,
    host: str,
    port: int,
    open_browser: bool,
    launch_url: str,
) -> None:
    """Run the aiohttp server until the quit signal fires, then tear down.

    Uses ``AppRunner`` + ``TCPSite`` (not ``web.run_app``) so the watchdog
    task and the ``/api/quit`` beacon can stop the server cleanly: they set
    ``app["watchdog_trigger"]``, which this coroutine awaits, then it
    calls ``runner.cleanup()`` (which runs ``on_cleanup`` →
    ``runtime.aclose()``) and returns. The process then exits.

    Lifecycle order:
      1. ``runner.setup()`` → fires ``on_startup`` (runtime.start, watchdog start)
      2. ``site.start()`` → begins serving on host:port
      3. open browser if requested
      4. ``await watchdog_trigger`` → blocks until quit beacon or watchdog timeout
      5. ``runner.cleanup()`` → fires ``on_cleanup`` (watchdog cancel, runtime.aclose)
    """

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host=host, port=port)
    await site.start()
    logger.info("web server serving on %s:%d", host, port)

    if open_browser:
        webbrowser.open(launch_url)

    try:
        await app["watchdog_trigger"].wait()
    finally:
        logger.info("web server shutting down")
        await runner.cleanup()


def run_web_server(
    *,
    runtime: HostRuntime,
    host: str,
    port: int,
    token: str,
    open_browser: bool,
) -> None:
    """Run the local web host until the browser tab closes (or interrupted).

    The runtime's ``start()`` and ``aclose()`` are bound to the server's
    ``on_startup`` / ``on_cleanup`` hooks, so they execute inside the
    aiohttp event loop (satisfying the anyio cross-loop rule).

    The server runs under an explicit ``AppRunner`` so the heartbeat
    watchdog (or the ``/api/quit`` beacon) can stop it cleanly: both set
    ``watchdog_trigger``, which ``_run_server_until_quit`` awaits, after
    which ``runner.cleanup()`` runs ``on_cleanup`` and the process exits.
    """

    app = _build_web_app(runtime=runtime, token=token)
    launch_url = web_url(host=host, port=port, token=token)
    logger.info("starting web server on %s:%d", host, port)

    try:
        asyncio.run(
            _run_server_until_quit(
                app=app,
                host=host,
                port=port,
                open_browser=open_browser,
                launch_url=launch_url,
            )
        )
    except KeyboardInterrupt:
        # Ctrl-C in the terminal: let the loop tear down via runner.cleanup.
        logger.info("web server interrupted (KeyboardInterrupt)")
        asyncio.run(_cleanup_runner_on_interrupt(app))


__all__ = [
    "DEFAULT_WEB_HOST",
    "DEFAULT_WEB_PORT",
    "SSETransport",
    "WebTransport",
    "build_web_index_html",
    "generate_web_token",
    "run_web_server",
    "web_url",
]

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
import logging
import secrets
import webbrowser
from dataclasses import asdict, is_dataclass
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
    """Build the lightweight local web chat page."""

    safe_token = html.escape(token, quote=True)
    return f"""<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Lisa Web</title>
    <style>
      :root {{
        color-scheme: light;
        --bg: #f4f6f8;
        --card: #ffffff;
        --text: #1f2933;
        --muted: #67788a;
        --accent: #1f6feb;
        --border: #dce3ea;
      }}
      body {{
        margin: 0;
        background: linear-gradient(140deg, #edf2f7 0%, #f8fafc 60%, #ebf5ff 100%);
        color: var(--text);
        font-family: "Segoe UI", "Aptos", "Calibri", sans-serif;
      }}
      .shell {{
        max-width: 980px;
        margin: 16px auto;
        padding: 0 16px 24px;
      }}
      .header {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
      }}
      .title {{
        margin: 0;
        font-size: 20px;
        letter-spacing: 0.3px;
      }}
      .subtitle {{
        margin: 2px 0 0;
        color: var(--muted);
        font-size: 13px;
      }}
      .badge {{
        border: 1px solid var(--border);
        background: var(--card);
        border-radius: 999px;
        padding: 6px 10px;
        font-size: 12px;
        color: var(--muted);
      }}
      .panel {{
        margin-top: 14px;
        border: 1px solid var(--border);
        border-radius: 12px;
        background: var(--card);
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
      }}
      .messages {{
        min-height: 360px;
        max-height: 60vh;
        overflow: auto;
        padding: 12px;
      }}
      .msg {{
        border: 1px solid var(--border);
        border-radius: 10px;
        margin-bottom: 10px;
        overflow: hidden;
      }}
      .msg .meta {{
        padding: 6px 10px;
        font-size: 12px;
        color: var(--muted);
        background: #f8fafc;
        border-bottom: 1px solid var(--border);
      }}
      .msg pre {{
        margin: 0;
        padding: 10px;
        white-space: pre-wrap;
        word-break: break-word;
        font-family: "Consolas", "Cascadia Mono", monospace;
        font-size: 13px;
      }}
      .composer {{
        padding: 12px;
        border-top: 1px solid var(--border);
      }}
      textarea {{
        width: 100%;
        min-height: 86px;
        resize: vertical;
        border-radius: 10px;
        border: 1px solid var(--border);
        background: #fff;
        color: var(--text);
        padding: 10px;
        box-sizing: border-box;
        font-family: "Segoe UI", "Aptos", "Calibri", sans-serif;
      }}
      .actions {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-top: 10px;
        gap: 10px;
      }}
      button {{
        background: var(--accent);
        border: none;
        color: #fff;
        border-radius: 10px;
        padding: 9px 14px;
        font-weight: 600;
        cursor: pointer;
      }}
      button[disabled] {{
        opacity: 0.6;
        cursor: not-allowed;
      }}
      details {{
        margin-top: 14px;
      }}
      details summary {{
        cursor: pointer;
        color: var(--muted);
      }}
      .context {{
        margin-top: 8px;
        border: 1px solid var(--border);
        border-radius: 10px;
        background: #fff;
      }}
      .context pre {{
        margin: 0;
        padding: 10px;
        overflow: auto;
        font-size: 12px;
      }}
    </style>
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

      <details>
        <summary>Last turn context</summary>
        <section class=\"context\"><pre id=\"context\">(none)</pre></section>
      </details>
    </main>

    <script>
      const state = {{
        token: "{safe_token}",
        busy: false,
      }};

      const messagesEl = document.getElementById("messages");
      const promptEl = document.getElementById("prompt");
      const sendEl = document.getElementById("send");
      const statusEl = document.getElementById("status");
      const contextEl = document.getElementById("context");

      function setBusy(next) {{
        state.busy = Boolean(next);
        sendEl.disabled = state.busy;
        statusEl.textContent = state.busy ? "working" : "ready";
      }}

      function appendMessage(kind, text) {{
        const wrapper = document.createElement("article");
        wrapper.className = "msg";

        const meta = document.createElement("div");
        meta.className = "meta";
        meta.textContent = kind;

        const body = document.createElement("pre");
        body.textContent = text;

        wrapper.appendChild(meta);
        wrapper.appendChild(body);
        messagesEl.appendChild(wrapper);
        messagesEl.scrollTop = messagesEl.scrollHeight;
      }}

      async function refreshContext() {{
        const response = await fetch(`/api/context?token=${{encodeURIComponent(state.token)}}`);
        if (!response.ok) {{
          return;
        }}
        const payload = await response.json();
        contextEl.textContent = JSON.stringify(payload, null, 2);
      }}

      async function sendPrompt() {{
        const message = promptEl.value.trim();
        if (!message || state.busy) {{
          return;
        }}
        appendMessage("you", message);
        promptEl.value = "";
        setBusy(true);
        try {{
          const response = await fetch("/api/turn", {{
            method: "POST",
            headers: {{
              "Content-Type": "application/json",
              "X-Lisa-Token": state.token,
            }},
            body: JSON.stringify({{ message }}),
          }});
          const payload = await response.json();
          if (!response.ok) {{
            appendMessage("error", payload.error || "request failed");
          }} else {{
            appendMessage("lisa", payload.response_text || "");
            contextEl.textContent = JSON.stringify(payload.last_turn_context || {{}}, null, 2);
          }}
        }} catch (error) {{
          appendMessage("error", String(error));
        }} finally {{
          setBusy(false);
          promptEl.focus();
        }}
      }}

      sendEl.addEventListener("click", sendPrompt);
      promptEl.addEventListener("keydown", (event) => {{
        if (event.key === "Enter" && event.ctrlKey) {{
          event.preventDefault();
          sendPrompt();
        }}
      }});

      // ── Heartbeat + quit-on-close ───────────────────────────────
      // Ping the server every few seconds so it knows the tab is alive.
      // On pagehide, send a quit beacon so the server can shut down
      // immediately instead of waiting for the watchdog timeout.
      async function pingHeartbeat() {{
        try {{
          await fetch("/api/heartbeat?token=" + encodeURIComponent(state.token), {{
            method: "POST",
            keepalive: true,
          }});
        }} catch (error) {{
          // Network errors are expected during shutdown; swallow them.
        }}
      }}

      function sendQuitBeacon() {{
        try {{
          navigator.sendBeacon("/api/quit?token=" + encodeURIComponent(state.token));
        }} catch (error) {{
          // sendBeacon may not be available or may fail during unload.
        }}
      }}

      window.addEventListener("pagehide", sendQuitBeacon);
      window.addEventListener("beforeunload", sendQuitBeacon);
      setInterval(pingHeartbeat, 5000);
      pingHeartbeat();

      refreshContext();
      promptEl.focus();
    </script>
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

    # ── Route handlers ───────────────────────────────────────────────
    async def _index(request: web.Request) -> web.Response:
        expected_token = str(request.app["web_token"])
        if not _is_token_valid(request, expected_token):
            return _unauthorized_response(
                request, "index", web.Response(status=401, text="Unauthorized")
            )
        html_page = build_web_index_html(token=expected_token)
        return web.Response(text=html_page, content_type="text/html")

    async def _health(_request: web.Request) -> web.Response:
        return web.json_response({"ok": True})

    async def _context(request: web.Request) -> web.Response:
        if not _is_token_valid(request, str(request.app["web_token"])):
            return _unauthorized_response(
                request, "context", web.json_response({"error": "unauthorized"}, status=401)
            )
        _mark_activity(request.app)
        rt: HostRuntime = request.app["runtime"]
        context = _turn_context_to_dict(rt.last_turn_context)
        return web.json_response(context)

    async def _heartbeat(request: web.Request) -> web.Response:
        if not _is_token_valid(request, str(request.app["web_token"])):
            return _unauthorized_response(
                request, "heartbeat", web.json_response({"error": "unauthorized"}, status=401)
            )
        _mark_activity(request.app)
        return web.json_response({"ok": True})

    async def _quit(request: web.Request) -> web.Response:
        if not _is_token_valid(request, str(request.app["web_token"])):
            return _unauthorized_response(
                request, "quit", web.json_response({"error": "unauthorized"}, status=401)
            )
        logger.info("web quit beacon received, shutting down")
        # Set the trigger; _run_server_until_quit is awaiting it and will
        # run runner.cleanup() (which fires on_cleanup → runtime.aclose).
        # We do NOT call app.shutdown() here — that would race the runner
        # teardown. Respond 200 so the beacon completes first.
        request.app["watchdog_trigger"].set()
        return web.json_response({"ok": True})

    async def _turn(request: web.Request) -> web.Response:
        if not _is_token_valid(request, str(request.app["web_token"])):
            return _unauthorized_response(
                request, "turn", web.json_response({"error": "unauthorized"}, status=401)
            )
        return await _handle_turn_request(request)

    app.router.add_get("/", _index)
    app.router.add_get("/health", _health)
    app.router.add_get("/api/context", _context)
    app.router.add_post("/api/turn", _turn)
    app.router.add_post("/api/heartbeat", _heartbeat)
    app.router.add_post("/api/quit", _quit)
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
    "WebTransport",
    "build_web_index_html",
    "generate_web_token",
    "run_web_server",
    "web_url",
]

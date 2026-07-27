/* Lisa Web — localhost chat UI */

const state = {
  token: window.LISA_TOKEN || "",
  busy: false,
};

const messagesEl = document.getElementById("messages");
const promptEl = document.getElementById("prompt");
const sendEl = document.getElementById("send");
const statusEl = document.getElementById("status");
const stageBarEl = document.getElementById("stage-bar");
const contextEl = document.getElementById("context");
const tokenHudEl = document.getElementById("token-hud");

// ── Helpers ──────────────────────────────────────────────────────────

function setBusy(next) {
  state.busy = Boolean(next);
  sendEl.disabled = state.busy;
  if (!state.busy) {
    statusEl.textContent = "ready";
    if (stageBarEl) {
      stageBarEl.textContent = "";
      stageBarEl.classList.add("ready");
    }
  }
}

function appendMessage(kind, text) {
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
  return wrapper;
}

// ── Stage bar rendering ──────────────────────────────────────────────

const TOOL_KIND_GLYPH = {
  operation: "→ op",
  pack_tool: "→ tool",
  mcp_tool: "→ mcp",
  transition: "↦",
};

function renderStageLine(data) {
  const label = data.label || "";
  const modeId = data.mode_id || "";
  const prefix = modeId ? "[" + modeId + "] " : "";

  // tool_dispatch gets a legible glyph.
  if (label === "tool_dispatch") {
    const tool = data.tool || "?";
    const kind = data.tool_kind || "";
    const glyph = TOOL_KIND_GLYPH[kind] || "→";
    let body = glyph + ": " + tool;

    // Transitions show the target.
    if (kind === "transition") {
      let target = "";
      if (data.handoff_to) {
        target = "→ " + data.handoff_to;
      } else if (data.transition === "fork") {
        const n = data.fork_children || 0;
        target = "fork (" + n + " children)";
      } else if (data.transitioned_to) {
        target = "→ " + data.transitioned_to;
      }
      if (target) {
        body = glyph + " " + tool + " " + target;
      }
    }

    const meta = [];
    if (data.iteration) meta.push("iter=" + data.iteration);
    if (data.tool_calls_so_far) meta.push("calls=" + data.tool_calls_so_far);
    if (meta.length) body += " (" + meta.join(", ") + ")";
    return prefix + "stage: tool_dispatch " + body;
  }

  // Normal stage line.
  const parts = [];
  for (const key of [
    "provider", "model_id", "iteration", "tool_calls_so_far",
    "tool_calls", "duration_ms", "stage_id", "turn_count",
    "prompt_tokens", "completion_tokens", "total_tokens",
  ]) {
    const value = data[key];
    if (value != null && value !== "" && value !== 0) {
      parts.push(key + "=" + value);
    }
  }
  const suffix = parts.length ? " (" + parts.join(", ") + ")" : "";
  return prefix + "stage: " + label + suffix;
}

// ── Token HUD ────────────────────────────────────────────────────────

async function refreshTokenHud() {
  if (!tokenHudEl) return;
  try {
    const resp = await fetch("/api/tokens?token=" + encodeURIComponent(state.token));
    if (!resp.ok) return;
    const data = await resp.json();
    const s = data.session || {};
    const parts = [];
    if (s.total_tokens) {
      parts.push((s.total_tokens / 1000).toFixed(1) + "k tokens");
    }
    if (s.turn_count) {
      parts.push(s.turn_count + " turns");
    }
    if (s.model_token_totals) {
      const models = Object.keys(s.model_token_totals).length;
      if (models) parts.push(models + " models");
    }
    tokenHudEl.textContent = parts.join(" · ") || "";
  } catch (_) {
    // Token HUD is best-effort.
  }
}

// ── Context refresh ──────────────────────────────────────────────────

async function refreshContext() {
  try {
    const response = await fetch("/api/context?token=" + encodeURIComponent(state.token));
    if (!response.ok) return;
    const payload = await response.json();
    contextEl.textContent = JSON.stringify(payload, null, 2);
  } catch (_) {
    // Context refresh is best-effort.
  }
}

// ── Send prompt (SSE streaming) ──────────────────────────────────────

async function sendPrompt() {
  const message = promptEl.value.trim();
  if (!message || state.busy) return;

  appendMessage("you", message);
  promptEl.value = "";
  setBusy(true);
  statusEl.textContent = "working";
  if (stageBarEl) {
    stageBarEl.textContent = "dispatch";
    stageBarEl.classList.remove("ready");
  }

  const url = "/api/turn/stream?message=" + encodeURIComponent(message) +
    "&token=" + encodeURIComponent(state.token);
  const es = new EventSource(url);

  let responseWrapper = null;
  let responseBody = null;
  let responseText = "";

  es.addEventListener("stage", function (event) {
    try {
      const data = JSON.parse(event.data);
      if (stageBarEl) {
        stageBarEl.textContent = renderStageLine(data);
        stageBarEl.classList.remove("ready");
      }
      const label = data.label || "";
      const modeId = data.mode_id || "";
      const prefix = modeId ? "[" + modeId + "] " : "";
      statusEl.textContent = prefix + label;
    } catch (_) {
      // Ignore parse errors in stage events.
    }
  });

  es.addEventListener("token", function (event) {
    try {
      const data = JSON.parse(event.data);
      const chunk = data.text || "";
      responseText += chunk;
      if (!responseWrapper) {
        responseWrapper = document.createElement("article");
        responseWrapper.className = "msg";
        const meta = document.createElement("div");
        meta.className = "meta";
        meta.textContent = "lisa";
        responseWrapper.appendChild(meta);
        responseBody = document.createElement("pre");
        responseWrapper.appendChild(responseBody);
        messagesEl.appendChild(responseWrapper);
      }
      responseBody.textContent = responseText;
      messagesEl.scrollTop = messagesEl.scrollHeight;
    } catch (_) {
      // Ignore parse errors in token events.
    }
  });

  es.addEventListener("done", function (event) {
    es.close();
    try {
      const data = JSON.parse(event.data);
      if (!responseWrapper) {
        // No tokens were streamed — render the full response now.
        const wrapper = document.createElement("article");
        wrapper.className = "msg";
        const meta = document.createElement("div");
        meta.className = "meta";
        meta.textContent = "lisa";
        const body = document.createElement("div");
        body.className = "msg-body";
        body.innerHTML = data.html || "";
        wrapper.appendChild(meta);
        wrapper.appendChild(body);
        messagesEl.appendChild(wrapper);
      } else {
        // Tokens were streamed — replace the <pre> with rendered HTML.
        const body = document.createElement("div");
        body.className = "msg-body";
        body.innerHTML = data.html || "";
        responseBody.replaceWith(body);
      }
      if (data.data && data.data.last_turn_context) {
        contextEl.textContent = JSON.stringify(data.data.last_turn_context, null, 2);
      }
    } catch (_) {
      // Ignore parse errors in done event.
    }
    setBusy(false);
    promptEl.focus();
    refreshTokenHud();
  });

  es.addEventListener("error", function (event) {
    es.close();
    let errorMsg = "request failed";
    try {
      if (event.data) {
        const data = JSON.parse(event.data);
        errorMsg = data.message || errorMsg;
      }
    } catch (_) {
      // event.data may not be JSON (native error event).
    }
    appendMessage("error", errorMsg);
    setBusy(false);
    promptEl.focus();
  });

  es.onerror = function () {
    // EventSource fires onerror for network errors and non-200
    // responses. The server sends an "error" event frame for
    // application-level errors (which the "error" listener above
    // handles). If we get here without a prior "error" event, it's
    // a transport failure.
    if (state.busy) {
      es.close();
      appendMessage("error", "connection lost");
      setBusy(false);
      promptEl.focus();
    }
  };
}

// ── Event listeners ──────────────────────────────────────────────────

sendEl.addEventListener("click", sendPrompt);
promptEl.addEventListener("keydown", function (event) {
  if (event.key === "Enter" && event.ctrlKey) {
    event.preventDefault();
    sendPrompt();
  }
});

// ── Heartbeat + quit-on-close ────────────────────────────────────────

function pingHeartbeat() {
  try {
    fetch("/api/heartbeat?token=" + encodeURIComponent(state.token), {
      method: "POST",
      keepalive: true,
    });
  } catch (_) {
    // Network errors are expected during shutdown; swallow them.
  }
}

function sendQuitBeacon() {
  try {
    navigator.sendBeacon("/api/quit?token=" + encodeURIComponent(state.token));
  } catch (_) {
    // sendBeacon may not be available or may fail during unload.
  }
}

window.addEventListener("pagehide", sendQuitBeacon);
window.addEventListener("beforeunload", sendQuitBeacon);
setInterval(pingHeartbeat, 5000);
pingHeartbeat();

// ── Initialization ───────────────────────────────────────────────────

refreshContext();
refreshTokenHud();
promptEl.focus();

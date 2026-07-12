/**
 * Lisa Framework — VS Code extension shell.
 *
 * This is a thin TypeScript transport layer. It spawns the Python
 * stdio bridge (`hosts.vscode.bridge`) as a child process and pipes
 * JSON-RPC messages to/from it. The Python side does all the work
 * (turn routing, MCP pool, model calls, context assembly). The TS
 * shell just:
 *
 *   1. Spawns `python -m hosts.vscode.bridge --data-dir <globalStoragePath>`.
 *   2. Registers a Chat participant handler for `@lisa`.
 *   3. Sends user messages to the bridge over stdin.
 *   4. Receives responses from stdout and renders them in the Chat view.
 *   5. Handles the `lisa.turn` command (opens the Chat with @lisa).
 *
 * Protocol: one JSON object per line over stdio.
 *   Request:  {"id": 1, "method": "turn", "params": {"message": "hello"}}
 *   Response: {"id": 1, "result": {"text": "...", "data": null, "is_error": false}}
 *   Notification: {"method": "notify", "params": {"stage": "frontdoor", ...}}
 *   Error notif: {"method": "error", "params": {"message": "..."}}
 */

import * as vscode from "vscode";
import { ChildProcess, spawn } from "child_process";
import { EventEmitter } from "events";
import * as path from "path";

// ── Bridge protocol types ────────────────────────────────────────────

interface BridgeRequest {
  id: number;
  method: string;
  params: Record<string, unknown>;
}

interface BridgeResponse {
  id: number;
  result?: { text: string; data: unknown; is_error: boolean };
  error?: { code: number; message: string };
}

interface BridgeNotification {
  method: "notify" | "error";
  params: Record<string, unknown>;
}

type BridgeMessage = BridgeResponse | BridgeNotification;

// ── Bridge client ─────────────────────────────────────────────────────

class BridgeClient extends EventEmitter {
  private proc: ChildProcess | null = null;
  private nextId = 1;
  private pending = new Map<number, (msg: BridgeMessage) => void>();
  private buffer = "";
  private started = false;

  constructor(
    private readonly pythonPath: string,
    private readonly dataDir: string,
  ) {
    super();
  }

  async start(): Promise<void> {
    if (this.started) {
      return;
    }
    this.started = true;

    const modulePath = "hosts.vscode.bridge";
    this.proc = spawn(this.pythonPath, ["-m", modulePath, "--data-dir", this.dataDir], {
      stdio: ["pipe", "pipe", "pipe"],
    });

    this.proc.on("error", (err) => {
      this.emit("error", err);
    });

    this.proc.on("exit", (code) => {
      this.emit("exit", code);
    });

    this.proc.on("close", (code) => {
      this.emit("close", code);
    });

    this.proc.stdout?.setEncoding("utf8");
    this.proc.stdout?.on("data", (chunk: string) => {
      this.buffer += chunk;
      this._drainBuffer();
    });

    this.proc.stderr?.setEncoding("utf8");
    this.proc.stderr?.on("data", (chunk: string) => {
      console.error("[lisa-bridge stderr]", chunk.trim());
    });
  }

  private _drainBuffer(): void {
    while (true) {
      const nl = this.buffer.indexOf("\n");
      if (nl === -1) {
        break;
      }
      const line = this.buffer.slice(0, nl).trim();
      this.buffer = this.buffer.slice(nl + 1);
      if (!line) {
        continue;
      }
      try {
        const msg = JSON.parse(line) as BridgeMessage;
        this._dispatch(msg);
      } catch (err) {
        console.error("[lisa-bridge] failed to parse line:", line, err);
      }
    }
  }

  private _dispatch(msg: BridgeMessage): void {
    // Notifications have no `id` and no `result`.
    if (!("id" in msg)) {
      this.emit("notification", msg as BridgeNotification);
      return;
    }
    const resolver = this.pending.get(msg.id);
    if (resolver) {
      this.pending.delete(msg.id);
      resolver(msg);
    }
  }

  async turn(message: string, onProgress?: (notif: BridgeNotification) => void): Promise<string> {
    if (!this.proc || !this.proc.stdin) {
      throw new Error("Bridge not started");
    }
    const id = this.nextId++;
    const request: BridgeRequest = { id, method: "turn", params: { message } };

    const onNotif = (notif: BridgeNotification) => {
      onProgress?.(notif);
    };
    this.on("notification", onNotif);

    try {
      const response = await new Promise<BridgeMessage>((resolve, reject) => {
        this.pending.set(id, resolve);
        const line = JSON.stringify(request) + "\n";
        this.proc!.stdin!.write(line, (err) => {
          if (err) {
            reject(new Error(`Failed to write to bridge: ${err.message}`));
          }
        });
      });

      if ("error" in response && response.error) {
        throw new Error(response.error.message);
      }
      if ("result" in response && response.result) {
        return response.result.text;
      }
      return "";
    } finally {
      this.off("notification", onNotif);
    }
  }

  async shutdown(): Promise<void> {
    if (!this.proc || !this.proc.stdin) {
      return;
    }
    const id = this.nextId++;
    const request: BridgeRequest = { id, method: "shutdown", params: {} };
    try {
      await new Promise<void>((resolve) => {
        this.pending.set(id, () => resolve());
        this.proc!.stdin!.write(JSON.stringify(request) + "\n", () => resolve());
      });
    } catch {
      // Best effort.
    }
    this.proc.kill();
    this.proc = null;
    this.started = false;
  }
}

// ── Extension activation ──────────────────────────────────────────────

let bridge: BridgeClient | null = null;

export async function activate(context: vscode.ExtensionContext): Promise<void> {
  const pythonPath =
    vscode.getConfiguration("lisa").get<string>("pythonPath") ||
    "python";

  const dataDir = context.globalStorageUri.fsPath;

  // Ensure the global storage dir exists.
  await vscode.workspace.fs.createDirectory(context.globalStorageUri);

  bridge = new BridgeClient(pythonPath, dataDir);
  await bridge.start();

  bridge.on("error", (err) => {
    console.error("[lisa] bridge error:", err);
  });
  bridge.on("exit", (code) => {
    console.warn(`[lisa] bridge exited with code ${code}`);
  });

  // ── Chat participant handler ────────────────────────────────────────
  const participant = vscode.chat.createChatParticipant(
    "lisa-framework.lisa",
    async (
      request: vscode.ChatRequest,
      _chatContext: vscode.ChatContext,
      stream: vscode.ChatResponseStream,
      _token: vscode.CancellationToken,
    ): Promise<void> => {
      if (!bridge) {
        stream.error("Lisa bridge is not running.");
        return;
      }

      try {
        const text = await bridge.turn(
          request.prompt,
          (notif) => {
            if (notif.method === "notify") {
              const stage = notif.params.stage as string | undefined;
              if (stage) {
                stream.progress(stage);
              }
            }
          },
        );
        stream.markdown(text);
      } catch (err) {
        stream.error(`Lisa turn failed: ${(err as Error).message}`);
      }
    },
  );
  participant.iconPath = new vscode.ThemeIcon("comment-discussion");

  context.subscriptions.push(participant);

  // ── Command handler ─────────────────────────────────────────────────
  const command = vscode.commands.registerCommand("lisa.turn", async () => {
    await vscode.commands.executeCommand("workbench.action.chat.open", "@lisa ");
  });
  context.subscriptions.push(command);
}

export async function deactivate(): Promise<void> {
  if (bridge) {
    await bridge.shutdown();
    bridge = null;
  }
}
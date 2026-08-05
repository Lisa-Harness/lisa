# Lisa Framework

The public Lisa framework packages: runtime-neutral kernel contracts, an author-facing SDK, a set of public packs, the terminal CLI host, and the MCP servers. This workspace is intentionally separate from the private production Lisa Teams bot app.

## What Is Included

- `kernel/`: runtime-neutral contracts — mode routing, workflow state (`ipsar`/`ipsar_fsm`), the operation registry, tool protocols, per-iteration model router discovery, the slice (pressure-driven housekeeping) contract, approval/capability seams, and testing helpers.
- `sdk/`: author-facing helper contracts for packs, rendering, the CEL-based model router, and integration contracts.
- `sdk/packs/core/`: baseline public modes (`default`, `workshop`) and shared prompt assets.
- `sdk/packs/memory/`: memory-backed learning behaviors (`learn`, memory ops, default override candidate).
- `sdk/packs/introspect/`: operational self-model and reflective self-assessment operations.
- `sdk/packs/checkpoint/`: LLM-enriched checkpoint/restore operations.
- `host/`: the shared transport-neutral runtime (`HostConfig`, `HostRuntime`, `TransportCallbacks`, provider gateways) used by transport hosts.
- `../lisa-hosts/cli/`: the terminal host (submodule of the `lisa-hosts` umbrella repo). Entry point: `lisa`. Provides the REPL, `once`, `verify`, `modes`, `models`, `operations`, `tools`, `mcp`, `pack`, and `intake` commands. Distribution: `lisa-host-cli` (`uv tool install lisa-host-cli`); meta-package `lisa-host` (`uv tool install lisa-host`).
- `mcp_server/`: MCP (Model Context Protocol) servers that expose Lisa's surface-scoped tools over stdio — `lisa-context`, `lisa-content`, `lisa-catalog`, `lisa-memory`, `lisa-guard`, `lisa-storage`.

## What Is Excluded

The production Lisa app layer is not part of this framework. The private app owns the Teams adapter, production model routing, Azure deployment, proprietary memory and proactive loops, CloudServers-specific extensions and packs, and market-intel surfaces.

Market intelligence and CloudServers operations logic stay private. Public packages may expose generic extension and evidence contracts, but they do not include CloudServers-specific collectors, private prompt assets, or product data.

## Quick Start

From this directory:

```powershell
uv sync --extra dev
uv run lisa verify
uv run lisa modes
uv run lisa models
uv run lisa once "/search public sdk design"
uv run lisa once "/markdown Draft a short extension checklist"
uv run lisa once "/discuss capability tradeoffs"
uv run lisa
```

Lisa uses a generated model manifest in `artifacts/models.yaml` (or your configured data-dir) with `default` and `reasoning` designations. Use `/models discover` then `/models apply` to generate and activate it. It supports live completion through `azure_openai`, `openai`, `anthropic`, `google`, and local `ollama` providers, plus local action/chat packs discovered through `SDK_CONTENT_ROOTS`; heavier workflow modes still report the missing host wiring they require.

`lisa-pack-memory` is an opt-in pack in this workspace. Install it through the host pack manager when you want memory-backed learning behavior:

```powershell
uv run lisa pack install "C:\Code\lisa-pack-memory" --no-log
```

Key CLI options:

- `--streaming none|stage|text|hybrid[:delay]` controls terminal streaming mode.
- `--user <slug>` selects authored user context and controls thread-log keying.
- `--timezone <IANA>` pins status-line local time and timezone display.
- `--log-level debug|info|warning|error` controls JSONL verbosity.
- `--log <file>` appends structured JSONL turn records.
- `--no-log` suppresses JSONL logging entirely.
- `--data-dir <path>` overrides the artifacts root.
- `--thread-dir <path>` overrides the thread persistence directory.
- `--no-thread` disables thread persistence.
- `--session-context <slug>` sets the session context slug.
- `--state-snapshot <file.json>` injects compact local snapshot context into prompt assembly.
- `--version` prints the lisa version and exits.

On Windows, you can package a standalone executable (from the `../lisa-hosts/cli/` checkout):

```powershell
uv run pyinstaller --name lisa --onefile hosts/cli/main.py
```

For a first custom pack, follow `sdk/README.md` and point `SDK_CONTENT_ROOTS` at your pack folder. The same guide covers signal connectors, processing loops, and host commands — validate connectors through `kernel.testing.extension_harness`.

## Package Layout

The root `pyproject.toml` defines a public-only `uv` workspace. The workspace members are:

- `lisa-kernel` (`kernel/`)
- `lisa-sdk` (`sdk/`)
- `lisa-mcp-servers` (`mcp_server/`)
- `lisa-pack-core` (`sdk/packs/core/`)
- `lisa-pack-introspect` (`sdk/packs/introspect/`)
- `lisa-pack-memory` (`sdk/packs/memory/`)
- `lisa-pack-checkpoint` (`sdk/packs/checkpoint/`)
- `lisa-pack-compaction` (`sdk/packs/compaction/`)
- `lisa-host-runtime` (`host/`)
- `lisa-host-cli` (`../lisa-hosts/cli/hosts/cli/`) — entry point `lisa` (submodule of the `lisa-hosts` umbrella repo)
- `lisa-host` (`../lisa-hosts/packaging/lisa-host/`) — meta-package (`uv tool install lisa-host`)

External public packs resolve from the sibling `../lisa-packs/` checkout and are pulled in with `uv sync --extra ext`: `lisa-pack-brave-search`, `lisa-pack-letta`, `lisa-pack-markdown`, `lisa-pack-mem0`, `lisa-pack-okf`, `lisa-pack-search`.

WIP packs resolve from `../lisa-packs-wip/` and are pulled in with `uv sync --extra wip`: `lisa-pack-cloudflare`, `lisa-pack-coding`, `lisa-pack-github`, `lisa-pack-gmail`, `lisa-pack-stripe`.

Private packages such as CloudServers operations, CloudServers packs, and market-intel collectors are intentionally absent.

## Validation

Useful focused checks:

```powershell
uv run python -c "import kernel, sdk; print(kernel.__file__); print(sdk.__file__)"
uv run lisa verify
uv run lisa modes
uv run lisa models
uv run pytest --import-mode=importlib kernel/tests sdk/tests hosts -m "not live" -v --tb=short
uv run --extra dev ruff check --select F,I kernel sdk mcp_server hosts
```

See `docs/BOUNDARIES.md`, `docs/CODING_WORKFLOW.md`, `sdk/README.md`, the Operations section in `kernel/README.md`, `sdk/AUTHORING.md`, `docs/SESSION_CONTEXT_AUTHORING_GUIDE.md`, `hosts/README.md`, and `kernel/README.md` for authoring, host-boundary, and architecture details.

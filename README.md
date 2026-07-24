# Lisa Framework

This directory stages the public Lisa framework packages before they are moved to a separate repository. It is intentionally separate from the private production Lisa Teams bot app in the repository root.

## What Is Included

- `kernel/`: runtime-neutral contracts, mode routing, workflow state, operation registry, tool protocols, and testing helpers.
- `sdk/`: author-facing helper contracts for packs, workshop execution, rendering, and integration contracts.
- `sdk/packs/core/`: baseline public modes (`default`, `workshop`) and shared prompt assets.
- `sdk/packs/memory/`: memory-backed learning behaviors (`learn`, memory ops, and default override candidate).
- `sdk/packs/introspect/`: operational self-model and reflective self-assessment operations.
- `sdk/packs/checkpoint/`: LLM-enriched checkpoint/restore operations.
- `hosts/little_lisa/`: a local non-production terminal host that loads public packs with manifest-driven model routing and OpenAI-compatible provider calls.
- `mcp_server/`: an MCP (Model Context Protocol) memory server that exposes Lisa's surface-scoped memory tools over stdio.

## What Is Excluded

The production Lisa app layer is not part of this staged framework. The private app owns the Teams adapter, production model routing, Azure deployment, proprietary memory and proactive loops, CloudServers-specific extensions and packs, and market-intel surfaces.

Market intelligence and CloudServers operations logic stay private. Public packages may expose generic extension and evidence contracts, but they do not include CloudServers-specific collectors, private prompt assets, or product data.

## Quick Start

From this directory:

```powershell
uv sync --extra dev
uv run little-lisa verify
uv run little-lisa modes
uv run little-lisa models
uv run little-lisa once "/search public sdk design"
uv run little-lisa once "/markdown Draft a short extension checklist"
uv run little-lisa once "/deep Evaluate migration risks"
uv run little-lisa once "/discuss capability tradeoffs"
uv run little-lisa cli
uv run little-lisa desktop
uv run little-lisa
```

Little Lisa uses a generated model manifest in `artifacts/models.yaml` (or your configured data-dir) with `default`, `deep`, and `reasoning` designations. Use `/models discover` then `/models apply` to generate and activate it. It supports live completion through `azure_openai`, `openai`, `anthropic`, and local `ollama` providers, plus local action/chat packs discovered through `SDK_CONTENT_ROOTS`; heavier workflow modes still report the missing host wiring they require.

`lisa-pack-memory` is now an opt-in pack in this workspace. Install it through the host pack manager when you want memory-backed learning behavior:

```powershell
uv run little-lisa pack install "C:\Code\lisa-pack-memory" --no-log
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
- `--version` prints the Little Lisa version and exits.

On Windows, you can package a standalone executable:

```powershell
uv sync --extra dev
uv run pyinstaller --name little_lisa --onefile hosts/little_lisa/main.py
```

For a first custom pack, follow `docs/PACK_AUTHORING_GUIDE.md` and point `SDK_CONTENT_ROOTS` at your pack folder. The same guide covers signal connectors, processing loops, and host commands — validate connectors through `kernel.testing.extension_harness`.

## Package Layout

The root `pyproject.toml` defines a public-only `uv` workspace. The memory pack is part of the public framework set but not included in the default dependency graph during the split-repo transition:

- `lisa-kernel`
- `lisa-sdk`
- `lisa-pack-coding`
- `lisa-pack-cloudflare`
- `lisa-pack-github`
- `lisa-pack-stripe`
- `lisa-pack-core`
- `little-lisa`

Install `lisa-pack-memory` through the pack manager when you want memory-backed learning workflows.

Private packages such as CloudServers operations, CloudServers packs, and market-intel collectors are intentionally absent.

## Validation

Useful focused checks:

```powershell
uv run python -c "import kernel, sdk; print(kernel.__file__); print(sdk.__file__)"
uv run little-lisa verify
uv run little-lisa modes
uv run little-lisa models
uv run pytest --import-mode=importlib hosts/little_lisa/test_little_lisa.py kernel/tests sdk/tests -m "not live" -v --tb=short
uv run --extra dev ruff check --select F,I kernel sdk mcp_server hosts
```

See `docs/BOUNDARIES.md`, `docs/CODING_WORKFLOW.md`, `docs/PACK_AUTHORING_GUIDE.md`, `docs/OPERATIONS.md`, `docs/OPERATION_AUTHORING_GUIDE.md`, `docs/SESSION_CONTEXT_AUTHORING_GUIDE.md`, `docs/LITTLE_LISA_GUIDE.md`, and `docs/KERNEL_ARCHITECTURE.md` for authoring, host-boundary, and architecture details.

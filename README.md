# Lisa Framework

This directory stages the public Lisa framework packages before they are moved to a separate repository. It is intentionally separate from the private production Lisa Teams bot app in the repository root.

## What Is Included

- `kernel/`: runtime-neutral contracts, mode routing, workflow state, operation registry, tool protocols, and testing helpers.
- `lisa_sdk/`: author-facing helper contracts for packs, extensions, workshop execution, and rendering.
- `sdk_extensions/`: public extensions for Clarity, Cloudflare, GitHub, and Stripe.
- `sdk_packs/core/`: baseline public modes, operations, and prompt assets.
- `sdk_packs/extras/`: optional general-purpose modes and authoring operations.
- `examples/little_lisa/`: a local non-production terminal host that loads public packs with manifest-driven model routing and OpenAI-compatible provider calls.
- `mcp_server/`: an MCP (Model Context Protocol) memory server that exposes Lisa's surface-scoped memory tools over stdio.

## What Is Excluded

The production Lisa app layer is not part of this staged framework. The private app owns the Teams adapter, production model routing, Azure deployment, proprietary memory and proactive loops, CloudServers-specific extensions and packs, and market-intel surfaces.

Market intelligence and CloudServers operations logic stay private. Public packages may expose generic extension and evidence contracts, but they do not include CloudServers-specific collectors, private prompt assets, or product data.

## Quick Start

From this directory:

```powershell
uv sync --extra dev
uv run little-lisa verify
uv run little-lisa setup
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

Little Lisa uses a model manifest (`examples/little_lisa/models.yaml`) with `front-door`, `default`, and `deep` designations. It supports live completion through `azure_openai`, `openai`, `anthropic`, and local `ollama` providers, plus local action/chat packs discovered through `SDK_CONTENT_ROOTS`; heavier workflow modes still report the missing host wiring they require.

Key CLI options:

- `--streaming none|stage|text|hybrid[:delay]` controls terminal streaming mode.
- `--user <slug>` selects authored user context and controls thread-log keying.
- `--timeout-profile fast|balanced|deep` tunes per-mode operation timeout budgets.
- `--timezone <IANA>` pins status-line local time and timezone display.
- `--log-level debug|info|warning|error` controls JSONL verbosity.
- `--log <file>` appends structured JSONL turn records.
- `--no-log` suppresses JSONL logging entirely.
- `--data-dir <path>` overrides the artifacts root.
- `--memory-db <path>` overrides the memory store file.
- `--thread-dir <path>` overrides the thread persistence directory.
- `--no-thread` disables thread persistence.
- `--session-context <slug>` sets the session context slug.
- `--state-snapshot <file.json>` injects compact local snapshot context into prompt assembly.
- `--version` prints the Little Lisa version and exits.

On Windows, you can package a standalone executable:

```powershell
uv sync --extra dev
uv run pyinstaller --name little_lisa --onefile examples/little_lisa/main.py
```

For a first custom pack, follow `docs/PACK_AUTHORING_GUIDE.md` and point `SDK_CONTENT_ROOTS` at your pack folder. For a first custom extension, follow `docs/SDK_EXTENSIONS_GUIDE.md` and validate it through `kernel.testing.extension_harness`.

## Package Layout

The root `pyproject.toml` defines a public-only `uv` workspace:

- `lisa-kernel`
- `lisa-sdk`
- `lisa-ext-clarity`
- `lisa-ext-cloudflare`
- `lisa-ext-github`
- `lisa-ext-stripe`
- `lisa-pack-core`
- `lisa-pack-extras`
- `mcp-server`

Private packages such as CloudServers operations, CloudServers packs, and market-intel collectors are intentionally absent.

## Validation

Useful focused checks:

```powershell
uv run python -c "import kernel, lisa_sdk; print(kernel.__file__); print(lisa_sdk.__file__)"
uv run little-lisa verify
uv run little-lisa modes
uv run little-lisa models
uv run pytest --import-mode=importlib examples/little_lisa/test_little_lisa.py kernel/tests lisa_sdk/tests sdk_extensions/cloudflare/tests sdk_extensions/github/tests sdk_extensions/stripe/tests sdk_extensions/clarity/tests -m "not live" -v --tb=short
uv run --extra dev ruff check --select F,I kernel lisa_sdk sdk_extensions sdk_packs examples
```

See `docs/BOUNDARIES.md`, `docs/PACK_AUTHORING_GUIDE.md`, `docs/SDK_EXTENSIONS_GUIDE.md`, `docs/LITTLE_LISA_GUIDE.md`, and `docs/KERNEL_ARCHITECTURE.md` for authoring, host-boundary, and architecture details.

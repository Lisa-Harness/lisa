# Lisa Framework

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://docs.astral.sh/ruff/)

Lisa is a runtime-neutral framework for building **agent hosts** and
**mode/operation packs**. The kernel defines the contracts — mode routing,
workflow state machines, the operation registry, tool and storage protocols —
without taking a position on transport, storage backend, or model provider.
The SDK ships author-facing helpers, a set of reference packs, and a CEL-based
model router. A transport-neutral host runtime wires a starter subset, and a
collection of first-party MCP servers expose Lisa's surface-scoped tools over
stdio.

This workspace contains the **public framework packages**. It is intentionally
separate from the private production Lisa Teams bot app. The terminal CLI host
lives in the sibling [`lisa-hosts/cli`][lisa-host-cli] repository.

## Table of contents

- [Lisa Framework](#lisa-framework)
  - [Table of contents](#table-of-contents)
  - [Architecture](#architecture)
  - [Repository layout](#repository-layout)
  - [Installation](#installation)
  - [Quick start](#quick-start)
  - [Configuration](#configuration)
  - [Commands](#commands)
  - [MCP servers](#mcp-servers)
  - [Packs](#packs)
  - [Model providers](#model-providers)
  - [Privacy and surface scope](#privacy-and-surface-scope)
  - [Validation](#validation)
  - [Contributing](#contributing)
  - [License](#license)

## Architecture

Lisa is layered so that reusable mechanics live in packages with no transport
or production dependencies. Each layer owns a single responsibility and is
forbidden from reaching into the one above it.

| Layer | Responsibility | Must not own |
| --- | --- | --- |
| `kernel` | Runtime-neutral contracts, mode catalog loading, workflow state (`ipsar`), the operation registry, tool and storage protocols, per-iteration model-router discovery, the slice (pressure-driven housekeeping) contract | Host transport SDKs, production deployment, proprietary app behavior |
| `sdk` | Author-facing helpers for rendering, integration contracts, pack composition, and the CEL-based model router | Private app imports, private model policy, production storage |
| `sdk/packs` | Bundled reference packs with modes, operations, authored content, and pack-registered slash commands | Private product strategy, market-intel workflows |
| `host` | Transport-neutral runtime: typed config (`HostConfig`), mutable state (`HostRuntime`), provider dispatch, MCP pool compilation, command dispatch, context assembly, thread persistence | Production chat transport, durable memory, production-grade model policy |
| `mcp_server` | First-party MCP servers that expose Lisa's surface-scoped tools over stdio | Transport, host wiring |

A real host injects concrete implementations for the runtime seams the
framework leaves open: model completion and resolution callbacks, tool catalog
and execution callbacks, blob/object storage (via the `lisa-storage` MCP
server), memory store/recall when memory operations are enabled, browser
automation when surf operations are enabled, the transport adapter, and any
scheduler for signal connector polling. The framework keeps these as protocols
and callbacks so pack authors can build reusable behavior without depending
on any production host.

See [`docs/BOUNDARIES.md`](docs/BOUNDARIES.md) for the full public/private
boundary and the host layer architecture.

## Repository layout

This is a [uv][uv] workspace; the root `pyproject.toml` ties the member
packages together.

```text
.
├── kernel/              # Runtime-neutral contracts, FSM engine, operation registry
├── sdk/                 # Author-facing helpers and integration contracts
│   ├── packs/           # Bundled reference packs
│   │   ├── core/        #   baseline mode (default) and shared assets
│   │   ├── memory/      #   memory-backed learning operations
│   │   ├── introspect/  #   operational self-model and reflective assessment
│   │   └── recovery/    #   thread-based /resume and recovery records
│   └── router/          # CEL-based model router and runtime support
├── host/                # Transport-neutral host runtime
├── mcp_server/          # First-party MCP servers (context, content, catalog, ...)
├── artifacts/           # Generated manifests, thread store, token usage, overlays
├── docs/                # Architecture and design documents
└── scripts/             # Repo-local tooling
```

## Installation

Lisa requires **Python 3.11 or newer** and is managed with [uv][uv]. Clone the
repo and sync the workspace, including the development extras:

```sh
git clone https://github.com/Lisa-Framework/lisa.git
cd lisa
uv sync --extra dev
```

The framework packages are installed as editable workspace members. To drive
Lisa from a terminal, install the CLI host from the sibling repo:

```sh
uv tool install lisa-host-cli
```

This provides the `lisa` entry point, which wraps the transport-neutral
`host` runtime with console rendering and a REPL loop.

## Quick start

From the repo root:

```sh
uv sync --extra dev
uv run lisa verify        # smoke-check the installed runtime
uv run lisa modes         # list available modes
uv run lisa models discover
uv run lisa models apply
uv run lisa once "summarize the current thread state"
uv run lisa               # enter interactive mode
```

The first `lisa models discover` / `lisa models apply` cycle generates and
activates the model manifest (see [Configuration](#configuration)). Once that is
in place, the runtime can answer natural-language turns through whichever
provider the manifest points at.

## Configuration

Lisa resolves configuration from environment variables and generated
artifacts under the data directory (`artifacts/` by default, or the path in
`LISA_DATA_DIR`).

- **Model manifest** — `artifacts/models.yaml` declares the `default` and
  `reasoning` model designations. Run `lisa models discover` to probe
  configured providers, then `lisa models apply` to write the active manifest.
  See [`docs/ROUTER_DESIGN.md`](docs/ROUTER_DESIGN.md) for the routing contract.
- **Content roots** — `SDK_CONTENT_ROOTS` is a `;`-separated list of extra pack
  roots. Local action/chat packs discovered this way run when they only need the
  shipped callbacks; heavier workflow modes report the missing host wiring
  they require.
- **Thread store** — append-only JSONL under the data dir. `lisa-pack-memory`
  is bundled and enabled by default.
- **Approval policy** — `config/policy.yaml` decides which actions need
  operator approval: the `ask`/`trust`/`plan` dial, sandbox auto-approval, and
  allow/ask/deny rules. See
  [`docs/POLICY_GUIDE.md`](docs/POLICY_GUIDE.md) for the authoring reference.

## Commands

Host commands are slash-prefixed. Built-ins ship in the host runtime; packs
register their own through the `lisa.extensions.host_commands` entry point.

| Command | Purpose |
| --- | --- |
| `lisa verify` | Smoke-check the installed runtime, manifest, and packs |
| `lisa modes` | List discovered modes and their owning packs |
| `lisa models` | Show the active model manifest |
| `lisa models discover` | Probe providers and propose a manifest |
| `lisa models apply` | Activate the proposed manifest |
| `lisa once "<prompt>"` | Run a single turn and exit |
| `lisa` | Enter interactive mode |
| `/resume [reply]` | Resume from the most recent yielded thread entry |

Pack-registered commands (for example `/search`, `/markdown`, `/discuss`,
`/help`) are added through `SDK_CONTENT_ROOTS` and the bundled packs.

## MCP servers

The `mcp_server` package ships first-party [MCP][mcp] servers that expose
Lisa's surface-scoped tools over stdio. Each is an independent entry point:

| Server | Entry point | Surface |
| --- | --- | --- |
| `lisa-context` | `mcp_server.context.server:main` | Context assembly and retrieval |
| `lisa-content` | `mcp_server.content.server:main` | Content loading and rendering |
| `lisa-catalog` | `mcp_server.catalog.server:main` | Mode/operation catalog |
| `lisa-memory` | `mcp_server.memory.server:main` | Surface-scoped memory store/recall |
| `lisa-guard` | `mcp_server.guard.server:main` | Privacy and scope guardrails |
| `lisa-storage` | `mcp_server.storage.server:main` | Blob/object storage |
| `lisa-thread` | `mcp_server.thread.server:main` | Append-only thread store |

## Packs

Packs are the unit of authoring. A pack contributes modes, operations, authored
content, and pack-registered slash commands under a single namespace. Bundled
reference packs:

- **`lisa-pack-core`** — baseline mode (`default`) and shared
  prompt assets.
- **`lisa-pack-memory`** — memory-backed learning operations (`learn`, memory
  ops). Enabled by default.
- **`lisa-pack-introspect`** — operational self-model and reflective
  self-assessment operations.
- **`lisa-pack-recovery`** — thread-based `/resume` and recovery records.

External integration packs live in the public
[`lisa-packs`][lisa-packs] repo (`brave`, `cloudservers`, `letta`, `markdown`,
`mem0`, `okf`, `personas`, `search`, `serper`, `tavily`) and the WIP
[`packs-wip`][packs-wip] repo, all under the
`lisa_packs.*` namespace.

## Model providers

Lisa supports live completion through any OpenAI-compatible endpoint plus the
first-party SDK providers:

- `anthropic`
- `azure_openai`
- `google`
- `ollama` (local)
- `openai`
- `openrouter`

Provider credentials and endpoints are resolved from the environment in the
usual way for each SDK (for example `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
`AZURE_OPENAI_*`). The active model manifest in `artifacts/models.yaml` pins
which model the `default` and `reasoning` designations resolve to.

## Privacy and surface scope

The public contracts model privacy explicitly — prompt wording is **not** a
privacy control. Hosts must enforce scope in storage keys, retrieval filters,
and delivery routing:

- `SurfaceContext` carries `surface_type`, `surface_id`, and `memory_scope`.
- Memory writes and recalls are scoped by the active surface and memory scope.
- Shared surfaces must not receive personal context unless a host implements
  explicit, audited sharing semantics.
- Proactive delivery targets are host-owned and must be explicit.

See [`docs/BOUNDARIES.md`](docs/BOUNDARIES.md) for the full privacy boundary.

## Validation

The local loop and CI run the same commands in the same order — there is no
warn-only stage. Run all checks from the repo root:

```powershell
# Format check
uv run ruff format --check kernel sdk mcp_server hosts

# Lint
uv run ruff check kernel sdk mcp_server hosts

# Type check
uv run mypy kernel sdk mcp_server

# Dead code check
uv run vulture kernel sdk mcp_server hosts vulture_allowlist.py

# Tests (no live API calls)
uv run pytest -m "not live" --tb=short -v

# Smoke test
uv run lisa verify
```

Focused sanity checks:

```powershell
uv run python -c "import kernel, sdk; print(kernel.__file__); print(sdk.__file__)"
uv run lisa modes
uv run lisa models
uv run pytest --import-mode=importlib kernel/tests sdk/tests hosts -m "not live" -v --tb=short
uv run --extra dev ruff check --select F,I kernel sdk mcp_server hosts
```

Tests marked `live` hit real external signal or provider APIs and are skipped
by default; `integration` tests are multi-module in-process and may require
local services; `eval` tests are deterministic agent regression evals with no
model calls or I/O.

## Contributing

Contributions are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the
full checklist, and [`docs/CODING_WORKFLOW.md`](docs/CODING_WORKFLOW.md) for the
coding pack workflow. In summary:

1. Fork and clone the repo, then `uv sync --extra dev`.
2. Make changes on a feature branch.
3. Run the [validation](#validation) suite from the repo root — any failure
   blocks.
4. Open a pull request with breaking changes noted in the description and
   [`CHANGELOG.md`](CHANGELOG.md).

Keep changes scoped to the current task. Follow the existing module boundaries
and naming patterns; prefer the smallest diff that fixes the root cause and
verify it with the relevant tests. See [`docs/ROADMAP.md`](docs/ROADMAP.md)
for the current work horizons.

## License

Lisa Framework is released under the [MIT License](LICENSE), © 2026 The Migus
Group.

[lisa-host-cli]: https://github.com/Lisa-Framework/lisa-host-cli
[lisa-packs]: https://github.com/Lisa-Framework/lisa-packs
[mcp]: https://modelcontextprotocol.io
[uv]: https://docs.astral.sh/uv

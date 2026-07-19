# Pack Authoring Guide

Packs add modes, operations, tools, authored content, signal connectors, and processing loops without changing host code. The framework discovers packs through `lisa.extensions.content_roots` entry points or through host-injected content roots.

## Packs vs. Extensions

Packs and extensions serve different architectural roles:

- **Packs** own *workflow* and *personality*: `_mode.md` routing, allowed operations, prompt templates, curated session contexts, and host commands.
- **Extensions** own *ambient telemetry*: background data collection (`SignalCollector`) and post-collection processing (`ProcessingLoop`). They run in the bottom half (a one-shot process triggered by a systemd timer or cron), pulling ambient state (e.g. GitHub PRs, Stripe billing) without waiting for a user query.
- **MCP Servers** own *actions* and *storage*: client-invoked tools, file access, search, and memory.

A single pack can provide all three: modes, operations, and signal connectors. The `lisa-pack-cloudflare`, `lisa-pack-github`, and `lisa-pack-stripe` packs do exactly that.

## Pack Locations

Packs live in one of two places depending on whether they are bundled with the framework or maintained as an external dependency:

| Location | Packages | Import namespace | When to use |
| --- | --- | --- | --- |
| `sdk/packs/<name>` | `lisa-pack-core`, `lisa-pack-memory`, `lisa-pack-models`, `lisa-pack-search`, `lisa-pack-checkpoint` | `sdk.packs.<name>` | Bundled reference packs shipped with the framework |
| `../lisa_packs/<name>` (repo: `github.com/lisa-framework/packs`) | `lisa-pack-brave-search`, `lisa-pack-cloudflare`, `lisa-pack-coding`, `lisa-pack-github`, `lisa-pack-gmail`, `lisa-pack-mem0`, `lisa-pack-stripe` | `lisa_packs.<name>` | External integration packs maintained in a separate repo |

The framework `pyproject.toml` consumes external packs via path dependencies (`{ path = "../lisa_packs/<name>" }`). Bundled packs are uv workspace members. Both are installed editable in the venv and discovered through entry points at runtime.

## Pack Shape

A pack normally contains:

```plaintext
sdk/packs/example/
  __init__.py
  pyproject.toml
  modes/
    example/
      _mode.md
      ask.md
      answer.md
  operations/
    example_operation.yml
    example_operation.py
  content/
    prompts/
      example.md
  tests/
    test_import_boundaries.py
    test_catalog.py
```

A pack can provide only modes, only operations, only content, or any combination of those.

## Mode Files

Every mode folder needs an `_mode.md` file with YAML front matter:

```markdown
---
mode_id: example
start_stage_id: ask
mode_kind: workflow
require_query: true
command_pattern: "^(?:/)?example(?:[\\s,:;]+(?P<query>.+))?$"
allowed_operation_ids: []
description: "Example workflow mode."
---

# Example Mode

Mode-level instructions go here.
```

Use `mode_kind: chat` for normal conversational modes, `mode_kind: workflow` for finite-state modes, and `mode_kind: action` for one-shot commands.

Optional routing metadata can also be authored in `_mode.md` front matter:

```markdown
---
routable: true
priority: 20
fast_path_patterns:
  - "^search\\b"
routing_intent_tags: [research]
routing_intent_keywords:
  research:
    - latest
    - news
satisfies_intents: [research]
---
```

- Use `fast_path_patterns` only for high-precision regex candidates suitable for verifier confirmation.
- `routing_intent_tags` declares the intents a mode **covers** when routing **to** it (e.g. `search` covers `research`, `markdown` covers `write`). The proactive routing hint generator uses this to pick a single-handoff target or a `fork:` fan-out.
- `routing_intent_keywords` maps each intent tag to the broad substring keywords that signal that specific intent in a user query (case-insensitive substring match, not anchored regex).
- `satisfies_intents` is the symmetric counterpart: the intents a mode **handles in-place** when it is the **current** mode, so the hint generator knows not to suggest a handoff for them. Author the same intent-tag vocabulary you use in `routing_intent_tags`. Default empty means every detected intent is a gap → a handoff hint is offered (the safe direction for modes that have not opted in, e.g. `default`). **Author this for every mode with a non-trivial tool surface** — otherwise the kernel will offer handoffs for intents your mode can already handle. For example, `coding/_mode.md` should declare `satisfies_intents: [coding]` (explicitly **not** `write` or `learn`), so a "write a markdown document" request from coding mode triggers a handoff to `markdown` rather than falling back to `coding_create_worktree`. The kernel never names tool IDs or hardcodes mode-specific intent semantics — the pack makes the claim and the kernel believes it.

### Extension Front Matter (`extra_front_matter`)

The kernel owns a fixed set of front-matter keys (routing, FSM, operation bindings, workshop config). Every other key in `_mode.md` front matter is **extension data** the kernel does not interpret: it is collected into `WorkflowModeConfig.extra_front_matter` and passed through untouched. Hosts and packs read `extra_front_matter` directly to pick up app-owned concepts like `lenses` (mode perspectives) without the kernel ever naming or curating them.

```markdown
---
mode_id: ponder
lenses: [lens.conversation, lens.reflection, lens.founder_journal]
...
---
```

The kernel never validates or allowlists extension keys — it only knows what it owns and protects. Adding a new app-owned front-matter key requires no kernel change.

### Front-Matter Overrides (`_overrides.yaml`)

Apps that need to manage extension front matter (or a small set of kernel-owned but overridable fields) without editing pack-owned `_mode.md` files can place an `_overrides.yaml` file beside the `_mode.md`:

```plaintext
sdk/packs/example/modes/example/
  _mode.md
  _overrides.yaml
```

```yaml
# _overrides.yaml
lenses: [lens.conversation, lens.reflection]
description: "Overridden description for the app's deployment."
```

Overrides are deep-merged on top of the `_mode.md` front matter at load time. The boundary is a **denylist**, not an allowlist: the kernel names only the keys it owns and protects (routing, FSM, operation bindings, workshop config). Attempting to override any kernel-owned key is rejected with a startup validation error. Every other key is app/pack-owned and passes through untouched — the kernel does not name, interpret, or allowlist those keys.

Kernel-owned keys that overrides may **not** touch include: `mode_id`, `start_stage_id`, `mode_kind`, `command_pattern`, `require_query`, `entry_operation_ids`, `pre_turn_operation_ids`, `post_turn_operation_ids`, `allowed_operation_ids`, `routable`, `priority`, `fast_path_patterns`, `routing_intent_tags`, `routing_intent_keywords`, `satisfies_intents`, `workshop_panels`, `workshop_budget`, `workshop_planner_prompt_file`, `workshop_synthesis_prompt_file`, `workshop_rendering`, `default_intent_bridges`, `artifact_type`.

Kernel-owned keys that overrides **may** replace (read into typed fields but not security-sensitive): `description`, `default_model`.

List values from overrides replace (not concatenate) list values from the base front matter, so extension sets stay deterministic.

## Operations

Operations are the behavior layer between modes and tools. A pack contributes
operations as colocated YAML + Python pairs (runtime) or as `.operation.md` Markdown
files (declarative). The Hello World example below shows the minimal shape; the mode
front matter decides when operations are invoked.

- For the conceptual model, lifecycle hooks, and the operation-vs-tool distinction
  see [OPERATIONS.md](OPERATIONS.md).
- For the full metadata reference, examples with tool calls, response models, and the
  testing checklist see [OPERATION_AUTHORING_GUIDE.md](OPERATION_AUTHORING_GUIDE.md).

## Content Rules

Authored text should live under `content/` and be loaded by host-provided readers. Avoid hard-coded absolute paths. Keep pack content generic unless the pack is explicitly private.

### Curated Session Contexts

Packs can provide customized reference contexts via `content/user_contexts/<pack_name>_optimal.md` (or any other descriptive name). Since these files are loaded through the registry during `--session-context` resolution, users can quickly invoke your curated context.

By including a curated context, pack authors can natively inject "Instructions", "You (Lisa)", and role configuration that best utilize their tools! Pack-supplied session contexts are resolved as explicit local files via `--session-context <path>`. The session context front matter must conform to the v2 strict schema (see the [Session Context Authoring Guide](SESSION_CONTEXT_AUTHORING_GUIDE.md)).

## Hello World Pack

The fastest end-to-end path is a local external pack folder. The CLI host discovers these packs from `SDK_CONTENT_ROOTS`, so you do not need a private app or a published wheel just to prove the pack shape.

Create this folder outside the framework repo (example path shown):

```plaintext
C:\Code\lisa-pack-hello-world\
  __init__.py
  modes/
    hello/
      _mode.md
  operations/
    hello_operation.py
    hello_operation.yml
```

`C:\Code\lisa-pack-hello-world\__init__.py`

```python
"""Hello world pack."""
```

`C:\Code\lisa-pack-hello-world\modes\hello\_mode.md`

```markdown
---
mode_id: hello
start_stage_id: hello
mode_kind: action
require_query: false
command_pattern: '^(?:/)?hello(?:[\s,:;]+(?P<query>.+))?$'
entry_operation_ids: [hello_operation]
allowed_operation_ids: [hello_operation]
description: "Hello world action mode."
---

# Hello
```

`C:\Code\lisa-pack-hello-world\operations\hello_operation.yml`

```yaml
operation_id: hello_operation
description: Say hello from a custom pack.
mode_allowlist: [hello]
timeout_seconds: 30
```

`C:\Code\lisa-pack-hello-world\operations\hello_operation.py`

```python
from kernel.models import OperationExecutionContext, OperationResult


async def execute(ctx: OperationExecutionContext) -> OperationResult:
    return OperationResult(output_text=f"Hello from custom pack: {ctx.query or 'hello'}")
```

From `lisa_framework/` on Windows:

```powershell
$env:SDK_CONTENT_ROOTS = "C:\Code\lisa-pack-hello-world"
uv run lisa modes
uv run lisa once "/hello framework"
```

Expected outcome:

- `modes` shows `hello` under supported modes.
- `/hello framework` prints `Hello from custom pack: framework`.

Use single quotes around regex-based `command_pattern` values in YAML front matter so backslashes like `\s` are not treated as YAML escape sequences.

## Installed Pack Discovery

For a packaged pack, export the content root from `__init__.py` and register it as an entry point. The framework kernel uses the same mechanism for installed public packs and extensions.

```python
from pathlib import Path


def get_content_root() -> Path:
    return Path(__file__).parent
```

```toml
[project.entry-points."lisa.extensions.content_roots"]
hello_world = "sdk.packs.hello_world:get_content_root"
```

## Extension Entry Points

Packs register extensions through entry-point groups in `pyproject.toml`. The active groups are:

| Entry-point group | Purpose |
| --- | --- |
| `lisa.extensions.content_roots` | Declare the pack's content root for mode/operation/tool discovery |
| `lisa.extensions.signal_connectors` | Register `SignalCollector` subclasses for background data collection |
| `lisa.extensions.processing_loops` | Register `ProcessingLoop` implementations for post-collection analysis |
| `lisa.extensions.host_commands` | Register slash-command handlers (`/checkpoint`, `/restore`, etc.) |
| `lisa.extensions.context_evidence` | Register context-evidence builders for prompt injection |

Hosts decide which groups they discover and how often extension outputs are collected.

## Signal Connectors

A signal connector is a stateless, repeatable monitor for an external system. Each connector:

- Defines what configuration is required (env vars or settings)
- Implements `collect()` to fetch fresh data from the external system
- Returns a typed Pydantic snapshot
- Provides fallback snapshots for misconfiguration or errors

Signal connectors are invoked by the bottom-half process (`little-lisa bottom --tick`), not by the turn loop. The top half reads the resulting `UnifiedState` via the host's state provider.

### Hello World Signal Connector

A minimal signal connector package:

```plaintext
C:\Code\lisa-pack-hello-world\
  __init__.py
  connector.py
  models.py
  pyproject.toml
```

`models.py`

```python
from pydantic import BaseModel


class HelloWorldSnapshot(BaseModel):
    collected_at: str
    status: str = "ok"
    message: str = "hello world"
```

`connector.py`

```python
import httpx

from kernel.intake.connectors import SignalCollector

from .models import HelloWorldSnapshot


class HelloWorldSignalCollector(SignalCollector[HelloWorldSnapshot]):
    collector_id = "hello_world"
    display_name = "Hello World"

    async def collect(
        self,
        client: httpx.AsyncClient,
        collected_at: str,
    ) -> HelloWorldSnapshot:
        del client
        return HelloWorldSnapshot(
            collected_at=collected_at, status="ok", message="hello world"
        )

    def build_unconfigured_snapshot(self, collected_at: str) -> HelloWorldSnapshot:
        return HelloWorldSnapshot(
            collected_at=collected_at,
            status="unconfigured",
            message="missing configuration",
        )

    def build_error_snapshot(self, collected_at: str) -> HelloWorldSnapshot:
        return HelloWorldSnapshot(
            collected_at=collected_at,
            status="error",
            message="connector failed",
        )
```

`pyproject.toml`

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["."]
include = ["sdk.packs.hello_world*"]

[project]
name = "lisa-pack-hello-world"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["lisa-kernel", "httpx>=0.27.0"]

[project.entry-points."lisa.extensions.signal_connectors"]
hello_world = "sdk.packs.hello_world.connector:HelloWorldSignalCollector"
```

Validate through the public extension harness:

```powershell
uv pip install -e "C:\Code\lisa-pack-hello-world"
uv run python -c "from kernel.testing.extension_harness import list_extension_entries; print([entry.model_dump() for entry in list_extension_entries() if entry.entry_id == 'hello_world'])"
uv run python -c "from kernel.testing.extension_harness import build_collector_contract_preview, load_signal_collector; connector = load_signal_collector('hello_world'); print(build_collector_contract_preview(connector))"
```

Those commands prove entry-point discovery and contract shape without relying on a live external API.

## Processing Loops

A processing loop runs after signal collection and produces a `ProcessedState` that the bottom half merges into `UnifiedState`. Common uses:

- Anomaly detection (examine snapshots, emit `active_alerts`)
- Trend analysis (examine snapshots, emit `intel_collections`)
- State derivation (examine snapshots, emit `signal_snapshots`)

```python
from kernel.intake.contracts import ProcessContext, ProcessedState, ProcessingLoop


class HelloWorldLoop:
    loop_id = "hello_world_loop"

    async def process(self, ctx: ProcessContext) -> ProcessedState:
        return ProcessedState(
            loop_id=self.loop_id,
            metadata={"greeting": "hello from processing loop"},
        )
```

Register in `pyproject.toml`:

```toml
[project.entry-points."lisa.extensions.processing_loops"]
hello_world_loop = "sdk.packs.hello_world.loop:HelloWorldLoop"
```

## Host Commands

Packs can register slash commands that the host dispatch layer consults after its own built-in commands. Each command is a `HostCommandSpec` with a prefix, handler, and help text.

```python
from kernel.host_commands import HostCommandSpec


async def handle_hello(runtime, stripped: str, message: str) -> str:
    return f"Hello from pack: {stripped or 'world'}"


hello_command = HostCommandSpec(
    prefix="/hello-pack",
    handler=handle_hello,
    help_text="Say hello from a custom pack.",
)
```

Register in `pyproject.toml`:

```toml
[project.entry-points."lisa.extensions.host_commands"]
hello_world = "sdk.packs.hello_world.commands:hello_command"
```

## Host Boundary

Packs and extensions should return typed snapshots, contribution fragments, or evidence records. They should not send chat messages, choose proactive recipients, own production storage, or import private host code. A host scheduler or runtime adapter owns those responsibilities.

Keep imports limited to `kernel`, `sdk`, and package-local modules. Add import-boundary tests to verify no private app imports.

## Testing Checklist

1. Import the package without private app modules.
2. Load the mode catalog and operation catalog.
3. Validate operation allowlists against discovered modes.
4. Run deterministic unit tests without live APIs.
5. Run the CLI host for one command path when practical.
6. Add opt-in live tests only for integrations that need them.
7. For signal connectors: validate through `kernel.testing.extension_harness` before wiring into a host.
8. Add import-boundary tests.

Focused local checks from `lisa_framework/`:

```powershell
uv run lisa modes
uv run lisa once "/markdown Draft a quick checklist"
uv run lisa once "/hello framework"  # after setting SDK_CONTENT_ROOTS
uv run ruff check sdk mcp_server hosts
```

Run extension tests without live API calls by default:

```powershell
uv run pytest sdk/packs/github/tests -m "not live"
uv run pytest sdk/packs/cloudflare/tests -m "not live"
uv run pytest sdk/packs/stripe/tests -m "not live"
```

Live tests must be explicit, opt-in, and secret-safe. Do not make live tests part of default local or CI validation.

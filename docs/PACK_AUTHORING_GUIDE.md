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
| `sdk/packs/<name>` | `lisa-pack-core`, `lisa-pack-introspect`, `lisa-pack-memory`, `lisa-pack-recovery` | `sdk.packs.<name>` | Bundled reference packs shipped with the framework |
| [`lisa-packs`](https://github.com/Lisa-Harness/lisa-packs) (`../lisa-packs/<name>` in a sibling checkout) | `lisa-pack-brave`, `lisa-pack-markdown`, `lisa-pack-okf`, `lisa-pack-search` | `lisa_packs.<name>` | External integration packs maintained in separate repos; public subset in lisa-packs |

See the [lisa-packs repo](https://github.com/Lisa-Harness/lisa-packs) for the public pack set, per-pack installation, and the submodule workflow.

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
- `satisfies_intents` is the symmetric counterpart: the intents a mode **handles in-place** when it is the **current** mode, so the hint generator knows not to suggest a handoff for them. Author the same intent-tag vocabulary you use in `routing_intent_tags`. Default empty means every detected intent is a gap → a handoff hint is offered (the safe direction for modes that have not opted in, e.g. `default`). **Author this for every mode with a non-trivial tool surface** — otherwise the kernel will offer handoffs for intents your mode can already handle. For example, `coding/_mode.md` should declare `satisfies_intents: [coding]` (explicitly **not** `write` or `learn`), so a "write a markdown document" request from coding mode triggers a handoff to `markdown` rather than falling back to `coding_create_worktree`. The kernel never names tool IDs or hardcodes mode-specific intent semantics — the pack makes the claim and the kernel believes it. **Chaining**: after a mid-turn handoff the kernel recomputes the hint against the new mode's `satisfies_intents`, so the remaining unfulfilled intents drive the *next* handoff. This lets an ad-hoc multi-step workflow (e.g. search → coding → markdown) chain one handoff per intent: each mode satisfies its intent, the gap shrinks, and the next hint names the next target. Author `satisfies_intents` for *every* routable action/chat mode (`search: [research]`, `markdown: [write]`, `learn: [learn]`, ...) so the gap shrinks correctly per step; without it the gap stays full and the kernel re-offers the same handoff or suggests a redundant fork.
- `required_capabilities` declares the `ModelCapability` strings (`tools`, `code`, `reasoning`, ...) the resolved model **must** declare for the mode to run safely. The turn runner validates the resolved model's `ManifestModelEntry.capabilities` against this list **before** the tool loop starts. When a model router is wired and the model is missing a required capability, the turn returns a clear user-facing block instead of letting a weak model reach for write-effect tools it cannot use reliably (e.g. `gemma4:cloud` calling `coding_create_worktree` for a "write a roadmap" request). When no router is wired the gate is advisory (no block) — a host without a router has no escalation path. Default empty means no capability gate (every model is admitted, backward compatible). Pair this with a router rule like `required_capabilities includes "code" → gpt-5.3-codex` so the router escalates to a capable model; the gate is the safety net behind the router. Author the same `ModelCapability` vocabulary used in `models.yaml`.

### Turn policy front matter

These optional keys decide what a mode does at the two lifecycle boundaries a pack cannot express with `effect:` declarations alone: after an approved action, and at turn completion.

```markdown
---
fork_allowed: true
yield_policy:
  allowed: false
  fallback_mode: discuss
approval_continuation: resume
task_completion: checked_commit
task_artifact_tool_id: coding_create_worktree
---
```

- `yield_policy` is a mapping (`kernel.models.YieldPolicy`): `allowed` (bool, default `true`), `fallback_mode` (str, default empty), `is_default_yield_target` (bool, default `false`), and `default_question` (str, default empty). It decides whether the model may emit the unified turn-lifecycle `yield` action in this mode. Action modes set `allowed: false` and name a `fallback_mode` (typically a chat mode such as `discuss`) so a forced or system yield is rerouted instead of asking the user inline; a mode that must be able to ask the user keeps the default and does **not** set `fork_allowed`.

- `fork_allowed` (bool, default `false`) is the mode's consent to be a **fork target** (parallel fan-out). A fork child runs with no interactive channel — the user cannot answer a mid-turn question — so a mode that opts in commits to completing or failing without asking. The routing fork decision and the fork enforcer both refuse a fork when any candidate target lacks this. A fork child's `run_turn` receives an unconditional `yield_policy_override` that disallows `yield`, so the `yield` tool is never advertised to a fork child. A mode that may need to ask the user must leave this `false`; it remains reachable via `delegate` (sequential, parent-mediated), where a child yield propagates to the user through the parent.

- `approval_continuation` (`"report"` default, or `"resume"`) decides what the post-approval turn does in this mode. `report` (default) reports the approved action's result in plain prose and stops — a one-shot host process would otherwise pause for a second approval on any new effectful action. `resume` continues the request that raised the approval: the continuation turn is told not to re-run the approved action, is given the original request, and continues from where it paused using the approved action's result. Set this on modes whose work legitimately spans an approval (e.g. an executor that must apply a patch, run checks, and commit). Policy auto-approved actions resume regardless of this field. `report` and `resume` are the only accepted values; anything else is a startup `WorkflowValidationError`.

- `task_completion` (`"none"` default, or `"checked_commit"`) and `task_artifact_tool_id` (str, default empty) opt a mode into the **checked-commit completion contract**. With `task_completion: checked_commit`, the post-turn completion contract no longer accepts the turn merely because a write receipt landed: the turn must own a **verified commit on a real, clean worktree** for the current task. Set `task_artifact_tool_id` to the producer tool that creates that worktree (the artifact store's producer). The turn runner collects the turn's effect receipts whose capability is the producer, and:
  - exactly one artifact must resolve — zero or more than one fails closed with "Cannot identify one worktree for this task";
  - a task identity and a wired `verify_task_completion` host service must exist (the bare/test host that wires neither fails closed);
  - the resolved `WorkProduct` must belong to the identity, match the task id, be `committed` / `pushed` / `pr_opened`, have a revision ≥ 1, changed paths, `commit_sha != base_sha`, `head_sha == commit_sha`, a branch, and a worktree path;
  - the verifier re-derives the tree **from Git itself** in the owned worktree — same `--git-common-dir` as the source repo, `HEAD` / `HEAD^` matching the artifact's `commit_sha` / `base_sha`, the tree differing from the parent tree, the recorded branch matching, `git status --porcelain --untracked-files=all` clean, and the recorded changed paths a subset of the actual committed paths;
  - every `required_check_id` must be `passed`, at the artifact's current `revision`, **and** carry `checked_tree_sha ==` the committed tree — checks must have run against the committed content, not a pre-commit working tree.

  When verification fails the model's prose is replaced with the verifier's reason, the turn is blocked, and `block_origin` is set to `task_completion` (a resumable task state). This is what makes a persona's "finish only after a real commit with required checks passing" instruction enforceable rather than a request the model can ignore.

These keys are **kernel-owned** (parsed into typed `WorkflowModeConfig` fields, so they never land in `extra_front_matter` and must not be used for app-owned data), with one difference in the overrides boundary: `yield_policy`, `approval_continuation`, `task_completion`, and `task_artifact_tool_id` are also **protected** — they are rejected in `_overrides.yaml` — whereas `fork_allowed` is overridable, like `description` and `default_model`. They are additive and backward compatible: an existing mode keeps the safe defaults (`yield_policy` allowed, `fork_allowed: false`, `approval_continuation: report`, `task_completion: none`). `task_artifact_tool_id` is required only when `task_completion: checked_commit` — an empty producer fails closed rather than silently passing. The runtime side of the checked-commit contract lives in `kernel/src/kernel/turn_phases/_completion_contract.py` (decision resolver), `kernel/src/kernel/turn_runner/__init__.py` (evidence collection), and the host's `verify_task_completion` service (the runtime wires `verify_checked_commit` in `runtime/src/runtime/code_artifacts.py`); approval continuation is resolved in `runtime/src/runtime/approval/orchestration.py`.

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

Overrides are deep-merged on top of the `_mode.md` front matter at load time. The boundary is a **denylist**, not an allowlist: the kernel names only the keys it owns and protects (routing, FSM, operation bindings, turn policy). Attempting to override any kernel-owned key is rejected with a startup validation error. Every other key is app/pack-owned and passes through untouched — the kernel does not name, interpret, or allowlist those keys.

Kernel-owned keys that overrides may **not** touch include: `mode_id`, `start_stage_id`, `mode_kind`, `command_pattern`, `require_query`, `entry_operation_ids`, `entry_op_kind`, `pre_turn_operation_ids`, `post_turn_operation_ids`, `allowed_operation_ids`, `allowed_tool_ids`, `tools_authoritative`, `accept_default_capabilities`, `excluded_operation_ids`, `excluded_tool_ids`, `routable`, `priority`, `fast_path_patterns`, `routing_intent_tags`, `routing_intent_keywords`, `satisfies_intents`, `required_capabilities`, `default_intent_bridges`, `artifact_type`, `yield_policy`, `approval_continuation`, `task_completion`, `task_artifact_tool_id`.

Kernel-owned keys that overrides **may** replace (read into typed fields but not security-sensitive): `description`, `default_model`, `fork_allowed`.

List values from overrides replace (not concatenate) list values from the base front matter, so extension sets stay deterministic.

### Exportable and default-available capabilities

Operations and tools declare two visibility flags in their YAML front matter:

- **`exportable`** (default `false`) — when `true`, the capability may be referenced by modes in other packs. When `false`, it is private to its owning pack; a foreign mode referencing it is rejected at startup by `validate_export_boundaries`.
- **`default_available`** (default `false`) — when `true`, the SDK catalog loader auto-merges the capability id into every mode's effective `allowed_operation_ids` / `allowed_tool_ids` (unless the mode opts out). Lets a pack ship ambient capabilities (e.g. `recall_memory`) without forcing every mode author to re-list them.

```yaml
# operation YAML
operation_id: recall_memory
exportable: true
default_available: true
```

```yaml
# tool YAML (.tool.yml)
tool_id: web_search
exportable: true
# default_available omitted — modes opt in via allowed_tool_ids
```

Modes opt out of the ambient merge via `accept_default_capabilities: false` (disables all defaults) or `excluded_operation_ids` / `excluded_tool_ids` (subtracts specific ids). The kernel distinguishes the authored allowlists from the merged effective ones so that `mode_completion_effects` (`kernel/operations/registry.py`) checks only the authored set plus `mode_allowlist`-admitted capabilities.

#### MCP-native ambient tools

MCP-native tools (discovered at runtime from server subprocesses) have no pack `.tool.yml`, so they cannot declare `default_available` in front matter. Instead, the host manifest (`mcp_servers.yaml`) declares ambient MCP-native tools per server:

```yaml
# runtime/mcp_servers.yaml
- server_id: lisa-storage
  runtime_kind: venv_bin
  venv_dir: "${FRAMEWORK_ROOT}"
  default_available_tools:
    - read_file      # bare name; merged as lisa-storage.read_file
```

The host resolves these into namespaced ids (`server_id.tool`, or `<pack_slug>.<server_id>.tool` for pack-contributed servers) and passes them to the catalog server via the `LISA_DEFAULT_MCP_TOOLS` env var. The SDK catalog loader unions them into every mode's `allowed_tool_ids` with the same opt-out / idempotent semantics as pack `default_available: true` tools. The turn-runner's existing MCP-tool filter (which keys on namespaced ids) then admits them with no further change.

## Operations

Operations are the behavior layer between modes and tools. A pack contributes
operations as colocated YAML + Python pairs (runtime) or as `.operation.md` Markdown
files (declarative). The Hello World example below shows the minimal shape; the mode
front matter decides when operations are invoked.

- For the conceptual model, lifecycle hooks, and the operation-vs-tool distinction
  see the Operations section in [`../kernel/README.md`](../kernel/README.md#operations).
- For the full metadata reference, examples with tool calls, response models, and the
  testing checklist see [`../sdk/AUTHORING.md`](../sdk/AUTHORING.md).

## Content Rules

Authored text should live under `content/` and be loaded by host-provided readers. Avoid hard-coded absolute paths. Keep pack content generic unless the pack is explicitly private.

### Curated Session Contexts

Packs can provide customized reference contexts via `content/user_contexts/<pack_name>_optimal.md` (or any other descriptive name). Since these files are loaded through the registry during `--session-context` resolution, users can quickly invoke your curated context.

By including a curated context, pack authors can natively inject "Instructions", "You (Lisa)", and role configuration that best utilize their tools! Pack-supplied session contexts are resolved as explicit local files via `--session-context <path>`. The session context front matter must conform to the v2 strict schema (see the [Session Context Authoring Guide](SESSION_CONTEXT_AUTHORING_GUIDE.md)).

### Inheritable Pack Sessions (`sessions/`)

A pack can ship session files that user sessions inherit block-by-block via the `from:` and `body:` directives (see the [Session Context Authoring Guide](SESSION_CONTEXT_AUTHORING_GUIDE.md#inheritance-via-from-and-body)). Place these under `<pack_root>/sessions/<name>.md`. For example, `lisa-pack-core` ships `sessions/default.md`, which a user session can pull from:

```yaml
# user's session_context.md
routing:
  from: { pack: lisa-pack-core, session: default }
```

Convention rules:

- A pack session file is a strict v2 session manifest (`schema_version: 2`) with a non-empty Markdown body. It is compiled through the same strict `SessionManifestV2` schema as a user session.
- **Pack sessions are leaves.** They cannot themselves use `from:` or `body:` — the strict schema's `extra="forbid"` rejects them. Inheritance is one level: user → pack, never pack → pack.
- A pack session does **not** need to declare `resources.packs.required` (it is not a runnable session on its own); it only needs to carry the blocks a user session might want to inherit.
- The `pack` reference in a user's `from:` accepts the full distribution ID (`lisa-pack-core`) or the bare slug (`core`). The pack must be installed (discoverable via content roots) at the time the user session is compiled.

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
| `lisa.extensions.host_commands` | Register slash-command handlers (`/resume`, `/stalls`, etc.) |
| `lisa.extensions.context_items` | Register context-item builders for prompt injection |

Hosts decide which groups they discover and how often extension outputs are collected.

## Signal Connectors

A signal connector is a stateless, repeatable monitor for an external system. Each connector:

- Defines what configuration is required (env vars or settings)
- Implements `collect()` to fetch fresh data from the external system
- Returns a typed Pydantic snapshot
- Provides fallback snapshots for misconfiguration or errors

Signal connectors are invoked by the bottom-half process (`lisa intake`), not by the turn loop. The top half reads the resulting `UnifiedState` via the host's state provider.

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
        return HelloWorldSnapshot(collected_at=collected_at, status="ok", message="hello world")

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
dependencies = ["lisa-harness-kernel", "httpx>=0.27.0"]

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

1. Run `packstest.catalog.validate_pack` over the pack from source (structure: entry points, tool/operation YAMLs, importable modules).
2. Import the package without private app modules.
3. Load the mode catalog and operation catalog.
4. Validate operation allowlists against discovered modes.
5. Run deterministic unit tests without live APIs; drive tools in-process via `packstest.runner.run_tool`.
6. Run the CLI host for one command path when practical.
7. Add opt-in live tests only for integrations that need them.
8. For signal connectors: validate through `kernel.testing.extension_harness` before wiring into a runtime.
9. Add import-boundary tests.
10. Ship at least one live end-to-end scenario under `tests/e2e/` that drives the pack's primary workflow and verifies the outcome outside Lisa; run it with `uv run --frozen lisa-e2e run <scenario.yml>`. See [`../e2etest/README.md`](../e2etest/README.md).

The `packstest` harness ([`../packstest/README.md`](../packstest/README.md)) activates packs in-process from source — no install step, no catalog subprocess, no model calls. Eval files live in the pack under `tests/evals/` with the `eval` marker.

Run extension tests without live API calls by default. External packs live in sibling checkouts (`../lisa-packs/` and `../lisa-packs-wip/`), not under `sdk/packs/`. Verify them with the `packstest` harness first (validate + in-process evals, no install step), then the pack's own suite:

```powershell
uv run pytest ../lisa-packs-wip/coding/lisa_packs/coding/tests -m "not live" -q
uv run pytest ../lisa-packs-wip/github/tests -m "not live" -q
uv run pytest ../lisa-packs-wip/cloudflare/tests -m "not live" -q
uv run pytest ../lisa-packs-wip/stripe/tests -m "not live" -q
```

Live tests must be explicit, opt-in, and secret-safe. Do not make live tests part of default local or CI validation.

### End-to-end scenarios

`packstest` proves each tool honors its contract; it cannot prove a model can
finish the workflow through routing, handoffs, approvals, memory, and the
sandbox. A pack's `tests/e2e/*.scenario.yml` does that: `lisa-e2e` boots a
real runtime in an isolated data dir, plays scripted turns against a live
model, answers approvals with real `/approve` commands, and judges the result
from receipts, stored records, and git state. The scenario format, oracle
reference, debugging loop, and fixture conventions are in
[`../e2etest/README.md`](../e2etest/README.md). Like other live tests,
scenarios are opt-in and stay out of default CI.

# Pack Authoring Guide

Packs add modes, operations, and authored content without changing host code. The framework discovers packs through `lisa.extensions.content_roots` entry points or through host-injected content roots.

## Pack Locations

Packs live in one of two places depending on whether they are bundled with the framework or maintained as an external dependency:

| Location | Packages | Import namespace | When to use |
| --- | --- | --- | --- |
| `sdk/packs/<name>` | `lisa-pack-core`, `lisa-pack-coding`, `lisa-pack-learning`, `lisa-pack-search`, `lisa-pack-checkpoint`, `lisa-pack-model-picker` | `sdk.packs.<name>` | Bundled reference packs shipped with the framework |
| `../lisa_packs/<name>` (repo: `github.com/lisa-framework/packs`) | `lisa-pack-brave-search`, `lisa-pack-cloudflare`, `lisa-pack-github`, `lisa-pack-gmail`, `lisa-pack-mem0`, `lisa-pack-stripe` | `lisa_packs.<name>` | External integration packs maintained in a separate repo |

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
route_hints:
  - "Prefer `search` when the user asks for recent external information."
examples:
  - reasoning: "User asked for fresh external data."
    selected_mode_id: search
    normalized_query: cloud costs
    clarification_needed: false
    clarification_question: ""
    direct_response: ""
    confidence_score: 0.92
---
```

- `route_hints` and `examples` are prompt-only guidance for the bounded router.
- Use `fast_path_patterns` only for high-precision regex candidates suitable for verifier confirmation.

### Extension Front Matter (`extra_front_matter`)

The kernel owns a fixed set of front-matter keys (routing, FSM, operation bindings, workshop config). Every other key in `_mode.md` front matter is **extension data** the kernel does not interpret: it is collected into `WorkflowModeConfig.extra_front_matter` and passed through untouched. Hosts and packs read `extra_front_matter` directly to pick up app-owned concepts like `lenses` (mode perspectives) without the kernel ever naming or curating them.

```markdown
---
mode_id: thinkabout
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
route_hints:
  - "Prefer `example` when the user asks for a deep-dive."
```

Overrides are deep-merged on top of the `_mode.md` front matter at load time. The boundary is a **denylist**, not an allowlist: the kernel names only the keys it owns and protects (routing, FSM, operation bindings, workshop config). Attempting to override any kernel-owned key is rejected with a startup validation error. Every other key is app/pack-owned and passes through untouched — the kernel does not name, interpret, or allowlist those keys.

Kernel-owned keys that overrides may **not** touch include: `mode_id`, `start_stage_id`, `mode_kind`, `command_pattern`, `require_query`, `entry_operation_ids`, `pre_turn_operation_ids`, `post_turn_operation_ids`, `allowed_operation_ids`, `routable`, `priority`, `fast_path_patterns`, `workshop_panels`, `workshop_budget`, `workshop_planner_prompt_file`, `workshop_synthesis_prompt_file`, `workshop_rendering`, `default_intent_bridges`, `artifact_type`.

Kernel-owned keys that overrides **may** replace (read into typed fields but not security-sensitive): `description`, `default_model`, `route_hints`, `examples`.

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

## Testing Checklist

1. Import the package without private app modules.
2. Load the mode catalog and operation catalog.
3. Validate operation allowlists against discovered modes.
4. Run deterministic unit tests without live APIs.
5. Run the CLI host for one command path when practical.
6. Add opt-in live tests only for integrations that need them.

Focused local checks from `lisa_framework/`:

```powershell
uv run lisa modes
uv run lisa once "/markdown Draft a quick checklist"
uv run lisa once "/hello framework"  # after setting SDK_CONTENT_ROOTS
uv run ruff check sdk mcp_server hosts
```

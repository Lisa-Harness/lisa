# Pack authoring guide

Packs add modes, operations, tools, authored content, MCP declarations, and
extension contributions without putting application behavior in the kernel.
Start with a deterministic source-based test, then exercise the host integration.

## Ownership and discovery

- Modes describe workflow/personality and the allowed capability surface.
- Operations implement typed Python behavior or prompt-authored tasks.
- Tools expose callable actions; MCP adapters own protocol transport.
- Context builders and signal/processing extensions contribute typed evidence.
- Hosts own credentials, transport, scheduling, approvals, and durable services.

Keep imports to kernel, SDK, and package-local code. Do not import runtime
internals or another pack. Use shared contracts/injected services instead.
See [architecture](BOUNDARIES.md), [bundled packs](../sdk/packs/README.md),
and the [external registry](https://github.com/Lisa-Harness/lisa-packs).

Content roots are exported through `lisa.extensions.content_roots`.
`SDK_CONTENT_ROOTS` can supply extra local roots (`;`-separated on every OS),
but does not install Python modules or dependencies.

## A minimal external pack

Create a checkout outside the framework with this shape:

```text
lisa-pack-hello-world/
  pyproject.toml
  lisa_packs/
    hello_world/
      __init__.py
      modes/
        hello/
          _mode.md
      operations/
        hello_operation.yml
        hello_operation.py
  tests/
```

`lisa_packs` is the external namespace; `sdk.packs` is for bundled reference
packs. Keep implementation subpackages importable and ship authored files
in the wheel.

### Distribution metadata

The checkout's `pyproject.toml`:

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "lisa-pack-hello-world"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["lisa-harness-kernel"]

[project.entry-points."lisa.extensions.content_roots"]
hello_world = "lisa_packs.hello_world:get_content_root"

[tool.hatch.build.targets.wheel]
packages = ["lisa_packs"]
```

`lisa_packs/hello_world/__init__.py`:

```python
from pathlib import Path


def get_content_root() -> Path:
    return Path(__file__).parent
```

### Mode

`lisa_packs/hello_world/modes/hello/_mode.md`:

```markdown
---
mode_id: hello
mode_kind: action
require_query: false
command_pattern: '^(?:/)?hello(?:[\s,:;]+(?P<query>.+))?$'
entry_operation_ids: [hello_operation]
allowed_operation_ids: [hello_operation]
description: Say hello through a deterministic operation.
---
# Hello

Run the greeting operation and report its result.
```

Use single-quoted YAML strings for regexes containing backslashes.
`mode_kind` is `chat` or `action`. The `workflow` kind and its authored stage
files (`{stage_id}.md` with `start_stage_id` / `allowed_next_stages`) were
retired — a staged FSM never ran in practice (no installed pack shipped stage
files, and `transitioned_to` was near-zero across two months of turns).
Multi-step work is declared with a `## Plan` block instead; see
[router design](ROUTER_DESIGN.md#durable-workflow-chains).

### Operation

`lisa_packs/hello_world/operations/hello_operation.yml`:

```yaml
operation_id: hello_operation
description: Return a greeting.
mode_allowlist: [hello]
timeout_seconds: 30
```

`lisa_packs/hello_world/operations/hello_operation.py`:

```python
from kernel.models import OperationExecutionContext, OperationResult


async def execute(ctx: OperationExecutionContext) -> OperationResult:
    return OperationResult(output_text=f"Hello: {ctx.query or 'world'}")
```

For declarative operations, tool calls, response models, error/status handling,
and the `ctx.host` seam, see [operation authoring](../sdk/AUTHORING.md).

### Validate before installation

From the framework workspace root, adjust the sibling checkout path:

```sh
uv run python -c "from pathlib import Path; from packstest.catalog import validate_pack; report = validate_pack(Path('../lisa-pack-hello-world')); print(report.summary()); assert report.ok"
```

Validation loads the real SDK scanners/entry points. It does not prove the
mode's entire model-driven workflow succeeds. Add an operation test using
`OperationExecutionContext`, catalog assertions, and a scenario where needed.

The flat namespace layout above is intentional: the current `validate_pack`
content-root locator does not infer every `src/` layout. For bundled/source-layout
packs, explicitly resolve their exported root and scan it as described in
[packstest](../packstest/README.md#source-layout-packs). A warning-only empty
scan is not evidence that the catalog was exercised.

Only after source tests pass, install through the host's `/pack install
<absolute-checkout>` workflow and select the pack in the session. Verify
the catalog with `/modes` and `/operations`; invoke `/hello framework` as
an input to that host. Use the external CLI's own command spelling rather
than assuming `uv run lisa` is available in this workspace.

## Routing and lifecycle metadata

The following are optional `_mode.md` fields:

| Field | Purpose |
| --- | --- |
| `routable`, `priority`, `fast_path_patterns` | Handoff eligibility, ordering, high-precision hint candidates |
| `routing_intent_tags`, `routing_intent_keywords` | Declare covered intents and their query vocabulary |
| `satisfies_intents` | Declare intents handled in-place so handoff gaps shrink |
| `required_capabilities` | Capabilities expected of the resolved manifest model |
| `yield_policy` | Whether/how the mode asks the user |
| `fork_allowed` | Noninteractive child eligibility; do not opt in if a path needs a question |
| `approval_continuation` | `report` versus `resume` after an approved action |
| `completion_budget_tokens` | Per-call completion ceiling |
| `turn_completion_budget_tokens` | Cumulative per-turn completion rail |
| `task_completion`, `task_artifact_tool_id` | Checked-commit gate and its exposed artifact producer |

Usually align `satisfies_intents` with `routing_intent_tags`. A routable
mode that omits in-place coverage can keep handing off needlessly.
`task_completion: checked_commit` requires a real task-owned artifact;
pair it with an artifact producer the mode actually exposes. A flag does
not create a worktree or authorize completion by prose.

See [routing](ROUTER_DESIGN.md), [token budgets](TOKEN_BUDGET.md), and
[coding workflow](CODING_WORKFLOW.md). Keep domain vocabulary in the pack/session,
not hardcoded in kernel routing logic.

## Tools and capability visibility

A Python tool normally has a `.tool.yml` metadata file and implementation
module returning `ToolResult` from `ToolExecutionContext`. Its metadata
declares ID, parameters, effect, timeout/call budget, and module location.
Use the [core fetch tool](../sdk/packs/core/src/sdk/packs/core/tools/fetch_page/)
as a real example and validate with `packstest.run_tool`.

Declare effects truthfully (`read`, `write`, `execute`, `external_write`).
Receipts and approvals depend on them. Model-requested file I/O uses the
storage seam and sandbox-relative paths; never write a model-supplied
absolute path directly. See [storage](STORAGE.md).

Operation/tool visibility has two independent flags:

- `exportable: true` permits references from another pack's modes.
  Private foreign references fail boundary validation.
- `default_available: true` contributes an ambient capability to effective
  mode allowlists.

Modes can opt out with `accept_default_capabilities: false` or subtract
`excluded_operation_ids` / `excluded_tool_ids`. Ambient defaults are not
automatically part of a mode's authored completion contract.
MCP-native ambient tools use host manifest `default_available_tools`, not
pack `.tool.yml` metadata.

Duplicate catalog IDs need an explicit session `providers` owner.
Do not silently depend on pack load order to resolve collisions.

## Authored content and sessions

Keep instructions/templates in versioned Markdown under `content/` and
load them through host-provided readers. Missing required content is an
error, not an empty success fallback.

Packs can ship strict v2 `sessions/<name>.md` files. User sessions may inherit
whole blocks with `from:` or instructions with `body:`. Pack sessions are
leaves: they cannot themselves inherit from another session.
See [session authoring](SESSION_CONTEXT_AUTHORING_GUIDE.md#inheritance-via-from-and-body).
Inspect the built wheel to ensure Markdown/YAML actually ships.

## Extension entry points

| Group | Contribution |
| --- | --- |
| `lisa.extensions.content_roots` | Catalog/content discovery |
| `lisa.extensions.host_commands` | Factory returning pack `HostCommandSpec` entries |
| `lisa.extensions.context_items` | Context evidence builder |
| `lisa.extensions.signal_connectors` | Repeatable external data collector |
| `lisa.extensions.processing_loops` | Processing of collected snapshots into evidence |
| `lisa.extensions.slice_providers` | Opportunistic housekeeping factory |
| `lisa.extensions.model_routers` | Model selection backend |
| `lisa.extensions.model_router_configs` | Default router configuration contribution |
| `lisa.extensions.provider_gateways` | Provider gateway backend |
| `lisa.extensions.sandbox` | Sandbox implementation declaration (floor owner) |
| `lisa.extensions.sandbox_floors` | Pack sandbox-floor needs (paths/postures contribution) |

### Sandbox capability contributions

The sandbox capability follows the memory-capability shape: the kernel owns
the concept (a named capability whose floor is a declarative vocabulary),
one pack owns the implementation (today `lisa-pack-bwrap`), and any pack
may contribute floor needs. Two groups cooperate:

- `lisa.extensions.sandbox` — the implementation pack declares itself: an
  entry returning `{"pack_name": ..., "validate_floor": callable}` where
  the validator parses the final merged floor and raises on values the
  implementation cannot enforce. Sessions select the owner explicitly via
  `providers.capabilities.sandbox: <pack>`; a single registered
  implementation auto-owns; ambiguity is a load error.
- `lisa.extensions.sandbox_floors` — a consumer pack declares its needs:
  an entry returning `(pack_config, context) -> mapping | None`, where the
  mapping is a subset of the floor vocabulary (`execution`,
  `persistence`, `network`, `device`, `ipc`, `paths`,
  `tmpfs_budget_bytes`, `timeout_ceiling_seconds`). `context` carries
  `data_dir` (the host data root, for dynamically-derived paths) and
  `workspace_root`.

Contributions merge **exclusively** onto the harness-authored floor
(`resources.packs.configuration[owner].sandbox.floor`) at session
assembly: scalar axes named by two packs must carry identical values
(disagreement is a load-time policy error naming both packs); `paths`
union with per-path agreement (overlapping grants with the same access
are compatible; `rw` vs `capture` on one path is an error, as are nested
paths with different access). A contribution may only *add* to the
authored floor — restating an axis the floor already satisfies is inert;
contradicting it is an error. Contributions require an authored floor to
attach to (even an empty `floor: {}`): the floor is the session's security
posture, so every effective floor starts from something the harness
wrote. The merged floor is validated by the implementation before it is
materialized, so load-time errors name the exact axis, values, and packs
involved — never a mid-turn dispatch failure.

Backends need not be content packs. Hosts decide which contribution groups
to discover. Entry-point loading must be import-safe, with no credential
reads, registration side effects, or network calls.

Signal collectors and processing loops run through the host's one-shot
intake path, not a timer embedded in the turn loop. Emit typed
`ContextItem` evidence rather than relying on retired `UnifiedState`
snapshot bags. Keep unconfigured/error states explicit and observable.
Use [the public extension harness](../kernel/src/kernel/testing/extension_harness.py)
before wiring to a host. Context items are labelled by the pack and admitted
by the session's [context rules](CONTEXT_ROUTING_GUIDE.md).

## Verification checklist

1. `validate_pack` passes from source; assert `report.ok`, not just print it.
2. Package import and catalog IDs/module paths work without host/private imports.
3. Unit tests/evals drive tools and operations with isolated temporary state.
4. Allowlists, export/default flags, effects, required capabilities, and limits
   match the intended workflow.
5. A new workflow has an opt-in `tests/e2e/*.scenario.yml`, session template,
   tiny fixture, and evidence-based outcome oracle; register it in the
   scenario harness's pack source map.
6. Built wheels include all required content and entry points.
7. Credentials, platform requirements, activation, and user commands are
   documented in the pack's own README.

Use [packstest](../packstest/README.md) for the first loop and
[e2etest](../e2etest/README.md) for integration. Live tests are opt-in and
must not be the default contributor or CI path.

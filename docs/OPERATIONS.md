# Operations

Operations are the **behavior layer** of the Lisa framework. They sit between modes
(which own routing and personality) and tools (which are low-level atomic capabilities).
An operation is a reusable, mode-gated, timeout-bounded unit of work that the kernel
can schedule automatically or offer to the model as a callable.

For step-by-step authoring instructions see [OPERATION_AUTHORING_GUIDE.md](OPERATION_AUTHORING_GUIDE.md).

## What an operation is — and what it is not

| Concept | Role | Example |
| --- | --- | --- |
| **Mode** | User-facing route; owns personality, FSM, and operation bindings | `stripe`, `search`, `discuss` |
| **Operation** | Internal reusable behavior bound to one or more modes | `stripe_status`, `web_search`, `checkpoint_create` |
| **Internal tool** | Atomic capability; single Python call; no orchestration | `web_search` (tool), `save_checkpoint`, `gmail_create_draft` |
| **MCP server tool** | External capability accessed over stdio/HTTP via the MCP protocol | `memory.add`, `memory.search`, `brave.search` |

Operations are **not** user-facing routes. A user never types the name of an operation;
a mode decides when to invoke it. Operations may call internal tools or MCP tools, but
tools cannot call operations.

## Two operation kinds

The `execution_kind` field on `OperationDefinition` (`kernel/models.py`) determines
which runtime is used.

### Runtime operations (`execution_kind: "runtime"`, default)

Authored as a YAML metadata file plus a Python implementation module. The Python
`async def execute(ctx)` function can do anything: read cached state, call tools,
run multi-step logic, validate YAML, or compose a completion call with custom retry
logic. Use this when behavior must be deterministic, non-LLM, or require branching
that is too complex to express in prose.

```plaintext
operations/stripe_status/
├── stripe_status.yml   ← metadata (operation_id, description, mode_allowlist, …)
└── stripe_status.py    ← async def execute(ctx: OperationExecutionContext) -> OperationResult
```

Real examples: `lisa_packs/stripe/operations/stripe_status/`,
`lisa_packs/github/operations/github_status/`,
`sdk/packs/models/operations/advise_models/`.

### Declarative (prompt) operations (`execution_kind: "prompt"`)

Authored as a single Markdown file whose YAML front matter carries the same metadata
fields and whose body is the LLM instruction. The `DeclarativeOperationModule` runtime
(`kernel/operations/declarative.py`) executes the operation by feeding the Markdown
body to the model as a system instruction and running a constrained tool loop. Use this
when the behavior is primarily an LLM task — synthesize, research, reflect — and the
only "logic" is "call these tools, then produce structured output following these rules."

```plaintext
operations/research/
└── web_search.operation.md   ← YAML front matter + Markdown body
```

A sibling `.py` file may optionally supply a Pydantic `response_format_model` class
without exposing an `execute` function. The catalog loader recognises that shape and
skips it as a runtime entry point, using it only as the response contract for the
declarative runtime.

Real examples: `sdk/packs/search/operations/research/web_search.operation.md`,
`sdk/packs/introspect/operations/self_model/reflect_self.operation.md`.

**Precedence rule:** if a directory contains both `<stem>.operation.md` and
`<stem>.py`, the Markdown operation takes precedence and the Python file is treated as
a response-format model only.

## Lifecycle and execution hooks

A mode binds operations in its `_mode.md` front matter under four hooks:

```yaml
entry_operation_ids: [checkout_context]   # run once on mode entry (action modes)
pre_turn_operation_ids: [web_briefing]    # run before every model completion
allowed_operation_ids: [recall, summarize] # callable by the model as tools
post_turn_operation_ids: [synthesize_learning] # run after every turn
```

The turn runner dispatches hooks in order: entry → pre-turn → model+tool loop
(where the model may call `allowed_operation_ids`) → post-turn. Pre-turn and post-turn
exceptions are caught and logged rather than aborting the turn; entry exceptions follow
the mode's `entry_failure_policy` (default `"abort"`).

Mode-bound operations are authoritative: listing an `operation_id` in any of the four
hook lists grants it permission for that mode regardless of its own `mode_allowlist`.
The operation's `mode_allowlist` is a secondary grant used when the mode does not
explicitly list the operation.

## How the model calls operations

Operations listed in `allowed_operation_ids` are advertised to the model as OpenAI
function-tool schemas during the turn, alongside internal pack tools and MCP tools.
The turn runner calls `build_operation_tool_schemas(mode_id)` to build those schemas
from each operation's optional `Parameters` Pydantic class. When the model emits a
tool call whose name matches an operation in the catalog, the turn runner dispatches
it to `execute_operation` (in-process) rather than to the MCP pool.

Modes with `tools_authoritative: true` (the default) expose **only** operations and
tools that are explicitly bound or allowlisted. A mode with an empty
`allowed_operation_ids` list will not offer any operation to the model, even if the
operations declare that mode in their `mode_allowlist`.

## Default-available and exportable capabilities

Two pack-authored flags on operation and tool YAML control cross-pack visibility
and ambient availability:

- **`exportable`** (default `false`) — when `true`, a capability (operation or
  tool) may be referenced by modes in *other* packs. When `false`, the capability
  is private to its owning pack; a foreign mode that lists it in
  `allowed_operation_ids` / `allowed_tool_ids` is rejected at startup by
  `validate_export_boundaries` (`kernel/operations/registry.py`). The check is a
  generic rule over `provider_id` + `exportable`; the kernel never names
  capability ids.
- **`default_available`** (default `false`) — when `true`, the SDK catalog loader
  (`sdk/content/_catalog.py :: _merge_default_capabilities`) auto-merges the
  capability id into every mode's effective `allowed_operation_ids` /
  `allowed_tool_ids` *before* the kernel sees them. This lets a pack ship ambient
  capabilities (e.g. `recall_memory`) without forcing every mode author to
  re-list them. The merge is idempotent (set union) and happens entirely in the
  SDK layer; the kernel's `tools_authoritative` filter is unchanged.

Modes opt out via three front-matter keys:

- **`accept_default_capabilities`** (default `true`) — when `false`, the mode keeps
  its explicit allowlists untouched and inherits no ambient capabilities. Set
  this to `false` for a mode that genuinely wants zero tools.
- **`excluded_operation_ids`** / **`excluded_tool_ids`** (default `[]`) — subtract
  specific ambient ids from the merge while keeping the rest. Does not affect ids
  the mode lists explicitly in `allowed_operation_ids` / `allowed_tool_ids`.

The kernel distinguishes the *authored* allowlists
(`authored_allowed_operation_ids` / `authored_allowed_tool_ids`) from the merged
*effective* ones (`allowed_operation_ids` / `allowed_tool_ids`) so that
`_fork_mode_requires_tools` checks only the authored set — a chat-only mode that
inherits `recall_memory` is not misflagged as requiring tools to complete.

```yaml
# recall.yml — ambient + exported
operation_id: recall_memory
exportable: true
default_available: true
```

```yaml
# web_search.tool.yml — exported but not ambient (costly)
tool_id: web_search
exportable: true
# default_available omitted (false) — modes opt in via allowed_tool_ids
```

## Tool capability budget: `tool_ids`

Every operation declares which tools it may use via `tool_ids`. This serves two
purposes:

1. **Schema injection** — before the operation executes, the registry builds
   `OperationExecutionContext.tool_schemas` from those declared tools so the
   operation's own LLM calls see the right function schemas.
2. **Enforcement for declarative operations** — the registry wraps
   `ctx.execute_tool_fn` with `_wrap_declarative_tool_executor`, which rejects any
   tool call outside the declared set at runtime
   (`kernel/operations/registry.py: _wrap_declarative_tool_executor`).

Runtime operations receive the same `ctx.execute_tool_fn`, but Python code is not
mechanically prevented from calling an undeclared tool through other paths. Honor the
contract; it is the basis for capability auditing.

```yaml
# checkpoint_create.yml
operation_id: checkpoint_create
description: Create a checkpoint enriched by the model and persist it to lisa-storage.
tool_ids: [save_checkpoint]   # this operation may call exactly one tool
mode_allowlist: [default, discuss, learn]
```

## Discovery and registration

The `lisa-catalog` MCP server (`mcp_server/catalog/`) scans all discovered content
roots via `sdk.content.build_operation_catalog_json`. The kernel calls
`list_operations` on startup through `register_operation_mcp(config)`, which populates
`_OPERATION_CATALOG` in `kernel/operations/registry.py` and validates every entry.

Content roots come from `lisa.extensions.content_roots` entry points registered in
each pack's `pyproject.toml`, plus any paths in the `SDK_CONTENT_ROOTS` environment
variable. Duplicate `operation_id` values across roots require an explicit session
provider selection in `LISA_SESSION_PROVIDER_SELECTIONS`; the catalog loader raises
`CatalogCollisionError` otherwise.

## Validation at startup

`validate_operation_catalog` enforces:

- Every `operation_id` is unique within the catalog.
- Every entry has a non-empty `description`.
- Every runtime operation has an implementation module with `async def execute(ctx)`.
- Every declarative operation has a non-empty `instruction_body`.
- `max_calls_per_turn >= 0` and `timeout_seconds >= 0`.
- Every `tool_id` in `tool_ids` exists in the tool catalog (if the tool catalog is
  already loaded when the operation catalog is validated).

## Execution details

`execute_operation(operation_id, ctx)` in `kernel/operations/registry.py`:

1. Looks up the `OperationDefinition`; raises `OperationExecutionError` if absent.
2. Checks mode permission via `_is_operation_allowed_for_mode`.
3. Injects `ctx.preferred_model_id` from the definition.
4. Builds `ctx.tool_schemas` from declared `tool_ids`.
5. Wraps `ctx.execute_tool_fn` to enforce the tool allowlist (declarative only).
6. Calls `definition.implementation_module.execute(ctx)` under `asyncio.wait_for`
   when `timeout_seconds > 0`.
7. Emits a `ToolExecutionTelemetry` event via `ctx.event_bus` with duration and status.
8. Accepts `str` or `OperationResult` return values; raises on anything else.

## Operation vs. tool: the short version

| | Operation | Internal tool |
| --- | --- | --- |
| **Purpose** | Behavior: orchestrate, reason, persist | Capability: one atomic action |
| **Author format** | `.yml` + `.py` or `.operation.md` | `.tool.yml` + `.py` |
| **Discovered from** | `operations/` subdirs in content roots | `tools/` subdirs in content roots |
| **Model-callable** | When listed in `allowed_operation_ids` | When listed in `allowed_tool_ids` |
| **Can call tools** | Yes, via `ctx.execute_tool_fn` | No |
| **LLM turn** | Optional (prompt ops use the model internally) | Never (pure Python) |
| **Return type** | `OperationResult` | `ToolResult` |
| **Execution kind** | `"runtime"` or `"prompt"` | `"runtime"` only |

## Related documents

- [OPERATION_AUTHORING_GUIDE.md](OPERATION_AUTHORING_GUIDE.md) — step-by-step guide
  with complete examples, metadata reference, and testing checklist.
- [PACK_AUTHORING_GUIDE.md](PACK_AUTHORING_GUIDE.md) — how operations fit inside a
  pack alongside modes and content.
- [KERNEL_ARCHITECTURE.md](KERNEL_ARCHITECTURE.md) — full turn execution flow and
  module map.
- [BOUNDARIES.md](BOUNDARIES.md) — public/private framework boundary rules.

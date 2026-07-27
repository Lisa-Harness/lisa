# Operation Authoring Guide

This guide walks through creating operations for the Lisa framework. For the conceptual
overview and lifecycle details see [OPERATIONS.md](OPERATIONS.md).

## Choose your operation kind

| Use case | Kind | Format |
| --- | --- | --- |
| Deterministic Python logic — read state, validate, format | **Runtime** | `.yml` + `.py` |
| LLM task — synthesize, research, reflect, guided tool use | **Declarative (prompt)** | `.operation.md` (optionally + `.py` response model) |

When in doubt, start declarative. Reach for a runtime operation only when branching,
validation, retry logic, or non-LLM data shaping makes a markdown instruction
impractical to maintain.

## Prerequisites

Operations live in `operations/` subdirectories within any content root. A content root
is any directory registered via the `lisa.extensions.content_roots` entry point in your
pack's `pyproject.toml`, or via the `SDK_CONTENT_ROOTS` environment variable. The
`lisa-catalog` MCP server discovers the catalog; you never register individual files.

```toml
# pyproject.toml
[project.entry-points."lisa.extensions.content_roots"]
my_pack = "my_pack:get_content_root"
```

```python
# my_pack/__init__.py
from pathlib import Path

def get_content_root() -> Path:
    return Path(__file__).parent
```

## Runtime operation

### Directory shape

```plaintext
my_pack/
  operations/
    my_feature/
      my_operation.yml   ← required metadata
      my_operation.py    ← required implementation
```

### Metadata file (`.yml`)

```yaml
operation_id: my_operation
description: One clear sentence describing what this operation does.
mode_allowlist: [my_mode]
tool_ids: []           # list tools this operation may call; empty is fine
timeout_seconds: 30    # default 30; 0 disables timeout
max_calls_per_turn: 1  # default 3
preferred_model_id: default   # optional; omit to inherit mode default
```

**Required:** `operation_id`, `description`.

**Optional:** `mode_allowlist` (default allows all modes), `tool_ids`, `timeout_seconds`,
`max_calls_per_turn`, `preferred_model_id`.

`mode_allowlist` can also be omitted — the operation will then be allowed for every
mode unless `tools_authoritative: true` is set on the mode and the operation is not
listed there.

### Implementation (`.py`)

```python
from kernel.models import OperationExecutionContext, OperationResult


async def execute(ctx: OperationExecutionContext) -> OperationResult:
    """Execute the operation and return a structured result."""
    return OperationResult(output_text=f"Received: {ctx.query}")
```

The module must export `async def execute(ctx: OperationExecutionContext) -> OperationResult`.
No other exports are required.

### Optional `Parameters` class

If you want the model to supply structured arguments when calling the operation as a
tool, export a Pydantic `Parameters` class. The catalog loader detects it automatically
and builds the function-tool schema from it.

```python
from pydantic import BaseModel, Field
from kernel.models import OperationExecutionContext, OperationResult


class Parameters(BaseModel):
    limit: int = Field(default=10, ge=1, le=100, description="Max items to return.")


async def execute(ctx: OperationExecutionContext) -> OperationResult:
    limit = int(ctx.params.get("limit", 10))
    # use limit ...
    return OperationResult(output_text=f"Returning up to {limit} items.")
```

### Calling tools from a runtime operation

List the tools in `tool_ids` and call them via `ctx.execute_tool_fn`:

```python
from kernel.models import OperationExecutionContext, OperationResult, ToolExecutionContext


async def execute(ctx: OperationExecutionContext) -> OperationResult:
    tool_ctx = ToolExecutionContext(
        tool_id="save_checkpoint",
        workspace_id=ctx.workspace_id,
        user_id=ctx.user_id,
        query=ctx.query,
        parameters={"label": "my checkpoint"},
    )
    result = await ctx.execute_tool_fn("save_checkpoint", tool_ctx)
    return OperationResult(output_text=result.output_text)
```

Matching YAML:

```yaml
operation_id: my_checkpoint_op
description: Save a labeled checkpoint.
tool_ids: [save_checkpoint]
mode_allowlist: [my_mode]
```

### Reading cached state

```python
from kernel.models import OperationExecutionContext, OperationResult, UnifiedState


async def execute(ctx: OperationExecutionContext) -> OperationResult:
    state: UnifiedState | None = ctx.state_provider() if ctx.state_provider else None
    if state is None:
        return OperationResult(output_text="No cached state available.")
    snapshot = state.signal_snapshots.get("my_service")
    if snapshot is None:
        return OperationResult(output_text="No snapshot yet.")
    return OperationResult(output_text=str(snapshot))
```

## Declarative (prompt) operation

### Directory shape

```plaintext
my_pack/
  operations/
    research/
      briefing.operation.md   ← metadata + instructions in one file
      briefing.py             ← optional Pydantic response model (no execute())
```

### Markdown file (`.operation.md`)

```markdown
---
operation_id: research_briefing
description: Produce a grounded research summary from web results.
tool_ids: [web_search]
mode_allowlist: [research]
timeout_seconds: 60
max_calls_per_turn: 1
completion_max_tokens: 2000
completion_temperature: 0.2
response_format_model_path: my_pack.operations.research.briefing.BriefingOutput
response_output_text_field: summary
---

# Research Briefing

You are a grounded research analyst. Use `web_search` to gather evidence, then
synthesize only from the tool results.

## Rules

1. Run 2–3 independent searches.
2. Never invent URLs or facts.
3. If no grounded results are available, set `summary` to the search error and leave
   other fields empty.

## Output

Return one JSON object matching the declared schema.
```

**Required front-matter fields:** `operation_id`, `description`.

**Optional fields:**

| Field | Default | Purpose |
| --- | --- | --- |
| `tool_ids` | `[]` | Tools the model may call in the instruction loop |
| `mode_allowlist` | (all modes) | Modes allowed to invoke this operation |
| `timeout_seconds` | `60` | Wall-clock limit for the full instruction loop |
| `max_calls_per_turn` | `3` | Max times this operation can be called in one turn |
| `completion_max_tokens` | `1500` | Token budget for the model completion |
| `completion_temperature` | `0.2` | Temperature for the model completion |
| `preferred_model_id` | (mode default) | Model alias to use for this operation |
| `response_format_model_path` | (none) | Dotted path to a Pydantic response model class |
| `response_output_text_field` | `"output_text"` | Field name to extract as `OperationResult.output_text` |
| `response_output_json_passthrough` | `{}` | Dotted-path mappings to copy from last tool output into result |

### Optional response-format model (`.py`)

```python
from pydantic import BaseModel, Field


class BriefingOutput(BaseModel):
    summary: str = Field(default="", description="Full markdown briefing.")
    sources: list[str] = Field(default_factory=list, description="Grounded URLs from tool results.")
    key_points: list[str] = Field(default_factory=list, description="Bullet-point highlights.")
```

When `response_format_model_path` is set, the declarative runtime:

1. Passes the model class to `ctx.complete_fn` as `response_format`.
2. On a parse failure or a provider rejection (`structured_output_unsupported`), appends
   a JSON-only retry instruction and retries without the `response_format` parameter.
3. Extracts `response_output_text_field` from the parsed model for `OperationResult.output_text`.

If you omit the response model, the operation returns the model's raw text.

## Binding operations to a mode

Operations are invoked by the turn runner when a mode declares them in its
`_mode.md` front matter:

```yaml
---
mode_id: my_mode
mode_kind: chat
entry_operation_ids: [load_context]          # run once on mode entry
pre_turn_operation_ids: [research_briefing]  # run before every model completion
allowed_operation_ids: [recall, summarize]   # model can call these as tools
post_turn_operation_ids: [synthesize_notes]  # run after every turn
---
```

Listing an operation in any of these four lists automatically grants it permission for
the mode, regardless of the operation's own `mode_allowlist`. Mode-owned bindings are
the authoritative source; `mode_allowlist` in the operation's metadata is a secondary
grant.

## Discovery rules and precedence

- `sdk.content.build_operation_catalog_json` scans `operations/` recursively.
- Markdown operations (`.operation.md`) are loaded first and take precedence.
- If a directory has both `my_op.operation.md` and `my_op.py`, the `.py` file is
  treated as the response-format model only — its `execute` function (if any) is
  ignored.
- Python runtime operations must have a sibling `.yml` file with matching `operation_id`
  or the catalog loader skips them with a warning.
- Module paths for Python modules are inferred automatically from their location
  relative to the package root. Use `module_path` in the `.yml` only when auto-inference
  is wrong.

## Testing

### Import boundary test (required)

Every pack must verify it imports cleanly without private app modules:

```python
# tests/test_import_boundaries.py
import importlib

def test_pack_imports_cleanly() -> None:
    importlib.import_module("my_pack")
```

### Catalog discovery test (recommended)

```python
# tests/test_catalog.py
from pathlib import Path
from sdk.content import build_operation_catalog_json
import json

def test_my_operation_discovered() -> None:
    import my_pack
    root = my_pack.get_content_root() / "operations"
    catalog = json.loads(build_operation_catalog_json([root]))
    assert "my_operation" in catalog
    op = catalog["my_operation"]
    assert op["description"]
    assert "my_mode" in op["mode_allowlist"]

def test_tool_ids_declared() -> None:
    import my_pack
    root = my_pack.get_content_root() / "operations"
    catalog = json.loads(build_operation_catalog_json([root]))
    assert catalog["my_checkpoint_op"]["tool_ids"] == ["save_checkpoint"]
```

### Validate against live modes

```powershell
# From lisa_framework/
$env:SDK_CONTENT_ROOTS = "C:\Code\my-pack"
uv run lisa modes         # verify my_mode appears
uv run lisa once "/my_mode test query"
```

## Pre-merge checklist

- `operation_id` is unique across all packs you intend to load simultaneously.
- `description` is a non-empty one-line summary.
- Every entry in `tool_ids` is registered in the tool catalog.
- `mode_allowlist` lists at least one mode, or the operation is intentionally global.
- For declarative operations: the front-matter body is non-empty and
  `response_format_model_path` resolves to a real Pydantic class.
- The pack imports cleanly with no host or private app imports.
- Catalog discovery test passes without live services.

## Common errors

| Error | Cause | Fix |
| --- | --- | --- |
| `OperationValidationError: … missing execute(ctx)` | Python module has no `execute` function | Add `async def execute(ctx)` |
| `OperationValidationError: … references unknown tool 'X'` | `tool_ids` entry not in tool catalog | Register the tool in your pack's `tools/` dir or remove the entry |
| `OperationExecutionError: … is not allowed for mode 'Y'` | Mode doesn't bind the operation and `mode_allowlist` doesn't include it | Add operation to `allowed_operation_ids` in the mode's `_mode.md` |
| `OperationExecutionError: cannot call undeclared tool 'X'` | Declarative operation called a tool not in `tool_ids` | Add the tool to `tool_ids` in the front matter |
| `CatalogCollisionError: duplicate operation_id 'X'` | Two packs define the same `operation_id` | Use `LISA_SESSION_PROVIDER_SELECTIONS` or rename one operation |

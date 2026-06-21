# Pack Authoring Guide

Packs add modes, operations, and authored content without changing host code. The framework discovers packs through `lisa.extensions.content_roots` entry points or through host-injected content roots.

## Pack Shape

A pack normally contains:

```plaintext
sdk_packs/example/
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

Optional front-door routing metadata can also be authored in `_mode.md` front matter:

```markdown
---
frontdoor_routable: true
frontdoor_priority: 20
frontdoor_fast_path_patterns:
  - "^search\\b"
frontdoor_route_hints:
  - "Prefer `search` when the user asks for recent external information."
frontdoor_examples:
  - |-
      User message: "Find recent cloud costs"
      Reasoning: The user asked for fresh external data.
      selected_mode_id: search
      normalized_query: cloud costs
      confidence_score: 0.92
---
```

- `frontdoor_route_hints` and `frontdoor_examples` are prompt-only guidance for the bounded router.
- Use `frontdoor_fast_path_patterns` only for high-precision regex candidates suitable for verifier confirmation.

## Operation Files

Runtime operations use colocated YAML metadata and Python implementation:

```yaml
operation_id: example_operation
description: Run one example operation.
mode_allowlist: [example]
timeout_seconds: 30
```

```python
from kernel.models import OperationExecutionContext, OperationResult


async def execute(ctx: OperationExecutionContext) -> OperationResult:
    return OperationResult(output_text=f"Example received: {ctx.query}")
```

Prompt-authored operations can use `.operation.md` files when the operation is mostly instructions plus model/tool callbacks. Declare `tool_ids` only for tools the host actually registers.

## Operation Rules

- Operations are internal reusable behavior, not user-facing routes.
- A mode decides which operations are allowed, pre-turn, post-turn, or entry operations.
- Operations must depend on `kernel` contracts and host-injected callbacks, not private app modules.
- Use typed Pydantic parameter and response models when the payload crosses a module boundary.
- Return `OperationResult` with text plus optional structured JSON.

## Content Rules

Authored text should live under `content/` and be loaded by host-provided readers. Avoid hard-coded absolute paths. Keep pack content generic unless the pack is explicitly private.

## Hello World Pack

The fastest end-to-end path is a local pack inside the extracted framework tree. Little Lisa discovers these packs from `SDK_CONTENT_ROOTS`, so you do not need a private app or a published wheel just to prove the pack shape.

Create this folder under `lisa_framework/`:

```plaintext
sdk_packs/hello_world/
  __init__.py
  modes/
    hello/
      _mode.md
  operations/
    hello_operation.py
    hello_operation.yml
```

`sdk_packs/hello_world/__init__.py`

```python
"""Hello world pack."""
```

`sdk_packs/hello_world/modes/hello/_mode.md`

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

`sdk_packs/hello_world/operations/hello_operation.yml`

```yaml
operation_id: hello_operation
description: Say hello from a custom pack.
mode_allowlist: [hello]
timeout_seconds: 30
```

`sdk_packs/hello_world/operations/hello_operation.py`

```python
from kernel.models import OperationExecutionContext, OperationResult


async def execute(ctx: OperationExecutionContext) -> OperationResult:
    return OperationResult(output_text=f"Hello from custom pack: {ctx.query or 'hello'}")
```

From `lisa_framework/` on Windows:

```powershell
$env:SDK_CONTENT_ROOTS = (Resolve-Path .\sdk_packs\hello_world)
uv run python examples/little_lisa/main.py --list-modes
uv run python examples/little_lisa/main.py --once "/hello framework"
```

Expected outcome:

- `--list-modes` shows `hello` under supported Little Lisa modes.
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
hello_world = "sdk_packs.hello_world:get_content_root"
```

## Testing Checklist

1. Import the package without private app modules.
2. Load the mode catalog and operation catalog.
3. Validate operation allowlists against discovered modes.
4. Run deterministic unit tests without live APIs.
5. Run Little Lisa for one command path when practical.
6. Add opt-in live tests only for integrations that need them.

Focused local checks from `lisa_framework/`:

```powershell
uv run python examples/little_lisa/main.py --list-modes
uv run python examples/little_lisa/main.py --once "/markdown Draft a quick checklist"
uv run python examples/little_lisa/main.py --once "/hello framework"  # after setting SDK_CONTENT_ROOTS
uv run ruff check sdk_packs examples
```

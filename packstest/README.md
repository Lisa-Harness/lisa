# packstest — in-process pack test/eval harness

`packstest` tests Lisa content packs **in-process, from source, in seconds** —
no pip install, no `uv sync`, no catalog MCP subprocess, no model calls. It
runs on pytest and the existing `kernel.testing` primitives; it adds no new
runtime dependencies.

## Why

The production pack route (entry-point discovery → content-root discovery →
catalog MCP subprocess → host wiring) is correct for runtime but far too slow
for the edit-test-fix loop. `packstest` activates a pack directly from its
checkout and drives its tools through the real contracts, so a failing eval
names the exact tool and error instead of surfacing a mysterious turn failure.

## Modules

| Module | Purpose |
| --- | --- |
| `packstest.context` | `ToolContextFactory` / `make_context` — fully-wired `ToolExecutionContext` (`pack_config`, host identity, artifact store, event bus, data dir) |
| `packstest.store` | `MemoryWorkProductStore` — in-memory `WorkProductStore` double (the host's file store is off-bounds for packs) |
| `packstest.catalog` | `validate_pack` / `build_tool_catalog` / `activate_tool` — activate a pack from source via the real `sdk.content` scanners |
| `packstest.runner` | `run_tool` (one in-process dispatch) and `run_scenario` (scripted tool-call plans with `$shared.*` parameter threading) |
| `packstest.report` | `Trajectory` — ordered record of every tool invocation, exportable as diffable JSON |
| `packstest.git` | `git_repo` fixture — throwaway repository plus bare remote, fully offline |
| `packstest.case` | Declarative `Case` executed under the framework's `eval` pytest marker |

## Quick start: validate a pack

```bash
uv run python -c "
from pathlib import Path
from packstest.catalog import validate_pack
report = validate_pack(Path('../lisa-packs-wip/<pack>'))
print(report.summary())
"
```

Validation checks (mirroring production contracts, not duplicating them):
distribution name declared, content subdirs present, every
`tools/**/*.tool.yml` parses, each `module_path` imports in-process with an
`execute()` callable, tool IDs unique, provider IDs resolved the way session
configuration expects, and every `lisa.extensions.*` entry point loads.

## Quick start: drive tools

```python
import asyncio
from pathlib import Path

from packstest.catalog import build_tool_catalog
from packstest.context import ToolContextFactory
from packstest.git import git_repo
from packstest.runner import run_tool

PACK = Path("../lisa-packs-wip/coding")


async def main() -> None:
    repo = await git_repo(tmp)  # throwaway repo + bare remote
    factory = ToolContextFactory(
        root=repo.root,
        pack_config={"workspace_root": str(repo.root)},
    )
    catalog = build_tool_catalog(PACK)
    run = await run_tool(catalog, "coding_create_worktree", factory, {"summary": "probe"})
    assert run.result.error == ""


asyncio.run(main())
```

`run_tool` accepts a pack checkout path, a full catalog dict, or a single tool
dict. Tool exceptions surface as `ToolRun(error="raised")` with the exception
text in `result.error` — mirroring the production dispatch contract, so a
crashing tool is visible in the trajectory instead of aborting the run.

## Eval cases

Eval files live in the pack under `tests/evals/` and use the `eval` marker
(register it in the pack's `pyproject.toml`):

```python
import pytest
from packstest.catalog import validate_pack

PACK_PATH = Path(__file__).resolve().parents[4]  # tests/evals/x.py → checkout


@pytest.mark.eval
def test_pack_validates_from_source() -> None:
    report = validate_pack(PACK_PATH)
    assert report.ok, report.summary()
```

The `eval` marker contract: **deterministic agent regression evals — no model
calls, no I/O beyond throwaway tmp dirs.** Model behavior is simulated by an
explicit tool-call plan (`packstest.runner.Scenario`), never by a live
provider. For LLM-judged evals use a separate, opt-in marker; keep them out of
the deterministic core.

## Writing a trajectory eval

Assert on outcomes and state transitions, not prose:

```python
@pytest.mark.eval
@pytest.mark.asyncio
async def test_full_chain(tmp_path) -> None:
    repo = await git_repo(tmp_path)
    factory = ToolContextFactory(root=repo.root, pack_config=_config(repo.root))
    catalog = build_tool_catalog(PACK_PATH)
    trajectory = Trajectory(name="chain")

    run = await run_tool(catalog, "coding_create_worktree", factory, {"summary": "x"})
    artifact_id = run.result.output_json["artifact_id"]
    trajectory.record(run)

    # ... subsequent steps, threading artifact_id ...

    assert trajectory.all_errors_absent(), trajectory.first_error()
    assert trajectory.tool_ids == ["coding_create_worktree", "coding_status", ...]
```

## Fidelity rules (keep the harness honest)

- The harness must call the **real** SDK loaders (`sdk.content.build_tool_catalog_json`,
  `resolve_tool_roots`) rather than reimplementing scans — otherwise it drifts
  from production.
- Validate tool `effect` values against `kernel.approval.EffectKind`
  (`get_args`), never a hardcoded set — `external_write` is a valid kind.
- `build_tool_catalog` must make the pack source importable **before**
  scanning: module-path inference walks `sys.path` for package anchors.
- `MemoryWorkProductStore` implements exactly the `WorkProductStore` methods
  pack code exercises. If pack code starts using a new method, add it here
  with a test; consider a contract test running both store implementations.

## Boundary rule

`packstest` imports `kernel` and `sdk` only — never hosts, never packs. Pack
repos consume it as a dev dependency (`lisa-packstest`) and keep their own
unit tests in `tests/` plus evals in `tests/evals/`.
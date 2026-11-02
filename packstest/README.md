# packstest — in-process pack test/eval harness

`packstest` tests Lisa content packs **in-process, from source, in seconds** —
no pip install, no `uv sync`, no catalog MCP subprocess, no model calls. It
runs on pytest and the existing `kernel.testing` primitives; it adds no new
runtime dependencies.

The framework development environment must already be installed.
See [testing](../docs/TESTING.md) and [pack authoring](../docs/PACK_AUTHORING_GUIDE.md).
The framework's publish workflow ships the `lisa-harness-packstest`
distribution to PyPI under the `pypi-lisa-harness-packstest` environment;
pack repos that consume it from the registry should verify availability
during their first dependency-managed install.

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
report = validate_pack(Path('../lisa-packs/markdown'))
print(report.summary())
assert report.ok, report.summary()
"
```

Validation checks (mirroring production contracts, not duplicating them):
distribution name declared, content subdirs present, every
`tools/**/*.tool.yml` parses, each `module_path` imports in-process with an
`execute()` callable, tool IDs unique, provider IDs resolved the way session
configuration expects, and every `lisa.extensions.*` entry point loads.

### Source-layout packs

The content-root locator resolves every layout from the pack's *declared*
entry point — a flat `lisa_packs/<name>` checkout, `src/lisa_packs/<name>`
(two-pack repos), and bundled `src/sdk/packs/<name>` trees all resolve to the
real content root, so `report.ok` implies a real tool scan whenever the pack
declares tools. When the entry point does not resolve (and no `src/` walk
matches), the checkout itself is used and a `content-root` warning reports
the empty scan — inspect warnings; a warning is the only way `ok=True`
carries no tool scan.

For operations and modes, assert the real SDK catalogs at the resolved root
as part of the pack's tests; `validate_pack` focuses on tool schemas/imports,
not every operation behavior.

## Quick start: drive tools

This pytest example assumes the external markdown pack checkout exists at
`PACK`. Supply any configuration/dispatch services required by the tool;
they are not inferred from its README.

```python
from pathlib import Path

import pytest

from packstest.catalog import build_tool_catalog
from packstest.context import ToolContextFactory
from packstest.runner import run_tool

PACK = Path("../lisa-packs/markdown")


@pytest.mark.asyncio
async def test_markdown_tool(tmp_path: Path) -> None:
    factory = ToolContextFactory(root=tmp_path, pack_config={})
    catalog = build_tool_catalog(PACK)
    run = await run_tool(catalog, "markdown_generate", factory, {"markdown": "# Hi\n"})
    assert run.result.error == ""
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
from pathlib import Path
from packstest.catalog import validate_pack

PACK_PATH = Path(__file__).resolve().parents[2]  # <checkout>/tests/evals/x.py


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
from pathlib import Path

import pytest
from packstest.catalog import build_tool_catalog
from packstest.context import ToolContextFactory
from packstest.report import Trajectory
from packstest.runner import run_tool

PACK_PATH = Path("../lisa-packs/markdown")


@pytest.mark.eval
@pytest.mark.asyncio
async def test_trajectory(tmp_path: Path) -> None:
    factory = ToolContextFactory(root=tmp_path, pack_config={})
    catalog = build_tool_catalog(PACK_PATH)
    trajectory = Trajectory(name="chain")

    run = await run_tool(catalog, "markdown_generate", factory, {"markdown": "# Hi\n"})
    outline = run.result.output_json
    trajectory.record(run)

    assert trajectory.all_errors_absent(), trajectory.first_error()
    assert trajectory.tool_ids == ["markdown_generate"]
```

For multi-step cases, record each call and pass its structured output to
the next invocation. Assert on the resulting artifacts/state, not only
the absence of an error. The example's external pack is not bundled.

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
repos consume it as a dev dependency (`lisa-harness-packstest`) and keep their own
unit tests in `tests/` plus evals in `tests/evals/`.

# Testing and quality gates

Run commands from the framework workspace root after
`uv sync --frozen --extra dev`. The root [pyproject.toml](../pyproject.toml)
defines pytest paths, markers, Ruff, mypy, and coverage settings;
[CI](../.github/workflows/ci.yml) is authoritative for required jobs.

## Fast development loop

Start with the smallest reproducing test, then the affected package:

```sh
uv run pytest runtime/tests/test_bootstrap.py -m "not live" -q
uv run pytest kernel/tests/test_session_plan.py -m "not live" -q
```

Root test discovery includes package `tests/` directories and several
colocated `src/**/tests` suites. Explicitly selecting only a package's
top-level `tests/` does not cover all its colocated tests.

| Marker/layer | Meaning |
| --- | --- |
| `live` | Calls real external APIs; opt in with credentials and cost awareness |
| `integration` | Multi-module tests; some require local services |
| `eval` | Deterministic agent regressions; no model calls, only isolated test data |
| `packstest` | Source-based catalog/tool/scenario tests with real SDK contracts |
| `lisa-e2e` | Whole-runtime scenarios, live by default or explicitly scripted |

Always specify `-m "not live"` for the offline suite. The root pytest
configuration defines markers but does not itself deselect live tests.
Not every integration test is necessarily offline or service-free just
because it is not marked `live`; check its fixtures.

## CI gates

Quality jobs run on Ubuntu/Python 3.12. Tests and per-package coverage gates
run on Python 3.11/3.12 across Linux, macOS, and Windows.

```sh
uv run ruff format --check .
uv run ruff check .
uv run python scripts/check_direct_io_policy.py
MYPYPATH="typings:kernel/src:sdk/src:mcp_server/src" \
  uv run mypy --explicit-package-bases --exclude '(^|/)(tests|test_[^/]*\.py$)' \
  kernel/src/kernel sdk/src/sdk mcp_server/src/mcp_server
uv run vulture kernel/src/kernel sdk/src/sdk mcp_server/src/mcp_server runtime/src/runtime vulture_allowlist.py
uv run pytest -m "not live" --tb=short -v
uv run coverage run -m pytest -m "not live" -q
uv run python scripts/check_coverage.py
```

For the mypy step in PowerShell, set the path using the platform separator:

```powershell
$env:MYPYPATH = @("typings", "kernel/src", "sdk/src", "mcp_server/src") -join [IO.Path]::PathSeparator
uv run mypy --explicit-package-bases --exclude '(^|/)(tests|test_[^/]*\.py$)' kernel/src/kernel sdk/src/sdk mcp_server/src/mcp_server
```

The mypy gate currently selects kernel, SDK, and MCP sources, not every
workspace package. The runtime is included in lint/dead-code/tests/coverage.
Do not describe the current type-check selection as workspace-wide.

Coverage is branch-based and excludes test source. The root global floor
is 76%; [check_coverage.py](../scripts/check_coverage.py) enforces separate
package floors. Consult that script for current numbers rather than copying
a dated coverage report. Ratchet floors with evidence; do not lower them
to hide a regression. `xfail_strict = true` makes unexpected passes fail.

The `lisa verify` CLI smoke check is separate: the framework workspace does
not install `lisa`, and the current framework CI does not run that command.

## Pack and host changes

Use [packstest](../packstest/README.md) before any installed-package route:

Follow the [source validation examples](../packstest/README.md#quick-start-validate-a-pack).
For `src/`-layout packs, explicitly inspect the resolved-root catalogs;
the current `validate_pack` locator can emit a warning-only empty scan.
Do not treat `report.ok` without expected catalog assertions as complete
validation.

Drive behavior with `run_tool` / `run_scenario`, isolated data, and offline
fixtures. New pack workflows also need an opt-in
[end-to-end scenario](../e2etest/README.md) asserting real artifacts,
receipts, memory, or Git state rather than trusting prose.

Runtime integration tests should construct typed configuration and injected
fakes through the [runtime factory](../runtime/src/runtime/testing/runtime_factory.py).
Do not depend on a developer's data directory, installed external packs,
credentials, or live model roster.

## Documentation changes

Check relative links, heading targets, command ownership, and example schema
validity. Validate complete examples against the current compiler/catalog;
label partial blocks explicitly. Documentation-only edits do not need a
live model run. Do not claim all quality gates passed based on a link check.

# Testing and quality gates

Run commands from the framework workspace root after
`uv sync --frozen --extra dev`. The root [pyproject.toml](../pyproject.toml)
defines pytest paths, markers, Ruff, mypy, and coverage settings;
[CI](../.github/workflows/ci.yml) is authoritative for required jobs.

## Test-selection tiers

Select tests by what changed; escalate only when the targeted tier is green.
The tiers are evidence levels, not alternatives — each one subsumes the
previous scope of knowledge, and the full suite remains the pre-push gate.

| Tier | Command | When |
| --- | --- | --- |
| Unit | `uv run pytest <test module(s)> -m "not live" -q` | After each edit; the test module(s) mirroring the changed source file |
| Package | `uv run python scripts/check_tests.py` | Before commit; selects the owning package testpaths from the worktree's changed paths |
| Full | `uv run pytest -m "not live" -q` | Before push/PR; the required CI scope |

[check_tests.py](../scripts/check_tests.py) computes the package tier from
Git state, so selection cannot be narrowed by declaring less work. It runs
the testpaths of every package the changed paths reach and escalates to the
full suite for `kernel` (the shared contract layer every package imports),
`pyproject.toml`, `uv.lock`, `scripts/`, `.github/`, `typings/`, any
`conftest.py`, and any path outside a known package. `--dry-run` prints the
command without running it; `--paths` selects explicitly. Its package table
is asserted against the configured `testpaths` by
`packstest/tests/test_check_tests.py`, so a new suite cannot silently fall
outside selection.

Map a changed file to its owning package's testpaths (all of them — a
package's tests are split between its top-level `tests/` and colocated
`src/**/tests/`, and selecting only one half misses the other):

| Changed source | Owning package testpaths |
| --- | --- |
| `kernel/src/**` | `kernel/tests kernel/src/kernel/context/tests` |
| `sdk/src/**` | `sdk/tests sdk/src/sdk/context/tests sdk/src/sdk/guard/tests sdk/src/sdk/memory/tests sdk/src/sdk/storage/tests` |
| `sdk/packs/<name>/src/**` | `sdk/packs/<name>/tests` |
| `mcp_server/src/**` | `mcp_server/tests mcp_server/src/mcp_server/fetch/tests mcp_server/src/mcp_server/memory/tests mcp_server/src/mcp_server/thread/tests` |
| `runtime/src/**` | `runtime/tests runtime/src/runtime/model_discovery/tests` |
| `packstest/src/**` | `packstest/tests` |
| `e2etest/src/**` | `e2etest/tests` |
| `scripts/check_docs.py` | `packstest/tests/test_check_docs.py` |
| `scripts/check_tests.py` | `packstest/tests/test_check_tests.py` |

Cross-package closure: `kernel` is imported by every other package, so a
kernel change that alters a contract the SDK, runtime, MCP servers, or packs
consume must also run those packages' testpaths — the import direction
(kernel ← sdk ← runtime/packs/mcp_server) is the dependency direction. A
change confined to one package's internal implementation runs that
package's tier only. When in doubt, run the full suite; it is the gate CI
enforces.

### Automated selection

[`scripts/check_tests.py`](../scripts/check_tests.py) performs the package tier
automatically. It reads the change set from Git rather than from a declaration,
so the selection cannot be narrowed by whoever is asking:

```sh
uv run python scripts/check_tests.py            # select from Git state
uv run python scripts/check_tests.py --dry-run  # print the pytest command
uv run python scripts/check_tests.py --base origin/main
uv run python scripts/check_tests.py --paths a.py b.py
```

The changed set is the union of the working tree (`git status --porcelain -z`,
including untracked and deleted files and both sides of a rename) and the
committed delta against a base (`git diff --name-only <base>...HEAD`). Reading
the tree alone would miss work that was already committed, which is the normal
state when the gate runs.

The base is `--base` when given — an explicit ref that does not resolve is an
error rather than a silent fallback — otherwise the merge-base of `HEAD` with
`origin/main`, then `main`, taking the most recent of those. A stale candidate
yields a merge-base older than the branch point, which would widen the delta to
include other people's commits; merge-bases of one `HEAD` are totally ordered by
ancestry, so the maximal one is the branch point.

The selector escalates to the full suite rather than guessing when it cannot
show that a subset covers the change: no base resolved, no changed paths
detected, a path outside every known package, a cross-cutting path
(`pyproject.toml`, `uv.lock`, `conftest.py`, `typings/`, `.github/`,
`vulture_allowlist.py`, or any `scripts/` file other than the two the packstest
suite loads), or a kernel change. Documentation-only paths select nothing and
exit 0, mirroring the CI path filter. The exit code is pytest's, so a failed,
errored, or uncollected run fails the gate.

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

The full offline suite shares one `lisa-catalog` subprocess for the whole
session (the kernel and runtime catalog fixtures are session-scoped with a
stable log path, so a per-module path does not spawn a new subprocess per
test module).

## CI gates

Quality jobs run on Ubuntu/Python 3.12. Tests and per-package coverage gates
run on Python 3.11/3.12 across Linux, macOS, and Windows.

Linear history is a repository setting, not a CI job: `main` requires linear
history and disallows merge commits, so GitHub rejects a non-linear merge at
the server. There is no redundant workflow check for it.

Ruff also enforces whitespace, Python-version pitfalls, comprehensions,
performance patterns, redundant code, modernization, and logging formatting
(`W`, `YTT`, `C4`, `PERF`, `PIE`, `FURB`, `G`). These rules apply to source and
tests; the existing complexity limits and scoped exceptions remain in
[pyproject.toml](../pyproject.toml).

```sh
uv run ruff format --check .
uv run ruff check .
uv run python scripts/check_direct_io_policy.py
uv run mypy
uv run vulture
uv run lint-imports
uv run python scripts/check_coverage.py --run
```

Mypy checks shipped sources in all ten workspace packages, including runtime,
bundled packs, and both harnesses. Tests and the Vulture allowlist are excluded;
this is source-wide enforcement, not strict typing of every test or script.
Source roots and namespace-package search paths are centralized in
[pyproject.toml](../pyproject.toml), so the same command works in PowerShell
without a separately maintained `MYPYPATH`. Vulture's roots live there too;
entry-point name prefixes are not blanket exemptions.

Import-linter enforces kernel/integration separation, SDK/runtime/server
separation, and bundled-pack independence. Pack thread reads use the SDK,
not lazy runtime imports.

Coverage is branch-based and excludes test source. The root global floor
is 82%; [check_coverage.py](../scripts/check_coverage.py) enforces separate
package floors. Consult that script for current numbers rather than copying
a dated coverage report. Ratchet floors with evidence; do not lower them
to hide a regression. `xfail_strict = true` makes unexpected passes fail.

The combined command runs the full offline suite **once**, checks the global
and package floors, then requires **95% coverage of changed executable lines**.
It includes quality scripts in measurement. Package floors remain branch-aware;
the changed-line percentage is line coverage, not proof of every branch.

CI runs the suite and package floors on each supported OS/Python pair, then
combines their coverage data before enforcing the changed-line threshold once.
This keeps platform-specific branches in the denominator while counting a line
as covered when any supported matrix job executes it.

[check_changed_coverage.py](../scripts/check_changed_coverage.py) compares the
current tree to its branch point, including committed, staged, unstaged, renamed,
and untracked Python sources. Tests, `conftest.py`, and the Vulture allowlist
are excluded. Comments and non-executable lines do not enter the denominator.
New source absent from coverage counts as uncovered; missing data, line-only
data, unresolved bases, and invalid source fail explicitly. Existing coverage
exclusion pragmas apply, so additions require review rather than using pragmas
to evade the gate.

Use `--base <ref>` with the combined command to name the comparison base
explicitly; the script resolves its merge-base with `HEAD`. CI supplies the PR
base SHA (or the previous main SHA on a push) and fetches full history. Locally,
the default is the newest merge-base with `origin/main` or `main`; unresolved
bases fail instead of checking only uncommitted work. The standalone changed
gate can inspect an existing report, but run the combined command to certify
fresh coverage of the current revision.

The `lisa verify` CLI smoke check is separate: the framework workspace does
not install `lisa`, and the current framework CI does not run that command.

### Targeted mutation testing

The optional Linux-only mutation extra checks the changed-coverage gate itself,
including missing/unmeasured data, Git deltas, and the exact 95% boundary:

```sh
uv sync --frozen --extra dev --extra mutation
uv run --no-sync mutmut run --max-children 2
uv run --no-sync mutmut results --all true
uv run --no-sync mutmut show scripts.check_changed_coverage.x_check_coverage__mutmut_1
```

Mutmut works in an ignored `mutants/` copy, not the shared source checkout.
The selected suite and cache-invalidation inputs live in
[pyproject.toml](../pyproject.toml). Expand the target only with a focused,
fast suite. Inspect survivors: they can expose missing assertions, harmless
message changes, or equivalent behavior. A successful `mutmut run` is **not**
a zero-survivor gate or proof of complete testing. The manually triggered
[mutation workflow](../.github/workflows/mutation.yml) publishes results for
review; it is advisory, separate from required PR gates.

Branch coverage grows through tests for real alternative outcomes, not a
blanket 95% branch requirement. SDK integrity tests now cover unreadable seals,
malformed entries, and missing mid-chain links; file-lock tests cover timeout,
cancellation, failed atomic replacement, and OS-error distinctions.

## Documentation gate

The [Docs workflow](../.github/workflows/docs.yml) runs a documentation-only
check on every push and pull request. It is stdlib-only, so it needs no
workspace install and stays cheap enough to run on every change:

```sh
uv run python scripts/check_docs.py
```

The gate checks the maintained corpus — the guides under `docs/`, the
root-level policy files, and any package `README.md` or `AUTHORING.md`
outside a package source tree — for relative links that do not resolve,
`#anchor` fragments that match no heading, code fences with no language or
no closing fence, files that do not end with exactly one newline, trailing
whitespace outside code blocks, and lists that start without a blank line.
Anchor slugs follow the same rules GitHub uses, so a heading's anchor here
matches the one GitHub renders.

Markdown that is *not* documentation is out of scope: pack content (modes,
operations, prompts, sessions), test fixtures, and Copilot customization
files are inputs to the runtime or the agent host, and the tests that load
them cover them. A change to one of those files still runs the full CI
matrix; a change to a guide does not.

## When CI runs

The [CI workflow](../.github/workflows/ci.yml) skips the quality, test, and
coverage matrix when a push or pull request changes only documentation:
the guides under `docs/`, the root-level `*.md` files, and any `README.md`
or `AUTHORING.md`. The documentation gate still runs, so a doc-only change
is still checked.

Markdown that the runtime or the agent host loads is *not* documentation and
does not trigger the skip. Pack content, test fixtures, and the files under
`.github/` that carry Copilot customizations all run the full matrix. A
`README.md` inside a package source tree is a runtime asset rather than a
guide, so it runs the matrix too. The path filter and the documentation
gate use the same rule, so a file is never documentation for one and
behavior for the other.

The release workflow calls both CI and the documentation gate, so a tag
build is validated regardless of which files the release commit touched.

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

The [documentation gate](#documentation-gate) checks links, anchors, code
fences, and whitespace in the maintained corpus. Run it before committing a
doc change:

```sh
uv run python scripts/check_docs.py
```

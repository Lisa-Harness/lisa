---
name: Python Development
description: "Use when writing, editing, or reviewing Python in the lisa_framework workspace. Enforces docstrings on every public surface, zero new ruff/mypy warnings, and co-maintained pytest tests. Applies to all .py files."
applyTo: "**/*.py"
---

# Python Development

## Docstrings — required on every public surface

Every **module**, **class**, and **public function/method** gets a docstring before a change is considered done. No exceptions for "it's obvious."

- **Modules**: a one-sentence summary of what the module owns, plus a short `DESIGN —` note when the module encodes a non-obvious contract (see `kernel/slice.py` for the house style).
- **Functions/methods**: a summary line, then `Args:`, `Returns:`, and `Raises:` sections (only the sections that apply). One blank line between the summary and the first section.
- **Classes**: a summary line describing the responsibility, not the fields.
- Private helpers (`_leading_underscore`) still get a one-line docstring; skip the `Args:`/`Returns:` sections only when they are genuinely trivial.
- Keep docstrings imperative and terse — describe what the callable does, not the type signature (the signature already says that).

## Logging — best practices

Every Python module must import the `logging` module and define a module-level logger:

```python
import logging
logger = logging.getLogger(__name__)
```

Use `logger.debug()`, `logger.info()`, `logger.warning()`, and `logger.error()` to record key logic branches, state changes, warnings, and errors respectively. Choose the appropriate level to reflect the message’s severity.

## Linter — zero new warnings, immediate hard-fail

The workspace is configured in the root `pyproject.toml`: `ruff` with `select = ["E","F","I","UP","B","SIM","PL","C90","RET","RUF"]` (only `PLR0913` ignored), `line-length = 100`, `target-version = "py311"`, plus `mypy >= 1.10` in `--strict` mode.

Before committing any Python change:

1. Run `ruff format --check .` then `ruff check .` from the repo root. Both must be clean for the files you touched. CI fails immediately on any violation — there is no warn-only stage.
2. Run `mypy` against the affected package. No new `error` or `warning` lines. CI fails immediately on any new typing error.
3. Do **not** add `# noqa` to silence a rule unless you can name the specific code (e.g. `# noqa: PLW0603`) and justify it in a one-line comment above. Broad `# noqa` is a review blocker.
4. Do **not** widen `ignore = [...]` in `pyproject.toml` to make a warning go away. Fix the code, or open a tracked discussion if a rule is genuinely wrong for this codebase.
5. Imports are sorted by `I` (isort) — let `ruff check --fix` handle import ordering; never hand-order imports against the tool.

## Complexity budgets — enforced by Ruff

These thresholds are not aspirational. They are configured in `pyproject.toml` and CI fails on violation. If a function exceeds a budget, split it — do not suppress the rule.

| Budget | Rule | Limit | Action when exceeded |
| --- | --- | --- | --- |
| Cyclomatic complexity | `C901` | ≤ 10 | Extract branch clusters into named helpers. |
| Too many returns | `PLR0911` | ≤ 6 | Consolidate exit paths; use early-guard + single result variable. |
| Too many branches | `PLR0912` | ≤ 12 | Move each branch group into its own function. |
| Too many statements | `PLR0915` | ≤ 50 | Split at logical phase boundaries (parse → validate → execute → render). |
| Too many arguments | `PLR0913` | (ignored) | Prefer a dataclass / Pydantic model / `TypedDict` parameter object over adding positional args. |
| Too many locals | `PLR0914` | ≤ 15 | Extract a small helper class or parameter object. |

## Structural rules — readable, low-complexity, pythonic

These rules are enforced by review and by the linter subset above. They exist so the codebase stays auditable by humans, not just by tools.

- **One responsibility per function.** If the docstring needs "and" to describe what the function does, split it.
- **Flatten closures.** Nested `def` / `async def` inside another function is a code smell. Extract it to a module-level function or a small class with the parent's context passed explicitly as arguments. Closures that capture mutable parent state are untestable and a review blocker.
- **Thin CLI handlers.** `@app.command()` bodies parse arguments and print final output only. Business logic lives in imported functions that return plain Python objects. Never pass `console.print` or Rich renderable construction into core logic.
- **No untyped state bags.** Do not use `dict[str, Any]` as a runtime state container. Model state with a Pydantic `BaseModel` or `@dataclass` so mutations are explicit and typed. Existing untyped bags must be replaced when touched — no compat shims.
- **Typed boundaries, untyped interiors are fine.** Public function signatures and class attributes must be fully typed. Internal helper bodies may rely on inference where it is obvious.
- **Exception handling is specific.** `except Exception:` and bare `except:` are review blockers unless the callable is a top-level process boundary (CLI entry point, server handler, background worker loop) and the catch is followed by structured logging + re-raise or clean exit. Catch the concrete exception type(s) you expect.
- **No f-strings without interpolation.** `f"literal text"` is an `RUF` finding; use a plain string.
- **No unused parameters.** If a protocol requires a parameter you don't use, name it `_` or prefix with `_` and add a one-line `# protocol-required` comment. `ARG002` is enforced.
- **Return early, return often.** Prefer guard clauses over nested `if/else` pyramids. `RET` rules enforce simple return patterns.

## Workflow

- Write the docstring in the same commit as the code it documents. "I'll add the docstring later" is how modules ship without one.
- If you touch an existing public callable that has no docstring, add one as part of the change — leave the file cleaner than you found it.
- If a linter warning is pre-existing on a line you are not editing, leave it; do not scope-creep a fix into an unrelated commit.

## Tests — required and co-maintained

Every public callable added or materially changed in a commit ships with a corresponding test in the same commit. "Materially changed" means the observable behavior changed — not a pure rename, formatting, or docstring-only edit.

- **Location**: tests live under the package's `tests/` directory (e.g. `kernel/tests/test_<module>.py`). Mirror the source module name in the test filename.
- **Coverage target**: every new public function/method gets at least one test exercising the happy path and one test for each documented `Raises:` failure mode. Pure dataclasses and trivial re-exports are exempt.
- **Style**: follow the existing house style — `pytest` + `pytest-asyncio`, `@pytest.mark.asyncio` for coroutines, `tmp_path` for filesystem fixtures, module-level docstring `"""Tests for kernel/<module>.py."""`.
- **Markers**: use the workspace markers from `pyproject.toml` — `live` for tests hitting real external APIs, `integration` for multi-module in-process tests, `eval` for deterministic agent regression evals. Unmarked = fast unit test.
- **Run before commit**: `pytest <package>/tests` must be green for the files you touched. Do not commit a change that leaves a previously-passing test failing.
- **Co-maintenance**: if you change a callable's signature or behavior, update its tests in the same commit. A test that no longer reflects the code is worse than no test — fix it or delete it with a one-line commit-message note explaining why.
- **No test-skipping without a ticket**: `pytest.skip` and `xfail` require a comment naming the tracking issue or a one-line reason. Silent skips accumulate rot.

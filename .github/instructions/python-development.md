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

## Linter — zero new warnings

The workspace is configured in the root `pyproject.toml`: `ruff` with `select = ["E","F","I","UP","B","SIM","PL"]` (only `PLR0913` ignored), `line-length = 100`, `target-version = "py311"`, plus `mypy >= 1.10`.

Before committing any Python change:

1. Run `ruff check .` and `ruff format --check .` from the repo root. Both must be clean for the files you touched.
2. Run `mypy` against the affected package. No new `error` or `warning` lines.
3. Do **not** add `# noqa` to silence a rule unless you can name the specific code (e.g. `# noqa: PLW0603`) and justify it in a one-line comment above. Broad `# noqa` is a review blocker.
4. Do **not** widen `ignore = [...]` in `pyproject.toml` to make a warning go away. Fix the code, or open a tracked discussion if a rule is genuinely wrong for this codebase.
5. Imports are sorted by `I` (isort) — let `ruff check --fix` handle import ordering; never hand-order imports against the tool.

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

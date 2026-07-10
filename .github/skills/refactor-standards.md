---
name: Refactor Standards
description: "Reusable checklist for lint-clean, low-complexity, readable, pythonic refactors in the lisa_framework workspace. Load when planning or executing a refactor, decomposition, or complexity-reduction pass. Covers complexity budgets, structural rules, closure flattening, state typing, exception narrowing, test-seam preservation, and the required verification loop."
applyTo: "**/*.py"
---

# Refactor Standards Skill

This skill is a quick-reference checklist for refactoring Python in the
lisa_framework workspace. It mirrors the authoritative rules in
`.github/instructions/python-development.md` and
`.github/instructions/code-quality.md` — when in doubt, those files win.

## When to load

- Planning a module decomposition or complexity-reduction pass.
- Executing a refactor via the `refactor` agent or manually.
- Reviewing a refactor commit before merging.

## Complexity budgets (Ruff-enforced, CI hard-fail)

| Budget | Rule | Limit |
| --- | --- | --- |
| Cyclomatic complexity | `C901` | ≤ 10 |
| Too many returns | `PLR0911` | ≤ 6 |
| Too many branches | `PLR0912` | ≤ 12 |
| Too many statements | `PLR0915` | ≤ 50 |
| Too many locals | `PLR0914` | ≤ 15 |
| Too many arguments | `PLR0913` | (ignored — use parameter objects) |

If a function exceeds a budget, split it. Do not suppress the rule.

## Structural rules

1. **One responsibility per function.** If the docstring needs "and", split.
2. **Flatten closures.** Move nested `def` / `async def` to module level. Pass parent context explicitly as arguments. Closures capturing mutable parent state are untestable.
3. **Thin CLI handlers.** `@app.command()` bodies parse args and print output only. Logic lives in imported functions returning plain objects. Never pass `console.print` into core logic.
4. **No untyped state bags.** Replace `dict[str, Any]` state containers with Pydantic `BaseModel` or `@dataclass`. Existing accessors stay as thin wrappers for test seams.
5. **Typed boundaries.** Public signatures and class attributes must be fully typed. Internal helpers may rely on inference.
6. **Specific exceptions.** `except Exception:` is a review blocker unless at a top-level process boundary with structured logging + re-raise or clean exit.
7. **No f-strings without interpolation.** `f"literal"` is an `RUF` finding.
8. **No unused parameters.** Prefix with `_` and add `# protocol-required` if a protocol forces the parameter.
9. **Return early.** Prefer guard clauses over nested `if/else` pyramids.
10. **No compat shims.** Hard cutover when adopting a new pattern. Old code goes away.

## Refactor workflow

1. **Baseline.** Run `ruff check <target>` and `mypy <target>`. Capture error counts.
2. **Plan.** List functions exceeding budgets, closures to flatten, state bags to type, exceptions to narrow. Propose target module structure in 5–10 bullets.
3. **Execute in small steps.** After each step: `ruff check <target>`, `mypy <target>`, `pytest <package>/tests`. Must be clean or improved, no regressions.
4. **Verify full package.** `ruff format --check <package>`, `ruff check <package>`, `mypy <package>`, `pytest <package>/tests -m "not live"`.
5. **Report.** Before → after metrics table, what changed, what was preserved, follow-ups.

## Test-seam preservation

Tests in this repo use `monkeypatch.setattr(module, "name", fake)` and
`monkeypatch.setitem(module._RUNTIME_STATE, key, value)`. When moving a name
to a submodule:

- Re-import the name on the parent module surface so `setattr` keeps working.
- Keep `_RUNTIME_STATE` as a plain dict (or a typed model with `__getitem__`/
  `__setitem__` compatibility) so `setitem` keeps working.
- Document the re-export with a one-line comment: `# re-exported for test compatibility`.

## Required verification loop (no warn-only stage)

```powershell
uv run ruff format --check kernel sdk sdk_packs hosts
uv run ruff check kernel sdk sdk_packs hosts
uv run mypy kernel sdk sdk_packs
uv run pytest -m "not live" --tb=short -v
uv run little-lisa verify
```

Any failure blocks the commit. CI enforces the same.


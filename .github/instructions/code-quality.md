---
name: Code Quality
description: "Required verification for framework changes; current commands and CI coverage live in docs/TESTING.md."
applyTo: "**/*"
---

# Code quality

- Start with the smallest reproducing case, then select tests by tier from
  [docs/TESTING.md](../../docs/TESTING.md#test-selection-tiers): the unit
  module(s) mirroring your edited file while iterating, the owning
  package's full testpaths before committing, and the full suite before
  push or PR. A package's tests are split between its top-level `tests/`
  and colocated `src/**/tests/` — always run both halves of an owning
  package; the testing guide's table maps changed files to testpaths.
- For kernel contract changes, run the testpaths of every package that
  consumes the changed contract (import direction = test scope); for a
  package-internal change, that package's tier is enough. When unsure,
  run the full suite — it is the CI gate.
- For code changes, run the complete quality/test/coverage loop in
  [docs/TESTING.md](../../docs/TESTING.md#ci-gates) before committing.
  CI is authoritative for the selected packages and platform matrix.
- Required gates are format, lint, direct-I/O policy, strict mypy on the
  configured sources, dead code, non-live tests, and per-package branch coverage.
  A dated baseline or a focused pass does not certify the full tree.
- For documentation-only changes, check links, paths, schema validity, and
  runnable examples; do not require a live model test.
- `lisa verify` belongs to the external CLI, is not installed by this workspace,
  and is not a current framework CI gate.
- Keep validation commands in the testing guide rather than duplicating them
  here. Never reference removed `hosts/` paths or nonexistent extras.

---
name: Stabilization Freeze
applyTo: "kernel/**,runtime/**,sdk/**,mcp_server/**"
description: Behavior freeze on kernel, runtime, sdk, and mcp_server. Only fixes with regression tests, and tests, are allowed. All other work must branch or wait.
---

# Stabilization freeze

The operator lifted the freeze for context-preservation work on 2026-10-10;
see the scoped exception in `docs/CODING_WORKFLOW.md` § Freeze.

As of 2026-10-04 these trees are frozen to behavior change. The full rules and
rationale live in `docs/CODING_WORKFLOW.md` § Freeze. Follow them.

## Before editing anything under a frozen tree

1. **Name the bug.** You must be able to state, in one sentence, which bug
   this change fixes. "Improving", "cleaning up", "simplifying", or
   "modernizing" are not bugs.
2. **Name the test.** You must be able to point at the test that will fail
   before and pass after. If it does not exist yet, write it first.
3. **Check for a carve-out.** Operator-declared exceptions to this freeze
   are listed in `docs/CODING_WORKFLOW.md` § "Declared carve-outs." Work
   inside a declared carve-out follows the carve-out's stated rules (same
   red/green test receipt, same commit scoping) instead of this gate.
4. **If you can do neither, stop.** Surface the freeze to the operator and
   ask whether to carve out an exception or park the work on a branch. Do
   not decide for them.

## Hard rules

- One bug per commit. A `src/` change must ship with its regression test in
  the same commit.
- No signature changes to anything imported by another package, even if the
  change "looks safe". Frozen means frozen for callers.
- No "while I'm here" edits. If you spot an adjacent issue, file it as a
  follow-up; do not expand the diff.
- The full verification loop (`.github/instructions/code-quality.md`) must
  pass entirely — a green check on only the touched files does not satisfy
  the freeze. A fix that breaks an untouched test is out of scope.
- Do not mutate settings singletons in tests without restoring them
  (`try/finally` or `monkeypatch`). Autouse guards in
  `kernel/tests/conftest.py` and `sdk/tests/conftest.py` fail the suite on
  leaks.
- `docs/`, `e2etest/`, `profiles/`, `scripts/`, `.github/` are not frozen.

---
name: Documentation Drift Audit
applyTo: "**/*.md,**/*.py"
description: "Use when auditing a module's code against its README/spec; treats the doc as the spec and traces every claim to implementation."
---

# Documentation Drift Audit

When asked to audit a module against its documentation (README, spec, or
design doc), follow this loop:

## 1. Extract assertions

Read the doc end-to-end. List every claimable assertion — behavior,
semantics, inputs, outputs, edge cases, ordering, and fallback behavior.
Each gets a stable ID (e.g. `A1`, `A2`).

## 2. Trace to code

For each assertion, locate the code that implements it. Search the module,
its dependencies, and its integration seams. If no implementation exists,
mark `UNIMPLEMENTED`. If the implementation differs, mark `MISMATCH`.

## 3. Classify

| Label | Meaning |
| --- | --- |
| `MATCH` | Code implements exactly what the doc says. |
| `MISMATCH` | Code and doc disagree on behavior. |
| `UNIMPLEMENTED` | Doc promises behavior the code doesn't have. |
| `UNDOCUMENTED` | Code does something the doc doesn't mention. |
| `DEAD` | Doc describes a path that is unreachable in production. |

## 4. Decide remediation direction

- Default: the doc is the spec; the code moves to match it.
- Exception: if the code's behavior is safer or more correct, update the
  doc and add a test that locks the code's behavior.
- Never silently change one without the other.

## 5. Remediate

Make the smallest change that closes the gap. One gap per commit when
practical. Preserve repository conventions and module boundaries.

## 6. Lock with a regression test

- For code fixes: add or update a test that would fail if the gap
  reappeared.
- For doc-only fixes: add a test that exercises the doc's examples or
  key claims against the code.
- For `DEAD` findings: either remove the dead path or add a test that
  proves it's reachable (if the doc is wrong).

## 7. Verify and commit

Run the targeted test suite. Review the diff. Verify the doc and code
agree. Commit with a message referencing the audit.

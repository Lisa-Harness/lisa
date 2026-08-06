---
name: Documentation Drift Audit
description: Audit a module's code against its README/spec. Treat the doc as the spec, trace every claim, classify gaps, and propose remediation.
---

# Documentation Drift Audit: $ARGUMENTS

You are auditing **$ARGUMENTS** (a module + its README/spec).
The document is the spec. The code must be true to it.

## Procedure

1. **Read the doc** (README.md or the named spec) end-to-end. Extract
   every claimable assertion about behavior, semantics, inputs, outputs,
   edge cases, ordering, and fallbacks. Give each a stable ID (`A1`,
   `A2`, ...).

2. **Read the code** — the module, its dependencies, and its integration
   seams (callers, protocols, tests). For each assertion, trace it to
   the implementing code.

3. **Classify each assertion** as `MATCH`, `MISMATCH`, `UNIMPLEMENTED`,
   `UNDOCUMENTED`, or `DEAD`. For each non-`MATCH` finding, cite the
   exact file and line.

4. **Produce a findings table**:

   | ID | Assertion (from doc) | Label | File:Line | Evidence |
   | --- | --- | --- | --- | --- |

5. **For each finding**, propose remediation:
   - Default: fix the code to match the doc.
   - Exception: if the code is safer/correct, fix the doc and add a test.
   - State the direction and the minimal change.

6. **List the regression tests** needed to lock each remediation. For
   doc-only fixes, list tests that exercise the doc's examples.

7. **Do not edit files yet.** Present the findings and proposed changes
   for review first. After approval, apply one fix per commit, run the
   targeted tests, and verify the doc and code agree.

## Rules

- Trace every assertion, not just the ones that look suspicious.
- Cite exact file paths and line numbers.
- Distinguish code bugs from doc bugs from integration bugs.
- Check the integration seam (callers), not just the module in isolation.
- Flag any `DEAD` path (doc describes something unreachable in production).
- Flag any sentinel/trap values that satisfy naive comparisons.
- Check __all__, re-exports, and entry points against what the doc claims
  is public.

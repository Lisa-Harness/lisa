---
name: Engineering Workflow
applyTo: "**/*"
description: Core workflow for making changes in this repository.
---

# Engineering workflow

- Start from the narrowest failing case. Prefer a targeted repro or test before broad edits.
- Keep changes scoped to the current task. Avoid unrelated refactors or formatting churn.
- Make the smallest change that solves the root cause, then verify it with the relevant tests.
- Preserve repository conventions. Follow the existing module boundaries and naming patterns.
- Prefer clear diffs and minimal new surface area. If a new concept is needed, document it in the nearest relevant module or guide.

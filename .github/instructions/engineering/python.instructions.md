---
name: Python Engineering
applyTo: "**/*.py"
description: Python-specific implementation guidance for Lisa.
---

# Python guidance

- Keep Python code straightforward and explicit. Prefer readable, typed functions over clever abstractions.
- Use existing repository patterns before introducing a new helper or framework.
- Favor deterministic behavior and small, testable units.
- Avoid adding unused imports, dead code, or speculative configuration.
- Keep tests close to the behavior they cover and use them to lock in regressions.

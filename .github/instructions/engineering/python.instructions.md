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
- Late (`import-outside-toplevel`, ruff `PLC0415` / pylint `C0415`) imports are allowed only for cycle-breaking, startup-latency deferrals, optional-dependency guards, or PEP 562 plugin discovery. When you intentionally keep a late import, suppress it inline with a reason clause: `# noqa: PLC0415  # <reason>`. A bare `# noqa: PLC0415` is not acceptable. Use one of: `lazy: breaks cycle with <module>`, `lazy: defers <dep> startup cost`, `lazy: optional dep <name>`, `lazy: plugin discovery via PEP 562`. Pure stdlib imports (`json`, `time`, `uuid`, `datetime`, `dataclasses`, `concurrent.futures`) have no cycle risk and must be hoisted to module top-level, not suppressed.

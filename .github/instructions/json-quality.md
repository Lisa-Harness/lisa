---
name: JSON Quality
description: "Use when writing, editing, or reviewing JSON configs in the lisa_framework workspace. Enforces 2-space indent, no trailing commas, and schema validation. Applies to *.json."
applyTo: "**/*.json"
---

# JSON

- Indent with 2 spaces.
- No trailing commas.
- Keep keys sorted alphabetically when the file is a config schema (not when order is semantically meaningful, e.g. CHANGELOG entries).
- Validate against the relevant schema before committing when a schema exists.

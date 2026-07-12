---
name: YAML Quality
description: "Use when writing, editing, or reviewing YAML front matter and configs in the lisa_framework workspace. Enforces required front-matter keys, tight glob scopes, and consistent quoting. Applies to *.yaml, *.yml."
applyTo: "**/*.{yaml,yml}"
---

# YAML (front matter and configs)

- Every agent / prompt / instruction / skill file starts with a YAML front-matter block delimited by `---`.
- Required front-matter keys for each file type:
  - Instructions: `name`, `description`, `applyTo`
  - Agents: `name`, `description`, optional `tools`, `user-invocable`, `argument-hint`
  - Prompts: `description`, `agent`
  - Skills: `name`, `description`, optional `applyTo`
- `description` values are string-quoted when they contain colons, em-dashes, or double quotes; otherwise bare strings are fine.
- `applyTo` uses glob patterns relative to the workspace root. Keep scopes tight — do not use `**/*` when a more specific pattern applies.

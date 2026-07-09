---
name: Code Quality
description: "Use when writing, editing, or reviewing any non-Python artifact in the lisa_framework workspace — Markdown docs, YAML front matter, JSON configs, shell scripts. Enforces lint-clean structure, explicit formatting, and scope-narrowed suppressions. Applies to *.md, *.yaml, *.yml, *.json, *.sh, *.ps1."
applyTo: "**/*.{md,yaml,yml,json,sh,ps1}"
---

# Code Quality

## Markdown

Follow the repo's `markdown-preferences.md` user memory: write lint-clean Markdown by default.

- Surround headings, lists, tables, and fenced code blocks with blank lines unless the file already uses a required exception.
- Use fenced code blocks with an explicit language. If the block is not source code, use `plaintext`.
- Prefer native Markdown over inline HTML. Only use inline HTML or markdownlint suppression comments when Markdown cannot express the required structure.
- Scope suppressions narrowly. If a markdownlint disable comment is unavoidable, limit it to the smallest possible block and preserve or restore the prior rule state.
- Tables must use one style per table. Do not mix aligned, compact, or tight pipe spacing inside the same table. For new tables, prefer compact pipe tables.
- Do not auto-fix repository-wide Markdown in normal workflows. Use non-fixing lint checks by default so unrelated files are not rewritten. Use auto-fix only when explicitly requested and scope it to files intended for cleanup.

## YAML (front matter and configs)

- Every agent / prompt / instruction / skill file starts with a YAML front-matter block delimited by `---`.
- Required front-matter keys for each file type:
  - Instructions: `name`, `description`, `applyTo`
  - Agents: `name`, `description`, optional `tools`, `user-invocable`, `argument-hint`
  - Prompts: `description`, `agent`
  - Skills: `name`, `description`, optional `applyTo`
- `description` values are string-quoted when they contain colons, em-dashes, or double quotes; otherwise bare strings are fine.
- `applyTo` uses glob patterns relative to the workspace root. Keep scopes tight — do not use `**/*` when a more specific pattern applies.

## JSON

- Indent with 2 spaces.
- No trailing commas.
- Keep keys sorted alphabetically when the file is a config schema (not when order is semantically meaningful, e.g. CHANGELOG entries).
- Validate against the relevant schema before committing when a schema exists.

## Shell (PowerShell on Windows, bash on Linux/macOS)

- Match the host OS detected in the session. Do not produce bash syntax on Windows or PowerShell syntax on Linux.
- Parameterize would-be hard-coded values (ports, paths, executables, timeouts, flags). No magic numbers or literal paths inside script bodies — every configurable value gets a `param()` entry with a sensible default.
- Never use forbidden-on-Windows bash commands in PowerShell: `head`, `grep`, `tail -f`, `find`, `rm -rf`, `mkdir -p`, `cp -r`, `mv`, `wc`, `diff`, `xargs`, `which`. Use the PowerShell / `rg` equivalents documented in the user's shell preferences.

## Dev loop — required order

The local dev loop and CI use the same commands in the same order. There is no warn-only stage; any failure blocks.

```powershell
# 1. Format check (deterministic; idempotent)
uv run ruff format --check kernel lisa_sdk sdk_extensions sdk_packs examples

# 2. Lint (complexity + style + import ordering)
uv run ruff check kernel lisa_sdk sdk_extensions sdk_packs examples

# 3. Type check (strict, all public packages)
uv run mypy kernel lisa_sdk sdk_extensions sdk_packs

# 4. Dead code check
uv run vulture kernel lisa_sdk sdk_extensions sdk_packs examples/little_lisa vulture_allowlist.py

# 5. Tests (no live API calls)
uv run pytest -m "not live" --tb=short -v

# 6. Smoke test
uv run little-lisa verify
```

# Copilot customization guide

This repository now uses a compact, intent-driven customization taxonomy under [.github/instructions](../.github/instructions).

## How to add guidance

- Put durable rules in the appropriate taxonomy folder:
  - [.github/instructions/engineering](../.github/instructions/engineering) for workflow and Python guidance.
  - [.github/instructions/architecture](../.github/instructions/architecture) for package boundaries.
  - [.github/instructions/formats](../.github/instructions/formats) for file-format expectations.
  - [.github/instructions/customizations](../.github/instructions/customizations) for customization governance.
- Keep each file short and task-oriented.
- Use the narrowest `applyTo` that still covers the intended files.
- Prefer code, tests, and docs over long instruction text when the detail belongs elsewhere.

## Always-on contract

The always-on project contract lives in [.github/copilot-instructions.md](../.github/copilot-instructions.md). It is the root rule file for repository-wide guidance.

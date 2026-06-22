---
name: Commit Workflow
description: "Use for every implementation change in the lisa_framework workspace. Agents must commit changes immediately after the implementation is complete and verification is green, without waiting for the user to ask."
applyTo: "**/*"
---

# Commit Workflow

## Commit immediately after implementation

When an agent completes an implementation change (code, docs, or config) and the required verification for that change is green or explicitly baseline-qualified, **commit the change immediately** — do not wait for the user to request a commit, and do not batch multiple unrelated changes into one commit.

- "Implementation is complete" means the intended behavior is in place, docstrings/tests required by `python-development.md` are added, and `ruff check`, `mypy`, and `pytest` for the touched files are green (or the failure is pre-existing and noted in the commit message).
- Use one normal commit per shippable slice. Do not amend or rewrite history after pushing.
- If later review or verification finds issues after the commit exists, create `fixup!` commits targeting the affected commit instead of amending.

## Commit scope and hygiene

- Stage and commit only files from the active task. When unrelated working-tree changes exist, continue past them — do not pause for confirmation unless commit scope is genuinely ambiguous.
- Never stage `.vscode/settings.json` unless the user explicitly asks (it carries persistent unrelated spell-check changes).
- Split agent-infrastructure docs (`copilot:`) from product/code commits (`feat:`, `imp:`, `fix:`, etc.). Use the same top-line commit name with the matching prefix.
- Follow the commit conventions in `CONTRIBUTING.md` (`feat:`, `fix:`, `imp:`, `docs:`, `infra:`).

## Release notes

After committing a materially shippable change, update `CHANGELOG.md` proactively in the same commit (or a follow-up `docs:` commit if the change is already pushed) — do not leave release-note intake for a later session.

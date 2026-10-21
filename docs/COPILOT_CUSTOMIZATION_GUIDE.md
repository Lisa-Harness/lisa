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

## Agents

This repository uses [agents](../.github/agents), not prompts. GitHub Copilot
deprecated prompt files, so a prompt is an agent with a different file
extension. An agent carries the same front matter as before (`name`,
`description`, and optionally `tools`, `user-invocable`, `argument-hint`)
and lives under `.github/agents` with a `.agent.md` suffix.

The [Code Reviewer](../.github/agents/code-review.agent.md) agent resolves
which commits to review when the user does not name a range, then gathers the
diff and renders the review. Pass a range or a list of SHAs to review
something specific; omit it to review the working tree or the most recent
commit.

Guidance that is not a task someone invokes belongs in an instruction, not
an agent. The [documentation drift audit](../.github/instructions/doc-drift-audit.md)
is an instruction because it applies whenever a doc or its code changes,
rather than only when a user asks for an audit.

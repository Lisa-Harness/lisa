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

## How customization bodies reach the model

Both scoping mechanisms decide *when* a body costs context, so a turn pays only
for the guidance it can use.

### Instructions — `applyTo` is a filter, not a label

- **Always-on files** (`copilot-instructions.md`, `AGENTS.md`, `CLAUDE.md`) and
  instructions that declare **no** `applyTo` are unconditional. They are
  assembled into the system prompt on every turn.
- **Instructions that declare `applyTo`** are held out of the system prompt.
  The runtime appends a matching body to the first pack-tool result that
  mentions a matching path, at most once per turn and per fork/delegate child.
  Guidance therefore arrives with the file it governs rather than on every turn
  that never touches it.

Two consequences worth knowing when authoring:

- A scoped instruction the model never reads a matching file for is never
  injected. That is the deferred behavior, not a bug — if a rule must always
  apply, leave `applyTo` off.
- Only paths the tool actually carries can match. A tool with no single file
  path (commit, run-check, a full-workspace listing) is never decorated. Write
  path-scoped rules for guidance a *file* should carry, and unscoped rules for
  process that applies to any turn.

### Skills — the index is eager, the body is on demand

A skill carries no `applyTo`, so it cannot be matched to a path; the model has
to choose. The system prompt therefore carries a **skill index** — name,
description, and the path of the `SKILL.md` to read — and never the body.

Loading is an ordinary file read of that path in source scope, so no dedicated
loader tool is needed and the behavior matches Copilot's progressive loading: a
skill's own reference files load only if its body names them, and a skill no
turn loads costs only its index line.

Two behaviors to be aware of:

- **`disable-model-invocation: true` is honored.** Such a skill is neither
  listed nor injected. Copilot keeps it reachable as a slash command; Lisa has
  no skill-slash surface, so listing guidance the model is forbidden to load
  would be worse than omitting it.
- **The description is the discovery surface.** It is what the model matches
  against, so put the trigger words in it. A vague description means the skill
  is never loaded.

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

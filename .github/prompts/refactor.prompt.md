---
description: "Resolve the refactor target from the current context, then hand the explicit target to the refactor agent. The agent only refactors the resolved target — it does not infer scope. Use when: refactor, decompose, split module, reduce complexity, flatten closures, fix linter errors, fix mypy errors, clean up god module."
agent: Refactor Engineer
---

Your only job in this prompt is to **resolve the refactor target** and pass it
to the `refactor` agent as an explicit file path or module glob. Do not
refactor the code yourself and do not delegate refactor decisions — the agent
owns the refactor once it has the target.

## Resolve the target

Determine the target to refactor, in this order of preference:

1. **Explicit input wins.** If the user supplied a file path, module glob, or
   description in the surrounding chat context, use that. If a natural-language
   description was given (e.g. "the Little Lisa main file", "the kernel turn
   runner"), resolve it to an explicit path by searching the workspace. State
   the resolved target and the inference rationale.
2. **Active editor file.** If there is no explicit input and a file is open in
   the editor, use that file's path as the target. Note that the review
   includes the active editor file.
3. **Working tree.** If there is no explicit input and no active editor file,
   but there are uncommitted working-tree or staged changes, treat the set of
   changed `.py` files as the target. Use `git diff --name-only HEAD` to list
   them.

If none of the above yields a target, stop and tell the user: "No refactor
target detected. Provide a file path, module glob, or description, or open a
file in the editor." Do not invoke the agent.

## Validate before delegating

Run `ruff check <target>` to confirm the target exists and has lint findings
worth refactoring. If the target is already clean, tell the user: "`<target>`
is already lint-clean — no refactor needed." Do not invoke the agent.

## Hand off to the agent

Invoke the `refactor` agent with:

- The **explicit resolved target** (e.g. `examples/little_lisa/main.py`,
  `kernel/turn_runner.py`, or a space-separated list of paths).
- A one-line **inference rationale** stating which resolution path was used.

The agent expects a concrete target only — do not pass natural-language
descriptions or vague pointers. The agent will read the repo instructions,
baseline diagnostics, plan the decomposition, execute, verify, and report.
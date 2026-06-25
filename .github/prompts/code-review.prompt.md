---
description: "Resolve the changeset to review from the current git state, then hand the explicit range to the code-review agent. The agent only reviews the resolved changeset — it does not infer scope. Use when: review commits, review diff, code review, security review, privacy review, review changes, review last commit, review HEAD, review fixup, pre-merge review, review PR, review session changes."
agent: Code Reviewer
---

Your only job in this prompt is to **resolve the changeset** and pass it to the
`code-review` agent as an explicit git range. Do not review the code yourself
and do not delegate review decisions — the agent owns the review once it has
the range.

## Resolve the changeset

Determine the range to review, in this order of preference:

1. **Explicit input wins.** If the user supplied a range, SHA list, or
   description in the surrounding chat context, use that. If a natural-language
   description was given (e.g. "the last two commits", "everything since the
   v0.4.0 tag"), resolve it to an explicit range by running
   `git log --oneline -20` (and `git log --oneline <tag>..HEAD` if a release tag
   is mentioned). State the resolved range and the inference rationale.
2. **Working tree.** If there is no explicit input and there are uncommitted
   working-tree or staged changes, treat `HEAD` plus the working tree as the
   change set. Use `git diff HEAD` for the diff and note that the review
   includes uncommitted changes.
3. **Recent commits.** Otherwise, review the most recent commit
   (`HEAD~1..HEAD`), or the last few commits if they clearly belong to the same
   logical change. Run `git log --oneline -n 10` and group by message prefix
   (`fixup!`, `feat:`, `imp:`, etc.) to decide the boundary.

If none of the above yields any changes, stop and tell the user: "No changes
detected to review. Provide a range, SHAs, or a description, or make a change
first." Do not invoke the agent.

## Validate before delegating

Run `git log --oneline <range>`. If it returns no output or exits with an error,
stop and tell the user: "Could not resolve range `<range>` to any commits.
Please verify the range is correct and that all referenced commits are
available locally (run `git fetch` if needed)." Do not invoke the agent.

## Hand off to the agent

Invoke the `code-review` agent with:

- The **explicit resolved range** (e.g. `HEAD~3..HEAD`, `abc123..def456`, or
  `HEAD` for working-tree review).
- A one-line **inference rationale** stating which resolution path was used.

The agent expects a concrete range only — do not pass natural-language
descriptions or vague pointers. The agent will gather the diff, read workspace
instructions, and render the structured review.

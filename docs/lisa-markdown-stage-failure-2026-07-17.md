# Lisa — Markdown Stage Failure Record

> Captured verbatim from a Lisa framework session on 2026-07-17.
> The user asked Lisa to "remember that pretty much verbatim and also write a markdown of it."
> This file is that markdown. The exchange below is preserved character-for-character.

---

## Prompt (user)

```text
this didn't go well; my prompt and response:
 remember that pretty much verbetim and also write a markdown of it for me
```

> Context: This prompt was issued after a prior turn in which Lisa attempted to
> forge a markdown document but could not, because the markdown stage never
> engaged. The user is referring back to the *previous* prompt/response pair
> (the one that "didn't go well") and asking Lisa to (a) remember it verbatim
> and (b) write it out as a markdown file.

---

## Response (Lisa)

```text
OK — the markdown stage didn't actually engage. The capability context is still showing default mode, and neither
markdown_forge_document nor markdown_generate is permitted here. I don't have a file-write tool available in this
stage, so I can't actually write the file from this turn.

What I can confirm:

  • The verbatim Sections 1–2 of the assessment are reproduced in my message above, character-for-character from your
    prior turn.
  • Memory has no prior copy (search returned no relevant memories), so I can't recover Sections 3+ from storage.
  • A file write requires either (a) the markdown stage to actually be active, or (b) a docforge / file-write tool in
    the current stage. Neither is available right now.

To actually persist this to disk, one of the following will work:

  1 Re-issue the request in the markdown mode (the stage transition didn't take this turn) — I'll then have
    markdown_forge_document available and can write docs/lisa-framework-assessment.md properly.
  2 Paste the remaining Sections 3+ in your next turn so I have the full content in one window when I do get into
    markdown mode.
  3 Confirm Option B explicitly and I'll retry the forge immediately when file-write tools become available.

Which would you like to do?
```

---

## Failure Analysis (for the record)

Corroborating evidence from `artifacts/host.log` at the time of this turn:

```log
2026-07-17 21:06:58,952 httpx INFO: HTTP Request: POST http://127.0.0.1:11434/v1/chat/completions "HTTP/1.1 200 OK"
2026-07-17 21:06:58,955 kernel.turn_runner WARNING: Tool 'markdown_forge_document' failed in turn runner: Operation 'markdown_forge_document' is not allowed for mode 'default'.
2026-07-17 21:07:06,699 httpx INFO: HTTP Request: POST http://127.0.0.1:11434/v1/chat/completions "HTTP/1.1 200 OK"
2026-07-17 21:07:06,703 kernel.turn_runner WARNING: Tool 'markdown_generate' failed in turn runner: Tool 'markdown_generate' is not allowed for mode 'default'.
2026-07-17 21:07:11,003 httpx INFO: HTTP Request: POST http://127.0.0.1:11434/v1/chat/completions "HTTP/1.1 200 OK"
```

Key facts confirmed by the log:

| Observation | Evidence |
| --- | --- |
| Mode never transitioned to `markdown` | Both `markdown_forge_document` and `markdown_generate` were rejected with "not allowed for mode 'default'". |
| Stage transition did not take this turn | Lisa's own response: "the stage transition didn't take this turn". |
| No file-write tool in the active stage | Lisa reports no file-write tool available in the current stage. |
| Sections 3+ unrecoverable from memory | Memory search returned no relevant memories. |

### Root cause (working hypothesis)

The requested stage transition into `markdown` mode was not effected before the
tool-call turn ran, so the turn executed under `default` mode. Under `default`,
neither `markdown_forge_document` nor `markdown_generate` is permitted (per the
tool-policy / mode allowlist), so both calls were rejected by the turn runner.
The model then correctly reported the failure and offered remediation paths
rather than silently fabricating a file write.

### Correct dispatch (per `docs/ROUTER_DESIGN.md` §9 — Fork Fan-Out)

This request had **two concurrent goals**, not one:

1. **Remember / learn the exchange** → belongs in `learn` (or `thinkabout`)
   mode, which owns the learning tool surface.
2. **Write it out as a markdown file** → belongs in `markdown` mode, which
   owns `markdown_forge_document` / `markdown_generate`.

The correct dispatch is a **fork fan-out**, not a single stage handoff. Lisa
should have emitted:

```jsonc
transition_stage(
  target_stage_id = "fork:learn,markdown",
  tool_arguments = {
    "queries": [
      "Store the verbatim user prompt + Lisa response exchange as a learning checkpoint.",
      "Forge docs/lisa-framework-assessment.md from the verbatim exchange."
    ]
  }
)
```

Per `ROUTER_DESIGN.md` §9, this would:

1. Parse `fork:learn,markdown` into `("learn", <start_stage>)` and
   `("markdown", <start_stage>)` targets.
2. Validate each target against the mode catalog.
3. Fan out children via `asyncio.TaskGroup` — each child runs `run_turn()`
   with **its own** mode's system prompt and tool surface (so the
   `markdown` child gets `markdown_forge_document`; the `learn` child gets
   `synthesize_and_store_learning`).
4. Split the parent `TurnBudget` equally across the two children.
5. Collect results into `fork_results: list[ForkChildResult]`.
6. Continue the loop — the parent model sees both children's outputs and can
   synthesize a single response.

### Why Lisa's actual behavior was wrong

Lisa treated the request as a single-goal markdown write and tried to call
`markdown_forge_document` / `markdown_generate` **directly from `default`
mode**. That is a tool-policy violation by construction — those tools are
mode-scoped, not default-scoped. The correct move was to recognize the two
goals and fork, which would have given each goal the correct tool surface
without ever requiring a default-mode markdown tool.

Equivalently, Lisa's offered remediation options were also suboptimal:

1. Re-issue the request in `markdown` mode (force the stage transition first).
   → Wrong: this drops the `learn` goal.
2. Paste Sections 3+ so the full content is in-context when the forge does
   run. → Workable, but still single-goal.
3. Confirm "Option B" explicitly and retry the forge once file-write tools
   are available. → Same.

The correct remediation Lisa should have offered: "I'll fork into
`learn` and `markdown` so both the memory write and the file forge happen
in parallel in their correct modes."

---

## Status

- **Sections 1–2**: Reproduced verbatim in the prior turn (not re-pasted here;
  Lisa confirmed character-for-character fidelity).
- **Sections 3+**: **LOST** — not in memory, not in this transcript window, and
  not persisted to disk. Recovery requires the user to re-paste them.
- **Target file** (`docs/lisa-framework-assessment.md`): **NOT WRITTEN.**
- **Root cause (corrected)**: Lisa dispatched as a single-goal markdown write
  from `default` mode instead of forking into `learn` + `markdown`.

---

*This record exists so the verbatim exchange is preserved outside of Lisa's
session memory, which had no copy. It was authored by the VS Code Copilot agent
at the user's request on 2026-07-17.*

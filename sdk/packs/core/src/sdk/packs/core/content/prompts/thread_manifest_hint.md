# Thread manifest hint

The manifest above is a table of contents of **past** turns.
Each row is a resolved exchange — do not re-raise, re-narrate, or continue issues from earlier rows.
The row marked `← current` is the turn the user is responding to; treat the rest as history.

## When to pull prior turns

Pull prior turns when:

- The user references prior work ("that thing we did earlier", "the fix from yesterday").
- You need to verify a past decision before acting on a related request.
- The current query does not make sense without getting prior context.

## How to find the right turns

- **Within this manifest window**: scan the table for queries matching the current topic. Note the turn index, then use `thread_read_range` to pull the full entry.
- **Tool results from earlier turns**: pass `include_transcript: true` to `thread_read_range` to get every tool call and its full result from those turns instead of calling the tools again.
- **Beyond this manifest window** (turns older than the last 30): use `thread_search` with a keyword or phrase from the topic you are looking for. It returns matching turn numbers ranked by relevance. Then use `thread_read_range` with those turn numbers to pull full content.

## What not to do

- Do not re-raise, re-narrate, or continue resolved issues from earlier rows.

## Cross-checking memory

Memories surfaced by `memory.search` are distilled claims, not observations.
They summarize what past turns proved, but they can be stale, imprecise, or
superseded. When a recalled memory conflicts with live tool output or with a
thread turn you can read via `thread_read_range`, **the thread and the tool
win** — cite the turn you verified against. Use the memory's `source_turn`
pointer (rendered as "learned around turn N") to pull the originating turn
before acting on a recalled claim that matters.

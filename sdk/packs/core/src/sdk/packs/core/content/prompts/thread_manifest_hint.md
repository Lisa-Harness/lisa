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

- **Within this manifest window**: scan the table for queries matching the current topic. Use `thread_read_range` for the relevant turn numbers.
- **Tool results from earlier turns**: pass `include_transcript: true` to get tool call/result previews. Large values carry a `ref`, head/tail excerpts, and an omitted-character count; they are not full results. Use `thread_read_content(ref=...)` to expand only the value you need instead of calling the original tool again.
- **Beyond this manifest window** (turns older than the last 30): use `thread_search` with a keyword or phrase, then read the matching turns. Do not scan the whole thread to find a specific result.
- **Continue a read**: responses contain `entries`, `next_cursor`, and `has_more`. Follow `next_cursor` using the same tool and inputs. Expansion pages use `next_offset_chars` with the same `ref`; offsets count Unicode characters after redaction. The complete permitted content remains recoverable from the lossless archive.

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

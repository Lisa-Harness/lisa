# Ledger Schema Documents

These files are the committed JSON Schema documents for every ledger topic
declared in [`kernel.evidence.TOPIC_SPECS`](../../../kernel/src/kernel/evidence.py).
They are **generated**, never hand-edited.

| File | Topic |
| --- | --- |
| `model.request.schema.json` | `model.request` |
| `routing.decision.schema.json` | `routing.decision` |
| `routing.forced.schema.json` | `routing.forced` |
| `tool.telemetry.schema.json` | `tool.telemetry` |
| `turn.decision.schema.json` | `turn.decision` |

## Why commit generated files at all

`kernel.evidence.render_schema(topic)` can produce these bytes on demand, so
the files are redundant as data. They are committed for one reason: a change
to a payload model then shows up as a reviewable diff in the pull request that
introduces it. A new ledger field is a change to the durable on-disk contract,
and a reviewer should see it.

## The rule that matters: write contract, not historical validator

Every document sets `"additionalProperties": false`. That is correct and
load-bearing **at write time**: the handler drops any key a topic does not
declare, so a publisher bug cannot leak content to disk
(see [the spec](../EVIDENCE-ENVELOPE.md) §6).

It is **not** a retrospective validator, and this is the trap:

- Adding a field bumps the topic's `schema_version`. Existing rows keep the
  old version and the old key set.
- Removing or retyping a field is a hard cutover with a new version. Rows
  written before the cutover still carry the retired key.
- Therefore validating a historical row against the current document would
  reject evidence that was written legally at the time.

**Readers must tolerate unknown keys in older rows rather than validate them
against the current schema.** Rows are versioned so a reader can decide what it
understands; a row newer than the reader is reported unreadable, never silently
skipped (spec §1).

That rule is restated inside each document as a top-level `$comment`. JSON has
no comment syntax, so `$comment` is the only way to ship commentary in a
`.json` file; it is a standard JSON Schema annotation keyword that validators
ignore by specification, and it survives a `json.dumps`/`json.loads` round
trip.

The `$comment` is deliberately short — one or two sentences. The schema object
is also consumed as a live Python value (prompts, tool descriptions, host
validation), so a long rationale there would ride along into every consumer.
This file carries the full explanation; the `$comment` carries the imperative
so an agent holding the schema object still sees the rule.

## Regenerating

The documents are byte-for-byte `render_schema(topic)`: `indent=2`,
`sort_keys=True`, and a trailing newline. Regenerate from the repository root
with the project interpreter:

```python
from pathlib import Path

from kernel.evidence import TOPIC_SPECS, render_schema, schema_filename

out = Path("docs/SPEC/schemas")
for topic in sorted(TOPIC_SPECS):
    (out / schema_filename(topic)).write_text(render_schema(topic), encoding="utf-8")
```
Only commit the regenerated result if it differs. The drift test
(`kernel/tests/test_evidence_envelope.py`) fails when a committed file is not
exactly `render_schema(topic)`, so a stale file cannot land silently.

If you changed a payload model, also bump that topic's `schema_version` in
`TOPIC_SPECS` — the drift test cannot know whether you intended a new version.

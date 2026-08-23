# Thread Store Format Specification

> **Status:** Active. Introduced 2026-08-10.
> **Implementation:** `host/thread.py`, `host/state_provider.py`

## Overview

The Lisa thread store is a **lossless**, **append-only** conversation
history backed by a three-tier file layout:

1. **Active segment** — plain JSONL, append-only. This is the only file
   the recent-N read ever touches, so reads are O(`limit`) in practice
   rather than O(history). Bounded by `segment_turns` entries.
2. **Sealed segments** — zstd-compressed JSONL, one per sealed chunk.
   Sealing happens at append time when the active segment crosses
   `segment_turns` — no background scheduler, no cron.
3. **Sparse index** — one JSON line per sealed segment recording the
   turn/timestamp range it covers. O(segments) = O(turns /
   `segment_turns`), i.e. tiny forever.

The store is keyed by a **base path** (`<thread_dir>/<user_slug>`, no
file extension). All derived file paths are computed from this base.

## On-disk layout

| File | Purpose |
| --- | --- |
| `<base>.current.jsonl` | Active segment — plain JSONL, append-only |
| `<base>.<seg_id>.jsonl.zst` | Sealed segment — zstd-compressed JSONL |
| `<base>.index.jsonl` | Sparse segment index — one JSON line per sealed segment |
| `<base>.digest.md` | Lossy digest sidecar (per-user) |
| `<base>.digest.marker` | Last-digested `seg_id` (plain text) |
| `<base>.dict.zstd` | Trained zstd dictionary (optional) |

Where `<base> = <thread_dir>/<user_slug>` and `<seg_id>` is a zero-padded
4-digit counter (`0001`, `0002`, ...).

### Legacy migration

A one-time rename guard in `append_thread_entry` moves a legacy
`<base>.jsonl` to `<base>.current.jsonl` on first write. The old path is
never read again after the rename (hard cutover, no compat shims, no
dual-path read code).

The legacy shared `thread_digest.md` sidecar (per-directory, not
per-user) is **orphaned** — it is not migrated to the new per-user
`<base>.digest.md`. Operators may delete stale `thread_digest.md` files
manually if they care; the new store starts a fresh digest per user.

## Record schemas

### Thread entry

One JSON object per line in the active and sealed segments:

```json
{
  "timestamp_utc": "2026-08-10T12:34:56.789000+00:00",
  "user_slug": "adam",
  "mode_id": "default",
  "query": "tell me about your context",
  "response_text": "here is my reflection",
  "provenance": {"source": "observed", "finish_reason": "stop", "model_id": "gpt-5.4-mini"}
}
```

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `timestamp_utc` | string (ISO 8601) | yes | UTC timestamp of the turn |
| `user_slug` | string | yes | User identity |
| `mode_id` | string | yes | Mode that handled the turn |
| `query` | string | yes | User's input |
| `response_text` | string | yes | Model/system response |
| `provenance` | object | no | Optional metadata (`source`, `finish_reason`, `model_id`) |
| `turn` | int | yes | 0-based global turn index |
| `effect_receipts` | list[object] | no | Each: `{capability_id, effect, artifact_id}` |
| `handoff_to` | string | no | Mode handoff target |
| `transitioned_to` | string | no | Stage transition target |
| `yielded` | bool | no | True when the turn paused to ask the user (YieldAction) |
| `yield_question` | string | no | The question surfaced when `yielded` is true |
| `turn_input` | object | no | Raw, unedited model input captured at the `hooks.complete` seam: `{"calls": [{iteration, messages, tools, model_id, provider, reasoning_model, temperature, max_tokens, response_format}, ...]}`, one entry per model round-trip in the turn. Absent for turns with no model call (host commands) or turns predating this field. Read back by `/debug [N]`. |

### Segment index record

One JSON line per sealed segment in `<base>.index.jsonl`:

```json
{
  "seg_id": "0001",
  "turn_lo": 0,
  "turn_hi": 255,
  "ts_lo": "2026-08-10T12:00:00+00:00",
  "ts_hi": "2026-08-10T14:30:00+00:00",
  "bytes_raw": 1048576,
  "bytes_zst": 218000
}
```

| Field | Type | Notes |
| --- | --- | --- |
| `seg_id` | string | Zero-padded 4-digit segment id |
| `turn_lo` | int | Global turn index of the first entry in this segment (0-based) |
| `turn_hi` | int | Global turn index of the last entry |
| `ts_lo` | string | Timestamp of the first entry |
| `ts_hi` | string | Timestamp of the last entry |
| `bytes_raw` | int | Uncompressed segment size |
| `bytes_zst` | int | Compressed segment size |

### Digest marker

Plain text file containing the `seg_id` of the last digested sealed
segment. When all sealed segments have been digested, this equals the
most recent `seg_id` in the index.

## Write path

### Append

1. Compute `current_path = <base>.current.jsonl`.
2. One-time rename guard: if `current_path` does not exist but the
   legacy `<base>.jsonl` does, rename it to `current_path`.
3. Count lines in `current_path`. If the count is >= `segment_turns`,
   seal the active segment first.
4. Append the new entry as JSON (`ensure_ascii=False`) to the (possibly
   fresh) active segment.

### Sealing

Sealing happens **synchronously at append time** when the active segment
reaches `segment_turns` entries. There is no background scheduler and no
cron — this matches the house rule "no cron in lightweight hosts". For a
256-turn segment, sealing (zstd compress + index append + unlink) takes
<10ms, so the user's turn ack is not delayed.

1. Read the active segment bytes.
2. Compress with zstd (level 5, optional trained dictionary).
3. Write `<base>.<seg_id>.jsonl.zst`.
4. Compute the global turn range from the index (sum of prior sealed
   segment turn counts).
5. Append a `SegmentRecord` to `<base>.index.jsonl`.
6. Delete the active segment file (a fresh one is created by the next
   append).
7. Opportunistic dictionary retraining (every 10 seals).

### Dictionary training

- The **first seal** uses raw zstd (no dictionary).
- After the first seal, a zstd dictionary is trained from the sealed
  segment and stored as `<base>.dict.zstd`.
- The dictionary is **retrained every 10 seals** from the most recent 10
  sealed segments. This exploits the repeated JSON keys and similar turn
  vocabulary for 4–6× compression.

## Read paths

### Recent-N (`load_recent_thread_entries`)

1. Read only `current.jsonl`. If it has >= `limit` entries, return the
   last `limit`.
2. If it has fewer, read the index, find the most recent sealed segment,
   decompress it, and take the needed tail entries from it.
3. **Two file touches max.** This fixes the legacy O(file-size) bug —
   the active segment is bounded by `segment_turns`, so the read is
   O(`limit`) in practice.

### Range query (`load_thread_range`)

1. Consult the index to find segments overlapping `[start_turn, end_turn]`.
2. Decompress only those segments, slice in memory.
3. For the active (unsealed) tail, read it directly and include entries
   in range.
4. Returns entries in chronological order.

### Lossless round-trip guarantee

zstd is a lossless compressor. Sealed segments decompress to the exact
bytes of the original active segment. The `load_thread_range` API gives
the harness a way to pull old turns back in by range on demand.

## Compaction / digest interaction

### `FileBackedStateProvider` slice

The deterministic slice summarizes **sealed segments** (not the active
segment) into the per-user `<base>.digest.md` sidecar:

1. Read `<base>.index.jsonl` to list sealed segments.
2. Read `<base>.digest.marker` for the last-digested `seg_id`.
3. Decompress undigested sealed segments (budget-bounded by
   `max_thread_entries` segments per slice).
4. Build digest entries (same host-command filtering, same 200-char
   preview as before).
5. Append to `<base>.digest.md`, apply the digest line cap.
6. Update `<base>.digest.marker`.

The active segment is **never truncated** — sealing already moves old
turns out of it. `max_thread_entries` now bounds how many sealed segments
to digest per slice (budget-bounded), not a truncation threshold.

## Configuration

| Env var | Default | Purpose |
| --- | --- | --- |
| `LISA_THREAD_DIR` | `<data_dir>/threads` | Thread directory |
| `LISA_THREAD_MAX_ENTRIES` | `50` | Recent-N window for context injection; also bounds digest segments per slice |
| `LISA_THREAD_SEGMENT_TURNS` | `256` | Seal threshold — active segment is sealed when it reaches this many entries |

`LISA_THREAD_MAX_ENTRIES` (50) and `LISA_THREAD_SEGMENT_TURNS` (256) are
independent: the former controls how many recent turns enter the model's
context, the latter controls how often the active segment is sealed.
Context injection is unchanged (last 50 turns) while the store scales to
near-infinite history.

## Scaling notes

### Segment size rationale

256 turns per segment was chosen so that whole-segment decompress takes
<10ms (measured on typical conversation data with zstd level 5). This
keeps both the append path (seal) and the read path (recent-N spill,
range query) well under user-perceptible latency.

### Seekable frames deferral

Whole-segment decompress is fine for 256-turn segments. For larger
segments, zstd's seekable frame format (`frame_dlog`) enables
byte-offset range reads within a compressed file without full
decompress. This is deferred until segment size grows beyond ~1000 turns
or decompress latency exceeds ~50ms. The deferral is marked with inline
`# TODO(seekable-frames):` comments at the compress/decompress call sites
in `host/thread.py` so a later agent can find it.

### Future generalization

`kernel/token_usage.py` has the identical "flat JSONL, read tail"
pattern (`open("a")` append; `read_text().splitlines()[-limit:]` tail
read). It is the next candidate for a shared `SegmentedJsonlStore`
helper. Implementation is deferred until this thread-store change is
complete, committed, and verified.

# Session Context Authoring Guide

This guide explains how to write `session_context.md` (or a `session_context/` folder) for Lisa hosts, including front-matter controls for routing, pack policy, and replay behavior.

## What Session Context Does

Session context lets an author define:

- A named session identity (`name`) that influences thread naming and status display.
- Per-turn context text injected into prompt assembly.
- Runtime startup controls (`start_mode`, pack policy, preferred providers).
- Replay controls (`replay.*`) for when/how history and diagnostics are replayed.

Precedence is:

1. Session context front matter
2. CLI flags and environment variables
3. Host defaults

## Where Lisa Loads It From

Resolution order for `--session-context` is:

1. Auto-discovery when empty:
   - `./session_context.md`
   - `./session_context/`
   - `<data-dir>/session_context.md`
   - `<data-dir>/session_context/`
2. Literal local path (file or directory)
3. Data-dir slug (`<data-dir>/<slug>.md` or `<data-dir>/<slug>/`)
4. Authored content slug (`user_contexts/<slug>.md`)
5. Remote registry slug (`https://raw.githubusercontent.com/lisa-framework/sessions/main/<slug>.md`)
6. Direct URL (`https://...`)

Remote fetches are cached in `<data-dir>/session_context_cache/` with conditional GET.

## Minimal Example

```yaml
---
name: acme-session
applies_to:
  - default
---

You are assisting the Acme platform team.
Focus on operational clarity and short action lists.
```

## Full Example

```yaml
---
name: acme-ops
surface:
  type: team_channel
  id: ops-war-room
  memory_scope: shared

start_mode: discuss

applies_to:
  - default
  - discuss

packs:
  lisa-pack-search:
    enabled: true
  lisa-pack-mem0:
    enabled: false

policy:
  always_on_packs:
    - lisa-pack-core
    - lisa-pack-learning
  exclude_servers:
    - lisa-pack-mem0.memory
  preferred_capability_providers:
    memory: lisa-pack-learning

context_routing:
  mode_intent_hints:
    discuss:
      - architecture
      - tradeoff

replay:
  trigger_mode: hybrid
  include_prior_conversation_digest: true
  include_recent_conversation: true
  include_diagnostic_reflection: false
  min_query_chars: 24
  history_max_tokens: 1000
  reflection_max_tokens: 500
  digest_max_lines: 120
  digest_summary_max_chars: 180
---

Primary goals this week:
- Ship migration with rollback checkpoints.
- Keep postmortems blameless and action-oriented.
```

## Front-Matter Reference

| Key | Type | Stage | Description |
| --- | --- | --- | --- |
| `name` | `string` | Startup + per-turn | Session display name and thread slug identity. |
| `applies_to` | `string` or `string[]` | Per-turn | Include this file only for matching modes. |
| `surface.type` | `string` | Per-turn | Override surface type (`personal_dm`, `team_channel`, etc.). |
| `surface.id` | `string` | Per-turn | Override surface ID. |
| `surface.memory_scope` | `string` | Per-turn | Override memory scope identity used by tools/guards. |
| `start_mode` | `string` | Startup | Seed initial mode and bypass front-door for startup routing. |
| `packs.<pack>.enabled` | `bool` | Startup | Pack intent in session controls for host policy wiring. |
| `policy.always_on_packs` | `string[]` | Startup | Packs that cannot be disabled. |
| `policy.exclude_servers` | `string[]` | Startup | MCP server IDs suppressed at startup. |
| `policy.preferred_capability_providers` | `map<string,string>` | Startup | Capability-to-pack provider preference. |
| `context_routing` | object | Startup | Optional context routing overrides (validated structure). |

### Replay Controls (`replay.*`)

| Key | Type | Stage | Behavior |
| --- | --- | --- | --- |
| `trigger_mode` | `always \| conditional \| manual \| hybrid` | Per-turn | Controls when history replay is eligible. |
| `include_prior_conversation_digest` | `bool` | Per-turn | Include/exclude digest lane section. |
| `include_recent_conversation` | `bool` | Per-turn | Include/exclude recent excerpt section. |
| `include_diagnostic_reflection` | `bool` | Per-turn | Include/exclude diagnostic reflection section. |
| `min_query_chars` | `int` | Per-turn | For `conditional`/`hybrid`, replay is gated until query length reaches this value. |
| `history_max_tokens` | `int` | Per-turn | Tighten-only cap for history lane allocation. |
| `reflection_max_tokens` | `int` | Per-turn | Tighten-only cap for reflection lane allocation. |
| `digest_max_lines` | `int` | Per-turn + slice | Max lines retained when replaying digest and when compaction writes digest. |
| `digest_summary_max_chars` | `int` | Slice | Max length for lossy summary line during digest rollover. |

## Tighten-Only Safety Model

Replay budget knobs are tighten-only:

- `history_max_tokens` and `reflection_max_tokens` may reduce effective lane caps.
- They cannot raise caps above kernel defaults.
- Invalid or non-positive values are ignored and defaults are used.

Likewise, malformed replay values degrade gracefully and do not abort startup.

## Trigger Semantics

- `always`: replay sections are included when present.
- `conditional`: replay sections require threshold checks (currently `min_query_chars`).
- `manual`: automatic history replay is suppressed.
- `hybrid`: conditional automatic behavior; compatible with future manual one-turn force paths.

## Directory Mode (`session_context/`)

If `session_context` is a directory, all files are read in sorted order:

- Each file is parsed for front matter.
- `applies_to` is evaluated per file.
- Matching file bodies are concatenated with blank lines.

Use this to split large context into topic files.

## Remote Context Authoring Tips

For remote URLs or registry slugs:

- Prefer immutable branch/tag URLs for reproducibility.
- Use `--session-context-refresh` when you need a hard refresh.
- Keep private contexts protected via `LISA_SESSION_CONTEXT_TOKEN` or `LISA_SESSION_CONTEXT_AUTH_HEADER`.
- Keep host allowlist strict via `LISA_SESSION_CONTEXT_ALLOWED_HOSTS`.

## Common Pitfalls

- Setting `start_mode` to a mode that your host does not support.
- Using replay caps expecting expansion; caps only tighten.
- Forgetting `applies_to` in multi-mode setups and unintentionally over-injecting context.
- Large digest growth without setting `replay.digest_max_lines`.

## Validation Workflow

1. Start host with explicit context:

```powershell
uv run lisa cli --session-context .\session_context.md --debug
```

1. Inspect runtime-effective settings:

```text
/settings
/debug context
```

1. Verify replay behavior by varying query length and mode.

1. Run tests when changing controls/wiring:

```powershell
pytest hosts/core/tests/test_context_assembly.py hosts/core/tests/test_core_services.py -k "replay or session_context" -q
```

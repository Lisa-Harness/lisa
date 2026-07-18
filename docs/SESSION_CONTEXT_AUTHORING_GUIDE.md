# Session Context Authoring Guide

A session context is a single Markdown file whose YAML front matter compiles into the **session plan** — the immutable, declarative control plane for one Lisa runtime. The Markdown body becomes the primary system prompt. This guide covers the v2 strict schema, every front-matter key, validation rules, and authoring patterns.

---

## Design Philosophy

The session plan is the **author-owned source of truth**. It replaces every legacy host-side knob:

- No `applies_to` mode filtering — the body is the primary prompt for every turn.
- No loose `packs` map or `policy` block — pack and server selection is explicit and validated.
- No `context_routing` namespaces — context admission is deterministic, rule-based, and session-owned.
- No `replay.*` controls — conversation history compaction is part of the context plan, not a separate replay subsystem.

The compiler (`kernel.session_plan.compile_session_plan`) is strict: `extra="forbid"` on every model. Unknown keys fail compilation. There is no silent degradation.

---

## Minimal Example

```markdown
---
schema_version: 2
name: lisa
surface:
  type: personal_dm
  id: host-console
  memory_scope: personal
resources:
  packs:
    required: [lisa-pack-core]
providers:
  modes:
    default: lisa-pack-core
  tools:
    web_search: lisa-pack-brave-search
routing:
  start_mode: default
context:
  sources:
    - rule_id: default
      lane_id: operational
      fallback: true
---
# Lisa Session

Use the available tools and supplied context carefully. Distinguish verified
facts from inferences, preserve the user's intent, and explain important
limits or uncertainty directly.
```

The Markdown body (after the closing `---`) is the **primary instruction body**. It must not be empty.

---

## Full Example

```markdown
---
schema_version: 2
name: acme-ops
surface:
  type: team_channel
  id: ops-war-room
  memory_scope: shared

resources:
  packs:
    include: [lisa-pack-core, lisa-pack-search, lisa-pack-learning]
    required: [lisa-pack-core]
    exclude: [lisa-pack-mem0]
  servers:
    exclude: [learning.lisa-memory]

providers:
  modes:
    default: lisa-pack-core
  operations:
    web_search: lisa-pack-search
  tools:
    web_search: lisa-pack-brave-search
  capabilities:
    memory: lisa-pack-learning

routing:
  start_mode: discuss

instructions:
  fragments:
    - ref: acme/ops-guardrails
      modes: [discuss]
      position: before_mode
      required: true

context:
  lanes:
    - lane_id: operational
      min_tokens: 256
      max_tokens: 1200
      render_order: 0
    - lane_id: general
      max_tokens: 1800
      render_order: 10
    - lane_id: history
      max_tokens: 1200
      render_order: 20
  sources:
    - rule_id: alerts
      lane_id: operational
      evidence_types: [alert]
      sources: [cloudflare, github_actions]
      priority: 100
      required: true
    - rule_id: financial
      lane_id: general
      tags: [billing, stripe]
      priority: 50
    - rule_id: fallback
      lane_id: general
      fallback: true
  ranking:
    relevance: 0.40
    mode_affinity: 0.15
    salience: 0.10
    freshness: 0.10
    authority: 0.10
    recency: 0.05
    diversity_lambda: 0.80
    relevance_provider: lexical
  budget:
    output_reserve_tokens: 1024
    safety_margin_tokens: 512
    max_materialized_candidates: 32
  history:
    digest_max_lines: 200
    digest_summary_max_chars: 240
---
Primary goals this week:
- Ship migration with rollback checkpoints.
- Keep postmortems blameless and action-oriented.
```

## Schema Reference

Every key below is defined in [kernel/session_plan.py](../kernel/session_plan.py) as a frozen Pydantic model with `extra="forbid"`. Unknown keys cause a `SessionPlanError` at compilation.

### Top-Level

| Key | Type | Required | Description |
| --- | --- | --- | --- |
| `schema_version` | `2` | Yes | Must be exactly `2`. |
| `name` | `string` | Yes | Session identity; influences thread naming and display. Min length 1. |
| `surface` | object | Yes | Privacy and interaction surface. |
| `resources` | object | No | Deterministic include/exclude/required selectors for packs and MCP servers. |
| `providers` | object | No | Explicit ownership decisions for duplicate catalog resources. |
| `routing` | object | No | Static route seed (start mode). |
| `instructions` | object | No | Explicit instruction fragment declarations. |
| `context` | object | No | The session-owned context mapper policy (lanes, sources, ranking, budget, history). |

### `surface`

| Key | Type | Required | Description |
| --- | --- | --- | --- |
| `type` | `string` | Yes | Surface type (`personal_dm`, `team_channel`, etc.). |
| `id` | `string` | Yes | Surface instance ID. |
| `memory_scope` | `string` | Yes | Memory scope identity for tools and guards. |

### `resources`

Each sub-key is a `ResourceSelection` with `include`, `exclude`, and `required` tuple fields. Include and exclude cannot overlap; required and exclude cannot overlap; no duplicate IDs within any single field.

| Key | Type | Description |
| --- | --- | --- |
| `packs` | `PackSelection` | Pack selection plus pack-owned configuration. Also accepts a `configuration` map. |
| `servers` | `ResourceSelection` | MCP server include/exclude (session-level). |

### `providers`

Explicit ownership decisions for duplicate catalog resources. When multiple active packs contribute the same mode, operation, or tool ID, the session plan must declare which pack owns it. Without an explicit selection, catalog compilation raises a `CatalogCollisionError`.

| Key | Type | Description |
| --- | --- | --- |
| `modes` | `map<str, str>` | `mode_id` to `pack_name` |
| `operations` | `map<str, str>` | `operation_id` to `pack_name` |
| `tools` | `map<str, str>` | `tool_id` to `pack_name` |
| `capabilities` | `map<str, str>` | `capability` to `pack_name` (e.g. `memory: lisa-pack-mem0`) |

All IDs must be non-empty strings.

### `routing`

Session-level routing is currently a single seed: the initial mode the runtime pins at startup. Per-mode routing priority is authored in each mode's `_mode.md` front matter (`priority`), not in the session plan.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `start_mode` | `string` | `default` | Initial mode seed. Does not bypass the router — it sets the router's starting point. |

### `instructions`

The Markdown body is the **primary instruction body** — it is always included. Fragments are additional authored instruction files resolved at plan compilation.

| Key | Type | Description |
| --- | --- | --- |
| `fragments` | `list[InstructionFragment]` | Explicit authored fragments resolved by reference. |

#### `InstructionFragment`

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `ref` | `string` | (required) | Reference to an authored instruction file. |
| `modes` | `tuple<str,...>` | `()` | Modes in which this fragment applies. Empty means all modes. |
| `stages` | `tuple<str,...>` | `()` | Stages in which this fragment applies. Empty means all stages. |
| `position` | `before_mode \| after_mode \| after_stage` | `after_mode` | Where the fragment is rendered relative to the mode instruction. |
| `required` | `bool` | `true` | If `true`, fragment resolution failure is a plan error. |

### `context`

The session-owned context mapper policy. This replaces the legacy `context_routing` block.

#### `context.lanes`

Named lanes that group context candidates for rendering. Each lane has a `render_order` that controls top-to-bottom prompt position.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `lane_id` | `string` | (required) | Unique lane identifier. |
| `min_tokens` | `int >= 0` | `0` | Minimum token budget for this lane. |
| `max_tokens` | `int >= 1 \| null` | `null` | Maximum token budget. `null` means unbounded. |
| `min_items` | `int >= 0` | `0` | Minimum number of candidates. |
| `max_items` | `int >= 1 \| null` | `null` | Maximum number of candidates. |
| `render_order` | `int >= 0` | `0` | Top-to-bottom ordering in the prompt. |

`min_tokens` cannot exceed `max_tokens`; `min_items` cannot exceed `max_items`. Lane IDs must be unique.

The default lanes are `session`, `operational`, `history`, and `retrieval`.

#### `context.sources`

Admission rules that assign context candidates to lanes. Each rule has a `rule_id` (must be unique) and a `lane_id` (must reference a declared lane). A rule needs at least one selector (`source_kinds`, `sources`, `packs`, `evidence_types`, `tags`, `modes`, `stages`) or `fallback: true`. At most one fallback rule is allowed.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `rule_id` | `string` | (required) | Unique rule identifier. |
| `lane_id` | `string` | (required) | Target lane. Must exist in `context.lanes`. |
| `source_kinds` | `tuple<str,...>` | `()` | Match candidate source kind. |
| `sources` | `tuple<str,...>` | `()` | Match candidate source ID. |
| `packs` | `tuple<str,...>` | `()` | Match contributing pack name. |
| `evidence_types` | `tuple<str,...>` | `()` | Match evidence type (e.g. `alert`). |
| `tags` | `tuple<str,...>` | `()` | Match any tag. |
| `modes` | `tuple<str,...>` | `()` | Match active mode. |
| `stages` | `tuple<str,...>` | `()` | Match active stage. |
| `priority` | `int` | `0` | Higher priority wins admission. |
| `required` | `bool` | `false` | Required candidates are always admitted if they match. |
| `pinned` | `bool` | `false` | Pinned candidates bypass MMR diversity filtering. |
| `min_freshness` | `float 0.0-1.0` | `0.0` | Minimum freshness score for admission. |
| `explicit_invocation` | `bool` | `false` | Only admit when explicitly invoked this turn. |
| `fallback` | `bool` | `false` | If `true`, this rule catches all unmatched candidates. |

#### `context.ranking`

Deterministic scoring weights for the context mapper. All weights are `>= 0.0`.

| Key | Default | Description |
| --- | --- | --- |
| `relevance` | `0.35` | Query relevance score. |
| `mode_affinity` | `0.10` | Affinity to the active mode. |
| `salience` | `0.15` | Author-declared salience. |
| `freshness` | `0.10` | Time-based freshness. |
| `authority` | `0.10` | Source authority. |
| `continuity` | `0.05` | Continuity with prior turns. |
| `user_priority` | `0.10` | User-declared priority. |
| `recency` | `0.05` | Recency of last access. |
| `conflict_penalty` | `0.10` | Penalty for conflicting candidates. |
| `token_cost_penalty` | `0.05` | Penalty for token cost. |
| `diversity_lambda` | `0.75` | MMR diversity trade-off (`0.0-1.0`). |
| `relevance_provider` | `lexical` | Relevance scoring backend: `lexical`, `semantic`, or `hybrid`. |

Positive weights are normalized to sum to 1.0 at runtime. At least one positive weight is required.

#### `context.budget`

Model-aware budget limits applied after fixed per-turn reservations.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `output_reserve_tokens` | `int >= 1` | `1024` | Tokens reserved for model output. |
| `safety_margin_tokens` | `int >= 0` | `512` | Safety margin tokens. |
| `max_materialized_candidates` | `int >= 1` | `32` | Maximum candidates materialized into the prompt. |

#### `context.history`

Conversation history compaction constraints. This replaces the legacy `replay.*` controls.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `digest_max_lines` | `int >= 1` | `200` | Max lines retained in the digest during compaction. |
| `digest_summary_max_chars` | `int >= 60` | `240` | Max chars for the lossy summary line during digest rollover. |

---

## Provider Selection

When multiple active packs contribute the same catalog resource ID (a mode, operation, or tool), the session plan **must** declare which pack owns it via the `providers` block. Without this, the catalog MCP server raises a collision error at startup.

```yaml
providers:
  modes:
    default: lisa-pack-core          # core owns the "default" mode
  tools:
    web_search: lisa-pack-brave-search  # Brave owns the web_search tool
  capabilities:
    memory: lisa-pack-mem0            # mem0 owns the memory capability
```

Provider IDs are normalized: `lisa-pack-brave-search`, `brave-search`, and `brave_pack` all match the same pack. Selection is exact — there is no ordered preference list. If the selected pack is not active, the catalog entry is absent (no fallback).

---

## Resolution and Loading

The host resolves a session context source at startup via `hosts.core.session_context.resolve_session_context_source`:

1. **Auto-discovery** (empty `--session-context`):
   - `./session_context.md` in the current working directory.
   - `<data-dir>/session_context.md`.
   - Bundled default: `sdk/packs/core/sessions/default.md`.
2. **Explicit local path**: a literal `.md` file path.
3. **Data-dir slug**: `<data-dir>/<slug>.md`.
4. **Registry slug**: fetched from `LISA_SESSION_CONTEXT_REGISTRY_URL` (default `https://raw.githubusercontent.com/lisa-framework/sessions/main`).
5. **Direct URL**: `https://...` fetched and cached.

Remote fetches are cached in `<data-dir>/session_context_cache/` with conditional GET. Use `--session-context-refresh` to force a re-fetch.

For private contexts, pass auth via `LISA_SESSION_CONTEXT_TOKEN` or `LISA_SESSION_CONTEXT_AUTH_HEADER`. Restrict allowed hosts via `LISA_SESSION_CONTEXT_ALLOWED_HOSTS` (default `github.com, raw.githubusercontent.com`). HTTP is blocked unless `LISA_SESSION_CONTEXT_ALLOW_HTTP=1`.

---

## Validation Rules

The compiler (`compile_session_plan`) enforces:

- `schema_version` must be exactly `2`.
- `name` and all `surface` fields are required and non-empty.
- Front matter must be present (no bare Markdown).
- The instruction body (Markdown after `---`) must not be empty.
- `resources.*.include` and `exclude` cannot overlap.
- `resources.*.required` and `exclude` cannot overlap.
- No duplicate IDs within any single resource selection field.
- `context.lanes` must have unique `lane_id` values.
- `context.sources` must have unique `rule_id` values.
- Every `context.sources[].lane_id` must reference a declared lane.
- At most one `fallback: true` source rule.
- Every source rule needs at least one selector or `fallback: true`.
- `providers` entries require non-empty resource and provider IDs.
- All `context.ranking` weights must be `>= 0.0`; `diversity_lambda` must be `0.0-1.0`.
- `context.lanes` `min_tokens <= max_tokens` and `min_items <= max_items`.

Any violation raises `SessionPlanError` (a `ValueError` subclass). The host renders this as a startup error and exits.

---

## Common Patterns

### Multi-Pack Session with Explicit Ownership

```yaml
providers:
  modes:
    default: lisa-pack-core
  operations:
    web_search: lisa-pack-search
  tools:
    web_search: lisa-pack-brave-search
  capabilities:
    memory: lisa-pack-mem0
```

### Operational Alerts Lane

```yaml
context:
  lanes:
    - lane_id: operational
      max_tokens: 1200
      render_order: 0
  sources:
    - rule_id: alerts
      lane_id: operational
      evidence_types: [alert]
      sources: [cloudflare, github_actions]
      priority: 100
      required: true
```

### Fallback Catch-All

```yaml
context:
  sources:
    - rule_id: catch-all
      lane_id: general
      fallback: true
```

At most one fallback rule is allowed. The fallback rule catches every candidate that no other rule matched.

### Restricting Active Packs

```yaml
resources:
  packs:
    include: [lisa-pack-core, lisa-pack-search]
    required: [lisa-pack-core]
    exclude: [lisa-pack-mem0]
```

Host-mandatory packs (declared in `mcp_servers.yaml` `pack_policy.always_on_packs`) cannot be excluded by the session plan.

---

## Common Pitfalls

- **Missing `schema_version: 2`**: the compiler rejects front matter without it.
- **Empty instruction body**: the Markdown after `---` must not be blank.
- **Duplicate catalog IDs without `providers`**: if two active packs ship the same `tool_id` or `mode_id`, the session plan must declare the owner. A `CatalogCollisionError` at startup means a `providers` entry is missing.
- **Unknown front-matter keys**: `extra="forbid"` means any unrecognized key fails compilation. There is no silent ignore.
- **Overlapping include/exclude**: `include` and `exclude` for the same resource kind cannot share an ID.
- **Source rule referencing undeclared lane**: every `lane_id` in `context.sources` must exist in `context.lanes`.
- **Multiple fallback rules**: at most one `fallback: true` source rule is allowed.
- **Source rule with no selector and no fallback**: a rule needs at least one matching selector or `fallback: true`.

---

## Validation Workflow

1. Compile the plan explicitly to catch validation errors before launching the host:

```powershell
uv run python -c "from kernel.session_plan import compile_session_plan; from pathlib import Path; compile_session_plan(text=Path('session_context.md').read_text(encoding='utf-8'), source_uri='session_context.md'); print('OK')"
```

2. Start the host with the session and verify startup. `lisa verify` compiles the plan during runtime initialization, so a malformed session context will fail the command — but note the displayed "Session context" check row only echoes the resolved slug/path and does not itself re-validate the plan.

```powershell
uv run lisa verify --session-context .\session_context.md --no-log
```

3. Run the session-plan regression tests:

```powershell
uv run pytest kernel/tests/test_session_plan.py -q
```

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

The Markdown body (after the closing `---`) is the **primary instruction body**.
It must not be empty. This is a minimal strict-schema example, not a complete
deployment: for live routing, inherit the core vocabulary as shown in
[configuration](CONFIGURATION.md#session-selection), and supply a model manifest.

---

## Full Example

This illustrates schema fields. The external packs and authored fragment
must exist in the target environment; it is not a self-contained offline
session.

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
    include: [lisa-pack-core, lisa-pack-search, lisa-pack-brave, lisa-pack-memory]
    required: [lisa-pack-core]
    exclude: [lisa-pack-mem0]

providers:
  modes:
    default: lisa-pack-core
  operations:
    web_search: lisa-pack-search
  tools:
    web_search: lisa-pack-brave
  capabilities:
    memory: lisa-pack-memory

routing:
  start_mode: default

instructions:
  fragments:
    - ref: acme/ops-guardrails
      modes: [default]
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
  history:
    recent_message_limit: 6
    # Bound the turn-start replay working set: total tool-result bytes
    # replayed verbatim for the open task before the rest collapse to
    # recoverable ref-stubs (the stubs re-expand via thread_read_content;
    # nothing is lost). Tighten-only against the host ceiling
    # (LISA_CONTEXT_BUDGET_REPLAY_MAX_BYTES, default 0 = no separate cap).
    # Omit this knob to let replay and live results fill the request's
    # remaining model window after input framing and output reservations.
    replay_max_bytes: 32768
    # Thread digest pushed into the retrieval lane each turn (all defaults shown)
    digest:
      enabled: true            # false → tool pulls only, nothing pushed
      recent_entries: 10       # exchanges summarized in the recent-history item
      recent_body_chars: 1800  # char budget for that summary
      task_view_budget_bytes: 4096
      max_open_objective_lines: 3
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
---
Primary goals this week:
- Ship migration with rollback checkpoints.
- Keep postmortems blameless and action-oriented.
```

## Schema Reference

Every key below is defined in [kernel/session_plan.py](../kernel/src/kernel/session_plan.py) as a frozen Pydantic model with `extra="forbid"`. Unknown keys cause a `SessionPlanError` at compilation.

### Top-Level

| Key | Type | Required | Description |
| --- | --- | --- | --- |
| `schema_version` | `2` | Yes | Must be exactly `2`. |
| `name` | `string` | Yes | Session identity; influences thread naming and display. Min length 1. |
| `surface` | object | Yes | Privacy and interaction surface. |
| `resources` | object | No | Deterministic include/exclude/required selectors for packs and MCP servers. |
| `providers` | object | No | Explicit ownership decisions for duplicate catalog resources. |
| `routing` | object | No | Static route seed (start mode). |
| `execution` | object | No | Tighten-only requests for turn/session resource limits. |
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
| `packs` | `PackSelection` | Pack selection plus pack-owned configuration. `required` drives auto-install; `include`/`exclude` filter catalog activation. Also accepts a `configuration` map and an `auto_install` map. |
| `servers` | `ResourceSelection` | MCP server include/exclude (session-level). |

#### `resources.packs.auto_install`

Customizes the install source/method for packs listed in `required`. The host auto-installs **every missing `required` pack** at boot; `auto_install` overrides the default git-root source for the named pack. Every `auto_install` key **must** also appear in `required` — the compiler rejects keys not in `required` with `SessionPlanError`.

| Key | Type | Required | Description |
| --- | --- | --- | --- |
| `source` | `string` | Yes | Bare pack name (expanded via `LISA_PACK_GIT_ROOT`), `git+https://...@ref`, or a local path. |
| `pin` | `string` | No | Explicit pin label (e.g. `v0.2.0`) for logging and provenance. |
| `editable` | `bool` | No | When `source` is a local directory, install with `uv pip install -e` so edits to the pack source are picked up without reinstall. Defaults to `True`. Ignored for git/PyPI sources. |

Git refs **must** pin `@tag/@branch/@sha`. The host treats pinning as mandatory for auto-install regardless of `LISA_PACK_REQUIRE_GIT_REF`; an unpinned `git+` source raises a startup error. Trust policy env vars (`LISA_PACK_ALLOWED_PREFIXES`, `LISA_PACK_GIT_ALLOW_HOSTS`, `LISA_PACK_GIT_ROOT`) are honored — a session context cannot override them.

Local paths are resolved relative to the session-context file's directory, so a session can reference packs checked out alongside it (e.g. `source: ../lisa-packs/brave`). Absolute paths are passed through unchanged.

Auto-install applies **only** to `required` packs. Failure of any required pack is a hard startup error; the runtime does not start in a degraded state. When at least one pack is installed, the host requests a restart so `importlib.metadata` entry-point discovery sees the new packs on the next boot pass.

```yaml
resources:
  packs:
    required: [lisa-pack-core, lisa-pack-cloudflare, lisa-pack-brave]
    auto_install:
      lisa-pack-cloudflare:
        source: git+https://github.com/Lisa-Harness/lisa-pack-cloudflare.git@v0.2.0
        pin: v0.2.0
      lisa-pack-brave:
        source: ../lisa-packs/brave
        editable: true
```

#### `required` vs `include` vs `auto_install`

These three fields have distinct responsibilities; they are not interchangeable:

| Field | Controls installation? | Controls catalog activation? | Default source when missing? |
| --- | --- | --- | --- |
| `required` | **Yes** — every missing `required` pack is auto-installed at boot. | No (but required packs that become importable are activated by default). | Git-root fallback (`git+https://...@main` via `LISA_PACK_GIT_ROOT`). |
| `include` | No. | **Yes** — restricts catalog activation to the listed packs (no include = all installed/enabled packs active). | N/A. |
| `auto_install` | No — does not decide *whether* a pack is installed, only *how*. | No. | Overrides the git-root fallback for the named pack with an explicit source/editable flag. |

The host installs **every missing `required` pack** regardless of whether it has an `auto_install` entry. `auto_install` only customizes the source and method (local path vs git ref, editable vs not). A `required` pack with no `auto_install` key falls back to the git-root template (`git+https://github.com/Lisa-Harness/lisa-pack-<name>@main`), which may fail if the pack is private or depends on unpublished packages.

#### Suppressing auto-install (operator-managed installs)

To stop the host from installing a pack, **remove it from `required`**. A pack listed only in `include` (or in neither field) is never auto-installed — it must already be importable or it will not appear in the catalog.

Use `include` (not `required`) when you manage the packs venv yourself (`uv pip install -e <pack dir>` into the packs venv) and want the host to activate the pack without trying to install it:

```yaml
# Operator manages the packs venv; host activates but does not install.
resources:
  packs:
    required: [lisa-pack-core]           # only the always-needed bundled pack
    include: [lisa-pack-cloudflare, lisa-pack-brave]  # operator-installed
```

If a pack is in `include` but not installed, it is silently absent from the catalog — the host does not raise a startup error. This is the intended behavior: `include` is a *catalog filter*, not an *install trigger*.

#### Where external packs must live

Use the host-managed packs venv for external installs; use `packstest` from
source for development. Catalog/content subprocesses activate that venv's
site-packages. The runtime deduplicates installed distributions by normalized
name, but conflicting roots/resource owners still need investigation.
Avoid installing multiple copies/revisions into different environments.

The framework root currently has a `dev` extra, not `ext` or `wip`.
Do not copy obsolete `uv sync --extra ext --extra wip` commands. Inspect
the actual environment and `/pack` state before uninstalling anything.

### `providers`

Explicit ownership decisions for duplicate catalog resources. When multiple active packs contribute the same mode, operation, or tool ID, the session plan must declare which pack owns it. Without an explicit selection, catalog compilation raises a `CatalogCollisionError`.

| Key | Type | Description |
| --- | --- | --- |
| `modes` | `map<str, str>` | `mode_id` to `pack_name` |
| `operations` | `map<str, str>` | `operation_id` to `pack_name` |
| `tools` | `map<str, str>` | `tool_id` to `pack_name` |
| `capabilities` | `map<str, str>` | `capability` to `pack_name` (e.g. `memory: lisa-pack-mem0`) |

All IDs must be non-empty strings.

### `execution`

Requests tighten the host's resolved envelope; they do not grant capabilities
or bypass approval. `null`/absence means no request.

| Key | Unit | Meaning |
| --- | --- | --- |
| `max_model_calls` | calls/turn | Model-call bound |
| `max_capability_calls` | calls/turn | Tool/operation-call bound |
| `max_total_tokens` | processed tokens/turn | Uncached input plus output; cache reads excluded |
| `max_completion_tokens` | completion tokens/turn | Cumulative output rail |
| `max_wall_ms` | milliseconds/turn | Elapsed-time bound |
| `max_session_tokens` | processed tokens/session | Pre-turn session admission bound, not a daily reset |
| `force_final_on_exhausted` | bool | Requested exhaustion behavior within lifecycle policy |
| `capabilities` | map | Per-capability `max_invocations_per_turn` / `max_internal_tool_calls` |
| `scopes` | map | Named scoped limit requests, applied when the host resolves those scopes |

Numeric declarations must be positive. See [token budgets](TOKEN_BUDGET.md)
for per-call model/mode ceilings, retries, and the distinction between
resource accounting and monetary cost.

### `routing`

Session-level routing seed plus the author-curated vocabulary the kernel's routing-hint generator reads. There is no pre-turn router — natural-language turns go directly to `current_mode_id` and hand off mid-turn via `transition_stage`; `start_mode` is a bypass pin, not a router seed. When `current_mode_id == start_mode != "default"`, the turn goes directly to the pinned mode; `start_mode: default` means "let the model decide via handoff." See [mode routing](ROUTER_DESIGN.md#turn-time-dispatch).

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `start_mode` | `string` | `default` | Bypass pin: the mode a turn goes directly to when it equals `current_mode_id`. `default` is exempt (routing hub, hands off mid-turn). |
| `aliases` | `map<str, str>` | `{}` | Slash-command aliases (`go: discuss` → `/go` routes like `/discuss`). |
| `priority_adjustments` | `map<str, int>` | `{}` | Override per-mode catalog ordering priority from the session. |
| `hints` | `map<str, tuple<str,...>>` | `{}` | Per-mode hint strings injected by the routing-hint fast path. |
| `sequential_connectors` | `tuple<str,...>` | `()` | Whitespace-padded substrings marking a query as an ordered chain (`then`, `based on`, `once`). A match routes sequential handoffs instead of a parallel `fork:`. Session-authored vocabulary — opaque to the kernel. |
| `resume_phrases` | `tuple<str,...>` | `()` | Lowercased imperatives marking a short message as a workflow/approval resume rather than a fresh query (`proceed`, `continue`, `go ahead`). Applied as prefix matches. |
| `producer_intents` | `tuple<str,...>` | `()` | Intent tags whose satisfaction produces evidence a consumer needs (`research`, `review`, `analyze`). The fork guard refuses a parallel fork when a gap mixes a producer with a consumer. |
| `consumer_intents` | `tuple<str,...>` | `()` | Intent tags whose satisfaction consumes a producer's output (`code`, `write`, `implement`). Counterpart to `producer_intents`. |
| `topic_abandonment_markers` | `tuple<str,...>` | `()` | Lowercased substrings signalling the user is *leaving* a topic (`side quest`, `never mind`), so a keyword mention does not route *into* that topic. |

```yaml
routing:
  start_mode: default
  sequential_connectors: [then, "and then", "followed by", "based on", once]
  resume_phrases: [proceed, continue, "go ahead", "keep going"]
  producer_intents: [research, gather, learn, read, analyze, review, architect]
  consumer_intents: [code, write, implement, document]
  topic_abandonment_markers: ["side quest", rewind, "never mind", forget]
```

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

The session-owned context mapper policy. This is the sole control surface for how evidence from intake, packs, and history is admitted, ranked, budgeted, and rendered into the prompt each turn.

**Wiring.** The `context:` block compiles to `SessionPlan.manifest.context` (a `ContextPlan` in `kernel/session_plan.py`). At turn time, `runtime/turn_hooks_builder.py` reads `runtime.session_plan.manifest.context` and passes it through `assemble_dynamic_user_context` → `build_session_mapped_context_markdown` → `ContextMapper.map(...)`, all in-process. There is no second, parallel routing system — the legacy namespace-based `context_routing` block and the `broker_context` MCP tool were removed; the `ContextPlan` is the only engine.

**Data flow.** `build_unified_state` (in `runtime/state_provider.py`) collects `ContextItem` records from intake and the extension-item hook into `UnifiedState.context_items` — a single flat feed. The mapper walks that feed against your `context.sources` rules, gates and ranks matches with your `context.ranking` weights, fits them to the model-aware window via `context.budget`, and renders the survivors into lane-ordered Markdown. The state carries no routing policy of its own; all policy lives here.

#### `context.models`

A reference to the model manifest file this session should load instead of
the default `<data-dir>/config/models.yaml`.

| Key | Type | Required | Description |
| --- | --- | --- | --- |
| `path` | `string` | Yes | Session-relative or absolute path to a `.yaml`/`.yml`/`.json` model manifest. |

The path is resolved relative to the session-context file's directory (for session-relative paths) or as an absolute path. The host validates existence and extension at boot; a missing file or wrong type raises a startup error. The compiler validates non-emptiness only — it has no filesystem access by design.

Remote URLs are not supported; manifest loading is local-only. The resolved path is logged at startup for operator audit.

```yaml
context:
  models:
    path: ./roster.yaml
```

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
| `required` | `bool` | `false` | Select first; raise a budget error if matched content cannot fit. |
| `pinned` | `bool` | `false` | Pinned candidates bypass MMR diversity filtering. |
| `min_freshness` | `float 0.0-1.0` | `0.0` | Minimum freshness score for admission. |
| `explicit_invocation` | `bool` | `false` | Only admit when explicitly invoked this turn. |
| `fallback` | `bool` | `false` | If `true`, this rule catches all unmatched candidates. |

All populated selector fields must match; listed alternatives within one
field use OR. Mode/stage, freshness, and explicit-invocation conditions
also apply. A source-matched candidate failing a condition is dropped,
not rescued by fallback. See [context routing](CONTEXT_ROUTING_GUIDE.md).

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

Conversation history settings for context assembly.

| Key | Default | Meaning |
| --- | --- | --- |
| `include_recent_conversation` | `true` | Include recent conversation evidence |
| `include_reflection` | `false` | Include the diagnostic reflection |
| `recent_message_limit` | `6` | Recent-message selection limit; non-negative |
| `replay_max_bytes` | absent | Optional 1–1048576-byte replay ceiling, tighten-only against a finite host ceiling |
| `digest` | object | Bounded task-state/recent-history/open-objective projection |

`digest` fields are `enabled` (true), `recent_entries` (10),
`recent_body_chars` (1800), `task_view_budget_bytes` (4096), and
`max_open_objective_lines` (3). This deterministic prompt projection is
not the removed lossy digest sidecar or in-place thread compaction.
The underlying thread remains lossless.

#### `context.memory`

Memory-store consolidation policy is pressure-driven and paid from the
configured slice slot/budget. It can make auxiliary model calls and add
latency; it does not compact the lossless thread in place.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `consolidation` | `MemoryConsolidationTuning` | (see below) | Episodic-pressure-triggered memory consolidation. |

##### `context.memory.consolidation`

When `enabled` is `true` (the default), a host that wires the memory-consolidation slice provider counts active episodic records each turn and fires the configured `consolidate_operation` (default: `consolidate_semantic`) when episodic pressure crosses a threshold. Set `enabled` to `false` to let episodic records accumulate until TTL expiry.

Thresholds are fractions of `max_records` (`0.0`-`1.0`). The slice pressure curve decides whether a between-thresholds turn actually runs the consolidation; `hard_threshold` overrides the curve when episodic pressure is critical.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `enabled` | `bool` | `true` | Master switch. Set `false` to disable automatic consolidation. |
| `trigger` | `post_turn` \| `pre_turn` \| `explicit` \| `never` | `post_turn` | Which slice slot fires consolidation. |
| `pressure_threshold` | `float 0.0-1.0` | `0.6` | Fire when `episodic_count / max_records` reaches this fraction. |
| `hard_threshold` | `float 0.0-1.0` | `0.85` | Guardrail: fire regardless of pressure budget or interval. Must be `>= pressure_threshold`. |
| `min_interval_turns` | `int >= 0` | `8` | Don't re-consolidate more often than every N turns (ignored when `hard_threshold` overrides). |
| `batch_size` | `int >= 1` | `10` | Records consolidated per invocation. |
| `min_episodic_count` | `int >= 1` | `5` | Don't fire below this many active episodic records. |
| `max_records` | `int >= 1` | `200` | Denominator for the pressure fraction. |
| `preferred_model_id` | `string` | `default` | Model ID used for the consolidate operation. |
| `consolidate_operation` | `string` | `consolidate_semantic` | Operation ID invoked to consolidate. |
| `mine_on_segment_seal` | `bool` | `true` | Mine newly sealed thread segments when consolidation runs. |

Example:

```yaml
context:
  memory:
    consolidation:
      enabled: true
      pressure_threshold: 0.6
      hard_threshold: 0.85
      min_interval_turns: 8
      batch_size: 10
      max_records: 200
```

#### `context.conflict_policy`

How the mapper resolves candidates that conflict on the same logical claim. The mapper computes a conflict risk for each candidate; this setting decides what happens when a high-authority and low-authority candidate disagree.

| Value | Behavior |
| --- | --- |
| `prefer_authority` (default) | Keep the higher-authority candidate; drop the lower. |
| `surface_conflict` | Keep both and annotate the conflict so the model can reason about it. |
| `drop_lower_authority` | Drop the lower-authority candidate without surfacing. |

```yaml
context:
  conflict_policy: surface_conflict
```

---

## Provider Selection

When multiple active packs contribute the same catalog resource ID (a mode, operation, or tool), the session plan **must** declare which pack owns it via the `providers` block. Without this, the catalog MCP server raises a collision error at startup.

```yaml
providers:
  modes:
    default: lisa-pack-core          # core owns the "default" mode
  tools:
    web_search: lisa-pack-brave  # Brave owns the web_search tool
  capabilities:
    memory: lisa-pack-mem0            # mem0 owns the memory capability
```

Provider IDs are normalized: `lisa-pack-brave`, `brave`, and `brave_pack` all match the same pack. Selection is exact — there is no ordered preference list. If the selected pack is not active, the catalog entry is absent (no fallback).

---

## Resolution and Loading

The host resolves a session context source at startup via `runtime.session_context.resolve_session_context_source` — from local disk only:

1. **Explicit path** (`--session-context <file>.md`): must exist and be a Markdown file; a missing file is an error, not a fallthrough.
2. **`LISA_SESSION_CONTEXT_PATH` env var**: an explicit `.md` path. When set, the file must exist — a configured-but-missing path raises instead of silently falling back.
3. **`<data-dir>/config/session_context.md`** when authored.
4. **Bundled default**: `sdk/packs/core/sessions/default.md`.

There is no remote fetch, registry, or slug resolution: the session plan is a local authored artifact, read from disk on every startup.

---

## Inheritance via `from:` and `body:`

A user session can pull whole blocks from a pack's bundled session file (`<pack_root>/sessions/<name>.md`) instead of redeclaring them. Two host-layer directives do this, resolved **before** the strict kernel compiler runs:

### `from:` (block inheritance)

Usable under any of `resources`, `providers`, `routing`, `context`, or `execution`. The block is the **unit of inheritance** — there is no per-field merge. The named block is pulled wholesale from the referenced pack session.

```yaml
routing:
  from:
    pack: lisa-pack-core
    session: default
```

Rules:

- **Mutually exclusive.** A block that uses `from:` may not declare any other key. `routing: { from: {...}, start_mode: x }` is a startup error.
- **Missing block errors.** If the referenced pack session omits the block, resolution fails with a `SessionPlanError` naming the pack, session, and block.
- **Identity stays local.** `schema_version`, `name`, `surface` are never pulled — there is no top-level `from:`. Every session declares its own identity.
- **Packs are leaves.** Pack session files cannot themselves use `from:` or `body:`. The strict v2 schema (`extra="forbid"`) rejects them, so a pack that tries to inherit fails at load.
- **Pack aliasing.** `pack` accepts the full ID (`lisa-pack-core`) or the bare slug (`core`). The pack must be installed (discoverable via content roots); it need not be in `resources.packs.required`.

Cherry-pick example — pull three blocks, author `context` locally:

```yaml
resources:
  from: { pack: lisa-pack-core, session: default }
providers:
  from: { pack: lisa-pack-core, session: default }
routing:
  from: { pack: lisa-pack-core, session: default }
context:
  sources:
    - rule_id: mine
      lane_id: operational
      fallback: true
```

Here `context` is authored locally and wins wholesale — no merge, no override semantics.

### `body:` (instruction body inheritance)

A top-level directive that controls the Markdown body. Three modes:

```yaml
body:
  from: { pack: lisa-pack-core, session: default }
  mode: pull          # pull | write | context (default: write)
```

| Mode | Behavior |
| --- | --- |
| `write` | Default when `body:` is omitted. The user's body is primary; the non-empty-body rule applies to it. |
| `pull` | The user's body is ignored (may be empty); the pack session's body becomes primary. |
| `context` | The pack session's body is prepended, wrapped in `<pack_session_body>` … `</pack_session_body>`, followed by the user's body. Both must be non-empty. |

`body:` is stripped from the front matter before the kernel schema sees it, so it does not violate `extra="forbid"`.

### Resolution order

`from:`/`body:` are resolved by `runtime.session_from_resolver.resolve_session_from_directives` after source resolution and before `compile_session_plan`. Pack discovery runs lazily — only when a directive is actually present — so sessions without directives pay zero discovery cost. When no directive is present, the resolver is a passthrough and existing sessions behave identically.

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
    web_search: lisa-pack-brave
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
  lanes:
    - lane_id: general
  sources:
    - rule_id: catch-all
      lane_id: general
      fallback: true
```

At most one fallback rule is allowed. The fallback catches candidates with
no matching non-fallback source selectors; it does not bypass failed
conditions on a source-matched rule.

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
- **`context_routing` is gone**: the legacy namespace-based routing block was removed. There is no `context_routing` key and no `ContextRoutingConfig`; use the `context` block (`lanes` + `sources`) instead. A `ContextRoutingConfig` reference in old code or docs is stale.
- **`ContextItem.namespace` is a label, not a selector**: the mapper selects on `source`, `pack`, `source_type`, `evidence_type`, and `tags` — never on `namespace`. Setting `namespace="alerts"` does not route an item to an `alerts` lane; author a `context.sources` rule matching `evidence_types: [alert]` or `sources: [...]` instead.
- **External packs installed in the main `.venv`**: an external pack present in both the main framework `.venv` and the packs venv produces a `Duplicate operation ID`/`Duplicate tool ID` collision at boot because the catalog subprocess sees two content-root paths for one pack. Keep external packs in the packs venv only; uninstall them from the main `.venv`. (Bundled packs are fine in the main `.venv` — they are not installed into the packs venv.)

---

## Validation Workflow

1. Compile the plan explicitly to catch validation errors before launching the host:

```powershell
uv run python -c "from kernel.session_plan import compile_session_plan; from pathlib import Path; compile_session_plan(text=Path('session_context.md').read_text(encoding='utf-8'), source_uri='session_context.md'); print('OK')"
```

For files using host-layer `from:`/`body:` directives, resolve those directives
before calling the kernel compiler; raw compilation deliberately rejects them.
See [the resolver](../runtime/src/runtime/session_from_resolver.py).

2. Start a compatible external host with the session and verify startup.
The framework workspace does not install `lisa`; consult the
[CLI reference](https://github.com/Lisa-Harness/lisa-cli) for its current
`verify` options. A displayed session path alone is not validation.

3. Run the session-plan regression tests:

```powershell
uv run pytest kernel/tests/test_session_plan.py -q
```

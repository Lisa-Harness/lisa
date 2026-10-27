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
    include: [lisa-pack-core, lisa-pack-brave, lisa-pack-memory]
    required: [lisa-pack-core]
    exclude: [lisa-pack-mem0]

providers:
  modes:
    default: lisa-pack-core
  tools:
    # Both lisa-pack-core (the contract stub) and lisa-pack-brave ship
    # tool_id `web_search`. This line resolves that collision.
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
  budget:
    output_reserve_tokens: 1024
    safety_margin_tokens: 512
    max_materialized_candidates: 32
---
Primary goals this week:
- Ship migration with rollback checkpoints.
- Keep postmortems blameless and action-oriented.
```

## Worked Example: Memory, Markdown, Personas, and Tavily

A complete, deployable session that wires four packs together: **`lisa-pack-memory`**
(long-term recall and learning), **`lisa-pack-markdown`** (document forging and
consolidation), **`lisa-pack-personas`** (the `thinker`/`researcher` modes and the
`discussion_briefing` pre-turn operation), and **`lisa-pack-tavily`** (the concrete
`web_search` *tool* provider). `lisa-pack-core` supplies the default mode, the
`fetch_page` tool, and the `web_search` *contract*.

You need a provider pack like `lisa-pack-tavily` (or `lisa-pack-serper`) for live
results, because research modes use `tool_id: web_search`. `lisa-pack-core` owns
the contract and ships a stub that just tells you to install one, like
`lisa-pack-tavily` which ships a real `web_search`.
However,
if you load both packs,
the kernel bootstrap raises a `CatalogCollisionError`.
To avoid that,
declare the which pack owns `web_search` in `providers.tools`.

> **The Tavily API key is required.** Without it `web_search` fails on every
> call with `Tavily search is not configured. Set TAVILY_SEARCH_API_KEY to
> enable it.` Export it in the host's environment before starting the session:
>
> ```bash
> export TAVILY_SEARCH_API_KEY="tvly-..."
> ```
>
> The other Tavily knobs — `TAVILY_SEARCH_ENDPOINT`, `TAVILY_SEARCH_MAX_RESULTS`,
> `TAVILY_SEARCH_SEARCH_DEPTH`, `TAVILY_SEARCH_TOPIC` — are optional and
> default to `https://api.tavily.com/search`, `5`, `basic`, and `general`.
> The key is read from the environment only; there is no session-front-matter
> slot for it, because a secret must never live in a versioned session file.

```markdown
---
schema_version: 2
name: research-desk
surface:
  type: personal_dm
  id: host-console
  memory_scope: personal

resources:
  packs:
    # `required` drives auto-install at boot; every missing required pack
    # is installed from the git-root template unless auto_install overrides it.
    #
    # lisa-pack-core is bundled with the framework (sdk/packs/core) and is
    # always-on via the host's pack_policy, so it needs no install source.
    # lisa-pack-memory is also bundled (sdk/packs/memory). Only the three
    # external packs below are auto-installed.
    required:
      - lisa-pack-core
      - lisa-pack-memory
      - lisa-pack-markdown
      - lisa-pack-personas
      - lisa-pack-tavily
    auto_install:
      lisa-pack-tavily:
        source: git+https://github.com/Lisa-Harness/lisa-pack-tavily.git@v0.1.0
        pin: v0.1.0
      lisa-pack-personas:
        source: git+https://github.com/Lisa-Harness/lisa-pack-personas.git@v0.1.0
        pin: v0.1.0
      lisa-pack-markdown:
        source: git+https://github.com/Lisa-Harness/lisa-pack-markdown.git@v0.1.0
        pin: v0.1.0
    configuration:
      lisa-pack-memory:
        # Real configuration key for the memory pack. The pack reads its own
        # block as `ctx.pack_config`; the framework does not interpret it.
        # This raises the distillation LLM token budget, which matters for
        # local OpenAI-compatible backends (Ollama) that truncate the JSON
        # envelope when the budget is too small.
        memory_distill_max_tokens: 16384

providers:
  modes:
    default: lisa-pack-core
  operations:
    # `discussion_briefing` is owned by the personas pack, alongside `thinker`
    # — its only consumer.
    discussion_briefing: lisa-pack-personas
    # Document forging and multi-source consolidation are the markdown pack's.
    markdown_forge_document: lisa-pack-markdown
    markdown_consolidate_sources: lisa-pack-markdown
    markdown_render_turn: lisa-pack-markdown
    # Recall and learning are the memory pack's.
    recall_memory: lisa-pack-memory
    synthesize_and_store_learning: lisa-pack-memory
  tools:
    # The core contract stub and lisa-pack-tavily both declare tool_id
    # `web_search`. This line is what resolves the collision.
    web_search: lisa-pack-tavily
    markdown_generate: lisa-pack-markdown
  capabilities:
    # The bundled pack provides the required memory.add/search/list contract.
    # Select an alternate provider explicitly when one is active.
    memory: lisa-pack-memory

routing:
  start_mode: default
  sequential_connectors: [then, "and then", "based on", once]
  resume_phrases: [proceed, continue, "go ahead", "keep going"]

context:
  lanes:
    - lane_id: operational
      max_tokens: 1200
      render_order: 0
    - lane_id: retrieval
      max_tokens: 2400
      render_order: 10
    - lane_id: history
      max_tokens: 1200
      render_order: 20
  sources:
    # The thread digest (task_state / recent_history / open_objectives) and
    # epistemics land in the retrieval lane. These are the *defaults*; they
    # are re-declared here because authoring `context.sources` replaces the
    # default rule set wholesale.
    - rule_id: thread_digest
      lane_id: retrieval
      evidence_types: [task_state, recent_history, open_objectives]
      priority: 100
    - rule_id: thread_digest_epistemics
      lane_id: retrieval
      evidence_types: [epistemics]
      priority: 100
      pinned: true
    # Memory recall results and Tavily search results are retrieval-grade
    # evidence: they are grounded, citable, and worth outranking the
    # catch-all.
    - rule_id: memory_recall
      lane_id: retrieval
      packs: [lisa-pack-memory]
      priority: 80
    - rule_id: web_search_results
      lane_id: retrieval
      packs: [lisa-pack-tavily, lisa-pack-personas]
      priority: 70
    # Everything else (live conversation excerpts, ambient items) goes to
    # the operational lane. Exactly one fallback rule is allowed.
    - rule_id: catch_all
      lane_id: operational
      fallback: true
  history:
    recent_message_limit: 8
    digest:
      enabled: true
      recent_entries: 10
      recent_body_chars: 1800
      task_view_budget_bytes: 4096
      max_open_objective_lines: 3
  memory:
    consolidation:
      enabled: true
      pressure_threshold: 0.6
      hard_threshold: 0.85
      min_interval_turns: 8
      batch_size: 10
      max_records: 200
  ranking:
    relevance: 0.40
    mode_affinity: 0.15
    salience: 0.10
    freshness: 0.10
    authority: 0.10
    recency: 0.05
    diversity_lambda: 0.80
  budget:
    output_reserve_tokens: 1024
    safety_margin_tokens: 512
    max_materialized_candidates: 32
---
You are a research desk with a memory.

Ground every claim in recalled memory or retrieved sources. When a claim
comes from web research, name the source. When it comes from memory, say
so. Prefer a documented, consolidated Markdown artifact over a chat
answer when the user asks for a deliverable.
```

### Why each block is there

| Block | Reason |
| --- | --- |
| `resources.packs.required` | All five packs are auto-installed at boot. A missing required pack is a **hard startup error** — the runtime does not start degraded. |
| `resources.packs.auto_install` | Pins each external pack to a tagged ref. Git refs **must** pin `@tag/@branch/@sha`; the host treats pinning as mandatory for auto-install regardless of `LISA_PACK_REQUIRE_GIT_REF`. |
| `providers.tools.web_search` | **Required.** `lisa-pack-core` (the contract stub) and `lisa-pack-tavily` both ship `tool_id: web_search`. Without this line the catalog raises `CatalogCollisionError` at startup. |
| `providers.operations.*` | Declares the owner for each research and document operation. Harmless when uncontested, but it makes ownership explicit and survives a future pack that ships a same-ID operation. |
| `providers.capabilities.memory` | Selects the owner of the required `memory.add`, `memory.search`, and `memory.list` contract. Provider-specific optional tools may vary. |
| `context.sources` | Authoring it **replaces** the default rule set, so the thread-digest rules are re-declared explicitly. The `catch_all` fallback preserves the operational lane. |
| `context.memory.consolidation` | Wired through the memory pack's slice provider: when active episodic records cross `pressure_threshold` of `max_records`, the slice fires `consolidate_semantic` outside the turn's latency budget. Set `enabled: false` to let records accumulate until TTL expiry. |

### Adding Tavily to a session that already has a search provider

If the session already selects Brave or Serper for `web_search`, adding
Tavily means **replacing** the owner, not adding a second one — the
`web_search` tool ID is singular:

```yaml
providers:
  tools:
    web_search: lisa-pack-tavily   # was: lisa-pack-brave
```

Both packs stay installed and their *operations* remain reachable, but only
one pack can own the `web_search` tool ID at a time.

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
| `servers` | `ResourceSelection` | MCP server include/exclude (session-level). Only `exclude` is read at runtime; `include`/`required` are accepted but inert. |
| `proxies` | `ResourceSelection` | Proxy server include/exclude/required. Unlike `servers`, all three fields are honored: `include` (when non-empty) is an allowlist, `exclude` drops a proxy, and `required` marks a proxy mandatory. |
| `modes` | `ResourceSelection` | Mode include/exclude/required. Accepted by the schema but **not read by the runtime** — mode activation is driven by pack selection and `providers.modes`, not by this selector. |
| `operations` | `ResourceSelection` | Operation include/exclude/required. Accepted by the schema but **not read by the runtime**. |
| `tools` | `ResourceSelection` | Tool include/exclude/required. Accepted by the schema but **not read by the runtime**. |
| `signals` | `ResourceSelection` | Signal include/exclude/required. Accepted by the schema but **not read by the runtime**. |
| `intake` | `ResourceSelection` | Intake include/exclude/required. Accepted by the schema but **not read by the runtime**. |
| `models` | `ResourceSelection` | Model include/exclude/required. Accepted by the schema but **not read by the runtime**; use `context.models.path` to point at a model manifest. |

Only `packs`, `servers.exclude`, and `proxies` currently affect runtime behavior. The remaining selectors are reserved for future use: they compile and validate, but authoring them has no effect today.

#### `resources.packs.configuration`

A two-level map: **pack ID → configuration object**.

```yaml
resources:
  packs:
    configuration:
      lisa-pack-bwrap:
        sandbox:
          floor:
            execution: sandbox
            network: deny
```

Each pack reads its own key; the framework does not interpret the contents. Keys naming a pack that is not active are ignored. A pack may only *tighten* the framework-authored floor — it cannot loosen it.

#### `resources.packs.auto_install`

Customizes the install source/method for packs listed in `required`. The host auto-installs **every missing `required` pack** at boot; `auto_install` overrides the default git-root source for the named pack. Every `auto_install` key **must** also appear in `required` — the compiler rejects keys not in `required` with `SessionPlanError`.

| Key | Type | Required | Description |
| --- | --- | --- | --- |
| `source` | `string` | Yes | Bare pack name (expanded via `LISA_PACK_GIT_ROOT`), `git+https://...@ref`, or a local path. |
| `pin` | `string` | No | Explicit pin label (e.g. `v0.2.0`) for logging and provenance. |
| `editable` | `bool` | No | When `source` is a local directory, install with `uv pip install -e` so edits to the pack source are picked up without reinstall. Defaults to `True`. Ignored for git/PyPI sources. |

Git refs **must** pin `@tag/@branch/@sha`. The host treats pinning as mandatory for auto-install regardless of `LISA_PACK_REQUIRE_GIT_REF`; an unpinned `git+` source raises a startup error. Trust policy env vars (`LISA_PACK_ALLOWED_PREFIXES`, `LISA_PACK_GIT_ALLOW_HOSTS`, `LISA_PACK_GIT_ROOT`) are honored — a session context cannot override them.

Local paths are resolved relative to the session-context file's directory, so a session can reference packs checked out alongside it (e.g. `source: ../lisa-packs/brave`). Absolute paths are passed through unchanged.

For `git+` sources, the host appends `egg=<pack name>` to the URL fragment unless one is already present, so the `auto_install` key names the package. This lets a pack live in a subdirectory of a differently named repository (e.g. `git+https://github.com/Lisa-Harness/lisa-go.git@v0.2.0#subdirectory=pack`). For `/pack install`, add `&egg=<pack name>` yourself. Packs that ship binary MCP servers without a `binary_download.version` download the release matching the installed pack version, so the git tag in `source` selects both the pack and its binaries.

Auto-install applies **only** to `required` packs. Failure of any required pack is a hard startup error; the runtime does not start in a degraded state. When at least one pack is installed, the host flags a pending reload so `importlib.metadata` entry-point discovery sees the new packs on the next boot pass.

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
| `capabilities` | `map<str, str>` | `capability` to provider pack or server ID (e.g. `memory: lisa-pack-mem0`) |

All IDs must be non-empty strings.

For MCP capabilities, `providers.capabilities.<name>` selects one server
implementation. Without a selection, a built-in host implementation wins;
when there is no built-in, a sole candidate is selected and competing
candidates require an explicit choice. The selected server publishes the
capability's stable `<name>.<tool>` IDs, while provider-specific argument
and result adaptation stays in its pack-owned MCP server.

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

The default lanes are `session`, `operational`, `history`, and `retrieval` (in that `render_order`).

`max_tokens` and `max_items` are **hard limits** — a candidate that would exceed either is dropped with reason `lane_token_budget_exhausted` or `lane_item_limit_reached`. `min_tokens` and `min_items` are **best-effort coverage targets**: they tell the mapper a lane is still under-covered and should keep admitting candidates, but they are not guarantees. If no eligible candidates remain, the minimum simply goes unmet.

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

##### How a candidate is routed

1. **Selectors** (`source_kinds`, `sources`, `packs`, `evidence_types`, `tags`) narrow the rule set. Every populated selector must match; alternatives inside one field are OR.
2. **Conditions** (`modes`, `stages`, `explicit_invocation`, `min_freshness`) are evaluated next. A candidate whose selectors matched but whose conditions fail is **dropped** — the fallback rule does not rescue it.
3. **Priority** picks the lane. Among the matching rules, the highest `priority` wins. If two rules at that priority name **different** lanes, the mapper raises a `ContextMappingError` — a real authoring conflict, not a silent pick. Ties within one lane are broken by the lowest `rule_id`.
4. **Fallback** catches candidates that matched no non-fallback rule at all.

See [context routing](CONTEXT_ROUTING_GUIDE.md) for the full selector semantics, the `required` vs `pinned` distinction, and worked examples.

##### Default source rules

Omitting `context.sources` entirely supplies these three rules. Authoring
`context.sources` **replaces them wholesale** — there is no merge, so a
session that declares its own rules loses the thread-digest and
history-fallback behavior unless it re-declares it.

| `rule_id` | `lane_id` | Selector | Priority | Notes |
| --- | --- | --- | --- | --- |
| `thread_digest` | `retrieval` | `evidence_types: [task_state, recent_history, open_objectives]` | `100` | The thread digest the model uses to know its recorded state. |
| `thread_digest_epistemics` | `retrieval` | `evidence_types: [epistemics]` | `100` | `pinned: true` — epistemics bypass MMR diversity filtering. |
| `history_fallback` | `history` | (fallback) | `0` | Catch-all for every other candidate. |

The digest also emits a `verified_facts` item, which no default rule names —
it falls through to `history_fallback` and lands in the `history` lane. See
[the `digest` section](#digest) for the fix.

This is why a session that declares a single `fallback: true` rule into
`operational` — as the bundled core session does — moves the catch-all out
of the `history` lane entirely.

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
| `relevance_provider` | `lexical` | Relevance scoring backend. **Not implemented — the mapper always scores relevance lexically**, regardless of this value. The field validates and compiles, but setting it to `semantic` or `hybrid` changes nothing. |

##### How a candidate is scored

Every candidate gets a component per weight, then one base score:

```text
base_score = Σ (normalized_weight × component)
           - conflict_penalty × conflict_component
           - token_cost_penalty × token_cost_component
```

The eight **positive** weights (`relevance`, `mode_affinity`, `salience`,
`freshness`, `authority`, `continuity`, `user_priority`, `recency`) are
normalized to sum to 1.0 among themselves. `conflict_penalty` and
`token_cost_penalty` are **not** part of that normalization — they are raw
multiplicative penalties applied after. Setting all eight positive weights
to `0.0` raises `SessionPlanError` at compile time.

##### The scoring components

| Weight | Range | How it is computed |
| --- | --- | --- |
| `relevance` | `0.0`–`1.0` | Lexical overlap. Query and candidate are lowercased, tokenized on `[a-z0-9']+`, and stripped of 19 stopwords (`a`, `about`, `an`, `and`, `are`, `at`, `be`, `for`, `from`, `how`, `in`, `is`, `it`, `of`, `on`, `or`, `the`, `to`, `with`). Candidate terms come from title + content + tags. The score is `matched_query_terms / total_query_terms` — the denominator is the **query** size, so a long query scores low even for a good match. An empty query scores `0.0`. |
| `mode_affinity` | `0.6`–`1.0` | `1.0` when the candidate's `mode_ids` include the active mode, otherwise `0.6`. A candidate with no `mode_ids` gets `0.6`. Candidates whose `mode_ids` exclude the active mode are gated out before scoring. |
| `salience` | `0.0`–`1.0` | The producer's declared salience, passed through unchanged. |
| `freshness` | `0.0`–`1.0` | `1.0` when the candidate has no expiry or has not expired. After expiry it decays as `exp(-stale_seconds / ttl_seconds)`, where TTL is `expires_at - collected_at`. A candidate with no `collected_at` scores `0.0`. `min_freshness` on a source rule compares against this component. |
| `authority` | `0.0`–`1.0` | `authority × confidence × provenance_multiplier`. The multiplier comes from the host's provenance table (below), so a speculative item with perfect authority and confidence still scores `0.30`. |
| `continuity` | `0.0`–`1.0` | The producer's declared continuity with prior turns, passed through unchanged. Defaults to `0.0`. |
| `user_priority` | `0.0`–`1.0` | The user-declared priority, passed through unchanged. Defaults to `0.0`. |
| `recency` | `0.0`–`1.0` | `exp(-ln(2) × age_hours / 24)` — a 24-hour half-life on `collected_at`. A candidate with no `collected_at` scores `0.5` (neutral, not zero). |
| `conflict` | `0.0`–`1.0` | From the host's conflict-risk table: `low` → `0.0`, `medium` → `0.5`, `high` → `1.0`. Subtracted, weighted by `conflict_penalty`. |
| `token_cost` | `0.0`–`1.0` | `token_count / available_context_tokens`, clamped to `1.0`. Token count is the candidate's own estimate, else `ceil(len(content) / 4)` (the host's chars-per-token ratio). Penalizes bulky candidates. |

The host's two scoring tables are configurable via
`CONTEXT_MAPPER_PROVENANCE_AUTHORITY` and `CONTEXT_MAPPER_CONFLICT_PENALTY`
(JSON objects). The defaults are:

| Provenance | Multiplier | | Conflict risk | Penalty |
| --- | --- | --- | --- | --- |
| `authoritative` | `1.00` | | `low` | `0.0` |
| `system_observed` | `0.98` | | `medium` | `0.5` |
| `user_asserted` | `0.75` | | `high` | `1.0` |
| `recalled_memory` | `0.70` | | | |
| `inferred` | `0.55` | | | |
| `diagnostic` | `0.45` | | | |
| `speculative` | `0.30` | | | |

##### Selection order and diversity

Selection runs in three passes:

1. **Forced.** Candidates marked `required` (by the rule or by the producer) and candidates marked `pinned` are selected first, in that order, before any scoring competition. A `required` candidate that does not fit the budget raises `ContextBudgetError`; a `pinned` candidate that does not fit is dropped with `pinned_budget_exhausted`. This is the only difference between the two: required cannot be silently dropped.
2. **Lane minimums.** Lanes still under their `min_items`/`min_tokens` coverage target get first claim on eligible candidates.
3. **MMR.** Remaining candidates are chosen by maximal marginal relevance: `diversity_lambda × base_score - (1 - diversity_lambda) × similarity`, where similarity is the maximum Jaccard term overlap against already-selected candidates. Two candidates sharing a non-empty `conflict_key` are treated as fully similar (`1.0`). `diversity_lambda: 1.0` is pure score ranking; `0.0` is pure anti-similarity. The default `0.75` weighs score 75% against similarity 25%.

#### `context.budget`

Model-aware budget limits applied after fixed per-turn reservations.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `output_reserve_tokens` | `int >= 1` | `1024` | Tokens reserved for model output. |
| `safety_margin_tokens` | `int >= 0` | `512` | Safety margin tokens. |
| `max_materialized_candidates` | `int >= 1` | `32` | Maximum candidates materialized into the prompt. **Not implemented — accepted by the schema but never read by the runtime.** Use lane `max_items` to bound how many candidates a lane can contribute. |

##### How the budget is computed

The mapper's global context allowance is:

```text
available_context_tokens = max(0,
    model_context_window_tokens
  - output_reserve_tokens
  - safety_margin_tokens
)
```

Both reservations are deducted **before** any candidate is admitted, so every
candidate competes inside the already-reduced window. `model_context_window_tokens`
comes from the resolved model's declared `context_window` in the model
manifest, falling back to the host default (131072) when the manifest entry
does not declare one.

A candidate is admitted only if it fits all three limits, checked in this
order:

1. the **global** remaining-token budget;
2. the lane's `max_tokens`;
3. the lane's `max_items`.

A candidate that fails check 1 is dropped as `global_budget_exhausted`; check 2
as `lane_token_budget_exhausted`; check 3 as `lane_item_limit_reached`. Lane
`min_tokens` and `min_items` are coverage *targets* — they make a lane
compete harder for candidates, but are not guarantees.

> These mapper reservations are **not** the whole provider-request budget.
> Whole-request sizing separately accounts for the serialized system and user
> messages, tool schemas and arguments, the response-format schema, the
> requested completion room, and a request margin. The mapper's
> `output_reserve_tokens` and `safety_margin_tokens` are an earlier, separate
> reservation for optional context — setting them does not change how the
> provider request is sized.

#### `context.history`

Conversation history settings for context assembly.

| Key | Default | Meaning |
| --- | --- | --- |
| `include_recent_conversation` | `true` | Include the recent-conversation excerpt. Read by the runtime. |
| `include_reflection` | `false` | Include the diagnostic reflection. **Not implemented — accepted by the schema but never read by the runtime.** |
| `recent_message_limit` | `6` | Recent-message selection limit; non-negative. **Not implemented — never read by the runtime.** The recent-conversation excerpt is built with a fixed three-item limit; use `digest.recent_entries` to size the digest's recent-history item. |
| `replay_max_bytes` | absent | Optional 1–1048576-byte replay ceiling, tighten-only against a finite host ceiling. |
| `digest` | object | Bounded task-state/recent-history/open-objective projection. |

##### `include_recent_conversation`

Gates one section only: the recent-conversation excerpt, which is appended to
the **`history`** lane. It does not gate the thread manifest or the digest
items. The excerpt is built from the thread store with a hard-coded
three-entry limit, and host-only command queries are filtered out of it.

The excerpt is additionally suppressed by the host's replay policy in
`manual` trigger mode and when the query is shorter than the configured
`min_query_chars` — so `include_recent_conversation: true` is a permission,
not a guarantee.

##### `replay_max_bytes`

Bounds the **turn-start replay working set**: the total tool-result bytes
replayed verbatim for the open task. Content beyond the ceiling collapses
into recoverable reference stubs that keep the addressing information for
the original result; the model re-expands a stub through
`thread_read_content`. Nothing is lost — the thread store remains the
lossless ground truth, and this knob governs *residency*, not storage.

The effective ceiling is the **minimum** of the session knob and the host's
`context_budget_replay_max_bytes` (env `LISA_CONTEXT_BUDGET_REPLAY_MAX_BYTES`,
default `0` = no host ceiling). So:

- session value **below** the host ceiling → the session value applies;
- session value **above** the host ceiling → the host ceiling wins, the
  session value is clamped down;
- host ceiling `0` → any positive session value applies;
- neither positive → no explicit ceiling; the request allocator decides.

##### `digest`

The digest is a **bounded, lossy projection** of the lossless thread, pushed
into the `retrieval` lane each turn so the model starts knowing its recorded
state instead of re-reading it through tools. It is not in-place thread
compaction, and it is not the removed lossy digest sidecar.

| Key | Default | Bound | Meaning |
| --- | --- | --- | --- |
| `enabled` | `true` | — | `false` → the digest is omitted entirely and the model falls back to tool pulls only. |
| `recent_entries` | `10` | 1–40 | How many recent thread exchanges the recent-history item summarizes. The digest loads up to `recent_entries × 4` source entries and selects the last `recent_entries`. Fewer available entries is not an error. |
| `recent_body_chars` | `1800` | 200–8000 | Character budget for the recent-history body. Rendering stops before the line that would exceed it. |
| `task_view_budget_bytes` | `4096` | 1024–16384 | Byte budget for the task view. The rendered view additionally caps each objective and latest-response field at 400 characters. |
| `max_open_objective_lines` | `3` | 1–10 | How many open-objective lines the item carries. |

The digest emits these evidence types, which the default `context.sources`
rules route as follows:

| Evidence type | Default rule | Default lane |
| --- | --- | --- |
| `task_state` | `thread_digest` | `retrieval` |
| `recent_history` | `thread_digest` | `retrieval` |
| `open_objectives` | `thread_digest` | `retrieval` |
| `epistemics` | `thread_digest_epistemics` (pinned) | `retrieval` |
| `verified_facts` | *(no dedicated rule)* | `history` via the fallback |

> **`verified_facts` is a gap in the default rules.** The thread digest emits
> a `verified_facts` item, but no default source rule names that evidence
> type, so it falls through to the `history_fallback` catch-all and lands in
> the `history` lane instead of `retrieval`. Verified facts are
> high-value retrieval-grade evidence, so a session that wants them in
> `retrieval` should declare a rule for them explicitly:
>
> ```yaml
> context:
>   sources:
>     - rule_id: verified_facts
>       lane_id: retrieval
>       evidence_types: [verified_facts]
>       priority: 100
> ```

#### `context.memory`

Memory-store consolidation policy is pressure-driven and paid from the
configured slice slot/budget. It can make auxiliary model calls and add
latency; it does not compact the lossless thread in place.

| Key | Type | Default | Description |
| --- | --- | --- | --- |
| `consolidation` | `MemoryConsolidationTuning` | (see below) | Episodic-pressure-triggered memory consolidation. |

The block only takes effect when the **memory pack is active** — it is
implemented by the memory pack's slice provider, which the host wires through
the `lisa.extensions.slice_providers` entry point. A session that does not
include `lisa-pack-memory` can set these knobs, but nothing will read them.

Two record kinds are involved:

- **Episodic** records are the active, accumulating source records. They are
  counted for pressure and loaded in batches for consolidation.
- **Semantic** records are durable distilled facts, produced either by
  `consolidate_semantic` (which also *retires* the episodic sources it
  distilled) or by `mine_thread_segment` (which does not retire anything).

Neither path rewrites or compacts the lossless thread store.

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
| `preferred_model_id` | `string` | `default` | Model ID used for the consolidate operation. **Not implemented — the value is carried into the slice config but the slice does not pass it when invoking the operation.** |
| `consolidate_operation` | `string` | `consolidate_semantic` | Operation ID invoked to consolidate. |
| `mine_on_segment_seal` | `bool` | `true` | Mine newly sealed thread segments on every slice pass. Independent of the consolidation fire decision. |

##### The fire decision

Pressure is `episodic_count / max_records`. The active count comes from
`memory.count` (filtered to `memory_kind: episodic`, `status: active`), with
a bounded `memory.list` fallback when `memory.count` is unavailable.

| Condition | Behavior |
| --- | --- |
| pressure below `pressure_threshold` | No fire. |
| pressure at or above `hard_threshold` | Fires immediately — the pressure budget and `min_interval_turns` are both overridden. |
| between the thresholds | Fires only if **all** hold: the count is at least `min_episodic_count`, the slice's pressure-derived `budget_ms` is at least 150 ms, and `min_interval_turns` turns have elapsed since the last fire. |

The "slice pressure curve" is the kernel's pressure-derived slice budget
(`SlicePolicy.budget_ms`) — the time the host is willing to spend on
post-turn work. The between-thresholds branch defers to a less-pressured
turn when the budget is too small to start an LLM distillation call.

`min_interval_turns` is tracked by a turn counter kept in a
`last_consolidation.json` sidecar beside the memory database, so the
interval survives restarts.

##### What a slice pass does

The slice runs **two independent jobs** every time it is invoked, in this
order. Only the second one is gated by the fire decision.

1. **Mine newly sealed thread segments.** If `mine_on_segment_seal` is on,
   the slice scans the thread segment index for sealed segments not yet
   mined, distills each into 1–3 semantic records carrying
   `thread_turn_lo`/`thread_turn_hi` provenance via `mine_thread_segment`,
   and advances a `mine.marker` sidecar. Mining is idempotent per segment
   and at most 2 segments are mined per pass. The lossless segment is
   retained — mining adds records, it does not replace them. **This job
   runs on every slice pass, whether or not consolidation fires.**
2. **Consolidate under memory pressure.** Only when the fire decision says
   so: load up to `batch_size` active episodic records, then invoke
   `consolidate_operation` (default `consolidate_semantic`) with the batch
   and the scope identifiers (`user_slug`, `workspace_id`, `memory_scope`,
   `surface_id`). The operation distills the batch into one durable semantic
   record and retires the episodic sources.

Every failure along the way — count, list, mining, the operation itself, or
the sidecar write — is caught, logged, and turned into a diagnostic outcome.
A consolidation failure never breaks the turn.

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

> **Not implemented.** The `conflict_policy` field validates and compiles, but no runtime code reads it. The mapper always applies a *soft* conflict penalty: conflicting candidates lose score through the `conflict_penalty` ranking weight and are de-duplicated by MMR similarity, but none of the three policies above is actually enacted. Treat the value as aspirational; do not rely on `surface_conflict` to surface a disagreement to the model.

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
    memory: lisa-pack-memory         # the memory pack owns the memory capability
```

Provider IDs are matched **exactly**, after lowercasing and converting
underscores to hyphens. Use the pack's **distribution name** — the
`[project] name` in its `pyproject.toml` — e.g. `lisa-pack-brave`. The bare
slug `brave` normalizes to `brave`, which does **not** match
`lisa-pack-brave`; a `providers` entry naming the bare slug will fail to
resolve and the catalog reports the available provider names in the error.

> Alias resolution (`lisa-pack-brave` ≡ `brave` ≡ `brave_pack`) applies to
> **pack activation** — `resources.packs.include`/`exclude`/`required` and
> the `from:` directive — where the host expands the alias set. It does
> **not** apply to `providers` ownership, which is an exact match against
> the discovered distribution name.

Selection is exact — there is no ordered preference list. If the selected
pack is not active, the catalog entry is absent (no fallback).

**When a collision actually happens.** The catalog groups candidates by resource ID. A single candidate is used as-is. Two or more candidates from the same pack's directory are one pack, not a collision. But `lisa-pack-core`, `lisa-pack-tavily`, `lisa-pack-brave`, and `lisa-pack-serper` each ship a tool with `tool_id: web_search` from *different* pack roots — that is a genuine collision and needs an explicit `providers.tools.web_search` entry.

Resolution order when several candidates share an ID:

1. One candidate → used directly.
2. Exactly one **overlay** candidate (an entry under the host's overlay root) → it wins automatically, even over an explicit `providers` selection.
3. Two or more overlay candidates → `CatalogCollisionError` (ambiguous).
4. Two or more non-overlay candidates → an explicit `providers.<kind>` selection is **required**; without it, `CatalogCollisionError`.

When a selection is present but matches zero or more than one candidate, `CatalogCollisionError` names the available providers.

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
- Every `resources.packs.auto_install` key must also appear in `resources.packs.required`.
- `execution` numeric limits must be `>= 1`; `execution.capabilities` and `execution.scopes` keys must be non-empty, and each nested numeric value must be `>= 1`.
- `context.memory.consolidation.hard_threshold` must be `>= pressure_threshold`.
- `context.history.replay_max_bytes` must be `1`-`1048576`; `context.history.digest` fields are bounded (`recent_entries` 1–40, `recent_body_chars` 200–8000, `task_view_budget_bytes` 1024–16384, `max_open_objective_lines` 1–10).
- `context.sources[].min_freshness` must be `0.0`-`1.0`; `context.budget.output_reserve_tokens` must be `>= 1` and `safety_margin_tokens` `>= 0`; `max_materialized_candidates` must be `>= 1`.

Any violation raises `SessionPlanError` (a `ValueError` subclass). The host renders this as a startup error and exits.

Two further errors are raised **after** compilation, at catalog build time, and are not `SessionPlanError`:

- `CatalogCollisionError` — two active packs contribute the same mode, operation, or tool ID and the session plan did not declare the owner in `providers`. See [provider selection](#provider-selection).
- `ContextMappingError` — two `context.sources` rules at the same priority match one candidate but name different lanes.

---

## Common Patterns

### Multi-Pack Session with Explicit Ownership

```yaml
providers:
  modes:
    default: lisa-pack-core
  operations:
    discussion_briefing: lisa-pack-personas  # the personas pack's pre-turn operation
  tools:
    web_search: lisa-pack-brave       # Brave owns the web_search tool
  capabilities:
    memory: lisa-pack-memory          # the memory pack owns the memory capability
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
    include: [lisa-pack-core, lisa-pack-personas]
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
- **Two search provider packs without a `providers.tools` entry**: `lisa-pack-core`, `lisa-pack-tavily`, `lisa-pack-brave`, and `lisa-pack-serper` all ship a tool with `tool_id: web_search`. Activating more than one without declaring the owner raises a `CatalogCollisionError` at startup. Exactly one pack can own `web_search` at a time.
- **A search provider pack without its API key**: the pack installs and the tool appears in the catalog, but every call fails with a "not configured" error naming the missing environment variable (for Tavily, `TAVILY_SEARCH_API_KEY`). The key is read from the environment, never from front matter — a secret does not belong in a versioned session file.
- **`lisa-pack-core` alone is not a working search**: its `web_search` tool is a deliberate contract stub that returns "provider unavailable". It keeps `tool_ids: [web_search]` resolvable so operation validation passes; a concrete provider pack (Tavily, Brave, Serper) supplies the live tool.
- **Authoring `context.sources` drops the default rules**: the default rule set is replaced wholesale, not merged. A session that declares its own sources loses the `thread_digest`, `thread_digest_epistemics`, and `history_fallback` rules unless it re-declares them.
- **Two source rules at the same priority naming different lanes**: the mapper raises a `ContextMappingError` rather than picking one. Split the priorities.
- **`auto_install` for a pack that is not `required`**: the compiler rejects it. `auto_install` customizes *how* a required pack installs; it never decides *whether* it installs.
- **`relevance_provider: semantic` does nothing**: the field is schema-only. Relevance is always scored lexically.
- **`conflict_policy: surface_conflict` does nothing**: the field is schema-only. Conflicting candidates are only softly penalized through the `conflict_penalty` ranking weight.
- **`include_reflection` does nothing**: the field is schema-only.
- **`recent_message_limit` does nothing**: the field is schema-only. The recent-conversation excerpt uses a fixed three-entry limit; size the digest's recent-history item with `digest.recent_entries` instead.
- **`max_materialized_candidates` does nothing**: the field is schema-only. Bound a lane's candidate count with `context.lanes[].max_items` instead.
- **`preferred_model_id` in the memory consolidation block does nothing**: the value reaches the slice config but is not passed to the operation. The consolidate operation picks its own model from its own definition.
- **`verified_facts` lands in the `history` lane, not `retrieval`**: the thread digest emits a `verified_facts` item, but no default source rule names that evidence type, so the `history_fallback` catch-all takes it. Add an explicit rule to route it to `retrieval`.

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

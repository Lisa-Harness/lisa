# Model Router

The Lisa Framework supports a pluggable model-router mechanism that selects
a model **per turn-loop iteration** based on turn state, rather than pinning
one model for the entire turn. This is layered on top of the existing static
`mode_models`/`roles` resolution — it does not replace it.

> **What's new.** A `ModelRouter` Protocol (`kernel/seams/protocols.py`) and
> `lisa.extensions.model_routers` entry-point group let packs contribute
> router backends. The built-in `lisa-router` pack ships a rules-based engine
> that makes zero extra model calls. External packs (e.g. `lisa-pack-aoi-router`
> in `../lisa_packs/`) can register vendor-backed routers under the same group.

---

## How it works

### Three-tier fallback

The turn loop (`kernel/turn_runner.py:_resolve_iteration_model`) resolves the
model for each iteration with this precedence:

1. **`ContextSeams.model_router`** — the formal `ModelRouter` Protocol with
   a widened `RouterRequest` carrying `query`, `last_tool_result`,
   `required_capabilities`, `mode_id`, and `roles_needed`. This is the
   pluggable router path.
2. **`ModelPolicy.choose`** — the legacy duck-typed optional method with the
   narrow `(default_model_id, iteration, tool_calls_so_far, budget_remaining)`
   signature. Preserved for backward compatibility.
3. **Static `model_def`** — the once-resolved `preferred_model_id`. This is
   the default behavior when no router and no `choose` are wired.

When no router is wired (the default), the loop behaves exactly as before.

### Data flow

```
models.yaml
  → load_model_manifest()               # kernel/model_manifest.py
  → ModelManifest(models, mode_models, roles, router)

run_turn()
  → resolve_model(preferred_model_id)  # mode_models → roles → find_model
  → per iteration:
      → ModelRouter.choose(RouterRequest) → RouterDecision(model_id)
      → resolve_model(decision.model_id) → ManifestModelEntry
      → complete(model_def=...) → ProviderGateway.complete()
```

The router's output is a `model_id` string that flows back through the
existing `resolve_model()` → provider-gateway path. **No new completion code
path.** The router only picks the model; completion still flows through the
existing gateway.

---

## Configuration

Add an optional `router:` block to `models.yaml`:

```yaml
mode_models: {}
roles: {default: 'gemma4:cloud', deep: 'gpt-5.4'}

router:
  backend: lisa-router          # or "aoi-router" (external pack)
  rules:
    # Capability gates — fire first, capture the whole turn.
    - when: required_capabilities includes "code"
      use: gpt-5.3-codex
    - when: required_capabilities includes "reasoning"
      use: gpt-5.4
    - when: required_capabilities includes "vision"
      use: gpt-5.4
    # Budget gate — downgrade long tool-heavy turns before paying for synthesis.
    - when: tool_calls_so_far >= 5
      use: gemma4:cloud
    # Phase gate — synthesize on the deep model after the first tool result.
    # Only reached on plain (non-capability) modes because the capability
    # gates above already captured code/reasoning/vision turns.
    - when: iteration > 1 and last_tool_result is not None
      use: gpt-5.4
  fallback: gemma4:cloud        # if no rule matches, stay on the default
```

- **`backend`** — names the active router backend. When empty or absent, no
  router is active and static `mode_models`/`roles` resolution is the only
  path.
- **`rules`** — ordered list of `{when, use}` entries evaluated
  first-match-wins.
- **`fallback`** — the model used when no rule matches. If empty, the router
  returns `None` and the loop falls back to the statically-resolved
  `preferred_model_id`.

### Rule ordering matters

The default config is ordered so rules never shadow each other:

1. **Capability gates** (rules 1-3): a mode that needs `code`, `reasoning`,
   or `vision` gets the deep model for the whole turn. These fire first so
   capability-needing modes never hit the budget or phase gates below.
2. **Budget gate** (rule 4): a long, tool-heavy turn on a plain mode
   downgrades to the default model before we pay for frontier synthesis.
3. **Phase gate** (rule 5): after the first tool returns, synthesize on the
   deep model. Only reached on plain (non-capability) modes because rules
   1-3 already captured code/reasoning/vision turns.

### Presets

`/models discover` writes the active `router:` block plus three commented
alternative presets to `models.proposed.yaml`. Each preset uses resolved
concrete model ids so they're uncomment-and-go:

- **cost-conscious**: cheap default, escalate to deep only on code or when
  the turn drags, downgrade to default early on tool-heavy turns.
- **quality-first**: frontier everywhere, downgrade to default only on very
  long tool-heavy turns.
- **coding-focused**: code model for code turns, deep for synthesis, code
  model as fallback.

To switch presets, uncomment the desired block in `models.proposed.yaml`,
comment out the active `router:` block, and run `/models apply`. The apply
step strips all YAML comments (including the preset annotations) so the
active `models.yaml` stays machine-clean.

### Relationship to `mode_models` / `roles`

These are **not replaced**. They become the router's input defaults:

- `mode_models[mode_id]` / `roles["deep"]` sets the **starting model** for
  the turn (`preferred_model_id`).
- The router receives this as `RouterRequest.default_model_id` and can
  override it per iteration.
- When the router is absent or returns `None`, the static default is used.

Remove the `router:` block (or set `backend: ""`) and you're back to pure
static selection — no code path changes.

### Pack-seeded router config

Installing a router pack can also **seed** the `router:` block
automatically — you don't have to hand-write the YAML. This is the
router-side parallel of pack-declared model roles
(`lisa.extensions.model_roles`):

- Packs declare a default router config through the
  `lisa.extensions.model_router_configs` entry-point group.
- `/models discover` calls `discover_pack_router_config()` (the first
  pack-declared config wins) and merges it into the proposed manifest.
- Symbolic `use`/`fallback` references (mode/role names like `"default"`
  or `"deep"`) are resolved to concrete model ids against the discovered
  model set. Rules referencing models not in the discovered set are
  stripped, so the proposal always loads cleanly.
- A user-provided `router:` block in `models.yaml` always wins. The pack
  default is only seeded when the user has not written one.

The built-in `lisa-router` pack ships a default config that routes
code-capable turns to the deep model, escalates to the deep model after
the first tool result, and downgrades to the default model when the
tool-call budget is running low. After `/models discover --force`, the
generated `models.proposed.yaml` includes a ready-to-use `router:` block
with resolved model ids — no manual editing required.

---

## Supported rule conditions (lisa-router)

The built-in `lisa-router` engine supports simple, dependency-free predicates:

| Condition | True when |
|---|---|
| `required_capabilities includes "code"` | Every listed capability is in `RouterRequest.required_capabilities` |
| `iteration > N` / `iteration >= N` | The iteration number exceeds / meets the threshold |
| `tool_calls_so_far >= N` | Tool calls so far meet the threshold |
| `last_tool_result is not None` | At least one tool has returned a result this turn |
| `last_tool_result is None` | No tool has returned yet |

Predicates can be joined with `and` / `or`:

```yaml
rules:
  - when: iteration > 1 and last_tool_result is not None
    use: gpt-5.4
  - when: required_capabilities includes "code" or iteration > 5
    use: gpt-5.2-codex
```

Conditions are intentionally simple — single predicates joined by `and` /
`or`. This keeps the engine dependency-free (no `eval`, no AST parser) and
auditable. Complex routing needs should use a vendor backend.

---

## Pluggable backends

### Built-in: `lisa-router`

- **Module**: `sdk/router/` (part of the `lisa-sdk` distribution)
- **Entry points** (declared in `sdk/pyproject.toml`):
  - `lisa.extensions.model_routers` → `sdk.router:get_router_factory`
    (runtime factory — the `ModelRouter` instance)
  - `lisa.extensions.model_router_configs` →
    `sdk.router.default_config:get_default_router_config`
    (static config — the default `router:` block for `/models discover`)
- **Zero model calls**: the engine evaluates rules locally against
  `ManifestModelEntry` metadata (`cost_tier`, `latency_tier`, `capabilities`)
  and `RouterRequest` fields.
- **Active when**: `router.backend == "lisa-router"` in the manifest.
- **Default config**: the backend declares a default `router:` block with
  symbolic refs (`"default"`, `"deep"`) that `/models discover` resolves
  to concrete model ids from the discovered set. See
  `sdk/router/default_config.py`.

> The `lisa-router` engine is **not a pack** — it ships no modes,
> operations, or authored content. It is a runtime `ModelRouter` backend
> bundled with the SDK, kin to `kernel.model_router_discovery` (which owns
> the discovery contract). External router backends (e.g. `lisa-pack-aoi-router`)
> still register under the same `lisa.extensions.model_routers` group from
> a real pack.

### External: `lisa-pack-aoi-router` (deferred)

- **Pack**: `../lisa_packs/aoi-router/` (independent repo)
- **Entry point**: same `lisa.extensions.model_routers` group
- **Model call**: the vendor router is itself a completion call (using
  `kernel.provider_core.AzureOpenAIGateway` with `response_format` for
  structured output), which returns a `model_id`. This is the latency cost
  you accept for ML-based routing.
- **Active when**: `router.backend == "aoi-router"` in the manifest.

### Writing a custom router

1. Create a pack with a `pyproject.toml` entry point:

```toml
[project.entry-points."lisa.extensions.model_routers"]
my-router = "my_pack.router:get_router_factory"

[project.entry-points."lisa.extensions.model_router_configs"]
my-router = "my_pack.router.default_config:get_default_router_config"
```

2. Implement the factory:

```python
from kernel.model_manifest import ModelManifest
from kernel.seams.protocols import ModelRouter, RouterDecision, RouterRequest


def get_router_factory():
    return _factory


def _factory(manifest: ModelManifest | None, session_plan) -> ModelRouter | None:
    if manifest is None or manifest.router is None:
        return None
    if manifest.router.backend != "my-router":
        return None
    return MyRouter(manifest)


class MyRouter(ModelRouter):
    def choose(self, req: RouterRequest) -> RouterDecision | None:
        # ... your routing logic ...
        return RouterDecision(model_id="...", reason="...", routed_by="my-router")
```

3. Return `None` from the factory when your backend is not active (so the
   host skips it and other factories get a chance).

The factory receives `(manifest, session_plan)`. The first non-`None` factory
result wins — ordering is deterministic by entry-point discovery order.

---

## Telemetry

Each `RouterDecision` carries:

- `model_id` — the chosen model
- `reason` — human-readable explanation (e.g. `"lisa-router: rule 'iteration > 1' matched at iteration 2"`)
- `routed_by` — which router made the decision (e.g. `"lisa-router"`)

The turn loop's `on_stage` hook emits `model_call` events that carry the
chosen `model_id` and `provider`. Router decisions are auditable via the
`reason` and `routed_by` fields.

---

## Design principles

- **Zero extra model calls (rules-based).** The `lisa-router` engine is
  pure-Python local evaluation — no model call. This preserves the "zero
  extra model calls" principle from `docs/ROUTER_DESIGN.md`. Vendor routers
  (e.g. `aoi-router`) are opt-in and accept the latency cost.
- **No replacement, layering.** `mode_models`/`roles` are demoted from
  "pinned model" to "starting model + fallback" but are not removed. They
  remain the default when the router is absent or declines.
- **First-non-None wins.** Router discovery uses the same first-non-None
  pattern as `lisa.extensions.slice_providers`, keeping ordering deterministic.
- **Graceful failure.** Router exceptions are caught and logged; the loop
  falls back to static resolution. A broken router never breaks the turn.

### Capability gate (router safety net)

When a mode declares `required_capabilities` in its `_mode.md` front matter
(pack-authored `ModelCapability` strings like `tools`, `code`), the turn
runner validates the resolved model's `ManifestModelEntry.capabilities`
**before** the tool loop starts. When a router is wired and the model is
missing a required capability, the turn returns a clear user-facing block
instead of letting a weak model reach for write-effect tools it cannot use
reliably — e.g. `gemma4:cloud` in `coding` mode calling
`coding_create_worktree` for a "write a roadmap" request it can't complete.

The gate is the safety net **behind** the router: a router rule like
`required_capabilities includes "code" → gpt-5.3-codex` should escalate to a
capable model first; the gate catches the cases where the router declined or
returned `None`. When no router is wired the gate is advisory (no block) — a
host without a router has no escalation path, so blocking would dead-end.

A mode with no `required_capabilities` (the default) never blocks — every
model is admitted, preserving the pre-field behavior. See
`docs/PACK_AUTHORING_GUIDE.md` for authoring.
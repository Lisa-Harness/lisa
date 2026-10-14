# Configuration

Configuration has several owners. A session manifest is not a substitute
for provider credentials, host launch policy, or authorization.

| Owner | Controls | Canonical reference |
| --- | --- | --- |
| Transport host | CLI arguments, `.env` loading, authentication, rendering | Host's own documentation |
| Runtime `HostConfig` | Data paths, logging, thread and context defaults | [runtime config](../runtime/src/runtime/config.py) |
| Session plan | Surface, packs, catalog owners, routing vocabulary, context, tighten-only execution requests | [Session authoring](SESSION_CONTEXT_AUTHORING_GUIDE.md) |
| Model manifest | Model IDs, providers, capabilities, mode pins, optional router | [Manifest loader](../kernel/src/kernel/model_manifest.py) and [model router](../sdk/src/sdk/router/README.md) |
| Host approval policy | Exact-action allow/ask/deny decisions | [Approval policy](POLICY_GUIDE.md) |
| Pack | Pack-owned settings and extension configuration | Pack README and declared configuration schema |

## Data directory and profiles

`HostConfig.from_env` chooses the data root in this order:

1. Explicit `data_dir` argument supplied by the host.
2. `LISA_DATA_DIR`.
3. `LISA_PROFILE`, resolving to `~/.lisa/profiles/<name>`.
4. `~/.lisa`.

A repository checkout does **not** change this default to `artifacts/`.
For an isolated checkout, set an explicit directory outside version control:

```sh
export LISA_DATA_DIR="$HOME/.lisa-dev"
```

In PowerShell:

```powershell
$env:LISA_DATA_DIR = Join-Path $HOME ".lisa-dev"
```

Use either `LISA_DATA_DIR` or `LISA_PROFILE`, not both. Profile names must
be single path-safe segments. Profiles isolate data-root-derived state;
the packs environment remains shared with the base directory. They are
not an OS-level or multi-tenant security boundary.

The canonical categories are defined by [sdk.layout](../sdk/src/sdk/layout.py):

| Path below the data root | Contents |
| --- | --- |
| `config/` | Active `models.yaml`, `session_context.md`, `policy.yaml`, overlays |
| `state/` | Model proposals, approvals, recovery records, pack locks, work products |
| `data/` | Memory database, threads and result objects, sandboxed files |
| `cache/` | Fetch and other regenerable caches |
| `logs/` | Runtime, MCP, and token-usage logs |
| `run/` | Per-session execution state, scratch, sandbox facts |

See [storage](STORAGE.md) before backing up, moving, or deleting state.

## Session selection

The runtime resolves a local Markdown session in this order:

1. An explicit host `session_context` argument (CLI `--session-context`).
2. `LISA_SESSION_CONTEXT_PATH`.
3. `<data-dir>/config/session_context.md`.
4. The [core default session](../sdk/packs/core/src/sdk/packs/core/sessions/default.md).

A configured but missing explicit file is an error. Session sources are
local-only. The Markdown body supplies primary instructions; YAML front
matter compiles into a strict v2 plan. Unknown keys fail validation.

For a small custom session, inherit the bundled routing vocabulary instead
of copying it:

```markdown
---
schema_version: 2
name: local-agent
surface:
  type: personal_dm
  id: local-console
  memory_scope: personal
resources:
  packs:
    required: [lisa-pack-core]
routing:
  from: {pack: lisa-pack-core, session: default}
---
# Local agent

Use tools carefully and distinguish verified results from assumptions.
```

`from:` is resolved by the runtime before strict kernel compilation.
Inheritance replaces a whole block; it does not merge fields.
See [session authoring](SESSION_CONTEXT_AUTHORING_GUIDE.md) for the complete
schema, including `execution` and context/model selection.

## Models

The default active path is `<data-dir>/config/models.yaml`. A session's
`context.models.path` may select another local YAML/JSON manifest, resolving
relative paths against the session file's directory.

The manifest lists models and declares `mode_models`, including a `default`
pin. Entries can declare aliases, capabilities, context windows, completion
ceilings, and pricing metadata. Optional `router` rules choose a model
per loop iteration; this is distinct from routing between modes.

Use host `/models discover`, `/models apply`, and `/models reload` to generate,
validate, activate, or reload a roster. Review the proposal before applying it:
provider discovery is not proof that every model can perform every workflow.
Do not copy a full manifest from an old incident report.

See [model-router configuration](../sdk/src/sdk/router/README.md#configuration)
and [token budgets](TOKEN_BUDGET.md) for capability and limit semantics.

## Provider access

The runtime resolves provider settings through [config_resolver](../runtime/src/runtime/config_resolver.py)
and [providers](../runtime/src/runtime/providers.py); SDK gateways receive
resolved configuration. Transport hosts own any `.env` loading.

Common CLI settings include `LISA_OPENAI_API_KEY`,
`LISA_AZURE_OPENAI_ENDPOINT` / `LISA_AZURE_OPENAI_API_KEY`,
`LISA_ANTHROPIC_API_KEY`, `LISA_GOOGLE_API_KEY`,
`LISA_OPENROUTER_API_KEY`, and `LISA_OLLAMA_HOST`.
Use the selected host/provider's reference for all required settings.
Optional gateways such as Bedrock require their extra dependencies.
Never assume a credential alone activates a model or pack.

## Pack installation and activation

- Bundled distributions are installed with the runtime, but optional packs
  are not necessarily active. Core and introspect are host-mandatory.
- `resources.packs.include` filters activation; it does not install packs.
- Missing `required` packs trigger startup auto-install. `auto_install`
  specifies their source; trust restrictions still apply. A missing required
  pack that cannot be installed prevents startup.
- Prefer reviewed, immutable refs for reproducible deployments. A branch
  ref such as `@main` is accepted but is not reproducible.
- External packs use the host-managed packs environment. Source testing
  uses `packstest`; do not repeatedly install into the framework `.venv`.
- `providers` chooses the owner of colliding catalog IDs. These are **pack
  owners**, not model-provider credentials.
- `SDK_CONTENT_ROOTS` supplies additional local roots separated by `;`,
  including on POSIX. It is a development/discovery hook, not a package
  dependency manager.

## Operational settings

| Setting | Runtime default | Purpose |
| --- | --- | --- |
| `LISA_LOG_LEVEL` | INFO effective level | Runtime log verbosity |
| `LISA_LOG_DEBUG` | Empty | Comma-separated logger names forced to DEBUG |
| `LISA_TIMEZONE` | UTC fallback | Timezone used by the host |
| `LISA_THREAD_DIR` | `<data-dir>/data/threads` | Override the thread directory |
| `LISA_THREAD_SEGMENT_TURNS` | 256 | Active-segment sealing threshold |
| `LISA_THREAD_REDACTION_ENABLED` | true | Read-time thread redaction |
| `LISA_THREAD_REDACTION_PATTERNS` | Defaults plus pack patterns | Newline-separated regex overrides; `none` disables |
| `LISA_PROMPT_CACHE_ENABLED` | true | Provider prompt-cache controls |
| `LISA_CONTEXT_BUDGET_REPLAY_MAX_BYTES` | 0 | No separate replay ceiling; use whole-request allocation |

Host `LISA_CONTEXT_BUDGET_*` settings cap prompt-assembly sections.
Kernel `Settings` controls loop limits and request allocation. Session
execution requests tighten host limits; they cannot authorize broader
execution. See [token budgets](TOKEN_BUDGET.md), [context lifecycle](CONTEXT_LIFECYCLE.md),
and [logging](LOGGING.md) rather than guessing from similarly named knobs.

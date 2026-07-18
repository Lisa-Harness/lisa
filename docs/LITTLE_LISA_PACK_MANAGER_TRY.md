# Pack Manager: Try Guide

This guide lets you exercise the new runtime pack capabilities in a local dev loop.

## What You Can Try

- Runtime pack list/enable/disable/install/uninstall commands.
- Activation gating: bundled packs are off until `/pack enable`.
- Package trust policy enforcement (prefix + Git host allowlist).
- Git-based pack refs (not just package-index refs).
- Auto-restart after pack lifecycle changes.
- Optional resume message after restart.
- Pack-contributed MCP server coexistence and host exclusions.
- Capability-provider ownership selection (for memory backends).

## Prerequisites

Run from the workspace root:

```powershell
Set-Location C:\Code\lisa_framework
```

Use your local venv if needed:

```powershell
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& .\.venv\Scripts\Activate.ps1)
```

## 1) Basic Pack Commands

List pack activation state (always-on, bundled, runtime-installed):

```powershell
uv run lisa pack list --no-log
```

From chat/REPL, same operation:

```powershell
uv run lisa once "/pack list" --no-log
```

Enable a bundled pack explicitly:

```powershell
uv run lisa once "/pack enable cloudflare" --no-log
```

Disable it again:

```powershell
uv run lisa once "/pack disable cloudflare" --no-log
```

## 2) Trust Policy Checks

By default, allowed package prefixes are `lisa-pack-` and allowed Git host is `github.com`.

Expected rejection (wrong package prefix):

```powershell
uv run lisa once "/pack install requests" --no-log
```

Expected rejection (wrong Git host):

```powershell
uv run lisa once "/pack install git+https://evil.example.com/acme/lisa-pack-cloudflare.git@main#egg=lisa-pack-cloudflare" --no-log
```

## 3) Configure Policy via Env Vars

Allow custom package prefixes:

```powershell
$env:LISA_PACK_ALLOWED_PREFIXES = "lisa-pack-,acme-pack-"
```

Allow additional Git hosts:

```powershell
$env:LISA_PACK_GIT_ALLOW_HOSTS = "github.com,dev.azure.com"
```

Require pinned Git refs (`@tag`, `@branch`, or `@sha`) (default: true):

```powershell
$env:LISA_PACK_REQUIRE_GIT_REF = "true"
```

Disable auto-restart after install/uninstall:

```powershell
$env:LISA_AUTO_RESTART = "false"
```

## 4) Install From Git

Example shape (replace with your real repo/ref):

```powershell
uv run lisa pack install "git+https://github.com/the-migus-group/lisa-pack-cloudflare.git@v0.1.0#egg=lisa-pack-cloudflare" --no-log
```

From REPL/host command surface:

```powershell
uv run lisa once "/pack install git+https://github.com/the-migus-group/lisa-pack-cloudflare.git@v0.1.0#egg=lisa-pack-cloudflare" --no-log
```

## 4b) Install From Local Folder (Split-Repos Workflow)

Local folder installs are supported so you can iterate on packs before publishing.

Example learning pack path:

```powershell
uv run lisa pack install "C:\Code\lisa-pack-memory" --no-log
```

Example cloudflare pack path:

```powershell
uv run lisa pack install "C:\Code\lisa-pack-cloudflare" --no-log
```

Requirements for local folder install:

- Folder must exist.
- `pyproject.toml` must include `project.name`.
- `project.name` must match allowed prefixes policy (default `lisa-pack-`).

## 4c) Mode/Tool Provider Selection (Session Plan)

When multiple active packs contribute the same mode, operation, or tool ID,
the session plan must declare which pack owns it. This is configured in the
`providers` block of your `session_context.md` front matter:

```yaml
providers:
  modes:
    default: lisa-pack-core
  tools:
    web_search: lisa-pack-brave-search
```

Result:

- `lisa-pack-core` owns the `default` mode.
- `lisa-pack-brave-search` owns the `web_search` tool, replacing the bundled stub.
- Without an explicit `providers` entry, a catalog collision error is raised at startup.
- End users do not need to edit core pack files to transition.

See the [Session Context Authoring Guide](SESSION_CONTEXT_AUTHORING_GUIDE.md) for the full schema.

## 4d) Capability Provider Selection (Memory)

Capability ownership is selected per capability (for example, `memory`) via
the `providers.capabilities` block in the session plan and lock-state overrides.

Installing an external memory provider pack (for example `lisa-pack-mem0`)
auto-selects it when no explicit capability provider has been chosen yet.

## 5) Resume a Request After Restart

You can attach a resume message to install/uninstall in REPL command form:

```plaintext
/pack install git+https://github.com/acme/lisa-pack-cloudflare.git@v0.1.0#egg=lisa-pack-cloudflare --resume check cloudflare zone status for my workspace
```

On next process start, the CLI host consumes that message and runs it as the first turn.

## 6) Uninstall a Pack

```powershell
uv run lisa pack uninstall lisa-pack-cloudflare --no-log
```

## 7) Inspect Installed Pack Metadata

Runtime metadata is persisted at:

```plaintext
<data-dir>\installed-packs.json
```

The lock now persists:

- Pack provenance (`source_kind`, `source_ref`)
- Activation state (`enabled`)
- Explicit capability provider selections (`selected_capability_providers`)

Default data-dir:

```plaintext
<data-dir>
```

## 8) MCP Pack Server Coexistence + Exclusions

Pack-contributed MCP servers are compiled with runtime namespace:

```plaintext
<pack_slug>.<server_id>
```

This allows multiple packs with the same declared `server_id` to co-exist.

To exclude specific pack-contributed server instances, edit:

```plaintext
hosts/core/mcp_servers.yaml
```

Use:

```yaml
pack_policy:
  exclude_servers:
    - cloudflare.github
    - cloudflare.*
```

Supported exclusion formats:

- `<pack_slug>.<server_id>` (preferred)
- `<pack_slug>:<server_id>`
- `<pack_slug>/<server_id>`
- `<pack_slug>.*`

## 9) Suggested Quick Validation Sequence

```powershell
uv run lisa pack list --no-log
uv run lisa once "/pack install requests" --no-log
uv run lisa once "/pack list" --no-log
```

Then test a real Git pack install with your own repo ref.

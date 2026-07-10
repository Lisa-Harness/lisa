# Little Lisa Pack Manager: Try Guide

This guide lets you exercise the new runtime pack capabilities in a local dev loop.

## What You Can Try

- Runtime pack list/install/uninstall commands.
- Package trust policy enforcement (prefix + Git host allowlist).
- Git-based pack refs (not just package-index refs).
- Auto-restart after pack lifecycle changes.
- Optional resume message after restart.
- Pack-contributed MCP server coexistence and host exclusions.

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

List runtime-installed packs metadata:

```powershell
uv run little-lisa packs list --no-log
```

From chat/REPL, same operation:

```powershell
uv run little-lisa once "/pack list" --no-log
```

## 2) Trust Policy Checks

By default, allowed package prefixes are `lisa-pack-` and allowed Git host is `github.com`.

Expected rejection (wrong package prefix):

```powershell
uv run little-lisa once "/pack install requests" --no-log
```

Expected rejection (wrong Git host):

```powershell
uv run little-lisa once "/pack install git+https://evil.example.com/acme/lisa-pack-cloudflare.git@main#egg=lisa-pack-cloudflare" --no-log
```

## 3) Configure Policy via Env Vars

Allow custom package prefixes:

```powershell
$env:LITTLE_LISA_PACK_ALLOWED_PREFIXES = "lisa-pack-,acme-pack-"
```

Allow additional Git hosts:

```powershell
$env:LITTLE_LISA_PACK_GIT_ALLOW_HOSTS = "github.com,dev.azure.com"
```

Require pinned Git refs (`@tag`, `@branch`, or `@sha`) (default: true):

```powershell
$env:LITTLE_LISA_PACK_REQUIRE_GIT_REF = "true"
```

Disable auto-restart after install/uninstall:

```powershell
$env:LITTLE_LISA_AUTO_RESTART = "false"
```

## 4) Install From Git

Example shape (replace with your real repo/ref):

```powershell
uv run little-lisa packs install "git+https://github.com/acme/lisa-pack-cloudflare.git@v0.1.0#egg=lisa-pack-cloudflare" --no-log
```

From REPL/host command surface:

```powershell
uv run little-lisa once "/pack install git+https://github.com/acme/lisa-pack-cloudflare.git@v0.1.0#egg=lisa-pack-cloudflare" --no-log
```

## 4b) Install From Local Folder (Split-Repos Workflow)

Local folder installs are supported so you can iterate on packs before publishing.

Example learning pack path:

```powershell
uv run little-lisa packs install "C:\Code\lisa-pack-learning" --no-log
```

Example cloudflare pack path:

```powershell
uv run little-lisa packs install "C:\Code\lisa-pack-cloudflare" --no-log
```

Requirements for local folder install:

- Folder must exist.
- `pyproject.toml` must include `project.name`.
- `project.name` must match allowed prefixes policy (default `lisa-pack-`).

## 4c) Default Mode Replacement By Install (No Core Edits)

Little Lisa host policy can prefer a pack-provided `mode_id: default` when that
pack is installed. This is configured in [hosts/little_lisa/mcp_servers.yaml](hosts/little_lisa/mcp_servers.yaml)
under:

```yaml
pack_policy:
  preferred_mode_providers:
    default:
      - lisa-pack-learning
      - learning
```

Result:

- If `lisa-pack-learning` is installed/enabled, its `default` mode is selected.
- If it is not installed, core `default` remains active.
- End users do not need to edit core pack files to transition.

## 5) Resume a Request After Restart

You can attach a resume message to install/uninstall in REPL command form:

```plaintext
/pack install git+https://github.com/acme/lisa-pack-cloudflare.git@v0.1.0#egg=lisa-pack-cloudflare --resume check cloudflare zone status for my workspace
```

On next process start, Little Lisa consumes that message and runs it as the first turn.

## 6) Uninstall a Pack

```powershell
uv run little-lisa packs uninstall lisa-pack-cloudflare --no-log
```

## 7) Inspect Installed Pack Metadata

Runtime metadata is persisted at:

```plaintext
<data-dir>\installed-packs.json
```

Default data-dir:

```plaintext
C:\Code\lisa_framework\hosts\little_lisa\artifacts
```

## 8) MCP Pack Server Coexistence + Exclusions

Pack-contributed MCP servers are compiled with runtime namespace:

```plaintext
<pack_slug>.<server_id>
```

This allows multiple packs with the same declared `server_id` to co-exist.

To exclude specific pack-contributed server instances, edit:

```plaintext
hosts/little_lisa/mcp_servers.yaml
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
uv run little-lisa packs list --no-log
uv run little-lisa once "/pack install requests" --no-log
uv run little-lisa once "/pack list" --no-log
```

Then test a real Git pack install with your own repo ref.


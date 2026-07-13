# lisa-pack-gmail

Gmail draft creation for [Lisa](https://github.com/TheMigusGroup/lisa-framework) — a self-contained content pack that drafts emails from natural-language requests and saves them as Gmail drafts via the Gmail REST API.

> **Draft-only by design.** This pack never sends email. It creates drafts the user reviews and sends from Gmail's UI. Sending is a separate, deliberate act.

## What it provides

| Layer | Name | Kind | Purpose |
| --- | --- | --- | --- |
| Mode | `gmail` | action | `/gmail <request>` — one-shot draft-and-save command |
| Operation | `gmail_draft` | runtime | Drafts the body via the model, then calls the tool |
| Tool | `gmail_create_draft` | runtime | Calls Gmail `drafts.create` with a base64url RFC 2822 message |

The pack is discovered via the `lisa.extensions.content_roots` entry point and reads configuration from environment variables only — it never imports `sdk` or `kernel` config, so it stays self-contained for externalization.

## Install

### As a workspace dependency

Add to the framework workspace `pyproject.toml`:

```toml
dependencies = [
    # ...existing deps...
    "lisa-pack-gmail",
]
```

Then reinstall the workspace:

```powershell
uv sync
```

### For local development (no publish)

Point `SDK_CONTENT_ROOTS` at the pack folder — the CLI host discovers it without a wheel:

```powershell
$env:SDK_CONTENT_ROOTS = "C:\Code\lisa_framework\lisa_pack_gmail"
uv run lisa modes
```

## Configure

The pack reads a single required environment variable:

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `GMAIL_TOKEN` | yes | — | OAuth2 access token with `gmail.compose` scope |
| `GMAIL_API_BASE` | no | `https://gmail.googleapis.com/gmail/v1` | Gmail REST API base URL |

The pack owns no token acquisition. The host (or your environment) must supply a valid OAuth2 access token. For local dev, use the Gmail API OAuth playground to mint a short-lived token; for production, wire a refresh-token flow outside the pack and inject the access token via `GMAIL_TOKEN`.

Example:

```powershell
$env:GMAIL_TOKEN = "<your OAuth2 access token>"
$env:GMAIL_API_BASE = "https://gmail.googleapis.com/gmail/v1"
```

## Use

```powershell
uv run lisa once "/gmail draft a Q3 update to Sam"
```

Flow:

1. The front-door router matches `gmail draft` and routes to the `gmail` action mode.
2. The mode's entry operation `gmail_draft` runs.
3. The operation calls the model to produce structured draft fields (`to`, `subject`, `body`) from the user's request.
4. The operation calls the `gmail_create_draft` tool, which POSTs a base64url-encoded RFC 2822 message to `POST /users/me/drafts`.
5. The tool returns a `draft_id`; the user reviews and sends from Gmail's UI.

If Gmail is not configured, the tool returns a clear error naming the missing `GMAIL_TOKEN` variable.

## Project layout

```plaintext
lisa_pack_gmail/
├── __init__.py                 # get_content_root() entry point export
├── pyproject.toml              # packaging + entry point declaration
├── config.py                   # environment-backed settings
├── models.py                   # Pydantic models (GmailDraftResult)
├── tools/
│   └── gmail_create_draft/
│       ├── __init__.py
│       ├── gmail_create_draft.py        # async execute(ctx) -> ToolResult
│       └── gmail_create_draft.tool.yml  # tool metadata
├── operations/
│   └── gmail_draft/
│       ├── __init__.py
│       ├── gmail_draft.py               # async execute(ctx) -> OperationResult
│       └── gmail_draft.yml              # operation metadata
├── modes/
│   └── gmail/
│       └── _mode.md                    # mode front-matter + instructions
├── content/
│   └── prompts/
│       └── drafting.md                 # authored drafting guidance
├── tests/
│   ├── __init__.py
│   └── test_catalog.py                 # import + metadata contract tests
└── README.md                           # this file
```

## Test

```powershell
uv run pytest lisa_pack_gmail/tests -v
```

The catalog tests are deterministic evals (no model calls, no I/O). They verify the pack imports cleanly, its content root resolves, and its tool/operation/mode metadata files are well-formed. A live test marker (`@pytest.mark.live`) is reserved for tests that hit the real Gmail API with a configured token.

## Boundaries

This pack follows the framework's pack contract:

- **No `sdk` or `kernel` config imports.** Configuration is environment-only via `config.py`.
- **No `hosts` imports.** The pack depends only on `kernel.models` contracts (`ToolExecutionContext`, `ToolResult`, `OperationExecutionContext`, `OperationResult`) and `httpx`.
- **No background schedulers.** All work is pressure-driven on the turn cadence.
- **No compat shims.** Hard cutover patterns only — no dual-path code.
- **Draft-only.** The tool calls `drafts.create`, never `drafts.send`. Sending is a future, separate pack (or a manual user step).

## License

MIT — same as the Lisa framework. See [LICENSE](../LICENSE).

## Status

Experimental. The pack is functionally complete but the OAuth flow is out of scope — you supply the token. A refresh-token helper and a `gmail_send_draft` companion tool are planned as separate, opt-in additions.

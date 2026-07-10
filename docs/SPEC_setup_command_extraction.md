# SPEC: _setup_command Step Extraction

## Status

**Complete** — 2026-07-08. Decomposed into 5 provider step functions with dispatch table. C901=19 → resolved. Zero test regressions.

## Problem

`_setup_command` in `main.py` is an interactive CLI wizard with C901=19
(budget: ≤ 10). It handles provider selection, credential collection,
validation, and `.env` file writing — all in one function body with deeply
nested if/elif chains for 5 provider types.

## Goal

Extract each provider setup step into a focused function and use a
provider-step dispatch table, mirroring the `handle_message` command
dispatch pattern.

## Design

### Provider step registry

```python
type ProviderSetupStep = Callable[
    [Path, Console],
    Awaitable[tuple[bool, str]],
]

_PROVIDER_SETUP_STEPS: dict[str, ProviderSetupStep] = {
    "ollama": _setup_ollama_step,
    "openai": _setup_openai_step,
    "azure_openai": _setup_azure_step,
    "anthropic": _setup_anthropic_step,
    "google": _setup_google_step,
}
```

Each step function:

1. Prompts for credentials via `console.input`.
2. Validates credentials by calling the provider's validation helper.
3. Returns `(success, message)`.

### Extracted functions

| Function | Responsibility |
| --- | --- |
| `_setup_ollama_step` | Ollama validation and env writing. |
| `_setup_openai_step` | OpenAI key collection and validation. |
| `_setup_azure_step` | Azure endpoint + key collection and validation. |
| `_setup_anthropic_step` | Anthropic key collection and validation. |
| `_setup_google_step` | Google key collection and validation. |
| `_setup_select_provider` | Display provider menu, return selection. |
| `_setup_write_env` | Write env updates to `.env` file (already exists). |

### Refactored _setup_command

```python
@app.command("setup", help="Interactive wizard to configure Little Lisa providers.")
async def _setup_command() -> None:
    """Interactive setup wizard for provider configuration."""
    env_path = _LITTLE_LISA_ENV_FILE
    console.print("[bold]Little Lisa Setup Wizard[/bold]\n")

    selection = await _setup_select_provider()
    if selection is None:
        console.print("[yellow]Setup cancelled.[/yellow]")
        return

    provider_id, provider_label = selection
    step = _PROVIDER_SETUP_STEPS.get(provider_id)
    if step is None:
        console.print(f"[red]Unknown provider: {provider_label}[/red]")
        return

    success, message = await step(env_path, console)
    if success:
        console.print(f"[green]{message}[/green]")
    else:
        console.print(f"[red]{message}[/red]")
```

## Test seam preservation

- `_setup_command` is a Typer command — it stays on `main.py`'s surface.
- Tests don't currently test `_setup_command` directly (it's interactive).
- The extracted step functions can be unit-tested with mocked `console.input`.

## Verification

1. `ruff check hosts/little_lisa/main.py --select C901` — zero violations
   for `_setup_command`.
2. `pytest hosts/little_lisa/ -m "not live" -p no:randomly` — 99 passed,
   7 pre-existing failures unchanged.
3. Manual: `little-lisa setup` produces the same menu and flow as before.

## Out of scope

- Changing the provider list or validation logic.
- Adding new providers.
- Making the wizard non-interactive.


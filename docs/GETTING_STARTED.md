# Getting started

Lisa Harness is the framework. The terminal application is the separate
[lisa-cli project](https://github.com/Lisa-Harness/lisa-cli). Choose the path
below that matches your goal.

## Prerequisites

- Python 3.11 or newer and [uv](https://docs.astral.sh/uv/getting-started/installation/).
- Git for working from source.
- For live use: access to a supported model provider, a credential where
  required, and a model with the capabilities your chosen mode needs.
- For additional packs: their dependencies, credentials, and platform
  requirements. A Linux bubblewrap integration is not portable merely
  because framework tests pass on Windows and macOS.

## Run the terminal host

Install the CLI into its own tool environment:

```sh
uv tool install lisa-cli
lisa --help
```

This is not installed by syncing the framework workspace. Follow the CLI's
[installation instructions](https://github.com/Lisa-Harness/lisa-cli#install)
if using its source installer or a local framework checkout.

1. Configure one provider in the environment used by the CLI. For example,
   set `LISA_OPENAI_API_KEY` for OpenAI or `LISA_OLLAMA_HOST` for an Ollama
   endpoint. Do not put credentials in a session file or commit them.
2. Choose a data directory. The runtime default is `~/.lisa`; setting
   `LISA_DATA_DIR` explicitly overrides it. The CLI owns loading any `.env`
   file. See [configuration](CONFIGURATION.md).
3. Discover and review models, then activate the proposal:

   ```sh
   lisa models discover
   lisa models apply
   lisa models
   lisa verify
   lisa modes
   ```

4. Make a live turn:

   ```sh
   lisa prompt "Explain what capabilities are available in this session."
   ```

Discovery can contact providers. A live turn sends instructions, conversation,
and admitted evidence to the selected provider and can incur charges.
`verify` is a setup diagnostic, not proof that a model or pack workflow works.

The active model manifest normally lives at `<data-dir>/config/models.yaml`;
proposals live at `<data-dir>/state/models.proposed.yaml`. A session can
select a different local manifest.

Host control commands and pack commands are separate from model prompts.
Use the CLI's `--help` for its current command spelling and exit codes;
use runtime `/help` for commands registered in the active session.
Do not assume an external `/search` or `/markdown` command is bundled.

## Develop the framework offline

From a source checkout:

```sh
git clone https://github.com/Lisa-Harness/lisa.git
cd lisa
uv sync --frozen --extra dev
uv run pytest packstest/tests -m "not live" -q
```

This quick check requires no provider credentials or model calls. It verifies
the pack-test harness, not the whole framework. Follow [testing](TESTING.md)
for the full CI gates and package-specific checks.

For source-based pack validation and runnable examples, see
[packstest](../packstest/README.md), including its
[source-layout caveat](../packstest/README.md#source-layout-packs).
For runtime integration, see [host integration](HOST_INTEGRATION.md).

## Next steps

- [Author a session](SESSION_CONTEXT_AUTHORING_GUIDE.md) to select packs,
  assign duplicate capability owners, and define context admission.
- [Write a pack](PACK_AUTHORING_GUIDE.md) rather than adding application
  behavior to the kernel.
- Read [approval policy](POLICY_GUIDE.md) before enabling effectful tools.
- Use [troubleshooting](TROUBLESHOOTING.md) if startup or a turn fails.

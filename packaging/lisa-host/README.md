# Lisa Host

Lisa is an agent framework with a terminal host, desktop web UI with system tray, pack manager, and MCP-first architecture.

## Quick start

```powershell
uv tool install lisa-host
lisa verify   # check your environment
lisa          # launch the interactive terminal REPL
lisa desktop  # launch the web UI + system tray
```

The `lisa` command gives you the full interactive experience with pack
management built in:

```text
/pack install lisa-pack-cloudflare
/pack list
```

Packs are installed into a dedicated `~/.lisa/packs-venv` and discovered
on the next launch.
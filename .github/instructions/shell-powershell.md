---
name: Shell Quality (PowerShell)
description: "Use when writing, editing, or reviewing PowerShell scripts in the lisa_framework workspace. Enforces OS-matched syntax, parameterized values, and forbidden-command avoidance. Applies to *.ps1."
applyTo: "**/*.ps1"
---

# Shell — PowerShell on Windows

- Match the host OS detected in the session. Do not produce bash syntax on Windows or PowerShell syntax on Linux.
- Parameterize would-be hard-coded values (ports, paths, executables, timeouts, flags). No magic numbers or literal paths inside script bodies — every configurable value gets a `param()` entry with a sensible default.
- Never use forbidden-on-Windows bash commands in PowerShell: `head`, `grep`, `tail -f`, `find`, `rm -rf`, `mkdir -p`, `cp -r`, `mv`, `wc`, `diff`, `xargs`, `which`. Use the PowerShell / `rg` equivalents documented in the user's shell preferences.

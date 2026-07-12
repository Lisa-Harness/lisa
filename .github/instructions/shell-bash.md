---
name: Shell Quality (bash)
description: "Use when writing, editing, or reviewing bash shell scripts in the lisa_framework workspace. Enforces OS-matched syntax and parameterized values. Applies to *.sh."
applyTo: "**/*.sh"
---

# Shell — bash on Linux/macOS

- Match the host OS detected in the session. Do not produce bash syntax on Windows or PowerShell syntax on Linux.
- Parameterize would-be hard-coded values (ports, paths, executables, timeouts, flags). No magic numbers or literal paths inside script bodies — every configurable value gets a `param()` entry with a sensible default.

# Docforge Usage

Usage: `/docforge <natural language instruction> [--document-path docs/name.md]`

The operation recalls relevant memory, synthesizes a markdown document from your instruction and the recalled context, validates its structure with the markdown-it-py outline tool, and stores it via the lisa-storage `write_file` tool.

Example: `/docforge write a runbook for restarting a stuck VM from the team's past incidents`

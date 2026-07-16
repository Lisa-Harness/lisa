# Markdown Usage

Usage: `/markdown <natural language instruction>`

The operation recalls relevant memory, synthesizes a markdown document from your instruction and the recalled context, validates its structure with the markdown-it-py outline tool, and stores it via the lisa-storage `write_file` tool.

To iterate on an existing document, pass `document_path` and `existing_document` (the current text). When `existing_document` is omitted, the operation reads the current document at `document_path` from storage automatically.

Example: `/markdown write a runbook for restarting a stuck VM from the team's past incidents`

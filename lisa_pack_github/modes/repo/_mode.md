---
mode_id: repo
mode_provider_id: lisa-pack-github
start_stage_id: none
mode_kind: action
routable: true
priority: 4
fast_path_patterns:
  - "\\brepo(?:sitory)?\\s+(?:search|browse|navigate|inspect|read)\\b"
  - "\\b(?:where|which)\\s+(?:file|module|function|class)\\b"
route_hints:
  - Prefer `repo` for read-only questions about the configured GitHub repository.
entry_operation_ids: [navigate_repo]
command_pattern: "^(?:/)?repo(?:[\\s,:;]+(?P<query>.+))?$"
description: "Read-only navigation of the configured GitHub repository with target-ref-grounded evidence."
---

# Repository Navigation

Use only the repository tools to answer questions about the configured GitHub
repository. Search can identify default-branch candidates; use exact lookup or
a file read before making claims about a requested Git ref. State uncertainty
plainly when the available evidence is partial.

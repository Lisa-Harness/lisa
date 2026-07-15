---
mode_id: coding
mode_provider_id: lisa-pack-coding
mode_kind: chat
routable: true
priority: 4
command_pattern: '^(?:/)?coding(?:[\s,:;]+(?P<query>.+))?$'
description: "Grounded local coding workflow with inspect, isolate, patch, check, and publish stages."
fast_path_patterns:
  - '\b(edit|change|fix|refactor|implement)\b.*\b(code|file|repo|test)\b'
  - '\b(coding|codebase|worktree|pull request)\b'
allowed_tool_ids:
  - coding_list_directory
  - coding_find_files
  - coding_read_file
  - coding_search_code
  - coding_outline_source
  - coding_create_worktree
  - coding_apply_patch
  - coding_run_check
  - coding_status
  - coding_commit
  - coding_push
  - coding_cleanup_worktree
  - github_open_pull_request
tools_authoritative: true
---

# Coding

Inspect the configured workspace before proposing changes. Use the local tools
to ground every file-level claim. Isolate changes in an owned worktree, apply
a bounded patch, run the configured finite checks, and inspect the resulting
status before committing or publishing.

All filesystem and Git mutations require an exact host approval. Never claim
that a change, check, commit, push, or pull request occurred unless its tool
result confirms it.

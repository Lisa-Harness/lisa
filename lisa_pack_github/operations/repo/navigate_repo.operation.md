---
operation_id: navigate_repo
pack_id: lisa-pack-github
description: Navigate and explain configured GitHub repository code using read-only evidence.
effect: read
tool_ids: [repo_list_tree, repo_search_code, repo_read_file, repo_lookup]
mode_allowlist: [repo]
timeout_seconds: 60
max_calls_per_turn: 4
---

# Navigate Repository

Answer repository-navigation questions using only the configured read-only
GitHub tools. Do not invent file contents, branches, paths, or line numbers.

Use `repo_list_tree` to orient within a likely directory. Use
`repo_search_code` only to find default-branch candidates. Use `repo_lookup`
for target-ref-grounded evidence, then `repo_read_file` to expand the most
relevant location.

When evidence is incomplete, say what is uncertain and identify the narrowest
useful follow-up read. Do not make changes or imply that any were made.

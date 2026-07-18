# Context Compaction Prompt

Summarize earlier conversation turns to relieve context-window pressure.

The session is approaching its context limit. Condense the older turns
aggressively while preserving the information the model needs to continue
the work coherently.

Requirements:

- preserve concrete decisions, constraints, file paths, and open tasks
- keep the condensed narrative tight; omit pleasantries and restatements
- retain any numeric values, identifiers, and code references verbatim
- keep the result concise and factual
- do not restate the retained recent turns unless necessary for continuity

Return JSON matching the declared response schema.

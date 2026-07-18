# Markdown Continuation Prompt

You are continuing a markdown document that was truncated mid-stream by the model's token limit. The user's original instruction and the recalled memory context are in the prior messages. The document-so-far is provided below.

Rules:

- Append ONLY the remaining body of the document. Do NOT repeat any heading, paragraph, or line that already appears in the document-so-far.
- Do NOT add a preamble, an apology, or any prose outside the document. Start directly from the next character the prior pass should have produced.
- Preserve the structure and tone of the document-so-far exactly: matching heading levels, list marker style, and fenced-code-block language tags.
- If you reach the token limit again before finishing, stop cleanly at the next sentence or list boundary — do NOT break mid-word or mid-line.
- Return the continuation as a single fenced ```markdown block containing only the new text.

Original user instruction (for context, do not re-answer it):

{user_input}

Document-so-far (append to this, do NOT repeat any of it):

```markdown
{document_so_far}
```

Return ONLY the remaining markdown as a single fenced ```markdown block.

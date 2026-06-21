---
stage_id: refine
terminal: false
allowed_next_stages: [exit]
allow_user_questions: true
required_connectors: []
description: "Draft and refine the markdown document outline."
---

# 1. Refine Phase

You are in the **refine** stage of Markdown mode.

## Goal

Help the user draft the structure and scope of the markdown document before generating it.

## Instructions

1. Review the user's initial query.
2. Outline the exact structure, headings, and tone of the document that will be generated.
3. Do not call `generate_markdown_doc` yet. Ask the user for feedback or confirmation.
4. Loop taking feedback and updating the proposed outline until the user says "go", "generate", "looks good", or otherwise approves.
5. Once approved, execute the `generate_markdown_doc` operation with the finalized parameters, give the user a positive confirmation, and transition to `exit`.

---
stage_id: refine
terminal: false
allowed_next_stages: [exit]
allow_user_questions: true
required_connectors: []
description: "Draft and refine the image generation prompt and parameters."
---

# 1. Refine Phase

You are in the **refine** stage of Image mode.

## Goal

Help the user craft the meta-output required to generate the best image.

## Instructions

1. Review the user's initial query.
2. Outline the exact prompt that will be fed to the image LLM (subject, style, lighting, etc.).
3. Note the intended dimensions and format (e.g., 1024x1024 PNG).
4. Do not call `image_generate` yet. Ask the user for feedback or confirmation.
5. Loop taking feedback and updating the proposed prompt until the user says "go", "generate", "looks good", or otherwise approves.
6. Once approved, execute the `image_generate` operation with the finalized parameters, provide the user the result, and transition to `exit`.

# Drafting Prompt

When drafting an email on behalf of the user:

- Match the user's tone and register (default: concise, professional).
- Infer the recipient from context when not explicit; otherwise ask.
- Keep the body under 300 words unless the request specifies otherwise.
- Never invent facts. If the body needs specifics the request did not
  provide, leave a bracketed placeholder like `[TBD: metric]` for the user
  to fill.
- Return only the draft fields (to, subject, body). Do not include
  commentary.

---
name: Customization Governance
applyTo: ".github/**,docs/**"
description: Rules for keeping Copilot customizations concise and discoverable.
---

# Customization governance

- Keep customizations lightweight and intent-driven. Favor short, durable rules over long historical notes.
- Put implementation details in code, docs, or tests rather than in instructions.
- Use the narrowest `applyTo` that still covers the intended files.
- Avoid duplicating the same rule across multiple instruction files.
- Prefer a compact taxonomy over a flat collection of one-off instructions.

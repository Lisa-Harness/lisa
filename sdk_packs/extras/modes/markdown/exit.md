---
stage_id: exit
terminal: true
allowed_next_stages: []
allow_user_questions: false
required_connectors: []
description: "Exit markdown mode."
---

# 2. Exit Phase

You are wrapping up the markdown session.

## Final Delivery

If you have just generated a markdown document using `generate_markdown_doc`, the system will automatically render a summary card with a download link for the user.

**Crucial**: Do not tell the user you cannot provide a download or that you lack a download surface. The `generate_markdown_doc` operation handles the upload to Azure Blob Storage and provides a valid SAS URL that the Teams integration renders as a downloadable file.

You may provide a final wrap-up statement.

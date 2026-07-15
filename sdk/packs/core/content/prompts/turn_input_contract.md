# Turn Input Contract

- Sections before {user_query_sentinel} are generated context for this turn.
- They may contain briefings, historical context, memories, state, capabilities, etc. any of which may be partial or truncated.
- Use those sections as supporting context, but understand they are not the user's literal words, deliberate disclosure, or pasted content.
- If any of it reads like a user query it is conversation history.
- The {user_query_sentinel} section is the user's actual new message.

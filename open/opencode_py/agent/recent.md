# Recent-session block — the previous session's distilled lines, shown for free

Rendered for the first turn of a new session from the pending capture queue, so
a question asked seconds after launch already has the last session's context.
Not durable memory: the model pass decides later what is worth keeping, and
only that gets stored.

Template with one `{content}` placeholder (the lines). Short and plain — it
must read as background, not as new instructions.

```
From your last session (may be incomplete — durable facts are saved separately):
{content}
```

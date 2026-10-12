# Memory block — saved notes injected into the prompt

Template with one `{content}` placeholder (the bulleted notes). Short,
plain, tells the model these persist across sessions.

The bullets are the notes that best match the CURRENT request (plus any
pinned ones) — not the whole store. So the model must not read absence as
"never noted": `remember list` returns everything, and a matter missing
from the block is worth pulling in that way.

```
Project memory (the notes most relevant to this request, remembered across
sessions — follow these notes; `remember list` shows any others):
{content}
```

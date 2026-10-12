"""Shared helpers for the chat surface.

`reasoning_summary` is used by both ChatView and the message bubble, so it
lives here rather than in either — importing it from chat_view would make the
bubble module circular.
"""

from __future__ import annotations

import re


def reasoning_summary(text: str) -> dict:
    """Mirror opencode's reasoningSummary: extract a bold `**Title**` block."""
    content = str(text).strip()
    match = re.match(r"^\*\*([^*\n]+)\*\*(?:\r?\n\r?\n|$)", content)
    if not match:
        return {"title": None, "body": content}
    return {"title": match.group(1).strip(), "body": content[match.end():].strip()}


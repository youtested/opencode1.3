"""Unified diff create/apply (pure Python).

Used by the edit tool and permission dialogs. Produces/consumes standard
`--- a/<path>` / `+++ b/<path>` unified diffs compatible with `patch`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_HEADER_RE = re.compile(r"^--- a/(.*)$")
_HEADER2_RE = re.compile(r"^\+\+\+ b/(.*)$")


@dataclass
class FileDiff:
    old_path: str
    new_path: str
    hunks: list[list[str]] = field(default_factory=list)

    @property
    def text(self) -> str:
        lines = [f"--- a/{self.old_path}", f"+++ b/{self.new_path}"]
        for hunk in self.hunks:
            lines.extend(hunk)
        return "\n".join(lines) + "\n"




def parse_diff(diff_text: str) -> list[FileDiff]:
    """Parse a unified diff into FileDiff objects."""
    files: list[FileDiff] = []
    current: FileDiff | None = None
    current_hunk: list[str] = []

    def flush_hunk() -> None:
        nonlocal current_hunk
        if current is not None and current_hunk:
            current.hunks.append(current_hunk)
        current_hunk = []

    for line in diff_text.splitlines(keepends=True):
        stripped = line.rstrip("\n")
        if stripped.startswith("--- a/") and current is None:
            flush_hunk()
            current = FileDiff(old_path=stripped[6:], new_path="")
            continue
        if stripped.startswith("+++ b/") and current is not None and current.new_path == "":
            current.new_path = stripped[6:]
            continue
        if stripped.startswith("@@") and current is not None:
            flush_hunk()
        if current is not None:
            current_hunk.append(line)
    flush_hunk()
    if current is not None and current.new_path == "":
        current.new_path = current.old_path
    if current is not None:
        files.append(current)
    return files





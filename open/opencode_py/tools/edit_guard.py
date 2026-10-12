"""One implementation of the write-safety rules, shared by every tool.

edit, write and apply_patch all have to answer the same four questions before
they touch a file:

    1. did the user lock this file on purpose?
    2. would this change make the file stop parsing?
    3. could this file's syntax not be checked (and is the caller told)?
    4. did the checker itself break?

Keeping those answers in one place means a language is never checked two ways,
a message is never improved in one tool and left stale in another, and the
"fail open but never fail silently" rule cannot be honoured in one tool only.

The checking itself lives with the language knowledge (verify.py) and with the
file primitives (write.py); this module is only the policy that joins them.
"""

from __future__ import annotations

import stat
from pathlib import Path

# Set when a checker raised. The guard must fail OPEN (a broken guard must never
# block a legitimate edit) but must never fail SILENTLY, so the fault is put in
# front of the caller instead of being swallowed.
_FAULT: str | None = None


def fault() -> str | None:
    """Why the last check could not run, or None when it ran fine."""
    return _FAULT


def is_read_only(path: Path) -> bool:
    """Whether the file was marked read-only (no write bit set for anybody)."""
    try:
        from .write import is_read_only as _impl
    except Exception:  # pragma: no cover - import cycle guard
        return False
    return _impl(path)


def read_only_message(path: Path) -> str:
    """Human reason to leave a read-only file alone, or "" when it is fine."""
    try:
        if not is_read_only(path):
            return ""
        mode = stat.S_IMODE(path.stat().st_mode)
    except OSError:
        return ""
    return (
        f"file is marked read-only (mode {mode:04o}) — you marked it that way on purpose. "
        f"Remove the flag (chmod u+w) or pass force=true to override"
    )


def syntax_check(path: Path, before: str, after: str) -> tuple[list[str], str]:
    """New syntax errors `after` introduces, plus a note when nothing was checked.

    Comparing the two sides is what makes refusing safe: a file that was ALREADY
    broken stays editable, so the guard can never deadlock a repair. Returns
    ([], "") when the guard is off, the language has no checker, or the file is
    over the size limit — with a note in the second slot so the gap is visible.
    """
    global _FAULT
    try:
        from .verify import EXPENSIVE, MAX_BYTES, kind_for, new_syntax_errors, too_big
        from .write import guard_allow_expensive, guard_enabled

        if not guard_enabled():
            return [], ""
        allow = guard_allow_expensive()
        kind = kind_for(path)
        if kind == "other":
            return [], ""
        if kind in EXPENSIVE and not allow:
            return [], (
                f"{kind} syntax not checked (starting node costs ~0.9s here) — "
                f"set OPENCODE_EDIT_GUARD=all to enable it"
            )
        if too_big(kind, len(after.encode("utf-8", "replace"))):
            return [], (
                f"{kind} syntax not checked (file is over the "
                f"{MAX_BYTES[kind] // 1024} KB limit for this language)"
            )
        problems = new_syntax_errors(path, before, after, allow_expensive=allow)
        _FAULT = None
        return problems, ""
    except Exception as e:
        _FAULT = f"{type(e).__name__}: {e}"
        return [], ""


def refusal_text(path: Path, problems: list[str], *, tool: str = "edit") -> str:
    """The message a caller shows when it refuses to write. One wording for all."""
    try:
        from .verify import kind_for

        kind = kind_for(path)
    except Exception:  # pragma: no cover
        kind = "file"
    lines = [
        f"Refused: this change would break the {kind} syntax of {path}.",
        "The file was NOT changed.",
    ]
    for problem in problems[:5]:
        lines.append(f"  new problem: {problem}")
    if len(problems) > 5:
        lines.append(f"  ...and {len(problems) - 5} more")
    lines.append(
        "Fix the new text and try again, or pass force=true to write it anyway if you "
        "are deliberately adding code that is not valid yet."
    )
    return "\n".join(lines)

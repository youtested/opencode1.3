"""checkpoint tool: the agent-facing face of the per-session snapshot store.

There is exactly ONE store — ``opencode_py/snapshots.py`` — and this module is
only the tool wrapper around it. An earlier version kept its own global
15-slot manifest, which meant two competing safety nets where the worse one
(blind to dotfiles, shared across projects, and deleting the OLDEST snapshot
when asked for the latest) was the one the agent was told to call.

Restore comes in two sizes, and the agent should prefer the small one:

* ``step``   — undo the single most recent recorded change. Cheap, targeted,
               keeps everything else exactly as it is. Use this first.
* ``rewind`` — put the WHOLE project back to the state before this session's
               first edit. Use only when stepping back is not enough.

Both are refused rather than guessed at when they are not safe: a rewind into a
different project, or from a capture that was truncated, will not run.
"""

from __future__ import annotations

import base64
import os
import shutil
from pathlib import Path
from typing import Any

from .. import snapshots as _snaps
from ..globals import resolve_worktree
from ..util.log import swallow
from .registry import Tool, schema_with

VERSION = 2
MAX_DIFF_ENTRIES = 80


def _context() -> tuple[str, Path]:
    """The session and project this call is about, from the thread-local the
    agent loop installs before every tool call. There is deliberately no
    override: a restore must act on the project the session is actually in,
    and a path argument is exactly how one would target the wrong tree."""
    sid, root, _title = _snaps.current()
    return sid, resolve_worktree(Path(root or os.getcwd()))


def _out(text: str, **meta: Any) -> dict[str, Any]:
    return {"output": text, "metadata": meta}


# --------------------------------------------------------------------- actions


def _action_step(session_id: str, root: Path) -> dict[str, Any]:
    res = _snaps.step_change(session_id, root)
    if not res.get("ok"):
        return _out(
            f"Nothing to step back: {res.get('reason') or 'no recorded changes'}.\n"
            "If the whole session went wrong, use action=rewind instead.",
            ok=False,
        )
    verb = "Restored" if res.get("kind") == "restored" else "Deleted (it did not exist before)"
    return _out(f"{verb}: {res['path']}", ok=True, path=res.get("path"), kind=res.get("kind"))


def _action_rewind(session_id: str, root: Path, deletions: bool, force: bool) -> dict[str, Any]:
    res = _snaps.rewind(session_id, root, deletions=deletions, force=force)
    if not res.get("ok"):
        return _out(f"Refusing to rewind: {res.get('reason')}", ok=False)
    lines = [
        f"Rewound {root} to the state before this session's first edit.",
        f"Restored {res['restored']} file(s).",
    ]
    if deletions:
        lines.append(f"Removed {res['removed']} file(s) that did not exist then.")
    else:
        lines.append("Files created during the session were left alone "
                     "(pass deletions=true to remove them too).")
    lines.append("A safety copy of the state before this rewind was kept — "
                 "rewind it back to undo the undo.")
    if res.get("failed"):
        lines.append(f"Could not restore {len(res['failed'])}: {', '.join(res['failed'])}")
    meta = {k: res[k] for k in ("restored", "removed", "failed")}
    return _out("\n".join(lines), ok=True, **meta)


def _action_take(session_id: str, root: Path) -> dict[str, Any]:
    res = _snaps.ensure_full(session_id, root, _snaps.current()[2])
    if res == "failed":
        return _out("Could not capture a snapshot for this session.", ok=False)
    if res == "exists":
        return _out(
            f"This session already has its starting snapshot of {root}. "
            "It is taken automatically before the first edit, so you rarely "
            "need this. Use action=list to see what is stored.",
            ok=True, created=False,
        )
    return _out(
        f"Captured a full snapshot of {root} for this session. "
        "Restore it with action=rewind.",
        ok=True, created=True,
    )


def _action_list(session_id: str) -> dict[str, Any]:
    rep = _snaps.report()
    if not rep["count"]:
        return _out("No snapshots stored yet. One is taken automatically before "
                    "a session's first edit.", ok=True, **rep)
    lines = [f"{rep['count']} snapshot folder(s), {rep['bytes'] / 1048576:.1f} MB total:", ""]
    for s in rep["sessions"]:
        tag = "DELETE" if s["delete"] else "keep"
        lines.append(
            f"  {s['id'][:30]:<32} {s['bytes'] / 1024:8.0f} KB  {s['age_days']:7.2f}d  {tag}"
        )
    if rep["delete"]:
        lines += ["", f"{len(rep['delete'])} folder(s) are old enough for /cleanup to remove."]
    lines += ["", "Restore: action=step (one change) or action=rewind (whole project)."]
    return _out("\n".join(lines), ok=True, **rep)


def _action_diff(session_id: str, root: Path) -> dict[str, Any]:
    payload = _snaps.read_full(session_id)
    if payload is None:
        return _out("This session has no starting snapshot to compare against.", ok=False)
    base = Path(str(payload.get("root") or root))
    stored = payload.get("files") or {}
    modified: list[str] = []
    added: list[str] = []
    deleted: list[str] = []
    for rel in sorted(stored):
        target = base / rel
        try:
            now = target.read_bytes()
        except OSError as e:
            swallow(e, "opencode_py/tools/checkpoint.py:_action_diff")
            deleted.append(rel)
            continue
        try:
            if now != base64.b64decode(str(stored[rel])):
                modified.append(rel)
        except ValueError as e:
            swallow(e, "opencode_py/tools/checkpoint.py:_action_diff")
            modified.append(rel)
    for path in _snaps._current_files(base):
        rel = path.relative_to(base).as_posix()
        if rel not in stored:
            added.append(rel)

    def block(title: str, items: list[str]) -> list[str]:
        if not items:
            return []
        out = [f"• {title} ({len(items)})"]
        out += [f"  {r}" for r in items[:MAX_DIFF_ENTRIES]]
        if len(items) > MAX_DIFF_ENTRIES:
            out.append(f"  ... and {len(items) - MAX_DIFF_ENTRIES} more")
        return out

    lines = block("modified", modified) + block("added", added) + block("deleted", deleted)
    if not lines:
        lines = ["Nothing has changed since this session's starting snapshot."]
    return _out("\n".join(lines), ok=True, modified=modified, added=added, deleted=deleted)


def _action_drop(target_id: str) -> dict[str, Any]:
    """Drop one named snapshot folder. ``id`` is REQUIRED.

    The old version defaulted a missing id to the first entry, so "delete the
    latest" silently destroyed the OLDEST safety net instead. Requiring the id
    removes the ambiguity rather than inventing a new default for it.
    """
    wanted = str(target_id or "").strip()
    if not wanted:
        return _out(
            "action=drop needs an explicit id — run action=list and pass the one "
            "you mean. (Refusing to guess: the old default deleted your oldest "
            "snapshot.)",
            ok=False,
        )
    known = {s["id"]: s for s in _snaps.report()["sessions"]}
    folder = _snaps.session_dir(wanted)
    if folder.exists() and not _snaps._owns(folder):
        return _out(
            f"Refusing: {wanted} exists but has no marker file, so it is not "
            "provably ours to delete.",
            ok=False,
        )
    if wanted not in known:
        near = [i for i in known if wanted in i][:5]
        hint = f" Closest: {', '.join(near)}." if near else ""
        return _out(f"No snapshot folder matches id {wanted!r}.{hint}", ok=False)
    try:
        shutil.rmtree(folder)
    except OSError as e:
        swallow(e, "opencode_py/tools/checkpoint.py:_action_drop")
        return _out(f"Could not delete {wanted}: {e}", ok=False)
    return _out(f"Dropped snapshot folder {wanted}.", ok=True, id=wanted)


# ------------------------------------------------------------------ the tool

_ACTIONS = ("take", "step", "rewind", "list", "diff", "drop")


def tool() -> Tool:
    description = """Snapshot and restore safety net, scoped to this session.

The app captures your whole project automatically before the session's first
edit, and records each file's contents before every later edit. You do not
normally need to call this.

Restoring comes in two sizes — PREFER THE SMALL ONE:
- action=step: undo the single most recent change. Try this FIRST. If it puts
  the file back you are done, and nothing else is disturbed.
- action=rewind: put the WHOLE project back to the start of the session. Use
  ONLY when step cannot fix it, because it touches every file. Add
  deletions=true to also remove files created during the session.

Also: list (what is stored), diff (what changed since the session started),
take (rarely needed, it happens automatically), drop (delete one named folder
from list — the id is required)."""

    def run(input: dict[str, Any]) -> dict[str, Any]:
        data = input if isinstance(input, dict) else {}
        action = str(data.get("action") or "list").strip().lower()
        session_id, root = _context()
        if not session_id:
            return _out(
                "No session context for this call, so there is nothing to act on. "
                "This tool only works inside an agent session.",
                ok=False,
            )
        try:
            if action == "step":
                return _action_step(session_id, root)
            if action == "rewind":
                return _action_rewind(
                    session_id, root,
                    bool(data.get("deletions")),
                    bool(data.get("force")),
                )
            if action == "take":
                return _action_take(session_id, root)
            if action == "list":
                return _action_list(session_id)
            if action == "diff":
                return _action_diff(session_id, root)
            if action == "drop":
                return _action_drop(str(data.get("id") or ""))
        except Exception as e:
            swallow(e, "opencode_py/tools/checkpoint.py:run")
            return _out(f"checkpoint {action} failed: {e}", ok=False)
        return _out(f"Unknown action {action!r}. Use one of: {', '.join(_ACTIONS)}", ok=False)

    return Tool(
        name="checkpoint",
        description=description,
        parameters=schema_with(
            {
                "action": {
                    "type": "string",
                    "enum": list(_ACTIONS),
                    "description": "step (small, preferred) / rewind (whole project) / list / diff / take / drop",
                    "optional": True,
                },
                "id": {
                    "type": "string",
                    "description": "Snapshot folder id from action=list. REQUIRED for drop.",
                    "optional": True,
                },
                "deletions": {
                    "type": "boolean",
                    "description": "rewind only: also delete files created during the session",
                    "optional": True,
                },
                "force": {
                    "type": "boolean",
                    "description": "rewind only: proceed even if the snapshot is from another project",
                    "optional": True,
                },
            },
            [],
        ),
        run=run,
        permission="checkpoint",
    )
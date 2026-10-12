"""Per-session project snapshots: one full capture before a session's first
edit, then one small record per change.

Design notes (why it looks like this):

* Stored under its OWN top-level dir (``data/snapshots``), never inside
  ``sessions/``. A delete bug here must not be able to reach real chats.
* Every session folder carries a marker file written LAST. The marker is the
  only proof of ownership, so pruning skips anything without it rather than
  trusting a name pattern. (A loose pattern is exactly how 15 real checkpoints
  got wiped once already.)
* Deletion renames the folder to ``.trash-*`` first — one atomic step — then
  empties it. A crash mid-empty leaves a trash folder that the next startup
  sweeps, so a snapshot is never observed half-deleted.
* The full capture deliberately does NOT reuse ``tools/checkpoint.py``'s
  ``_scan_tree``: that one skips dotfiles, ignore-listed dirs, binaries and big
  files, which are exactly the holes that made it unsafe to restore from. This
  walks everything except ``.git`` (restoring a stale ``.git`` would overwrite
  real commit history).

Nothing here raises: every failure is reported as a return value.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import threading
import time
import zlib
from pathlib import Path
from typing import Any

from .globals import Path as GPath
from .session import _write_durable
from .tools.write import _atomic_write_bytes
from .util.log import swallow

# Sub-agents share the top-level session's full capture (one copy per job), so
# every engine in a job tree writes into the SAME folder — hence the lock.
_LOCK = threading.RLock()

MARKER = ".snapshot"
INFO = "info.json"
START = "start.json.z"
CHANGES = "changes"

# Full-capture ceilings. A project past these is written anyway but flagged
# `truncated`, and a truncated capture REFUSES to be restored rather than
# deleting files it never saw. Same safety rule as tools/checkpoint.py.
MAX_FULL_FILES = 20000
MAX_FULL_BYTES = 64 * 1024 * 1024

MAX_CHANGES = 500
KEEP_FLOOR = 5        # never delete below this many most-recent sessions
KEEP_DAYS = 14        # never delete anything newer than this
ACTIVE_GRACE = 300.0  # seconds; a folder touched this recently is in use

# Compiler output, not source: Python regenerates these on the next run, and a
# STALE .pyc restored from a snapshot is worse than none. They were also the
# single biggest chunk of the capture (see test_full_capture_skips_bytecode).
SKIP_DIRS = {"__pycache__"}
SKIP_SUFFIXES = (".pyc",)


# ----------------------------------------------------------------- locations

# Thread-local "which session am I in", installed by the agent loop before every
# tool call — the same pattern tools/glob.py uses for its worktree. Thread-local
# because mutating tools are serialised per agent but sub-agents run on their
# own worker threads, and a parallel-safe context must not be shared.
_THREAD = threading.local()


def set_session(session_id: str, root: str, title: str = "") -> None:
    _THREAD.session_id = str(session_id or "")
    _THREAD.root = str(root or "")
    _THREAD.title = str(title or "")


def current() -> tuple[str, str, str]:
    """(session_id, root, title) for the calling thread. ("", "", "") if unset."""
    return (
        str(getattr(_THREAD, "session_id", "") or ""),
        str(getattr(_THREAD, "root", "") or ""),
        str(getattr(_THREAD, "title", "") or ""),
    )


def _store_dir() -> Path:
    return GPath.data / "snapshots"


def _safe(session_id: str) -> str:
    """Percent-encode to an injective single path segment (mirrors
    session.py's session_path so a session and its snapshots share one id)."""
    out = "".join(
        c if (c.isascii() and c.isalnum()) or c in "-_" else f"%{ord(c):02X}"
        for c in str(session_id)
    )
    return out or "invalid"


def session_dir(session_id: str) -> Path:
    return _store_dir() / _safe(session_id)


def _owns(d: Path) -> bool:
    """True only for a folder we can PROVE is ours."""
    return d.is_dir() and (d / MARKER).is_file()


# -------------------------------------------------------------- blob helpers


def _write_blob(path: Path, payload: dict[str, Any]) -> bool:
    """Compressed + atomic + fsync'd. Returns True on success, never raises.

    zlib level 1, not 6: on a phone the full capture is dominated by
    compression (measured 2.8s at level 6 for this repo vs 0.9s at level 1), and
    a safety net cares far more about being written promptly than about the last
    few percent of disk.
    """
    try:
        blob = zlib.compress(json.dumps(payload).encode("utf-8"), 1)
    except (TypeError, ValueError) as e:
        swallow(e, "snapshots.py:_write_blob")
        return False
    tmp = path.with_name(path.name + ".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "wb") as fh:
            fh.write(blob)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except OSError as e:
        swallow(e, "snapshots.py:_write_blob")
        return False
    return True


def _read_blob(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(zlib.decompress(path.read_bytes()).decode("utf-8"))
    except (OSError, ValueError, zlib.error) as e:
        swallow(e, "snapshots.py:_read_blob")
        return None
    return data if isinstance(data, dict) else None


# --------------------------------------------------------------- full capture


def _scan_all(root: Path) -> tuple[dict[str, bytes], bool]:
    """Every file under root except .git. Returns (files, truncated)."""
    files: dict[str, bytes] = {}
    total = 0
    truncated = False
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != ".git" and d not in SKIP_DIRS]
        for fn in filenames:
            if fn.endswith(SKIP_SUFFIXES):
                continue
            try:
                data = (Path(dirpath) / fn).read_bytes()
            except OSError as e:
                swallow(e, "snapshots.py:_scan_all")
                continue
            total += len(data)
            files[(Path(dirpath) / fn).relative_to(root).as_posix()] = data
            if len(files) > MAX_FULL_FILES or total > MAX_FULL_BYTES:
                return files, True
    return files, truncated


def ensure_full(session_id: str, root: Path, title: str = "") -> str:
    """Take the whole-project capture ONCE per session, before its first edit.

    Returns "created", "exists", or "failed". Never raises. The marker is
    written last so an interrupted capture leaves an unproven folder that
    pruning skips instead of trusting.
    """
    sid = str(session_id or "").strip()
    if not sid:
        return "failed"
    with _LOCK:
        d = session_dir(sid)
        if _owns(d) and (d / START).is_file():
            return "exists"
        files, truncated = _scan_all(root)
        created = time.time()
        if not _write_blob(
            d / START,
            {
                "root": str(root),
                "title": title,
                "created": created,
                "truncated": truncated,
                "files": {
                    rel: base64.b64encode(data).decode("ascii")
                    for rel, data in files.items()
                },
            },
        ):
            return "failed"
        _write_durable(
            d / INFO,
            json.dumps(
                {
                    "title": title,
                    "root": str(root),
                    "created": created,
                    "truncated": truncated,
                    "files": len(files),
                }
            ),
        )
        _write_durable(d / MARKER, str(created))
        return "created"


def read_full(session_id: str) -> dict[str, Any] | None:
    """The session's full capture, or None."""
    return _read_blob(session_dir(session_id) / START)


# ------------------------------------------------------------ change records


def record_change(
    session_id: str,
    root: Path,
    path: str,
    before: bytes | None,
    dirs: list[str] | None = None,
) -> bool:
    """Persist one file's pre-edit bytes for this session. Never raises.

    ``before is None`` means the file did not exist, so undo deletes it.
    Called once per mutating tool call, for the files that call touched.
    """
    sid = str(session_id or "").strip()
    if not sid or not path:
        return False
    with _LOCK:
        d = session_dir(sid) / CHANGES
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            swallow(e, "snapshots.py:record_change")
            return False
        ok = _write_blob(
            d / f"{time.time_ns():019d}.json.z",
            {
                "root": str(root),
                "path": path,
                "time": time.time(),
                "existed": before is not None,
                "b64": base64.b64encode(before).decode("ascii") if before is not None else "",
                "dirs": [str(d) for d in (dirs or [])],
            },
        )
        _trim_changes(d)
    return ok


def list_changes(session_id: str) -> list[dict[str, Any]]:
    """Change records oldest-first. Unreadable records are skipped, not fatal."""
    d = session_dir(session_id) / CHANGES
    out: list[dict[str, Any]] = []
    try:
        for p in sorted(d.glob("*.json.z")):
            rec = _read_blob(p)
            if rec is not None:
                out.append(rec)
    except OSError as e:
        swallow(e, "snapshots.py:list_changes")
    return out


def _trim_changes(d: Path) -> None:
    """Keep the newest MAX_CHANGES, dropping the oldest. Never raises."""
    try:
        files = sorted(d.glob("*.json.z"))
        for old in files[:-MAX_CHANGES]:
            old.unlink(missing_ok=True)
    except OSError as e:
        swallow(e, "snapshots.py:_trim_changes")


# ----------------------------------------------------------------- restoring


def _under(root: Path, rel: str) -> Path | None:
    """Resolve `rel` inside `root`, or None if it escapes. Stale notes and a
    hand-edited store must never be able to write outside the project."""
    if not rel:
        return None
    try:
        target = (root / rel).resolve()
        target.relative_to(root.resolve())
    except (OSError, ValueError):
        return None
    return target


def _retire(d: Path, note: Path) -> None:
    """Move an applied note out of the pending set. The rename is atomic, so a
    note is either pending or applied — never both, never neither."""
    try:
        done = d.with_name(".done")
        done.mkdir(parents=True, exist_ok=True)
        os.replace(note, done / note.name)
    except OSError as e:
        swallow(e, "snapshots.py:_retire")


def step_change(session_id: str, root: Path) -> dict[str, Any]:
    """Undo the single most recent recorded change. Never raises.

    Prefers the newest note belonging to THIS project. Notes for other projects
    are left alone, and a note whose path escapes the project is skipped.
    """
    sid = str(session_id or "").strip()
    base = {"ok": False, "path": "", "kind": "", "reason": ""}
    if not sid:
        return {**base, "reason": "no session id"}
    rootp = Path(root)
    with _LOCK:
        d = session_dir(sid) / CHANGES
        try:
            notes = sorted(d.glob("*.json.z"), reverse=True)
        except OSError as e:
            swallow(e, "snapshots.py:step_change")
            return {**base, "reason": "no change log"}
        for note in notes:
            rec = _read_blob(note)
            if rec is None:
                _retire(d, note)  # unreadable: drop it rather than block forever
                continue
            if str(rec.get("root") or "") != str(rootp.resolve()):
                continue
            target = _under(rootp, str(rec.get("path") or ""))
            if target is None:
                _retire(d, note)
                continue
            _retire(d, note)
            try:
                if rec.get("existed"):
                    target.parent.mkdir(parents=True, exist_ok=True)
                    _atomic_write_bytes(target, base64.b64decode(str(rec.get("b64") or "")))
                    kind = "restored"
                else:
                    if target.exists():
                        target.unlink()
                    kind = "deleted"
                    for raw in rec.get("dirs") or []:
                        dd = _under(rootp, str(raw))
                        if dd is None or not dd.is_dir():
                            continue
                        try:
                            if not any(dd.iterdir()):
                                dd.rmdir()
                        except OSError as e:
                            swallow(e, "snapshots.py:step_change")
            except (OSError, ValueError) as e:
                swallow(e, "snapshots.py:step_change")
                return {**base, "path": str(target), "reason": f"restore failed: {e}"}
            return {"ok": True, "path": str(target), "kind": kind, "reason": ""}
    return {**base, "reason": "nothing left to step back"}


def _current_files(root: Path) -> list[Path]:
    out: list[Path] = []
    try:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d != ".git" and d not in SKIP_DIRS]
            for fn in filenames:
                if not fn.endswith(SKIP_SUFFIXES):
                    out.append(Path(dirpath) / fn)
    except OSError as e:
        swallow(e, "snapshots.py:_current_files")
    return out


def rewind(
    session_id: str,
    root: Path,
    *,
    deletions: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Restore the whole project to the state captured before this session's
    first edit. Never raises.

    Guards, in order: no capture, different project (unless `force`), and a
    truncated capture — restoring an incomplete one would delete files it never
    saw. A safety copy of the CURRENT state is taken first; if that fails,
    nothing is touched.
    """
    sid = str(session_id or "").strip()
    base = {"ok": False, "restored": 0, "removed": 0, "failed": [], "reason": ""}
    if not sid:
        return {**base, "reason": "no session id"}
    with _LOCK:
        payload = read_full(sid)
        if payload is None:
            return {**base, "reason": "this session has no full snapshot"}
        rootp = Path(root).resolve()
        snap_root = str(payload.get("root") or "")
        if snap_root != str(rootp) and not force:
            return {**base, "reason": (
                f"snapshot belongs to {snap_root or '?'}, you are in {rootp}. "
                "Pass force=true to rewind anyway."
            )}
        if payload.get("truncated"):
            return {**base, "reason": (
                "snapshot is incomplete (too many files/large), so rewinding "
                "could delete files it never captured. Refusing."
            )}
        files = payload.get("files") or {}
        if ensure_full(f"{sid}--rewind-safety", rootp) == "failed":
            return {**base, "reason": "could not take a safety copy; nothing was changed"}

        restored = 0
        failed: list[str] = []
        for rel, b64 in files.items():
            target = _under(rootp, str(rel))
            if target is None:
                failed.append(str(rel))
                continue
            try:
                data = base64.b64decode(str(b64))
                target.parent.mkdir(parents=True, exist_ok=True)
                # durable=False: the rename still makes each file all-or-nothing
                # (a crash leaves the old bytes, never a truncated file), but
                # skipping the fsyncs took a 440-file rewind from 11.8s to well
                # under a second. All-or-nothing ACROSS files is not claimed here
                # — the safety copy taken above is what makes a partial rewind
                # recoverable.
                _atomic_write_bytes(target, data, durable=False)
                restored += 1
            except (OSError, ValueError) as e:
                swallow(e, "snapshots.py:rewind")
                failed.append(str(rel))

        removed = 0
        # Opt-in only: removing files the session created is the destructive
        # half of a rewind, so it never happens unless asked for.
        if deletions and not failed:
            for path in _current_files(rootp):
                rel = path.relative_to(rootp).as_posix()
                if rel in files:
                    continue
                try:
                    path.unlink()
                    removed += 1
                except OSError as e:
                    swallow(e, "snapshots.py:rewind")
    return {"ok": not failed, "restored": restored, "removed": removed,
            "failed": failed[:8], "reason": "" if not failed else "some files could not be restored"}


# ------------------------------------------------------------------- pruning


def _created(d: Path) -> float:
    try:
        return float((d / MARKER).read_text(encoding="utf-8").strip())
    except (OSError, ValueError) as e:
        swallow(e, "snapshots.py:_created")
        try:
            return (d / MARKER).stat().st_mtime
        except OSError as e2:
            swallow(e2, "snapshots.py:_created")
            return 0.0


def _touched(d: Path) -> float:
    """Newest mtime inside the folder — "was anything written here lately?".

    The age policy uses the marker's CREATION stamp, but a session created weeks
    ago that is still being edited keeps that old stamp, so creation time alone
    cannot tell "abandoned" from "running right now". The mtime can.
    """
    newest = 0.0
    try:
        for f in d.rglob("*"):
            try:
                newest = max(newest, f.stat().st_mtime)
            except OSError as e:
                swallow(e, "snapshots.py:_touched")
    except OSError as e:
        swallow(e, "snapshots.py:_touched")
    return newest


def _candidates() -> list[tuple[float, Path]]:
    """Owned session folders, oldest first."""
    out: list[tuple[float, Path]] = []
    try:
        with os.scandir(_store_dir()) as it:
            for e in it:
                if e.name.startswith(".") or not e.is_dir():
                    continue
                p = Path(e.path)
                if _owns(p):
                    out.append((_created(p), p))
    except OSError as e:
        swallow(e, "snapshots.py:_candidates")
    return sorted(out, key=lambda t: t[0])


def report(keep_ids: set[str] | None = None) -> dict[str, Any]:
    """What is stored, and what pruning WOULD remove. Changes nothing.

    ``keep_ids`` are session ids that are open RIGHT NOW (the caller passes the
    live session). They are never proposed for deletion regardless of age: the
    age policy cannot tell an abandoned session from one the user simply left
    open over lunch, and the caller can.
    """
    now = time.time()
    keep = {_safe(s) for s in (keep_ids or set()) if str(s).strip()}
    cands = _candidates()
    total = 0
    sessions: list[dict[str, Any]] = []
    delete: list[str] = []
    for i, (created, p) in enumerate(cands):
        size = 0
        try:
            for f in p.rglob("*"):
                if f.is_file():
                    size += f.stat().st_size
        except OSError as e:
            swallow(e, "snapshots.py:report")
        total += size
        title = ""
        try:
            title = str(json.loads((p / INFO).read_text(encoding="utf-8")).get("title") or "")
        except (OSError, ValueError, AttributeError) as e:
            swallow(e, "snapshots.py:report")
        age_days = (now - created) / 86400.0
        # Delete only when ALL THREE say so — the floor and the age limit are
        # independent guards, so the policy can only ever be too generous.
        if p.name in keep:
            why = None
        elif now - _touched(p) < ACTIVE_GRACE:
            why = None
        elif i >= len(cands) - KEEP_FLOOR:
            why = None
        elif age_days < KEEP_DAYS:
            why = None
        else:
            why = "older than %dd, outside the %d most recent, untouched %ds" % (
                KEEP_DAYS, KEEP_FLOOR, ACTIVE_GRACE,
            )
            delete.append(p.name)
        sessions.append(
            {
                "id": p.name,
                "title": title,
                "bytes": size,
                "age_days": round(age_days, 2),
                "delete": bool(why),
                "reason": why or "kept",
            }
        )
    return {"sessions": sessions, "bytes": total, "delete": delete, "count": len(cands)}


def _drop(d: Path) -> bool:
    """Atomic rename out of the way, then empty. Never raises."""
    trash = d.with_name(f".trash-{d.name}-{int(time.time())}")
    try:
        os.replace(d, trash)
    except OSError as e:
        swallow(e, "snapshots.py:_drop")
        return False
    # ponytail: rmtree is synchronous, so on a huge tree this blocks the
    # caller for a while. Pruning only runs at startup or on demand, never
    # mid-edit; if that ever changes, move it to a worker.
    shutil.rmtree(trash, ignore_errors=True)
    return True


def sweep_trash() -> int:
    """Finish deletions interrupted by a crash. Returns how many were cleared."""
    n = 0
    try:
        with os.scandir(_store_dir()) as it:
            for e in it:
                if e.name.startswith(".trash-"):
                    shutil.rmtree(e.path, ignore_errors=True)
                    n += 1
    except OSError as e:
        swallow(e, "snapshots.py:sweep_trash")
    return n


def prune(keep_ids: set[str] | None = None) -> list[str]:
    """Delete only what report() marked for deletion. Returns what went."""
    with _LOCK:
        gone = []
        for name in report(keep_ids)["delete"]:
            d = _store_dir() / name
            if not _owns(d):  # re-check under the lock, never trust the report
                continue
            if _drop(d):
                gone.append(name)
        return gone
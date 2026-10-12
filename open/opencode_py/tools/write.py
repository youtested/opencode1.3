"""write tool: create a NEW file (edits belong to edit/apply_patch).

RULES (full guidance for maintainers; the sent schema is short):
- Atomic (temp + rename), creates parent dirs.
- Returns stats only (bytes/lines) — never echoes content back (saves 2x data).
- Identical content -> no-op, 0 bytes written.
- Existing file with different content -> error guiding to edit/apply_patch
  (or force=true to blast, dry_run=true to preview size first).
- Never proactively create docs (*.md/README) unless asked.
"""

from __future__ import annotations

from ..util.log import swallow
import os
import stat
import tempfile
from pathlib import Path

from .registry import Tool, schema_with


def guard_enabled() -> bool:
    """Whether the syntax guard refuses edits that would break a file."""
    return os.environ.get("OPENCODE_EDIT_GUARD", "on").strip().lower() != "off"


def guard_allow_expensive() -> bool:
    """Whether to run checkers that must launch another program (node: ~0.9s)."""
    return os.environ.get("OPENCODE_EDIT_GUARD", "on").strip().lower() == "all"


def confirm_write(path: Path, expected: str) -> str | None:
    """Re-read a file and confirm it holds exactly what we meant to write.

    None when it matches, otherwise a message saying what went wrong. Reporting
    success without this is how a truncated or mistargeted write is mistaken
    for a good edit. Shared by every tool that writes.
    """
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            got = fh.read()
    except (OSError, UnicodeError) as e:
        return f"could not read the file back to confirm the write ({e})"
    if got != expected:
        return "the bytes on disk do not match what was written"
    return None


def _existing_mode(path: Path) -> int | None:
    """Permission bits of the file we are about to replace, or None if new.

    ``tempfile.mkstemp`` creates 0600 and ``os.replace`` keeps the NEW file's
    mode, so without this every save silently turned a 0644 file into 0600 and
    stripped the execute bit off a 0755 script. Only the permission bits are
    carried over: setuid/setgid/sticky belong to the old inode and must not be
    inherited by a new one.
    """
    try:
        return stat.S_IMODE(path.stat().st_mode)
    except OSError:
        return None


def is_read_only(path: Path) -> bool:
    """True when nobody has a write bit set — the file was marked read-only.

    Saving goes through a temp file and a rename, and renaming over a file only
    needs write access to the DIRECTORY. So a file the user deliberately locked
    with ``chmod a-w`` was silently edited anyway; the flag was never consulted.
    A file with any write bit set (0644, 0664, 0775 …) is not read-only and is
    not reported as such.
    """
    mode = _existing_mode(path)
    return mode is not None and not mode & 0o222


def _atomic_write_bytes(
    path: Path, data: bytes, durable: bool = True, new_mode: int | None = None
) -> None:
    """Write via temp file + rename, so the target is never left half-written.

    ``durable=False`` still gives crash-atomicity per file (rename is atomic, so
    a crash leaves the OLD contents, not a truncated file) but skips the fsyncs.
    That matters in bulk: restoring a 440-file snapshot with the fsyncs on cost
    11.8s because each file paid two flushes. Callers restoring many files
    should pass durable=False, because the guarantee that actually matters there
    is "no truncated file", and a safety copy covers the all-or-nothing case.

    ``new_mode`` is the mode for a file that does not exist yet. ``mkstemp``
    creates 0600, so without it every NEW file came out owner-only: a fresh
    ``run.sh`` was not executable and not group/other readable until the user
    chmod-ed it by hand. Existing files always keep their own mode, and callers
    that want 0600 for fresh files (snapshot blobs) simply leave it None.
    """
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    mode = _existing_mode(path)
    want = mode if mode is not None else new_mode
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            if durable:
                try:
                    fh.flush()
                    os.fsync(fh.fileno())
                except OSError as _oc_exc:
                    swallow(_oc_exc, "opencode_py/tools/write.py:87")
                    pass
        if want is not None:
            # Do this while the temp file still has our own ownership: after
            # the rename the path may be a file we no longer have the right to
            # chmod, and a best-effort chmod there would fail silently.
            try:
                os.chmod(tmp, want)
            except OSError as _oc_exc:
                swallow(_oc_exc, "opencode_py/tools/write.py:95")
                pass
        os.replace(tmp, path)
        if not durable:
            return
        try:
            dfd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(dfd)
            except OSError as _oc_exc:
                swallow(_oc_exc, "opencode_py/tools/write.py:102")
                pass
            finally:
                try:
                    os.close(dfd)
                except OSError as _oc_exc:
                    swallow(_oc_exc, "opencode_py/tools/write.py:107")
                    pass
        except OSError as _oc_exc:
            swallow(_oc_exc, "opencode_py/tools/write.py:109")
            pass
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError as _oc_exc:
            swallow(_oc_exc, "opencode_py/tools/write.py:114")
            pass
        raise


def _atomic_write(path: Path, content: str) -> None:
    """Text form of :func:`_atomic_write_bytes`.

    The text writer used ``newline=""``, i.e. it never translated line endings,
    so encoding to utf-8 here produces exactly the bytes it used to write.

    ``new_mode=0o644`` so a newly created file matches what any other tool on
    this box would produce, instead of inheriting mkstemp's 0600.
    """
    _atomic_write_bytes(path, content.encode("utf-8"), new_mode=0o644)


def _write(filePath: str, content: str, dry_run: bool = False, force: bool = False) -> dict:
    path = Path(filePath)
    if not path.is_absolute():
        path = Path.cwd() / path
    asked_for = path
    path = path.resolve()
    content = content if isinstance(content, str) else str(content)
    nbytes = len(content.encode("utf-8", errors="replace"))
    nlines = content.count("\n") + (0 if content.endswith("\n") or not content else 1)
    existed = path.exists()
    notes: list[str] = []
    if existed and is_read_only(path):
        # force=true is the DOCUMENTED override (see edit_guard.read_only_message),
        # so do not block it -- but say so, because the file still LOOKS locked
        # afterwards and the user otherwise cannot tell it was edited.
        mode_bits = _existing_mode(path)
        notes.append(
            f"note: {path} was marked read-only (mode {mode_bits:04o}); "
            f"force=true overrode that lock."
        )
    if existed:
        try:
            current = path.read_text(encoding="utf-8", errors="replace")
        except (OSError, UnicodeError) as e:
            return {"output": f"Error reading file {path}: {e}", "error": True}
        if current == content:
            return {
                "output": f"Already up to date ({nbytes} bytes, {nlines} lines) — wrote nothing.",
                "metadata": {"filePath": str(path), "bytes": nbytes, "lines": nlines,
                             "identical": True, "created": False},
            }
        if dry_run:
            return {
                "output": f"Dry run — no changes written. Would overwrite {path} "
                           f"({len(current.encode('utf-8', errors='replace'))}B -> {nbytes}B).",
                "metadata": {"filePath": str(path), "bytes": nbytes, "lines": nlines,
                             "dry_run": True, "created": False},
            }
        if not force:
            return {
                "output": (
                    f"File exists with different content ({path}). "
                    f"Use edit (small change) or apply_patch (multi-file), "
                    f"or retry write with force=true to overwrite "
                    f"({len(current.encode('utf-8', errors='replace'))}B -> {nbytes}B). "
                    f"Dry run first: dry_run=true previews size without writing."
                ),
                "error": True,
                "metadata": {"filePath": str(path), "bytes": nbytes, "lines": nlines},
            }
    if dry_run:
        return {
            "output": f"Dry run — no changes written. Would create {path} ({nbytes}B, {nlines} lines).",
            "metadata": {"filePath": str(path), "bytes": nbytes, "lines": nlines,
                         "dry_run": True, "created": not existed},
        }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if existed and force:
            # Re-check the body right before overwriting: another writer may
            # have changed it since the first read — refuse instead of
            # silently losing their update (TOCTOU guard).
            try:
                again = path.read_text(encoding="utf-8", errors="replace")
                if again != current:
                    return {
                        "output": (
                            f"File changed since read ({path}). Re-read it and "
                            f"retry — refusing to silently overwrite another writer."
                        ),
                        "error": True,
                    }
            except (OSError, UnicodeError) as e:
                return {"output": f"Error re-reading file {path}: {e}", "error": True}
        _atomic_write(path, content)
    except (OSError, UnicodeError) as e:
        return {"output": f"Error writing file {path}: {e}", "error": True}
    if asked_for != path:
        notes.append(
            f"note: you asked for {asked_for}, which resolves to {path} "
            f"(symlink or indirect path) -- that is what was written."
        )
    # Read the file back BEFORE claiming success. edit.py and apply_patch.py both
    # do this; write did not, so it was the one writer that could report
    # "Created"/"Overwrote" for a write that did not actually land.
    unverified = confirm_write(path, content)
    if unverified:
        return {
            "output": f"Wrote {path} but could not confirm it: {unverified}",
            "error": True,
            "metadata": {"filePath": str(path), "bytes": nbytes, "lines": nlines,
                         "created": not existed, "unverified": True},
        }
    # feed the verify tool's homework checker
    from .verify import track

    track(path, "write")
    verb = "Created" if not existed else "Overwrote (force)"
    diag: dict = {}
    try:
        # guard_enabled() is honoured here for the first time. It used to run
        # pyflakes inline no matter what OPENCODE_EDIT_GUARD said, so a 12k-line
        # file blocked the write for ~4.3s on this phone with no way to opt out.
        if path.suffix == ".py" and guard_enabled():
            from .lsp import _diagnostics_py
            from .verify import too_big

            # Reuse the size cap the edit guard already applies, so a huge file
            # cannot stall the write; the gap is stated, never silently hidden.
            if too_big("python", nbytes):
                notes.append(
                    f"note: python syntax not checked ({nbytes}B is over the "
                    f"64KB limit) -- run verify, or lsp diagnostics, on it directly."
                )
                issues = None
            else:
                issues = _diagnostics_py(path)
            if issues is None:
                pass
            elif issues:
                notes.append(
                    "\n\nLSP errors detected in this file, please fix:\n"
                    + "\n".join(f"- {x}" for x in issues[:10])
                )
                diag = {"diagnostics": {str(path): issues[:20]}}
            elif issues == []:
                diag = {"diagnostics": {}}
    except Exception as _oc_exc:
        swallow(_oc_exc, "tools/write.py:_write diagnostics")
        pass
    note = ("\n" + "\n".join(notes)) if notes else ""
    return {
        "output": f"{verb} {path} ({nbytes} bytes, {nlines} lines)." + note,
        "metadata": {"filePath": str(path), "bytes": nbytes, "lines": nlines,
                     "created": not existed, **diag},
    }


def tool() -> Tool:
    description = """Write a NEW file, creating parent dirs. Stats back, no content echo. Existing file? Use edit/apply_patch (or force=true). dry_run previews size."""

    def run(input: dict) -> dict:
        # A bare KeyError used to escape the tool here: a model that left out a
        # required field produced a red traceback instead of a usable message.
        missing = [k for k in ("filePath", "content") if not input.get(k)]
        if missing:
            return {
                "output": "write needs " + " and ".join(missing) + " -- no file was written.",
                "error": True,
            }
        return _write(input["filePath"], input["content"],
                      dry_run=bool(input.get("dry_run", False)),
                      force=bool(input.get("force", False)))

    return Tool(
        name="write",
        description=description,
        parameters=schema_with(
            {
                "content": {"type": "string", "description": "File content"},
                "filePath": {"type": "string", "description": "Absolute path"},
                "dry_run": {"type": "boolean", "description": "Preview size without writing", "optional": True},
                "force": {"type": "boolean", "description": "Overwrite an existing different file", "optional": True},
            },
            ["content", "filePath"],
        ),
        run=run,
        permission="edit",
    )

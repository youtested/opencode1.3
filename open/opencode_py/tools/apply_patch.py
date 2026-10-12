"""apply_patch tool: safe multi-file changes from one unified diff.

RULES (full guidance for maintainers; the sent schema is short):
- One diff may touch MANY files; every hunk verified BEFORE anything is
  written — default all-or-nothing aborts everything with all problems
  reported. partial=true applies clean files and reports only bad hunks
  (best-way: 1 bad hunk costs 1 hunk resend, not N files).
- Position-tolerant (exact context searched nearby), never fuzzy content.
  Miss reports the 3 closest real lines + read offset= anchor.
- `--- /dev/null` creates, `+++ /dev/null` deletes. dry_run previews.
- undo reverts the whole patch; history lists past patches.
- SAFETY, the same layers the edit tool uses: a file marked read-only is
  refused, a hunk that would introduce NEW syntax errors is refused, every
  write is read back and confirmed, and a failure part-way through is rolled
  back — and SAYS SO if the rollback itself failed. force=true overrides the
  refusals.
"""

from __future__ import annotations

from ..util.log import swallow
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

from ..globals import Path as GPath
from .registry import Tool, schema_with
from .write import _atomic_write

_LOCK = threading.Lock()
# Session worktree for relative paths + journal scoping (see set_worktree).
_WORKTREE = threading.local()

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")

JOURNAL_PATH_NAME = "patch_journal.json"
MAX_JOURNAL_ENTRIES = 20
MAX_SHIFT = 200  # how far a hunk may drift from its declared position


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

def _clean_path(p: str) -> str:
    p = p.strip()
    if p.startswith('"') and p.endswith('"') and len(p) > 1:
        p = p[1:-1]
    if p.startswith("a/") or p.startswith("b/"):
        p = p[2:]
    return p


class _FilePatch:
    __slots__ = ("old_path", "new_path", "hunks")

    def __init__(self, old_path: str, new_path: str) -> None:
        self.old_path = old_path  # cleaned, or "/dev/null"
        self.new_path = new_path
        self.hunks: list[tuple[int, int, list[str]]] = []


def _parse_patches(diff_text: str) -> list[_FilePatch]:
    """Parse a multi-file unified diff.

    Unlike util/diff.parse_diff this accepts git-style ``--- /dev/null``
    headers (new files) and tolerates header lines without the a//b/ prefix.
    """
    patches: list[_FilePatch] = []
    cur: _FilePatch | None = None
    in_hunk = False

    for raw in diff_text.splitlines():
        stripped = raw.rstrip("\n")
        is_old_header = stripped.startswith("--- ")
        is_new_header = stripped.startswith("+++ ")
        m = _HUNK_RE.match(stripped)

        if is_old_header and not in_hunk:
            target = _clean_path(stripped[4:])
            cur = _FilePatch(target, "")
            patches.append(cur)
            continue
        if is_new_header and cur is not None and not cur.new_path and not cur.hunks:
            cur.new_path = _clean_path(stripped[4:])
            continue
        if m and cur is not None:
            old_start = int(m.group(1))
            old_count = int(m.group(2) if m.group(2) is not None else 1)
            cur.hunks.append((old_start, old_count, []))
            in_hunk = True
            continue
        if in_hunk and cur is not None and cur.hunks:
            if stripped.startswith(("--- ", "+++ ")) or (
                stripped.startswith("@@") and not m
            ):
                # next file's header or malformed hunk: stop this body
                if stripped.startswith("--- "):
                    in_hunk = False
                    cur = _FilePatch(_clean_path(stripped[4:]), "")
                    patches.append(cur)
                elif stripped.startswith("+++ ") and patches and cur.hunks[-1][2]:
                    pass  # stray header inside a hunk body: keep as data
                continue
            if stripped.startswith("diff ") or stripped.startswith("Index: "):
                in_hunk = False
                cur = None
                continue
            cur.hunks[-1][2].append(stripped)
    return [p for p in patches if p.hunks]


# ---------------------------------------------------------------------------
# tolerant hunk application
# ---------------------------------------------------------------------------

def _split_hunk(body: list[str]) -> tuple[list[str], list[str], int, int]:
    """Split a hunk body into (expected_old_lines, new_block, n_add, n_del)."""
    expected: list[str] = []
    new_block: list[str] = []
    n_add = 0
    n_del = 0
    for line in body:
        if line.startswith("\\"):  # "\ No newline at end of file"
            continue
        tag, rest = (line[0], line[1:]) if line else (" ", "")
        if tag == " " or tag == "":
            expected.append(rest)
            new_block.append(rest)
        elif tag == "-":
            expected.append(rest)
            n_del += 1
        elif tag == "+":
            new_block.append(rest)
            n_add += 1
    return expected, new_block, n_add, n_del


def _find_offset(
    lines: list[str], expected: list[str], declared_idx: int, last_end: int
) -> int | None:
    """Locate ``expected`` exactly at/before/after its declared position."""
    n = len(expected)
    span = len(lines) - n
    if span < 0:
        return None
    lo = max(0, declared_idx - MAX_SHIFT, last_end)
    hi = min(span, declared_idx + MAX_SHIFT)
    for delta in range(0, MAX_SHIFT + 1):
        for cand in (declared_idx + delta, declared_idx - delta):
            if cand < lo or cand > hi:
                continue
            if lines[cand : cand + n] == expected:
                return cand
        if declared_idx + delta > hi and declared_idx - delta < lo:
            break
    return None


_WORD_RE = re.compile(r"\w{3,}")


def _closest_hint(lines: list[str], expected: list[str], n: int = 3) -> str:
    """3 closest real lines to the expected block, plus a read anchor.

    Keeps each line's real leading whitespace: a patch usually fails because of
    something invisible, and a hint that has been stripped cannot be copied back
    verbatim. Candidates come from the target's rarest words, so a big file
    cannot produce a match from the wrong end of it — and when nothing similar
    exists it says nothing rather than guessing.
    """
    import difflib as _dl
    try:
        first = ""
        for ln in expected:
            if ln.strip():
                first = ln[:120]
                break
        if not first:
            return ""
        words = _WORD_RE.findall(first)
        # The most useful thing to say is often "that exact line IS there" —
        # which means the hunk's declared position, or its other context lines,
        # are what is wrong. Cheap to check, and exact.
        exact = [i for i, raw in enumerate(lines, 1) if raw == first]
        if exact:
            locs = ", ".join(str(i) for i in exact[:3])
            return (
                f" That exact line IS in the file at line {locs}: {first[:80]!r}"
                f" — the hunk's line number or its other context lines must be off."
            )
        probes = (
            sorted(set(words), key=lambda w: sum(1 for ln in lines if w in ln))[:3]
            if words
            else [first]
        )
        seen: dict[str, list[int]] = {}
        buckets: dict[str, list[str]] = {w: [] for w in probes}
        for i, raw in enumerate(lines, 1):
            if not raw.strip() or raw == first:
                continue
            if raw in seen:
                seen[raw].append(i)
                continue
            seen[raw] = [i]
            for probe in probes:
                if probe in raw:
                    buckets[probe].append(raw)
        pool: list[str] = []
        for probe in probes:  # rarest word first
            hits = buckets[probe]
            if hits and len(hits) <= max(20, len(seen) // 4):
                pool = hits
                break
        if not pool:
            if len(seen) > 500:
                return ""  # any answer would be a guess
            pool = list(seen)
            if not pool:
                return ""
        if len(pool) > 200:
            pool = pool[:200]
        close = _dl.get_close_matches(first, pool, n=n, cutoff=0.4) or pool[:n]
        parts: list[str] = []
        for c in close:
            where = seen.get(c) or []
            if not where:
                continue
            also = ", ".join(str(x) for x in where[1:4])
            parts.append(
                f"line {where[0]}" + (f" (also {also})" if also else "") + f": {c[:80]!r}"
            )
        if not parts:
            return ""
        s0 = (seen.get(close[0]) or [1])[0]
        return " Closest: " + " | ".join(parts) + f" — read offset={max(1, s0 - 2)} limit=7."
    except Exception:
        return ""


def _apply_hunks(
    lines: list[str], hunks: list[tuple[int, int, list[str]]]
) -> tuple[list[str], list[int]]:
    """Apply ordered hunks to a no-newline-suffix line list.

    Returns (new_lines, shifts) where shifts[i] is how far hunk i moved from
    its declared position. Raises ValueError on any conflict (nothing applied).
    """
    out: list[str] = []
    pos = 0
    last_end = 0
    shifts: list[int] = []
    for idx, (start, count, body) in enumerate(hunks):
        expected, new_block, _, _ = _split_hunk(body)
        if not expected and not new_block:
            raise ValueError(f"hunk {idx + 1} is empty")
        declared = start - 1 if start > 0 else 0
        if not expected:
            # A pure insertion has no context to match, so the declared line
            # number used to be believed blindly — "line 99" of a 4-line file
            # was accepted and the text landed wherever. Anchor instead on the
            # line that must FOLLOW the insertion, and refuse when the position
            # is off the end of the file or the anchor is not there.
            # GNU inserts AFTER old line `start`; start=0 means "before line 1".
            after = max(0, start)
            if after < 0 or after > len(lines):
                raise ValueError(
                    f"hunk {idx + 1}: insert position {after} is outside the file "
                    f"({len(lines)} line(s)) — nothing applied"
                )
            if after < len(lines):
                anchor = lines[after]
                near = [
                    i
                    for i in range(max(0, after - MAX_SHIFT), min(len(lines), after + MAX_SHIFT + 1))
                    if lines[i] == anchor
                ]
                if not near:
                    raise ValueError(
                        f"hunk {idx + 1}: insert anchor {anchor[:60]!r} (the line that "
                        f"should follow the insertion) is not in the file near line "
                        f"{after + 1} — nothing applied"
                    )
                found = near[0]
            else:
                found = len(lines)
        else:
            found = _find_offset(lines, expected, declared, last_end)
            if found is None:
                first = expected[0] if expected else "(no context)"
                hint = _closest_hint(lines, expected)
                raise ValueError(
                    f"hunk {idx + 1}: no exact match near line {start} "
                    f"(expected block starting {first!r}).{hint}"
                )
        out.extend(lines[pos:found])
        out.extend(new_block)
        pos = found + len(expected)
        last_end = pos
        shifts.append(found - declared)
    out.extend(lines[pos:])
    return out, shifts


def _guard_plan(plan: dict[str, Any], before: str, force: bool) -> dict[str, Any]:
    """Refuse a patch that would introduce NEW syntax errors.

    Delegates to edit_guard — the same module the edit tool uses — so the two
    tools can never disagree about what "safe to write" means.
    """
    if force or plan.get("conflict") or plan.get("new_content") is None:
        return plan
    from . import edit_guard

    path = Path(plan["file"])
    problems, note = edit_guard.syntax_check(path, before, plan["new_content"])
    if note:
        plan["note"] = note
    if problems:
        plan["conflict"] = edit_guard.refusal_text(path, problems)
    return plan


# ---------------------------------------------------------------------------
# per-file planning / application
# ---------------------------------------------------------------------------

def _resolve(base: Path, relpath: str) -> Path:
    p = Path(relpath)
    if p.is_absolute():
        base_res = base.resolve()
        try:
            pr = p.resolve()
            pr.relative_to(base_res)
        except ValueError:
            raise ValueError(f"refusing absolute path outside worktree: {relpath!r}")
        return p
    joined = (base / p).resolve()
    base_res = base.resolve()
    try:
        joined.relative_to(base_res)
    except ValueError:
        raise ValueError(f"refusing path traversal outside worktree: {relpath!r}")
    return joined


def _read_text(path: Path) -> tuple[str | None, str]:
    """Read preserving exact bytes-as-text; returns (text|None, error)."""
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            return fh.read(), ""
    except FileNotFoundError:
        return None, ""
    except (OSError, UnicodeError) as e:
        return None, str(e)


def _plan_one(patch: _FilePatch, base: Path, force: bool = False) -> dict[str, Any]:
    creates = patch.old_path == "/dev/null"
    deletes = patch.new_path == "/dev/null"
    target_rel = patch.new_path if creates else patch.old_path
    if deletes and patch.old_path:
        target_rel = patch.old_path
    try:
        path = _resolve(base, target_rel)
    except ValueError as e:
        return {"file": str(base / target_rel), "conflict": str(e), "creates": creates, "deletes": deletes}

    plan: dict[str, Any] = {
        "file": str(path),
        "creates": creates,
        "deletes": deletes,
        "conflict": "",
        "shifts": [],
        "added": 0,
        "removed": 0,
    }

    if creates:
        text, err = _read_text(path)
        if err:
            plan["conflict"] = f"cannot check existing file: {err}"
            return plan
        if text is not None:
            added = "".join(l[1:] + "\n" for h in patch.hunks for l in h[2] if l.startswith("+"))
            if text != added:
                plan["conflict"] = "file already exists with different content"
                return plan
        content = "".join(l[1:] + "\n" for h in patch.hunks for l in h[2] if l.startswith("+"))
        plan["new_content"] = content
        plan["added"] = sum(1 for h in patch.hunks for l in h[2] if l.startswith("+"))
        return _guard_plan(plan, "", force)

    text, err = _read_text(path)
    if err:
        plan["conflict"] = f"cannot read file: {err}"
        return plan
    if text is None:
        plan["conflict"] = "file does not exist"
        return plan

    had_nl = text.endswith("\n")
    # Two invisible prefixes used to make a file unpatchable, because a diff
    # carries neither of them:
    #   * CRLF — the incoming diff is normalised to LF (see _action_apply) while
    #     the file's lines kept their "\r", so the context could never match.
    #   * a UTF-8 BOM, which is read as a leading U+FEFF on line 1.
    # Match in clean text space and put both back on write. CRLF is only
    # normalised when EVERY line ending is CRLF — a file with mixed endings is
    # left alone rather than silently rewritten.
    has_bom = text.startswith("\ufeff")
    body = text[1:] if has_bom else text
    crlf = body.count("\r\n")
    use_crlf = crlf > 0 and crlf == body.count("\n")
    work = body.replace("\r\n", "\n") if use_crlf else body
    lines = work.split("\n")
    if had_nl and lines and lines[-1] == "":
        lines.pop()

    try:
        new_lines, shifts = _apply_hunks(lines, patch.hunks)
    except ValueError as e:
        plan["conflict"] = str(e)
        return plan

    new_text = "\n".join(new_lines)
    if had_nl:
        new_text += "\n"
    if use_crlf:
        new_text = new_text.replace("\n", "\r\n")
    if has_bom:
        new_text = "\ufeff" + new_text  # put the invisible prefix back

    if deletes:
        removed_ok = not any(l for l in new_lines)
        if not removed_ok:
            plan["conflict"] = "deletion leaves non-empty remainder; diff does not match file"
            return plan
        plan["delete_target"] = True
    else:
        plan["new_content"] = new_text

    # What the hunks were matched against. The commit step re-checks this before
    # writing, so a file someone else changed after planning is not clobbered.
    plan["original"] = text

    plan["shifts"] = shifts
    for _, _, body in patch.hunks:
        plan["removed"] += sum(1 for l in body if l.startswith("-"))
        plan["added"] += sum(1 for l in body if l.startswith("+"))
    return _guard_plan(plan, text or "", force)


# ---------------------------------------------------------------------------
# journal (undo support)
# ---------------------------------------------------------------------------

def _journal_file() -> Path:
    return GPath.data / JOURNAL_PATH_NAME


def _journal_load() -> list[dict[str, Any]]:
    try:
        data = json.loads(_journal_file().read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def _journal_save(entries: list[dict[str, Any]]) -> None:
    path = _journal_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(entries)
    from ..session import _write_durable
    _write_durable(path, data, file_sync=True, dir_sync=True)


def set_worktree(path) -> None:
    """Session worktree that relative patch paths and the journal resolve against.

    Thread-local, installed by the agent loop before every tool call — the same
    pattern ``tools/glob.py`` already uses. Both call sites here used
    ``Path.cwd()``, so launching opencode from a subdirectory made apply_patch
    edit a different tree than ``edit``/``write`` (which resolve against the
    session worktree) and scope the journal to a different project than the one
    actually being worked on. Falls back to cwd when unset, which is what the
    headless tests rely on.
    """
    try:
        _WORKTREE.path = None if path is None else Path(path)
    except Exception as _oc_exc:
        swallow(_oc_exc, "tools/apply_patch.py:set_worktree")


def _base_dir() -> Path:
    p = getattr(_WORKTREE, "path", None)
    return Path(p) if p is not None else Path.cwd()


def _project_root() -> str:
    """Absolute path of the project this patch belongs to."""
    try:
        return str(_base_dir().resolve())
    except OSError:  # pragma: no cover - cwd vanished
        return str(_base_dir())


def _entry_root(entry: dict[str, Any]) -> str:
    """Project of a journal entry; "" for entries written before this existed."""
    return str(entry.get("root") or "")


def _entry_matches(entry: dict[str, Any], root: str) -> bool:
    """Whether this journal entry belongs to the given project.

    Entries written before projects were recorded carry no root, so their
    project is inferred from the absolute paths they touched. Without that, a
    project would claim every legacy entry in the shared journal — which is the
    cross-project undo this is meant to prevent.
    """
    entry_root = _entry_root(entry)
    if entry_root:
        return entry_root == root
    for item in entry.get("files", []) or []:
        touched = str(item.get("path") or "")
        if touched == root or touched.startswith(root + os.sep):
            return True
    return False


def _journal_append(entry: dict[str, Any]) -> None:
    with _LOCK:
        entries = _journal_load()
        entry = dict(entry)
        entry.setdefault("root", _project_root())
        entries.append(entry)
        while len(entries) > MAX_JOURNAL_ENTRIES:
            entries.pop(0)
        _journal_save(entries)


def _journal_pop() -> dict[str, Any] | None:
    """Take the newest entry BELONGING TO THIS PROJECT.

    The journal is one file for the whole app, so an unguarded pop would let a
    project undo another project's work without warning.
    """
    with _LOCK:
        entries = _journal_load()
        if not entries:
            return None
        root = _project_root()
        for i in range(len(entries) - 1, -1, -1):
            if _entry_matches(entries[i], root):
                entry = entries.pop(i)
                _journal_save(entries)
                return entry
        return None


def _journal_for_project() -> tuple[list[dict[str, Any]], int]:
    """Entries for this project, and how many belong to other projects."""
    root = _project_root()
    all_entries = _journal_load()
    mine = [e for e in all_entries if _entry_matches(e, root)]
    return mine, len(all_entries) - len(mine)


# ---------------------------------------------------------------------------
# actions
# ---------------------------------------------------------------------------

def _action_apply(
    diff_text: str, dry_run: bool, message: str, partial: bool = False, force: bool = False
) -> dict:
    diff_text = (diff_text or "").replace("\r\n", "\n")
    if not diff_text.strip():
        return {"output": "Empty diff: nothing to apply.", "error": True}
    patches = _parse_patches(diff_text)
    if not patches:
        return {
            "output": (
                "Could not parse any file sections from the diff. Expected "
                "unified format:\n--- a/file\n+++ b/file\n@@ ... @@\n..."
            ),
            "error": True,
        }

    plans: list[dict[str, Any]] = []
    base = _base_dir()
    for patch in patches:
        plans.append(_plan_one(patch, base, force))

    # A diff may name the SAME file twice. Each section is planned on its own,
    # but the commit step finds a file by name, so only the first section could
    # ever be written — the second was silently dropped while the summary still
    # claimed "2 file(s) OK" and listed both. Refuse instead of losing work.
    duplicates: list[str] = []
    counted: set[str] = set()
    for p in plans:
        if p["file"] in counted:
            duplicates.append(p["file"])
        counted.add(p["file"])
    if duplicates:
        listed = "\n".join(f"  {d}" for d in dict.fromkeys(duplicates))
        return {
            "output": (
                "Apply refused: the diff names the same file more than once:\n"
                f"{listed}\n"
                "Each file must appear in ONE section. Combine the hunks into a single "
                "--- / +++ pair for that file and resend. Nothing was written."
            ),
            "error": True,
            "metadata": {"files": [], "applied": 0, "failed": len(duplicates),
                         "duplicate_files": list(dict.fromkeys(duplicates))},
        }

    problems = [p for p in plans if p.get("conflict")]
    ok_files = [p for p in plans if not p.get("conflict")]
    label = "Dry run" if dry_run else "Apply"
    per_file = [{"file": p["file"], "ok": not bool(p.get("conflict")),
                 "added": p.get("added", 0), "removed": p.get("removed", 0),
                 "conflict": p.get("conflict", "")} for p in plans]

    if problems and not (partial and ok_files):
        lines = [f"{label} FAILED — {len(problems)} of {len(plans)} files conflict. "
                 "Nothing was written."]
        for p in problems:
            lines.append(f"  {p['file']}: {p['conflict']}")
        if ok_files:
            lines.append(f"(remaining {len(ok_files)} file(s) would apply cleanly — retry with partial=true to apply them)")
        return {"output": "\n".join(lines), "error": True,
                "metadata": {"files": per_file, "applied": 0, "failed": len(problems)}}
    failed = problems if (partial and ok_files) else []
    if failed:
        plans = ok_files

    summary_lines = [f"{label}: {len(plans)} file(s) OK."]
    for p in plans:
        kind = "create" if p.get("creates") else ("delete" if p.get("delete_target") else "modify")
        shift_note = ""
        real_shifts = [abs(s) for s in p.get("shifts", [])]
        if real_shifts:
            shift_note = f" (hunk offset {max(real_shifts):+d})" if max(real_shifts) else ""
        summary_lines.append(f"  {kind} {p['file']}  +{p['added']} -{p['removed']}{shift_note}")

    # Never let a coverage gap be silent: say which files went unchecked.
    notes = [f"  NOTE {p['file']}: {p['note']}" for p in plans if p.get("note")]
    from . import edit_guard

    if edit_guard.fault():
        notes.append(
            f"  NOTE the syntax guard could not run ({edit_guard.fault()}); "
            f"nothing was syntax-checked"
        )
    if notes:
        summary_lines.append("")
        summary_lines.append("Syntax checking:")
        summary_lines.extend(notes)

    if dry_run:
        if failed:
            summary_lines.append("")
            summary_lines.append(f"SKIPPED ({len(failed)} file(s) conflict — fix and resend only these):")
            for p in failed:
                summary_lines.append(f"  SKIP {p['file']}: {p['conflict']}")
        meta_files = ([{"file": p["file"], "ok": True, "added": p.get("added", 0),
                        "removed": p.get("removed", 0)} for p in plans]
                      + [{"file": p["file"], "ok": False, "conflict": p.get("conflict", "")} for p in failed])
        return {"output": "\n".join(summary_lines),
                "metadata": {"dry_run": True, "files": meta_files,
                             "applied": 0, "failed": len(failed),
                             "partial": bool(failed)}}

    # ---- commit: backup originals FIRST, then write/delete everything ----
    backup: list[dict[str, Any]] = []
    errors: list[str] = []
    for p in plans:
        path = Path(p["file"])
        existed = path.exists()
        original, err = (_read_text(path) if existed else (None, ""))
        if existed and err:
            errors.append(f"{path}: could not back up ({err}); aborting before any write")
            break
        if existed and not force:
            from .edit_guard import read_only_message

            locked = read_only_message(path)
            if locked:
                errors.append(f"{path}: {locked}")
                break
        backup.append({"path": str(path), "existed": existed,
                       "content": original, "creates": bool(p.get("creates"))})
        # What the patch left behind: the new text, or None for a delete (the
        # file must be absent). Used by undo to tell "still my change" from
        # "somebody changed it after me".
        backup[-1]["after"] = None if p.get("delete_target") else (p.get("new_content") or "")

    if errors:
        return {"output": "Apply aborted (nothing was written):\n" + "\n".join(errors),
                "error": True}

    def rollback() -> list[str]:
        """Put every backed-up file back. Returns what could NOT be restored."""
        failed: list[str] = []
        for item in backup:
            path = Path(item["path"])
            try:
                if item["existed"]:
                    _atomic_write(path, item["content"])
                elif path.exists() and item["creates"]:
                    path.unlink()
            except (OSError, UnicodeError) as e:
                # Never swallow this: claiming a rollback that did not happen is
                # worse than the original failure.
                failed.append(f"could not restore {path}: {e}")
        return failed

    for item in backup:
        path = Path(item["path"])
        p = next(x for x in plans if x["file"] == str(path))
        if p.get("creates") or p.get("delete_target"):
            continue
        # The hunk was matched against the file as it looked during PLANNING.
        # If something else wrote to it since (another agent, an editor, a
        # background job), writing now would silently destroy that change.
        # Checked immediately before this file's own write, so the window is as
        # small as it can be.
        baseline = p.get("original")
        if baseline is not None:
            current, rerr = _read_text(path)
            unchanged = not rerr and current == baseline
        else:  # nothing to compare against (a create) — the backup is the truth
            unchanged = True
        if not unchanged:
            errors.append(
                f"{path}: the file changed on disk after the patch was planned "
                f"(another writer got there first) — not written. Re-read it and resend."
            )
            continue
        try:
            _atomic_write(path, p["new_content"])
        except (OSError, UnicodeError) as e:
            errors.append(f"write failed mid-patch: {path}: {e}")
    # Read every file back: a save that silently did not land must not be
    # reported as applied.
    for item in backup:
        path = Path(item["path"])
        p = next(x for x in plans if x["file"] == str(path))
        if p.get("creates") or p.get("delete_target"):
            continue
        from .write import confirm_write

        bad = confirm_write(path, p["new_content"])
        if bad:
            errors.append(f"write not verified: {path}: {bad}")
    if errors:
        unrestored = rollback()
        head = "Apply FAILED mid-write and was rolled back" if not unrestored else \
               "Apply FAILED mid-write and the ROLLBACK WAS INCOMPLETE"
        lines = [head + ":"] + [f"  {e}" for e in errors]
        if unrestored:
            lines.append("")
            lines.append("These files may still hold the patched content — check them by hand:")
            lines.extend(f"  {e}" for e in unrestored)
        return {"output": "\n".join(lines), "error": True,
                "metadata": {"applied": 0, "rolled_back": not unrestored}}

    for item in backup:
        path = Path(item["path"])
        p = next(x for x in plans if x["file"] == str(path))
        try:
            if p.get("creates"):
                path.parent.mkdir(parents=True, exist_ok=True)
                _atomic_write(path, p["new_content"])
            elif p.get("delete_target"):
                path.unlink()
        except (OSError, UnicodeError) as e:
            errors.append(f"{path}: {e}")
    if errors:
        # A create/delete can fail AFTER modified files were already written.
        # "all-or-nothing" is the promise, so undo them rather than leaving a
        # half-applied patch behind.
        unrestored = rollback()
        head = ("Apply FAILED while creating/deleting and was rolled back"
                if not unrestored else
                "Apply FAILED while creating/deleting and the ROLLBACK WAS INCOMPLETE")
        lines = [head + ":"] + [f"  {e}" for e in errors]
        if unrestored:
            lines.append("")
            lines.append("These files may still hold the patched content — check them by hand:")
            lines.extend(f"  {e}" for e in unrestored)
        return {"output": "\n".join(lines), "error": True,
                "metadata": {"applied": 0, "rolled_back": not unrestored}}

    # feed the verify tool's homework checker: track writes, forget deletes
    from .verify import track, untrack

    # Bookkeeping must never destroy the result of a patch that already landed.
    # A full disk or an unwritable journal used to raise straight out of here,
    # crashing the tool AFTER the files were changed: the caller saw a crash
    # instead of a result and could not tell whether anything had happened.
    bookkeeping: list[str] = []

    for item in backup:
        path = Path(item["path"])
        p = next(x for x in plans if x["file"] == str(path))
        if p.get("delete_target"):
            untrack(path)
        else:
            track(path, "apply_patch")

    try:
        _journal_append({
            "time": time.time(),
            "message": message or "",
            "files": backup,
        })
    except Exception as e:
        bookkeeping.append(
            f"the patch WAS applied, but could not be recorded for undo ({e}) — "
            f"this one cannot be undone with action=undo"
        )

    diag: dict = {}
    try:
        from .lsp import _diagnostics_py
        all_issues: dict[str, list[str]] = {}
        for p in plans:
            fp = Path(p["file"])
            if fp.suffix == ".py" and fp.exists():
                issues = _diagnostics_py(fp)
                if issues:
                    all_issues[str(fp)] = issues[:20]
        if all_issues:
            summary_lines.append("")
            summary_lines.append("LSP errors detected in patched files, please fix:")
            for f, issues in list(all_issues.items())[:5]:
                summary_lines.append(f"  {f}:")
                for x in issues[:5]:
                    summary_lines.append(f"    - {x}")
            diag = {"diagnostics": all_issues}
        else:
            diag = {"diagnostics": {}}
    except Exception as _oc_exc:
        swallow(_oc_exc, "tools/apply_patch.py:824")
        pass

    if errors:
        summary_lines.append("")
        summary_lines.append("PARTIAL FAILURES (see above files):")
        summary_lines.extend(f"  {e}" for e in errors)
    if failed:
        summary_lines.insert(1, f"Partial: {len(plans)} file(s) applied, {len(failed)} skipped.")
        summary_lines.append("")
        summary_lines.append(f"SKIPPED ({len(failed)} — resend only these hunks):")
        for p in failed:
            summary_lines.append(f"  SKIP {p['file']}: {p['conflict']}")
    if bookkeeping:
        summary_lines.append("")
        summary_lines.extend(f"WARNING: {w}" for w in bookkeeping)
    else:
        summary_lines.append("Undo available: run apply_patch with action=\"undo\".")
    meta_files = ([{"file": p["file"], "ok": True, "added": p["added"], "removed": p["removed"]}
                   for p in plans]
                  + [{"file": p["file"], "ok": False, "conflict": p.get("conflict", "")} for p in failed])
    return {"output": "\n".join(summary_lines),
            "metadata": {"applied": True, "partial": bool(failed), "files": list(meta_files),
                         "failed": [p["file"] for p in failed], **diag}}


def _action_undo() -> dict:
    entry = _journal_pop()
    if entry is None:
        mine, others = _journal_for_project()
        if others:
            return {
                "output": (
                    f"Nothing to undo in THIS project. The journal holds {others} patch(es) from "
                    f"other projects, which undo will not touch. Change into the project you "
                    f"edited and run undo there."
                ),
                "error": True,
            }
        return {"output": "Nothing to undo: the journal is empty.", "error": True}
    restored: list[str] = []
    errors: list[str] = []
    skipped: list[str] = []
    no_record = object()  # journal entries written before this check
    for item in reversed(entry.get("files", [])):
        # Undo used to write the snapshot over whatever was on disk now, so a
        # change made AFTER this patch (by you, by another agent, by an editor)
        # was destroyed without a word. Only restore a file that still looks the
        # way this patch left it; report the others instead of overwriting them.
        expected_after = item.get("after", no_record)
        path = Path(item["path"])
        if expected_after is not no_record:
            current, rerr = _read_text(path)
            unchanged = (current is None) if expected_after is None else (not rerr and current == expected_after)
            if not unchanged:
                skipped.append(
                    f"{path}: changed since this patch was applied, so undo left it alone"
                )
                continue
        try:
            if item.get("existed") and item.get("content") is not None:
                _atomic_write(path, item["content"])
                restored.append(f"restored {path}")
            else:
                if path.exists():
                    path.unlink()
                    restored.append(f"deleted created file {path}")
        except (OSError, UnicodeError) as e:
            errors.append(f"{path}: {e}")
    msg = entry.get("message") or "(no message)"
    when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(entry.get("time", 0)))
    lines = [f"Undid patch \"{msg}\" ({when}):"]
    lines.extend(f"  {r}" for r in restored)
    if skipped:
        lines.append("LEFT ALONE (changed after this patch — undo would have lost that work):")
        lines.extend(f"  {s}" for s in skipped)
    if errors:
        lines.append("ERRORS during undo:")
        lines.extend(f"  {e}" for e in errors)
    remaining = len(_journal_for_project()[0])
    lines.append(f"Patches left for this project: {remaining}")
    return {"output": "\n".join(lines), "error": bool(errors),
            "metadata": {"undone": True, "restored": restored}}


def _action_history() -> dict:
    entries, others = _journal_for_project()
    if not entries:
        if others:
            return {
                "output": (
                    f"No patches for this project. The journal holds {others} patch(es) from "
                    f"other projects, which are not shown here."
                )
            }
        return {"output": "Patch journal is empty."}
    lines = [f"Last {len(entries)} applied patch(es) in this project:"]
    for i, e in enumerate(entries, 1):
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(e.get("time", 0)))
        files = ", ".join(Path(f["path"]).name for f in e.get("files", [])[:5])
        more = "" if len(e.get("files", [])) <= 5 else ", ..."
        lines.append(f"{i}. [{when}] {e.get('message') or '(no message)'} — {files}{more}")
    if others:
        lines.append(f"({others} patch(es) from other projects are not listed and cannot be undone here.)")
    lines.append("Undo reverts the LAST one.")
    return {"output": "\n".join(lines)}


# ---------------------------------------------------------------------------
# tool
# ---------------------------------------------------------------------------

_ACTIONS = ("apply", "undo", "history")


def tool() -> Tool:
    description = """Apply one unified diff across many files, atomically (all-or-nothing). partial=true applies clean files, lists bad hunks (~0.3KB resend). dry_run previews; undo reverts. A hunk that would break a file's syntax, or that targets a read-only file, is refused and nothing is written (force=true overrides)."""

    def run(input: dict) -> dict:
        action = str(input.get("action") or "apply").strip().lower()
        if action not in _ACTIONS:
            return {"output": f"Unknown action {action!r} (want: {', '.join(_ACTIONS)}).",
                    "error": True}
        if action == "undo":
            return _action_undo()
        if action == "history":
            return _action_history()
        return _action_apply(
            str(input.get("diff") or ""),
            dry_run=bool(input.get("dry_run", False)),
            message=str(input.get("message") or ""),
            partial=bool(input.get("partial", False)),
            force=bool(input.get("force", False)),
        )

    return Tool(
        name="apply_patch",
        description=description,
        parameters=schema_with(
            {
                "diff": {
                    "type": "string",
                    "description": "Unified diff (multi-file ok)",
                    "optional": True,
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "Preview without writing",
                    "optional": True,
                },
                "message": {
                    "type": "string",
                    "description": "Journal label",
                    "optional": True,
                },
                "partial": {
                    "type": "boolean",
                    "description": "Apply clean files, skip bad hunks (1 hunk resend, not N files)",
                    "optional": True,
                },
                "force": {
                    "type": "boolean",
                    "description": "Apply even if a hunk would break the file's syntax or the file is read-only",
                    "optional": True,
                },
                "action": {
                    "type": "string",
                    "enum": list(_ACTIONS),
                    "description": "apply, undo, or history",
                    "optional": True,
                },
            },
            [],
        ),
        run=run,
        permission="apply_patch",
    )
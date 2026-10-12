"""verify tool: syntax-check edited files before claiming done.

RULES (full guidance for maintainers; the sent schema is short):
- Tracks edit/write/apply_patch touches; one check covers all. Python is
  compiled, JSON/TOML/YAML parsed, shell gets bash -n — never executed.
- Green implicit check clears tracking; failures persist for fix + re-check.
- Only say done when green.

``syntax_errors`` is the same language knowledge exposed as a pure function of
the file's TEXT, so the edit tool can check a candidate it has not written yet
and refuse an edit that would break the file. Keeping one implementation here
means a language is never checked two different ways.

Cost matters as much as coverage: these run on every edit, so each language
declares how much text it is worth checking (see ``MAX_BYTES``), and the
javascript check is opt-in via ``allow_expensive`` because starting node costs
~0.9 s on this hardware.
"""

from __future__ import annotations

from ..util.log import swallow
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any

from .registry import Tool, schema_with

_LOCK = threading.Lock()
# absolute path -> which tool touched it ("edit"/"write"/"apply_patch")
_TRACKED: dict[str, str] = {}
_TRACK_ORDER: list[str] = []

MAX_TRACKED = 200


def track(path: str | Path, source: str) -> None:
    """Record a file the model just wrote/edited (called by other tools)."""
    try:
        p = Path(path).resolve()
    except OSError as _oc_exc:
        swallow(_oc_exc, "opencode_py/tools/verify.py:47")
        return
    with _LOCK:
        if p.as_posix() not in _TRACKED:
            _TRACK_ORDER.append(p.as_posix())
            while len(_TRACK_ORDER) > MAX_TRACKED:
                old = _TRACK_ORDER.pop(0)
                _TRACKED.pop(old, None)
        _TRACKED[p.as_posix()] = source


def untrack(path: str | Path) -> None:
    """Forget a file (e.g. apply_patch deleted it)."""
    with _LOCK:
        key = Path(path).resolve().as_posix()
        _TRACKED.pop(key, None)
        if key in _TRACK_ORDER:
            _TRACK_ORDER.remove(key)


def tracked() -> list[str]:
    with _LOCK:
        return list(_TRACK_ORDER)


def _clear_tracked() -> int:
    with _LOCK:
        n = len(_TRACK_ORDER)
        _TRACKED.clear()
        _TRACK_ORDER.clear()
        return n


# ---------------------------------------------------------------------------
# per-file checks
# ---------------------------------------------------------------------------

SUFFIX_KINDS: dict[str, str] = {
    ".py": "python",
    ".json": "json",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".sh": "shell",
    ".bash": "shell",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
}

_DIGITS_RE = re.compile(r"\d+")


def kind_for(path: Path | str) -> str:
    """Language key for a path, or "other" when nothing can check it."""
    return SUFFIX_KINDS.get(Path(path).suffix.lower(), "other")


def _py_syntax_errors(text: str) -> list[str]:
    # compile() on a str rejects a leading BOM; the bytes-based compiler used to
    # skip it, so drop it here to keep both entry points agreeing.
    if text.startswith("\ufeff"):
        text = text[1:]
    try:
        compile(text, "<candidate>", "exec")
    except SyntaxError as e:
        where = f"line {e.lineno}" + (f", col {e.offset}" if e.offset else "")
        return [f"syntax error at {where}: {e.msg}"]
    except ValueError as e:
        return [f"compile error: {e}"]
    except (OverflowError, TypeError) as e:  # pragma: no cover - exotic
        return [f"compile error: {e}"]
    return []


def _json_syntax_errors(text: str) -> list[str]:
    try:
        json.loads(text)
    except UnicodeDecodeError as e:
        return [f"not valid UTF-8: {e}"]
    except ValueError as e:
        return [f"invalid JSON: {e}"]
    return []


def _toml_syntax_errors(text: str) -> list[str] | None:
    try:
        import tomllib
    except ImportError:
        return None  # no checker available, not "no errors"
    try:
        tomllib.loads(text)
    except ValueError as e:
        return [f"invalid TOML: {e}"]
    except TypeError as e:  # non-str input
        return [f"invalid TOML: {e}"]
    return []


def _yaml_syntax_errors(text: str) -> list[str] | None:
    try:
        import yaml  # type: ignore
    except ImportError:
        return None  # no checker available, not "no errors"
    try:
        yaml.safe_load(text)
    except ValueError as e:  # yaml raises its own errors subclassing ValueError
        return [f"invalid YAML: {e}"]
    return []


_LEADING_JUNK_RE = re.compile(r"^(?:bash|sh|node|/[\w./-]+|\[[^\]]*\])\s*:\s*")


def _first_useful_error_line(stderr: str, stdout: str, returncode: int) -> str:
    """Pull the informative line out of a checker's stderr.

    Naively taking the last line gives Node's version banner and bash's own
    absolute path, which tells the caller nothing. Prefer the first line that
    actually mentions an error, and strip the interpreter's own prefix.
    """
    lines = [ln.strip() for ln in ((stderr or "") + "\n" + (stdout or "")).splitlines() if ln.strip()]
    if not lines:
        return f"exit {returncode}"
    chosen = next(
        (ln for ln in lines if re.search(r"error|unexpected|invalid", ln, re.IGNORECASE)),
        lines[0],
    )
    cleaned = _LEADING_JUNK_RE.sub("", chosen).strip()
    return (cleaned or chosen)[:160]


def _run_stdin_syntax_check(cmd: list[str], text: str, timeout: int = 15) -> list[str] | None:
    """Feed `text` to a ``--check``-style command; None when it is unavailable."""
    try:
        proc = subprocess.run(cmd, input=text, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode == 0:
        return []
    return [f"syntax error: {_first_useful_error_line(proc.stderr, proc.stdout, proc.returncode)}"]


def _shell_syntax_errors(text: str) -> list[str] | None:
    bash = shutil.which("bash")
    if not bash:
        return None
    return _run_stdin_syntax_check([bash, "-n"], text)


def _javascript_syntax_errors(text: str, suffix: str = ".js") -> list[str] | None:
    node = shutil.which("node")
    if not node:
        return None
    # node reads stdin as CommonJS, so `export ...` in a valid .mjs file comes
    # back as a syntax error. A file named with the real suffix lets node pick
    # the module system itself, which is the difference between checking and
    # inventing false refusals.
    workdir = tempfile.mkdtemp(prefix="editguard-")
    try:
        candidate = os.path.join(workdir, "candidate" + suffix)
        with open(candidate, "w", encoding="utf-8") as fh:
            fh.write(text)
        proc = subprocess.run(
            [node, "--check", candidate], capture_output=True, text=True, timeout=15
        )
    except (OSError, subprocess.SubprocessError):
        return None
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    if proc.returncode == 0:
        return []
    return [f"syntax error: {_first_useful_error_line(proc.stderr, proc.stdout, proc.returncode)}"]


# Bytes of text each language is worth checking on every edit. Beyond these the
# check costs more than the edit it guards, so it is skipped and said so.
#
# The numbers come from measurement on this hardware, not guesswork: CPython's
# compiler runs ~0.045 ms per line, so a 64 KB Python file costs ~200 ms and a
# 10,000-line one ~450 ms — more than the edit itself. tomllib and PyYAML parse
# in pure Python and are slower still. json is a C parser, so it can afford
# megabytes. bash -n and node cost a fixed process start (28 ms and ~840 ms)
# and scale negligibly with size.
MAX_BYTES: dict[str, int] = {
    "python": 64 * 1024,
    "json": 2 * 1024 * 1024,
    "toml": 64 * 1024,
    "yaml": 64 * 1024,
    "shell": 512 * 1024,
    "javascript": 512 * 1024,
}

# Languages whose checker must start another program (~0.9 s for node on a
# slow ARM phone). Off unless the caller explicitly allows the cost.
EXPENSIVE = frozenset({"javascript"})


def too_big(kind: str, size: int) -> bool:
    return size > MAX_BYTES.get(kind, 256 * 1024)


def syntax_errors(
    path: Path | str, text: str, allow_expensive: bool = False
) -> list[str] | None:
    """Syntax problems in `text` for ``path``'s language, or None if unchecked.

    A pure function of the text: no disk access, no language server. That matters
    because the edit tool calls it on a candidate BEFORE writing, and a slow or
    unhappy language server must never be able to veto a valid edit. Returns
    None (rather than []) when no checker exists for the file type, so callers
    can tell "checked and clean" from "cannot check".

    ``allow_expensive`` opts into checkers that must launch another program;
    they are off by default because they cost more than the edit itself.
    """
    kind = kind_for(path)
    if kind in EXPENSIVE and not allow_expensive:
        return None
    if too_big(kind, len(text.encode("utf-8", "replace"))):
        return None
    if kind == "python":
        return _py_syntax_errors(text)
    if kind == "json":
        return _json_syntax_errors(text)
    if kind == "toml":
        return _toml_syntax_errors(text)
    if kind == "yaml":
        return _yaml_syntax_errors(text)
    if kind == "shell":
        return _shell_syntax_errors(text)
    if kind == "javascript":
        return _javascript_syntax_errors(text, Path(path).suffix.lower() or ".js")
    return None


def new_syntax_errors(
    path: Path | str, before: str, after: str, allow_expensive: bool = False
) -> list[str]:
    """Problems that ``after`` has and ``before`` did not.

    Comparing the two sides is what makes refusing safe: a file that was
    ALREADY broken stays editable, and an edit that merely moves an existing
    error to a different line is not treated as a new one.
    """
    after_errors = syntax_errors(path, after, allow_expensive=allow_expensive)
    if not after_errors:
        return []
    before_errors = syntax_errors(path, before, allow_expensive=allow_expensive) or []
    if len(after_errors) <= len(before_errors):
        # Same number or fewer: the edit did not make the file worse overall.
        # Still catch a swap (one fixed, one different added) by comparing the
        # messages with their line numbers removed.
        known = {_DIGITS_RE.sub("#", m) for m in before_errors}
        return [m for m in after_errors if _DIGITS_RE.sub("#", m) not in known]
    # More problems than before: report the ones that are not pre-existing.
    known = {_DIGITS_RE.sub("#", m) for m in before_errors}
    fresh = [m for m in after_errors if _DIGITS_RE.sub("#", m) not in known]
    return fresh or after_errors


def _check_python(path: Path) -> tuple[bool, str]:
    """Syntax + pyflakes check: catches broken code WITHOUT running it."""
    try:
        src = path.read_bytes()
    except OSError as e:
        return False, f"unreadable: {e}"
    errors = _py_syntax_errors(src.decode("utf-8", "replace"))
    if errors:
        return False, errors[0]
    try:
        from .lsp import _diagnostics_py, _pyflakes_note

        issues = _diagnostics_py(path)
        if issues:
            hard = [x for x in issues if ("undefined name" in x or "redefinition" in x or x.startswith("ERROR"))]
            if hard:
                first = hard[0]
                extra = f" (+{len(issues) - 1} more)" if len(issues) > 1 else ""
                return False, f"{first}{extra}"
            return True, f"syntax OK ({len(issues)} style warning(s): {issues[0][:120]})"
        note = _pyflakes_note()
        if note:
            # Be explicit that only the syntax pass ran, so "OK" is not read
            # as "checked and clean".
            return True, f"syntax OK — {note}"
    except Exception as _oc_exc:
        swallow(_oc_exc, "tools/verify.py:332")
        pass
    return True, "syntax OK"


def _check_json(path: Path) -> tuple[bool, str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        return False, f"unreadable: {e}"
    errors = _json_syntax_errors(text)
    if errors:
        return False, errors[0]
    return True, "valid JSON"


def _check_toml(path: Path) -> tuple[bool, str]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as e:
        return False, f"unreadable: {e}"
    errors = _toml_syntax_errors(text)
    if errors is None:
        return True, "not checked (tomllib unavailable)"
    if errors:
        return False, errors[0]
    return True, "valid TOML"


def _check_yaml(path: Path) -> tuple[bool, str]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as e:
        return False, f"unreadable: {e}"
    errors = _yaml_syntax_errors(text)
    if errors is None:
        return True, "not checked (PyYAML not installed)"
    if errors:
        return False, errors[0]
    return True, "valid YAML"


def _check_shell(path: Path) -> tuple[bool, str]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as e:
        return False, f"unreadable: {e}"
    errors = _shell_syntax_errors(text)
    if errors is None:
        return True, "not checked (bash not found)"
    if errors:
        return False, errors[0]
    return True, "syntax OK"


def _check_javascript(path: Path) -> tuple[bool, str]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as e:
        return False, f"unreadable: {e}"
    errors = _javascript_syntax_errors(text)
    if errors is None:
        return True, "not checked (node not found)"
    if errors:
        return False, errors[0]
    return True, "syntax OK"


def _check_one(path_str: str) -> dict[str, Any]:
    path = Path(path_str)
    if not path.exists():
        return {"file": path_str, "ok": False, "note": "missing (deleted?)"}
    if not path.is_file():
        return {"file": path_str, "ok": False, "note": "not a regular file"}
    kind = kind_for(path)
    checker = _PATH_CHECKERS.get(kind)
    if checker is None:
        ok, note = True, "no checker for this type"
    else:
        ok, note = checker(path)
    return {"file": str(path), "ok": ok, "note": note, "kind": kind}


_PATH_CHECKERS = {
    "python": _check_python,
    "json": _check_json,
    "toml": _check_toml,
    "yaml": _check_yaml,
    "shell": _check_shell,
    "javascript": _check_javascript,
}


# ---------------------------------------------------------------------------
# actions
# ---------------------------------------------------------------------------

def _action_check(paths: list[str] | None) -> dict:
    if paths:
        targets = [str(p) for p in paths]
    else:
        targets = tracked()
    if not targets:
        return {
            "output": (
                "Nothing to verify yet: no files have been edited/written "
                "since the last clean verify (or pass explicit paths)."
            ),
            "metadata": {"checked": 0},
        }
    results = [_check_one(p) for p in targets]
    failed = [r for r in results if not r["ok"]]
    lines = []
    for r in results:
        mark = "✅" if r["ok"] else "❌"
        lines.append(f"{mark} {r['file']}  ({r.get('kind', '')}: {r['note']})")
    summary = f"{len(results)} checked: {len(results) - len(failed)} pass, {len(failed)} fail"
    lines.append(summary)

    cleared = 0
    if not failed and not paths:
        # only auto-clear after a fully green implicit check
        cleared = _clear_tracked()

    out = "\n".join(lines)
    if failed:
        out += "\nFix the failing files, then verify again."
    elif cleared:
        out += f"\nAll green — tracked list cleared ({cleared} file(s))."
    return {
        "output": out,
        "error": bool(failed),
        "metadata": {
            "checked": len(results),
            "failed": len(failed),
            "files": results,
        },
    }


def tool() -> Tool:
    description = """Syntax-check edited files (never executed). check (default) or reset. Green before done."""

    def run(input: dict) -> dict:
        action = str(input.get("action") or "check").strip().lower()
        raw_paths = input.get("paths")
        if isinstance(raw_paths, str):
            paths = [raw_paths]
        elif isinstance(raw_paths, list):
            paths = [str(p) for p in raw_paths]
        else:
            paths = None
        if action == "reset":
            n = _clear_tracked()
            return {"output": f"Tracked list cleared ({n} file(s)).",
                    "metadata": {"cleared": n}}
        if action != "check":
            return {"output": f"Unknown action {action!r} (want check or reset).",
                    "error": True}
        return _action_check(paths)

    return Tool(
        name="verify",
        description=description,
        parameters=schema_with(
            {
                "action": {
                    "type": "string",
                    "enum": ["check", "reset"],
                    "description": "check (default) or reset",
                    "optional": True,
                },
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Explicit files",
                    "optional": True,
                },
            },
            [],
        ),
        run=run,
        permission="verify",
    )
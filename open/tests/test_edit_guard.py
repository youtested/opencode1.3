"""The edit tool's safety net: refuse, confirm, and never fail silently.

Before this, an edit that broke a file was saved first and reported afterwards
(and only for .py). Now the candidate text is syntax-checked before the write,
the write is re-read to confirm it, and every gap is stated out loud.
"""

import pytest

from opencode_py.tools import edit as E
from opencode_py.tools.edit import _edit
from opencode_py.tools.verify import (
    kind_for,
    new_syntax_errors,
    syntax_errors,
    too_big,
)


# --- the checkers themselves --------------------------------------------

def test_python_syntax_detected():
    assert syntax_errors("a.py", "x = 1\n") == []
    assert syntax_errors("a.py", "x = (\n") != []


def test_json_toml_shell_detected():
    assert syntax_errors("a.json", '{"a": 1}') == []
    assert syntax_errors("a.json", '{"a": }') != []
    assert syntax_errors("a.toml", "a = 1") == []
    assert syntax_errors("a.toml", "a = = 1") != []
    assert syntax_errors("a.sh", "f() {\n  echo hi\n}\n") == []
    assert syntax_errors("a.sh", "f() {\n  echo hi\n") != []


def test_no_checker_returns_none_not_empty():
    """None means 'cannot check', which callers must be able to tell apart."""
    assert syntax_errors("a.txt", "anything (((") is None
    assert syntax_errors("a.md", "# (((") is None
    assert syntax_errors("a.ts", "let a: = ;") is None  # node cannot parse TS


def test_javascript_is_off_by_default_and_understands_esm_when_on():
    from opencode_py.tools.verify import syntax_errors as se

    assert se("a.js", "const b = (;") is None          # not run by default
    # a valid .mjs must NOT be reported as broken: node reads a stdin .mjs as
    # CommonJS, which is why the real check writes a file with the right suffix
    assert se("a.mjs", "export const a = 1;\n", allow_expensive=True) == []
    assert se("a.mjs", "export const a = ;\n", allow_expensive=True)


def test_kind_for_covers_the_languages_we_can_check():
    assert kind_for("a.py") == "python"
    assert kind_for("a.json") == "json"
    assert kind_for("a.toml") == "toml"
    assert kind_for("a.yml") == "yaml"
    assert kind_for("a.sh") == "shell"
    assert kind_for("a.js") == "javascript"
    assert kind_for("a.bin") == "other"


def test_size_cap_is_honoured():
    assert too_big("python", 10) is False
    assert too_big("python", 10 * 1024 * 1024) is True
    # over the cap, python reports "cannot check" rather than taking 2s
    huge = "x = 1\n" * 60_000
    assert syntax_errors("a.py", huge) is None


# --- only NEW problems are treated as new --------------------------------

def test_new_syntax_errors_ignores_preexisting_problems():
    # already broken, and still just as broken: not a new problem
    assert new_syntax_errors("a.py", "def f(:\n", "def g(:\n") == []
    # already broken, and the edit repairs it
    assert new_syntax_errors("a.py", "def f(:\n", "def f():\n    pass") == []
    # the error merely moved to another line
    assert new_syntax_errors("a.py", "x = (1\ny = 2\n", "a = 0\nx = (1\n") == []
    # clean file becomes broken
    assert new_syntax_errors("a.py", "x = 1\n", "x = (\n")


def test_new_syntax_errors_on_unchecked_files_is_empty():
    assert new_syntax_errors("a.txt", "ok\n", "(((\n") == []


# --- the guard refuses, and the file is left alone -----------------------

@pytest.mark.parametrize("name,init,old,new", [
    ("m.py", "def f():\n    return 1\n", "    return 1", "    return 1("),
    ("m.py", "def f():\n    pass\n", "def f():", "def f()"),
    ("m.json", '{"a": 1}\n', '{"a": 1}', '{"a": }'),
    ("m.toml", "a = 1\n", "a = 1", "a = = 1"),
    ("m.sh", "f() {\n  echo hi\n}\n", "  echo hi", "  echo hi\n}"),
])
def test_breaking_edit_is_refused_and_file_untouched(tmp_path, name, init, old, new):
    p = tmp_path / name
    p.write_text(init)
    r = _edit(str(p), old, new)
    assert r.get("error") is True, r
    assert r["metadata"]["guard"] == "refused"
    assert p.read_text() == init, "the file must be untouched"
    assert "Refused" in r["output"]
    assert "force=true" in r["output"]  # tells the model how to proceed


def test_refusal_names_the_actual_problem(tmp_path):
    p = tmp_path / "m.py"
    p.write_text("x = 1\n")
    r = _edit(str(p), "x = 1", "x = (")
    assert "never closed" in r["output"] or "unmatched" in r["output"]


def test_a_valid_edit_is_never_blocked(tmp_path):
    p = tmp_path / "m.py"
    p.write_text("def f():\n    return 1\n")
    r = _edit(str(p), "    return 1", "    return 2")
    assert r.get("error") is not True
    assert r["metadata"]["guard"] == "ok"
    assert p.read_text() == "def f():\n    return 2\n"


def test_an_already_broken_file_stays_editable(tmp_path):
    """The guard must never deadlock a repair: no NEW problem, no refusal."""
    p = tmp_path / "m.py"
    p.write_text("def f(:\n")
    r = _edit(str(p), "def f(:", "def f():\n    pass")
    assert r.get("error") is not True
    assert p.read_text() == "def f():\n    pass\n"


def test_force_overrides_the_refusal_and_says_so(tmp_path):
    p = tmp_path / "m.py"
    p.write_text("x = 1\n")
    r = _edit(str(p), "x = 1", "x = (", force=True)
    assert r.get("error") is not True
    assert r["metadata"]["guard"] == "forced"
    assert p.read_text() == "x = (\n"


def test_line_range_mode_is_guarded_too(tmp_path):
    p = tmp_path / "m.py"
    p.write_text("def f():\n    return 1\n")
    r = _edit(str(p), "", "return 1(", offset=2, limit=1)
    assert r.get("error") is True
    assert r["metadata"]["guard"] == "refused"
    assert p.read_text() == "def f():\n    return 1\n"


def test_dry_run_warns_but_changes_nothing(tmp_path):
    p = tmp_path / "m.py"
    p.write_text("x = 1\n")
    r = _edit(str(p), "x = 1", "x = ((", dry_run=True)
    assert r.get("error") is not True          # a preview is not a failure
    assert r["metadata"]["guard"] == "would_refuse"
    assert "REFUSED" in r["output"]
    assert p.read_text() == "x = 1\n"


def test_guard_can_be_switched_off(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENCODE_EDIT_GUARD", "off")
    p = tmp_path / "m.py"
    p.write_text("x = 1\n")
    r = _edit(str(p), "x = 1", "x = (")
    assert r.get("error") is not True          # guard off: applied as before
    assert p.read_text() == "x = (\n"


def test_expensive_checks_switched_on(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENCODE_EDIT_GUARD", "all")
    p = tmp_path / "m.js"
    p.write_text("const a = 1;\n")
    r = _edit(str(p), "const a = 1;", "const a = (;")
    assert r.get("error") is True              # now javascript is checked
    assert p.read_text() == "const a = 1;\n"


def test_unchecked_language_is_reported_out_loud(tmp_path):
    """Silence about an unchecked file is what the user complained about."""
    p = tmp_path / "m.js"
    p.write_text("const a = 1;\n")
    r = _edit(str(p), "const a = 1;", "const a = 2;")
    assert r.get("error") is not True
    assert "not checked" in r["output"]


def test_no_note_for_a_fully_checked_file(tmp_path):
    p = tmp_path / "m.py"
    p.write_text("x = 1\n")
    r = _edit(str(p), "x = 1", "x = 2")
    assert "not checked" not in r["output"]


# --- the write is confirmed, not assumed --------------------------------

def test_write_is_read_back_and_confirmed(tmp_path, monkeypatch):
    p = tmp_path / "m.py"
    p.write_text("x = 1\n")
    calls = []
    real = E._confirm_write
    monkeypatch.setattr(E, "_confirm_write", lambda path, expected: calls.append(path) or real(path, expected))
    r = _edit(str(p), "x = 1", "x = 2")
    assert calls == [p]              # it really did read the file back
    assert r.get("error") is not True


def test_a_write_that_does_not_land_is_reported_as_failure(tmp_path, monkeypatch):
    p = tmp_path / "m.py"
    p.write_text("x = 1\n")
    monkeypatch.setattr(
        E, "_confirm_write", lambda path, expected: "the bytes on disk do not match"
    )
    r = _edit(str(p), "x = 1", "x = 2")
    assert r.get("error") is True
    assert r["metadata"]["guard"] == "unverified"
    assert "check it before continuing" in r["output"]


def test_confirm_write_detects_a_real_mismatch(tmp_path):
    p = tmp_path / "m.py"
    p.write_text("something else\n")
    assert E._confirm_write(p, "what we meant to write") is not None
    assert E._confirm_write(p, "something else\n") is None


def test_confirm_write_reports_an_unreadable_file(tmp_path):
    missing = tmp_path / "gone.py"
    assert "could not read the file back" in (E._confirm_write(missing, "x") or "")


# --- the guard itself must never be a liability -------------------------

def test_guard_never_raises(tmp_path, monkeypatch):
    """A crashing checker must allow the edit, not block it — and say so."""
    p = tmp_path / "m.py"
    p.write_text("x = 1\n")
    import opencode_py.tools.verify as V

    def boom(*a, **k):
        raise RuntimeError("checker exploded")

    monkeypatch.setattr(V, "new_syntax_errors", boom)
    r = _edit(str(p), "x = 1", "x = (")
    assert r.get("error") is not True, "a broken guard must not block an edit"
    assert p.read_text() == "x = (\n"
    assert "could not run" in r["output"]
    assert "checker exploded" in r["output"]


def test_guard_off_note_absent_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENCODE_EDIT_GUARD", "off")
    p = tmp_path / "m.js"
    p.write_text("const a = 1;\n")
    r = _edit(str(p), "const a = 1;", "const a = 2;")
    assert "not checked" not in r["output"]

"""Round-2 edit.py fixes: indent repair, hint quality, scan count, dry runs.

Each test pins one behaviour that was wrong before:
- a multi-line fuzzy replacement indented only its first line (syntax error);
- ``_closest_hints`` ranked ``sorted(...)[:500]``, so on a big file every hint
  came from the alphabetically first lines, and hints were stripped of the
  indentation the caller needs in order to copy them;
- a fuzzy edit scanned the whole file twice;
- ``dry_run=True`` marked the file as edited for the verify checker.
"""

import pytest

from opencode_py.tools import edit as E
from opencode_py.tools.edit import _closest_hints, _edit, _reindent


# --- A. multi-line indent repair ----------------------------------------

def test_reindent_prefixes_every_flush_line():
    assert _reindent("a = 1\nb = 2", "    ") == "    a = 1\n    b = 2"


def test_reindent_leaves_deeper_lines_alone():
    # a line that already starts with whitespace is left exactly as it is
    assert _reindent("a = 1\n    b = 2", "    ") == "    a = 1\n    b = 2"


def test_reindent_does_not_add_trailing_space_to_blanks():
    assert _reindent("a = 1\n\nb = 2", "    ") == "    a = 1\n\n    b = 2"


def test_reindent_single_line():
    assert _reindent("a = 1", "\t") == "\ta = 1"


def test_multiline_fuzzy_edit_keeps_every_line_indented(tmp_path):
    p = tmp_path / "m.py"
    p.write_text("def f():\n        a = 1\n        b = 2\n        return a\n")
    r = _edit(str(p), "      a = 1\n      b = 2", "a = 1\nb = 2")  # wrong indent -> fuzzy
    assert r.get("error") is None
    assert p.read_text() == "def f():\n        a = 1\n        b = 2\n        return a\n"
    compile(p.read_text(), str(p), "exec")  # must still be valid Python


def test_three_line_fuzzy_edit_keeps_every_line_indented(tmp_path):
    p = tmp_path / "m3.py"
    p.write_text("class C:\n    def f(self):\n        a = 1\n        b = 2\n        c = 3\n")
    r = _edit(str(p), "   a = 1\n   b = 2\n   c = 3", "a = 1\nb = 2\nc = 3")
    assert r.get("error") is None
    compile(p.read_text(), str(p), "exec")
    assert "        a = 1\n        b = 2\n        c = 3" in p.read_text()


def test_tabs_are_preserved_by_the_repair(tmp_path):
    p = tmp_path / "t.py"
    p.write_text("def f():\n\ta = 1\n\tb = 2\n")
    r = _edit(str(p), "  a = 1\n  b = 2", "a = 1\nb = 2")
    assert r.get("error") is None
    assert p.read_text() == "def f():\n\ta = 1\n\tb = 2\n"


def test_exact_multi_line_edit_is_untouched(tmp_path):
    """An exact match must not be re-indented — the model said what it meant.

    Done at module level, where flush-left text is still valid, so this asserts
    the indent repair really did not run rather than tripping the syntax guard.
    """
    p = tmp_path / "e.py"
    p.write_text("def f():\n    return 1\n\nx = 1\ny = 2\n")
    r = _edit(str(p), "x = 1\ny = 2", "a = 1\nb = 2")
    assert r.get("error") is None
    assert p.read_text() == "def f():\n    return 1\n\na = 1\nb = 2\n"


def test_exact_edit_that_would_break_the_file_is_refused(tmp_path):
    """Same text as above, but moving code out of its block: must not be written."""
    p = tmp_path / "e2.py"
    p.write_text("def f():\n    a = 1\n    b = 2\n")
    r = _edit(str(p), "    a = 1\n    b = 2", "a = 1\nb = 2")
    assert r.get("error") is True
    assert p.read_text() == "def f():\n    a = 1\n    b = 2\n"


def test_single_line_fuzzy_edit_still_indents(tmp_path):
    p = tmp_path / "one.py"
    p.write_text("def f():\n    a = 1\n")
    r = _edit(str(p), "  a =  1", "a = 9")  # extra inner space -> fuzzy, not exact
    assert r.get("error") is None
    assert p.read_text() == "def f():\n    a = 9\n"


# --- B. hint quality -----------------------------------------------------

def test_hint_finds_the_line_deep_inside_a_big_file():
    """The old alphabetical [:500] pool could only ever answer from the top."""
    body = "".join(f"    def handler_{i}(self):\n        pass\n\n" for i in range(3000))
    h = _closest_hints(body, "        retun 2999\nq")
    assert h, "expected a hint"
    assert any("handler_2999" in x for x in h)


def test_hint_keeps_the_real_indentation():
    """A hint is meant to be copied verbatim, so the indent must be in it."""
    body = "".join(f"    def handler_{i}(self):\n        pass\n\n" for i in range(50))
    h = _closest_hints(body, "  def handler_7(self):\nq")
    assert h
    assert any(x.endswith("    def handler_7(self):") for x in h), h


def test_hint_reports_all_line_numbers_of_a_repeated_line():
    body = "x = 1\n" + "y = 2\n" * 3 + "z = 3\n"
    h = _closest_hints(body, "y == 2\nq", 1)
    assert h and "line 2" in h[0] and "also 3, 4" in h[0]


def test_hint_says_nothing_rather_than_guessing_on_a_big_file():
    """Nothing shared with a 5k-line file: silence beats a random line."""
    body = "".join(f"value_{i} = compute({i})\n" for i in range(5000))
    assert _closest_hints(body, "zzz_qqq_nothing_shared\nq") == []


def test_hint_still_helps_on_a_small_file():
    body = "import os\nx = 1\ntarget_name = 3\ny = 2\n"
    h = _closest_hints(body, "target_nme = 3", 2)
    assert h == ["line 3: target_name = 3"]


def test_hint_on_empty_and_degenerate_input():
    assert _closest_hints("", "anything") == []
    assert _closest_hints("a\nb\n", "") == []
    assert _closest_hints("a\nb\n", "\n\n") == []


@pytest.mark.parametrize("size", [10, 1000, 20000])
def test_hints_stay_fast_on_files_of_any_size(size):
    import time

    body = "".join(f"line_{i} = compute({i})\n" for i in range(size))
    t0 = time.perf_counter()
    _closest_hints(body, f"line_{size - 1} = compute({size - 1})\nq", 3)
    elapsed = time.perf_counter() - t0
    assert elapsed < 2.0, f"hints took {elapsed:.2f}s on a {size}-line file"


# --- D. one scan per fuzzy edit -----------------------------------------

def test_fuzzy_edit_scans_the_file_once(tmp_path, monkeypatch):
    calls = []
    real = E.find_matches

    def counted(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(E, "find_matches", counted)
    p = tmp_path / "f.py"
    p.write_text("def f():\n    a = 1\n    b = 2\n")
    r = _edit(str(p), "  a =  1", "a = 9")
    assert r.get("error") is None
    assert len(calls) == 1


def test_candidate_spans_accepts_a_precomputed_exact_list():
    content = "a\nb\na\n"
    exact = E.find_matches(content, "a")
    assert E._candidate_spans(content, "a", exact=exact) == exact


# --- E. dry runs leave no trace -----------------------------------------

def test_dry_run_does_not_track_the_file(tmp_path, monkeypatch):
    from opencode_py.tools import verify as V

    calls = []
    monkeypatch.setattr(V, "track", lambda path, source, *a, **kw: calls.append(source))
    p = tmp_path / "d.txt"
    p.write_text("a\nb\nc\n")
    r = _edit(str(p), "b", "B", dry_run=True)
    assert r.get("error") is None
    assert calls == []
    assert p.read_text() == "a\nb\nc\n"


def test_line_range_dry_run_does_not_track(tmp_path, monkeypatch):
    from opencode_py.tools import verify as V

    calls = []
    monkeypatch.setattr(V, "track", lambda path, source, *a, **kw: calls.append(source))
    p = tmp_path / "d2.txt"
    p.write_text("a\nb\nc\n")
    _edit(str(p), "", "B", offset=2, limit=1, dry_run=True)
    assert calls == []


def test_real_edit_still_tracks_after_a_dry_run(tmp_path, monkeypatch):
    from opencode_py.tools import verify as V

    calls = []
    monkeypatch.setattr(V, "track", lambda path, source, *a, **kw: calls.append(source))
    p = tmp_path / "d3.txt"
    p.write_text("a\nb\nc\n")
    _edit(str(p), "b", "B", dry_run=True)
    _edit(str(p), "b", "B")
    assert calls == ["edit"]
    assert p.read_text() == "a\nB\nc\n"


# --- the six original fixes are still in place ---------------------------

def test_no_unused_imports_came_back():
    import inspect

    src = inspect.getsource(E)
    import ast

    imported = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            imported.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.update(a.asname or a.name for a in node.names)
    assert "tempfile" not in imported
    assert not hasattr(E, "tempfile")
    # the guard's env switch is read in write.py now; edit.py must not import
    # os just for it
    assert "os" not in imported or "os." in inspect.getsource(E)


def test_candidate_spans_docstring_is_still_live():
    assert E._candidate_spans.__doc__
    assert "cascade" in E._candidate_spans.__doc__


def test_no_disabled_code_left_in_the_module():
    import inspect

    for fn in (E._span_diff, E._edit_lines, E._candidate_spans, E._closest_hints):
        assert "if False" not in inspect.getsource(fn)

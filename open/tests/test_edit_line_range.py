"""Line-range mode (offset/limit) of the edit tool: bug-fix regressions.

Before these fixes the line-range path was untested and carried six defects:
CRLF files were rewritten to LF, non-UTF-8 bytes were replaced with U+FFFD
while reporting success, no diff was produced (the TUI renders nothing without
metadata.diff), verify.track was never called, _candidate_spans had an
unreachable docstring, and os/tempfile were imported but unused.
"""

import inspect

from opencode_py.tools import edit as edit_mod
from opencode_py.tools.edit import _edit


CRLF = b"\x0d\x0a"
FFFD = b"\xef\xbf\xbd"


def _diff_body(result):
    """Diff text without the ---/+++ header lines (they carry the file path)."""
    return "\n".join((result.get("metadata") or {}).get("diff", "").splitlines()[2:])


# --- bug 1: CRLF must survive a line-range edit -------------------------

def test_line_range_preserves_crlf(tmp_path):
    p = tmp_path / "win.txt"
    p.write_bytes(b"alpha" + CRLF + b"beta" + CRLF + b"gamma" + CRLF + b"delta" + CRLF)
    r = _edit(str(p), "", "BETA2", offset=2, limit=1)
    assert r.get("error") is None
    assert p.read_bytes() == b"alpha" + CRLF + b"BETA2" + CRLF + b"gamma" + CRLF + b"delta" + CRLF


def test_line_range_crlf_multiline_replacement_uses_crlf(tmp_path):
    p = tmp_path / "win2.txt"
    p.write_bytes(b"a" + CRLF + b"b" + CRLF + b"c" + CRLF)
    r = _edit(str(p), "", "x\ny\nz", offset=2, limit=1)
    assert r.get("error") is None
    assert p.read_bytes() == b"a" + CRLF + b"x" + CRLF + b"y" + CRLF + b"z" + CRLF + b"c" + CRLF


def test_line_range_lf_file_stays_lf(tmp_path):
    p = tmp_path / "unix.txt"
    p.write_bytes(b"a\nb\nc\n")
    r = _edit(str(p), "", "B", offset=2, limit=1)
    assert r.get("error") is None
    assert p.read_bytes() == b"a\nB\nc\n"


def test_line_range_no_trailing_newline_preserved(tmp_path):
    p = tmp_path / "nonl.txt"
    p.write_bytes(b"a\nb\nc")
    r = _edit(str(p), "", "B", offset=2, limit=1)
    assert r.get("error") is None
    assert p.read_bytes() == b"a\nB\nc"


def test_line_range_crlf_and_string_mode_agree(tmp_path):
    """Both modes must produce byte-identical results for the same change."""
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    raw = b"one" + CRLF + b"two" + CRLF + b"three" + CRLF
    a.write_bytes(raw)
    b.write_bytes(raw)
    r1 = _edit(str(a), "", "TWO", offset=2, limit=1)
    r2 = _edit(str(b), "two", "TWO")
    assert r1.get("error") is None and r2.get("error") is None
    assert a.read_bytes() == b.read_bytes()


# --- bug 2: unreadable bytes must never be rewritten silently -----------

def test_line_range_non_utf8_refuses_and_leaves_file_alone(tmp_path):
    p = tmp_path / "latin.txt"
    raw = b"caf\xe9 latin1" + b"\nsecond line\nthird line\n"
    p.write_bytes(raw)
    r = _edit(str(p), "", "CHANGED", offset=2, limit=1)
    assert r.get("error") is True
    assert "could not read" in r["output"].lower()
    assert p.read_bytes() == raw
    assert FFFD not in p.read_bytes()


def test_line_range_utf8_multibyte_still_edits(tmp_path):
    p = tmp_path / "utf8.txt"
    p.write_text("héllo\nwörld\ntail\n", encoding="utf-8")
    r = _edit(str(p), "", "WÖRLD", offset=2, limit=1)
    assert r.get("error") is None
    assert p.read_text(encoding="utf-8") == "héllo\nWÖRLD\ntail\n"


# --- bug 3: a real diff must be produced and shown in full --------------

def test_line_range_metadata_has_diff(tmp_path):
    p = tmp_path / "d.txt"
    p.write_text("".join(f"row{i}\n" for i in range(1, 60)))
    r = _edit(str(p), "", "X", offset=10, limit=1)
    assert "diff" in (r.get("metadata") or {})
    assert (r["metadata"]["diff"]).startswith("--- a/")


def test_line_range_diff_is_not_truncated_to_ten_lines(tmp_path):
    p = tmp_path / "big.txt"
    p.write_text("".join(f"row{i}\n" for i in range(1, 60)))
    r = _edit(str(p), "", "\n".join(f"new{i}" for i in range(1, 41)), offset=1, limit=40)
    diff = (r.get("metadata") or {}).get("diff", "")
    removed = [l for l in diff.splitlines() if l.startswith("-") and not l.startswith("---")]
    added = [l for l in diff.splitlines() if l.startswith("+") and not l.startswith("+++")]
    assert len(removed) == 40
    assert len(added) == 40
    assert "-row40" in diff
    assert "+new40" in diff


def test_line_range_diff_matches_string_mode_for_same_change(tmp_path):
    """Both modes must agree on the resulting file AND on the rendered diff."""
    body = "".join(f"row{i}\n" for i in range(1, 60))
    old = "\n".join(f"row{i}" for i in range(1, 41))
    new = "\n".join(f"new{i}" for i in range(1, 41))
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_text(body)
    b.write_text(body)
    r_str = _edit(str(a), old, new)
    r_rng = _edit(str(b), "", new, offset=1, limit=40)
    assert r_str.get("error") is None and r_rng.get("error") is None
    assert a.read_bytes() == b.read_bytes()
    assert _diff_body(r_str) == _diff_body(r_rng)


def test_line_range_crlf_diff_matches_string_mode(tmp_path):
    crlf = CRLF.decode()
    body = "".join(f"row{i}" + crlf for i in range(1, 30))
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_bytes(body.encode())
    b.write_bytes(body.encode())
    r_str = _edit(str(a), "row5", "X")
    r_rng = _edit(str(b), "", "X", offset=5, limit=1)
    assert r_str.get("error") is None and r_rng.get("error") is None
    assert a.read_bytes() == b.read_bytes()
    assert a.read_bytes().count(CRLF) == body.encode().count(CRLF)
    assert _diff_body(r_str) == _diff_body(r_rng)


def test_line_range_dry_run_diff_and_no_write(tmp_path):
    p = tmp_path / "dry.txt"
    p.write_text("a\nb\nc\n")
    r = _edit(str(p), "", "B", offset=2, limit=1, dry_run=True)
    assert r.get("error") is None
    assert (r.get("metadata") or {}).get("dry_run") is True
    assert "-b" in r["output"]
    assert p.read_text() == "a\nb\nc\n"


def test_line_range_diff_shows_real_removed_and_added_lines(tmp_path):
    p = tmp_path / "vis.txt"
    p.write_text("keep1\nOLDLINE\nkeep2\n")
    r = _edit(str(p), "", "NEWLINE", offset=2, limit=1)
    diff = r["metadata"]["diff"]
    assert "-OLDLINE" in diff
    assert "+NEWLINE" in diff
    assert " keep1" in diff  # context survives


# --- bug 4: verify.track must fire for line-range edits ------------------

def test_line_range_edit_is_tracked(tmp_path, monkeypatch):
    from opencode_py.tools import verify as V

    calls = []
    monkeypatch.setattr(V, "track", lambda path, source, *a, **kw: calls.append((str(path), source)))
    p = tmp_path / "t.txt"
    p.write_text("a\nb\nc\n")
    r = _edit(str(p), "", "B", offset=2, limit=1)
    assert r.get("error") is None
    assert len(calls) == 1
    assert calls[0][1] == "edit"


def test_failed_line_range_edit_is_not_tracked(tmp_path, monkeypatch):
    from opencode_py.tools import verify as V

    calls = []
    monkeypatch.setattr(V, "track", lambda path, source, *a, **kw: calls.append(source))
    p = tmp_path / "t2.txt"
    p.write_text("a\nb\n")
    r = _edit(str(p), "", "Z", offset=99, limit=1)
    assert r.get("error") is True
    assert calls == []


def test_string_mode_still_tracked(tmp_path, monkeypatch):
    from opencode_py.tools import verify as V

    calls = []
    monkeypatch.setattr(V, "track", lambda path, source, *a, **kw: calls.append(source))
    p = tmp_path / "t3.txt"
    p.write_text("a\nb\nc\n")
    _edit(str(p), "b", "B")
    assert calls == ["edit"]


# --- bug 5: the docstring must actually exist ---------------------------

def test_candidate_spans_docstring_is_live():
    assert edit_mod._candidate_spans.__doc__
    assert "cascade" in edit_mod._candidate_spans.__doc__


def test_candidate_spans_doc_is_reachable_not_under_the_guard():
    import inspect

    lines = inspect.getsource(edit_mod._candidate_spans).splitlines()
    doc = next(i for i, l in enumerate(lines) if l.strip().startswith('"""'))
    guard = next(i for i, l in enumerate(lines) if l.strip().startswith("if len(old)"))
    assert doc < guard


# --- bug 6: no unused imports -------------------------------------------

def test_no_unused_tempfile_import(monkeypatch):
    """tempfile is never needed, and the env switch really is wired up."""
    assert not hasattr(edit_mod, "tempfile")
    assert not hasattr(edit_mod, "os") or "os." in inspect.getsource(edit_mod)
    # behaviour, not source text: the switch must actually change the guard
    monkeypatch.delenv("OPENCODE_EDIT_GUARD", raising=False)
    assert edit_mod._guard_enabled() is True
    assert edit_mod._guard_allow_expensive() is False
    monkeypatch.setenv("OPENCODE_EDIT_GUARD", "off")
    assert edit_mod._guard_enabled() is False
    monkeypatch.setenv("OPENCODE_EDIT_GUARD", "all")
    assert edit_mod._guard_enabled() is True
    assert edit_mod._guard_allow_expensive() is True


# --- behaviour that must not regress ------------------------------------

def test_offset_past_end_errors_without_writing(tmp_path):
    p = tmp_path / "short.txt"
    p.write_text("a\nb\n")
    r = _edit(str(p), "", "Z", offset=50, limit=1)
    assert r.get("error") is True
    assert "past end" in r["output"].lower()
    assert p.read_text() == "a\nb\n"


def test_oldstring_takes_precedence_over_line_range(tmp_path):
    p = tmp_path / "prec.txt"
    p.write_text("a\nb\nc\nd\n")
    r = _edit(str(p), "c", "C", offset=2, limit=1)
    assert r.get("error") is None
    assert p.read_text() == "a\nb\nC\nd\n"


def test_line_range_metadata_reports_offset_and_limit(tmp_path):
    p = tmp_path / "meta.txt"
    p.write_text("a\nb\nc\nd\ne\n")
    r = _edit(str(p), "", "X\nY", offset=2, limit=2)
    meta = r["metadata"]
    assert meta["offset"] == 2
    assert meta["limit"] == 2
    assert p.read_text() == "a\nX\nY\nd\ne\n"

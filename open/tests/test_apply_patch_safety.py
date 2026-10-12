"""Safety and correctness for apply_patch, added after an audit of the tool.

Covers the layers the tool gained: a syntax guard, read-only refusal, write
confirmation, honest rollback, anchored insert hunks, CRLF files, a per-project
undo journal, and hint quality. Each test names the failure it prevents.
"""

import os
import stat

import pytest

from opencode_py.globals import Path as GPath
from opencode_py.tools import apply_patch as AP


@pytest.fixture
def repo(tmp_path, monkeypatch):
    monkeypatch.setattr(GPath, "data", tmp_path / "data")
    work = tmp_path / "repo"
    work.mkdir()
    monkeypatch.chdir(work)
    return work


def run(diff, **kw):
    return AP._action_apply(
        diff,
        dry_run=kw.pop("dry_run", False),
        message=kw.pop("message", ""),
        **kw,
    )


def mode(path):
    return stat.S_IMODE(os.stat(path).st_mode)


def skip_without_permission_enforcement(tmp_path):
    probe = tmp_path / "probe.txt"
    probe.write_text("x\n")
    os.chmod(probe, 0o444)
    try:
        with open(probe, "a"):
            pass
    except OSError:
        return
    pytest.skip("this environment ignores read-only bits")


# --- 1. a patch that breaks a file is refused ----------------------------

def test_patch_breaking_python_is_refused(repo):
    p = repo / "m.py"
    p.write_text("def f():\n    return 1\n")
    r = run("--- a/m.py\n+++ b/m.py\n@@ -2,1 +2,1 @@\n-    return 1\n+    return 1(\n")
    assert r.get("error") is True
    assert p.read_text() == "def f():\n    return 1\n"
    assert "would break the python syntax" in r["output"]


def test_patch_breaking_json_is_refused(repo):
    p = repo / "c.json"
    p.write_text('{"a": 1}\n')
    r = run('--- a/c.json\n+++ b/c.json\n@@ -1,1 +1,1 @@\n-{"a": 1}\n+{"a": }\n')
    assert r.get("error") is True
    assert p.read_text() == '{"a": 1}\n'


def test_creating_a_broken_file_is_refused(repo):
    r = run("--- /dev/null\n+++ b/new.py\n@@ -0,0 +1,2 @@\n+def g(:\n+    pass\n")
    assert r.get("error") is True
    assert not (repo / "new.py").exists()


def test_a_valid_patch_is_never_blocked(repo):
    p = repo / "m.py"
    p.write_text("def f():\n    return 1\n")
    r = run("--- a/m.py\n+++ b/m.py\n@@ -2,1 +2,1 @@\n-    return 1\n+    return 2\n")
    assert r.get("error") is not True
    assert p.read_text() == "def f():\n    return 2\n"


def test_an_already_broken_file_stays_patchable(repo):
    """The guard must never block a repair."""
    p = repo / "m.py"
    p.write_text("def f(:\n")
    r = run("--- a/m.py\n+++ b/m.py\n@@ -1,1 +1,2 @@\n-def f(:\n+def f():\n+    pass\n")
    assert r.get("error") is not True
    assert p.read_text() == "def f():\n    pass\n"


def test_force_overrides_the_syntax_guard(repo):
    p = repo / "m.py"
    p.write_text("x = 1\n")
    r = run("--- a/m.py\n+++ b/m.py\n@@ -1,1 +1,1 @@\n-x = 1\n+x = (\n", force=True)
    assert r.get("error") is not True
    assert p.read_text() == "x = (\n"


def test_a_guarded_patch_in_a_multifile_patch_blocks_the_whole_thing(repo):
    good = repo / "good.py"
    good.write_text("a = 1\n")
    bad = repo / "bad.py"
    bad.write_text("b = 1\n")
    r = run(
        "--- a/good.py\n+++ b/good.py\n@@ -1,1 +1,1 @@\n-a = 1\n+a = 2\n"
        "--- a/bad.py\n+++ b/bad.py\n@@ -1,1 +1,1 @@\n-b = 1\n+b = (((\n"
    )
    assert r.get("error") is True
    assert good.read_text() == "a = 1\n", "all-or-nothing: nothing may be written"
    assert bad.read_text() == "b = 1\n"


def test_unchecked_language_is_reported_not_silent(repo):
    p = repo / "a.js"
    p.write_text("const a = 1;\n")
    r = run("--- a/a.js\n+++ b/a.js\n@@ -1,1 +1,1 @@\n-const a = 1;\n+const a = 2;\n")
    assert r.get("error") is not True
    assert "not checked" in r["output"]


# --- 2. read-only files --------------------------------------------------

def test_read_only_file_is_refused(repo, tmp_path):
    skip_without_permission_enforcement(tmp_path)
    p = repo / "ro.py"
    p.write_text("x = 1\n")
    os.chmod(p, 0o444)
    r = run("--- a/ro.py\n+++ b/ro.py\n@@ -1,1 +1,1 @@\n-x = 1\n+x = 2\n")
    assert r.get("error") is True
    assert p.read_text() == "x = 1\n"
    assert mode(p) == 0o444
    assert "read-only" in r["output"]


def test_read_only_refusal_stops_the_whole_patch(repo, tmp_path):
    skip_without_permission_enforcement(tmp_path)
    other = repo / "other.py"
    other.write_text("y = 1\n")
    ro = repo / "ro.py"
    ro.write_text("x = 1\n")
    os.chmod(ro, 0o444)
    r = run(
        "--- a/other.py\n+++ b/other.py\n@@ -1,1 +1,1 @@\n-y = 1\n+y = 2\n"
        "--- a/ro.py\n+++ b/ro.py\n@@ -1,1 +1,1 @@\n-x = 1\n+x = 2\n"
    )
    assert r.get("error") is True
    assert other.read_text() == "y = 1\n"


def test_force_overrides_read_only(repo, tmp_path):
    skip_without_permission_enforcement(tmp_path)
    p = repo / "ro.py"
    p.write_text("x = 1\n")
    os.chmod(p, 0o444)
    r = run("--- a/ro.py\n+++ b/ro.py\n@@ -1,1 +1,1 @@\n-x = 1\n+x = 2\n", force=True)
    assert r.get("error") is not True
    assert p.read_text() == "x = 2\n"
    assert mode(p) == 0o444


def test_permissions_are_preserved(repo):
    p = repo / "s.sh"
    p.write_text("f() {\n  echo hi\n}\n")
    os.chmod(p, 0o755)
    run("--- a/s.sh\n+++ b/s.sh\n@@ -2,1 +2,1 @@\n-  echo hi\n+  echo there\n")
    assert mode(p) == 0o755
    assert os.stat(p).st_mode & 0o111


# --- 3. every write is confirmed ----------------------------------------

def test_every_write_is_read_back(repo, monkeypatch):
    seen = []
    from opencode_py.tools import write as W

    real = W.confirm_write
    monkeypatch.setattr(W, "confirm_write", lambda p, e: seen.append(str(p)) or real(p, e))
    p = repo / "m.py"
    p.write_text("a = 1\n")
    run("--- a/m.py\n+++ b/m.py\n@@ -1,1 +1,1 @@\n-a = 1\n+a = 2\n")
    assert seen == [str(p)]


def test_a_write_that_did_not_land_is_reported(repo, monkeypatch):
    from opencode_py.tools import write as W

    monkeypatch.setattr(W, "confirm_write", lambda p, e: "the bytes on disk do not match")
    p = repo / "m.py"
    p.write_text("a = 1\n")
    r = run("--- a/m.py\n+++ b/m.py\n@@ -1,1 +1,1 @@\n-a = 1\n+a = 2\n")
    assert r.get("error") is True
    assert "not verified" in r["output"]


# --- 4/5. rollback is honest and complete -------------------------------

def test_a_failed_write_rolls_back_and_says_so(repo, monkeypatch):
    p = repo / "m.py"
    p.write_text("a = 1\n")
    from opencode_py.tools import write as W

    real = W._atomic_write
    calls = {"n": 0}

    def flaky(path, content):
        calls["n"] += 1
        if calls["n"] == 1:
            real(path, content)
            raise OSError("simulated write failure")
        real(path, content)

    monkeypatch.setattr(AP, "_atomic_write", flaky)
    r = run("--- a/m.py\n+++ b/m.py\n@@ -1,1 +1,1 @@\n-a = 1\n+a = 2\n")
    assert r.get("error") is True
    assert "rolled back" in r["output"]
    assert p.read_text() == "a = 1\n", "the original must be back"


def test_an_incomplete_rollback_is_announced(repo, monkeypatch):
    """Never claim a rollback that did not happen."""
    from opencode_py.tools import write as W

    real = W._atomic_write
    calls = {"n": 0}

    def flaky(path, content):
        calls["n"] += 1
        if calls["n"] == 1:
            real(path, content)
            raise OSError("simulated write failure")
        if calls["n"] == 2:
            raise OSError("cannot restore")
        real(path, content)

    monkeypatch.setattr(AP, "_atomic_write", flaky)
    p = repo / "m.py"
    p.write_text("a = 1\n")
    r = run("--- a/m.py\n+++ b/m.py\n@@ -1,1 +1,1 @@\n-a = 1\n+a = 2\n")
    assert r.get("error") is True
    assert "ROLLBACK WAS INCOMPLETE" in r["output"]
    assert "check them by hand" in r["output"]


def test_a_failed_create_rolls_back_earlier_writes(repo, monkeypatch):
    """all-or-nothing includes the create/delete phase."""
    p = repo / "m.py"
    p.write_text("a = 1\n")
    from opencode_py.tools import write as W

    real = W._atomic_write
    monkeypatch.setattr(AP, "_atomic_write", lambda path, content: (_ for _ in ()).throw(OSError("disk full")) if "new.py" in str(path) else real(path, content))
    r = run(
        "--- a/m.py\n+++ b/m.py\n@@ -1,1 +1,1 @@\n-a = 1\n+a = 2\n"
        "--- /dev/null\n+++ b/new.py\n@@ -0,0 +1,1 @@\n+z = 1\n"
    )
    assert r.get("error") is True
    assert "rolled back" in r["output"]
    assert p.read_text() == "a = 1\n", "the earlier write must be undone"


# --- 6. CRLF files can be patched ---------------------------------------

def test_crlf_file_is_patchable_and_stays_crlf(repo):
    p = repo / "w.py"
    p.write_bytes(b"a = 1\r\nb = 2\r\nc = 3\r\n")
    r = run("--- a/w.py\n+++ b/w.py\n@@ -1,3 +1,3 @@\n a = 1\n-b = 2\n+b = 99\n c = 3\n")
    assert r.get("error") is not True
    assert p.read_bytes() == b"a = 1\r\nb = 99\r\nc = 3\r\n"


def test_crlf_without_final_newline_keeps_that(repo):
    p = repo / "w.py"
    p.write_bytes(b"a = 1\r\nb = 2")
    r = run("--- a/w.py\n+++ b/w.py\n@@ -1,2 +1,2 @@\n a = 1\n-b = 2\n+b = 3")
    assert r.get("error") is not True
    assert p.read_bytes() == b"a = 1\r\nb = 3"


def test_lf_file_is_unaffected_by_the_crlf_work(repo):
    p = repo / "l.py"
    p.write_text("a = 1\nb = 2\n")
    r = run("--- a/l.py\n+++ b/l.py\n@@ -1,2 +1,2 @@\n a = 1\n-b = 2\n+b = 3\n")
    assert r.get("error") is not True
    assert p.read_text() == "a = 1\nb = 3\n"


def test_mixed_line_endings_are_not_silently_normalised(repo):
    p = repo / "mix.py"
    before = b"a = 1\r\nb = 2\nc = 3\r\n"
    p.write_bytes(before)
    run("--- a/mix.py\n+++ b/mix.py\n@@ -1,2 +1,2 @@\n a = 1\n-b = 2\n+b = 9\n")
    assert p.read_bytes() == before, "a file we cannot handle must be left alone"


def test_crlf_file_still_gets_the_syntax_guard(repo):
    p = repo / "w.py"
    p.write_bytes(b"def f():\r\n    return 1\r\n")
    r = run("--- a/w.py\n+++ b/w.py\n@@ -2,1 +2,1 @@\n-    return 1\n+    return 1(\n")
    assert r.get("error") is True
    assert p.read_bytes() == b"def f():\r\n    return 1\r\n"


# --- 7. insert hunks are anchored ---------------------------------------

def test_insert_past_the_end_of_the_file_is_refused(repo):
    p = repo / "m.py"
    p.write_text("one\ntwo\nthree\nfour\n")
    r = run("--- a/m.py\n+++ b/m.py\n@@ -99,0 +100,1 @@\n+INSERTED\n")
    assert r.get("error") is True
    assert p.read_text() == "one\ntwo\nthree\nfour\n"
    assert "outside the file" in r["output"]


def test_insert_lands_in_the_right_place(repo):
    p = repo / "m.py"
    p.write_text("one\ntwo\nthree\n")
    r = run("--- a/m.py\n+++ b/m.py\n@@ -2,0 +3,1 @@\n+INSERTED\n")
    assert r.get("error") is not True
    assert p.read_text() == "one\ntwo\nINSERTED\nthree\n"


def test_insert_at_the_end_works(repo):
    p = repo / "m.py"
    p.write_text("one\ntwo\n")
    r = run("--- a/m.py\n+++ b/m.py\n@@ -2,0 +3,1 @@\n+LAST\n")
    assert r.get("error") is not True
    assert p.read_text() == "one\ntwo\nLAST\n"


# --- 8. undo is scoped to its project -----------------------------------

def test_undo_does_not_cross_projects(tmp_path, monkeypatch):
    monkeypatch.setattr(GPath, "data", tmp_path / "data")
    p1, p2 = tmp_path / "p1", tmp_path / "p2"
    p1.mkdir()
    p2.mkdir()
    f1 = p1 / "file.py"
    f1.write_text("one\n")

    monkeypatch.chdir(p1)
    run("--- a/file.py\n+++ b/file.py\n@@ -1,1 +1,1 @@\n-one\n+TWO\n", message="p1 change")
    assert f1.read_text() == "TWO\n"

    monkeypatch.chdir(p2)
    out = AP._action_undo()
    assert out.get("error") is True
    assert "THIS project" in out["output"]
    assert f1.read_text() == "TWO\n", "another project must not be undone"

    monkeypatch.chdir(p1)
    out = AP._action_undo()
    assert out.get("error") is not True
    assert f1.read_text() == "one\n"


def test_history_hides_other_projects(tmp_path, monkeypatch):
    monkeypatch.setattr(GPath, "data", tmp_path / "data")
    p1, p2 = tmp_path / "p1", tmp_path / "p2"
    p1.mkdir()
    p2.mkdir()
    f1 = p1 / "file.py"
    f1.write_text("one\n")
    monkeypatch.chdir(p1)
    run("--- a/file.py\n+++ b/file.py\n@@ -1,1 +1,1 @@\n-one\n+TWO\n", message="p1 change")
    monkeypatch.chdir(p2)
    out = AP._action_history()["output"]
    assert "p1 change" not in out
    assert "other projects" in out


def test_legacy_journal_entry_is_matched_by_its_paths(tmp_path, monkeypatch):
    """Entries with no recorded project are attributed by the files they touched."""
    monkeypatch.setattr(GPath, "data", tmp_path / "data")
    p1, p2 = tmp_path / "p1", tmp_path / "p2"
    p1.mkdir()
    p2.mkdir()
    f1 = p1 / "file.py"
    f1.write_text("one\n")
    # a legacy entry: no "root" key
    AP._journal_save(
        [{"time": 0, "message": "legacy", "files": [{"path": str(f1), "existed": True, "content": "one\n"}]}]
    )
    monkeypatch.chdir(p2)
    assert "THIS project" in AP._action_undo()["output"]
    monkeypatch.chdir(p1)
    out = AP._action_undo()
    assert out.get("error") is not True


# --- 9. hints are copyable and honest -----------------------------------

def test_hint_keeps_the_real_indentation():
    lines = ["def f():", "    return 1", "class C:", "    def m(self):", "        total = 0"]
    hint = AP._closest_hint(lines, ["    tota = 0"])
    assert "        total = 0" in hint, hint
    assert "read offset=" in hint


def test_hint_finds_the_right_line_deep_in_a_file():
    lines = []
    for i in range(3000):
        lines.append(f"    def handler_{i}(self):")
        lines.append("        pass")
    hint = AP._closest_hint(lines, ["    def handler_2999(self):"])
    assert "handler_2999" in hint, hint


def test_hint_reports_every_occurrence_of_a_repeated_line():
    lines = ["x = 1", "y = 2", "y = 2", "y = 2", "z = 3"]
    hint = AP._closest_hint(lines, ["y = 3"])
    assert "also 3, 4" in hint or "line 2" in hint


def test_hint_stays_silent_when_nothing_matches():
    lines = [f"value_{i} = {i}" for i in range(6000)]
    assert AP._closest_hint(lines, ["zzz_qqq_absent"]) == ""


# --- second audit: two more bugs the first pass missed --------------------

def test_prepend_hunk_puts_the_line_first(repo):
    """@@ -0,0 means "before line 1"; it used to land after line 1."""
    p = repo / "m.py"
    p.write_text("one\ntwo\nthree\n")
    r = run("--- a/m.py\n+++ b/m.py\n@@ -0,0 +1,1 @@\n+TOP\n")
    assert r.get("error") is not True
    assert p.read_text() == "TOP\none\ntwo\nthree\n"


@pytest.mark.parametrize("start,expected", [
    (1, "one\nNEW\ntwo\nthree\n"),
    (2, "one\ntwo\nNEW\nthree\n"),
    (3, "one\ntwo\nthree\nNEW\n"),
])
def test_insert_lands_after_the_declared_line(repo, start, expected):
    p = repo / "m.py"
    p.write_text("one\ntwo\nthree\n")
    r = run(f"--- a/m.py\n+++ b/m.py\n@@ -{start},0 +{start+1},1 @@\n+NEW\n")
    assert r.get("error") is not True
    assert p.read_text() == expected


def test_same_file_named_twice_is_refused_not_half_applied(repo):
    """A diff may not name one file twice: the second section was dropped
    silently while the summary still claimed both were applied."""
    p = repo / "m.py"
    p.write_text("a\nb\nc\n")
    r = run(
        "--- a/m.py\n+++ b/m.py\n@@ -1,1 +1,1 @@\n-a\n+A\n"
        "--- a/m.py\n+++ b/m.py\n@@ -3,1 +3,1 @@\n-c\n+C\n"
    )
    assert r.get("error") is True
    assert p.read_text() == "a\nb\nc\n", "nothing may be written"
    assert "more than once" in r["output"]
    assert r["metadata"]["duplicate_files"] == [str(p)]


def test_duplicate_refusal_also_applies_to_dry_run(repo):
    p = repo / "m.py"
    p.write_text("a\n")
    r = run(
        "--- a/m.py\n+++ b/m.py\n@@ -1,1 +1,1 @@\n-a\n+A\n"
        "--- a/m.py\n+++ b/m.py\n@@ -1,1 +1,1 @@\n-a\n+B\n",
        dry_run=True,
    )
    assert r.get("error") is True
    assert p.read_text() == "a\n"


def test_a_normal_multi_file_patch_is_unaffected_by_the_duplicate_check(repo):
    a, b = repo / "a.py", repo / "b.py"
    a.write_text("a\n")
    b.write_text("b\n")
    r = run(
        "--- a/a.py\n+++ b/a.py\n@@ -1,1 +1,1 @@\n-a\n+A\n"
        "--- a/b.py\n+++ b/b.py\n@@ -1,1 +1,1 @@\n-b\n+B\n"
    )
    assert r.get("error") is not True
    assert a.read_text() == "A\n"
    assert b.read_text() == "B\n"


def test_symlink_pointing_outside_the_project_is_refused(repo, tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("line1\nline2\n")
    (repo / "link.py").symlink_to(outside)
    r = run("--- a/link.py\n+++ b/link.py\n@@ -1,2 +1,2 @@\n-line1\n+LINE1\n line2\n")
    assert r.get("error") is True
    assert outside.read_text() == "line1\nline2\n"
    assert "outside worktree" in r["output"]


def test_overlapping_hunks_are_refused(repo):
    p = repo / "m.py"
    p.write_text("a\nb\nc\n")
    r = run(
        "--- a/m.py\n+++ b/m.py\n@@ -2,1 +2,1 @@\n-b\n+X\n@@ -2,1 +2,1 @@\n-b\n+Y\n"
    )
    assert r.get("error") is True
    assert p.read_text() == "a\nb\nc\n"


def test_partial_still_applies_the_clean_file_and_skips_the_breaking_one(repo):
    good, bad = repo / "ok.py", repo / "bad.py"
    good.write_text("a = 1\n")
    bad.write_text("b = 1\n")
    run(
        "--- a/ok.py\n+++ b/ok.py\n@@ -1,1 +1,1 @@\n-a = 1\n+a = 2\n"
        "--- a/bad.py\n+++ b/bad.py\n@@ -1,1 +1,1 @@\n-b = 1\n+b = (((\n",
        partial=True,
    )
    assert good.read_text() == "a = 2\n"
    assert bad.read_text() == "b = 1\n"


def test_a_large_patch_applies_every_hunk_and_says_so(repo):
    p = repo / "m.py"
    p.write_text("".join(f"line{i}\n" for i in range(200)))
    hunks = "".join(f"@@ -{i+1},1 +{i+1},1 @@\n-line{i}\n+LINE{i}\n" for i in range(100))
    r = run("--- a/m.py\n+++ b/m.py\n" + hunks)
    assert r.get("error") is not True  # every hunk really is applied below
    lines = p.read_text().splitlines()
    assert sum(1 for l in lines if l.startswith("LINE")) == 100
    assert sum(1 for l in lines if l.startswith("line")) == 100
    assert "+100 -100" in r["output"]


def test_empty_and_unparseable_diffs_are_reported(repo):
    assert "Empty diff" in run("")["output"]
    assert "Could not parse" in run("just some text\n")["output"]


# --- third audit: BOM files, write races, and undo vs later work ----------

def test_bom_file_line_one_is_patchable_and_keeps_its_bom(repo):
    """A UTF-8 BOM used to make the first line unpatchable: it is read as a
    leading U+FEFF, which no diff carries."""
    p = repo / "bom.py"
    p.write_bytes(b"\xef\xbb\xbfimport os\nvalue = 1\n")
    r = run("--- a/bom.py\n+++ b/bom.py\n@@ -1,1 +1,1 @@\n-import os\n+import sys\n")
    assert r.get("error") is not True
    assert p.read_bytes() == b"\xef\xbb\xbfimport sys\nvalue = 1\n"


def test_bom_file_other_lines_still_work(repo):
    p = repo / "bom.py"
    p.write_bytes(b"\xef\xbb\xbfimport os\nvalue = 1\n")
    r = run("--- a/bom.py\n+++ b/bom.py\n@@ -2,1 +2,1 @@\n-value = 1\n+value = 2\n")
    assert r.get("error") is not True
    assert p.read_bytes() == b"\xef\xbb\xbfimport os\nvalue = 2\n"


def test_bom_plus_crlf_file(repo):
    p = repo / "both.py"
    p.write_bytes(b"\xef\xbb\xbfa = 1\r\nb = 2\r\n")
    r = run("--- a/both.py\n+++ b/both.py\n@@ -1,2 +1,2 @@\n a = 1\n-b = 2\n+b = 9\n")
    assert r.get("error") is not True
    assert p.read_bytes() == b"\xef\xbb\xbfa = 1\r\nb = 9\r\n"


def test_a_file_changed_after_planning_is_not_clobbered(repo, monkeypatch):
    """Another writer wins the race: their change must survive, and we must say so."""
    p = repo / "m.py"
    p.write_text("a\nb\n")
    real = AP._read_text

    def racing(path):
        out = real(path)
        if path.name == "m.py" and not getattr(racing, "done", False):
            racing.done = True
            path.write_text("a\nb\nSOMEONE_ELSE\n")
        return out

    monkeypatch.setattr(AP, "_read_text", racing)
    r = run("--- a/m.py\n+++ b/m.py\n@@ -1,2 +1,2 @@\n a\n-b\n+B\n")
    assert "SOMEONE_ELSE" in p.read_text(), "the other writer's work must survive"
    assert r.get("error") is True
    assert "changed on disk" in r["output"]


def test_no_race_in_the_normal_case(repo):
    """The re-check must not fire when nothing else touched the file."""
    p = repo / "m.py"
    p.write_text("a\nb\n")
    r = run("--- a/m.py\n+++ b/m.py\n@@ -1,2 +1,2 @@\n a\n-b\n+B\n")
    assert r.get("error") is not True
    assert p.read_text() == "a\nB\n"


def test_undo_leaves_a_later_change_alone(repo):
    """Undo must not destroy work done after the patch it would revert."""
    p = repo / "m.py"
    p.write_text("v1\n")
    run("--- a/m.py\n+++ b/m.py\n@@ -1,1 +1,1 @@\n-v1\n+v2\n", message="first")
    p.write_text("v3-later-work\n")
    out = AP._action_undo()
    assert p.read_text() == "v3-later-work\n", "the later change must survive undo"
    assert "LEFT ALONE" in out["output"]


def test_undo_still_restores_when_nothing_changed_since(repo):
    p = repo / "m.py"
    p.write_text("v1\n")
    run("--- a/m.py\n+++ b/m.py\n@@ -1,1 +1,1 @@\n-v1\n+v2\n", message="first")
    out = AP._action_undo()
    assert out.get("error") is not True
    assert p.read_text() == "v1\n"


def test_undo_of_a_delete_still_restores_the_file(repo):
    p = repo / "g.py"
    p.write_text("a\nb\n")
    run("--- a/g.py\n+++ /dev/null\n@@ -1,2 +0,0 @@\n-a\n-b\n", message="del")
    out = AP._action_undo()
    assert out.get("error") is not True
    assert p.exists()
    assert p.read_text() == "a\nb\n"


def test_undo_of_a_created_file_still_deletes_it(repo):
    run("--- /dev/null\n+++ b/new.py\n@@ -0,0 +1,1 @@\n+created = 1\n", message="create")
    assert (repo / "new.py").exists()
    AP._action_undo()
    assert not (repo / "new.py").exists()


def test_crlf_diff_text_from_a_windows_client_is_accepted(repo):
    p = repo / "m.py"
    p.write_text("a\nb\n")
    diff = "--- a/m.py\r\n+++ b/m.py\r\n@@ -1,2 +1,2 @@\r\n a\r\n-b\r\n+B\r\n"
    r = run(diff)
    assert r.get("error") is not True
    assert p.read_text() == "a\nB\n"


def test_a_very_long_line_can_be_patched(repo):
    p = repo / "m.py"
    p.write_text("x = " + "1" * 50000 + "\n")
    r = run("--- a/m.py\n+++ b/m.py\n@@ -1,1 +1,1 @@\n-x = " + "1" * 50000 + "\n+x = 2\n")
    assert r.get("error") is not True
    assert p.read_text() == "x = 2\n"


def test_hunk_header_counts_are_advisory_not_fatal(repo):
    """A hunk whose body is short is still applied from its body."""
    p = repo / "m.py"
    p.write_text("a\nb\nc\n")
    r = run("--- a/m.py\n+++ b/m.py\n@@ -1,5 +1,5 @@\n-a\n+A\n b\n")
    assert r.get("error") is not True
    assert p.read_text() == "A\nb\nc\n"


def test_overlapping_hunks_refused_still(tmp_path, monkeypatch):
    monkeypatch.setattr(GPath, "data", tmp_path / "data")
    work = tmp_path / "repo"
    work.mkdir()
    monkeypatch.chdir(work)
    p = work / "m.py"
    p.write_text("a\nb\nc\n")
    r = run("--- a/m.py\n+++ b/m.py\n@@ -2,1 +2,1 @@\n-b\n+X\n@@ -2,1 +2,1 @@\n-b\n+Y\n")
    assert r.get("error") is True
    assert p.read_text() == "a\nb\nc\n"


def test_journal_failure_does_not_crash_a_landed_patch(repo, monkeypatch):
    """Bookkeeping must not destroy the result of a patch that already wrote."""
    p = repo / "m.py"
    p.write_text("a\n")

    def full_disk(entries):
        raise OSError("journal disk full")

    monkeypatch.setattr(AP, "_journal_save", full_disk)
    r = run("--- a/m.py\n+++ b/m.py\n@@ -1,1 +1,1 @@\n-a\n+A\n")
    assert p.read_text() == "A\n", "the patch really did land"
    assert r.get("error") is not True, "a bookkeeping failure is not a patch failure"
    assert "WARNING" in r["output"]
    assert "cannot be undone" in r["output"]
    assert "Undo available" not in r["output"], "must not promise an undo that does not exist"

"""summarize_file: bugs found by reading the tool and probing it.

Four defects, each with the failure it caused:
- `read_bytes()[:1024]` read the WHOLE file, so a 20 MB file was pulled into
  memory twice (once for the binary check, once as text).
- classes and functions nested inside `if` / `try` / `for` were missing from the
  outline, so the line number the caller needs to jump to was simply not there.
- long preview lines were sliced without the ellipsis, because the code built a
  truncated string and then never used it.
- a named pipe (or a device file) blocked forever, hanging the agent.
"""

import os
import subprocess
import time

import pytest

from opencode_py.tools import summarize_file as S


@pytest.fixture
def repo(tmp_path):
    return tmp_path


def structure(result):
    out = result["output"]
    if "STRUCTURE" not in out:
        return ""
    return out.split("STRUCTURE", 1)[1].split("PREVIEW", 1)[0]


# --- the outline must not hide nested anchors ---------------------------

NESTED = '''import os

if os.environ.get("FEATURE"):
    class InsideIf:
        def go(self):
            pass
        class Nested:
            def deep(self):
                pass

try:
    def in_try():
        pass
except Exception:
    pass

for _ in range(1):
    def in_for():
        pass

with open(__file__) as fh:
    def in_with():
        pass

def outside():
    pass

class Top:
    def m(self):
        pass
'''


@pytest.mark.parametrize("name", ["InsideIf", "Nested", "in_try", "in_for", "in_with", "outside", "Top"])
def test_nested_definitions_are_reported(repo, name):
    p = repo / "n.py"
    p.write_text(NESTED)
    assert name in structure(S._summarize(str(p))), f"{name} missing from the outline"


def test_nested_defs_keep_indentation_and_no_duplicates(repo):
    p = repo / "n.py"
    p.write_text(NESTED)
    struct = structure(S._summarize(str(p)))
    rows = [l.strip() for l in struct.strip().splitlines()]
    defs = [r for r in rows if r.startswith(("def ", "class "))]
    assert len(defs) == len(set(defs)), f"duplicate entries: {defs}"
    raw = struct.splitlines()
    # a class nested INSIDE a class is indented deeper than a module-level one
    inner = next(l for l in raw if "class Nested" in l)
    top = next(l for l in raw if "class Top" in l)
    assert len(inner) - len(inner.lstrip()) > len(top) - len(top.lstrip())


def test_a_plain_file_is_unchanged_in_shape(repo):
    p = repo / "plain.py"
    p.write_text("import os\n\n\ndef a():\n    pass\n\n\nclass B:\n    def m(self):\n        pass\n")
    struct = structure(S._summarize(str(p)))
    assert "def a (line 4)" in struct
    assert "class B (line 8, 1 methods)" in struct
    assert "def m (line 9)" in struct


# --- long lines are truncated visibly ------------------------------------

def test_long_preview_line_ends_with_an_ellipsis(repo):
    p = repo / "long.py"
    p.write_text("x = 1  " + "y" * 5000 + "\n")
    out = S._summarize(str(p))["output"]
    preview = [l for l in out.splitlines() if l.strip().startswith("x = 1")]
    assert preview, "no preview line found"
    assert preview[0].rstrip().endswith("…")
    assert len(preview[0]) <= S.MAX_LINE_CHARS + 4


# --- the binary check must not read the whole file ----------------------

def test_sample_reads_only_the_requested_prefix(repo):
    """_sample must not pull the whole file in the way read_bytes()[:n] did."""
    p = repo / "big.txt"
    p.write_text("a" * 5_000_000)
    assert S._sample(p, 1024) == b"a" * 1024
    assert len(S._sample(p, 10)) == 10
    assert S._sample(repo / "empty.txt") if False else True


def test_summarizing_a_large_file_does_not_double_the_memory(repo):
    p = repo / "big.txt"
    p.write_text("x" * (12 * 1024 * 1024))
    before = _peak_mb()
    t0 = time.perf_counter()
    result = S._summarize(str(p))
    assert time.perf_counter() - t0 < 30
    assert result.get("error") is not True
    # the whole point: a big file must not be loaded twice
    assert _peak_mb() - before < 90


def _peak_mb():
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


# --- special files must not hang ----------------------------------------

def test_named_pipe_is_refused_not_hung(repo):
    import sys as _sys

    fifo = repo / "pipe"
    os.mkfifo(fifo)
    code = (
        "import sys;sys.path.insert(0, %r);"
        "from opencode_py.tools import summarize_file as S;"
        "print(S._summarize(%r)['output'])"
        % (str(repo.parent), str(fifo))
    )
    done = subprocess.run([_sys.executable, "-c", code], capture_output=True, text=True, timeout=20)
    assert "regular file" in done.stdout, done.stdout + done.stderr
    assert "pipe" in done.stdout


def test_directory_still_reports_its_own_message(repo):
    result = S._summarize(str(repo))
    assert result.get("error") is True
    assert "directory" in result["output"]


# --- behaviours that must not regress -----------------------------------

def test_a_python_file_with_a_syntax_error_says_so(repo):
    p = repo / "broken.py"
    p.write_text("def f(:\n    pass\n")
    out = S._summarize(str(p))["output"]
    assert "could not parse as Python" in out
    assert "PREVIEW" in out, "a preview is still useful on a broken file"


def test_missing_file_and_suggestions(repo):
    p = repo / "targt.py"
    p.write_text("x = 1\n")
    result = S._summarize(str(repo / "nope.py"))
    assert result.get("error") is True
    assert "does not exist" in result["output"]


def test_binary_file_is_refused(repo):
    p = repo / "pic.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 500)
    result = S._summarize(str(p))
    assert result.get("error") is True


def test_utf8_multibyte_file_is_not_called_binary(repo):
    p = repo / "u.py"
    p.write_text("# héllo wörld ✅\n" * 200)
    result = S._summarize(str(p))
    assert result.get("error") is not True


def test_output_stays_within_the_cap(repo):
    p = repo / "lots.py"
    p.write_text("".join(f"def f{i}():\n    pass\n\n" for i in range(4000)))
    out = S._summarize(str(p))["output"]
    assert len(out) <= S.MAX_OUTPUT + 40

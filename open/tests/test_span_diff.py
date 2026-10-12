"""_span_diff: the lazy fast path must be indistinguishable from the old one.

The diff writer can serve line slices straight from the ``\n`` offsets instead
of splitting the whole file, but only for files whose line breaks are all
``\\n``/``\\r\\n``. These tests pin that gate: whatever it decides, the rendered
diff must be byte-identical to the whole-file ``splitlines`` path.
"""

import pytest

from opencode_py.tools import edit as E


def force_whole_file_split(monkeypatch):
    """Make the fast path unreachable so the old behaviour is produced."""
    monkeypatch.setattr(E, "_lf_only_lines", lambda text: False)


def render(text, span, new, monkeypatch=None, ctx=3):
    starts = E._line_starts(text)
    n_lines = text.count("\n") + (0 if not text or text.endswith("\n") else 1)
    s = starts[min(span[0], n_lines) - 1]
    e = starts[span[1]] if span[1] < len(starts) else len(text)
    new_content = text[:s] + new + text[e:]
    return E._span_diff(text, new_content, [(s, e)], [new], "a/f", "b/f", context=ctx)


# --- the gate's own classification ---------------------------------------

def test_lf_only_lines_true_for_plain_lf():
    assert E._lf_only_lines("a\nb\nc\n") is True


def test_lf_only_lines_true_for_crlf():
    assert E._lf_only_lines("a\r\nb\r\n") is True


def test_lf_only_lines_false_for_lone_cr():
    assert E._lf_only_lines("a\rb\n") is False


def test_lf_only_lines_false_for_form_feed():
    assert E._lf_only_lines("a\x0cb\n") is False


@pytest.mark.parametrize("ch", ["\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029"])
def test_lf_only_lines_false_for_every_splitlines_breaker(ch):
    assert E._lf_only_lines("a" + ch + "b\n") is False


def test_lf_only_lines_empty_and_tiny():
    assert E._lf_only_lines("") is True
    assert E._lf_only_lines("no newline at all") is True


# --- the fast path must not change any output ---------------------------

PLAIN = "alpha\nbeta\ngamma\ndelta\nepsilon\nzeta\n"
CRLF = "alpha\r\nbeta\r\ngamma\r\ndelta\r\nepsilon\r\nzeta\r\n"
FORMFEED = "alpha\x0c\nbeta\ngamma\ndelta\n"
LONE_CR = "alpha\r\nbeta\r\ngamma\ndelta\n"
NO_FINAL_NL = "alpha\nbeta\ngamma"


@pytest.mark.parametrize("text", [PLAIN, CRLF, FORMFEED, LONE_CR, NO_FINAL_NL])
@pytest.mark.parametrize("span,new", [
    ((2, 2), "NEW"),
    ((1, 3), "X\nY"),
    ((3, 3), "ONE\nTWO\nTHREE"),
    ((1, 1), "APPENDED\n"),
    ((4, 4), ""),
])
def test_fast_path_matches_whole_file_split(text, span, new, monkeypatch):
    """Same diff whether the fast path is taken or not."""
    fast = render(text, span, new)
    force_whole_file_split(monkeypatch)
    slow = render(text, span, new)
    monkeypatch.undo()
    assert fast == slow


def test_span_between_cr_and_lf_still_matches_whole_file_split(monkeypatch):
    """Cutting a span inside a CRLF pair leaves a lone CR in the new content."""
    text = "a\r\nb\r\nc\r\n"
    starts = E._line_starts(text)
    s = starts[0] + 2  # between the "\r" and the "\n" of the first line
    e = starts[1]
    new_content = text[:s] + "NEW" + text[e:]
    fast = E._span_diff(text, new_content, [(s, e)], ["NEW"], "a/f", "b/f", context=2)
    force_whole_file_split(monkeypatch)
    slow = E._span_diff(text, new_content, [(s, e)], ["NEW"], "a/f", "b/f", context=2)
    assert fast == slow


def test_detection_reports_a_cut_inside_crlf():
    text = "a\r\nb\r\n"
    starts = E._line_starts(text)
    assert E._crlf_cut_by(text, [(starts[0] + 2, starts[1])]) is True
    assert E._crlf_cut_by(text, [(0, 1)]) is False


# --- the diff must actually describe the change -------------------------

def test_diff_reconstructs_the_new_file():
    """Walk context + added lines and compare with the real new content."""
    text = PLAIN
    starts = E._line_starts(text)
    s, e = starts[2], starts[3]
    new = "NEW1\nNEW2"
    new_content = text[:s] + new + text[e:]
    diff = E._span_diff(text, new_content, [(s, e)], [new], "a/f", "b/f", context=99)
    rebuilt = []
    in_hunk = False
    for ln in diff.splitlines(keepends=True):
        if ln.startswith("@@"):
            in_hunk = True
            continue
        if not in_hunk or ln.startswith("\\"):
            continue
        if ln[:1] in ("+", " "):
            rebuilt.append(ln[1:])
    assert "".join(rebuilt) == new_content

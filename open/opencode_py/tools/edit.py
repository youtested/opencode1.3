"""edit tool: exact string replacement + diff verification.

RULES (full guidance for maintainers; the sent schema is short):
- oldString must match EXACTLY incl. indentation (never the `N:` line
  prefix from read output). Not found -> error WITH 3 closest line hints
  so the next try hits; found many times -> add context or use replaceAll.
- dry_run=true previews the diff without writing (~0.3KB).
- Have grep offset/limit? Use offset=+limit=+newString (line_range mode):
  no verbatim copy needed, no mismatch fail.
- Prefer editing over write. Cascade tries fuzzy fallbacks for near-miss.
- SAFETY: before writing, the candidate text is syntax-checked for its own
  language (see verify.syntax_errors). An edit that would introduce NEW
  syntax errors is REFUSED and the file is left alone; a file that was already
  broken stays editable, so this cannot deadlock a repair. After writing, the
  file is re-read and compared, so "applied successfully" means the bytes on
  disk are the ones intended. force=true overrides a refusal on purpose.
  OPENCODE_EDIT_GUARD=off disables the guard, =all adds the slow checks.
- A missing newString is an ERROR, never "delete this text": the schema marks
  it required, but a schema is advice to the model, not a gate, and treating an
  absent value as "" silently deleted the matched text and reported success.
  To remove a line on purpose, use offset/limit covering it AND the line after
  it, repeating the following line's text (oldString may not end in a newline,
  so it cannot be used to delete a line).
- A file the user marked read-only (no write bit set for anybody) is REFUSED.
  Saving renames a temp file over the target, and a rename only needs write
  access to the directory, so such protection was silently ignored. force=true
  overrides this too, and the message shows the file's mode so the reason for
  the refusal is never a mystery.
"""

from __future__ import annotations

from ..util.log import swallow
import bisect
import re
from pathlib import Path
from typing import Callable

from .registry import Tool, schema_with

from .write import _atomic_write


def _guard_enabled() -> bool:
    from .write import guard_enabled

    return guard_enabled()


def _guard_allow_expensive() -> bool:
    """Whether to run checkers that must launch another program (node: ~0.9s)."""
    from .write import guard_allow_expensive

    return guard_allow_expensive()


# The rules themselves live in edit_guard, shared with apply_patch so the two
# tools can never drift apart. These are thin delegates kept so callers and
# tests inside this module keep working.
_GUARD_FAULT: str | None = None


def _guard_problems(path: Path, before: str, after: str) -> list[str]:
    """Syntax errors ``after`` introduces that ``before`` did not have.

    Empty when nothing new is wrong, when the language has no checker, or when
    the guard is switched off. Never raises: a guard that can crash must not be
    able to block a legitimate edit.
    """
    from . import edit_guard

    problems, _note = edit_guard.syntax_check(path, before, after)
    global _GUARD_FAULT
    _GUARD_FAULT = edit_guard.fault()
    return problems


def _guard_skip_note(path: Path, text: str) -> str:
    """Say out loud when a file's syntax was NOT checked, so a gap is never silent."""
    if not _guard_enabled():
        return ""
    if _GUARD_FAULT:
        return f"\nNote: the syntax guard could not run ({_GUARD_FAULT}); this edit was not checked."
    from . import edit_guard

    _problems, note = edit_guard.syntax_check(path, text, text)
    return f"\nNote: {note}" if note else ""


def _refusal(path: Path, problems: list[str]) -> str:
    from .edit_guard import refusal_text

    return refusal_text(path, problems)


def _confirm_write(path: Path, expected: str) -> str | None:
    """Re-read the file and confirm it holds exactly what we meant to write.

    Thin wrapper over write.confirm_write, which every tool that writes shares.
    """
    from .write import confirm_write

    return confirm_write(path, expected)


def _read_only_refusal(path: Path) -> str | None:
    """Message when the target is marked read-only, else None.

    Checked in both writing paths so no route through this module can bypass it.
    """
    from .edit_guard import read_only_message

    reason = read_only_message(path)
    if not reason:
        return None
    return f"Refused: {path}: {reason}. Nothing was changed."


def _py_diag_note(path: Path) -> str:
    try:
        if path.suffix != ".py":
            return ""
        from .lsp import _diagnostics_py
        issues = _diagnostics_py(path)
        if not issues:
            return ""
        lines = ["", "LSP errors detected in this file, please fix:"]
        lines.extend(f"- {x}" for x in issues[:10])
        return "\n" + "\n".join(lines)
    except Exception:
        return ""


def _py_diag_meta(path: Path) -> dict:
    try:
        if path.suffix != ".py":
            return {}
        from .lsp import _diagnostics_py
        issues = _diagnostics_py(path)
        if not issues:
            return {"diagnostics": {}}
        return {"diagnostics": {str(path): issues[:20]}}
    except Exception:
        return {}

_HSPACE_RE = re.compile(r"[ \t]+")


def find_matches(content: str, old: str) -> list[tuple[int, int]]:
    """Return all (start, end) exact-match spans of `old` in `content`."""
    matches = []
    start = 0
    while True:
        idx = content.find(old, start)
        if idx == -1:
            break
        matches.append((idx, idx + len(old)))
        start = idx + 1
    return matches


_WORD_RE = re.compile(r"\w{3,}")


def _reindent(text: str, indent: str) -> str:
    """Put ``indent`` back on every line of ``text`` that sits at the margin.

    The fuzzy-match repair used to prefix only the FIRST line, so a two-line
    replacement came out as ``<indent>line1`` + ``line2`` — a syntax error in
    Python and a mis-indented block in every other language. Lines that already
    start with whitespace are left alone (the model meant that extra depth) and
    blank lines are not given trailing whitespace.
    """
    lines = text.split("\n")
    out = [indent + lines[0]]
    out.extend(indent + ln if ln and not ln[0].isspace() else ln for ln in lines[1:])
    return "\n".join(out)


def _line_offsets(content: str) -> list[int]:
    """Character offset of the start of each line (plus one past the end)."""
    offsets = [0]
    for line in content.split("\n"):
        offsets.append(offsets[-1] + len(line) + 1)
    return offsets


def _line_starts(text: str) -> list[int]:
    """Character offset where each line begins (no trailing sentinel).

    ``text.find`` skips forward, so this is O(n) total even on huge files;
    feeding the result to ``bisect`` gives O(log n) line-index lookups.
    """
    starts = [0]
    i = text.find("\n")
    while i != -1:
        starts.append(i + 1)
        i = text.find("\n", i + 1)
    return starts


def _line_window_spans(
    content: str,
    content_lines: list[str],
    offsets: list[int],
    old: str,
    key,
    allow_shorter: bool,
    keyed_content: list[str],
) -> list[tuple[int, int]]:
    """Return char spans of line windows whose keyed lines equal keyed old lines.

    ``key`` normalizes a single line (e.g. ``str.strip``, whitespace collapse).
    A single trailing empty old line (from a trailing ``\\n``) is dropped so we
    don't require the file itself to end with a newline.

    The caller precomputes ``content``/``content_lines``/``offsets``/
    ``keyed_content`` once (this helper is used by several successive fallback
    passes; re-splitting for each would be wasteful). The span covers the
    matched lines' text. When ``old`` does NOT end with a newline, the span
    stops before the newline that follows the last matched line (the natural
    exact-match semantics); when it does end with ``\\n`` the newline is part of
    the match and stays inside the span.
    """
    old_lines = old.split("\n")
    ends_with_nl = old.endswith("\n")
    if old_lines and old_lines[-1] == "":
        old_lines.pop()
    if not allow_shorter and any(line.strip() == "" for line in old_lines):
        return []
    keyed_old = [key(line) for line in old_lines]
    if not keyed_old:
        return []

    width = len(keyed_old)
    spans = []
    for i in range(len(content_lines) - width + 1):
        if keyed_content[i : i + width] == keyed_old:
            start = offsets[i]
            end = offsets[i + width]
            # offsets[i+width] points at the START of the next line, i.e. right
            # after the line terminator ending the last matched line. If the
            # caller's old text does not itself end with a newline, that line
            # terminator must not be swallowed by the replacement (it would
            # merge the next line into the edit). Walk back over the full
            # terminator so CRLF files don't leave a stray \r behind.
            if not ends_with_nl:
                while end > start and content[end - 1] in "\r\n":
                    end -= 1
            spans.append((start, end))
    return spans


def _collapse_hspace(line: str) -> str:
    return _HSPACE_RE.sub(" ", line).strip()


def _candidate_spans(content: str, old: str, exact: list[tuple[int, int]] | None = None) -> list[tuple[int, int]]:
    """Return best-match spans of `old` in `content` under the replacer cascade.

    Successively fuzzier normalizers are tried; the first that finds any match
    wins. Exact matches are returned as plain substring spans (fast path).
    The lines/offsets/keyed-lines are computed ONCE and shared across the
    fallback passes (they used to be rebuilt for each pass). ``exact`` may be
    passed in when the caller already has it, so the file is not scanned twice
    (the caller needs to know whether the result was exact).
    """
    # Oversized input: skip the fuzzy passes entirely. Bucketing every line of a
    # huge file is wasted work when only exact matches can realistically hit.
    if len(old) > 200_000 or old.count("\n") > 5000:
        return find_matches(content, old) if exact is None else exact
    if exact is None:
        exact = find_matches(content, old)
    if exact:
        return exact
    content_lines = content.split("\n")
    offsets = _line_offsets(content)
    keyed_cache: dict[Callable[[str], str], list[str]] = {}
    for key, allow_shorter in (
        (str.strip, False),     # leading/trailing whitespace per line
        (str.rstrip, True),     # trailing-only (file may not end with newline)
        (_collapse_hspace, False),
        (_collapse_hspace, True),
    ):
        keyed_content = keyed_cache.get(key)
        if keyed_content is None:
            keyed_content = [key(line) for line in content_lines]
            keyed_cache[key] = keyed_content
        spans = _line_window_spans(
            content, content_lines, offsets, old, key, allow_shorter, keyed_content
        )
        if spans:
            return spans
    return []


_BREAK_DELETE = {ord(c): None for c in "\x0b\x0c\x1c\x1d\x1e\x85\u2028\u2029"}


def _lf_only_lines(text: str) -> bool:
    """True when ``text.splitlines(keepends=True)`` breaks only on ``\n``.

    ``str.splitlines`` also breaks on a lone ``\r`` (old-Mac endings, stray CR)
    and on ``\v``, ``\f`` (form feeds in legacy C/PostScript sources),
    ``\x1c``-``\x1e``, ``\x85``, ``U+2028`` and ``U+2029``. Where any of those
    occur, splitting on ``\n`` alone would return different pieces than the
    whole-file ``splitlines`` this replaces, so those files keep the old path.
    A ``\r`` that is part of ``\r\n`` is fine and must not count — that is
    ordinary Windows line endings.

    Two C-level scans (14 ms on a 1.4 MB file), paid once per call and skipped
    entirely for the ``new_texts``, which are tiny.
    """
    if text.count("\r") != text.count("\r\n"):
        return False
    return len(text.translate(_BREAK_DELETE)) == len(text)


def _crlf_cut_by(text: str, spans: list[tuple[int, int]]) -> bool:
    """True if any span boundary falls between a ``\r`` and its ``\n``.

    A file with no lone ``\r`` can still gain one: cutting a span between the
    two halves of a ``\r\n`` leaves a ``\r`` that is no longer followed by a
    newline, which is enough to make the result break differently.
    """
    n = len(text)
    for s, e in spans:
        if 0 < s < n and text[s - 1] == "\r" and text[s] == "\n":
            return True
        if 0 < e < n and text[e - 1] == "\r" and text[e] == "\n":
            return True
    return False


class _LineView:
    """Slice-addressed line index over a string, materialising only what is read.

    ``text.splitlines(keepends=True)`` allocates one Python string per line for
    the WHOLE file even when the diff shows a seven-line hunk (64 ms on a
    50k-line file, doubled because both sides are split). This serves slices
    straight from the ``\n`` offsets that are computed anyway for the bisect
    lookups, so only the windows the caller asks for are ever materialised.

    Only slices are supported — every use in ``_span_diff`` is a slice, never a
    bare index or a length. Callers gate this on ``_lf_only_lines`` so the
    pieces it returns are identical to the ``splitlines`` ones being replaced.
    """

    __slots__ = ("text", "starts", "n")

    def __init__(self, text: str, starts: list[int]) -> None:
        self.text = text
        self.starts = starts
        # A trailing "" from split() is not a line, so count real terminators.
        self.n = text.count("\n") + (0 if not text or text.endswith("\n") else 1)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, sl: slice) -> list[str]:
        if not isinstance(sl, slice):
            raise TypeError("_LineView supports slicing only")
        a = 0 if sl.start is None else sl.start
        b = self.n if sl.stop is None else sl.stop
        if a < 0 or b < a:
            return []
        a = min(a, self.n)
        b = min(b, self.n)
        if a == b:
            return []
        start = self.starts[a]
        # starts has one entry per "\n"; the final line has no entry of its own.
        end = self.starts[b] if b < len(self.starts) else len(self.text)
        parts = self.text[start:end].split("\n")
        lines = [p + "\n" for p in parts[:-1]]
        if parts[-1]:
            lines.append(parts[-1])
        return lines


def _span_diff(
    content: str,
    new_content: str,
    spans: list[tuple[int, int]],
    new_texts: list[str],
    old_path: str,
    new_path: str,
    context: int = 3,
    old_starts: list[int] | None = None,
    new_starts: list[int] | None = None,
) -> str:
    """Build a unified diff for edits whose changed regions are KNOWN upfront.

    The edit tool always knows exactly which char spans it replaced, so it does
    not need difflib — diffing two whole files is quadratic on long runs of
    unique lines (1.7 s on a 50k-line file). Instead we emit hunks snapped to
    LINE boundaries:

    - the removed block is the *whole* old lines each edit touches, and
    - the added block is the *whole* new lines at the same (offset-adjusted)
      position, including the residual blank line a mid-line deletion leaves.

    Edits within ``2*context`` lines merge into one hunk (as difflib/GNU do);
    farther apart stay separate so a small edit never drags in a huge file as
    context. Context lines are strictly bounded by the neighboring edits, so
    every hunk is patch-applicable. ``spans``/``new_texts`` are zipped: for
    span index i, char range ``content[spans[i]]`` was replaced by
    ``new_texts[i]`` when building ``new_content``.

    Two cosmetic differences from GNU/difflib are accepted: hunk headers always
    carry an explicit ``,count`` (no compact ``-1 +1`` form), and when the
    replacement text contains a copy of a removed line (e.g. wrapping a line
    with itself), difflib's LCS alignment keeps the original line as context
    while this span-based writer emits the extra removal/addition instead. Both
    produce a valid, patch-applicable hunk; only the display differs.

    Line numbers are computed by bisecting precomputed line-start offsets
    (``str.find`` skips forward, so building them is O(n); each span's lookup
    is O(log n)) — never a per-line Python loop over the whole file, which is
    the dominant cost on slow (32-bit ARM) machines.
    """
    # Line-start offsets for O(log n) line-index lookups via bisect — a handful
    # of ``str.count``/``find`` scans per span was O(span offsets), quadratic on
    # replace_all with thousands of spans. Callers that already built them pass
    # them in, so it is done once per call.
    if old_starts is None:
        old_starts = _line_starts(content)
    if new_starts is None:
        new_starts = _line_starts(new_content)
    if (
        _lf_only_lines(content)
        and all(_lf_only_lines(t) for t in new_texts)
        and not _crlf_cut_by(content, spans)
    ):
        old_lines: list[str] | _LineView = _LineView(content, old_starts)
        new_lines: list[str] | _LineView = _LineView(new_content, new_starts)
    else:
        # Files whose line breaks are not all "\n" keep the whole-file split.
        # Measured: both paths emit the same diff for such files, so the only
        # reason to choose between them is speed.
        old_lines = content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)
    out: list[str] = []
    delta = 0  # cumulative new_len - old_len from prior (left) spans
    spans = [(s, e, t) for (s, e), t in zip(spans, new_texts) if not (s == e and not t)]
    if not spans:
        return ""
    # Per-span line extents, tracked SEPARATELY in old and new coordinates.
    # For old, `lo`/`hi_old` derive from the original char offsets. For new,
    # the region lands at new_s = s + delta (leftward edits shift it), so its
    # line index in the final new content is a fresh lookup — it only equals
    # `lo` for the first/multi-span-free case.
    recs = []
    for s, e, t in spans:
        new_s = s + delta
        new_e = new_s + len(t)
        lo = bisect.bisect_right(old_starts, s) - 1
        hi_old = lo if e == s else bisect.bisect_right(old_starts, e - 1)
        new_lo = bisect.bisect_right(new_starts, new_s) - 1
        # The unchanged suffix resumes at the first line boundary on or after
        # new_e (the byte where content[e:] lands); hi_new is that line's index.
        hi_new = bisect.bisect_left(new_starts, new_e + 1)
        delta += len(t) - (e - s)
        if not old_lines[lo:hi_old] and not new_lines[new_lo:hi_new]:
            continue
        recs.append((lo, hi_old, new_lo, hi_new))

    # Collapse spans whose OLD line ranges touch or overlap into one change
    # region — several spans can sit inside a single line (e.g. replace_all of
    # one character), and treating them separately would duplicate the line in
    # the removed/added blocks. The region's added block spans the union of the
    # new ranges, so it naturally dedups too.
    regions: list[tuple[int, int, int, int]] = []
    for lo, hi_old, new_lo, hi_new in recs:
        if regions and lo <= regions[-1][1]:
            plo, phi, pnl, pnh = regions[-1]
            regions[-1] = (plo, max(phi, hi_old), min(pnl, new_lo), max(pnh, hi_new))
        else:
            regions.append((lo, hi_old, new_lo, hi_new))

    # Group consecutive regions into hunks. Regions within 2*context lines merge
    # into one hunk (as difflib/GNU do); farther apart stay separate so a small
    # edit never drags in half a huge file as context. Context is also clamped
    # to the neighboring hunks, so it never crosses a changed line and every
    # hunk stays patch-applicable.
    groups: list[list[tuple[int, int, int, int]]] = []
    for reg in regions:
        if groups:
            _, prev_hi_old, _, prev_hi_new = groups[-1][-1]
            lo, hi_old, new_lo, hi_new = reg
            if lo - prev_hi_old < 2 * context or new_lo - prev_hi_new < 2 * context:
                groups[-1].append(reg)
                continue
        groups.append([reg])

    out.extend([f"--- a/{old_path}\n", f"+++ b/{new_path}\n"])
    for gi, group in enumerate(groups):
        lo0, _, new_lo0, _ = group[0]
        _, _, _, hi_newN = group[-1]
        # Context stops at the neighboring hunks so it only ever shows lines
        # that exist (unchanged) in both files.
        prev_old = groups[gi - 1][-1][1] if gi else 0
        prev_new = groups[gi - 1][-1][3] if gi else 0
        old_ctx = max(prev_old, lo0 - context)
        new_ctx = max(prev_new, new_lo0 - context)
        next_new = groups[gi + 1][0][2] if gi + 1 < len(groups) else float("inf")
        ctx_before = old_lines[old_ctx:lo0]
        ctx_after = new_lines[hi_newN : min(next_new, hi_newN + context)]

        blocks: list[tuple[str, list[str]]] = []
        for idx, (lo, hi_old, new_lo, hi_new) in enumerate(group):
            if idx:
                prev_hi_old, prev_hi_new = group[idx - 1][1], group[idx - 1][3]
                old_gap = old_lines[prev_hi_old:lo]
                new_gap = new_lines[prev_hi_new:new_lo]
                if old_gap == new_gap:
                    # A shared unchanged run between the regions is context.
                    blocks.append((" ", list(old_gap)))
                else:
                    # The gaps diverge (a nearby edit changed line counts);
                    # show them as real removals/additions instead of false
                    # shared context.
                    blocks.append(("-", list(old_gap)))
                    blocks.append(("+", list(new_gap)))
            blocks.append(("-", list(old_lines[lo:hi_old])))
            blocks.append(("+", list(new_lines[new_lo:hi_new])))

        # Standard hunk header: 1-based start of the FIRST shown line (not the
        # first changed line) and the TOTAL line count for that side — context
        # lines included. The old/new starts can differ once earlier edits
        # changed line counts.
        old_start = old_ctx + 1
        new_start = new_ctx + 1
        old_count = len(ctx_before) + sum(len(lines) for kind, lines in blocks if kind != "+") + len(ctx_after)
        new_count = len(ctx_before) + sum(len(lines) for kind, lines in blocks if kind != "-") + len(ctx_after)
        out.append(f"@@ -{old_start},{old_count} +{new_start},{new_count} @@\n")

        def _emit(prefix: str, lines: list[str]) -> None:
            for ln in lines:
                if ln.endswith("\n"):
                    out.append(prefix + ln)
                else:
                    # Only the file's final line lacks a newline; GNU diff and
                    # opencode (jsdiff) flag it so a patch tool doesn't silently
                    # normalize the line ending.
                    out.append(prefix + ln + "\n")
                    out.append("\\ No newline at end of file\n")

        _emit(" ", ctx_before)
        for kind, lines in blocks:
            _emit(kind, lines)
        _emit(" ", ctx_after)
    return "".join(out)


def _closest_hints(content: str, old: str, n: int = 3) -> list[str]:
    """Best-way fail help: the file lines closest to old's first line (~0.3KB).

    Kills blind retries: the caller is told to copy a hint verbatim, so the hint
    carries the line's real leading whitespace and its line number. That matters
    most when the miss was an indentation or trailing-space mistake — the one
    case where the text is right and only the invisible part is wrong.

    Candidates are found with a cheap substring probe and only those are ranked
    with difflib. Two reasons for that shape:

    - ``SequenceMatcher.ratio`` costs ~0.3 ms per candidate on this hardware,
      so ranking a 50k-line file takes 17 s and a failed edit cannot afford it.
    - The pool used to be ``sorted(...)[:500]``, i.e. the 500 alphabetically
      first distinct lines. On a big file that threw away everything else, so
      every hint came from an unrelated part of it (asking about line 49999 got
      suggestions from line 10420). Ranking by file order with a probe means
      the answer is the closest line wherever it lives.

    The probe words are the target's own words ranked by how RARE they are in
    the file, not by length: the longest word is often a common one that every
    line shares ("compute" in 50k generated lines), which would rebuild the
    whole-file pool and reintroduce the 17 s ranking.
    """
    import difflib as _dl
    try:
        first = (old.split("\n", 1)[0] if old else "").rstrip()[:120]
        if not first.strip():
            return []
        # Raw lines, not stripped ones — see the docstring. Only trailing
        # whitespace is trimmed, as noise the caller should not copy.
        raws = [ln.rstrip() for ln in content.split("\n")]
        words = set(_WORD_RE.findall(first))
        probes = sorted(words, key=lambda w: content.count(w))[:3] or [first.strip()]
        # One pass: index every distinct line and keep the ones that share a
        # probe word with the target. Substring tests are C-level, so this stays
        # fast on huge files; a repeated line keeps all its line numbers. A line
        # byte-identical to what the caller already sent is dropped, since
        # repeating it back teaches nothing.
        seen: dict[str, list[int]] = {}
        buckets: dict[str, list[str]] = {w: [] for w in probes}
        for i, raw in enumerate(raws, 1):
            if not raw.strip() or raw == first:
                continue
            if raw in seen:
                seen[raw].append(i)
                continue
            seen[raw] = [i]
            for p in probes:
                if p in raw:
                    buckets[p].append(raw)
        pool: list[str] = []
        for p in probes:  # rarest word first
            hits = buckets[p]
            # A word that matches most of the file pins nothing down. The exact
            # target is excluded above, so a caller whose text differs only in
            # indentation has an empty rarest bucket — and guessing from the top
            # of the file is worse than admitting there is no useful hint.
            if hits and len(hits) <= max(20, len(seen) // 4):
                pool = hits
                break
        if len(pool) > 200:
            # Even the rarest word of the target is this common; ranking all of
            # them would stall the error message for no better answer.
            pool = pool[:200]
        if not pool:
            # Nothing in the file shares a word with the target. Only small
            # files are worth ranking exhaustively; on a big file any answer
            # would be a guess, so say nothing rather than mislead.
            if len(seen) > 500:
                return []
            pool = list(seen)
            if not pool:
                return []
        close = _dl.get_close_matches(first.strip(), pool, n=n, cutoff=0.4)
        if not close:
            # Shares vocabulary but scores below the cutoff — still far more
            # use than a line from the top of the alphabet.
            close = pool[:n]
        out = []
        for c in close:
            where = seen.get(c)
            if not where:
                continue
            also = ", ".join(str(x) for x in where[1:4])
            out.append(f"line {where[0]}" + (f" (also {also})" if also else "") + f": {c[:120]}")
        return out
    except Exception:
        return []


def _do_edit(path: Path, old: str, new: str, replace_all: bool = False, dry_run: bool = False, force: bool = False) -> dict:
    # Read/write with newline="" to preserve the file's exact line endings
    # (universal newlines mode would silently convert CRLF files to LF on any
    # edit, rewriting the whole file's bytes for a one-line change).
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            content = fh.read()
    except (OSError, UnicodeError) as e:
        return {"output": f"Could not read file {path}: {e}", "error": True}

    if not force:
        locked = _read_only_refusal(path)
        if locked:
            return {"output": locked, "error": True, "metadata": {"guard": "read_only"}}

    if old == new:
        return {"output": "oldString and newString are identical. No changes made.", "error": True}
    if not old:
        return {"output": "oldString is empty. Use the write tool to replace the whole file.", "error": True}
    if old[0] == "\n" or old[-1] == "\n":
        return {
            "output": (
                "oldString must not start or end with a newline. "
                "Include only the text to replace; the surrounding line breaks "
                "are preserved automatically."
            ),
            "error": True,
        }

    # Scanned once here and handed to the cascade: the indent repair below needs
    # to know whether the hit was exact, and re-running find_matches meant a
    # second full pass over the file on every fuzzy edit.
    exact = find_matches(content, old)
    matches = _candidate_spans(content, old, exact=exact)
    if not matches:
        hints = _closest_hints(content, old)
        msg = "oldString not found in content."
        if hints:
            msg += " Closest " + str(len(hints)) + ": " + " | ".join(hints)
            msg += " — copy exact text, or use offset=/limit= from grep."
        else:
            msg += " Use offset=/limit=+newString from your grep anchor (no verbatim copy needed)."
        return {"output": msg, "error": True, "metadata": {"hints": hints}}

    if len(matches) > 1 and not replace_all:
        starts = _line_starts(content)
        locs = []
        for s, e in matches[:5]:
            try:
                import bisect as _bi
                locs.append(str(_bi.bisect_right(starts, s)))
            except Exception as _oc_exc:
                swallow(_oc_exc, "tools/edit.py:697")
                pass
        msg = "Found multiple matches for oldString. Provide more surrounding lines in oldString to identify the correct match."
        if locs:
            msg += f" Matches at lines: {', '.join(locs)} — or use offset=<one line>+limit= to target exactly."
        return {"output": msg, "error": True, "metadata": {"match_lines": locs}}

    if replace_all:
        # replace non-overlapping spans right-to-left so offsets stay valid
        spans = sorted(matches, key=lambda s: (s[0], s[1]))
        chosen = []
        last_end = -1
        for s, e in spans:
            if s >= last_end:
                chosen.append((s, e))
                last_end = e
        new_texts = [new] * len(chosen)
        # Repetitive whole-string splicing is O(n²) for thousands of spans
        # (75 s on a 50k-line replace_all); assemble in one pass instead.
        parts: list[str] = []
        last = 0
        for s_off, e_off in chosen:
            parts.append(content[last:s_off])
            parts.append(new)
            last = e_off
        parts.append(content[last:])
        new_content = "".join(parts)
    else:
        start, end = matches[0]
        new_text = new
        # A fuzzy (non-exact) match replaces the whole matched line region, so
        # the file's original leading indentation would be lost when the model's
        # new string doesn't carry it. Restore it on every line the model left
        # flush against the margin, not just the first: a two-line replacement
        # used to come out with only its first line indented, which is a syntax
        # error in Python and a mis-indented block anywhere else.
        if not exact:
            region = content[start:end]
            first_line = region.split("\n", 1)[0]
            leading = first_line[: len(first_line) - len(first_line.lstrip())]
            if leading and not new.startswith((" ", "\t")):
                new_text = _reindent(new, leading)
        chosen = [(start, end)]
        new_texts = [new_text]
        new_content = content[:start] + new_text + content[end:]

    diff = _span_diff(content, new_content, chosen, new_texts, str(path), str(path))
    detected = _guard_problems(path, content, new_content)
    problems = [] if force else detected
    guard_state = "forced" if (force and detected) else ("refused" if problems else "ok")
    if dry_run:
        head = "Dry run — no changes written.\n"
        if problems:
            head = (
                "Dry run — no changes written. WARNING: applying this would be REFUSED:\n"
                + _refusal(path, problems)
                + "\n\n"
            )
        return {
            "output": head + diff,
            "metadata": {"diff": diff, "replaceAll": replace_all, "dry_run": True, "guard": "would_refuse" if problems else guard_state},
        }
    if problems:
        return {
            "output": _refusal(path, problems),
            "error": True,
            "metadata": {"diff": diff, "guard": "refused", "problems": problems[:10]},
        }
    try:
        _atomic_write(path, new_content)
    except (OSError, UnicodeError) as e:
        return {"output": f"Error writing file {path}: {e}", "error": True}
    unverified = _confirm_write(path, new_content)
    if unverified:
        return {
            "output": (
                f"Edit was written but could NOT be verified: {unverified}. "
                f"Open {path} and check it before continuing."
            ),
            "error": True,
            "metadata": {"diff": diff, "guard": "unverified"},
        }
    return {
        "output": "Edit applied successfully." + _py_diag_note(path) + _guard_skip_note(path, new_content),
        "metadata": {"diff": diff, "replaceAll": replace_all, "guard": guard_state, **_py_diag_meta(path)},
    }


def _edit_lines(path: Path, offset: int, limit: int, new: str, dry_run: bool = False, force: bool = False) -> dict:
    """Line-range mode: offset/limit from a grep anchor + newString.
    No verbatim copy, so no mismatch fail. Best with dry_run first."""
    # Read exactly like _do_edit: newline="" so CRLF survives (universal
    # newlines rewrote a Windows file to LF end-to-end for a one-line change)
    # and strict decoding so unreadable bytes raise instead of silently
    # becoming U+FFFD (errors="replace" rewrote them into the file and still
    # reported success — unrecoverable data loss).
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            text = fh.read()
    except (OSError, UnicodeError) as e:
        return {"output": f"Could not read file {path}: {e}", "error": True}

    if not force:
        locked = _read_only_refusal(path)
        if locked:
            return {"output": locked, "error": True, "metadata": {"guard": "read_only"}}

    # Splice on char offsets instead of split/join: a CRLF file's "\r" lives at
    # the END of each split line, so rebuilding by joining lines silently
    # dropped it on the replaced line. The untouched prefix/suffix are copied
    # verbatim. One line index is built and handed to the diff below.
    starts = _line_starts(text)
    # A trailing "" from split() is not a line, so count real terminators.
    total = text.count("\n") + (0 if not text or text.endswith("\n") else 1)
    s = max(1, int(offset or 1))
    lim = max(1, int(limit or 1))
    if s > total:
        return {"output": f"offset {s} past end of file ({total} lines).", "error": True}
    e = min(total, s + lim - 1)
    # Match the file's own line ending so a CRLF file stays CRLF end-to-end.
    eol = "\r\n" if "\r\n" in text else "\n"
    start = starts[s - 1]
    end = starts[e] if e < len(starts) else len(text)
    block = new.replace("\r\n", "\n").replace("\n", eol)
    # Keep the file's own terminator instead of replacing it: shrink the span
    # back over it so the untouched tail still carries it. Same rule _do_edit
    # applies to an oldString that does not end in a newline, which makes both
    # modes produce an identical file AND an identical diff for the same change.
    if end > start and text[end - 1] == "\n":
        term = 2 if text[end - 2:end] == "\r\n" else 1
        if block.endswith(eol):
            block = block[: -len(eol)]
        end -= term
    new_content = text[:start] + block + text[end:]
    # Real span-based diff (same writer _do_edit uses) so metadata.diff is
    # populated — the TUI renders nothing for an edit without it, and the old
    # hand-rolled preview silently truncated to 10 lines.
    diff = _span_diff(text, new_content, [(start, end)], [block], str(path), str(path), old_starts=starts)
    detected = _guard_problems(path, text, new_content)
    problems = [] if force else detected
    guard_state = "forced" if (force and detected) else ("refused" if problems else "ok")
    if dry_run:
        head = "Dry run — no changes written.\n"
        if problems:
            head = (
                "Dry run — no changes written. WARNING: applying this would be REFUSED:\n"
                + _refusal(path, problems)
                + "\n\n"
            )
        return {
            "output": head + diff,
            "metadata": {"diff": diff, "offset": s, "limit": e - s + 1, "dry_run": True, "guard": "would_refuse" if problems else guard_state},
        }
    if problems:
        return {
            "output": _refusal(path, problems),
            "error": True,
            "metadata": {"diff": diff, "offset": s, "limit": e - s + 1, "guard": "refused", "problems": problems[:10]},
        }
    try:
        _atomic_write(path, new_content)
    except (OSError, UnicodeError) as ex:
        return {"output": f"Error writing file {path}: {ex}", "error": True}
    unverified = _confirm_write(path, new_content)
    if unverified:
        return {
            "output": (
                f"Edit was written but could NOT be verified: {unverified}. "
                f"Open {path} and check it before continuing."
            ),
            "error": True,
            "metadata": {"diff": diff, "offset": s, "limit": e - s + 1, "guard": "unverified"},
        }
    return {
        "output": f"Replaced lines {s}-{e} ({e - s + 1} lines)." + _py_diag_note(path) + _guard_skip_note(path, new_content),
        "metadata": {"diff": diff, "offset": s, "limit": e - s + 1, "guard": guard_state, **_py_diag_meta(path)},
    }


def _edit(filePath: str, oldString: str, newString: str, replaceAll: bool = False, dry_run: bool = False, offset: int = 0, limit: int = 0, force: bool = False) -> dict:
    path = Path(filePath)
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if not path.exists():
        return {"output": f"File does not exist: {path}", "error": True}
    if offset and limit and not (oldString or "").strip():
        result = _edit_lines(path, int(offset), int(limit), newString, dry_run, force)
    else:
        result = _do_edit(path, oldString, newString, replaceAll, dry_run=dry_run, force=force)
    if result.get("error") is not True and not dry_run:
        # Feed the verify tool's homework checker. Both modes come through here
        # (line-range used to return early, so the checker never saw those
        # edits), but a dry run writes nothing and must not count as an edit.
        from .verify import track

        track(path, "edit")
    return result


def _require_new_string(input: dict) -> str:
    """The replacement text, or a loud error if the model forgot to send it.

    ``input.get("newString") or ""`` used to turn a missing or null value into
    an empty string, which the replacer then wrote — silently deleting whatever
    ``oldString`` matched while reporting success. The schema already says this
    field is required, but a schema is advice, not a gate.
    """
    if "newString" not in input:
        raise KeyError(
            "newString is required and was not sent, so nothing was changed. "
            "To remove a line, use offset+limit covering it and the next line, "
            "repeating the next line's text."
        )
    value = input["newString"]
    if value is None:
        raise KeyError("newString was null, so nothing was changed. Send the replacement text.")
    if not isinstance(value, str):
        raise KeyError(f"newString must be a string, got {type(value).__name__}, so nothing was changed.")
    return value


def tool() -> Tool:
    description = """Exact string replacement in a file. oldString must match verbatim (no line-number prefixes); miss returns 3 closest lines, multi-match returns line numbers. dry_run previews. Have grep offset/limit? Use offset+limit+newString (no verbatim copy). An edit that would break the file's syntax, or that targets a file marked read-only, is refused and nothing is written (force=true to override both)."""

    def run(input: dict) -> dict:
        # The loop wraps tool calls in try/except, so a bare KeyError surfaced as
        # the raw "edit failed: 'filePath'". Say what is missing instead.
        fp = input.get("filePath")
        if not isinstance(fp, str) or not fp.strip():
            return {
                "output": "edit needs a filePath (the path to the file to change). "
                "Nothing was changed.",
                "error": True,
            }
        try:
            off = int(input.get("offset") or 0)
        except (TypeError, ValueError):
            off = 0
        try:
            lim = int(input.get("limit") or 0)
        except (TypeError, ValueError):
            lim = 0
        return _edit(
            fp,
            input.get("oldString") or "",
            _require_new_string(input),
            replaceAll=bool(input.get("replaceAll", False)),
            dry_run=bool(input.get("dry_run", False)),
            offset=off,
            limit=lim,
            force=bool(input.get("force", False)),
        )

    return Tool(
        name="edit",
        description=description,
        parameters=schema_with(
            {
                "filePath": {"type": "string", "description": "Absolute file path"},
                "oldString": {"type": "string", "description": "Text to replace, verbatim (or omit with offset+limit)", "optional": True},
                "newString": {"type": "string", "description": "Replacement text"},
                "replaceAll": {"type": "boolean", "description": "Replace every match", "optional": True},
                "dry_run": {"type": "boolean", "description": "Preview diff without writing", "optional": True},
                "offset": {"type": "integer", "description": "Start line from grep anchor (use with limit, no oldString needed)", "optional": True},
                "limit": {"type": "integer", "description": "Lines to replace (use with offset)", "optional": True},
                "force": {"type": "boolean", "description": "Write even if it would break the file's syntax or the file is read-only (refused otherwise)", "optional": True},
            },
            ["filePath", "newString"],
        ),
        run=run,
        permission="edit",
    )

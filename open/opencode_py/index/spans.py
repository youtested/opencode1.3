"""Block-span detection for any language: exact (start, end) of one def/class.

Python uses the real AST (exact). Every other language uses a fast
string/comment-aware brace scan (~70us): from the header line, the first
`{` opens depth 1, the matching `}` at depth 0 closes the block. Strings,
line/block comments, and template interpolations never count. Header-only
lines (prototypes, fields, single-line bodies) return just their own line;
indent languages without braces (lua/ruby/perl/shell/...) fall back to an
indent scan. Never raises — worst case returns (start, start).
"""

from __future__ import annotations

from ..util.log import swallow
import ast as _ast

# languages whose blocks open with `{` (brace scan); everything else listed
# here uses indent scan. Unknown languages try brace scan first (a `{` on or
# after the header decides), else indent scan.
_BRACE_LANGS = frozenset({
    "javascript", "typescript", "c", "cpp", "c++", "csharp", "c#", "java",
    "go", "rust", "swift", "php", "kotlin", "scala", "dart", "zig",
})

_INDENT_LANGS = frozenset({
    "python", "lua", "ruby", "perl", "r", "bash", "sh", "shell", "fish",
    "sql", "yaml", "yml", "haskell", "elixir", "clojure", "vim",
})


def _strip_to_code(text: str) -> str:
    """Replace strings/comments with blanks, keeping newlines and braces.

    Handles '...' "..." `...${...}` C/JS block comments and // # -- line
    comments. Escape sequences inside strings are skipped so `"\\""` can't
    end a string early. Output length == input length (line map preserved).
    """
    out: list[str] = []
    i, n = 0, len(text)
    quote: str | None = None
    line_c = False
    block_c = False
    while i < n:
        c = text[i]
        nx = text[i + 1] if i + 1 < n else ""
        if line_c:
            if c == "\n":
                line_c = False
                out.append(c)
            i += 1
            continue
        if block_c:
            if c == "*" and nx == "/":
                block_c = False
                i += 2
            else:
                out.append("\n" if c == "\n" else " ")
                i += 1
            continue
        if quote:
            if c == "\\":
                out.append("  ")
                i += 2
                continue
            if c == quote:
                quote = None
                out.append(" ")
            elif quote == "`" and c == "$" and nx == "{":
                out.append("  ")
                i += 2
                continue
            else:
                out.append(" " if c != "\n" else "\n")
            i += 1
            continue
        if c == "/" and nx == "/":
            line_c = True
            i += 2
            continue
        if c == "/" and nx == "*":
            block_c = True
            i += 2
            continue
        if c == "#" and quote is None:
            line_c = True
            i += 1
            continue
        if c in "\"'`":
            quote = c
            out.append(" ")
            i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


_BRACE_CACHE: list[tuple[list, list, list, bool]] = []
_BRACE_CACHE_MAX = 2


def _brace_index(san: list[str], total: int):
    """Per-line (opens, closes) for `san`, cached, plus a "mixed lines" flag.

    `_brace_end` used to walk characters from the declaration to the end of the
    file on every call — 5.8ms per call on a 2,900-line header, called once per
    declaration. Counting braces per line once turns each of those walks into
    two array reads.
    """
    for ref, o, c, mixed in _BRACE_CACHE:
        if ref is san:
            return o, c, mixed
    o = [0] * total
    c = [0] * total
    mixed = False
    for i in range(min(total, len(san))):
        line = san[i]
        if "{" in line or "}" in line:
            oi = line.count("{")
            ci = line.count("}")
            o[i] = oi
            c[i] = ci
            if oi and ci:
                mixed = True
    _BRACE_CACHE.append((san, o, c, mixed))
    while len(_BRACE_CACHE) > _BRACE_CACHE_MAX:
        _BRACE_CACHE.pop(0)
    return o, c, mixed


def _brace_end(san: list[str], start: int, total: int) -> int | None:
    """Closing 1-indexed line of the `{` block opened at/after `start`.

    Returns None when no brace opens (header-only line) or it never closes.
    A `}` on the header line itself (single-line body) closes immediately.
    """
    opens, closes, mixed = _brace_index(san, total)
    if mixed:
        # Some line both opens and closes, so brace *order* decides the answer
        # (`} else {`, default arguments `= {}`). Counting cannot resolve that,
        # and guessing changed 0.5% of results, so fall back to the exact walk.
        return _brace_end_exact(san, start, total)
    depth = 0
    started = False
    for i in range(start - 1, total):
        oi = opens[i]
        ci = closes[i]
        if not started:
            if not oi:
                continue  # closes before any open are ignored, as before
            started = True
            depth = oi
            continue
        depth += oi - ci
        if depth <= 0:
            return i + 1
    return None


def _brace_end_exact(san: list[str], start: int, total: int) -> int | None:
    """Character-accurate `_brace_end`, for files the counting path rejects."""
    depth = 0
    started = False
    for i in range(start - 1, total):
        for c in san[i]:
            if c == "{":
                depth += 1
                started = True
            elif c == "}":
                if not started:
                    continue
                depth -= 1
                if depth <= 0:
                    return i + 1
    return None


_END_LANGS = frozenset({"ruby", "lua", "perl", "bash", "sh", "shell", "fish"})

_END_OPEN = ("def ", "class ", "module ", "function ", "if ", "unless ",
             "while ", "until ", "for ", "do", "do |", "begin", "case ",
             "local function ", "function(")


def _end_keyword_close(lines: list[str], start: int, total: int) -> int | None:
    """Matching `end`/`}`/`fi`/`done` at indent <= header indent (ruby/lua/shell).

    Returns the closing line, or None when no opener shape matches (caller
    falls back to the indent scan). Nested same-kind blocks are counted.
    """
    import re as _re
    try:
        header = lines[start - 1]
        base = len(header) - len(header.lstrip())
        stripped = header.strip()
    except IndexError:
        return None
    low = stripped.lower()
    is_fn = low.startswith(_END_OPEN) or re_match_fn(low)
    if not is_fn:
        # still allow: any header in an end-language gets end-scan if a
        # deeper-indented body follows, else indent scan wins below
        try:
            nxt = lines[start] if start < total else ""
            if not nxt.strip() or len(nxt) - len(nxt.lstrip()) <= base:
                return None
        except IndexError:
            return None
    depth = 1
    for i in range(start, total):
        ln = lines[i]
        s = ln.strip()
        if not s:
            continue
        indent = len(ln) - len(ln.lstrip())
        if indent < base:
            return None  # left the block without an `end`: not end-shaped
        if s == "end" and indent <= base:
            depth -= 1
            if depth <= 0:
                return i + 1
        elif s == "end" and depth > 1:
            depth -= 1  # closes a nested block (deeper indent)
        elif indent > base and _re.match(r"^(?:def |class |module |function\b|if\b|unless\b|while\b|until\b|for\b|case\b|begin\b|do\b)", s):
            depth += 1
    return None


def re_match_fn(low: str) -> bool:
    import re as _re
    return bool(_re.match(r"^(?:local\s+)?function\s*[\w.:]+", low) or _re.match(r"^[\w.:]+\s*=\s*function\b", low))


def _indent_end(lines: list[str], start: int, total: int) -> int:
    """First line before the next same-or-lower-indent line (or EOF)."""
    try:
        base = len(lines[start - 1]) - len(lines[start - 1].lstrip())
    except IndexError:
        return start
    for i in range(start, total):
        try:
            ln = lines[i]
        except IndexError as _oc_exc:
            swallow(_oc_exc, "opencode_py/index/spans.py:242")
            break
        if not ln.strip():
            continue
        if len(ln) - len(ln.lstrip()) <= base:
            return i
    return total


_SAN_CACHE: list[tuple[list, list[str]]] = []
_SAN_CACHE_MAX = 2


def _sanitized(lines: list[str]) -> list[str]:
    """Comment/string-stripped view of `lines`, computed once per file.

    `block_span` is called once per declaration, and it used to rebuild this
    from scratch every time: a char-by-char pass over the whole file, per
    declaration. A 2,700-line C header with ~500 decls is 500 full passes and
    it took minutes on a phone.

    The cache holds a strong reference to `lines`, so the identity check stays
    sound (a freed list's id cannot be recycled onto a different object while
    we still hold it). Two entries is enough to cover a per-file index pass
    and keeps the extra memory to one copy of the file.
    """
    for ref, san in _SAN_CACHE:
        if ref is lines:
            return san
    san = _strip_to_code("\n".join(lines)).split("\n")
    _SAN_CACHE.append((lines, san))
    while len(_SAN_CACHE) > _SAN_CACHE_MAX:
        _SAN_CACHE.pop(0)
    return san


def block_span(lines: list[str], start: int, language: str = "") -> tuple[int, int]:
    """Exact (start, end) 1-indexed lines of the block at `start`. Never raises."""
    try:
        total = len(lines)
        if total <= 0 or start < 1 or start > total:
            return start, start
        lang = (language or "").strip().lower()
        if lang == "python":
            try:
                tree = _ast.parse("\n".join(lines))
            except (SyntaxError, ValueError):
                tree = None
            if tree is not None:
                best = None
                for node in _ast.walk(tree):
                    if isinstance(node, (_ast.ClassDef, _ast.FunctionDef, _ast.AsyncFunctionDef)):
                        s = node.lineno
                        e = node.end_lineno or s
                        if s == start and (best is None or e > best[1]):
                            best = (s, e)
                if best is not None:
                    return best
        if lang in _INDENT_LANGS and lang != "python":
            if lang in _END_LANGS:
                end = _end_keyword_close(lines, start, total)
                if end is not None:
                    return start, end
            return start, _indent_end(lines, start, total)
        san = _sanitized(lines)
        end = _brace_end(san, start, total)
        if end is not None:
            return start, end
        if lang in _BRACE_LANGS or not lang:
            return start, start
        return start, _indent_end(lines, start, total)
    except Exception:
        try:
            return start, start
        except Exception:
            return 1, 1

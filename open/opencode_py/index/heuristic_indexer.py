"""Heuristic line-based indexers for languages without ``ast`` on the phone.

Two back-ends, best-effort as high as a pure-Python budget allows:

- **language rules** — per-family compiled regexes for declarations, imports
  and identifier usage (JS/TS, C/C++, Java, C#, Swift, PHP, Go, Rust, Lua,
  Ruby, Perl, R, SQL, shell, and a generic catch-all).
- **universal-ctags** — when the ``ctags`` binary is present it produces exact
  definitions for ~100 languages with one syscall. If its JSON output works we
  prefer its definitions for every non-Python file; the regex rules still
  provide the reference (usage) index, which ctags does not do.

Everything degrades gracefully: a missing binary, an empty pattern table or an
unreadable file never raises — they just yield fewer records.
"""

from __future__ import annotations

from ..util.log import swallow
import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .model import FileIndex, ImportRecord, Ref, Symbol

NAME = r"[A-Za-z_$][A-Za-z0-9_$]*"
WORD = re.compile(r"\b([A-Za-z_$][A-Za-z0-9_$]*)\b")

_MAX_CTAGS_LINES = 20000


def _id(name: str) -> str:
    return r"(?P<name>" + name + r")"


def _build_javascript() -> LangRule:
    decls = [
        ("class", re.compile(r"(?m)^\s*(?:export\s+default\s+|export\s+)?(?:abstract\s+)?class\s+" + _id(NAME))),
        ("interface", re.compile(r"(?m)^\s*(?:export\s+default\s+|export\s+)?interface\s+" + _id(NAME))),
        ("type", re.compile(r"(?m)^\s*(?:export\s+default\s+|export\s+)?type\s+" + _id(NAME))),
        ("enum", re.compile(r"(?m)^\s*(?:export\s+default\s+|export\s+)?(?:const\s+)?enum\s+" + _id(NAME))),
        ("function", re.compile(r"(?m)^\s*(?:export\s+default\s+|export\s+)?(?:async\s+)?function\s*\*?\s*" + _id(NAME))),
        ("namespace", re.compile(r"(?m)^\s*(?:export\s+)?namespace\s+" + _id(NAME))),
        ("function", re.compile(r"(?m)^\s*(?:(?:public|private|protected|static|async|get|set)\s+)*" + _id(NAME) + r"\s*\([^;]*\)\s*\{")),
        ("variable", re.compile(r"(?m)^\s*(?:export\s+default\s+|export\s+)?(?:const|let|var)\s+" + _id(NAME))),
    ]
    imports = [
        re.compile(r"""\bimport\s+(?:[\w$*{},\s]+\s+from\s+)?["']([^"']+)["']"""),
        re.compile(r"""\brequire\s*\(\s*["']([^"']+)["']\s*\)"""),
        re.compile(r"""\bimport\s*\(\s*["']([^"']+)["']\s*\)"""),
    ]
    keywords = frozenset(
        "break case catch class const continue debugger default delete do else enum "
        "export extends false finally for function if import in instanceof let new "
        "null of return static super switch this throw true try typeof var void "
        "while with yield await async get set interface type".split()
    )
    return LangRule(
        "javascript", keywords, decls, imports,
        comment=re.compile(r"\s*//.*$"),
        ctags_kinds={"c": "class", "m": "method", "f": "function", "C": "constant",
                      "v": "variable", "e": "enum", "i": "interface",
                      "n": "namespace", "t": "type", "p": "function"},
    )


def _build_c() -> LangRule:
    # C and C++ share a rule set; the differences (class/namespace/template) are
    # additive, so one set keeps them consistent. `cpp` is registered separately
    # only to advertise the extra kinds and enable the C++ decl pass.
    decls = [
        ("macro", re.compile(r"(?m)^\s*#\s*define\s+" + _id(NAME))),
        ("struct", re.compile(r"(?m)^\s*(?:typedef\s+)?struct\s+" + _id(NAME))),
        ("struct", re.compile(r"(?m)^\s*typedef\s+struct\s*\{")),
        ("class", re.compile(r"(?m)^\s*(?:template\s*<[^>]*>\s*)?(?:typedef\s+)?class\s+" + _id(NAME))),
        ("enum", re.compile(r"(?m)^\s*enum(?:\s+class)?\s+" + _id(NAME))),
        ("union", re.compile(r"(?m)^\s*(?:typedef\s+)?union\s+" + _id(NAME))),
        ("type", re.compile(r"(?m)^\s*typedef\s+\w+(?:\s*\*)?\s+" + _id(NAME))),
        # NO function pattern here on purpose. Matching a parameter list with a
        # regex needs a nested quantifier, which backtracks catastrophically on
        # the long prototype lines real headers contain (a 2.7k-line GL header
        # took minutes). _c_decl_map finds C/C++ functions by scanning to the
        # first `(` and inspecting the tail, which is linear and exact enough.
    ]
    imports = [re.compile(r"""^\s*#\s*include\s*[<"]([^>"\n]+)[>"]""")]
    keywords = frozenset(
        "alignas alignof auto break case char class const constexpr continue "
        "decltype default delete do double else enum explicit extern false float "
        "for friend goto if inline int long mutable namespace new noexcept nullptr "
        "operator private protected public register reinterpret_cast return short "
        "signed sizeof static static_assert static_cast struct switch template this "
        "throw true try typedef typeid typename union unsigned using virtual void "
        "volatile while".split()
    )
    return LangRule(
        "c", keywords, decls, imports,
        comment=re.compile(r"\s*(?://.*$|/\*.*?\*/)"),
        ctags_kinds={"c": "class", "s": "struct", "u": "union", "g": "enum",
                      "f": "function", "p": "function", "m": "member",
                      "v": "variable", "d": "macro", "t": "type"},
    )


# tokens that can sit between a return type and the name in a declaration;
# keeps the function matcher from reporting `if (x) {` as a function
_C_DECL_NOISE = frozenset(
    "if else for while switch return sizeof case do goto break continue "
    "defined __attribute__ __asm__ asm _Alignof alignof static_assert".split()
)

# a declaration is: [modifiers/type tokens] name ( ... ) [; { or = initialiser]
_C_FUNC_HEAD = re.compile(
    r"^(?P<pre>[A-Za-z_][\w:]*|struct|enum|union|class|const|static|extern|inline|"
    r"virtual|explicit|constexpr|thread_local|__inline__|__inline|_Noreturn|"
    r"restrict|__restrict|long|short|unsigned|signed)"
    r"(?P<rest>[\s\*&:<>,()\[\]\w]{0,300}?)\b(?P<name>[A-Za-z_]\w*)\s*\(")

# what may follow a parameter list: a terminator, a comma, or a qualifier
_C_TAIL = re.compile(r"^[^;{}\n]{0,300}?\)\s*(?:;|\{|=|,|\)|const\b|noexcept\b|override\b)")


def _join_continuations(lines: list[str], limit: int = 40,
                        max_chars: int = 2000) -> list[tuple[int, str]]:
    """Merge lines that continue into one logical declaration.

    Real C splits prototypes across lines:
        extern int  foo (int a,
                         int b);
    A line-anchored regex cannot see that, and headers are mostly like it.

    Only *parentheses* join. Braces must not: a class body is a sequence of
    separate member declarations, so joining across `{` would fuse a whole
    class into one string and hide every member but the first. It also made a
    2.7k-line header one ~100KB "line" whose nested-quantifier match then ran
    for minutes.

    `max_chars` is a second backstop so an unbalanced-paren file cannot build an
    unbounded string either.
    """
    out: list[tuple[int, str]] = []
    buf: list[str] = []
    size = 0
    start = 0
    depth = 0
    for i, raw in enumerate(lines, 1):
        if depth <= 0:
            start = i
            buf = [raw]
            size = len(raw)
        else:
            buf.append(raw)
            size += len(raw) + 1
        if raw.rstrip().endswith("\\"):
            depth += 1
        else:
            depth += raw.count("(") - raw.count(")")
        if depth <= 0 or len(buf) >= limit or size >= max_chars:
            out.append((start, " ".join(x.strip() for x in buf)))
            buf = []
            size = 0
            depth = 0
    if buf:
        out.append((start, " ".join(x.strip() for x in buf)))
    return out


def _c_decl_map(lines: list[str], lang_id: str) -> dict[int, tuple[str, str, int]]:
    """start line -> (kind, name, end line) for C/C++ declarations.

    A dedicated pass rather than the shared per-line loop, because a declaration
    can span lines. The caller consumes the map from the normal loop, so refs,
    imports and spans keep behaving exactly as before.
    """
    out: dict[int, tuple[str, str, int]] = {}
    kws = LANG_RULES["c"].keywords
    for start, text in _join_continuations(lines):
        s = text.strip()
        if not s or s.startswith("#"):
            if s.startswith("#"):
                m = re.match(r"^#\s*define\s+(?P<name>[A-Za-z_]\w*)", s)
                if m:
                    out.setdefault(start, ("macro", m.group("name"), start))
            continue
        end_line = start + text.count(";")
        if s.startswith("typedef"):
            # typedef ret (*NAME)(args);  function-pointer alias
            m = re.match(r"^typedef\s+[\w:*\s<>,]*\(\s*\*\s*(?P<name>[A-Za-z_]\w*)\s*\)", s)
            if m and m.group("name") not in kws:
                out.setdefault(start, ("type", m.group("name"), start))
                continue
            # typedef struct Foo Bar;   alias of a tag
            m = re.match(r"^typedef\s+(?:struct|union|enum)(?:\s+class)?\s+"
                         r"[A-Za-z_]\w*\s+(?P<name>[A-Za-z_]\w*)\s*[;\[]", s)
            if m and m.group("name") not in kws:
                out.setdefault(start, ("type", m.group("name"), start))
                continue
            # typedef ret NAME;         simple alias
            m = re.match(r"^typedef\s+[\w:*\s<>,]+?\**\s*(?P<name>[A-Za-z_]\w*)\s*(?:\[[^\]]*\])?\s*;$", s)
            if m and m.group("name") not in kws:
                out.setdefault(start, ("type", m.group("name"), start))
                continue
        m = re.match(r"^(?:typedef\s+)?(?:struct|union)\s+(?P<name>[A-Za-z_]\w*)", s)
        if m and m.group("name") not in kws:
            out.setdefault(start, ("struct", m.group("name"), start))
            continue
        m = re.match(r"^(?:template\s*<[^>]*>\s*)?(?:typedef\s+)?enum(?:\s+class)?\s+"
                     r"(?P<name>[A-Za-z_]\w*)", s)
        if m and m.group("name") not in kws:
            out.setdefault(start, ("enum", m.group("name"), start))
            continue
        m = re.match(r"^(?:template\s*<[^>]*>\s*)?(?:typedef\s+)?class\s+"
                     r"(?P<name>[A-Za-z_]\w*)", s)
        if m and m.group("name") not in kws:
            out.setdefault(start, ("class", m.group("name"), start))
            continue
        if lang_id == "cpp":
            m = re.match(r"^namespace\s+(?P<name>[A-Za-z_]\w*)", s)
            if m and m.group("name") not in kws:
                out.setdefault(start, ("namespace", m.group("name"), start))
                continue
        m = _C_FUNC_HEAD.match(s)
        if m:
            name = m.group("name")
            if (name not in kws and name not in _C_DECL_NOISE
                    and m.group("pre") not in _C_DECL_NOISE):
                tail = s[m.end():]
                # bounded scan: `[^;{]*` followed by `\)` backtracks badly on a
                # long parameter list, and a declaration's tail is short
                if _C_TAIL.match(tail):
                    out.setdefault(start, ("function", name, end_line))
    return out


def _build_cpp() -> LangRule:
    rule = _build_c()
    decls = list(rule.decls) + [
        ("namespace", re.compile(r"(?m)^\s*namespace\s+" + _id(NAME))),
    ]
    return LangRule("cpp", rule.keywords, decls, rule.imports,
                    comment=rule.comment, ctags_kinds=rule.ctags_kinds)



def _build_java() -> LangRule:
    mod = r"(?:(?:public|private|protected|static|final|abstract|default|sealed)\s+)*"
    decls = [
        ("class", re.compile(r"(?m)^\s*" + mod + r"(?:class|record)\s+" + _id(NAME))),
        ("interface", re.compile(r"(?m)^\s*" + mod + r"interface\s+" + _id(NAME))),
        ("enum", re.compile(r"(?m)^\s*" + mod + r"enum\s+" + _id(NAME))),
        ("function", re.compile(r"(?m)^\s*" + mod + r"[\w<>\[\]?.,\s]+\s+" + _id(NAME) + r"\s*\([^;{]*\)\s*(?:throws\s+[\w,\s]+)?\{")),
    ]
    imports = [re.compile(r"^\s*import\s+(?:static\s+)?([\w.]+)\s*;")]
    keywords = frozenset(
        "abstract assert boolean break byte case catch char class const continue "
        "default do double else enum extends final finally float for goto if "
        "implements import instanceof int interface long native new package "
        "private protected public return short static strictfp super switch "
        "synchronized this throw throws transient try void volatile while yield "
        "record sealed permits".split()
    )
    return LangRule(
        "java", keywords, decls, imports,
        comment=re.compile(r"\s*(?://.*$|/\*.*?\*/)"),
        ctags_kinds={"c": "class", "i": "interface", "e": "enum", "m": "method",
                      "f": "function", "P": "package", "v": "variable",
                      "C": "constant", "n": "namespace", "r": "record",
                      "t": "type"},
    )


def _build_go() -> LangRule:
    decls = [
        ("function", re.compile(r"(?m)^\s*func\s+(?:\([^)]*\)\s+)?" + _id(NAME))),
        ("struct", re.compile(r"(?m)^\s*type\s+" + _id(NAME) + r"\s+(?:struct\s*\{|interface\s*\{)")),
        ("type", re.compile(r"(?m)^\s*type\s+" + _id(NAME) + r"\s*=")),
        ("variable", re.compile(r"(?m)^\s*(?:var|const)\s+" + _id(NAME))),
        ("package", re.compile(r"(?m)^\s*package\s+" + _id(NAME))),
    ]
    imports = [re.compile(r'^\s*import\s+"([^"]+)"')]
    keywords = frozenset(
        "break case chan const continue default defer else fallthrough for func "
        "go goto if import interface map package range return select struct "
        "switch type var".split()
    )
    return LangRule(
        "go", keywords, decls, imports,
        comment=re.compile(r"\s*//.*$"),
        ctags_kinds={"f": "function", "t": "type", "s": "struct", "i": "interface",
                      "v": "variable", "c": "constant", "p": "package",
                      "m": "member", "P": "package"},
    )


_PUB = r"(?:(?:pub|pub\([^)]*\))\s+)?"


def _build_rust() -> LangRule:
    decls = [
        ("function", re.compile(r"(?m)^\s*" + _PUB + r"(?:async\s+)?fn\s+" + _id(NAME))),
        ("struct", re.compile(r"(?m)^\s*" + _PUB + r"struct\s+" + _id(NAME))),
        ("enum", re.compile(r"(?m)^\s*" + _PUB + r"enum\s+" + _id(NAME))),
        ("trait", re.compile(r"(?m)^\s*" + _PUB + r"trait\s+" + _id(NAME))),
        ("module", re.compile(r"(?m)^\s*" + _PUB + r"mod\s+" + _id(NAME))),
        ("type", re.compile(r"(?m)^\s*" + _PUB + r"(?:type|union)\s+" + _id(NAME))),
        ("constant", re.compile(r"(?m)^\s*" + _PUB + r"(?:const|static)\s+(?:mut\s+)?" + _id(NAME))),
    ]
    imports = [
        re.compile(r"^\s*" + _PUB + r"use\s+([A-Za-z_:]+)"),
    ]
    keywords = frozenset(
        "as async await break const continue crate dyn else enum extern false fn "
        "for if impl in let loop match mod move mut pub ref return self Self "
        "static struct super trait true type unsafe use where while".split()
    )
    return LangRule(
        "rust", keywords, decls, imports,
        comment=re.compile(r"\s*//.*$"),
        ctags_kinds={"f": "function", "c": "class", "s": "struct", "g": "enum",
                      "t": "type", "i": "interface", "M": "module", "v": "variable",
                      "C": "constant", "n": "namespace", "m": "method"},
    )


def _build_csharp() -> LangRule:
    mod = r"(?:(?:public|private|protected|internal|static|abstract|sealed|partial|readonly)\s+)*"
    decls = [
        ("class", re.compile(r"(?m)^\s*" + mod + r"class\s+" + _id(NAME))),
        ("interface", re.compile(r"(?m)^\s*" + mod + r"interface\s+" + _id(NAME))),
        ("struct", re.compile(r"(?m)^\s*" + mod + r"struct\s+" + _id(NAME))),
        ("enum", re.compile(r"(?m)^\s*" + mod + r"enum\s+" + _id(NAME))),
        ("function", re.compile(r"(?m)^\s*" + mod + r"[\w<>\[\]?.,\s]+\s+" + _id(NAME) + r"\s*\([^;]*\)\s*(?:=>|\{)")),
    ]
    imports = [re.compile(r"^\s*using\s+([\w.]+)\s*;")]
    keywords = frozenset(
        "abstract as base bool break byte case catch char checked class const "
        "continue decimal default delegate do double else enum event explicit "
        "extern false finally fixed float for foreach goto if implicit in int "
        "interface internal is lock long namespace new null object operator out "
        "override params private protected public readonly ref return sbyte "
        "sealed short sizeof stackalloc static string struct switch this throw "
        "true try typeof uint ulong unchecked unsafe ushort using var virtual "
        "void volatile while".split()
    )
    return LangRule(
        "csharp", keywords, decls, imports,
        comment=re.compile(r"\s*(?://.*$|/\*.*?\*/)"),
        ctags_kinds={"c": "class", "s": "struct", "i": "interface", "g": "enum",
                      "f": "function", "m": "method", "n": "namespace",
                      "v": "variable", "p": "namespace"},
    )


def _build_swift() -> LangRule:
    mod = r"(?:(?:public|private|internal|fileprivate|open|static|class|final|override|async|mutating|nonmutating|throws?|rethrows)\s+)*"
    # The modifier group must be repeatable AND optional. It used to allow
    # exactly one modifier and required one, so `public final class Store` and
    # a bare `struct Config` were both missed — the two most common shapes.
    vis = r"(?:(?:public|private|internal|fileprivate|open|final)\s+)*"
    decls = [
        ("function", re.compile(r"(?m)^\s*" + mod + r"func\s+" + _id(NAME))),
        ("class", re.compile(r"(?m)^\s*" + vis + r"class\s+" + _id(NAME))),
        ("struct", re.compile(r"(?m)^\s*" + vis + r"struct\s+" + _id(NAME))),
        ("enum", re.compile(r"(?m)^\s*" + vis + r"enum\s+" + _id(NAME))),
        ("protocol", re.compile(r"(?m)^\s*" + vis + r"protocol\s+" + _id(NAME))),
        ("extension", re.compile(r"(?m)^\s*" + vis + r"extension\s+" + _id(NAME))),
        ("type", re.compile(r"(?m)^\s*" + vis + r"typealias\s+" + _id(NAME))),
        ("variable", re.compile(r"(?m)^\s*" + vis + r"(?:static\s+)?(?:let|var)\s+" + _id(NAME))),
    ]
    imports = [re.compile(r"^\s*import\s+([\w.]+)")]
    keywords = frozenset(
        "associatedtype break case catch class continue default defer deinit do "
        "else enum extension fallthrough false fileprivate final for func guard "
        "if import in internal is let nil open operator override private "
        "protocol public repeat rethrows return self Self static struct super "
        "subscript switch throw throws true try typealias var where while".split()
    )
    return LangRule(
        "swift", keywords, decls, imports,
        comment=re.compile(r"\s*(?://.*$|/\*.*?\*/)"),
        ctags_kinds={"c": "class", "s": "struct", "g": "enum", "f": "function",
                      "m": "method", "p": "protocol", "e": "extension",
                      "t": "type", "v": "variable"},
    )


def _build_php() -> LangRule:
    mod = r"(?:(?:public|private|protected|static|final|abstract|async)\s+)*"
    decls = [
        ("function", re.compile(r"(?m)^\s*" + mod + r"function\s+" + _id(NAME))),
        ("class", re.compile(r"(?m)^\s*(?:abstract|final)?\s*class\s+" + _id(NAME))),
        ("interface", re.compile(r"(?m)^\s*interface\s+" + _id(NAME))),
        ("trait", re.compile(r"(?m)^\s*trait\s+" + _id(NAME))),
        ("constant", re.compile(r"(?m)^\s*const\s+" + _id(NAME))),
    ]
    imports = [re.compile(r"^\s*(?:require|include|require_once|include_once)\s*[\(']?\s*([^)']+(?:\.php)?)")]
    keywords = frozenset(
        "abstract and array as break callable case catch class clone const "
        "continue declare default do echo else elseif empty enddeclare endfor "
        "endforeach endif endswitch endwhile enum extends final finally fn for "
        "foreach function global goto if implements include instanceof insteadof "
        "interface isset list match namespace new or print private protected "
        "public readonly require return static switch throw trait try unset use "
        "var while xor yield".split()
    )
    return LangRule(
        "php", keywords, decls, imports,
        comment=re.compile(r"\s*(?://|#).*$|/\*.*?\*/"),
        ctags_kinds={"f": "function", "c": "class", "i": "interface",
                      "t": "trait", "d": "constant", "v": "variable",
                      "m": "method", "n": "namespace", "p": "function"},
    )


def _build_lua() -> LangRule:
    decls = [
        ("function", re.compile(r"(?m)^\s*(?:local\s+)?function\s+" + _id(r"[A-Za-z_]\w*[\w.:]*"))),
    ]
    imports = [re.compile(r"""^\s*local\s+\w+\s*=\s*require\s*\(?\s*['"]([^'"]+)['"]""")]
    keywords = frozenset(
        "and break do else elseif end false for function goto if in local nil "
        "not or repeat return then true until while".split()
    )
    return LangRule(
        "lua", keywords, decls, imports,
        comment=re.compile(r"\s*--.*$"),
        ctags_kinds={"f": "function"},
    )


def _build_ruby() -> LangRule:
    # `def self.run` and `def Klass.run` are singleton/class methods, very
    # common in Ruby. Without an optional receiver the symbol was recorded under
    # the *receiver* name, so a file with several `def self.x` produced several
    # symbols all called `self` and `x` was never indexed at all.
    recv = r"(?:self\.|[A-Z]\w*(?:::\w+)*\.|[A-Za-z_@]\w*\.|[A-Z]\w*(?:::\w+)*::)?"
    decls = [
        ("function", re.compile(r"(?m)^\s*def\s+" + recv + _id(r"[A-Za-z_]\w*[?!=]?"))),
        ("class", re.compile(r"(?m)^\s*class\s+" + _id(r"[A-Za-z_]\w*(?:::\w+)*"))),
        ("module", re.compile(r"(?m)^\s*module\s+" + _id(r"[A-Za-z_]\w*(?:::\w+)*"))),
        ("constant", re.compile(r"(?m)^\s*" + _id(r"[A-Z][A-Z0-9_]*") + r"\s*=")),
    ]
    imports = [re.compile(r"^\s*(?:require|require_relative|load)\s+['\"]([^'\"]+)['\"]")]
    keywords = frozenset(
        "alias and begin break case class def defined do else elsif end ensure "
        "false for if in module next nil not or redo rescue retry return self "
        "super then true undef unless until when while yield".split()
    )
    return LangRule(
        "ruby", keywords, decls, imports,
        comment=re.compile(r"\s*#.*$"),
        ctags_kinds={"f": "function", "c": "class", "M": "module", "C": "constant",
                      "m": "method", "v": "variable"},
    )


def _build_perl() -> LangRule:
    decls = [
        ("function", re.compile(r"(?m)^\s*sub\s+" + _id(NAME))),
        ("package", re.compile(r"(?m)^\s*package\s+" + _id(r"[A-Za-z_]\w*(?:::\w+)*"))),
    ]
    imports = [re.compile(r"^\s*use\s+([\w:]+)")]
    keywords = frozenset(
        "unless until eq ne cmp and or not xor if elsif else while for foreach "
        "sub my our local state package use require return last next redo goto".split()
    )
    return LangRule(
        "perl", keywords, decls, imports,
        comment=re.compile(r"^\s*#.*$"),
        ctags_kinds={"f": "function", "p": "package"},
    )


def _build_r() -> LangRule:
    decls = [
        ("function", re.compile(r"(?m)^\s*" + _id(NAME) + r"\s*(?:<-|=)\s*function\s*\(")),
    ]
    imports = [re.compile(r"^\s*(?:library|require)\s*\(\s*(['\"]?)([A-Za-z0-9._]+)\1\s*\)")]
    keywords = frozenset(
        "function if else repeat while for in next break TRUE FALSE NULL Inf "
        "NaN NA".split()
    )
    return LangRule(
        "r", keywords, decls, imports,
        comment=re.compile(r"^\s*#.*$"),
        ctags_kinds={"f": "function"},
    )


def _build_sql() -> LangRule:
    decls = [
        ("table", re.compile(r"(?im)^\s*CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?" + _id(r"[A-Za-z_][\w$]*"))),
        ("procedure", re.compile(r"(?im)^\s*CREATE\s+(?:OR\s+REPLACE\s+)?PROCEDURE\s+" + _id(r"[A-Za-z_][\w$]*"))),
        ("function", re.compile(r"(?im)^\s*CREATE\s+(?:OR\s+REPLACE\s+)?FUNCTION\s+" + _id(r"[A-Za-z_][\w$]*"))),
        ("view", re.compile(r"(?im)^\s*CREATE\s+(?:OR\s+REPLACE\s+)?VIEW\s+" + _id(r"[A-Za-z_][\w$]*"))),
    ]
    imports = []
    keywords = frozenset(
        "SELECT FROM WHERE INSERT INTO UPDATE DELETE CREATE TABLE VIEW FUNCTION "
        "PROCEDURE ALTER DROP INDEX PRIMARY KEY FOREIGN NOT NULL AND OR AS ON "
        "JOIN INNER LEFT RIGHT OUTER CROSS UNION ALL DISTINCT LIMIT OFFSET ORDER "
        "GROUP BY HAVING CASE WHEN THEN ELSE END BEGIN COMMIT ROLLBACK TRIGGER "
        "GRANT REVOKE SET VALUES".split()
    )
    return LangRule(
        "sql", keywords, decls, imports,
        comment=re.compile(r"\s*--.*$"),
        ctags_kinds={"t": "table", "v": "view", "f": "function", "p": "procedure"},
    )


def _build_shell() -> LangRule:
    decls = [
        ("function", re.compile(r"(?m)^\s*(?:function\s+)?" + _id(r"[A-Za-z_][A-Za-z0-9_-]*") + r"\s*\(\s*\)\s*\{")),
        ("variable", re.compile(r"(?m)^\s*" + _id(r"[A-Za-z_][A-Za-z0-9_-]*") + r"\s*=")),
        ("alias", re.compile(r"(?m)^\s*alias\s+" + _id(r"[A-Za-z_][A-Za-z0-9_-]*"))),
    ]
    imports = [re.compile(r"^\s*(?:source|\.)\s+([./~A-Za-z0-9_/-]+)")]
    keywords = frozenset(
        "if then else elif fi case esac for while until do done function select "
        "in local return exit export readonly set unset shift source alias echo "
        "printf test break continue trap declare typeset eval exec builtin".split()
    )
    return LangRule(
        "bash", keywords, decls, imports,
        comment=re.compile(r"^\s*#.*$"),
        ctags_kinds={"f": "function", "v": "variable", "a": "alias"},
    )


def _build_fish() -> LangRule:
    decls = [
        ("function", re.compile(r"(?m)^\s*function\s+" + _id(r"[A-Za-z_][A-Za-z0-9_-]*"))),
        ("variable", re.compile(r"(?m)^\s*set\s+" + _id(r"[A-Za-z_][A-Za-z0-9_-]*"))),
    ]
    imports = [re.compile(r"^\s*source\s+([./~A-Za-z0-9_/-]+)")]
    keywords = frozenset(
        "and begin break case command contains continue else end for function if "
        "in not or return set switch while".split()
    )
    return LangRule(
        "fish", keywords, decls, imports,
        comment=re.compile(r"^\s*#.*$"),
        ctags_kinds={"f": "function", "v": "variable"},
    )


def _build_generic() -> LangRule:
    """Catch-all for languages we don't special-case.

    Deliberately conservative: only unambiguous declaration keywords and the
    two most portable import forms, so we never invent definitions.
    """
    decls = [
        ("function", re.compile(r"(?m)^\s*(?:export\s+)?(?:public\s+|private\s+|internal\s+)?(?:async\s+)?function\s+" + _id(NAME))),
        ("function", re.compile(r"(?m)^\s*(?:export\s+)?(?:public\s+)?def\s+" + _id(NAME))),
        ("function", re.compile(r"(?m)^\s*(?:export\s+)?(?:public\s+)?fn\s+" + _id(NAME))),
        ("function", re.compile(r"(?m)^\s*(?:export\s+)?(?:public\s+)?func\s+" + _id(NAME))),
        ("function", re.compile(r"(?m)^\s*(?:export\s+)?(?:public\s+)?proc\s+" + _id(NAME))),
        ("class", re.compile(r"(?m)^\s*(?:export\s+)?(?:public\s+)?(?:abstract\s+)?class\s+" + _id(NAME))),
        ("class", re.compile(r"(?m)^\s*(?:export\s+)?(?:public\s+)?(?:struct|interface|enum|trait|union)\s+" + _id(NAME))),
        ("constant", re.compile(r"(?m)^\s*(?:export\s+)?(?:public\s+)?const\s+" + _id(NAME))),
    ]
    imports = [
        re.compile(r"""^\s*(?:import|include|require)\s+[('"]?([A-Za-z0-9_./:-]+)(?:\.([A-Za-z0-9_]+))?"""),
        re.compile(r"^\s*from\s+([A-Za-z0-9_.-]+)\s+import\s+\w+"),
    ]
    keywords = frozenset(
        "if else for while return class function def fn func proc struct interface "
        "enum trait import include require new var let const static public private "
        "protected this self super".split()
    )
    return LangRule("generic", keywords, decls, imports, comment=None, ctags_kinds={})


@dataclass
class LangRule:
    id: str
    keywords: frozenset[str]
    decls: list[tuple[str, re.Pattern]]
    imports: list[re.Pattern]
    comment: re.Pattern | None = None
    ctags_kinds: dict[str, str] = field(default_factory=dict)


LANG_RULES: dict[str, LangRule] = {}
for _lang, _builder in {
    "javascript": _build_javascript,
    "c": _build_c,
    "csharp": _build_csharp,
    "java": _build_java,
    "go": _build_go,
    "rust": _build_rust,
    "swift": _build_swift,
    "php": _build_php,
    "lua": _build_lua,
    "ruby": _build_ruby,
    "perl": _build_perl,
    "r": _build_r,
    "sql": _build_sql,
    "bash": _build_shell,
    "fish": _build_fish,
    "generic": _build_generic,
    "cpp": _build_cpp,
}.items():
    LANG_RULES[_lang] = _builder()


def rule_for(language: str) -> LangRule:
    if language == "typescript":
        language = "javascript"
    return LANG_RULES.get(language) or LANG_RULES["generic"]


def _clean_line(line: str, rule: LangRule) -> str:
    if rule.comment is not None:
        return rule.comment.sub("", line)
    return line


# languages the regex rules handle well on their own; for these we skip ctags
# even if installed (the parser is already exact enough for their data format)
_HEURISTIC_ONLY = {"bash", "fish", "sql", "lua", "r"}


def index_file(
    root: Path,
    rel: str,
    src: str,
    size: int,
    mtime: float,
    language: str,
    use_ctags: bool = True,
) -> FileIndex:
    """Index one non-Python file with the language rules (+ ctags if usable)."""
    lang_id = "javascript" if language == "typescript" else language
    rule = rule_for(lang_id)
    lines = src.split("\n")

    symbols: list[Symbol] = []
    ctags_symbols: list[Symbol] = []
    if use_ctags and lang_id not in _HEURISTIC_ONLY:
        ctags_symbols = _ctags_index(rel, src, rule, mtime)

    decl_patterns = rule.decls
    import_patterns = rule.imports
    imports: list[ImportRecord] = []
    refs: list[Ref] = []
    go_in_import_block = False
    # C/C++ declarations get their own pass because they can span lines; the
    # map is keyed by the line a declaration starts on, so the loop below keeps
    # its original per-line behaviour for everything else.
    cdecls = _c_decl_map(lines, lang_id) if lang_id in ("c", "cpp") else {}
    # Lazily-built comment-stripped view of the whole file, shared by the
    # anonymous-struct branch below. It used to re-join and re-parse every line
    # *per match*; a 2.7k-line GL header has 81 `typedef struct {` occurrences,
    # so that was 81 full passes over the file and it took minutes.
    _san_cache: list[str] | None = None
    # (kind, name) pairs already emitted, so a declaration matched by two rules
    # does not produce two nodes with the same id
    c_seen: set[tuple[str, str]] = set()

    for idx, raw_line in enumerate(lines, 1):
        line = _clean_line(raw_line, rule)
        stripped = line.strip()
        if not stripped:
            continue

        # imports ----------------------------------------------------------
        matched_import = False
        for pat in import_patterns:
            m = pat.search(line)
            if m:
                mod = (m.group(1) or "").strip()
                if mod:
                    imports.append(
                        ImportRecord(mod, rel, idx, _local_hint(mod, language), [])
                    )
                matched_import = True
                break
        # go import block continuation ("import (" then quoted lines)
        if language == "go" and "import" in raw_line and "(" in raw_line:
            go_in_import_block = True
            continue
        if language == "go" and go_in_import_block:
            if raw_line.strip() == ")":
                go_in_import_block = False
                continue
            m = re.match(r'^\s*"([^"]+)"', raw_line)
            if m:
                imports.append(
                    ImportRecord(m.group(1), rel, idx, _local_hint(m.group(1), language), [])
                )
            continue
        if matched_import:
            continue

        # declarations -----------------------------------------------------
        decl_name = None
        decl_kind = None
        c_end = None
        hit = cdecls.get(idx) if cdecls else None
        if hit is not None:
            decl_kind, decl_name, c_end = hit
        else:
            for kind, pat in decl_patterns:
                try:
                    m = pat.match(line)
                    nm = m.group("name") if m else None
                except Exception:
                    m, nm = None, None
                if m and nm:
                    decl_name = nm
                    decl_kind = kind
                    break
        if decl_name is None and re.match(r"^\s*typedef\s+struct\s*\{", line):
            # anonymous struct: the name trails at the close (`} Point;`)
            try:
                if _san_cache is None:
                    from .spans import _strip_to_code as _san2
                    _san_cache = _san2("\n".join(lines)).split("\n")
                _san = _san_cache
                _depth = 0
                _started = False
                _close = None
                for _j in range(idx - 1, len(lines)):
                    for _c in _san[_j]:
                        if _c == "{":
                            _depth += 1
                            _started = True
                        elif _c == "}" and _started:
                            _depth -= 1
                            if _depth <= 0:
                                _close = _j + 1
                                break
                    if _close is not None:
                        break
                if _close is not None:
                    _m2 = re.match(r"^\s*}\s*([A-Za-z_$][A-Za-z0-9_$]*)\s*;", lines[_close - 1])
                    if _m2:
                        try:
                            from .spans import block_span as _spanA
                            _, _endA = _spanA(lines, idx, lang_id)
                        except Exception:
                            _endA = _close
                        symbols.append(Symbol(_m2.group(1), "struct", rel, idx, _endA, stripped, "", lang_id))
                        continue
            except Exception as _oc_exc:
                swallow(_oc_exc, "index/heuristic_indexer.py:749")
                pass
        if decl_name:
            # one symbol per (kind, name) per file: a declaration can be matched
            # by two paths (a `typedef struct X` and a bare `struct X`), and a
            # duplicate node id would re-point its edges
            _key = (decl_kind, decl_name)
            if _key in c_seen:
                continue
            c_seen.add(_key)
            if c_end and c_end > idx:
                # the C/C++ pass already knows where the declaration ends,
                # which matters for a multi-line prototype
                _end = c_end
            else:
                try:
                    from .spans import block_span as _span
                    _, _end = _span(lines, idx, lang_id)
                except Exception:
                    _end = idx
            symbols.append(Symbol(decl_name, decl_kind, rel, idx, _end, stripped, "", lang_id))
            continue

        # identifier usages ------------------------------------------------
        for ident in WORD.findall(stripped):
            if ident in rule.keywords:
                continue
            refs.append(Ref(ident, rel, idx, "use", ""))

    # merge ctags definitions (preferred) over heuristic definitions
    if ctags_symbols:
        ctags_names = {s.name for s in ctags_symbols}
        heuristic_kept = [s for s in symbols if s.name not in ctags_names]
        symbols = list(ctags_symbols) + heuristic_kept

    return FileIndex(
        rel, mtime, size, lang_id,
        symbols=symbols,
        refs=refs,
        imports=imports,
    )


# ---------------------------------------------------------------------------
# universal-ctags back-end
# ---------------------------------------------------------------------------

_ctags_available: bool | None = None
_ctags_json_ok: bool | None = None


def ctags_available() -> bool:
    global _ctags_available
    if _ctags_available is None:
        _ctags_available = shutil.which("ctags") is not None
    return _ctags_available


def _ctags_index(rel: str, src: str, rule: LangRule, mtime: float) -> list[Symbol]:
    """Definitions for one file via universal-ctags JSON output (or [])."""
    global _ctags_json_ok
    if not ctags_available():
        return []
    if _ctags_json_ok is False:
        return []
    try:
        if _ctags_json_ok is None:
            ver = subprocess.run(
                ["ctags", "--version"], capture_output=True, text=True, timeout=10
            )
            _ctags_json_ok = "universal" in (ver.stdout + ver.stderr).lower()
        if not _ctags_json_ok:
            return []
        proc = subprocess.run(
            ["ctags", "--output-format=json", "--fields=+nKSt", "-u", "-f", "-", "-"],
            input=src, capture_output=True, text=True, timeout=60,
        )
        if proc.returncode != 0:
            return []
    except (OSError, subprocess.SubprocessError):
        _ctags_json_ok = False
        return []

    out: list[Symbol] = []
    for line in proc.stdout.splitlines():
        if len(out) >= _MAX_CTAGS_LINES:
            break
        try:
            obj = json.loads(line)
        except ValueError as _oc_exc:
            swallow(_oc_exc, "opencode_py/index/heuristic_indexer.py:840")
            continue
        if obj.get("_type") == "file":
            continue
        name = obj.get("name")
        if not name or name.startswith("__anon"):
            continue
        try:
            lineno = int(obj.get("line", 0) or 0)
        except (ValueError, TypeError):
            lineno = 0
        kind_letter = str(obj.get("kind", ""))
        kind = rule.ctags_kinds.get(kind_letter, "unknown")
        # drop kinds we don't understand so they can be picked up by heuristics
        if kind == "unknown":
            continue
        try:
            from .spans import block_span as _span2
            _lines2 = src.split("\n")
            _, _end2 = _span2(_lines2, lineno, rule.id)
        except Exception:
            _end2 = lineno
        out.append(
            Symbol(name, kind, rel, lineno, _end2, obj.get("pattern", ""), "", rule.id)
        )
    return out


def _local_hint(mod: str, language: str) -> bool | None:
    """Guess local vs external from the import form."""
    m = mod.strip()
    if language in ("bash", "fish"):
        return False if (m.startswith(("/", "~/", "~", "$")) or "." in m) else None
    if m.startswith(("<", '"')) or m.endswith((">", '"')):
        return False  # system/absolute include
    return None
"""Tests for the resolved dependency graph (index/graph.py) and its queries.

Covers the multi-hop modes added to find_symbols — impact, path, dead, cycles,
communities, graphstats — plus the invariants that make them trustworthy:

- a call edge's caller is the innermost callable whose real span contains the
  call site (verified against `ast`, not against the indexer)
- node ids are unique, so two same-named definitions never overwrite each other
- only callables can be a caller
- the model-visible surfaces (system prompt + tool schema) actually advertise
  the new verbs, so the agent can reach them
"""

from __future__ import annotations

import ast

import pytest

from opencode_py.globals import Path as GPath
from opencode_py.index import graph as G
from opencode_py.index.engine import query


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(GPath, "cache", tmp_path)
    return tmp_path


@pytest.fixture
def repo(tmp_path):
    """A small repo with a known call chain, a cycle, and one dead function."""
    root = tmp_path / "repo"
    (root / "pkg").mkdir(parents=True)

    (root / "pkg" / "core.py").write_text(
        "def leaf(x):\n"
        "    return x + 1\n"
        "\n"
        "def middle(x):\n"
        "    return leaf(x) * 2\n"
        "\n"
        "def top(x):\n"
        "    return middle(x) + 1\n"
        "\n"
        "def orphan():\n"
        "    return 'nobody calls me'\n",
        encoding="utf-8",
    )
    (root / "pkg" / "cycle_a.py").write_text(
        "from pkg.cycle_b import b_fn\n"
        "\n"
        "def a_fn():\n"
        "    return b_fn()\n",
        encoding="utf-8",
    )
    (root / "pkg" / "cycle_b.py").write_text(
        "from pkg.cycle_a import a_fn\n"
        "\n"
        "def b_fn():\n"
        "    return a_fn()\n",
        encoding="utf-8",
    )
    (root / "app.py").write_text(
        "from pkg.core import top\n"
        "\n"
        "def main():\n"
        "    return top(1)\n",
        encoding="utf-8",
    )
    return root


# ---------------------------------------------------------------------------
# graph build
# ---------------------------------------------------------------------------

def test_graph_builds_nodes_and_call_edges(repo, cache_dir):
    from opencode_py.index.engine import _ENGINE

    files = _ENGINE.ensure(repo, force=True)
    g = G.graph_for(files, str(repo))
    names = {m["name"] for m in g.nodes.values()}
    assert {"leaf", "middle", "top", "orphan", "main"} <= names
    assert g.stats()["call_edges"] > 0


def test_node_ids_are_unique(repo, cache_dir):
    """Two definitions sharing name+container must not collapse into one node.

    A class that defines the same method name twice is enough to trigger this,
    and silently merging them re-points one definition's edges at the other.
    """
    from opencode_py.index.engine import _ENGINE

    (repo / "pkg" / "dup.py").write_text(
        "class C:\n"
        "    def m(self):\n"
        "        return 1\n"
        "def _mk():\n"
        "    return C\n"
        "class C:\n"
        "    def m(self):\n"
        "        return 2\n"
        "def caller():\n"
        "    return C().m()\n",
        encoding="utf-8",
    )
    files = _ENGINE.ensure(repo, force=True)
    g = G.graph_for(files, str(repo))
    lines = [m["line"] for m in g.nodes.values() if m["name"] == "m"]
    assert len(lines) == len(set(lines)), "two definitions of `m` collapsed"


# ---------------------------------------------------------------------------
# caller attribution
# ---------------------------------------------------------------------------

def test_caller_is_the_innermost_callable_containing_the_call(repo, cache_dir):
    from opencode_py.index.engine import _ENGINE

    files = _ENGINE.ensure(repo, force=True)
    g = G.graph_for(files, str(repo))
    for path, fi in files.items():
        scans = [
            (s.line, s.end_line,
             f"{s.file}::{s.container}.{s.name}@{s.line}" if s.container
             else f"{s.file}::{s.name}@{s.line}")
            for s in fi.symbols
            if s.kind in G._CALLABLE_KINDS and s.end_line >= s.line
        ]
        scans.sort()
        for name, line, role in G._iter_refs(fi):
            if role != "call" or not line:
                continue
            nid = G._enclosing(scans, line) or f"@{path}"
            if nid.startswith("@"):
                continue
            meta = g.nodes[nid]
            assert meta["file"] == path
            assert meta["line"] <= line <= (meta["end_line"] or 10 ** 9), (
                f"{path}:{line} attributed to a span that does not contain it")
            assert meta["kind"] in G._CALLABLE_KINDS, "a non-callable became a caller"


# ---------------------------------------------------------------------------
# the multi-hop queries
# ---------------------------------------------------------------------------

def test_impact_reports_transitive_dependents(repo, cache_dir):
    res = query("impact leaf", root=repo)
    assert "error" not in res
    files = [c["name"] for c in res["metadata"]["items"]]
    assert "middle" in files, "direct caller missing"
    assert "top" in files, "transitive caller missing"
    assert res["metadata"]["depth"] >= 2
    assert "pkg/core.py" in res["metadata"]["files"]


def test_impact_unknown_name_is_reported_not_guessed(repo, cache_dir):
    res = query("impact no_such_symbol_anywhere", root=repo)
    assert res["metadata"]["found"] is False
    assert "nothing can depend on it" in res["output"]


def test_path_finds_the_call_chain(repo, cache_dir):
    res = query("path main to leaf", root=repo)
    assert res["metadata"]["found"] is True
    hops = [h.get("name") for h in res["metadata"]["hops"] if isinstance(h, dict)]
    assert hops == ["main", "top", "middle", "leaf"]
    assert res["metadata"]["hop_count"] == 3


def test_path_reports_when_no_chain_exists(repo, cache_dir):
    res = query("path leaf to main", root=repo)
    assert res["metadata"]["found"] is False
    assert res["metadata"].get("reason")


def test_path_needs_two_names(repo, cache_dir):
    res = query("path top", root=repo)
    assert res.get("error") is True
    assert "two names" in res["output"]


def test_dead_code_finds_the_orphan_only_in_python(repo, cache_dir):
    res = query("dead", root=repo)
    names = [i["name"] for i in res["metadata"]["items"]]
    assert "orphan" in names
    assert "leaf" not in names, "a called function was reported dead"
    assert all(i["confidence"] == "exact" for i in res["metadata"]["items"])


def test_cycles_detect_the_import_cycle(repo, cache_dir):
    res = query("cycles", root=repo)
    assert res["metadata"]["count"] >= 1
    groups = res["metadata"]["groups"]
    flat = {f for g in groups for f in g}
    assert "pkg/cycle_a.py" in flat and "pkg/cycle_b.py" in flat


def test_communities_group_by_package(repo, cache_dir):
    res = query("communities", root=repo)
    comms = res["metadata"]["communities"]
    assert "pkg" in comms
    assert comms["pkg"] == 3, "core.py + cycle_a.py + cycle_b.py"
    assert "(root)" in comms and comms["(root)"] == 1, "app.py is a top-level module"


def test_coupling_lists_dependencies(repo, cache_dir):
    res = query("communities", root=repo)
    links = res["metadata"]["coupling"]
    assert any(l["from"] == "(root)" and l["to"] == "pkg" for l in links)


def test_graphstats_reports_confidence_mix(repo, cache_dir):
    res = query("graphstats", root=repo)
    m = res["metadata"]
    assert m["nodes"] > 0
    assert m["call_edges"] >= m["exact_edges"] + m["likely_edges"] + m["weak_edges"] - 1
    assert m["build_ms"] >= 0
    assert "exact" in res["output"]


def test_bare_graphstats_is_not_read_as_communities(repo, cache_dir):
    """`graph` used to match as a verb and steal `stats` as its term."""
    assert query("graphstats", root=repo)["metadata"].get("build_ms") is not None
    assert "Knowledge graph status" in query("graphstats", root=repo)["output"]


# ---------------------------------------------------------------------------
# the agent can actually reach these
# ---------------------------------------------------------------------------

def test_system_prompt_advertises_the_graph_queries(repo, cache_dir):
    from opencode_py.agent.system import build_system_prompt
    from opencode_py.config import Config

    prompt = build_system_prompt(
        directory=repo, worktree=repo, provider_id="p", model_id="m",
        cfg=Config(), agent="build",
    )
    assert "impact <name>" in prompt, "graph guidance missing from the system prompt"
    for verb in ("impact", "path", "communities", "graphstats"):
        assert verb in prompt, f"{verb} not advertised to the model"


def test_tool_schema_advertises_the_graph_queries():
    from opencode_py.tools.find_symbols import tool

    surface = tool().description + str(tool().parameters)
    for verb in ("impact run_tool", "path run_tool to send", "dead",
                 "cycles", "communities", "graphstats"):
        assert verb in surface, f"{verb} missing from the tool schema"


def test_graph_survives_an_empty_repo(tmp_path, cache_dir):
    from opencode_py.index.engine import _ENGINE

    empty = tmp_path / "empty"
    empty.mkdir()
    files = _ENGINE.ensure(empty, force=True)
    g = G.graph_for(files, str(empty))
    assert g.stats()["call_edges"] == 0
    assert G.import_cycles(files, str(empty)) == []


def test_ast_reference_coverage_is_near_total(repo, cache_dir):
    """Guards the ast walker: nearly every Load-context name is indexed.

    This is the invariant that the whole call graph rests on — if refs stop
    being captured, edges silently disappear and impact/path go quiet rather
    than wrong, which is the failure mode that is hardest to notice.
    """
    from opencode_py.index.python_indexer import _SKIP_BUILTINS, index_file

    src = (repo / "pkg" / "core.py").read_text(encoding="utf-8")
    fi = index_file(repo, "pkg/core.py", src, len(src.encode()), 0)
    got = {(r.line, r.name) for r in fi.refs}
    defined = {s.name for s in fi.symbols}
    missed = []
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            name = n.id
        elif isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Load):
            name = n.attr
        else:
            continue
        if name in _SKIP_BUILTINS or name in defined:
            continue
        if (n.lineno, name) not in got:
            missed.append(f"line {n.lineno}: {name}")
    assert not missed, f"indexer dropped Load-context refs: {missed}"


def test_assignment_target_chain_is_indexed(tmp_path, cache_dir):
    """`a.b.c = x` reads a and a.b on the way to the write."""
    from opencode_py.index.python_indexer import index_file

    src = "def setup(self):\n    self.session.state.flag = True\n"
    fi = index_file(tmp_path, "m.py", src, len(src.encode()), 0)
    names = {r.name for r in fi.refs}
    assert "session" in names and "state" in names
    assert "flag" not in names, "the written attribute is not a read"


# ---------------------------------------------------------------------------
# the reported numbers must not be silently capped
# ---------------------------------------------------------------------------

def _star_graph(g, hub, n):
    """hub <- n callers, so the reverse BFS from hub reaches n nodes."""
    for i in range(n):
        nid = f"f.py::caller{i}@{i + 1}"
        g.nodes[nid] = {"name": f"caller{i}", "kind": "function", "file": "f.py",
                        "line": i + 1, "end_line": i + 1, "container": "",
                        "signature": "", "language": "python"}
        g.out.setdefault(nid, [])
        g.inn.setdefault(nid, [])
        g.out[nid].append(hub)
        g.inn.setdefault(hub, []).append(nid)


def test_bfs_cap_does_not_overshoot_one_level():
    """A per-level cap check let a whole level through: cap=400 reported 817."""
    g = G.Graph()
    hub = "f.py::hub@1"
    g.nodes[hub] = {"name": "hub", "kind": "function", "file": "f.py", "line": 1,
                    "end_line": 1, "container": "", "signature": "", "language": "python"}
    g.out[hub] = []
    g.inn[hub] = []
    _star_graph(g, hub, 900)
    assert len(G._bfs_up(g, [hub], 0)) == 900, "uncapped must reach everything"
    assert len(G._bfs_up(g, [hub], 50)) == 50, "cap must be honoured exactly"


def test_impact_reports_the_true_total_not_a_capped_one(repo, cache_dir):
    from opencode_py.index.engine import _ENGINE

    files = _ENGINE.ensure(repo, force=True)
    uncapped = G.impact_of(files, "leaf", cap=0, root_key=str(repo))
    capped = G.impact_of(files, "leaf", cap=1, root_key=str(repo))
    assert uncapped["count"] > capped["count"]
    assert capped["truncated"] is True, "a bounded query must admit it is bounded"
    assert uncapped["truncated"] is False


def test_dead_reports_the_true_total_not_the_display_slice(repo, cache_dir):
    """Counting the displayed rows made this say 60 when the total was 454."""
    res = query("dead", root=repo, limit=1)
    assert res["metadata"]["count"] >= 1
    assert res["metadata"]["shown"] <= 1
    assert res["metadata"]["count"] >= res["metadata"]["shown"]
    assert "and" in res["output"] or res["metadata"]["count"] == 1


def test_bare_dead_is_python_only(repo, cache_dir):
    """`*` is the term a bare `dead` gets; it must not opt into every language."""
    plain = query("dead", root=repo)
    wide = query("dead all", root=repo)
    assert plain["metadata"]["all_languages"] is False
    assert wide["metadata"]["all_languages"] is True
    assert all(i["confidence"] == "exact" for i in plain["metadata"]["items"])


def test_dead_all_marks_weak_hits(repo, cache_dir):
    (repo / "widget.js").write_text(
        "function neverCalled() { return 1; }\n", encoding="utf-8")
    res = query("dead all", root=repo)
    assert res["metadata"]["weak"] >= 1
    assert "regex-derived" in res["output"]


def test_path_reports_ambiguous_endpoints(repo, cache_dir):
    """`run` defined in several files resolves to one; the answer must say so."""
    (repo / "pkg" / "other.py").write_text(
        "def leaf2(x):\n    return x\n", encoding="utf-8")
    res = query("path definitely_not_here_a to leaf2", root=repo)
    assert res["metadata"]["ambiguous"] is not None
    res2 = query("path main to nope_not_real", root=repo)
    assert "ambiguous" in res2["metadata"]


# ---------------------------------------------------------------------------
# C / C++ declaration extraction
# ---------------------------------------------------------------------------

C_HEADER = """\
#ifndef T_H
#define T_H
#include <stddef.h>
struct Point { int x, y; };
typedef struct Point PointRef;
typedef int (*Callback)(int a, int b);
ssize_t read_all(int fd, char *buf, size_t n);
int  process(Callback cb, void *ctx);
extern int  split_proto(
    int a,
    int b);
static inline int add(int a, int b) { return a + b; }
#endif
"""

CPP_HEADER = """\
namespace app { class Widget; }
class Widget {
public:
    void render(int x) const;
    int  value() const noexcept { return v_; }
private:
    int v_;
};
template <typename T>
T clamp(T lo, T hi,
        T v) {
    return v;
}
"""


def _hx(tmp_path, name, src, lang):
    from opencode_py.index.heuristic_indexer import index_file

    rel = f"x{_EXT.get(lang, '')}"
    return index_file(tmp_path, rel, src, len(src), 0, lang, use_ctags=False)


_EXT = {"c": ".c", "cpp": ".hpp"}


def test_c_finds_prototypes_in_headers(tmp_path, cache_dir):
    """A header is prototypes; the old rule only matched `{` bodies, so every
    real header declaration was invisible."""
    fi = _hx(tmp_path, "t.h", C_HEADER, "c")
    got = {s.name for s in fi.symbols}
    for want in ("read_all", "process", "split_proto", "add"):
        assert want in got, f"{want} not found; got {sorted(got)}"


def test_c_finds_typedefs_and_structs(tmp_path, cache_dir):
    fi = _hx(tmp_path, "t.h", C_HEADER, "c")
    got = {(s.kind, s.name) for s in fi.symbols}
    assert ("struct", "Point") in got
    assert ("type", "PointRef") in got, "typedef of a struct tag"
    assert ("type", "Callback") in got, "function-pointer typedef"
    assert ("macro", "T_H") in got


def test_c_spans_a_multiline_prototype(tmp_path, cache_dir):
    fi = _hx(tmp_path, "t.h", C_HEADER, "c")
    s = next(x for x in fi.symbols if x.name == "split_proto")
    assert s.end_line > s.line, "multi-line prototype collapsed to one line"


def test_c_emits_no_duplicate_symbols(tmp_path, cache_dir):
    """Two matches for one declaration would create two nodes with the same id
    and silently re-point its edges at the other."""
    fi = _hx(tmp_path, "t.h", C_HEADER, "c")
    pairs = [(s.kind, s.name) for s in fi.symbols]
    assert len(pairs) == len(set(pairs)), f"duplicates: {pairs}"


def test_cpp_has_real_rules_not_the_generic_fallback(tmp_path, cache_dir):
    from opencode_py.index.heuristic_indexer import rule_for

    assert rule_for("cpp").id == "cpp", "C++ was falling back to the generic rule"


SWIFT_HEADER = """\
import Foundation

public final class Store: Storable {
    public static let maxRetries = 3
    public func put(_ key: String) {}
}

struct Config {
    let name: String
}

enum State {
    case ready
}
"""

RUBY_FILE = """\
module Example
  class Store
    def initialize
      @items = {}
    end

    def self.build
      new
    end
  end
end
"""


def test_swift_handles_repeated_and_absent_modifiers(tmp_path, cache_dir):
    """`public final class` and a bare `struct` were both missed.

    The modifier group allowed exactly one modifier and required one, which is
    neither shape. Swift recall went 41.7% -> 91.7% on the sample suite.
    """
    fi = _hx(tmp_path, "M.swift", SWIFT_HEADER, "swift")
    got = {s.name for s in fi.symbols}
    for want in ("Store", "Config", "State", "put"):
        assert want in got, f"{want} not found; got {sorted(got)}"


def test_ruby_records_the_method_name_not_the_receiver(tmp_path, cache_dir):
    """`def self.build` was indexed as a symbol called `self`.

    The pattern had no optional receiver, so every singleton method in a file
    produced a symbol named after the receiver and the method was never indexed.
    Ruby recall went 85.7% -> 100% on the sample suite.
    """
    fi = _hx(tmp_path, "m.rb", RUBY_FILE, "ruby")
    got = {s.name for s in fi.symbols}
    assert "build" in got, f"`def self.build` not indexed; got {sorted(got)}"
    assert "self" not in got, "the receiver leaked in as a symbol name"
    assert "initialize" in got and "Store" in got and "Example" in got


def test_cpp_finds_class_members_and_namespaces(tmp_path, cache_dir):
    fi = _hx(tmp_path, "t.hpp", CPP_HEADER, "cpp")
    got = {s.name for s in fi.symbols}
    for want in ("Widget", "render", "value", "clamp", "app"):
        assert want in got, f"{want} not found; got {sorted(got)}"


def test_c_and_cpp_rules_do_not_break_other_languages(tmp_path, cache_dir):
    cases = {
        "go": ("package m\nfunc Hello(x int) int { return h(x) }\n", {"Hello"}),
        "rust": ("pub fn run(x: i32) -> i32 { helper(x) }\n", {"run"}),
        "javascript": ("export function run(x) { return helper(x); }\n", {"run"}),
        "typescript": ("export function run(x: number) { return helper(x); }\n",
                       {"run"}),
    }
    for lang, (src, want) in cases.items():
        fi = _hx(tmp_path, "x", src, lang)
        got = {s.name for s in fi.symbols}
        assert want <= got, f"{lang} regressed: got {sorted(got)}"


def test_brace_index_is_exact_and_fast(tmp_path):
    """The counting fast path must agree with the character walk exactly.

    It is only used on files with no line that both opens and closes a brace;
    anything else falls back, so a divergence here means the guard is wrong.
    """
    from opencode_py.index import spans

    def exact(san, start, total):
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

    for src in (
        "int a;\nint f(void) {\n  return 1;\n}\nint b;\n",
        "struct S { int x; };\nvoid g(void) { }\n",
        "void h(void) { int x = 1; }\nvoid i(void) {\n  if (x) { }\n}\n",
        "int j(void)\n{\n  return 0;\n}\n",
    ):
        san = spans._strip_to_code(src).split("\n")
        total = len(san)
        for start in range(1, total + 1):
            assert spans._brace_end(san, start, total) == exact(san, start, total), (
                f"start={start} in {src!r}")


def test_sanitized_view_is_cached_per_file(tmp_path):
    """`_strip_to_code` used to run per declaration; that was the hang."""
    from opencode_py.index import spans

    lines = ["void f(void) {", "  int x = 1;", "}", "int y;"]
    first = spans._sanitized(lines)
    second = spans._sanitized(lines)
    assert first is second, "sanitized view rebuilt instead of reused"


def test_enclosing_does_not_claim_lines_past_a_known_span():
    """An open-ended (end_line == 0) symbol must not swallow the rest of the file.

    It used to answer for any line after it started, so a call 200 lines below a
    function that ended at line 10 was attributed to that function — inventing an
    edge that does not exist.
    """
    scans = sorted([(10, 0, "unknown_end"), (20, 100, "outer"), (30, 50, "inner")])
    assert G._enclosing(scans, 10) == "unknown_end"
    assert G._enclosing(scans, 25) == "outer"
    assert G._enclosing(scans, 35) == "inner"
    assert G._enclosing(scans, 200) is None, "claimed a line outside every span"


def test_enclosing_still_works_when_no_span_is_known():
    """Files whose indexers record no end line must keep caller attribution."""
    scans = sorted([(10, 0, "u1"), (50, 0, "u2")])
    assert G._enclosing(scans, 11) == "u1"
    assert G._enclosing(scans, 55) == "u2"
    assert G._enclosing(scans, 300) == "u2"


def test_exact_match_is_not_crowded_out_by_substring_noise(repo, cache_dir):
    """`def X` sorted by document order let the limit cut the real definition.

    70 `zz_*_target` decoys filled the 60-result limit and `target` itself was
    missing from the answer, so the agent was shown 60 wrong hits instead of
    the one it asked for. Results must rank exact matches first.
    """
    (repo / "noisy.py").write_text(
        "".join(f"def zz_x{i}_target():\n    return {i}\n\n" for i in range(70))
        + "def target():\n    return 999\n",
        encoding="utf-8",
    )
    res = query("def target", root=repo, limit=60)
    names = [i["name"] for i in res["metadata"]["items"]]
    assert "target" in names, "the exact match was crowded out"
    assert names[0] == "target", "exact match must rank first"

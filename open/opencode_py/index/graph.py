"""Resolved dependency graph over the symbol index — standard library only.

The index keeps definitions (:class:`Symbol`) and usages (:class:`Ref`, whose
``container`` names the enclosing function) as two independent flat lists. Every
existing query is therefore a single hop: find a name, list where it appears.

This module joins those two lists into an actual graph so questions that need
*traversal* have something to walk:

- ``impact_of``      — everything that transitively depends on a symbol
- ``path_between``   — the shortest chain of calls from A to B
- ``dead_symbols``   — definitions nobody calls or reads
- ``import_cycles``  — circular file dependencies
- ``communities``    — which files belong to the same subsystem
- ``coupling``       — which subsystems call each other, and how hard

Edge confidence mirrors the honest-audit convention of the upstream projects:
``exact`` when the call site sits in the same file as the definition (a Python
``ast`` fact, not a guess), ``likely`` when the name was disambiguated by the
caller's own imports, ``weak`` when only a project-unique name matched. Names
still ambiguous after all three passes are dropped *and counted*, so a wrong
edge can never masquerade as a real one.

Only Python is parsed exactly. The other 13 languages come from line-based
regex rules, so ``dead_symbols`` refuses to guess outside Python unless asked.

Performance notes, since this runs on a phone:
- ref rows are read as raw compact tuples; touching ``fi.refs`` would build a
  Ref object per row (59K of them) and dominates the runtime
- a name that no definition matches (builtins, stdlib) is dropped before any
  work beyond one dict lookup
- the graph is rebuilt only when the index fingerprint changes, and every
  expensive traversal is memoised on top of that
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import Any, Iterable, Iterator

from .model import FileIndex, Ref, Symbol

# kinds that can be a call target
_CALLABLE_KINDS = {"function", "method", "class", "constructor"}

# ref roles that mean "somebody referenced this name somewhere"
_READ_ROLES = ("use", "attribute", "import")

# Words the line-based indexers can mistake for a function name. Python never
# hits this (its defs come from the ast), but `dead all` on a JS file happily
# reported `if` and `while` as unreferenced functions, which is noise, not a
# finding.
_NOT_A_NAME = frozenset({
    # python
    "if", "else", "elif", "for", "while", "try", "except", "finally", "raise",
    "return", "break", "continue", "pass", "yield", "lambda", "with", "as",
    "import", "from", "def", "class", "global", "nonlocal", "assert", "del",
    "and", "or", "not", "in", "is", "match", "case", "self", "super",
    # c / c++ / java / c# / go / rust / js / ts
    "switch", "do", "catch", "new", "delete", "void", "int", "char", "long",
    "short", "float", "double", "bool", "const", "let", "var", "static",
    "public", "private", "protected", "struct", "union", "enum", "interface",
    "func", "fn", "impl", "pub", "crate", "mod", "package", "namespace",
    "using", "template", "typename", "this", "sizeof", "typeof", "instanceof",
    "function", "vararg", "defer", "go", "chan", "map", "range", "select",
    "throw", "throws", "try", "synchronized", "transient", "volatile",
    "abstract", "final", "extends", "implements", "super", "else if",
    # shell / ruby / perl / lua / sql / generic
    "fi", "esac", "done", "then", "elif", "end", "echo", "local", "export",
    "function", "endfunction", "begin", "until", "unless", "sub", "my",
    "our", "print", "puts", "require", "nil", "foreach", "select", "where",
    "insert", "update", "delete", "create", "table", "index", "declare",
    "goto", "label", "program", "procedure", "defun", "call", "repeat",
})


class Graph:
    """Resolved node/edge store. Built once per index snapshot, then queried."""

    __slots__ = ("nodes", "out", "inn", "mod_out", "mod_in", "exact_edges",
                 "likely_edges", "weak_edges", "dropped_ambiguous", "refs_seen",
                 "fingerprint", "build_ms")

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.out: dict[str, list[str]] = {}
        self.inn: dict[str, list[str]] = {}
        self.mod_out: dict[str, list[str]] = {}
        self.mod_in: dict[str, list[str]] = {}
        self.exact_edges = 0
        self.likely_edges = 0
        self.weak_edges = 0
        self.dropped_ambiguous = 0
        self.refs_seen = 0
        self.fingerprint: tuple = ()
        self.build_ms = 0.0

    def stats(self) -> dict[str, Any]:
        total = self.exact_edges + self.likely_edges + self.weak_edges
        return {
            "nodes": len(self.nodes),
            "call_edges": total,
            "exact_edges": self.exact_edges,
            "likely_edges": self.likely_edges,
            "weak_edges": self.weak_edges,
            "resolved_pct": round(100.0 * total / self.refs_seen, 1) if self.refs_seen else 0.0,
            "dropped_ambiguous": self.dropped_ambiguous,
            "import_edges": sum(len(v) for v in self.mod_out.values()),
            "modules": len(self.mod_out),
            "build_ms": round(self.build_ms, 1),
        }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _id_of(file: str, container: str, name: str, line: int) -> str:
    if container:
        return f"{file}::{container}.{name}@{line}"
    return f"{file}::{name}@{line}"


def _node_id(sym: Symbol) -> str:
    """Unique id for a definition.

    The definition line is part of the id because name+container is NOT unique:
    one class can define ``on_result`` twice, and the line-based indexers
    produce plenty of repeats. 130 of 5232 ids collided, and because a dict
    keeps the last writer, the earlier definition's edges were silently
    re-pointed at the later one.
    """
    if sym.container:
        return f"{sym.file}::{sym.container}.{sym.name}@{sym.line}"
    return f"{sym.file}::{sym.name}@{sym.line}"


def _iter_refs(fi: FileIndex) -> Iterator[tuple[str, int, str]]:
    """Yield (name, line, role) without materialising one Ref per row.

    Cache rows are compact: [name, file, line] / +role / +container. Reading the
    raw tuples avoids allocating a Ref for every row, which is otherwise the
    single largest cost in the build.
    """
    raw = fi.__dict__.get("_refs_raw") or ()
    for row in raw:
        if type(row) is tuple or type(row) is list:
            if len(row) < 3:
                continue
            yield str(row[0]), int(row[2] or 0), (str(row[3]) if len(row) > 3 else "use")
        elif isinstance(row, Ref):
            yield row.name, row.line, row.role
        elif isinstance(row, dict):
            name = row.get("name")
            if name:
                yield str(name), int(row.get("line") or 0), str(row.get("role", "use"))


def _fingerprint(files: dict[str, FileIndex]) -> tuple:
    """Cheap staleness key: file count + mtime/size totals.

    O(files), not O(symbols) — ~180 iterations, so callers can re-check on every
    query without paying for it. A content change preserving every mtime *and*
    size slips through: the same trade the index cache already makes.
    """
    return (len(files), sum(f.mtime for f in files.values()),
            sum(f.size for f in files.values()))


def _module_candidates(module: str) -> tuple[str, ...]:
    m = module.replace(".", "/").lstrip("./")
    if not m:
        return ()
    return (f"{m}.py", f"{m}/__init__.py", m)


def _stem_map(files: dict[str, FileIndex]) -> dict[str, list[str]]:
    """File stem -> the indexed paths that could declare it.

    Built once per snapshot. The previous per-import fallback looped every
    file calling os.path.basename/splitext, which on a phone came to 232K
    posixpath calls and 90% of the entire build. One dict lookup replaces it.
    """
    out: dict[str, list[str]] = {}
    for path in files:
        base = path.rsplit("/", 1)[-1]
        dot = base.rfind(".")
        stem = base[:dot] if dot > 0 else base
        bucket = out.get(stem)
        if bucket is None:
            out[stem] = [path]
        else:
            bucket.append(path)
    return out


def _resolve_module(files: dict[str, FileIndex], stems: dict[str, list[str]],
                    module: str, path: str = "") -> str | None:
    """Dotted module -> an indexed file path, or None (stdlib / external).

    Most confident first: exact candidate path, then a sibling of the importing
    file (a relative import points next to its caller), then a project-unique
    stem. An ambiguous stem returns None rather than guessing between two real
    files.
    """
    for cand in _module_candidates(module):
        if cand in files:
            return cand
    hit = stems.get(module.rsplit(".", 1)[-1])
    if not hit:
        return None
    if path:
        head = path.rsplit("/", 1)[0] if "/" in path else ""
        for cand in hit:
            if (cand.rsplit("/", 1)[0] if "/" in cand else "") == head:
                return cand
    return hit[0] if len(hit) == 1 else None


def _enclosing(scans: list[tuple[int, int, str]], line: int) -> str | None:
    """Innermost *callable* definition whose line span contains `line`.

    `scans` is one file's callable definitions as (start, end, node_id) sorted
    by start. Only callables belong in it: a module-level constant or a local
    variable has a span too, and letting one answer "who called this?" produced
    callers that were variables.

    Nesting means the container chain is ordered by start, so the innermost
    match is the *last* candidate with start <= line that still contains it.
    Bisect to the first candidate at/before the line, then walk back.

    A span with ``end == 0`` means the end is unknown (the line-based indexers
    do not record one). Those are accepted only after every symbol with a
    known, containing span has been ruled out — treating unknown as
    "contains everything" attributed 210 call sites to a function that did not
    actually contain them.
    """
    lo, hi = 0, len(scans)
    while lo < hi:
        mid = (lo + hi) >> 1
        if scans[mid][0] <= line:
            lo = mid + 1
        else:
            hi = mid
    saw_known = False
    for i in range(lo - 1, -1, -1):
        start, end, nid = scans[i]
        if end:
            saw_known = True
            if end >= line:
                return nid
        else:
            # An unknown end means the span is open-ended, so it *may* contain
            # this line — but only trust that when nothing with a known span
            # starts before it. Otherwise a symbol ending at line 10 would
            # claim a call on line 200 and invent a false edge.
            if not saw_known:
                return nid
    return None


def package_of(file: str) -> str:
    """Subsystem a file belongs to: `opencode_py/tui`, `(root)`, ..

    A single-segment path is a top-level module, so it belongs to the root
    package. Bucketing by its own filename instead made every top-level file
    its own "community" — 30 subsystems, most holding one file.
    """
    parts = [p for p in str(file).replace(os.sep, "/").split("/") if p]
    if len(parts) <= 1:
        return "(root)"
    if parts[0] in ("opencode_py", "src", "lib", "app") and len(parts) >= 3:
        return f"{parts[0]}/{parts[1]}"
    return parts[0]


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def build(files: dict[str, FileIndex]) -> Graph:
    """Join definitions + usages into a resolved graph. Never raises."""
    g = Graph()
    t0 = time.monotonic()
    try:
        _build(files, g)
    except Exception:
        # A malformed snapshot must degrade to an empty graph, never crash a
        # query: callers see zero nodes and report that honestly.
        g = Graph()
    g.fingerprint = _fingerprint(files)
    g.build_ms = (time.monotonic() - t0) * 1000
    return g


def _build(files: dict[str, FileIndex], g: Graph) -> None:
    nodes = g.nodes
    out: dict[str, list[str]] = defaultdict(list)
    inn: dict[str, list[str]] = defaultdict(list)
    mod_out: dict[str, list[str]] = defaultdict(list)
    mod_in: dict[str, list[str]] = defaultdict(list)

    # 1. definitions -> nodes, plus a name -> definitions lookup
    by_name: dict[str, list[Symbol]] = defaultdict(list)
    scans: dict[str, list[tuple[int, int, str]]] = {}
    for path, fi in files.items():
        file_scans: list[tuple[int, int, str]] = []
        for s in fi.symbols:
            nid = _id_of(s.file, s.container, s.name, s.line)
            nodes[nid] = {
                "name": s.name, "kind": s.kind, "file": s.file, "line": s.line,
                "end_line": s.end_line, "container": s.container,
                "signature": s.signature, "language": s.language,
            }
            if s.kind in _CALLABLE_KINDS:
                by_name[s.name].append(s)
                # only callables can *contain* a call site
                if s.end_line >= s.line:
                    file_scans.append((s.line, s.end_line, nid))
        file_scans.sort()
        scans[path] = file_scans
    by_name_get = by_name.get  # bind once: 59K lookups follow

    # 2. imports -> module edges (drives cycles + coupling)
    stems = _stem_map(files)
    for path, fi in files.items():
        here = f"@{path}"
        for imp in fi.imports:
            target = _resolve_module(files, stems, imp.module, path)
            if not target or target == path:
                continue
            there = f"@{target}"
            mod_out[here].append(there)
            mod_in[there].append(here)

    # per-file import sets let an ambiguous name be resolved through the
    # caller's own dependencies instead of dropped
    imported_by: dict[str, set[str]] = {
        here[1:]: {t[1:] for t in targets} for here, targets in mod_out.items()}

    # 3. call refs -> caller -> callee edges
    for path, fi in files.items():
        file_scans = scans.get(path)
        if not file_scans:
            continue
        local_imports = imported_by.get(path, ())
        for name, line, role in _iter_refs(fi):
            if role != "call":
                continue
            cands = by_name_get(name)
            if not cands:
                continue  # builtin / stdlib / unknown: nothing to link
            if len(cands) == 1:
                target = cands[0]
                confidence = 1 if target.file == path else 2
            else:
                target = _disambiguate(cands, path, local_imports)
                if target is None:
                    g.dropped_ambiguous += 1
                    continue
                confidence = 1 if target.file == path else 3
            tid = _id_of(target.file, target.container, target.name, target.line)
            caller = _enclosing(file_scans, line) or f"@{path}"
            # a direct recursive call is still a real call site. Dropping it
            # hid the recursion from `callers` AND made every recursive
            # function look dead, because nothing else pointed at it.
            out[caller].append(tid)
            inn[tid].append(caller)
            if confidence == 1:
                g.exact_edges += 1
            elif confidence == 2:
                g.likely_edges += 1
            else:
                g.weak_edges += 1
            g.refs_seen += 1

    def _freeze(table: dict[str, list[str]]) -> dict[str, list[str]]:
        for key, val in table.items():
            if len(val) > 1:
                table[key] = sorted(set(val))
        return dict(table)

    g.out = _freeze(out)
    g.inn = _freeze(inn)
    g.mod_out = _freeze(mod_out)
    g.mod_in = _freeze(mod_in)
    for nid in nodes:
        g.out.setdefault(nid, [])
        g.inn.setdefault(nid, [])


def _disambiguate(cands: list[Symbol], path: str,
                  local_imports: Iterable[str]) -> Symbol | None:
    """Pick one definition among same-named ones, most confident first.

    1. defined in the caller's own file   -> exact
    2. defined in a file the caller imports -> likely
    3. otherwise                          -> None (dropped, and counted)

    A name defined twice in one file stays ambiguous: picking one would invent
    an edge, and inventing edges is worse than reporting the gap.
    """
    local = [s for s in cands if s.file == path]
    if local:
        return local[0] if len(local) == 1 else None
    if local_imports:
        imp = [s for s in cands if s.file in local_imports]
        if len(imp) == 1:
            return imp[0]
        if imp:
            return None
    return None


# ---------------------------------------------------------------------------
# caching
# ---------------------------------------------------------------------------

_GRAPH_CACHE: dict[str, tuple[tuple, Graph]] = {}
_MEMO: dict[tuple, Any] = {}


def graph_for(files: dict[str, FileIndex], root_key: str = "") -> Graph:
    """Return a cached Graph, rebuilding only when the index snapshot changed."""
    fp = _fingerprint(files)
    hit = _GRAPH_CACHE.get(root_key)
    if hit is not None and hit[0] == fp:
        return hit[1]
    g = build(files)
    if len(_GRAPH_CACHE) > 8:
        _GRAPH_CACHE.clear()
        _MEMO.clear()
    _GRAPH_CACHE[root_key] = (fp, g)
    return g


def _memoized(root_key: str, files: dict[str, FileIndex], kind: str, fn):
    """Cache an expensive whole-graph derivation on the graph fingerprint."""
    graph = graph_for(files, root_key)
    key = (root_key, kind, graph.fingerprint)
    if key in _MEMO:
        return _MEMO[key]
    value = fn()
    _MEMO[key] = value
    return value


def invalidate() -> None:
    """Drop cached graphs (call after a force reindex)."""
    _GRAPH_CACHE.clear()
    _MEMO.clear()


# ---------------------------------------------------------------------------
# traversals
# ---------------------------------------------------------------------------

def _bfs_up(graph: Graph, start: list[str], cap: int) -> dict[str, int]:
    """Reverse BFS: hop distance from `start` back to every node reaching it."""
    inn = graph.inn
    seen: dict[str, int] = {}
    seed = set(start)
    frontier = list(start)
    depth = 0
    # cap == 0 means no cap. The check is per node, not per level: a per-level
    # check lets a whole BFS level overshoot, which is how a 400 cap ended up
    # reporting a count of 817 and silently lying about the total.
    while frontier:
        depth += 1
        nxt: list[str] = []
        for nid in frontier:
            for caller in inn.get(nid, ()):
                if caller in seen or caller in seed:
                    continue
                seen[caller] = depth
                nxt.append(caller)
                if cap and len(seen) >= cap:
                    return seen
        frontier = nxt
    return seen


def _bfs_forward(graph: Graph, start: str, goal: str, cap: int) -> list[str] | None:
    out = graph.out
    prev: dict[str, str | None] = {start: None}
    frontier = [start]
    while frontier and len(prev) < cap:
        nxt: list[str] = []
        for nid in frontier:
            for callee in out.get(nid, ()):
                if callee in prev:
                    continue
                prev[callee] = nid
                if callee == goal:
                    path = [callee]
                    cur: str | None = nid
                    while cur is not None:
                        path.append(cur)
                        cur = prev[cur]
                    path.reverse()
                    return path
                nxt.append(callee)
        frontier = nxt
    return None


def _tarjan(nodes: list[str], out: dict[str, list[str]]) -> list[list[str]]:
    """Iterative Tarjan SCC — finds every cycle group, no recursion limit."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on: set[str] = set()
    stack: list[str] = []
    result: list[list[str]] = []
    counter = 0
    for root in nodes:
        if root in index:
            continue
        work: list[tuple[str, int]] = [(root, 0)]
        while work:
            node, pi = work[-1]
            if pi == 0:
                index[node] = low[node] = counter
                counter += 1
                stack.append(node)
                on.add(node)
            succ = out.get(node, ())
            recursed = False
            for i in range(pi, len(succ)):
                nxt = succ[i]
                if nxt not in index:
                    work[-1] = (node, i + 1)
                    work.append((nxt, 0))
                    recursed = True
                    break
                if nxt in on:
                    low[node] = min(low[node], index[nxt])
            if recursed:
                continue
            if low[node] == index[node]:
                comp: list[str] = []
                while True:
                    w = stack.pop()
                    on.discard(w)
                    comp.append(w)
                    if w == node:
                        break
                if len(comp) > 1 or node in succ:
                    result.append(sorted(comp))
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
    return result


# ---------------------------------------------------------------------------
# public queries
# ---------------------------------------------------------------------------

def _pick(graph: Graph, term: str) -> list[str]:
    """Resolve a user term to node ids, preferring exact name matches."""
    if not term:
        return []
    if term in graph.nodes:
        return [term]
    names = graph.nodes
    exact = sorted(n for n, m in names.items() if m["name"] == term)
    if exact:
        return exact
    short = term.rsplit(".", 1)[-1]
    if short != term:
        hit = sorted(n for n, m in names.items() if m["name"] == short)
        if hit:
            return hit[:8]
    return []


def impact_of(files: dict[str, FileIndex], term: str, cap: int = 0,
              root_key: str = "") -> dict[str, Any]:
    """Blast radius: every node that transitively reaches `term`.

    Uncapped by default. The widest hub in a real repo had 1281 dependents and
    the full reverse BFS took 3ms, so a cap bought nothing and cost accuracy:
    the 400-node default made `append` report 817 dependents instead of 1281,
    with no indication anything was missing. Pass ``cap`` to bound it; when it
    bites, ``truncated`` is set so the caller can say so.
    """
    graph = graph_for(files, root_key)
    targets = _pick(graph, term)
    if not targets:
        return {"term": term, "found": False, "callers": [], "count": 0, "depth": 0,
                "files": [], "file_count": 0, "truncated": False, "stats": graph.stats()}
    distances = _bfs_up(graph, targets, cap)
    truncated = bool(cap) and len(distances) >= cap
    nodes = graph.nodes
    rows = []
    for nid, depth in distances.items():
        meta = nodes.get(nid)
        if meta is None:
            continue
        rows.append({
            "node": nid, "name": meta["name"], "kind": meta["kind"],
            "file": meta["file"], "line": meta["line"],
            "container": meta["container"], "depth": depth,
        })
    rows.sort(key=lambda r: (r["depth"], r["file"], r["line"]))
    touched = sorted({r["file"] for r in rows})
    return {
        "term": term, "found": True, "target": targets[0],
        "target_count": len(targets), "callers": rows, "count": len(rows),
        "depth": max((r["depth"] for r in rows), default=0),
        "files": touched, "file_count": len(touched), "truncated": truncated,
        "stats": graph.stats(),
    }


def path_between(files: dict[str, FileIndex], src: str, dst: str,
                 cap: int = 60000, root_key: str = "") -> dict[str, Any]:
    """Shortest call chain from `src` to `dst`."""
    graph = graph_for(files, root_key)
    a, b = _pick(graph, src), _pick(graph, dst)
    if not a or not b:
        return {"from": src, "to": dst, "found": False, "hops": [],
                "reason": "term not found", "ambiguous": {"from": len(a), "to": len(b)},
                "stats": graph.stats()}
    start, goal = a[0], b[0]
    chain = _bfs_forward(graph, start, goal, cap)
    note = ""
    if len(a) > 1 or len(b) > 1:
        # a name defined in several files resolves to one arbitrary instance;
        # say so instead of letting the answer look authoritative
        note = (f" (note: {src} matches {len(a)} definitions, {dst} matches "
                f"{len(b)}; used {start} -> {goal})")
    if chain is None:
        return {"from": src, "to": dst, "found": False, "hops": [],
                "reason": "no forward call path",
                "ambiguous": {"from": len(a), "to": len(b)}, "stats": graph.stats()}
    return {
        "from": src, "to": dst, "found": True,
        "hops": [graph.nodes.get(n, {"name": n, "file": n}) for n in chain],
        "hop_count": len(chain) - 1, "chain": chain, "note": note,
        "ambiguous": {"from": len(a), "to": len(b)}, "stats": graph.stats(),
    }


def dead_symbols(files: dict[str, FileIndex], limit: int = 200,
                 python_only: bool = True, root_key: str = "") -> list[dict[str, Any]]:
    """Callable definitions nobody calls and nobody reads.

    ``python_only`` (default) restricts the answer to Python, the one language
    the index parses exactly. The other 13 come from line-based regex rules, so
    a "dead" hit there is a guess and would be worse than no answer at all.
    """
    def _compute() -> list[dict[str, Any]]:
        graph = graph_for(files, root_key)
        read_anywhere: set[str] = set()
        for _path, fi in files.items():
            for name, _line, role in _iter_refs(fi):
                if role in _READ_ROLES:
                    read_anywhere.add(name)
        inn = graph.inn
        out: list[dict[str, Any]] = []
        for nid, meta in graph.nodes.items():
            if meta["kind"] not in ("function", "method", "constructor"):
                continue
            name = meta["name"]
            if name in _NOT_A_NAME:
                continue  # a keyword the regex indexer mistook for a def
            if name.startswith("__") or name.startswith("test_"):
                continue
            lang = meta.get("language")
            if python_only and lang != "python":
                continue
            segs = meta["file"].replace(os.sep, "/").split("/")
            if "test" in segs[:2] or name.endswith("_test"):
                continue
            if inn.get(nid) or name in read_anywhere:
                continue
            out.append({
                "node": nid, "name": name, "kind": meta["kind"],
                "file": meta["file"], "line": meta["line"],
                "container": meta["container"], "signature": meta["signature"],
                "confidence": "exact" if lang == "python" else "weak",
            })
        out.sort(key=lambda r: (r["file"], r["line"]))
        return out

    rows = _memoized(root_key, files, f"dead:{python_only}", _compute)
    return rows[:limit]


def import_cycles(files: dict[str, FileIndex], root_key: str = "") -> list[list[str]]:
    """Circular file-import groups (Tarjan over the module graph)."""
    def _compute() -> list[list[str]]:
        graph = graph_for(files, root_key)
        return _tarjan(list(graph.mod_out.keys()), graph.mod_out)

    return _memoized(root_key, files, "cycles", _compute)


def communities(files: dict[str, FileIndex], root_key: str = "") -> dict[str, list[str]]:
    """Group files by subsystem (package/dir), largest first.

    Package-level on purpose: connected components over the raw import web
    collapse into one blob for a project this coupled, which answers nothing.
    """
    def _compute() -> dict[str, list[str]]:
        groups: dict[str, list[str]] = defaultdict(list)
        for path in files:
            groups[package_of(path)].append(path)
        for v in groups.values():
            v.sort()
        return dict(sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])))

    return _memoized(root_key, files, "communities", _compute)


def coupling(files: dict[str, FileIndex], root_key: str = "") -> list[dict[str, Any]]:
    """Which subsystems depend on which, strongest coupling first."""
    def _compute() -> list[dict[str, Any]]:
        graph = graph_for(files, root_key)
        counts: dict[tuple[str, str], int] = {}
        for here, targets in graph.mod_out.items():
            src = package_of(here[1:])
            for there in targets:
                dst = package_of(there[1:])
                if src == dst:
                    continue
                counts[(src, dst)] = counts.get((src, dst), 0) + 1
        rows = [{"from": a, "to": b, "edges": n} for (a, b), n in counts.items()]
        rows.sort(key=lambda r: (-r["edges"], r["from"], r["to"]))
        return rows

    return _memoized(root_key, files, "coupling", _compute)


def summary(files: dict[str, FileIndex], root_key: str = "") -> dict[str, Any]:
    """Whole-graph health snapshot in one call."""
    graph = graph_for(files, root_key)
    groups = communities(files, root_key)
    return {
        **graph.stats(),
        "communities": len(groups),
        "largest_community": max((len(v) for v in groups.values()), default=0),
        "import_cycles": len(import_cycles(files, root_key)),
        "dead_python_symbols": len(dead_symbols(files, limit=10_000, root_key=root_key)),
    }

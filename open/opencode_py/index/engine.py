"""Symbol index engine: build/update a per-worktree index and answer queries.

Query verbs (first token decides what is asked; everything else is a
case-insensitive substring match against names):

    def/define/definition/where     -> the definition(s) of a name
    callers/calls/called/call       -> usage sites that *invoke* a name
    refs/references/uses/usages     -> every usage site
    deps/dependencies/depends       -> imports of a file (by path or module)
    imports/who-imports             -> files that import a module/file
    symbols/list                    -> every definition in a file

    impact/blast/what-breaks X      -> multi-hop blast radius of X
    path A to B                     -> shortest call chain from A to B
    dead/deadcode                   -> unreferenced Python definitions
    cycles/circular                 -> circular file-import groups
    communities/subsystems          -> which files group into which subsystem
    graphstats/stats                -> whole-graph health in one call

The first six are single-hop name lookups. The rest traverse the resolved
dependency graph in index/graph.py, so they can follow a relationship across
files and functions rather than reporting one file's contents at a time.

Bare names (no verb) return definitions first, then references.
"""

from __future__ import annotations

from ..util.log import swallow
import os
import threading
import time
from pathlib import Path

from .cache import SymbolCache
from .heuristic_indexer import index_file as heuristic_index_file
from .model import FileIndex, ImportRecord, Ref, Symbol
from .python_indexer import index_file as python_index_file

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

DEFAULT_IGNORE = {
    ".git", ".hg", ".svn", ".bzr",
    "node_modules", "__pycache__", ".pytest_cache", ".mypy_cache",
    "dist", "build", "target", "out", "_build", "vendor", "third_party",
    ".venv", "venv", "env", ".tox", ".gradle", ".idea", ".vscode", "Pods",
    ".cache", ".next", ".nuxt", ".coverage", "coverage", "elm-stuff", "deps",
}

# languages given a real/rule-based indexer
CODE_LANGS = {
    "python", "javascript", "c", "csharp", "java", "kotlin", "scala", "groovy",
    "go", "rust", "swift", "php", "lua", "ruby", "perl", "r", "sql", "bash",
    "fish", "generic",
}

# explicit extensions routed to the generic heuristic indexer
_GENERIC_EXTS = {
    ".dart", ".m", ".mm", ".tcl", ".pas", ".v", ".vh", ".vhd", ".asm", ".s",
    ".nix", ".lisp", ".clj", ".cljs", ".elm", ".erl", ".hrl", ".ex", ".exs",
    ".gleam", ".zig", ".odin", ".nim", ".hs", ".ml", ".fs", ".fsx", ".vb",
    ".awk", ".sed", ".ps1", ".bat", ".cmd", ".d", ".hcl", ".proto", ".thrift",
    ".jl", ".pony", ".smt2", ".cr", ".e", ".gnu", ".m4", ".sls",
}

# data/doc languages: parsed but produce no symbols (noise)
_DATA_LANGS = {"json", "markdown", "yaml", "toml", "xml", "html", "css", "text"}

MAX_FILE_BYTES = 1_500_000  # skip oversized (typically vendored) files
MAX_RESULTS = 60


def _is_binary(src_byte_prefix: bytes) -> bool:
    from ..tools.read import _is_binary_sample

    return _is_binary_sample(src_byte_prefix)


def _language_for(rel: str) -> str:
    from .model import language_for

    lang = language_for(rel)
    if lang:
        return lang
    lower = rel.lower()
    idx = lower.rfind(".")
    if idx >= 0 and lower[idx:] in _GENERIC_EXTS:
        return "generic"
    return ""


# ---------------------------------------------------------------------------
# index builder
# ---------------------------------------------------------------------------

class IndexEngine:
    """Holds one in-memory index per root, refreshed lazily per query."""

    def __init__(self) -> None:
        self._roots: dict[str, dict[str, FileIndex]] = {}
        self._lock = threading.Lock()
        # ponytail: per-root exact-name index (lower name -> refs). 46K-ref
        # scans become dict hits for exact/case-exact; prefix/substring
        # still scan. Rebuilt when the root map is swapped.
        self._ref_exact: dict[str, dict[str, list]] = {}
        # roots whose exact-name bucket is stale; built on demand (see
        # ref_bucket) because materialising 46K Refs costs ~3s on a phone
        self._ref_dirty: set[str] = set()
        # rel -> why it was not indexed (oversize / dotfile / no language).
        # Skipping silently made the index look complete when it was not, so
        # graphstats reports the tally instead.
        self.skipped: dict[str, str] = {}

    def ensure(
        self,
        root: Path,
        additional_ignores: list[str] | None = None,
        force: bool = False,
    ) -> dict[str, FileIndex]:
        """Return the current per-file index for `root`, rebuilding as needed."""
        root = root.resolve()
        with self._lock:
            cached = self._roots.get(str(root))
            ignores = set(DEFAULT_IGNORE) | set(additional_ignores or [])
            snapshot = dict(cached) if cached is not None else None
        # The os.walk + full parse runs OUTSIDE the lock: one big repo no
        # longer blocks every other root's query. The fresh map is swapped
        # back in atomically under the lock.
        if snapshot is not None and not force:
            # refresh incrementally against disk (per-file mtime/size
            # compare — only changed files are re-parsed; the walk itself
            # is the remaining cost and can't be skipped without risking
            # stale results right after an edit)
            fresh, changed = self._refresh(snapshot, root, ignores)
        else:
            fresh, _changed = self._walk_and_index(root, ignores, force)
            changed = _changed
        with self._lock:
            self._roots[str(root)] = fresh
            if not changed:
                # Unchanged refresh: keep the existing exact-name bucket
                # (rebuilding it walks all 46K refs + re-saves 4MB cache —
                # the 0.5s warm-query cost). Bucket rebuilds only when files
                # actually changed.
                return fresh
            # Building that bucket materialises 46K Ref objects, ~3s on a
            # phone — and impact/path/dead/communities/graphstats never read
            # it. Mark it stale and build it only when a refs/callers query
            # actually asks for it.
            self._ref_dirty.add(str(root))
            return fresh

    def ref_bucket(self, root: Path, files: dict[str, FileIndex]) -> dict[str, list]:
        """Exact-name -> refs map, built on first use after files change."""
        key = str(root.resolve())
        with self._lock:
            if key not in self._ref_dirty:
                got = self._ref_exact.get(key)
                if got is not None:
                    return got
        bucket: dict[str, list] = {}
        for fi in files.values():
            try:
                for r in fi.refs:
                    try:
                        bucket.setdefault(r.name, []).append(r)
                        ln = r.name.lower()
                        if ln != r.name:
                            bucket.setdefault(ln, []).append(r)
                    except Exception as _oc_exc:
                        swallow(_oc_exc, "index/engine.py:171")
                        continue
            except Exception as _oc_exc:
                swallow(_oc_exc, "index/engine.py:173")
                continue
        with self._lock:
            self._ref_exact[key] = bucket
            self._ref_dirty.discard(key)
        return bucket

    def _refresh(
        self, cached: dict[str, FileIndex], root: Path, ignores: set[str]
    ) -> tuple[dict[str, FileIndex], bool]:
        """Compare on-disk stamps to the cached ones; re-index only changes.

        Returns (fresh map, changed). Callers skip bucket rebuilds + cache
        re-saves when nothing changed.
        """
        new: dict[str, FileIndex] = {}
        changed = False
        seen = set()
        skipped: dict[str, str] = {}
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                d for d in dirnames
                if d not in ignores and not (d.startswith(".") and d not in (".", ".."))
            ]
            dirnames.sort()
            rel_dir = os.path.relpath(dirpath, root)
            is_root = rel_dir == "."
            for fn in sorted(filenames):
                if fn.startswith("."):
                    skipped[fn if is_root else os.path.join(rel_dir, fn)] = "dotfile"
                    continue
                rel = fn if is_root else os.path.join(rel_dir, fn)
                seen.add(rel)
                path = os.path.join(dirpath, fn)
                try:
                    st = os.stat(path)
                except OSError:
                    skipped[rel] = "unreadable"
                    continue
                if st.st_size > MAX_FILE_BYTES:
                    skipped[rel] = f"oversize ({st.st_size} > {MAX_FILE_BYTES} bytes)"
                    continue
                if not _language_for(rel):
                    skipped[rel] = "no language rule for extension"
                    continue
                if _language_for(rel) in _DATA_LANGS:
                    skipped[rel] = "data/doc language (yields no symbols)"
                    continue
                rec = cached.get(rel)
                if rec is not None and rec.size == st.st_size and rec.mtime == st.st_mtime_ns:
                    # Same size+mtime usually means unchanged — but same-size
                    # edits, coarse filesystems, and git checkouts can keep
                    # both stamps while changing content. Records that carry a
                    # content hash get a cheap 1KB head check; records without
                    # one (older caches) are re-parsed once, earning a hash.
                    if rec.content_hash and self._head_hash(path) == rec.content_hash:
                        new[rel] = rec
                        continue
                    if not rec.content_hash:
                        pass  # fall through to re-parse below
                fi = self._index_one(root, rel, path, st)
                if fi is not None:
                    new[rel] = fi
                    changed |= True
        if set(cached) != seen:
            # Only INDEXABLE files count: _index_one skips data langs
            # (markdown/toml/json...), dotfiles, oversize and binary — the
            # walk sees them but the cache never holds them. Comparing raw
            # `seen` re-saved the 4MB cache + rebuilt the 46K bucket on
            # EVERY query. ponytail: mirrors _index_one's skip rules; a new
            # skip rule must be added here too.
            try:
                indexable = set()
                for r in seen:
                    try:
                        lang = _language_for(str(r))
                        if not lang or lang in _DATA_LANGS or lang == "text":
                            continue
                        if str(r).startswith("."):
                            continue
                        indexable.add(r)
                    except Exception as _oc_exc:
                        swallow(_oc_exc, "index/engine.py:254")
                        continue
            except Exception:
                indexable = seen
            if set(cached) != indexable:
                changed |= True  # files deleted or renamed
        self.skipped = skipped
        if changed:
            SymbolCache(root).save(new)
        return new, changed

    @staticmethod
    def _head_hash(path: str) -> str:
        # Whole-file digest, not the first 4KB: an edit below byte 4096 that
        # preserved size and mtime used to be served from a stale cache.
        # Reading the file is cheap next to the graph build it protects.
        try:
            import hashlib as _hl

            h = _hl.sha1()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(65536), b""):
                    h.update(chunk)
            return h.hexdigest()[:16]
        except OSError:
            return ""

    def _walk_and_index(
        self, root: Path, ignores: set[str], force: bool
    ) -> tuple[dict[str, FileIndex], bool]:
        files: dict[str, FileIndex] = {}
        disk_file_map: dict[str, tuple[Path, os.stat_result]] = {}
        skipped: dict[str, str] = {}
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [
                d for d in dirnames
                if d not in ignores and not (d.startswith(".") and d not in (".", ".."))
            ]
            dirnames.sort()
            rel_dir = os.path.relpath(dirpath, root)
            is_root = rel_dir == "."
            for fn in sorted(filenames):
                if fn.startswith("."):
                    skipped[fn if is_root else os.path.join(rel_dir, fn)] = "dotfile"
                    continue
                rel = fn if is_root else os.path.join(rel_dir, fn)
                path = os.path.join(dirpath, fn)
                try:
                    st = os.stat(path)
                except OSError:
                    skipped[rel] = "unreadable"
                    continue
                if st.st_size > MAX_FILE_BYTES:
                    skipped[rel] = f"oversize ({st.st_size} > {MAX_FILE_BYTES} bytes)"
                    continue
                if not _language_for(rel):
                    skipped[rel] = "no language rule for extension"
                    continue
                if _language_for(rel) in _DATA_LANGS:
                    skipped[rel] = "data/doc language (yields no symbols)"
                    continue
                disk_file_map[rel] = (path, st)
        self.skipped = skipped

        # consult cache unless forcing a full rebuild
        cached: dict[str, FileIndex] = {}
        if not force:
            try:
                cached = SymbolCache(root).load() or {}
            except Exception:
                cached = {}
        changed = not force
        todo: list[tuple[str, str, os.stat_result]] = []
        for rel, (path, st) in disk_file_map.items():
            rec = cached.get(rel) if not force else None
            if not force and rec is not None and rec.size == st.st_size and rec.mtime == st.st_mtime_ns:
                if rec.content_hash and self._head_hash(path) == rec.content_hash:
                    changed |= False
                    files[rel] = rec
                    continue
                if rec.content_hash:
                    pass  # hash mismatch — re-parse below
                # no hash yet (older cache): re-parse once below
            todo.append((rel, path, st))
        # Parsing is CPU-bound pure-Python (ast walk), so threads would only
        # fight the GIL. Processes spread it across cores: a full rebuild of
        # 177 files took 9.7s serially and ~2s across 6 workers on an 8-core
        # phone. Below the threshold the pool costs more than it saves.
        for rel, fi in self._parse_many(root, todo):
            if fi is not None:
                files[rel] = fi
                changed |= True
        if cached and set(cached) != set(disk_file_map):
            changed |= True
        if changed:
            SymbolCache(root).save(files)
        return files, changed

    def _parse_many(self, root: Path, todo: list[tuple[str, str, "os.stat_result"]]) -> list[tuple[str, "FileIndex | None"]]:
        """Index `todo` files, in parallel when it is worth the process pool.

        Degrades to a plain serial loop on any failure: a sandbox with no
        usable /dev/shm must still be able to index, just slower.
        """
        if not todo:
            return []
        workers = min(len(todo), (os.cpu_count() or 1))
        if workers < 2 or len(todo) < 8:
            return [(rel, self._index_one(root, rel, path, st))
                    for rel, path, st in todo]
        try:
            from concurrent.futures import ProcessPoolExecutor

            payload = [(str(root), rel, path, st.st_mtime_ns, st.st_size)
                       for rel, path, st in todo]
            # longest file first: app.py alone costs ~1s, so starting it late
            # would leave every other worker idle waiting on the tail
            payload.sort(key=lambda t: -t[4])
            out: list[tuple[str, "FileIndex | None"]] = []
            with ProcessPoolExecutor(max_workers=workers) as pool:
                for rel, fi in pool.map(_index_worker, payload, chunksize=4):
                    out.append((rel, fi))
            return out
        except Exception:
            return [(rel, self._index_one(root, rel, path, st))
                    for rel, path, st in todo]

    def _index_one(
        self, root: Path, rel: str, path: str, st: os.stat_result
    ) -> FileIndex | None:
        lang = _language_for(rel)
        if not lang or lang in _DATA_LANGS or lang == "text":
            return None
        try:
            with open(path, "rb") as fh:
                head = fh.read(1024)
                if _is_binary(head):
                    return None
                fh.seek(0)
                data = fh.read()
        except OSError:
            return None
        src = data.decode("utf-8", errors="replace")
        mtime = st.st_mtime_ns
        size = st.st_size
        try:
            import hashlib as _hl

            ch = _hl.sha1(data).hexdigest()[:16]
        except Exception:
            ch = ""
        if lang == "python":
            fi = python_index_file(root, rel, src, size, mtime)
        else:
            fi = heuristic_index_file(root, rel, src, size, mtime, lang)
        if fi is not None:
            fi.content_hash = ch
        return fi


def _index_worker(task):
    """Index one file in a worker process.

    Module level (not a method) so ProcessPoolExecutor can pickle it, and it
    takes only primitives so the payload stays small. `_index_one` touches no
    instance state, so calling it through the inherited singleton is enough.
    """
    root_str, rel, path, _mtime, _size = task
    try:
        st = os.stat(path)
    except OSError:
        return rel, None
    try:
        return rel, _ENGINE._index_one(Path(root_str), rel, path, st)
    except Exception:
        return rel, None


_ENGINE = IndexEngine()


# ---------------------------------------------------------------------------
# queries
# ---------------------------------------------------------------------------

_VERB_DEF = ("def", "define", "definition", "definitions", "where", "what is", "what's")
_VERB_CALL = ("callers", "calls", "called", "call", "usages", "usage", "who calls", "who invokes")
_VERB_REF = ("refs", "references", "uses")
_VERB_DEPS = ("deps", "dependencies", "depends", "depend", "imports of", "what does")
_VERB_IMPORT = ("imports", "who imports", "used by")
_VERB_FILE = ("symbols", "list", "contents", "in file")
# multi-hop graph queries (see index/graph.py)
_VERB_IMPACT = ("impact", "blast", "blast radius", "dependents", "dependents of",
                "what breaks", "who breaks", "affected by", "downstream")
_VERB_PATH = ("path", "call path", "trace", "chain", "how does", "how do")
_VERB_DEAD = ("dead", "deadcode", "dead code", "unused", "unreachable", "orphans")
_VERB_CYCLES = ("cycles", "circular", "circular imports", "import cycles", "loops")
_VERB_GRAPHSTATS = ("graphstats", "graph stats", "graph summary", "stats")
_VERB_COMM = ("communities", "subsystems", "graph", "structure", "layers")


def _strip(q: str) -> str:
    return q.strip().strip("?:!.").strip()


# graph modes that take no search term — matched before the verb loop so
# `graphstats` is not misread as `graph` + term "stats"
_BARE_MODES = {
    "graphstats": "graphstats", "graph stats": "graphstats",
    "graph summary": "graphstats", "stats": "graphstats", "graph": "graphstats",
    "cycles": "cycles", "circular": "cycles", "circular imports": "cycles",
    "import cycles": "cycles",
    "communities": "communities", "subsystems": "communities",
    "structure": "communities", "layers": "communities",
    "dead": "dead", "deadcode": "dead", "dead code": "dead",
    "unused": "dead", "orphans": "dead",
}


def _parse_query(q: str) -> tuple[str, str]:
    """Return (mode, term). term is the rest-of-query name/path."""
    q0 = _strip(q)
    bare = _BARE_MODES.get(q0.lower())
    if bare is not None:
        return bare, "*"
    low = " " + q0.lower() + " "
    for verb, mode in (
        (_VERB_GRAPHSTATS, "graphstats"),
        (_VERB_IMPACT, "impact"),
        (_VERB_PATH, "path"),
        (_VERB_DEAD, "dead"),
        (_VERB_CYCLES, "cycles"),
        (_VERB_COMM, "communities"),
        (_VERB_FILE, "symbols"),
        (_VERB_DEPS, "deps"),
        (_VERB_IMPORT, "imports"),
        (_VERB_DEF, "def"),
        (_VERB_CALL, "callers"),
        (_VERB_REF, "refs"),
    ):
        for v in sorted(verb, key=len, reverse=True):
            start = low.find(" " + v)
            if start >= 0:
                end = start + 1 + len(v)
                term = q0[end:].strip().strip("?\"'`")
                if term:
                    return mode, term
    # "who imports X"
    term = q0
    for stop in (" in ", " of "):
        i = low.find(stop)
        if i > 0:
            pre = q0[:i]
            if pre in ("imports", "what", "who", "which files"):
                return "imports", q0[i + len(stop):].strip()
            if pre in ("deps", "dependencies", "depends"):
                return "deps", q0[i + len(stop):].strip()
    return "def", q0


def _match_rank(name: str, term: str) -> int:
    """0 = exact, 1 = case-insensitive exact, 2 = startswith, 3 = substring, -1 none."""
    if name == term:
        return 0
    ln, lt = name.lower(), term.lower()
    if ln == lt:
        return 1
    if ln.startswith(lt):
        return 2
    if lt in ln:
        return 3
    return -1


def _resolve_file(root: Path, term: str) -> str | None:
    """Resolve a user-supplied path/module/filename to an index 'path' key."""
    t = term.strip()
    if not t:
        return None
    p = Path(t)
    if p.is_absolute():
        try:
            return str(p.relative_to(root))
        except ValueError:
            return None
    candidate = root / p
    if candidate.exists():
        rel = p.as_posix()
        return rel
    # dotted module name -> possible files
    dotted = t.replace("/", ".").lstrip(".")
    for suffix in (".py", "/__init__.py", ""):
        mid = dotted.replace(".", "/")
        if suffix.startswith("/"):
            cand = mid + suffix
        else:
            cand = (mid or t) + suffix
        if (root / cand.lstrip("/")).exists():
            return cand.lstrip("/")
    # fuzzy basename match
    base = os.path.basename(t.rstrip("/"))
    return base or None


def _query_defs(files: dict[str, FileIndex], term: str, kind: str) -> list[Symbol]:
    out: list[Symbol] = []
    for fi in files.values():
        for s in fi.symbols:
            rank = _match_rank(s.name, term)
            if rank < 0:
                continue
            if kind and s.kind != kind:
                # allow kind=fn to also match methods for callers ergonomics
                if not (kind in ("fn", "function") and s.kind in ("function", "method")):
                    continue
            out.append(s)
    # Rank by match quality FIRST. Sorting by (file, line) let substring noise
    # fill the result limit and push the exact match out: asking `def target`
    # with 70 `zz_*_target` decoys returned 60 wrong hits and omitted `target`
    # itself. rank() is already computed above; order by it, then document
    # order within a rank so results stay stable and readable.
    out.sort(key=lambda s: (_match_rank(s.name, term), s.file, s.line))
    return out


def _query_refs(files: dict[str, FileIndex], term: str, only_calls: bool) -> list[Ref]:
    # Fast path: exact/case-exact hits via the per-root name index when the
    # caller passes it (engine wires it below); otherwise fall back to scan.
    # ponytail: index lives on the engine, not the file map — no new class.
    out: list[Ref] = []
    for fi in files.values():
        for r in fi.refs:
            rank = _match_rank(r.name, term)
            if rank < 0:
                continue
            if only_calls and r.role != "call":
                continue
            out.append(r)
    out.sort(key=lambda r: (r.file, r.line))
    return out


def _query_refs_indexed(exact: dict[str, list] | None, files: dict[str, FileIndex], term: str, only_calls: bool) -> list[Ref]:
    """Same as _query_refs but exact terms hit the name index (no 46K scan)."""
    try:
        if exact:
            lt = term.lower()
            if term in exact or lt in exact:
                out: list[Ref] = []
                seen: set[int] = set()
                for key in (term, lt) if term != lt else (term,):
                    for r in exact.get(key, ()):  # type: ignore[union-attr]
                        try:
                            if id(r) in seen:
                                continue
                            seen.add(id(r))
                            if only_calls and r.role != "call":
                                continue
                            # exact dict may hold lowercase bucket: verify rank
                            if _match_rank(r.name, term) < 0:
                                continue
                            out.append(r)
                        except Exception as _oc_exc:
                            swallow(_oc_exc, "index/engine.py:616")
                            continue
                # exact hit: still need substring/startswith cousins — scan
                # ONLY when the exact bucket was empty (rare term).
                if out:
                    out.sort(key=lambda r: (r.file, r.line))
                    return out
    except Exception as _oc_exc:
        swallow(_oc_exc, "index/engine.py:623")
        pass
    return _query_refs(files, term, only_calls)


# ---------------------------------------------------------------------------
# formatting
# ---------------------------------------------------------------------------

def _fmt_def(s: Symbol, root: Path) -> str:
    try:
        rel = Path(s.file).as_posix()
    except Exception:
        rel = str(s.file)
    where = s.container + "." if s.container else ""
    sig = s.signature or s.name
    end = getattr(s, "end_line", 0) or s.line
    span = f"{s.line}-{end}" if end != s.line else f"{s.line}"
    return f"{rel}:{span}\n  {sig}  ({s.kind} in {where or 'module'}) — read symbol='{s.name}'"


def _run_query(files: dict[str, FileIndex], mode: str, term: str,
               kind: str, limit: int, root: Path) -> dict:
    matches: list = []
    if mode == "def":
        matches = _query_defs(files, term, kind)
        title = f"Definition{'' if len(matches) == 1 else 's'} of `{term}`"
        if matches:
            # a bare name defined in N places is not a single answer; say so
            # instead of letting a flat list read as one authoritative hit
            exact = [m for m in matches if m.name == term]
            where = sorted({m.file for m in exact}) or sorted({m.file for m in matches})
            if len(where) > 1:
                title += (f"  [AMBIGUOUS: defined in {len(where)} files — "
                          f"{', '.join(Path(w).name for w in where[:5])}"
                          f"{'...' if len(where) > 5 else ''}]")
    elif mode in ("callers", "refs"):
        try:
            exact = _ENGINE.ref_bucket(root, files)
        except Exception:
            exact = None
        matches = _query_refs_indexed(exact, files, term, only_calls=(mode == "callers"))
        title = f"`{term}` is referenced {len(matches)} time"
        title += "" if len(matches) == 1 else "s"
    elif mode == "deps":
        return _query_deps(files, term, root, limit)
    elif mode == "imports":
        return _query_imports(files, term, root, limit)
    elif mode == "symbols":
        return _query_symbols(files, term, root, limit)
    elif mode == "impact":
        return _query_impact(files, term, root, limit)
    elif mode == "path":
        return _query_path(files, term, root, limit)
    elif mode == "dead":
        return _query_dead(files, term, root, limit)
    elif mode == "cycles":
        return _query_cycles(files, term, root, limit)
    elif mode == "communities":
        return _query_communities(files, term, root, limit)
    elif mode == "graphstats":
        return _query_graphstats(files, term, root, limit)
    else:
        return {"output": f"Unhandled query mode: {mode}", "error": True}

    if not matches:
        return {
            "output": f"No {mode} found for `{term}`.",
            "metadata": {"term": term, "mode": mode, "matches": 0, "items": []},
        }

    shown = matches[:limit]
    if mode == "def":
        lines = [title]
        last = None
        for s in shown:
            rel = s.file
            if rel != last:
                lines.append(f"{rel}:")
                last = rel
            lines.append(_fmt_def(s, root))
    else:
        lines = [title]
        last = None
        for r in shown:
            if r.file != last:
                lines.append(r.file + ":")
                last = r.file
            role = r.role if r.role != "use" else ""
            lines.append(f"  line {r.line}" + (f"  ({role})" if role else ""))
    if len(matches) > limit:
        lines.append(f"... and {len(matches) - limit} more (limit={limit})")

    structured = {"term": term, "mode": mode, "matches": len(matches), "items": [
        {"file": s.file, "line": s.line, "end_line": getattr(s, "end_line", 0) or s.line,
         "name": s.name, "kind": s.kind,
         "signature": s.signature, "container": s.container} for s in shown[:limit]
    ]} if mode == "def" else {
        "term": term, "mode": mode, "matches": len(matches), "items": [
            {"file": r.file, "line": r.line, "name": r.name, "role": r.role}
            for r in shown[:limit]
        ]
    }
    return {"output": "\n".join(lines), "metadata": structured}


def _query_deps(files: dict[str, FileIndex], term: str, root: Path, limit: int) -> dict:
    rel = _resolve_file(root, term)
    if rel is None:
        return {"output": f"Could not resolve `{term}` to a file.", "error": True}
    # exact or fuzzy file key
    key = rel
    fi = files.get(key)
    if fi is None:
        for k in files:
            if k == rel or k.endswith(rel) or os.path.basename(k) == rel:
                fi = files.get(k)
                rel = k
                break
    if fi is None or fi.language not in CODE_LANGS:
        return {"output": f"No indexed file matches `{term}`.", "error": True}
    local: list[ImportRecord] = []
    external: list[ImportRecord] = []
    for imp in fi.imports:
        (local if imp.local else external).append(imp)
    lines = [f"Dependencies of {rel}:", ""]
    lines.append(f"\u2022 Local ({len(local)}):")
    for imp in sorted(local, key=lambda x: x.module):
        target = _module_to_file(imp.module)
        if target:
            lines.append(f"  {imp.module}  ->  {target}")
        else:
            lines.append(f"  {imp.module}")
    lines.append(f"\u2022 External / stdlib ({len(external)}):")
    for imp in sorted(external, key=lambda x: x.module):
        lines.append(f"  {imp.module}")
    return {
        "output": "\n".join(lines),
        "metadata": {"file": rel, "local": [i.module for i in local],
                     "external": [i.module for i in external]},
    }


def _module_to_file(module: str) -> str | None:
    """Best-effort module name -> relative file path (for local deps)."""
    m = module.replace(".", "/").lstrip("/")
    if not m:
        return None
    return f"{m}.py"


def _query_imports(files: dict[str, FileIndex], term: str, root: Path, limit: int) -> dict:
    results: list[tuple[str, str, int]] = []  # (file, module, line)
    lowered = term.lower().rstrip("/")
    for fi in files.values():
        for imp in fi.imports:
            if imp.module.lower() == lowered or imp.module.lower().endswith("." + lowered):
                results.append((fi.path, imp.module, imp.line))
    results.sort()
    if not results:
        return {"output": f"No files import `{term}`.", "metadata": {"term": term, "items": []}}
    shown = results[:limit]
    lines = [f"Files importing `{term}` ({len(results)}):"]
    last = None
    for file, module, line in shown:
        if file != last:
            lines.append(f"{file}:")
            last = file
        lines.append(f"  line {line}: import {module}")
    if len(results) > limit:
        lines.append(f"... and {len(results) - limit} more (limit={limit})")
    return {
        "output": "\n".join(lines),
        "metadata": {"term": term, "items": [
            {"file": f, "line": ln, "module": m} for f, m, ln in shown]},
    }


def _query_symbols(files: dict[str, FileIndex], term: str, root: Path, limit: int) -> dict:
    rel = _resolve_file(root, term)
    if rel is None:
        return {"output": f"Could not resolve `{term}` to a file.", "error": True}
    fi = files.get(rel)
    if fi is None:
        for k in files:
            if k == rel or os.path.basename(k) == rel:
                fi = files.get(k)
                rel = k
                break
    if fi is None:
        return {"output": f"No indexed file matches `{term}`.", "error": True}
    syms = sorted(fi.symbols, key=lambda s: s.line)
    if not syms:
        return {"output": f"No definitions found in {rel}.", "metadata": {"file": rel, "items": []}}
    shown = syms[:limit]
    lines = [f"Definitions in {rel}:"]
    for s in shown:
        lines.append(_fmt_def(s, root))
    lines.append("Tip: read symbol='<name>' returns the block directly — no offset math.")
    if len(syms) > limit:
        lines.append(f"... and {len(syms) - limit} more (limit={limit})")
    return {
        "output": "\n".join(lines),
        "metadata": {"file": rel, "items": [
            {"name": s.name, "kind": s.kind, "line": s.line,
             "signature": s.signature, "container": s.container} for s in shown]},
    }


# ---------------------------------------------------------------------------
# multi-hop graph queries (index/graph.py)
# ---------------------------------------------------------------------------

def _graph_stats_line(stats: dict) -> str:
    return (f"Graph: {stats['nodes']} nodes, {stats['call_edges']} call edges "
            f"({stats['exact_edges']} exact / {stats['likely_edges']} likely / "
            f"{stats['weak_edges']} weak), {stats['import_edges']} import edges, "
            f"{stats['dropped_ambiguous']} ambiguous dropped, built in {stats['build_ms']}ms")


def _query_impact(files: dict[str, FileIndex], term: str, root: Path, limit: int) -> dict:
    """Blast radius: everything that transitively depends on `term`."""
    from .graph import impact_of

    res = impact_of(files, term, root_key=str(root))
    if not res["found"]:
        return {"output": f"No indexed definition matches `{term}`, so nothing "
                         f"can depend on it.", "metadata": {"term": term, "found": False}}
    shown = res["callers"][:limit]
    noun = "symbol" if res["count"] == 1 else "symbols"
    files_n = "file" if res["file_count"] == 1 else "files"
    lines = [f"Impact of `{res['term']}` — {res['count']} dependent {noun} across "
             f"{res['file_count']} {files_n} (max depth {res['depth']})", ""]
    if res.get("truncated"):
        lines.insert(1, f"  TRUNCATED: traversal cap reached at {limit}. This is a "
                        f"FLOOR, not the real total. Re-run with a higher limit.")
    if res["target_count"] > 1:
        lines.insert(1, f"  Note: `{term}` is defined {res['target_count']} times; "
                        f"impact counts everything that reaches any of them.")
    last = None
    for c in shown:
        if c["file"] != last:
            lines.append(f"{c['file']}:")
            last = c["file"]
        who = f"{c['container']}.{c['name']}" if c["container"] else c["name"]
        lines.append(f"  depth {c['depth']}  line {c['line']}  {who}  ({c['kind']})")
    if res["count"] > len(shown):
        lines.append(f"... and {res['count'] - len(shown)} more (limit={limit})")
    if res["file_count"]:
        lines.append("")
        lines.append("Files that would be affected:")
        for f in res["files"]:
            lines.append(f"  {f}")
    lines.append("")
    lines.append(_graph_stats_line(res["stats"]))
    return {"output": "\n".join(lines), "metadata": {
        "term": res["term"], "found": True, "count": res["count"],
        "depth": res["depth"], "files": res["files"],
        "truncated": res.get("truncated", False),
        "target_count": res["target_count"],
        "items": [{"file": c["file"], "line": c["line"], "name": c["name"],
                   "container": c["container"], "depth": c["depth"],
                   "kind": c["kind"]} for c in shown],
    }}


def _query_path(files: dict[str, FileIndex], term: str, root: Path, limit: int) -> dict:
    """Shortest call chain `A to B`."""
    from .graph import path_between

    parts = [p for p in term.replace("->", " to ").split(" to ") if p.strip()]
    if len(parts) < 2:
        return {"output": "Give two names: `path <name> to <name>` "
                         "(e.g. `path run_tool to send`).", "error": True}
    src, dst = parts[0].strip(), parts[1].strip()
    res = path_between(files, src, dst, root_key=str(root))
    if not res["found"]:
        amb = res.get("ambiguous") or {}
        # a name that resolves to several definitions may simply have picked the
        # wrong one, so the ambiguity has to be visible even when nothing was found
        caveat = ""
        if amb.get("from", 0) > 1 or amb.get("to", 0) > 1:
            caveat = (f" Note: {src} matches {amb.get('from', 0)} definitions, "
                      f"{dst} matches {amb.get('to', 0)}; one of each was used, so "
                      f"a path may exist between other instances.")
        return {"output": f"No call path from `{src}` to `{dst}`"
                         + (f" ({res.get('reason')})." if res.get("reason") else ".")
                         + caveat,
                "metadata": {"from": src, "to": dst, "found": False,
                             "reason": res.get("reason", ""),
                             "ambiguous": amb}}
    lines = [f"Call path `{src}` -> `{dst}` ({res['hop_count']} hops):", ""]
    if res.get("note"):
        lines.append(res["note"])
        lines.append("")
    for i, hop in enumerate(res["hops"], 1):
        sig = hop.get("signature") or hop.get("name", "?")
        lines.append(f"  {i}. {hop.get('file', '?')}:{hop.get('line', 0)}  {sig}")
    lines.append("")
    lines.append(_graph_stats_line(res["stats"]))
    return {"output": "\n".join(lines), "metadata": {
        "from": src, "to": dst, "found": True, "hop_count": res["hop_count"],
        "chain": res["chain"], "ambiguous": res.get("ambiguous", {}),
        "hops": [{"file": h.get("file", ""), "line": h.get("line", 0),
                  "name": h.get("name", ""),
                  "container": h.get("container", "")} for h in res["hops"]],
    }}


def _query_dead(files: dict[str, FileIndex], term: str, root: Path, limit: int) -> dict:
    """Definitions nobody calls and nobody reads (Python only — it is exact).

    `dead all` (or `dead all-languages`) opts into the other 13 languages.
    Those come from line-based regex rules, so hits there are guesses, and the
    output says so rather than presenting them as findings.
    """
    from .graph import dead_symbols

    # NOTE: bare `dead` arrives with term "*" from _BARE_MODES, so "*" must NOT
    # be treated as an opt-in or the default silently becomes every language
    wide = term.strip().lower() in ("all", "all-languages", "all languages")
    # ask for everything, then show a slice: counting the displayed slice made
    # this report "60 definitions" when the real total was 454
    every = dead_symbols(files, limit=1_000_000, python_only=not wide, root_key=str(root))
    total = len(every)
    rows = every[:limit]
    if not rows:
        scope = "any language" if wide else "Python"
        return {"output": f"No unreferenced {scope} definitions found.",
                "metadata": {"items": [], "all_languages": wide, "count": 0}}
    exact = [r for r in every if r.get("confidence") == "exact"]
    weak = [r for r in every if r.get("confidence") != "exact"]
    scope = "any language" if wide else "Python"
    lines = [f"Possibly dead — {total} {scope} definition"
             f"{'' if total == 1 else 's'} with no caller and no reader:", ""]
    if weak:
        lines += [f"  {len(exact)} verified with ast (exact). "
                  f"{len(weak)} are regex-based guesses on other languages —",
                  "  treat those as leads only, they are NOT confirmed.", ""]
    else:
        lines += ["Verified with ast, so these are real unreferenced defs, not",
                  "regex guesses. Entry points and __dunder__ are excluded already;",
                  "confirm before deleting (a plugin or test may load them by name).",
                  ""]
    last = None
    for r in rows:
        if r["file"] != last:
            lines.append(f"{r['file']}:")
            last = r["file"]
        who = f"{r['container']}.{r['name']}" if r["container"] else r["name"]
        mark = "" if r.get("confidence") == "exact" else "  [weak: regex-derived]"
        lines.append(f"  line {r['line']}  {who}  ({r['kind']}){mark}")
    if total > len(rows):
        lines.append(f"... and {total - len(rows)} more (showing {len(rows)}, "
                     f"raise `limit` to see them all)")
    return {"output": "\n".join(lines), "metadata": {
        "items": rows, "all_languages": wide, "count": total,
        "shown": len(rows), "exact": len(exact), "weak": len(weak)}}


def _query_cycles(files: dict[str, FileIndex], term: str, root: Path, limit: int) -> dict:
    """Circular file-import groups."""
    from .graph import import_cycles

    cycles = import_cycles(files, root_key=str(root))
    if not cycles:
        return {"output": "No circular file imports found.", "metadata": {"cycles": []}}
    lines = [f"{len(cycles)} circular import group"
             f"{'' if len(cycles) == 1 else 's'}:", ""]
    shown = cycles[:limit]
    for group in shown:
        lines.append(f"  group of {len(group)}:")
        for f in group[:20]:
            lines.append(f"    {f}")
        if len(group) > 20:
            lines.append(f"    ... and {len(group) - 20} more")
        lines.append("")
    lines.append("A cycle can surface as a partially-initialised module at import "
                 "time. Large groups are usually a sign a layer boundary leaked.")
    return {"output": "\n".join(lines), "metadata": {
        "count": len(cycles),
        "groups": [[f.lstrip("@") for f in g] for g in shown],
    }}


def _query_communities(files: dict[str, FileIndex], term: str, root: Path, limit: int) -> dict:
    """Subsystems and how strongly they depend on each other."""
    from .graph import communities, coupling

    key = str(root)
    groups = communities(files, root_key=key)
    links = coupling(files, root_key=key)
    lines = [f"Project subsystems ({len(groups)}), largest first:", ""]
    for name, members in list(groups.items())[:limit]:
        lines.append(f"  {name:34} {len(members):>3} files")
    lines.append("")
    lines.append("Strongest coupling (who depends on whom):")
    for link in links[:12]:
        lines.append(f"  {link['edges']:>3} edges   {link['from']}  ->  {link['to']}")
    return {"output": "\n".join(lines), "metadata": {
        "communities": {k: len(v) for k, v in groups.items()}, "coupling": links[:12]}}


def _query_graphstats(files: dict[str, FileIndex], term: str, root: Path, limit: int) -> dict:
    """One-call health snapshot of the whole graph."""
    from .graph import summary

    res = summary(files, root_key=str(root))
    lines = [
        "Knowledge graph status:", "",
        f"  indexed files      : {len(files)}",
        f"  graph nodes        : {res['nodes']}",
        f"  call edges         : {res['call_edges']} "
        f"({res['exact_edges']} exact, {res['likely_edges']} likely, {res['weak_edges']} weak)",
        f"  import edges       : {res['import_edges']}",
        f"  ambiguous dropped  : {res['dropped_ambiguous']} "
        f"(named more than once — left unlinked rather than guessed)",
        f"  subsystems         : {res['communities']} (largest {res['largest_community']} files)",
        f"  import cycles      : {res['import_cycles']}",
        f"  unreferenced python: {res['dead_python_symbols']}",
        f"  build time         : {res['build_ms']} ms", "",
        "exact  = call site and definition in the same file (ast fact)",
        "likely = resolved by the caller's own imports",
        "weak   = matched by unique name only; 14 languages are line-based, so",
        "         only Python edges are as trustworthy as `def` results.",
    ]
    skipped = getattr(_ENGINE, "skipped", None) or {}
    if skipped:
        by_reason: dict[str, list[str]] = {}
        for rel, why in skipped.items():
            by_reason.setdefault(why, []).append(rel)
        lines.insert(-6, f"  skipped files      : {len(skipped)}")
        lines.append("")
        lines.append("NOT indexed (the index is not the whole tree):")
        for why, rels in sorted(by_reason.items(), key=lambda kv: -len(kv[1])):
            sample = ", ".join(sorted(rels)[:3])
            more = f" (+{len(rels) - 3} more)" if len(rels) > 3 else ""
            lines.append(f"  {len(rels):>4}  {why} — e.g. {sample}{more}")
    return {"output": "\n".join(lines), "metadata": res}


def query(
    q: str,
    root: Path | str | None = None,
    kind: str = "",
    limit: int = MAX_RESULTS,
    ignore_extra: list[str] | None = None,
    force: bool = False,
) -> dict:
    """Run a query against the (lazily built) symbol index for `root`."""
    from ..globals import resolve_worktree

    start = time.monotonic()
    root_path = Path(root) if root else Path.cwd()
    root_path = resolve_worktree(root_path)
    mode, term = _parse_query(q)
    if not term:
        return {"output": f"Empty query. Try e.g. `def {term or 'main'}`, `callers {term or 'x'}`, `deps {term or 'file.py'}`, `symbols {term or 'file.py'}`.", "error": True}
    files = _ENGINE.ensure(root_path, ignore_extra, force)
    elapsed_ms = (time.monotonic() - start) * 1000
    kind = (kind or "").strip().lower()
    result = _run_query(files, mode, term, kind, int(limit or MAX_RESULTS), root_path)
    result.setdefault("metadata", {})["indexed_files"] = len(files)
    result.setdefault("metadata", {})["elapsed_ms"] = round(elapsed_ms, 1)
    return result
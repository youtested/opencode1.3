"""remember tool: notes that survive across sessions.

RULES (full guidance for maintainers; the sent schema is short):
- Save project rules/quirks/recurring fixes when the user says remember/note
  this — or when you find a stable fact worth keeping. Short + reusable.
- Notes tag the current project and auto-load into future prompts.
- add (needs text) / list / delete (by id) / clear. Never raises.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

from ..globals import Path as GPath
from .registry import Tool, schema_with

MAX_ENTRIES = 300
MAX_TEXT = 2000
# How many notes ride along in one prompt. Notes are scored against the user's
# message, so a small K is enough — the point is to stop paying context for
# notes that have nothing to do with the current turn.
INJECT_LIMIT = 12
KINDS = ("rule", "fact", "decision", "fix")

_LOCK = threading.Lock()

_ACTIONS = ("add", "list", "review", "delete", "clear")

# Retrieval is lexical (BM25 over short notes): an armv7 phone cannot afford a
# real embedding model, and notes are a few words each, so term overlap with an
# idf weight separates them fine.
_STOP = frozenset(
    """a an and are as at be but by do does for from had has have if in into is
    it its of on or so than that the their then there these they this to was
    were will with you your not no can just am""".split()
)
_TOKEN_RE = re.compile(r"[a-z0-9_]+")

_K1 = 1.2
_B = 0.6
_RECENT_WEIGHT = 0.30
_HIT_WEIGHT = 0.20
_PROJECT_BONUS = 0.25
_PINNED_BONUS = 1000.0
_DAY = 86400.0
_HALF_LIFE_DAYS = 30.0

# note id -> times it was injected. Held in memory and folded into the file on
# the next write: scoring must never write, because that bumps memory.json's
# mtime and would invalidate the whole system-prompt cache on every turn.
_HITS: dict[Any, int] = {}
# note id -> times the agent's own reply actually reflected it. Merged into the
# file on the next write, same as _HITS.
_USED: dict[Any, int] = {}

# Bumped by every write, so the corpus cache below invalidates even when a
# rewrite leaves mtime_ns and size looking unchanged.
_GEN = 0


def _stem(tok: str) -> str:
    """Crude suffix folding so a query word still matches the note's inflection:
    deploy/deploys, theme/themes, note/notes. Not linguistically correct, just
    enough that plain term overlap is not defeated by plurals."""
    if len(tok) >= 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) >= 6 and tok.endswith("ing"):
        return tok[:-3]
    if len(tok) >= 5 and tok.endswith("ed"):
        return tok[:-2]
    if len(tok) >= 5 and tok.endswith("es"):
        return tok[:-2]
    if len(tok) >= 3 and tok.endswith("s") and not tok.endswith("ss"):
        return tok[:-1]
    return tok


def _tokens(text: str) -> list[str]:
    """Lowercase tokens; underscores split identifiers, stopwords/digits dropped,
    plurals/gerunds folded to a common stem."""
    out: list[str] = []
    for raw in _TOKEN_RE.findall((text or "").lower()):
        for part in raw.split("_"):
            if len(part) >= 2 and not part.isdigit() and part not in _STOP:
                out.append(_stem(part))
    return out


def _hits_of(entry: dict) -> int:
    try:
        return max(int(entry.get("hits") or 0), _HITS.get(entry.get("id"), 0))
    except (TypeError, ValueError):
        return 0


def _recency(entry: dict, now: float) -> float:
    try:
        age = max(0.0, now - float(entry.get("created") or 0.0))
    except (TypeError, ValueError):
        age = 0.0
    return 1.0 / (1.0 + age / _DAY / _HALF_LIFE_DAYS)


def _bonus(entry: dict, now: float) -> float:
    """Everything except term overlap: recency, reuse, specificity, pinned."""
    if entry.get("superseded"):
        return -1.0  # replaced facts are the first thing to go, and never injected
    score = _RECENT_WEIGHT * _recency(entry, now)
    score += _HIT_WEIGHT * min(_hits_of(entry), 10) / 10.0
    if entry.get("project"):
        score += _PROJECT_BONUS
    if entry.get("pinned"):
        score += _PINNED_BONUS
    return score


def _rank_docs(
    docs: list[tuple[list[str], dict, int, dict]],
    df: dict[str, int],
    n: int,
    avg_len: float,
    terms: list[str],
    now: float,
) -> list[dict]:
    """Score pre-tokenized (tokens, term-freqs, length, entry) docs against the
    query terms, best-first. tf/length are precomputed by the corpus cache, so
    a turn only pays for the term lookups."""
    if not terms:
        return [
            e
            for _t, _tf, _l, e in sorted(
                docs,
                key=lambda d: (_bonus(d[3], now), float(d[3].get("created") or 0.0)),
                reverse=True,
            )
        ]
    scored: list[tuple[float, float, dict]] = []
    for _toks, tf, length, e in docs:
        overlap = 0.0
        for t in terms:
            freq = tf.get(t, 0)
            if not freq:
                continue
            idf = math.log(1.0 + n / (1.0 + df.get(t, 0)))
            denom = freq + _K1 * (1.0 - _B + _B * length / avg_len)
            overlap += idf * (freq * (_K1 + 1.0)) / denom
        scored.append((overlap * 2.0 + _bonus(e, now), float(e.get("created") or 0.0), e))
    # the key never touches the dict in slot 2, so ties can't compare dicts
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return [e for _s, _c, e in scored]


def _build_corpus(entries: list[dict]) -> tuple[list, dict, int, float]:
    docs: list[tuple[list[str], dict, int, dict]] = []
    df: dict[str, int] = {}
    total = 0
    for e in entries:
        toks = _tokens(str(e.get("text") or ""))
        tf: dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        for t in tf:
            df[t] = df.get(t, 0) + 1
        total += len(toks)
        docs.append((toks, tf, len(toks) or 1, e))
    return docs, df, len(docs), (total / len(docs) if docs else 1.0) or 1.0


def rank(entries: list[dict], query: str, now: float | None = None) -> list[dict]:
    """Entries best-first for `query`. No query text -> recency order, which is
    what the old last-N injection did."""
    now = time.time() if now is None else now
    if not entries:
        return []
    terms = list(dict.fromkeys(_tokens(query)))
    docs, df, n, avg = _build_corpus(entries)
    return _rank_docs(docs, df, n, avg, terms, now)


# Tokenizing every note on every turn costs more than the rest of the memory
# read put together, and the notes only change when the file does. So the
# corpus is cached on the file's stamp plus an in-process generation counter
# (bumped by _save, so a same-size rewrite still invalidates it).
_CORPUS: dict[str, Any] = {"stamp": None, "docs": [], "df": {}, "n": 0, "avg": 1.0}


def _corpus() -> tuple[list, dict, int, float]:
    try:
        st = _memory_file().stat()
        stamp = (int(st.st_mtime_ns), int(st.st_size), _GEN)
    except OSError:
        stamp = (0, -1, _GEN)
    except Exception:
        stamp = (0, -1, _GEN)
    if _CORPUS.get("stamp") == stamp:
        return _CORPUS["docs"], _CORPUS["df"], _CORPUS["n"], _CORPUS["avg"]
    entries: list[dict] = []
    try:
        data = _load()
        raw = data.get("entries") if isinstance(data, dict) else None
        if isinstance(raw, list):
            entries = [e for e in raw if isinstance(e, dict) and e.get("text")]
    except Exception:
        entries = []
    docs, df, n, avg = _build_corpus(entries)
    _CORPUS.update(stamp=stamp, docs=docs, df=df, n=n, avg=avg)
    return docs, df, n, avg


def select_notes(worktree: str, query: str = "", limit: int = INJECT_LIMIT) -> list[dict]:
    """The notes worth injecting this turn: pinned ones always, then the
    best-scoring of the rest. Oldest first, for a stable display. Tolerant of a
    corrupt/absent file — never raises."""
    try:
        docs, df, n, avg = _corpus()
        if not docs:
            return []
        # project scoping after the cache: cheap, and keeps one corpus for all.
        # Superseded notes are dropped here — a replaced fact must never reach
        # the prompt alongside the fact that replaced it.
        terms = list(dict.fromkeys(_tokens(query)))
        scoped = [
            d for d in docs
            if (not d[3].get("project") or d[3].get("project") == worktree)
            and not d[3].get("superseded")
        ]
        if not scoped:
            return []
        ranked = _rank_docs(scoped, df, n, avg, terms, time.time())
        pinned = [e for e in ranked if e.get("pinned")]
        room = max(0, limit - len(pinned))
        picked = pinned + [e for e in ranked if not e.get("pinned")][:room]
        for e in picked:
            nid = e.get("id")
            _HITS[nid] = _HITS.get(nid, 0) + 1
        picked.sort(key=lambda e: (float(e.get("created") or 0.0), e.get("id", 0)))
        return picked
    except Exception:
        return []


def mark_used(text: str) -> int:
    """Credit a note whose content showed up in the agent's own reply.

    This is the only real quality signal available: a note that keeps getting
    injected but is never reflected in an answer is a bad note — wrong, vague,
    or too long — and the store should stop paying for it.
    """
    terms = [t for t in _tokens(text) if len(t) >= 4]
    if not terms:
        return 0
    hits = 0
    for _toks, _tf, _len_, entry in _corpus()[0]:
        if entry.get("superseded"):
            continue
        eterms = set(_tokens(str(entry.get("text") or "")))
        if eterms and len(set(terms) & eterms) / float(len(terms)) >= 0.4:
            nid = entry.get("id")
            _HITS[nid] = _HITS.get(nid, 0) + 1
            _USED[nid] = _USED.get(nid, 0) + 1
            entry["used"] = _USED[nid]
            entry["last_used"] = time.time()
            hits += 1
    return hits


def unused_notes(min_injections: int = 3) -> list[dict]:
    """Injected repeatedly, never once reflected in a reply. Candidates for
    review — the store should not grow more of these."""
    out = []
    for _toks, _tf, _len_, entry in _corpus()[0]:
        if entry.get("superseded") or entry.get("pinned"):
            continue
        if _hits_of(entry) >= min_injections and not int(
            entry.get("used") or _USED.get(entry.get("id"), 0)
        ):
            out.append(entry)
    return out


def _gc(entries: list[dict]) -> list[dict]:
    """Over budget, drop the least valuable notes — never a pinned one. Replaces
    the old drop-the-oldest, which let a burst of chatter evict a hard-won note
    just for being old."""
    if len(entries) <= MAX_ENTRIES:
        return entries
    now = time.time()
    kept = sorted(entries, key=lambda e: _bonus(e, now), reverse=True)[:MAX_ENTRIES]
    kept.sort(key=lambda e: (float(e.get("created") or 0.0), e.get("id", 0)))
    return kept


# Jaccard over stemmed tokens. Auto-capture runs unattended, so without this
# it would happily re-store the same fact every session until the store fills
# with near-copies. 0.75 keeps genuinely different facts with shared words.
_DUP_THRESHOLD = 0.75


def _norm_set(text: str) -> set:
    return set(_tokens(str(text or "")))


def similarity(a: str, b: str) -> float:
    """Jaccard similarity of two notes' token sets, 0.0..1.0."""
    sa, sb = _norm_set(a), _norm_set(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    return inter / float(len(sa | sb))


def is_duplicate(text: str, worktree: str, threshold: float = _DUP_THRESHOLD) -> bool:
    """True when a note this similar is already stored for this project."""
    docs, _df, _n, _avg = _corpus()
    for _toks, _tf, _len_, entry in docs:
        if entry.get("project") and entry.get("project") != worktree:
            continue
        if entry.get("superseded"):
            continue
        if similarity(text, str(entry.get("text") or "")) >= threshold:
            return True
    return False


def supersede(note_id: int, new_text: str, worktree: str) -> bool:
    """Mark note `note_id` as replaced by `new_text`. The old note is kept (so
    history stays auditable) but stops being injected and stops blocking a
    re-add. Returns False when the id is unknown or out of scope."""
    with _LOCK:
        data = _load()
        entries = data.get("entries") or []
        target = None
        for e in entries:
            if not isinstance(e, dict) or e.get("id") != note_id:
                continue
            if e.get("project") and e.get("project") != worktree:
                return False
            target = e
            break
        if target is None:
            return False
        target["superseded"] = True
        target["superseded_by"] = str(new_text or "")[:MAX_TEXT]
        target["superseded_at"] = time.time()
        _save(data)
    return True


def _memory_file() -> Path:
    return GPath.data / "memory.json"


def _load() -> dict[str, Any]:
    try:
        raw = _memory_file().read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, ValueError):
        return {"version": 1, "entries": []}
    if not isinstance(data, dict):
        data = {"version": 1}
    entries = data.get("entries")
    if not isinstance(entries, list):
        entries = []
    return {"version": data.get("version", 1), "entries": entries}


def _save(data: dict[str, Any]) -> None:
    global _GEN
    _GEN += 1
    entries = data.get("entries") or []
    live = set()
    for e in entries:
        if not isinstance(e, dict):
            continue
        nid = e.get("id")
        live.add(nid)
        if nid in _HITS:
            try:
                e["hits"] = max(int(e.get("hits") or 0), _HITS[nid])
            except (TypeError, ValueError):
                e["hits"] = _HITS[nid]
        if nid in _USED:
            e["used"] = max(int(e.get("used") or 0), _USED[nid])
    # counters for notes that are gone would otherwise accumulate forever
    for gone in [k for k in _HITS if k not in live]:
        _HITS.pop(gone, None)
    for gone in [k for k in _USED if k not in live]:
        _USED.pop(gone, None)
    path = _memory_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _project() -> str:
    from ..globals import resolve_worktree

    try:
        return str(resolve_worktree(Path.cwd()))
    except OSError:
        return ""


def _next_id(entries: list[dict]) -> int:
    return max((e.get("id", 0) for e in entries), default=0) + 1


def _fmt_date(ts: float) -> str:
    try:
        import datetime

        return datetime.date.fromtimestamp(ts).isoformat()
    except (OSError, ValueError, OverflowError):
        return "?"


def _add(text: str, project: str, kind: str = "", pinned: bool = False) -> dict:
    text = (text or "").strip()
    if not text:
        return {"output": "Nothing to remember: the text was empty.", "error": True}
    if len(text) > MAX_TEXT:
        text = text[:MAX_TEXT] + "…"
    kind = kind if kind in KINDS else ""
    with _LOCK:
        data = _load()
        entries = data["entries"]
        entry = {
            "id": _next_id(entries),
            "text": text,
            "project": project,
            "created": time.time(),
            "kind": kind,
            "pinned": bool(pinned),
            "hits": 0,
        }
        entries.append(entry)
        data["entries"] = _gc(entries)
        _save(data)
    where = "project memory" if project else "global memory"
    tags = "".join([f", {kind}" if kind else "", ", pinned" if pinned else ""])
    return {"output": f"Remembered ({where}, id {entry['id']}{tags}):\n{text}"}


def _list(project: str) -> dict:
    with _LOCK:
        data = _load()
    entries = data["entries"]
    proj = [e for e in entries if e.get("project") == project]
    glob = [e for e in entries if not e.get("project")]

    def fmt(group: list[dict]) -> str:
        lines = []
        for e in group:
            tag = f"[{e['kind']}]" if e.get("kind") else ("[pinned]" if e.get("pinned") else "")
            if e.get("superseded"):
                tag = "[superseded]"
            lines.append(
                f"{e.get('id', 0)}. ({_fmt_date(e.get('created', 0))}"
                f"{' ' + tag if tag else ''}) "
                f"{str(e.get('text', '')).replace(chr(10), ' ')}"
            )
        return "\n".join(lines) if lines else "(none)"

    out: list[str] = []
    out.append(f"Project memory for {project or '(current folder)'}:")
    out.append(fmt(proj))
    if glob:
        out.append("Global memory:")
        out.append(fmt(glob))
    out.append(f"Total notes: {len(entries)}")
    return {"output": "\n".join(out)}


def _delete(spec: Any, project: str) -> dict:
    with _LOCK:
        data = _load()
        entries = data["entries"]
        target_id = None
        if isinstance(spec, bool):
            target_id = None
        elif isinstance(spec, (int, float)):
            target_id = int(spec)
        else:
            m = re.fullmatch(r"\s*#?(\d+)\s*", str(spec))
            if m:
                target_id = int(m.group(1))
        if target_id is None:
            return {
                "output": "Give the id of the note to delete (see `list` for ids).",
                "error": True,
            }
        kept = [e for e in entries if e.get("id") != target_id]
        if len(kept) == len(entries):
            return {"output": f"No note with id {target_id}.", "error": True}
        data["entries"] = kept
        _save(data)
    return {"output": f"Deleted note {target_id}."}


def _review() -> dict:
    """Notes that keep getting injected but never show up in a reply. These cost
    context every turn and earn nothing, so they should be fixed or deleted."""
    try:
        rows = unused_notes()
    except Exception as e:
        return {"output": f"Could not review notes: {e}", "error": True}
    if not rows:
        return {"output": "No unused notes — everything injected has been used."}
    lines = [
        f"{e.get('id', 0)}. injected {_hits_of(e)}x, never used: "
        f"{str(e.get('text', '')).replace(chr(10), ' ')[:120]}"
        for e in rows
    ]
    lines.append(
        "These are shown to the model every turn but never reflected in a "
        "reply. Re-word them to be specific, or delete them."
    )
    return {"output": "\n".join(lines)}


def _clear(project: str) -> dict:
    with _LOCK:
        data = _load()
        entries = data["entries"]
        if project:
            data["entries"] = [e for e in entries if e.get("project") != project]
            where = f"Cleared project memory for {project}."
        else:
            data["entries"] = []
            where = "Cleared global memory."
        _save(data)
    return {"output": f"{where} (was {len(entries)} notes.)"}


def tool() -> Tool:
    description = """Cross-session notes (auto-loaded into future prompts, picked by
relevance to the current request; pinned notes are always included).
add/list/review/delete/clear. `review` finds notes that are injected every turn
but never actually used — re-word or delete those."""

    def run(input: dict) -> dict:
        action = str(input.get("action") or "add").strip().lower()
        if action not in _ACTIONS:
            return {
                "output": f"Unknown action {action!r} (want one of {', '.join(_ACTIONS)}).",
                "error": True,
            }
        project = _project()
        if action == "add":
            return _add(
                str(input.get("text") or ""),
                project,
                str(input.get("kind") or "").strip().lower(),
                bool(input.get("pinned")),
            )
        if action == "list":
            return _list(project)
        if action == "review":
            return _review()
        if action == "delete":
            return _delete(input.get("id"), project)
        return _clear(project)

    return Tool(
        name="remember",
        description=description,
        parameters=schema_with(
            {
                "action": {
                    "type": "string",
                    "description": "add (default), list, delete, clear",
                    "enum": list(_ACTIONS),
                    "optional": True,
                },
                "text": {
                    "type": "string",
                    "description": "Note text",
                    "optional": True,
                },
                "id": {
                    "type": "integer",
                    "description": "Note id",
                    "optional": True,
                },
                "kind": {
                    "type": "string",
                    "description": "Note type: " + ", ".join(KINDS),
                    "enum": list(KINDS),
                    "optional": True,
                },
                "pinned": {
                    "type": "boolean",
                    "description": "Always inject this note, never evict it",
                    "optional": True,
                },
            },
            [],
        ),
        run=run,
    )

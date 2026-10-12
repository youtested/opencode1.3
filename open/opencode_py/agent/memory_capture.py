"""Session-end memory capture: distil durable facts out of a conversation so
the agent stops depending on remembering to call `remember`.

WHY TWO PASSES
- The model pass is accurate but costs a round trip (~20s on a slow free
  model). A question asked in the first seconds after launch would otherwise
  see nothing new.
- The quick pass costs ~90ms and no model call, but measured at 120 notes per
  session of which almost all are throwaway chatter ("ok what is this
  project"). It is good enough to *answer* with, and far too noisy to *store*.
So: the quick pass answers this turn and is thrown away; the model pass decides
what is actually worth keeping.

WHY PIGGYBACK-SHAPED
- One small completion per real session, not per turn. Marginal cost per turn
  stays zero.
- Notes are SHORT and atomic. The hand-saved store held 8 notes averaging 441
  chars (whole session summaries pasted in); long notes dilute term overlap
  and make top-K selection far less useful.

Never raises. A failed capture must not break a session.
"""

from __future__ import annotations

from ..util.log import swallow
import json
import re
import time
from pathlib import Path
from typing import Any

from ..globals import Path as GPath

MAX_NOTES = 5
MAX_NOTE_CHARS = 220
TRANSCRIPT_BUDGET = 4500
QUICK_BUDGET = 14000
MIN_TOOL_CALLS = 3
RETRY_DELAYS = (0.0, 2.0, 5.0)

_SYSTEM = "You distil durable memory for a coding agent. You output JSON only."

_INSTRUCTIONS = """\
Extract the facts from this conversation that will still be true and useful in
FUTURE sessions. Rules:

- One fact per note. At most {max_notes} notes. Return [] if nothing qualifies.
- Each note: one sentence, under {max_chars} characters, no date/time stamps,
  no "we did X today" status reports, no file paths of throwaway experiments.
- Keep only: standing user preferences, project conventions/rules, decisions
  and the reason behind them, non-obvious gotchas, and reusable fixes.
- Prefer a specific factual statement ("oauth refresh lives in session_store.py")
  over a vague one ("we worked on auth issues").
- If a note restates something already in MEMORY, do not repeat it.
- If a note CONTRADICTS a MEMORY entry, return it with "supersedes": <id>.

MEMORY (already stored; id = memory id):
{memory}

Return exactly this JSON shape, nothing else:
{{"notes": [{{"text": "...", "kind": "rule|fact|decision|fix", "supersedes": 0}}]}}"""

# A line that carries a conclusion, not chatter. Drives both the importance
# ranking and the quick pass.
_CONCLUSION = re.compile(
    r"\b(cause|root cause|bug is|the fix|fix is|turns out|found that|issue is|"
    r"because|which means|so the|always|never|must|instead of|the reason|"
    r"prefer|rule|convention|remember that)\b",
    re.I,
)
_SPEAKER = re.compile(r"^\[[a-z]+\]:\s*", re.I)


def _clip(text: Any, limit: int) -> str:
    text = str(text or "")
    return text if len(text) <= limit else text[:limit] + "…"


def _message_text(m: dict) -> str:
    c = m.get("content")
    if isinstance(c, list):
        return " ".join(str(p.get("text") or "") for p in c if isinstance(p, dict))
    return str(c or "")


def _tool_calls(m: dict) -> int:
    """How many tool calls this message carries — the 'was this real work' test."""
    c = m.get("content")
    if isinstance(c, list):
        return sum(1 for p in c if isinstance(p, dict) and p.get("type") == "tool_use")
    tc = m.get("tool_calls")
    return len(tc) if isinstance(tc, list) else 0


def is_real_work(messages: list[dict], min_tool_calls: int = MIN_TOOL_CALLS) -> bool:
    """A throwaway chat ("what's 2+2") is not worth 20s of capture."""
    if not messages:
        return False
    return sum(_tool_calls(m) for m in messages if isinstance(m, dict)) >= min_tool_calls


# ---------------------------------------------------------------- selection


def _rank_line(text: str, role: str, position: float) -> int:
    """Cheap importance for one line. No model call."""
    score = 0
    if role == "user":
        score += 3          # the user's own words are the decisions
    if _CONCLUSION.search(text):
        score += 3
    if re.search(r"[/.]\w{3,}", text):
        score += 1          # names a path or dotted symbol
    if re.search(r"\b(always|never|must|prefer|instead of|rule)\b", text, re.I):
        score += 2
    return score + int(position * 3)   # recency breaks ties


def _candidate_lines(messages: list[dict]) -> list[tuple[int, str, str]]:
    """(score, role, line) for every line worth considering."""
    total = max(len(messages), 1)
    out: list[tuple[int, str, str]] = []
    for i, m in enumerate(messages):
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or "")
        if role not in ("user", "assistant"):
            continue
        text = _message_text(m).strip()
        if not text:
            continue
        for part in re.split(r"\n+|(?<=[.!?])\s+", text):
            part = " ".join(_SPEAKER.sub("", part).split())
            if len(part) < 25:
                continue
            out.append((_rank_line(part, role, i / total), role, part))
    return out


def select_transcript(messages: list[dict], budget: int = TRANSCRIPT_BUDGET) -> str:
    """The most important lines of a conversation, within `budget` characters.

    Replaces a blunt head+tail window: the user's own lines and conclusion-
    bearing lines win on score, recency breaks ties, and the survivors are
    re-emitted in original order so the model still reads the session
    chronologically.
    """
    cands = _candidate_lines(messages)
    if not cands:
        return ""
    ranked = sorted(range(len(cands)), key=lambda i: (cands[i][0], -i), reverse=True)
    chosen: list[int] = []
    spent = 0
    for i in ranked:
        # +1 per line: the join adds a newline that must fit the budget too
        cost = len(cands[i][2]) + 1
        if spent + cost > budget:
            continue
        chosen.append(i)
        spent += cost
    return "\n".join(cands[i][2] for i in sorted(chosen))


def quick_notes(messages: list[dict], budget: int = QUICK_BUDGET) -> list[str]:
    """Instant, no-model candidates for answering the first turn of a session.

    Deliberately generous: this is shown to the model but never stored, so
    noise costs a little context and nothing more.
    """
    cands = _candidate_lines(messages)
    ranked = sorted(range(len(cands)), key=lambda i: (cands[i][0], -i), reverse=True)
    out, spent = [], 0
    for i in ranked:
        line = cands[i][2]
        if spent + len(line) > budget:
            continue
        out.append(line)
        spent += len(line)
    return out


# ------------------------------------------------------------------- store


def _existing_block(worktree: str, cap: int = 40) -> str:
    try:
        from ..tools.remember import select_notes

        notes = select_notes(worktree, query="", limit=cap)
    except Exception:
        return "(none)"
    if not notes:
        return "(none)"
    return "\n".join(
        f"{e.get('id')}: {_clip(e.get('text'), MAX_NOTE_CHARS)}" for e in notes
    )


def build_prompt(messages: list[dict], worktree: str = "",
                 max_notes: int = MAX_NOTES) -> str:
    body = select_transcript(messages)
    if not body:
        return ""
    return _INSTRUCTIONS.format(
        max_notes=max_notes,
        max_chars=MAX_NOTE_CHARS,
        memory=_existing_block(worktree),
    ) + "\n\nCONVERSATION:\n" + body


def parse_notes(raw: str) -> list[dict[str, Any]]:
    """Tolerant parse. Accepts {"notes": [...]}, a bare array, or JSON inside a
    ```json fence; entries may be strings or objects. Never raises."""
    text = str(raw or "").strip()
    if not text:
        return []
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    data: Any = None
    try:
        data = json.loads(text)
    except ValueError:
        for lo, hi in (("[", "]"), ("{", "}")):
            s, e = text.find(lo), text.rfind(hi)
            if s >= 0 and e > s:
                try:
                    data = json.loads(text[s : e + 1])
                    break
                except ValueError:
                    data = None
    out: list[dict[str, Any]] = []
    items = data.get("notes") if isinstance(data, dict) else data
    if not isinstance(items, list):
        return []
    for item in items:
        if isinstance(item, str):
            item = {"text": item}
        if not isinstance(item, dict):
            continue
        body = str(item.get("text") or "").strip()
        if not body:
            continue
        try:
            sup = int(item.get("supersedes") or 0)
        except (TypeError, ValueError):
            sup = 0
        kind = str(item.get("kind") or "").strip().lower()
        out.append({
            "text": _clip(body, MAX_NOTE_CHARS),
            "kind": kind if kind in ("rule", "fact", "decision", "fix") else "",
            "supersedes": sup,
        })
    return out


def _completion(provider: Any, messages: list[dict], max_tokens: int) -> tuple[str, dict]:
    """One completion. Prefers the non-streaming path; falls back to streaming
    for providers that only implement stream_chat. Returns (text, usage)."""
    kwargs = {"max_tokens": max_tokens, "temperature": 0.2}
    complete = getattr(provider, "complete", None)
    if callable(complete):
        resp = complete(messages, None, **kwargs)
        usage = resp.get("usage") or {} if isinstance(resp, dict) else {}
        choices = resp.get("choices") or [] if isinstance(resp, dict) else []
        text = ""
        if choices and isinstance(choices[0], dict):
            text = str((choices[0].get("message") or {}).get("content") or "")
        return text, usage if isinstance(usage, dict) else {}
    chunks: list[str] = []

    def on_event(evt: Any) -> None:
        if getattr(evt, "kind", "") == "text_delta":
            chunks.append(str(getattr(evt, "text", "") or ""))

    provider.stream_chat(messages, None, on_event, **kwargs)
    return "".join(chunks), {}


def capture(
    messages: list[dict],
    provider: Any,
    worktree: str = "",
    max_notes: int = MAX_NOTES,
    require_real_work: bool = True,
) -> dict[str, Any]:
    """Distil `messages` into stored notes. Retries a few times on an empty
    reply — measured at roughly 1 in 3 failures on a slow free model, which is
    not acceptable to leave unhandled. Returns a report; never raises.

    `require_real_work` is False when replaying the pending queue: the queue has
    already been filtered, and its messages are a single distilled turn that
    carries no tool calls to judge.
    """
    started = time.time()
    report: dict[str, Any] = {
        "ok": False, "captured": 0, "written": 0, "skipped_duplicate": 0,
        "superseded": 0, "prompt_chars": 0, "output_chars": 0, "usage": {},
        "elapsed_ms": 0.0, "attempts": 0, "error": "", "notes": [],
    }
    try:
        if require_real_work and not is_real_work(messages):
            report.update(ok=True, error="skipped: not real work")
            return report
        prompt = build_prompt(messages, worktree, max_notes)
        report["prompt_chars"] = len(prompt)
        if not prompt.strip():
            report["error"] = "no transcript survived selection"
            return report
        messages_in = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": prompt},
        ]
        budget = min(1200, 200 * max_notes)
        raw = ""
        for attempt, delay in enumerate(RETRY_DELAYS, start=1):
            report["attempts"] = attempt
            if delay:
                time.sleep(delay)
            raw, usage = _completion(provider, messages_in, budget)
            if usage:
                report["usage"] = usage
            if parse_notes(raw):
                break
        report["output_chars"] = len(raw or "")
        notes = parse_notes(raw)
        report["captured"] = len(notes)
        report["notes"] = notes
        if notes:
            report.update(_write_notes(notes, worktree))
        report["ok"] = True
    except Exception as e:  # a failed capture must never break a session
        report["error"] = f"{type(e).__name__}: {e}"
    finally:
        report["elapsed_ms"] = round((time.time() - started) * 1000, 1)
    return report


def _write_notes(notes: list[dict], worktree: str) -> dict[str, Any]:
    from ..tools import remember as R

    added = dupes = superseded = 0
    for note in notes:
        if R.is_duplicate(note["text"], worktree):
            dupes += 1
            continue
        sup = int(note.get("supersedes") or 0)
        if sup:
            if not R.supersede(sup, note["text"], worktree):
                continue
            superseded += 1
        R._add(note["text"], project=worktree, kind=note.get("kind") or "")
        added += 1
    return {"written": added, "skipped_duplicate": dupes, "superseded": superseded}


# ------------------------------------------------- pending-session queue
#
# Capture runs on the NEXT launch, not on session close. A phone has no
# business running a background daemon and Android kills them anyway; this way
# there is no background process at all, closing stays instant, and the work
# happens while the user is already typing.


def _pending_file() -> Path:
    return GPath.data / "capture_pending.json"


def mark_pending(session_id: str, messages: list[dict]) -> bool:
    """Record a finished session for later capture. Stores the distilled
    selection, not the whole transcript, so the queue stays small. A chat that
    was never real work is dropped here, so the queue only ever holds work."""
    try:
        if not is_real_work(messages):
            return False
        body = select_transcript(messages)
        if not body:
            return False
        path = _pending_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                data = {}
        except (OSError, ValueError):
            data = {}
        data[str(session_id)] = {"text": body, "queued": time.time()}
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(path)
        return True
    except Exception:
        return False


def peek_pending() -> list[dict]:
    """Look at the queue WITHOUT consuming it.

    The first turn of a session uses this: it shows the previous session's
    distilled lines immediately (~90ms, no model call) so a question asked in
    the opening seconds still has that context. `run_pending` consumes the same
    entries later to decide what is actually worth storing.
    """
    try:
        data = json.loads(_pending_file().read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return []
    except (OSError, ValueError):
        return []
    return [
        {"session_id": k, "text": str(v.get("text") or "")}
        for k, v in sorted(data.items(), key=lambda kv: kv[1].get("queued", 0))
        if isinstance(v, dict) and str(v.get("text") or "").strip()
    ]


def take_pending() -> list[dict]:
    """Return and clear everything queued, oldest first."""
    try:
        data = json.loads(_pending_file().read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return []
    except (OSError, ValueError):
        return []
    out = [
        {"session_id": k, "text": str(v.get("text") or "")}
        for k, v in sorted(data.items(), key=lambda kv: kv[1].get("queued", 0))
        if isinstance(v, dict) and str(v.get("text") or "").strip()
    ]
    try:
        _pending_file().unlink()
    except OSError as _oc_exc:
        swallow(_oc_exc, "opencode_py/agent/memory_capture.py:432")
        pass
    return out


def run_pending(provider: Any, worktree: str = "",
                max_notes: int = MAX_NOTES) -> dict[str, Any]:
    """Capture everything queued since the last launch. Never raises."""
    summary: dict[str, Any] = {
        "queued": 0, "captured": 0, "written": 0, "skipped_duplicate": 0,
        "superseded": 0, "elapsed_ms": 0.0, "errors": [],
    }
    started = time.time()
    try:
        items = take_pending()
    except Exception:
        items = []
    summary["queued"] = len(items)
    if not items:
        summary["elapsed_ms"] = round((time.time() - started) * 1000, 1)
        return summary
    # the stored selection is one flat body; hand it over as a single turn
    messages = [{"role": "user", "content": it["text"]} for it in items]
    try:
        rep = capture(messages, provider, worktree, max_notes,
                      require_real_work=False)
        for key in ("captured", "written", "skipped_duplicate", "superseded"):
            summary[key] = rep.get(key, 0)
        if rep.get("error"):
            summary["errors"].append(rep["error"])
    except Exception as e:
        summary["errors"].append(f"{type(e).__name__}: {e}")
    summary["elapsed_ms"] = round((time.time() - started) * 1000, 1)
    return summary

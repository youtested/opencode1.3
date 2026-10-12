"""Tests for automatic session capture and the memory quality loop.

Covers what the before/after measurement exercised:
- importance-based transcript selection replaces the blunt head+tail window
- the quick pass is instant, never stored, and shown on the first turn only
- an empty model reply is retried instead of silently losing the session
- the pending queue defers capture to the next launch and survives a restart
- trivial sessions are never queued
- superseded notes stop being injected; duplicates are rejected
- a note injected every turn but never used is reported by `review`
"""

from __future__ import annotations

import json
import time

import pytest

from opencode_py.agent import memory_capture as MC
from opencode_py.agent.system import build_system_prompt, labeled_prompt_parts
from opencode_py.config import Config
from opencode_py.globals import Path as GPath
from opencode_py.tools import remember as R


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(GPath, "data", tmp_path)
    R._CORPUS["stamp"] = None
    R._HITS.clear()
    R._USED.clear()
    yield tmp_path
    R._CORPUS["stamp"] = None
    R._HITS.clear()
    R._USED.clear()


def _work(messages, n_tools=4):
    """A conversation that counts as real work (>= MIN_TOOL_CALLS)."""
    out = list(messages)
    for i in range(n_tools):
        out.append({
            "role": "assistant",
            "content": [{"type": "tool_use", "name": "read", "input": {"path": f"f{i}.py"}}],
        })
    return out


# ------------------------------------------------------------- selection


def test_selection_keeps_the_middle_not_just_head_and_tail():
    """The old window dropped everything between the first and last 8k chars.
    Importance ranking must keep a conclusion from the middle of a long chat."""
    msgs = [
        {"role": "user", "content": "start of a very long chat about nothing much at all here"},
        {"role": "assistant", "content": "acknowledged, nothing important in that reply either"},
    ]
    for i in range(60):
        msgs.append({"role": "assistant", "content": f"filler line number {i} with no signal"})
    msgs.append({
        "role": "user",
        "content": "IMPORTANT: never delete test files, even when they look unused, because "
                   "they are the only regression guard we have for the parser",
    })
    for i in range(60):
        msgs.append({"role": "assistant", "content": f"more filler line number {i} here"})
    msgs.append({"role": "user", "content": "the end of the chat, also not important itself"})

    picked = MC.select_transcript(msgs, budget=900)
    assert "never delete test files" in picked, "middle-of-session rule was dropped"
    assert len(picked) <= 900, "budget not respected"


def test_selection_prefers_user_lines_and_conclusions():
    msgs = [
        {"role": "user", "content": "please always run pytest before saying a task is done ok"},
        {"role": "assistant", "content": "the cause of the failure was a stale cached index"},
        {"role": "assistant", "content": "done, updated three files in the parser module"},
    ]
    picked = MC.select_transcript(msgs, budget=120)
    assert "run pytest" in picked or "cause of the failure" in picked
    assert "done, updated three files" not in picked, "filler outranked a signal"


def test_selection_respects_budget_and_is_chronological():
    msgs = [
        {"role": "user", "content": f"a durable rule number {i} that the user stated clearly here"}
        for i in range(80)
    ]
    picked = MC.select_transcript(msgs, budget=400)
    assert len(picked) <= 400
    nums = [int(x.split("number ")[1].split(" ")[0]) for x in picked.splitlines()]
    assert nums == sorted(nums), "survivors must stay in original order"


def test_no_transcript_returns_empty():
    assert MC.select_transcript([]) == ""
    assert MC.build_prompt([]) == ""


# ------------------------------------------------------------- real work


def test_trivial_chat_is_not_real_work():
    assert not MC.is_real_work([
        {"role": "user", "content": "what is 2+2"},
        {"role": "assistant", "content": "4"},
    ])
    assert MC.is_real_work(_work([{"role": "user", "content": "do the real thing here"}]))


def test_capture_skips_trivial_conversation():
    class P:
        def complete(self, *a, **k):
            raise AssertionError("must not call the model for a throwaway chat")

    rep = MC.capture([{"role": "user", "content": "hi"}], P(), worktree="")
    assert rep["ok"] and "not real work" in rep["error"]
    assert rep["written"] == 0


# ------------------------------------------------------------ quick pass


def test_quick_pass_is_instant_and_not_stored(data_dir):
    msgs = _work([
        {"role": "user", "content": "always run the tests before reporting, never skip them"},
        {"role": "assistant", "content": "the fix is in apply_patch.py, line range mode"},
    ])
    t0 = time.perf_counter()
    notes = MC.quick_notes(msgs)
    ms = (time.perf_counter() - t0) * 1000
    assert ms < 200, f"quick pass must be instant, took {ms:.0f}ms"
    assert notes, "quick pass produced nothing"
    # the crucial part: it must NOT be written to the store
    assert R._load()["entries"] == []


def test_recent_block_only_appears_when_supplied(data_dir, tmp_path):
    with_ = build_system_prompt(
        directory=tmp_path, worktree=tmp_path, provider_id="p", model_id="m",
        cfg=Config(), query="x", recent="the last session decided to use pytest",
    )
    without = build_system_prompt(
        directory=tmp_path, worktree=tmp_path, provider_id="p", model_id="m",
        cfg=Config(), query="x",
    )
    assert "last session" in with_
    assert "last session" not in without
    assert len(with_) > len(without)


def test_labeled_parts_still_match_build(data_dir, tmp_path):
    sent = build_system_prompt(
        directory=tmp_path, worktree=tmp_path, provider_id="p", model_id="m",
        cfg=Config(), query="q", recent="r",
    )
    parts = labeled_prompt_parts(
        directory=tmp_path, worktree=tmp_path, provider_id="p", model_id="m",
        cfg=Config(), agent="build", query="q", recent="r",
    )
    assert sent == "\n\n".join(t for _l, t in parts)
    labels = [l for l, _t in parts]
    assert labels[0] == "base.md"
    assert "recent session" in labels


# --------------------------------------------------------------- retries


class _Flaky:
    """Returns empty `empties` times, then a real answer."""

    def __init__(self, empties=1, notes=None):
        self.calls = 0
        self.empties = empties
        self.notes = notes or [{"text": "always run pytest before reporting done"}]

    def complete(self, messages, tools=None, **kw):
        self.calls += 1
        if self.calls <= self.empties:
            return {"choices": [{"message": {"content": ""}}], "usage": {}}
        return {
            "choices": [{"message": {"content": json.dumps({"notes": self.notes})}}],
            "usage": {"total_tokens": 42},
        }


def test_empty_reply_is_retried(data_dir, monkeypatch):
    monkeypatch.setattr(MC, "RETRY_DELAYS", (0.0, 0.0, 0.0))
    p = _Flaky(empties=2)
    rep = MC.capture(_work([{"role": "user", "content": "do the work and state the rules"}]),
                     p, worktree="")
    assert rep["attempts"] == 3, rep
    assert rep["written"] == 1
    assert p.calls == 3


def test_capture_never_raises_on_a_dead_provider(data_dir):
    class Dead:
        def complete(self, *a, **k):
            raise RuntimeError("network down")

    rep = MC.capture(_work([{"role": "user", "content": "x" * 40}]), Dead(), worktree="")
    assert rep["ok"] is False
    assert "network down" in rep["error"]
    assert R._load()["entries"] == []


# ------------------------------------------------------- pending queue


def test_mark_pending_ignores_trivial_sessions(data_dir):
    assert MC.mark_pending("s1", [{"role": "user", "content": "hi"}]) is False
    assert MC.take_pending() == []


def test_pending_survives_and_is_consumed_once(data_dir):
    msgs = _work([{"role": "user", "content": "always pin the release notes as a rule here"}])
    assert MC.mark_pending("s1", msgs) is True
    peeked = MC.peek_pending()
    assert peeked and "release notes" in peeked[0]["text"]
    taken = MC.take_pending()
    assert len(taken) == 1
    assert MC.take_pending() == [], "queue must be consumed exactly once"


def test_pending_is_bounded_not_the_whole_transcript(data_dir):
    msgs = _work([{"role": "user", "content": f"durable rule {i} stated by the user clearly"} 
                  for i in range(400)])
    MC.mark_pending("s1", msgs)
    on_disk = (data_dir / "capture_pending.json").read_text()
    assert len(on_disk) < 20000, "queue must store the distilled selection, not the session"


def test_run_pending_captures_and_stores(data_dir, monkeypatch):
    monkeypatch.setattr(MC, "RETRY_DELAYS", (0.0,))
    MC.mark_pending("s1", _work([
        {"role": "user", "content": "the fix belongs in apply_patch.py, not edit.py"},
    ]))
    out = MC.run_pending(
        _Flaky(empties=0, notes=[{"text": "the fix belongs in apply_patch.py"}]),
        worktree="",
    )
    assert out["queued"] == 1
    assert out["written"] == 1
    assert "apply_patch" in R._list("")["output"]


def test_run_pending_is_a_noop_when_empty(data_dir):
    out = MC.run_pending(_Flaky(), worktree="")
    assert out["queued"] == 0 and out["elapsed_ms"] >= 0


# ------------------------------------------------- dedupe + supersede


def test_capture_skips_duplicates_of_existing_notes(data_dir, monkeypatch):
    monkeypatch.setattr(MC, "RETRY_DELAYS", (0.0,))
    note = "always run pytest before reporting the task is done"
    R._add(note, project="")
    R._CORPUS["stamp"] = None
    MC.mark_pending("s1", _work([{"role": "user", "content": "x" * 40}]))
    out = MC.run_pending(_Flaky(empties=0, notes=[{"text": note}]), worktree="")
    assert out["skipped_duplicate"] == 1
    assert out["written"] == 0
    assert len(R._load()["entries"]) == 1, "store must not grow a near-copy"


def test_supersede_removes_the_old_fact_from_the_prompt(data_dir):
    from opencode_py.agent.system import _load_memory

    R._add("the project lives in ~/old_path", project="", kind="fact")
    old_id = R._load()["entries"][0]["id"]
    R._CORPUS["stamp"] = None
    assert "old_path" in _load_memory("", query="where is the project")

    assert R.supersede(old_id, "the project lives in ~/new_path", "") is True
    R._CORPUS["stamp"] = None
    assert "old_path" not in _load_memory("", query="where is the project")
    R._add("the project lives in ~/new_path", project="", kind="fact")
    R._CORPUS["stamp"] = None
    assert "new_path" in _load_memory("", query="where is the project")


def test_supersede_unknown_id_is_false(data_dir):
    assert R.supersede(9999, "x", "") is False


# ------------------------------------------------------ the quality loop


def test_mark_used_credits_a_reflected_note(data_dir):
    R._add("the oauth refresh race lives in session_store.py", project="")
    R._CORPUS["stamp"] = None
    hits = R.mark_used(
        "I looked at the oauth refresh race in session_store.py and fixed the ordering."
    )
    assert hits == 1
    # the credit is in-process on purpose: writing memory.json here would bump
    # its mtime and invalidate the system-prompt cache on every single turn
    entry = R._corpus()[0][0][3]
    assert entry.get("used") == 1
    R._add("a later note forces a write", project="")
    assert R._load()["entries"][0].get("used") == 1


def test_review_reports_injected_but_never_used(data_dir):
    R._add("a vague note that matches lots of queries about widgets", project="")
    R._CORPUS["stamp"] = None
    for _ in range(4):
        R.select_notes("", query="widget stuff")
    R._CORPUS["stamp"] = None
    out = R.tool().run({"action": "review"})["output"]
    assert "never used" in out and "vague note" in out


def test_review_is_quiet_when_notes_are_used(data_dir):
    R._add("the release notes live in CHANGELOG.md", project="")
    R._CORPUS["stamp"] = None
    for _ in range(4):
        R.select_notes("", query="changelog")
    R.mark_used("the release notes are in CHANGELOG.md as expected")
    out = R.tool().run({"action": "review"})["output"]
    assert "No unused notes" in out


def test_pinned_notes_are_never_flagged_unused(data_dir):
    R._add("a pinned rule about the widget pipeline", project="", pinned=True)
    R._CORPUS["stamp"] = None
    for _ in range(5):
        R.select_notes("", query="widget")
    R._CORPUS["stamp"] = None
    assert "No unused notes" in R.tool().run({"action": "review"})["output"]


def test_scoring_never_writes_the_store(data_dir):
    R._add("a stable note about the tokenizer budget", project="")
    before = (data_dir / "memory.json").stat().st_mtime_ns
    for _ in range(5):
        R.select_notes("", query="tokenizer budget")
        R.mark_used("the tokenizer budget is fine")
    assert (data_dir / "memory.json").stat().st_mtime_ns == before

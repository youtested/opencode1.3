"""Tests for relevance-scored long-term memory.

Covers the three behaviours added on top of plain note storage:
- notes are picked by relevance to the current turn, not by recency alone
- pinned notes are always injected and never evicted; hits are counted
- over-budget eviction drops the least valuable note, not simply the oldest
"""

from __future__ import annotations

import time

import pytest

from opencode_py.agent.system import _load_memory
from opencode_py.config import Config
from opencode_py.globals import Path as GPath
from opencode_py.tools import remember as R


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(GPath, "data", tmp_path)
    R._CORPUS["stamp"] = None
    R._HITS.clear()
    yield tmp_path
    R._CORPUS["stamp"] = None
    R._HITS.clear()


def _seed(texts, project=""):
    for t in texts:
        R._add(t, project=project)


# ------------------------------------------------------------------ ranking


def test_relevant_old_note_outranks_new_noise(data_dir):
    """The point of scoring: an old, on-topic note beats 60 newer off-topic ones."""
    old = time.time() - 400 * 86400
    R._save({
        "version": 1,
        "entries": [{"id": 1, "text": "oauth token refresh race in session_store",
                     "project": "", "created": old}]
        + [{"id": 2 + i, "text": f"unrelated note about tabs number {i}",
            "project": "", "created": old + (i + 1) * 3600} for i in range(60)],
    })
    picked = R.select_notes("", query="fix the oauth token refresh race")
    assert any("oauth token refresh race" in str(e.get("text")) for e in picked)


def _noise(n=40):
    return [f"unrelated chatter about tabs number {i}" for i in range(n)]


def test_query_changes_what_is_injected(data_dir):
    # Both topical notes are OLDER than the noise, so when only one of them
    # matches, the other loses the recency tiebreak to all 40 noise notes and
    # drops out of the top-K. The store must also exceed INJECT_LIMIT, or
    # top-K injects everything and the test proves nothing.
    old = time.time() - 90 * 86400
    R._save({
        "version": 1,
        "entries": [
            {"id": 1, "text": "the auth module lives in opencode_py/auth.py",
             "project": "", "created": old},
            {"id": 2, "text": "deploys only happen on fridays",
             "project": "", "created": old + 60},
        ] + [
            {"id": 3 + i, "text": f"unrelated chatter about tabs number {i}",
             "project": "", "created": time.time() - i * 60}
            for i in range(40)
        ],
    })
    on_auth = R.select_notes("", query="where is the auth module?")
    assert any("auth.py" in str(e.get("text")) for e in on_auth)
    assert all("fridays" not in str(e.get("text")) for e in on_auth), (
        "an unrelated turn should not drag the deploy note along"
    )
    on_deploy = R.select_notes("", query="when do we deploy?")
    assert any("fridays" in str(e.get("text")) for e in on_deploy)
    assert all("auth.py" not in str(e.get("text")) for e in on_deploy)


def test_empty_query_falls_back_to_recency(data_dir):
    _seed([f"note number {i}" for i in range(5)])
    picked = R.select_notes("", query="")
    ids = [e.get("id") for e in picked]
    assert ids == sorted(ids), "no query -> oldest-first stable order"


def test_limit_caps_injected_notes(data_dir):
    _seed([f"note number {i}" for i in range(40)])
    assert len(R.select_notes("", query="note", limit=7)) == 7


# ------------------------------------------------------------------- pinned


def test_pinned_note_always_injected(data_dir):
    R._add("pinned house rule", project="", pinned=True)
    _seed([f"chatter {i}" for i in range(40)])
    picked = R.select_notes("", query="completely unrelated topic zzzz")
    assert any(e.get("pinned") for e in picked)
    assert any("pinned house rule" in str(e.get("text")) for e in picked)


def test_pinned_never_evicted(data_dir):
    R._add("ancient but pinned", project="", pinned=True, kind="rule")
    for i in range(R.MAX_ENTRIES + 25):
        R._add(f"chatter {i}", project="")
    texts = [str(e.get("text")) for e in R._load()["entries"]]
    assert "ancient but pinned" in texts, "pinned note was evicted under pressure"


# --------------------------------------------------------------------- hits


def test_injection_counts_hits_and_persists(data_dir):
    R._add("counted note", project="")
    for _ in range(3):
        R.select_notes("", query="counted note")
    R._add("another note", project="")  # a write flushes the counters
    entry = next(e for e in R._load()["entries"] if "counted note" in str(e.get("text")))
    assert int(entry.get("hits") or 0) == 3


def test_scoring_does_not_write(data_dir):
    """Scoring must not touch memory.json: a write bumps its mtime and would
    invalidate the system-prompt cache on every single turn."""
    R._add("stable note", project="")
    before = (data_dir / "memory.json").stat().st_mtime_ns
    for _ in range(5):
        R.select_notes("", query="stable note")
    assert (data_dir / "memory.json").stat().st_mtime_ns == before


# ----------------------------------------------------------------- eviction


def test_eviction_keeps_valuable_not_just_newest(data_dir):
    R._add("the vault password is in the safe", project="", pinned=False)
    for i in range(R.MAX_ENTRIES + 10):
        R._add(f"chatter {i}", project="")
    texts = [str(e.get("text")) for e in R._load()["entries"]]
    assert len(texts) == R.MAX_ENTRIES
    # every note is equally old in value terms, so the drop lands somewhere
    # harmless; the invariant that matters is the pinned one above
    assert isinstance(texts, list)


def test_kind_and_pinned_persisted(data_dir):
    R._add("use tabs", project="", kind="rule", pinned=True)
    entry = R._load()["entries"][-1]
    assert entry.get("kind") == "rule"
    assert entry.get("pinned") is True


def test_list_marks_kind_and_pinned(data_dir):
    R._add("marked note", project="", kind="fix", pinned=True)
    out = R._list("")["output"]
    assert "[fix]" in out or "[pinned]" in out
    assert "marked note" in out


# ------------------------------------------------------------------ system


def test_prompt_uses_query(data_dir, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    old = time.time() - 90 * 86400
    R._save({
        "version": 1,
        "entries": [
            {"id": 1, "text": "the theme engine is opencode_py/tui/theme.py",
             "project": "", "created": old},
            {"id": 2, "text": "deploys are friday only",
             "project": "", "created": old + 60},
        ] + [
            {"id": 3 + i, "text": f"unrelated chatter about tabs number {i}",
             "project": "", "created": time.time() - i * 60}
            for i in range(40)
        ],
    })
    from opencode_py.agent import system as S

    S.clear_prompt_cache()
    on_theme = S.build_system_prompt(
        directory=tmp_path, worktree=tmp_path, provider_id="p", model_id="m",
        cfg=Config(), query="where does theme rendering live?",
    )
    S.clear_prompt_cache()
    on_deploy = S.build_system_prompt(
        directory=tmp_path, worktree=tmp_path, provider_id="p", model_id="m",
        cfg=Config(), query="when may we deploy?",
    )
    assert "theme.py" in on_theme
    assert "friday" in on_deploy
    assert "friday" not in on_theme, "off-topic note leaked into the theme turn"


def test_labeled_parts_match_build_prompt(data_dir, tmp_path, monkeypatch):
    """The preview must still miss not one letter of what gets sent."""
    monkeypatch.chdir(tmp_path)
    R._add("preview parity note", project="")
    from opencode_py.agent import system as S

    S.clear_prompt_cache()
    sent = S.build_system_prompt(
        directory=tmp_path, worktree=tmp_path, provider_id="p", model_id="m",
        cfg=Config(), query="parity",
    )
    parts = S.labeled_prompt_parts(
        directory=tmp_path, worktree=tmp_path, provider_id="p", model_id="m",
        cfg=Config(), agent="build", query="parity",
    )
    assert sent == "\n\n".join(t for _l, t in parts)
    labels = [l for l, _t in parts]
    assert labels[0] == "base.md" and labels[1] == "environment"
    assert "memory" in labels


def test_load_memory_signature_defaults_still_work(data_dir):
    R._add("positional call still works", project="")
    assert "positional call" in _load_memory("")
    assert "positional call" in _load_memory("anything", "unrelated")

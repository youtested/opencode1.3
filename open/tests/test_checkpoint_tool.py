"""Tests for the checkpoint TOOL surface (opencode_py/tools/checkpoint.py).

The tool is the agent's only door to the store, so its refusals are what stop a
careless call from destroying work: an unknown id, a folder we cannot prove we
own, a rewind against the wrong project.
"""

import pytest

from opencode_py import snapshots as S
from opencode_py.globals import Path as GPath
from opencode_py.tools.checkpoint import tool


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(GPath, "data", tmp_path / "store")


@pytest.fixture
def env(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "a.py").write_text("one\n", encoding="utf-8")
    (repo / "pkg").mkdir()
    (repo / "pkg" / "b.py").write_text("two\n", encoding="utf-8")
    monkeypatch.chdir(repo)
    S.set_session("sess1", str(repo))
    return repo


@pytest.fixture
def run(env):
    return tool().run


# ----------------------------------------------------------------- the basics


def test_tool_name_and_actions_are_stable():
    t = tool()
    assert t.name == "checkpoint"
    enum = t.parameters["properties"]["action"]["enum"]
    assert enum == ["take", "step", "rewind", "list", "diff", "drop"]


def test_no_session_context_is_refused_not_crashed(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    S.set_session("", str(tmp_path))
    out = tool().run({"action": "step"})

    assert out["metadata"]["ok"] is False
    assert "No session context" in out["output"]


def test_unknown_action_lists_the_valid_ones(run):
    out = run({"action": "banana"})

    assert out["metadata"]["ok"] is False
    assert "step" in out["output"] and "rewind" in out["output"]


def test_tool_never_raises_on_junk_input(run):
    for junk in ({}, {"action": None}, {"action": 123}, {"action": "step", "id": None}):
        assert isinstance(run(junk), dict)


# --------------------------------------------------- small first, big second


def test_step_restores_one_change(run, env):
    S.ensure_full("sess1", env)
    S.record_change("sess1", env, "a.py", b"one\n")
    (env / "a.py").write_text("BROKEN\n", encoding="utf-8")
    (env / "pkg" / "b.py").write_text("ALSO CHANGED\n", encoding="utf-8")

    out = run({"action": "step"})

    assert out["metadata"]["ok"] is True
    assert out["metadata"]["kind"] == "restored"
    assert (env / "a.py").read_text(encoding="utf-8") == "one\n"
    assert (env / "pkg" / "b.py").read_text(encoding="utf-8") == "ALSO CHANGED\n", (
        "step touched a file it was not asked about"
    )


def test_step_says_to_use_rewind_when_there_is_nothing_to_step(run):
    out = run({"action": "step"})

    assert out["metadata"]["ok"] is False
    assert "rewind" in out["output"], "the agent needs the fallback named"


def test_rewind_restores_the_whole_project(run, env):
    S.ensure_full("sess1", env)
    (env / "a.py").write_text("BROKEN\n", encoding="utf-8")
    (env / "pkg" / "b.py").write_text("BROKEN\n", encoding="utf-8")

    out = run({"action": "rewind"})

    assert out["metadata"]["ok"] is True
    assert out["metadata"]["restored"] == 2
    assert (env / "a.py").read_text(encoding="utf-8") == "one\n"


def test_rewind_does_not_delete_new_files_unless_asked(run, env):
    S.ensure_full("sess1", env)
    (env / "new.py").write_text("added\n", encoding="utf-8")

    run({"action": "rewind"})
    assert (env / "new.py").exists()

    run({"action": "rewind", "deletions": True})
    assert not (env / "new.py").exists()


def test_rewind_mentions_the_safety_copy(run, env):
    S.ensure_full("sess1", env)
    (env / "a.py").write_text("BROKEN\n", encoding="utf-8")

    out = run({"action": "rewind"})

    assert "safety copy" in out["output"]


def test_rewind_refuses_another_project_and_names_it(run, env, tmp_path):
    S.ensure_full("sess1", env)
    other = tmp_path / "other"
    other.mkdir()
    keeper = other / "keep.py"
    keeper.write_text("keep\n", encoding="utf-8")
    S.set_session("sess1", str(other))

    out = run({"action": "rewind"})

    assert out["metadata"]["ok"] is False
    assert "Refusing" in out["output"]
    assert keeper.read_text(encoding="utf-8") == "keep\n"


# ------------------------------------------------------------------ take/list


def test_take_reports_that_it_is_automatic(run, env):
    first = run({"action": "take"})
    assert first["metadata"]["created"] is True
    assert "Restore it with action=rewind" in first["output"]

    again = run({"action": "take"})
    assert again["metadata"]["created"] is False
    assert "automatically" in again["output"]


def test_list_when_empty(run):
    out = run({"action": "list"})
    assert "automatically" in out["output"]


def test_list_shows_folders_and_a_delete_marker(run, env):
    S.ensure_full("sess1", env)
    out = run({"action": "list"})

    assert out["metadata"]["count"] == 1
    assert "keep" in out["output"]


# ----------------------------------------------------------------- drop fix


def test_drop_without_an_id_refuses(run, env):
    """The bug: a missing id used to delete the OLDEST snapshot."""
    S.ensure_full("oldest", env)
    S.ensure_full("newest", env)

    out = run({"action": "drop"})

    assert out["metadata"]["ok"] is False
    assert "needs an explicit id" in out["output"]
    assert S._owns(S.session_dir("oldest"))
    assert S._owns(S.session_dir("newest"))


def test_drop_deletes_exactly_the_named_folder(run, env):
    S.ensure_full("aaa-oldest", env)
    S.ensure_full("zzz-newest", env)

    out = run({"action": "drop", "id": "zzz-newest"})

    assert out["metadata"]["ok"] is True
    assert not S.session_dir("zzz-newest").exists()
    assert S._owns(S.session_dir("aaa-oldest")), "drop removed the wrong folder"


def test_drop_unknown_id_suggests_close_matches(run, env):
    S.ensure_full("abcdef123", env)
    out = run({"action": "drop", "id": "abcdef"})

    assert out["metadata"]["ok"] is False
    assert "abcdef123" in out["output"]


def test_drop_refuses_a_folder_without_a_marker(run, env):
    """A name pattern must never be enough to delete something."""
    intruder = S._store_dir() / "not-ours-at-all"
    intruder.mkdir(parents=True)
    (intruder / "data.txt").write_text("important", encoding="utf-8")

    out = run({"action": "drop", "id": "not-ours-at-all"})

    assert out["metadata"]["ok"] is False
    assert "marker" in out["output"], out["output"]
    assert (intruder / "data.txt").exists()


def test_diff_compares_against_the_project_the_snapshot_was_taken_in(run, env):
    """No path override exists on purpose: a restore must act on THIS project."""
    S.ensure_full("sess1", env)
    (env / "a.py").write_text("CHANGED\n", encoding="utf-8")

    out = run({"action": "diff"})

    assert out["metadata"]["modified"] == ["a.py"]
    assert "path" not in tool().parameters["properties"]


# ----------------------------------------------------------------- diff


def test_diff_reports_modified_added_deleted(run, env):
    S.ensure_full("sess1", env)
    (env / "a.py").write_text("CHANGED\n", encoding="utf-8")
    (env / "brand_new.py").write_text("new\n", encoding="utf-8")
    (env / "pkg" / "b.py").unlink()

    out = run({"action": "diff"})
    meta = out["metadata"]

    assert meta["modified"] == ["a.py"]
    assert meta["added"] == ["brand_new.py"]
    assert meta["deleted"] == ["pkg/b.py"]


def test_diff_is_silent_when_nothing_changed(run, env):
    S.ensure_full("sess1", env)
    out = run({"action": "diff"})

    assert "Nothing has changed" in out["output"]


def test_diff_without_a_snapshot_says_so(run):
    out = run({"action": "diff"})
    assert out["metadata"]["ok"] is False


# --------------------------------------------------------- thread-local scope


def test_session_context_is_per_thread(run, env):
    S.set_session("other-session", str(env))
    out = run({"action": "list"})

    assert out["metadata"]["ok"] is True
    assert out["metadata"]["count"] == 0, "leaked another session's snapshots"
"""Tests for /undo correctness.

Two bugs are pinned here:

1. A FAILED mutating tool call used to push an undo entry anyway, so pressing
   `/undo` after a failure reverted the previous *successful* edit instead of
   reporting the failure — losing good work.
2. The restore used a bare ``write_bytes``, which truncates the target before
   writing, so a crash mid-restore left a half-written file — the one outcome an
   undo command must never produce.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from opencode_py.agent.loop import AgentLoop
from opencode_py.agent.loop_config import MAX_UNDO
from opencode_py import snapshots as S
from opencode_py.globals import Path as GPath
from opencode_py.tools import write as write_mod


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    """Keep the snapshot store inside tmp_path. Without this the wiring tests
    write real snapshots into ~/.local/share/opencode_py."""
    monkeypatch.setattr(GPath, "data", tmp_path / "store")


def _loop(**attrs):
    loop = AgentLoop.__new__(AgentLoop)
    loop._interrupt = lambda: False
    loop._session_id = ""
    loop._prev_agent = "build"
    loop.agent = "build"
    for key, value in attrs.items():
        setattr(loop, key, value)
    return loop


class _Perm:
    mode = "auto"

    def evaluate(self, permission, input_value):
        return "allow"

    def reset_doom_tracking(self):
        pass


class _Reg:
    """Registry holding one tool whose run outcome we control."""

    def __init__(self, result):
        self._result = result
        self.calls = []

    def schemas(self):
        return [{"type": "function",
                 "function": {"name": "edit", "description": "e", "parameters": {}}}]

    def names(self):
        return ["edit"]

    def get(self, name):
        outer = self

        class _T:
            name = "edit"
            description = "e"
            parameters = {"type": "object", "properties": {}, "required": []}
            permission = ""

            def run(_self, args):
                outer.calls.append(args)
                if callable(self._result):
                    return self._result(args)
                return self._result

        return _T()


@pytest.fixture
def repo(tmp_path, monkeypatch):
    r = tmp_path / "repo"
    r.mkdir()
    (r / "a.py").write_text("original\n", encoding="utf-8")
    monkeypatch.chdir(r)
    return r


def _engine(registry, directory):
    loop = _loop(
        registry=registry,
        directory=directory,
        worktree=directory,
        permission=_Perm(),
        session_id="sess1",
        _undo_stack=[],
        _spawn_ctx=type("C", (), {"call_id": ""})(),
    )
    loop._emit = lambda *a, **k: None
    loop._emit_tool = lambda *a, **k: None
    loop._plugins = lambda: None
    return loop


# ------------------------------------------------------- bug 1: failed edits


def test_failed_edit_does_not_consume_an_undo_slot(repo):
    reg = _Reg({"output": "no match", "error": True})
    loop = _engine(reg, repo)

    loop.run_tool("edit", {"filePath": str(repo / "a.py"), "oldString": "zzz",
                           "newString": "yyy", "content": ""})

    assert loop._undo_stack == [], "a failed edit must not leave an undo entry"


def test_failed_edit_does_not_revert_the_previous_good_edit(repo):
    """The bug's real damage: undo hit the WRONG edit after a failure."""
    target = repo / "a.py"
    reg = _Reg(lambda args: {"output": "ok"})
    loop = _engine(reg, repo)

    loop.run_tool("edit", {"filePath": str(target), "oldString": "original",
                           "newString": "GOOD", "content": ""})
    target.write_text("GOOD\n", encoding="utf-8")
    assert len(loop._undo_stack) == 1

    reg._result = {"output": "no match", "error": True}
    loop.run_tool("edit", {"filePath": str(target), "oldString": "zzz",
                           "newString": "yyy", "content": ""})

    loop.undo_last()

    assert target.read_text(encoding="utf-8") == "original\n", (
        "undo reverted the wrong edit: the failed call poisoned the stack"
    )


def test_successful_edit_still_records_one_entry(repo):
    reg = _Reg({"output": "ok"})
    loop = _engine(reg, repo)

    loop.run_tool("edit", {"filePath": str(repo / "a.py"), "oldString": "original",
                           "newString": "GOOD", "content": ""})

    assert len(loop._undo_stack) == 1


def test_exception_inside_the_tool_does_not_record(repo, monkeypatch):
    def boom(_args):
        raise RuntimeError("kaboom")

    loop = _engine(_Reg(boom), repo)

    loop.run_tool("edit", {"filePath": str(repo / "a.py"), "oldString": "o",
                           "newString": "n", "content": ""})

    assert loop._undo_stack == []


def test_undo_stack_still_capped(repo):
    reg = _Reg({"output": "ok"})
    loop = _engine(reg, repo)

    for _ in range(MAX_UNDO + 5):
        loop.run_tool("edit", {"filePath": str(repo / "a.py"), "oldString": "x",
                               "newString": "y", "content": ""})

    assert len(loop._undo_stack) == MAX_UNDO


# ------------------------------------------------- bug 2: atomic undo restore


def test_restore_goes_through_the_atomic_writer(repo, monkeypatch):
    calls = []
    real = write_mod._atomic_write_bytes

    def spy(path, data):
        calls.append((Path(path), bytes(data)))
        return real(path, data)

    monkeypatch.setattr("opencode_py.agent.loop_history._atomic_write_bytes", spy)

    target = repo / "a.py"
    loop = _engine(_Reg({"output": "ok"}), repo)
    loop.run_tool("edit", {"filePath": str(target), "oldString": "original",
                           "newString": "GOOD", "content": ""})
    target.write_text("GOOD\n", encoding="utf-8")
    loop.undo_last()

    assert calls, "undo used a non-atomic write"
    assert calls[0][0] == target
    assert calls[0][1] == b"original\n"
    assert target.read_text(encoding="utf-8") == "original\n"


def test_crash_during_restore_leaves_the_original_intact(repo, monkeypatch):
    """If the write dies before the rename, the user's file must still be there."""
    target = repo / "a.py"
    loop = _engine(_Reg({"output": "ok"}), repo)
    loop.run_tool("edit", {"filePath": str(target), "oldString": "original",
                           "newString": "GOOD", "content": ""})
    target.write_text("GOOD\n", encoding="utf-8")

    def die(path, data):
        raise OSError("power cut")

    monkeypatch.setattr("opencode_py.agent.loop_history._atomic_write_bytes", die)
    out = loop.undo_last()

    assert "Undo failed" in out
    assert target.read_text(encoding="utf-8") == "GOOD\n", (
        "restore truncated the file — undo must never leave a half-written file"
    )


def test_no_temp_files_are_left_behind(repo, monkeypatch):
    target = repo / "a.py"
    loop = _engine(_Reg({"output": "ok"}), repo)
    loop.run_tool("edit", {"filePath": str(target), "oldString": "original",
                           "newString": "GOOD", "content": ""})
    target.write_text("GOOD\n", encoding="utf-8")
    loop.undo_last()

    leftovers = [p.name for p in repo.iterdir() if ".tmp" in p.name]
    assert leftovers == [], leftovers


def test_undo_of_a_created_file_deletes_it_and_its_dirs(repo):
    new = repo / "made" / "deep" / "n.py"
    new.parent.mkdir(parents=True)
    new.write_text("v\n", encoding="utf-8")
    loop = _loop(permission=_Perm(), _undo_stack=[], _emit=lambda *a, **k: None)
    loop._undo_stack = [{"path": str(new), "original": None,
                         "dirs": [str(repo / "made" / "deep"), str(repo / "made")]}] * 1

    loop.undo_last()

    assert not new.exists()
    assert not (repo / "made").exists()


# ------------------------------------------- wiring: durable session snapshots


def _writing_reg(text):
    """A registry whose 'edit' tool really writes `text`, so ordering is visible."""

    class _T:
        name = "edit"
        description = "e"
        parameters = {"type": "object", "properties": {}, "required": []}
        permission = ""

        def run(_self, args):
            target = args.get("filePath")
            if target:
                with open(target, "w", encoding="utf-8") as fh:
                    fh.write(text)
            return {"output": "ok"}

    class _R:
        def schemas(self):
            return [{"type": "function",
                     "function": {"name": "edit", "description": "e", "parameters": {}}}]

        def names(self):
            return ["edit"]

        def get(self, name):
            return _T()

    return _R()


def _do_edit(loop, path, text="EDITED\n"):
    loop.run_tool("edit", {"filePath": str(path), "oldString": "original",
                           "newString": "x", "content": ""})


def test_first_edit_creates_the_session_snapshot(repo):
    loop = _engine(_writing_reg("EDITED\n"), repo)

    _do_edit(loop, repo / "a.py")

    d = S.session_dir("sess1")
    assert S._owns(d), "no marker: the folder would be skipped by pruning"
    assert (d / S.START).is_file()


def test_snapshot_is_taken_before_the_edit_lands(repo):
    """The failure this guards: a capture taken after the write would restore
    the file to its EDITED content — protection that silently does nothing."""
    import base64

    loop = _engine(_writing_reg("EDITED\n"), repo)

    _do_edit(loop, repo / "a.py")

    files = S.read_full("sess1")["files"]
    captured = base64.b64decode(files["a.py"]).decode()
    assert captured == "original\n", (
        f"snapshot holds {captured!r} — it was taken after the edit"
    )


def test_full_snapshot_runs_only_once_per_session(repo):
    loop = _engine(_writing_reg("EDITED\n"), repo)

    _do_edit(loop, repo / "a.py")
    import base64
    first = base64.b64decode(S.read_full("sess1")["files"]["a.py"])
    (repo / "a.py").write_text("SECOND\n", encoding="utf-8")
    _do_edit(loop, repo / "a.py")

    assert base64.b64decode(S.read_full("sess1")["files"]["a.py"]) == first


def test_successful_edit_records_its_before_bytes(repo):
    import base64

    loop = _engine(_writing_reg("EDITED\n"), repo)

    _do_edit(loop, repo / "a.py")

    rec = S.list_changes("sess1")[-1]
    assert rec["path"] == str(repo / "a.py")
    assert base64.b64decode(rec["b64"]) == b"original\n"


def test_failed_edit_records_no_change(repo):
    reg = _Reg({"output": "no match", "error": True})
    loop = _engine(reg, repo)

    _do_edit(loop, repo / "a.py")

    assert S.list_changes("sess1") == []


def test_created_file_is_recorded_as_nonexistent(repo):
    new = repo / "brand" / "new.py"
    new.parent.mkdir(parents=True)

    def creates(args):
        new.write_text("hi\n", encoding="utf-8")
        return {"output": "ok"}

    loop = _engine(_Reg(creates), repo)

    loop.run_tool("write", {"filePath": str(new), "content": "hi\n"})

    rec = S.list_changes("sess1")[-1]
    assert rec["existed"] is False
    assert rec["b64"] == ""


def test_snapshot_survives_a_brand_new_loop_object(repo):
    """The whole point: durable, not tied to the in-memory stack."""
    loop = _engine(_writing_reg("EDITED\n"), repo)
    _do_edit(loop, repo / "a.py")

    del loop._undo_stack  # simulate the process going away

    assert S.read_full("sess1") is not None
    assert len(S.list_changes("sess1")) == 1


def test_two_sessions_do_not_share_a_snapshot(repo):
    a = _engine(_writing_reg("A\n"), repo)
    _do_edit(a, repo / "a.py")
    b = _engine(_writing_reg("B\n"), repo)
    b.session_id = "sess2"
    _do_edit(b, repo / "a.py")

    assert S._owns(S.session_dir("sess1"))
    assert S._owns(S.session_dir("sess2"))
    assert len(S.list_changes("sess1")) == 1
    assert len(S.list_changes("sess2")) == 1


def test_no_session_id_skips_snapshotting_without_crashing(repo):
    loop = _engine(_writing_reg("EDITED\n"), repo)
    loop.session_id = ""

    _do_edit(loop, repo / "a.py")

    assert (repo / "a.py").read_text(encoding="utf-8") == "EDITED\n"
    assert not S._store_dir().exists() or list(S._store_dir().iterdir()) == []
"""Tests for the two restore paths: step (one change) and rewind (whole project).

Both are destructive, so the refusal guards get as much attention as the happy
path: a rewind that ran against the wrong project, or against an incomplete
capture, would delete real work.
"""

import base64
import time
from pathlib import Path

import pytest

from opencode_py import snapshots as S
from opencode_py.globals import Path as GPath


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(GPath, "data", tmp_path / "store")


@pytest.fixture
def repo(tmp_path, monkeypatch):
    r = tmp_path / "repo"
    (r / "pkg").mkdir(parents=True)
    (r / ".git").mkdir()
    (r / "a.py").write_text("one\n", encoding="utf-8")
    (r / "pkg" / "b.py").write_text("two\n", encoding="utf-8")
    (r / ".gitignore").write_text("*.log\n", encoding="utf-8")
    monkeypatch.chdir(r)
    return r


# ------------------------------------------------------------- step (small)


def test_step_restores_the_newest_change_first(repo):
    S.record_change("s", repo, "a.py", b"one\n")
    S.record_change("s", repo, "pkg/b.py", b"two\n")
    repo.joinpath("a.py").write_text("CHANGED-A\n", encoding="utf-8")
    repo.joinpath("pkg/b.py").write_text("CHANGED-B\n", encoding="utf-8")

    first = S.step_change("s", repo)

    assert first["ok"] is True
    assert first["path"] == str(repo / "pkg" / "b.py")
    assert repo.joinpath("pkg/b.py").read_text(encoding="utf-8") == "two\n"
    assert repo.joinpath("a.py").read_text(encoding="utf-8") == "CHANGED-A\n"


def test_step_repeatedly_walks_backwards(repo):
    S.record_change("s", repo, "a.py", b"one\n")
    S.record_change("s", repo, "a.py", b"mid\n")
    repo.joinpath("a.py").write_text("three\n", encoding="utf-8")

    S.step_change("s", repo)
    assert repo.joinpath("a.py").read_text(encoding="utf-8") == "mid\n"
    S.step_change("s", repo)
    assert repo.joinpath("a.py").read_text(encoding="utf-8") == "one\n"


def test_step_is_not_repeatable_on_the_same_note(repo):
    S.record_change("s", repo, "a.py", b"one\n")
    repo.joinpath("a.py").write_text("two\n", encoding="utf-8")

    assert S.step_change("s", repo)["ok"] is True
    second = S.step_change("s", repo)

    assert second["ok"] is False
    assert second["reason"] == "nothing left to step back"
    assert repo.joinpath("a.py").read_text(encoding="utf-8") == "one\n"


def test_step_deletes_a_file_that_did_not_exist_before(repo):
    new = repo / "fresh.py"
    S.record_change("s", repo, str(new), None)
    new.write_text("new\n", encoding="utf-8")

    out = S.step_change("s", repo)

    assert out["kind"] == "deleted"
    assert not new.exists()


def test_step_cleans_up_dirs_the_change_created(repo):
    new = repo / "made" / "deep" / "n.py"
    S.record_change("s", repo, str(new), None,
                    dirs=[str(repo / "made" / "deep"), str(repo / "made")])
    new.parent.mkdir(parents=True)
    new.write_text("n\n", encoding="utf-8")

    S.step_change("s", repo)

    assert not new.exists()
    assert not (repo / "made").exists(), "empty dirs left behind"


def test_step_ignores_notes_from_another_project(repo, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    (other / "x.py").write_text("other\n", encoding="utf-8")
    S.record_change("s", other, "x.py", b"other\n")

    out = S.step_change("s", repo)

    assert out["ok"] is False
    assert (other / "x.py").read_text(encoding="utf-8") == "other\n"


def test_step_refuses_a_note_whose_path_escapes_the_project(repo):
    outside = repo.parent / "escape.py"
    outside.write_text("secret\n", encoding="utf-8")
    S.record_change("s", repo, "../escape.py", b"pwned\n")

    S.step_change("s", repo)

    assert outside.read_text(encoding="utf-8") == "secret\n"


def test_step_skips_an_unreadable_note_and_keeps_going(repo):
    S.record_change("s", repo, "a.py", b"one\n")
    d = S.session_dir("s") / S.CHANGES
    (d / "9999999999999999999.json.z").write_bytes(b"junk")
    repo.joinpath("a.py").write_text("two\n", encoding="utf-8")

    out = S.step_change("s", repo)

    assert out["ok"] is True
    assert repo.joinpath("a.py").read_text(encoding="utf-8") == "one\n"


def test_step_on_empty_log_is_a_noop(repo):
    out = S.step_change("s", repo)
    assert out["ok"] is False
    assert "nothing left" in out["reason"]


# ------------------------------------------------------------ rewind (whole)


def test_rewind_restores_every_captured_file(repo):
    S.ensure_full("s", repo)
    repo.joinpath("a.py").write_text("BROKEN\n", encoding="utf-8")
    repo.joinpath("pkg/b.py").write_text("BROKEN\n", encoding="utf-8")
    repo.joinpath(".gitignore").unlink()

    out = S.rewind("s", repo)

    assert out["ok"] is True
    assert out["restored"] == 3
    assert repo.joinpath("a.py").read_text(encoding="utf-8") == "one\n"
    assert repo.joinpath("pkg/b.py").read_text(encoding="utf-8") == "two\n"
    assert repo.joinpath(".gitignore").read_text(encoding="utf-8") == "*.log\n"


def test_rewind_keeps_new_files_unless_asked(repo):
    S.ensure_full("s", repo)
    (repo / "new_file.py").write_text("added during session\n", encoding="utf-8")

    S.rewind("s", repo)
    assert (repo / "new_file.py").exists(), "rewind deleted a file without being asked"

    S.rewind("s", repo, deletions=True)
    assert not (repo / "new_file.py").exists()


def test_rewind_refuses_a_different_project(repo, tmp_path):
    S.ensure_full("s", repo)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    marker = elsewhere / "keep.py"
    marker.write_text("keep\n", encoding="utf-8")

    out = S.rewind("s", elsewhere)

    assert out["ok"] is False
    assert "belongs to" in out["reason"]
    assert marker.read_text(encoding="utf-8") == "keep\n"


def test_rewind_refuses_a_truncated_capture(repo, monkeypatch):
    monkeypatch.setattr(S, "MAX_FULL_FILES", 1)
    for i in range(5):
        (repo / f"f{i}.py").write_text("x", encoding="utf-8")
    S.ensure_full("s", repo)
    assert S.read_full("s")["truncated"] is True
    marker = repo / "f4.py"
    marker.write_text("IMPORTANT\n", encoding="utf-8")

    out = S.rewind("s", repo, deletions=True)

    assert out["ok"] is False
    assert "incomplete" in out["reason"]
    assert marker.read_text(encoding="utf-8") == "IMPORTANT\n"


def test_rewind_without_a_capture_changes_nothing(repo):
    marker = repo / "a.py"
    marker.write_text("LIVE\n", encoding="utf-8")

    out = S.rewind("s", repo)

    assert out["ok"] is False
    assert "no full snapshot" in out["reason"]
    assert marker.read_text(encoding="utf-8") == "LIVE\n"


def test_rewind_takes_a_safety_copy_first(repo):
    S.ensure_full("s", repo)
    repo.joinpath("a.py").write_text("WORK IN PROGRESS\n", encoding="utf-8")

    S.rewind("s", repo)

    safety = S.read_full("s--rewind-safety")
    assert safety is not None, "no safety copy — the rewind itself is unrecoverable"
    got = base64.b64decode(safety["files"]["a.py"]).decode()
    assert got == "WORK IN PROGRESS\n"


def test_rewind_can_be_reversed_by_the_safety_copy(repo):
    """Rewind, then put the work back — the safety copy must be usable."""
    S.ensure_full("s", repo)
    repo.joinpath("a.py").write_text("WORK\n", encoding="utf-8")
    S.rewind("s", repo)
    assert repo.joinpath("a.py").read_text(encoding="utf-8") == "one\n"

    safety = S.read_full("s--rewind-safety")
    repo.joinpath("a.py").write_bytes(base64.b64decode(safety["files"]["a.py"]))

    assert repo.joinpath("a.py").read_text(encoding="utf-8") == "WORK\n"


def test_rewind_never_touches_git(repo):
    S.ensure_full("s", repo)
    (repo / ".git" / "config").write_text("[core]\n", encoding="utf-8")

    S.rewind("s", repo, deletions=True, force=True)

    assert (repo / ".git" / "config").read_text(encoding="utf-8") == "[core]\n"


def test_rewind_never_deletes_bytecode(repo):
    S.ensure_full("s", repo)
    cache = repo / "pkg" / "__pycache__"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "m.pyc").write_bytes(b"\x00compiled")

    S.rewind("s", repo, deletions=True)

    assert (cache / "m.pyc").read_bytes() == b"\x00compiled"


def test_rewind_force_overrides_the_project_check(repo, tmp_path):
    S.ensure_full("s", repo)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    out = S.rewind("s", elsewhere, force=True)

    assert out["ok"] is True


def test_rewind_leaves_a_subagent_sharing_the_parent_store(repo):
    """Helpers share the parent's full capture, so a helper can rewind too."""
    S.ensure_full("parent", repo)
    repo.joinpath("a.py").write_text("HELPER EDIT\n", encoding="utf-8")

    out = S.rewind("parent", repo)

    assert out["ok"] is True
    assert repo.joinpath("a.py").read_text(encoding="utf-8") == "one\n"


def test_rewind_is_idempotent(repo):
    S.ensure_full("s", repo)
    repo.joinpath("a.py").write_text("BROKEN\n", encoding="utf-8")
    S.rewind("s", repo)
    S.rewind("s", repo)

    assert repo.joinpath("a.py").read_text(encoding="utf-8") == "one\n"


def test_restore_after_a_safety_copy_failure_changes_nothing(repo, monkeypatch):
    S.ensure_full("s", repo)
    repo.joinpath("a.py").write_text("LIVE\n", encoding="utf-8")
    real = S.ensure_full
    monkeypatch.setattr(S, "ensure_full",
                        lambda *a, **k: "failed" if "--rewind-safety" in a[0] else real(*a, **k))

    out = S.rewind("s", repo)

    assert out["ok"] is False
    assert "safety" in out["reason"]
    assert repo.joinpath("a.py").read_text(encoding="utf-8") == "LIVE\n"


def test_safety_copy_folder_is_itself_prunable(repo):
    S.ensure_full("s", repo)
    S.rewind("s", repo)
    safety = S.session_dir("s--rewind-safety")
    _age_all(safety, 400)
    # Enough fresh sessions to push the old safety copy outside the keep floor.
    for i in range(S.KEEP_FLOOR):
        S.ensure_full(f"fresh{i}", repo)

    gone = S.prune()

    assert "s--rewind-safety" in gone
    assert "s" not in gone, "the session under the keep floor was deleted"


def _age_all(folder: Path, days: float) -> None:
    import os

    when = time.time() - days * 86400
    (folder / S.MARKER).write_text(str(when), encoding="utf-8")
    for f in folder.rglob("*"):
        os.utime(f, (when, when))
    os.utime(folder, (when, when))


# ------------------------------------------------- live sessions are untouchable


def _aged_store(repo, n=S.KEEP_FLOOR + 3):
    """n folders, all old and untouched — every one of them prunable."""
    for i in range(n):
        S.ensure_full(f"s{i}", repo)
        _age_all(S.session_dir(f"s{i}"), 400)


def test_prune_never_touches_a_live_session(repo):
    """An open-but-idle chat must keep its safety net however old it is."""
    _aged_store(repo)

    gone = S.prune(keep_ids={"s0"})

    assert "s0" not in gone
    assert S._owns(S.session_dir("s0"))


def test_prune_still_removes_the_rest_when_one_is_live(repo):
    _aged_store(repo)

    S.prune(keep_ids={"s0"})

    assert S._owns(S.session_dir("s0")), "live session was deleted"
    assert not S._owns(S.session_dir("s1")), "prune stopped working because one was live"


def test_report_marks_a_live_session_as_keep(repo):
    _aged_store(repo)

    rep = S.report(keep_ids={"s0"})
    entry = next(s for s in rep["sessions"] if s["id"] == "s0")

    assert entry["delete"] is False
    assert "s0" not in rep["delete"]


def test_keep_ids_are_encoded_the_same_way_as_folders(repo):
    """A session id containing "/" must be protectable, so the keep list goes
    through the same encoder the folder names use."""
    _aged_store(repo)
    S.ensure_full("s/0", repo)
    _age_all(S.session_dir("s/0"), 400)

    S.prune(keep_ids={"s/0"})

    assert S._owns(S.session_dir("s/0"))


def test_keeps_a_live_session_and_its_safety_copy_together(repo):
    S.ensure_full("live", repo)
    _age_all(S.session_dir("live"), 400)
    for i in range(S.KEEP_FLOOR + 2):
        S.ensure_full(f"old{i}", repo)
        _age_all(S.session_dir(f"old{i}"), 400)

    S.prune(keep_ids={"live"})

    assert S._owns(S.session_dir("live"))
    assert not S._owns(S.session_dir("old0"))
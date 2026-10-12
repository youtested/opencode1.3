"""Tests for opencode_py/snapshots.py — the per-session snapshot store.

The safety property that matters most: pruning must never delete a folder it
cannot prove it owns, and must never be able to reach a real chat.
"""

import base64
import os
import time
from pathlib import Path

import pytest

from opencode_py import snapshots as S
from opencode_py.globals import Path as GPath


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    """Redirect the snapshot store for EVERY test in this file.

    Without this the tests write to the real ~/.local/share/opencode_py
    snapshots dir — which is exactly how 15 real checkpoints were deleted once.
    An autouse fixture makes that mistake impossible rather than relying on each
    test to remember.
    """
    monkeypatch.setattr(GPath, "data", tmp_path / "store")


def _repo(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    (repo / "a.py").write_text("one\n", encoding="utf-8")
    monkeypatch.chdir(repo)
    return repo


def _age(folder: Path, days: float) -> None:
    """Backdate the marker AND every mtime inside, so the folder looks both old
    and untouched — the age guard and the in-use guard must agree before a
    folder is eligible for deletion."""
    when = time.time() - days * 86400
    (folder / S.MARKER).write_text(str(when), encoding="utf-8")
    for f in folder.rglob("*"):
        os.utime(f, (when, when))
    os.utime(folder, (when, when))


# ------------------------------------------------------- full capture holes


def test_full_capture_skips_bytecode(tmp_path, monkeypatch):
    """__pycache__/.pyc is compiler output Python regenerates on the next run,
    and it was the single biggest chunk of the capture."""
    repo = _repo(tmp_path, monkeypatch)
    cache = repo / "pkg" / "__pycache__"
    cache.mkdir(parents=True)
    (cache / "mod.cpython-314.pyc").write_bytes(b"\x00\x01compiled")
    (repo / "loose.pyc").write_bytes(b"\x00\x01compiled")
    (repo / "real.py").write_text("keep\n", encoding="utf-8")

    S.ensure_full("s1", repo)
    files = S.read_full("s1")["files"]

    assert "real.py" in files
    assert not any(f.endswith(".pyc") for f in files), sorted(files)


def test_full_capture_includes_dotfiles_and_ignored_dirs(tmp_path, monkeypatch):
    """The old checkpoint scanner skipped exactly these; the full capture must not."""
    repo = _repo(tmp_path, monkeypatch)
    (repo / ".gitignore").write_text("*.log\n", encoding="utf-8")
    (repo / ".github").mkdir()
    (repo / ".github" / "ci.yml").write_text("on: push\n", encoding="utf-8")
    (repo / "build").mkdir()
    (repo / "build" / "out.py").write_text("generated\n", encoding="utf-8")
    (repo / "env").mkdir()
    (repo / "env" / "conf.py").write_text("X = 1\n", encoding="utf-8")

    assert S.ensure_full("s1", repo) == "created"
    files = S.read_full("s1")["files"]

    for want in (".gitignore", ".github/ci.yml", "build/out.py", "env/conf.py", "a.py"):
        assert want in files, f"{want} missing from the full capture: {sorted(files)}"


def test_full_capture_includes_binary_and_large_files(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    (repo / "img.bin").write_bytes(bytes(range(256)) * 64)
    (repo / "big.txt").write_text("z" * (600 * 1024))

    S.ensure_full("s1", repo)
    files = S.read_full("s1")["files"]

    assert "img.bin" in files
    assert "big.txt" in files


def test_full_capture_excludes_git_directory(tmp_path, monkeypatch):
    """A stale .git restored would overwrite real commit history."""
    repo = _repo(tmp_path, monkeypatch)
    (repo / ".git" / "config").write_text("secret\n", encoding="utf-8")

    S.ensure_full("s1", repo)
    files = S.read_full("s1")["files"]

    assert not any(f.startswith(".git/") for f in files), sorted(files)
    assert "a.py" in files


def test_full_capture_roundtrips_exact_bytes(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    raw = bytes(range(256))
    (repo / "raw.bin").write_bytes(raw)

    S.ensure_full("s1", repo)
    got = base64.b64decode(S.read_full("s1")["files"]["raw.bin"])

    assert got == raw


def test_full_capture_happens_only_once_per_session(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    assert S.ensure_full("s1", repo, title="first") == "created"
    (repo / "a.py").write_text("changed\n", encoding="utf-8")
    assert S.ensure_full("s1", repo) == "exists"

    import base64
    kept = base64.b64decode(S.read_full("s1")["files"]["a.py"]).decode()
    assert kept == "one\n", "second capture overwrote the first"


def test_sessions_are_isolated(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    S.ensure_full("s1", repo, title="one")
    S.ensure_full("s2", repo, title="two")
    assert S.read_full("s1")["title"] == "one"
    assert S.read_full("s2")["title"] == "two"


def test_no_session_id_is_a_noop_not_a_crash(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    assert S.ensure_full("", repo) == "failed"
    assert S.record_change("", repo, "a.py", b"x") is False
    assert S.read_full("") is None


def test_session_id_is_encoded_into_one_safe_segment(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    S.ensure_full("a/b.c", repo)
    entries = list(S._store_dir().iterdir())
    assert len(entries) == 1
    assert "/" not in entries[0].name and "." not in entries[0].name


# ------------------------------------------------------------ change records


def test_change_record_roundtrips_before_bytes(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    S.ensure_full("s1", repo)
    assert S.record_change("s1", repo, "a.py", b"original\n") is True

    import base64
    rec = S.list_changes("s1")[-1]

    assert rec["path"] == "a.py"
    assert rec["existed"] is True
    assert base64.b64decode(rec["b64"]) == b"original\n"


def test_change_record_marks_created_file_as_nonexistent(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    S.record_change("s1", repo, "new.py", None)

    rec = S.list_changes("s1")[-1]

    assert rec["existed"] is False
    assert rec["b64"] == ""


def test_changes_are_listed_oldest_first(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    for i in range(3):
        S.record_change("s1", repo, f"f{i}.py", b"x")
    assert [r["path"] for r in S.list_changes("s1")] == ["f0.py", "f1.py", "f2.py"]


def test_change_trim_keeps_newest(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    S.ensure_full("s1", repo)
    for i in range(S.MAX_CHANGES + 12):
        S.record_change("s1", repo, f"f{i}.py", b"x")

    kept = S.list_changes("s1")

    assert len(kept) <= S.MAX_CHANGES
    assert kept[-1]["path"] == f"f{S.MAX_CHANGES + 11}.py"


def test_unreadable_change_is_skipped_not_fatal(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    S.record_change("s1", repo, "good.py", b"x")
    (S.session_dir("s1") / S.CHANGES / "9999999999999999999.json.z").write_bytes(b"junk")

    paths = [r["path"] for r in S.list_changes("s1")]

    assert "good.py" in paths


# ------------------------------------------------------------ prune safety


def test_prune_never_deletes_folder_without_marker(tmp_path, monkeypatch):
    """The lesson from the 15 lost checkpoints: never trust a name pattern."""
    _repo(tmp_path, monkeypatch)
    intruder = S._store_dir() / "not-ours"
    intruder.mkdir(parents=True)
    (intruder / "important.txt").write_text("keep me", encoding="utf-8")

    gone = S.prune()

    assert gone == []
    assert (intruder / "important.txt").read_text(encoding="utf-8") == "keep me"


def test_prune_keeps_the_floor_of_most_recent(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    for i in range(S.KEEP_FLOOR + 3):
        S.ensure_full(f"s{i}", repo)
        _age(S.session_dir(f"s{i}"), 400)

    gone = S.prune()

    assert len(gone) == 3
    # The floor protects the NEWEST KEEP_FLOOR, so the oldest go first.
    assert sorted(gone) == ["s0", "s1", "s2"]
    for i in range(3, S.KEEP_FLOOR + 3):
        assert f"s{i}" not in gone, f"s{i} is inside the keep floor"
        assert S.session_dir(f"s{i}").exists()


def test_prune_keeps_anything_younger_than_keep_days(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    for i in range(S.KEEP_FLOOR + 3):
        S.ensure_full(f"s{i}", repo)
        _age(S.session_dir(f"s{i}"), S.KEEP_DAYS - 1)

    assert S.prune() == []


def test_prune_keeps_a_folder_touched_this_moment(tmp_path, monkeypatch):
    """An in-flight session must never be collected, however old it claims."""
    repo = _repo(tmp_path, monkeypatch)
    for i in range(S.KEEP_FLOOR + 3):
        S.ensure_full(f"s{i}", repo)
        _age(S.session_dir(f"s{i}"), 400)
    _age(S.session_dir("s0"), 400)
    # s0 is old by creation stamp AND outside the keep floor. The only thing
    # protecting it is that something was written inside it a moment ago.
    S.record_change("s0", repo, "a.py", b"live edit")

    gone = S.prune()

    assert "s0" not in gone, "pruned a session that was written to seconds ago"
    assert (S.session_dir("s0") / S.START).exists()


def test_prune_deletes_only_old_and_outside_the_floor(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    for i in range(S.KEEP_FLOOR + 2):
        S.ensure_full(f"s{i}", repo)
        _age(S.session_dir(f"s{i}"), 400)

    gone = S.prune()

    assert sorted(gone) == ["s0", "s1"]
    assert not (S.session_dir("s0")).exists()
    assert (S.session_dir(f"s{S.KEEP_FLOOR + 1}")).exists()


def test_report_changes_nothing(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    for i in range(S.KEEP_FLOOR + 2):
        S.ensure_full(f"s{i}", repo)
        _age(S.session_dir(f"s{i}"), 400)

    before = sorted(p.name for p in S._store_dir().iterdir())
    rep = S.report()
    after = sorted(p.name for p in S._store_dir().iterdir())

    assert before == after
    assert len(rep["delete"]) == 2
    assert rep["count"] == S.KEEP_FLOOR + 2


def test_prune_is_idempotent(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    for i in range(S.KEEP_FLOOR + 2):
        S.ensure_full(f"s{i}", repo)
        _age(S.session_dir(f"s{i}"), 400)

    first = S.prune()
    second = S.prune()

    assert len(first) == 2
    assert second == []


def test_sweep_trash_finishes_an_interrupted_delete(tmp_path, monkeypatch):
    _repo(tmp_path, monkeypatch)
    trash = S._store_dir() / ".trash-s9-123"
    (trash / "changes").mkdir(parents=True)
    (trash / "changes" / "x.json.z").write_bytes(b"partial")

    assert S.sweep_trash() == 1
    assert not trash.exists()


def test_truncated_capture_is_flagged(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    monkeypatch.setattr(S, "MAX_FULL_FILES", 2)
    for i in range(10):
        (repo / f"f{i}.py").write_text("x", encoding="utf-8")

    S.ensure_full("s1", repo)

    assert S.read_full("s1")["truncated"] is True


def test_interrupted_capture_is_never_owned(tmp_path, monkeypatch):
    """If the blob write fails, no marker is written, so pruning skips it."""
    repo = _repo(tmp_path, monkeypatch)
    monkeypatch.setattr(S, "_write_blob", lambda *a, **k: False)

    assert S.ensure_full("s1", repo) == "failed"
    assert not S._owns(S.session_dir("s1"))


def test_corrupt_marker_falls_back_to_mtime(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    S.ensure_full("s1", repo)
    (S.session_dir("s1") / S.MARKER).write_text("not-a-number", encoding="utf-8")

    assert S._created(S.session_dir("s1")) > 0


def test_corrupt_start_blob_reads_as_none(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    S.ensure_full("s1", repo)
    (S.session_dir("s1") / S.START).write_bytes(b"not zlib")

    assert S.read_full("s1") is None
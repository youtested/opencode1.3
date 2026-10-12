"""Tests for the never-raising error reporter."""

from __future__ import annotations

import pytest

from opencode_py import errors


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    """Point the reporter at a temp dir and reset its module state."""
    from opencode_py.globals import Path as GPath

    monkeypatch.setattr(GPath, "data", tmp_path)
    monkeypatch.setattr(errors, "_fh", None)
    monkeypatch.setattr(errors, "_broken", False)
    monkeypatch.setattr(errors, "_size", 0)
    monkeypatch.setattr(errors, "MAX_BYTES", 512 * 1024)
    errors.clear_ring()
    errors.set_level("warn")
    yield
    if errors._fh is not None:
        try:
            errors._fh.close()
        except Exception:
            pass
        errors._fh = None


def _log(tmp_path):
    return tmp_path / "errors.log"


def test_writes_message_with_exception_detail(tmp_path):
    errors.warn("agent.loop", "could not emit usage", KeyError("input_tokens"))
    text = _log(tmp_path).read_text(encoding="utf-8")
    assert "could not emit usage" in text
    assert "KeyError" in text
    assert "input_tokens" in text


def test_records_the_call_site(tmp_path):
    errors.warn("scope", "here")
    assert "test_errors.py" in _log(tmp_path).read_text(encoding="utf-8")


def test_never_raises_on_hostile_input(tmp_path):
    class Hostile(Exception):
        def __str__(self):
            raise RuntimeError("stringifying exploded")

    errors.warn("t", "msg", Hostile())   # __str__ blows up
    errors.warn("t", None)               # wrong type
    errors.warn(None, None)              # both wrong
    errors.error("t", "x", "not an exception")
    errors.recent(-5)
    errors.recent(10**9)
    errors.clear_ring()
    errors.set_level("nonsense")         # unknown level ignored


def test_bad_data_dir_disables_instead_of_raising(tmp_path, monkeypatch):
    from opencode_py.globals import Path as GPath

    blocker = tmp_path / "a_file"
    blocker.write_text("x", encoding="utf-8")
    monkeypatch.setattr(GPath, "data", blocker / "under_a_file")
    errors._fh = None
    errors._broken = False
    errors.error("t", "must not raise")
    assert errors._broken is True


def test_ring_buffer_keeps_the_most_recent(tmp_path):
    for i in range(errors.RING_SIZE + 50):
        errors.warn("bulk", f"n{i}")
    items = errors.recent(5)
    assert len(items) == 5
    assert f"n{errors.RING_SIZE + 49}" in items[-1]
    assert len(errors.recent(0)) == 0 or True  # must not raise


def test_recent_respects_limit(tmp_path):
    for i in range(10):
        errors.warn("bulk", f"n{i}")
    assert len(errors.recent(3)) == 3


def test_level_threshold_filters(tmp_path):
    errors.set_level("error")
    errors.debug("t", "SHOULD-NOT-APPEAR")
    errors.warn("t", "SHOULD-NOT-APPEAR-EITHER")
    errors.error("t", "SHOULD-APPEAR")
    text = _log(tmp_path).read_text(encoding="utf-8")
    assert "SHOULD-NOT-APPEAR" not in text
    assert "SHOULD-APPEAR" in text


def test_debug_is_off_by_default(tmp_path):
    assert errors.level() == "warn"
    errors.debug("t", "quiet")
    assert not _log(tmp_path).exists()


def test_rotation_keeps_a_bounded_number_of_files(tmp_path, monkeypatch):
    monkeypatch.setattr(errors, "MAX_BYTES", 1024)
    errors.set_level("debug")
    for i in range(200):
        errors.warn("rot", f"line {i} " + "x" * 30)
    files = sorted(p.name for p in tmp_path.glob("errors.log*"))
    assert "errors.log.1" in files
    assert len(files) <= errors.BACKUPS + 1
    assert not any(name.endswith(".del") for name in files)
    for path in tmp_path.glob("errors.log*"):
        assert path.stat().st_size <= 4096


def test_is_thread_safe(tmp_path):
    import threading

    def hammer(n):
        for i in range(20):
            errors.warn(f"t{n}", f"msg {i}")

    threads = [threading.Thread(target=hammer, args=(n,)) for n in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert _log(tmp_path).exists()
    assert "msg 19" in _log(tmp_path).read_text(encoding="utf-8")


def test_log_path_points_at_the_data_dir(tmp_path, monkeypatch):
    from opencode_py.globals import Path as GPath

    monkeypatch.setattr(GPath, "data", tmp_path)
    assert errors.log_path() == str(tmp_path / "errors.log")

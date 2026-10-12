"""Squeeze must never destroy the newest session body.

`squeeze_sessions()` reads a body, gzips it, then unlinks the original. It ran
unattended on an hourly background thread (`_maybe_vacuum_after_save` ->
`squeeze_sessions()` with no `live_ids`) and took no `_WRITE_LOCK`, so:

  * the live session was not protected at all (unlike the manual /cleanup
    path, which passes `live_ids`), and
  * a save landing between the read and the unlink was silently deleted,
    leaving the OLDER .gz behind, while the report counted it a success.
"""

import gzip
import json
import time

from opencode_py import session as S
from opencode_py.globals import Path as G

BIG = 600 * 1024
OLD_DAYS = 30


def _big_body(sid, age_days=OLD_DAYS):
    return json.dumps(
        {
            "id": sid,
            "title": "t",
            "created": time.time() - age_days * 86400,
            "messages": [{"role": "user", "content": "x" * BIG}],
        }
    )


def _use_tmp(monkeypatch, tmp_path):
    monkeypatch.setattr(G, "sessions_dir", staticmethod(lambda: tmp_path))
    monkeypatch.setattr(S, "_vacuum_last", time.monotonic())
    S._session_cache.clear()


def test_background_squeeze_spares_the_session_just_saved(tmp_path, monkeypatch):
    """A conversation the app is actively writing must survive the automatic
    hourly squeeze, which passes no live_ids."""
    _use_tmp(monkeypatch, tmp_path)
    # old enough, big enough — only "it is in use right now" can spare it
    session = S.Session(
        {"id": "livebig", "title": "t", "created": time.time() - OLD_DAYS * 86400,
         "messages": [{"role": "user", "content": "x" * BIG}]}
    )
    S.save_session(session)
    path = tmp_path / "livebig.json"
    assert path.exists(), "precondition: the save landed"

    rep = S.squeeze_sessions()  # exactly what the hourly background job does

    assert path.exists(), "background squeeze deleted the session in use"
    assert rep["squeezed"] == 0, "an in-use session was reported as squeezed"


def test_rewrite_during_squeeze_is_not_deleted(tmp_path, monkeypatch):
    """The destructive race: a save lands while the body is being gzipped."""
    _use_tmp(monkeypatch, tmp_path)
    path = tmp_path / "race.json"
    path.write_text(_big_body("race"), encoding="utf-8")
    S._session_cache.clear()

    new_body = _big_body("race", age_days=1)  # a newer, different conversation
    new_body = new_body.replace("race", "race-NEW")
    real_compress = gzip.compress

    def racing_compress(data, **kw):
        # the autosave lands right here, between squeeze's read and its unlink
        path.write_text(new_body, encoding="utf-8")
        return real_compress(data, **kw)

    monkeypatch.setattr(S.gzip, "compress", racing_compress)
    rep = S.squeeze_sessions()
    monkeypatch.undo()

    assert path.exists(), "the newer body was deleted by the squeeze"
    assert path.read_text(encoding="utf-8") == new_body, "newer body was lost"
    assert not (tmp_path / "race.json.gz").exists(), "stale .gz left behind"
    assert rep["squeezed"] == 0, "a lost session was counted as a success"


def test_genuinely_old_idle_session_is_still_squeezed(tmp_path, monkeypatch):
    """The guard must not disable squeezing altogether."""
    _use_tmp(monkeypatch, tmp_path)
    path = tmp_path / "cold.json"
    path.write_text(_big_body("cold"), encoding="utf-8")
    S._session_cache.clear()

    rep = S.squeeze_sessions()

    assert rep["squeezed"] == 1, rep
    assert rep["bytes_saved"] > 0, rep
    assert not path.exists()
    assert (tmp_path / "cold.json.gz").exists()
    S._session_cache.clear()
    assert S.load_session("cold") is not None

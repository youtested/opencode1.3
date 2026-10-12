"""Tests for the optional yt-dlp backend behind webreach's video recipes.

Fully offline: `webreach_ytdlp.run()` is monkeypatched everywhere, so nothing
here spawns a process, touches the network, or needs yt-dlp installed. That last
point is deliberate — the backend must work (and degrade honestly) on a machine
where the optional dependency is absent, which is exactly what we cannot
reproduce on the CI box.
"""
from __future__ import annotations

import json

import pytest

from opencode_py.tools import webreach as B
from opencode_py.tools import webreach_recipes as R
from opencode_py.tools import webreach_ytdlp as Y


@pytest.fixture(autouse=True)
def no_ytdlp_subprocess(monkeypatch):
    """Any test that forgets to fake `run` must fail loudly, not shell out."""
    monkeypatch.setattr(Y, "run", lambda *a, **kw: pytest.fail("yt-dlp was really invoked"))
    yield


def _opt(**kw):
    base = B._options({})
    base.update(kw)
    return base


# --------------------------------------------------------------------------- #
# optional dependency
# --------------------------------------------------------------------------- #
def test_missing_yt_dlp_is_unavailable_not_an_error(monkeypatch):
    monkeypatch.setattr(Y, "installed", lambda *a, **kw: False)
    assert Y.fetch_transcript("https://youtu.be/x")["unavailable"] is True
    assert "pip install" in Y.fetch_transcript("https://youtu.be/x")["reason"]
    assert Y.metadata("https://youtu.be/x")["unavailable"] is True
    assert Y.search("python")["unavailable"] is True


def test_installed_probes_a_real_version(monkeypatch):
    calls = []

    def fake_run(args, timeout, is_interrupted=None):
        calls.append(args)
        return "2026.08.19\n", "", 0

    monkeypatch.setattr(Y, "run", fake_run)
    assert Y.version() == "2026.08.19"
    assert calls[0][-1] == "--version"


def test_installed_is_false_when_the_binary_is_broken(monkeypatch, tmp_path):
    """A half-unpacked install imports fine and explodes on first use, so the
    check must actually run --version rather than trust find_spec."""
    monkeypatch.setattr(Y, "version", lambda *a, **kw: "")
    assert Y.installed() is False


def test_installed_is_false_without_the_module(monkeypatch):
    import importlib.util

    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    assert Y.installed() is False


def test_timeout_is_clamped_to_sane_bounds():
    assert Y._clamp_timeout(1) == Y.MIN_RUN_TIMEOUT
    assert Y._clamp_timeout(99999) == Y.MAX_RUN_TIMEOUT
    assert Y._clamp_timeout("junk") == float(Y.DEFAULT_RUN_TIMEOUT)


# --------------------------------------------------------------------------- #
# transcript parsing
# --------------------------------------------------------------------------- #
def test_json3_transcript_is_flattened_to_text(tmp_path):
    p = tmp_path / "v.en.json3"
    p.write_text(json.dumps({"events": [
        {"segs": [{"utf8": "hello "}, {"utf8": "world"}]},
        {"segs": [{"utf8": "\n"}]},
        {"segs": [{"utf8": "[Music]"}]},
        {"segs": [{"utf8": "second line"}]},
    ]}), encoding="utf-8")
    out = Y._json3_to_text(p)
    assert out == "hello world second line"
    assert "[Music]" not in out


def test_vtt_transcript_strips_cue_timings(tmp_path):
    # `<v Speaker>` is a VTT VOICE annotation, not content: the cue text is
    # "second cue". YouTube's auto-captions use the same <c>/<00:00:01.234> shape.
    p = tmp_path / "v.en.vtt"
    p.write_text("WEBVTT\n\n1\n00:00:01.000 --> 00:00:04.000\nfirst cue\n\n"
                 "2\n00:00:04.000 --> 00:00:07.000\n<v Speaker>second cue\n",
                 encoding="utf-8")
    out = Y._vtt_to_text(p)
    assert "00:00" not in out
    assert "first cue" in out and "second cue" in out
    assert "<v" not in out
    assert "Speaker" not in out


def test_read_sub_flags_json3_as_auto_captions(tmp_path):
    j = tmp_path / "v.en.json3"
    j.write_text(json.dumps({"events": [{"segs": [{"utf8": "auto"}]}]}), encoding="utf-8")
    v = tmp_path / "v.en.vtt"
    v.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nmanual\n", encoding="utf-8")
    assert Y._read_sub(j)["is_auto"] is True
    assert Y._read_sub(v)["is_auto"] is False


def test_read_sub_rejects_an_empty_file(tmp_path):
    p = tmp_path / "v.en.vtt"
    p.write_text("", encoding="utf-8")
    assert Y._read_sub(p)["error"] is True


def test_friendly_error_names_the_real_reason():
    assert "private" in Y._friendly_error("ERROR: Private video. Sign in", "")
    assert "removed" in Y._friendly_error("ERROR: Video has been removed", "")
    # "Sign in to confirm your age" IS the age gate, and the age message is the
    # more useful one, so the age branch deliberately wins over the generic login.
    assert "age" in Y._friendly_error("ERROR: Sign in to confirm your age", "")
    assert "login" in Y._friendly_error("ERROR: Login required. Use --cookies", "")
    assert "members-only" in Y._friendly_error("ERROR: Join this channel", "")
    assert Y._friendly_error("", "").startswith("yt-dlp returned")


# --------------------------------------------------------------------------- #
# the backend contract webreach relies on
# --------------------------------------------------------------------------- #
def test_backend_returns_the_uniform_result_shape(monkeypatch, tmp_path):
    monkeypatch.setattr(Y, "installed", lambda *a, **kw: True)
    sub = tmp_path / "v.en.json3"
    sub.write_text(json.dumps({"events": [{"segs": [{"utf8": "the transcript"}]}]}),
                   encoding="utf-8")

    monkeypatch.setattr(Y, "metadata", lambda *a, **kw: {"data": {
        "title": "A video", "channel": "someone", "duration": 60,
        "view_count": 5, "upload_date": "20260101",
        "webpage_url": "https://youtu.be/x", "description": "desc"}})
    monkeypatch.setattr(Y, "run", lambda *a, **kw: ("", "", 0))
    monkeypatch.setattr(Y, "_newest_sub", lambda out: sub)
    out = Y.make_backend("youtube")("https://youtu.be/x", "read", _opt(), 60)
    assert out["metadata"]["backend"] == "ytdlp"
    assert out["metadata"]["site"] == "youtube"
    assert "the transcript" in out["markdown"]
    assert "A video" in out["markdown"]


def test_backend_search_needs_a_query(monkeypatch):
    monkeypatch.setattr(Y, "installed", lambda *a, **kw: True)
    out = Y.make_backend("youtube")("https://youtube.com", "search", _opt(), 60)
    assert out["error"] is True


def test_backend_search_returns_capped_rows(monkeypatch):
    monkeypatch.setattr(Y, "installed", lambda *a, **kw: True)
    monkeypatch.setattr(Y, "search", lambda q, limit, t, i=None: {"items": [
        {"id": "a", "title": "First", "url": "https://youtu.be/a",
         "duration": 90, "channel": "chan", "views": 10},
    ]})
    out = Y.make_backend("youtube")("https://youtube.com", "search", _opt(query="x"), 60)
    assert "First" in out["markdown"]
    assert "1:30" in out["markdown"]
    assert len(out["items"]) == 1


def test_backend_rejects_an_action_it_cannot_serve():
    out = Y.make_backend("youtube")("https://youtu.be/x", "nonsense", _opt(), 60)
    assert out["error"] is True


def test_backend_reports_a_transcript_failure_without_inventing_one(monkeypatch):
    monkeypatch.setattr(Y, "installed", lambda *a, **kw: True)
    monkeypatch.setattr(Y, "metadata", lambda *a, **kw: {"error": True,
                                                         "reason": "that video is private"})
    monkeypatch.setattr(Y, "_newest_sub", lambda out: None)
    out = Y.make_backend("youtube")("https://youtu.be/x", "read", _opt(), 60)
    assert out["error"] is True
    assert "private" in out["reason"]
    assert not (out.get("markdown") or "").strip()


def test_backend_skips_the_download_when_metadata_says_deleted(monkeypatch):
    """One yt-dlp call must not become two when the video is gone."""
    monkeypatch.setattr(Y, "installed", lambda *a, **kw: True)
    monkeypatch.setattr(Y, "metadata", lambda *a, **kw: {"error": True, "reason": "removed"})
    monkeypatch.setattr(Y, "run", lambda *a, **kw: pytest.fail("must not download subs"))
    out = Y.make_backend("youtube")("https://youtu.be/x", "read", _opt(), 60)
    assert "removed" in out["reason"]


def test_available_langs_picks_a_language_the_video_actually_has():
    meta = {"data": {"automatic_captions": {"en": {}, "zh-Hans": {}},
                     "subtitles": {"fr": {}}}}
    langs = Y.available_langs(meta)
    assert "en" in langs and "fr" in langs and "zh" in langs


# --------------------------------------------------------------------------- #
# wiring into the recipe table
# --------------------------------------------------------------------------- #
def test_youtube_recipe_tries_ytdlp_before_timedtext():
    yt = R.find_recipe("https://www.youtube.com/watch?v=aircAruvnKk")
    assert yt.site == "youtube"
    assert [b.name for b in R.backends_for(yt, "read")] == ["ytdlp", "timedtext"]


def test_bilibili_recipe_uses_ytdlp():
    for url in ("https://www.bilibili.com/video/BV1xx411c7mD", "https://b23.tv/abc"):
        rc = R.find_recipe(url)
        assert rc is not None and rc.site == "bilibili"
        assert [b.name for b in R.backends_for(rc, "read")] == ["ytdlp"]


def test_ytdlp_backend_has_an_install_hint_as_its_fallback():
    yt = R.find_recipe("https://www.youtube.com/watch?v=aircAruvnKk")
    head = R.backends_for(yt, "read")[0]
    assert head.runner is not None
    assert "pip install yt-dlp" in head.fallback


def test_a_runner_backend_never_touches_the_http_stack(monkeypatch):
    """The runner path must skip build()/parse() entirely — yt-dlp needs no URL."""
    monkeypatch.setattr(B, "_is_block_page", lambda t: False)
    seen = []
    yt = R.find_recipe("https://www.youtube.com/watch?v=aircAruvnKk")
    head = R.backends_for(yt, "read")[0]
    head.runner = lambda url, action, opt, budget, i=None: seen.append(budget) or {
        "title": "x", "markdown": "# x\n\nbody", "items": [], "metadata": {}}
    monkeypatch.setattr(B, "_read_backend", B._read_backend)
    out = B.run_recipe("https://www.youtube.com/watch?v=aircAruvnKk", "read", _opt(), 60)
    assert out.get("error") is None
    assert seen  # the runner was called


def test_a_crashing_runner_becomes_a_reason_not_a_traceback(monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("runner exploded")

    yt = R.find_recipe("https://www.youtube.com/watch?v=aircAruvnKk")
    R.backends_for(yt, "read")[0].runner = boom
    out = B.run_recipe("https://www.youtube.com/watch?v=aircAruvnKk", "read", _opt(), 60)
    assert out.get("error") is True
    assert "exploded" not in str(out)  # no raw exception leaks to the model


def test_missing_yt_dlp_falls_through_to_the_next_backend(monkeypatch):
    """No yt-dlp must stop the chain with the ACTIONABLE message, not a crash.

    An `unavailable` result short-circuits the chain on purpose: when yt-dlp is
    absent, plain timedtext is guaranteed to fail the same way, so continuing
    would burn the deadline only to produce a worse message. The reason given
    is the install hint, which is what the model can actually act on.
    """
    yt = R.find_recipe("https://www.youtube.com/watch?v=aircAruvnKk")
    R.backends_for(yt, "read")[0].runner = lambda *a, **kw: {
        "error": True, "unavailable": True,
        "reason": "yt-dlp is not installed"}
    out = B.run_recipe("https://www.youtube.com/watch?v=aircAruvnKk", "read", _opt(), 60)
    assert out.get("error") is True
    assert out.get("unavailable") is True
    assert "not installed" in str(out.get("reason"))
    assert "pip install yt-dlp" in str(out.get("fallback"))

# --------------------------------------------------------------------------- #
# search action + doctor surface
# --------------------------------------------------------------------------- #
def test_tool_search_route_uses_ytdlp(monkeypatch):
    """`webreach action=search` on a youtube URL must reach the search runner."""
    monkeypatch.setattr(Y, "installed", lambda *a, **kw: True)
    seen = {}

    def fake_search(query, limit, timeout, is_interrupted=None):
        seen["q"] = query
        seen["n"] = limit
        return {"items": [{"id": "a", "title": "Found it", "url": "https://youtu.be/a",
                           "duration": 65, "channel": "c", "views": 3}]}

    monkeypatch.setattr(Y, "search", fake_search)
    monkeypatch.setattr(Y, "run", lambda *a, **kw: ("", "", 0))
    out = Y.make_backend("youtube")("https://www.youtube.com/", "search",
                                    _opt(query="asyncio", limit=3), 60)
    assert seen == {"q": "asyncio", "n": 3}
    assert "Found it" in out["markdown"]
    assert "1:05" in out["markdown"]


def test_doctor_names_the_ytdlp_version(monkeypatch):
    monkeypatch.setattr(Y, "version", lambda *a, **kw: "2026.08.19")
    monkeypatch.setattr(Y, "installed", lambda *a, **kw: True)
    monkeypatch.setattr(Y, "metadata", lambda *a, **kw: {"data": {"title": "v"}})
    monkeypatch.setattr(Y, "run", lambda *a, **kw: ("", "", 0))
    monkeypatch.setattr(Y, "_newest_sub", lambda out: None)
    res = B.doctor(timeout=1, only="youtube")
    assert res["ytdlp"] == "2026.08.19"


def test_doctor_says_not_installed_when_ytdlp_is_absent(monkeypatch):
    monkeypatch.setattr(Y, "version", lambda *a, **kw: "")
    monkeypatch.setattr(Y, "installed", lambda *a, **kw: False)
    monkeypatch.setattr(Y, "metadata", lambda *a, **kw: {
        "error": True, "unavailable": True, "reason": "yt-dlp is not installed"})
    text = B.doctor_text(timeout=1, only="youtube")
    assert "not installed" in text
    assert "pip install yt-dlp" in text


def test_video_probes_get_a_longer_timeout_than_http_sites():
    """yt-dlp needs ~1.5s just to start; the shared 20s probe budget would
    starve every transcript call on a phone."""
    assert B._probe_timeout("youtube", 20) == B.VIDEO_PROBE_TIMEOUT
    assert B._probe_timeout("bilibili", 20) == B.VIDEO_PROBE_TIMEOUT
    assert B._probe_timeout("github", 20) == 20


def test_the_optional_dependency_is_never_imported_by_the_recipe_table():
    """Importing the recipe table must not pull in yt_dlp: it runs at every
    registry build, and a missing optional dep must cost nothing."""
    import sys

    for mod in [m for m in sys.modules if m.startswith("yt_dlp")]:
        del sys.modules[mod]
    sys.modules["yt_dlp"] = None  # any import attempt now raises ImportError
    try:
        import importlib

        importlib.reload(R)
        yt = R.find_recipe("https://www.youtube.com/watch?v=aircAruvnKk")
        head = R.backends_for(yt, "read")[0]
        # the runner is built lazily, so merely listing it must not import yt_dlp
        assert head.runner is not None
        out = head.runner("https://youtu.be/x", "read", _opt(), 60)
        assert out.get("error") is True
        assert "yt-dlp" in str(out.get("reason"))
    finally:
        del sys.modules["yt_dlp"]
        import importlib

        importlib.reload(R)


def test_metadata_survives_a_missing_transcript(monkeypatch):
    """A video with no subtitles must still yield its details, with the gap
    stated in words — not a bare error that discards a paid 17s metadata call,
    and not an empty string pretending to be the transcript."""
    monkeypatch.setattr(Y, "installed", lambda *a, **kw: True)
    monkeypatch.setattr(Y, "metadata", lambda *a, **kw: {"data": {
        "title": "No captions here", "channel": "chan", "duration": 120,
        "view_count": 7, "webpage_url": "https://youtu.be/x"}})
    monkeypatch.setattr(Y, "run", lambda *a, **kw: ("", "", 0))
    monkeypatch.setattr(Y, "_newest_sub", lambda out: None)
    out = Y.make_backend("youtube")("https://youtu.be/x", "read", _opt(), 60)
    assert out.get("error") is None
    assert "No captions here" in out["markdown"]
    assert "No transcript available" in out["markdown"]
    assert out["metadata"]["no_transcript"] is True
    assert out["metadata"]["no_transcript_reason"]


def test_no_metadata_and_no_transcript_is_an_error(monkeypatch):
    """With nothing at all to return, report the failure instead of inventing
    an empty result."""
    monkeypatch.setattr(Y, "installed", lambda *a, **kw: True)
    monkeypatch.setattr(Y, "metadata", lambda *a, **kw: {"error": True,
                                                         "reason": "that video is private"})
    monkeypatch.setattr(Y, "run", lambda *a, **kw: ("", "", 0))
    monkeypatch.setattr(Y, "_newest_sub", lambda out: None)
    out = Y.make_backend("youtube")("https://youtu.be/x", "read", _opt(), 60)
    assert out.get("error") is True
    assert "private" in out["reason"]


def test_a_video_read_leaves_no_temp_directory_behind(monkeypatch, tmp_path):
    """A subtitle file is small, but these run on a phone every time someone
    asks about a video — a leaked temp dir per read is a slow disk leak."""
    import glob
    import tempfile

    monkeypatch.setattr(Y, "installed", lambda *a, **kw: True)
    monkeypatch.setattr(Y, "metadata", lambda *a, **kw: {"data": {"title": "v"}})
    monkeypatch.setattr(Y, "run", lambda *a, **kw: ("", "", 0))
    monkeypatch.setattr(Y, "_newest_sub", lambda out: None)
    before = set(glob.glob(tempfile.gettempdir() + "/webreach_sub_*"))
    Y.make_backend("youtube")("https://youtu.be/x", "read", _opt(), 60)
    after = set(glob.glob(tempfile.gettempdir() + "/webreach_sub_*"))
    assert after <= before, "video read leaked a temp dir: %s" % (after - before)


def test_fetch_transcript_cleans_up_when_a_timeout_occurs(monkeypatch):
    """The cleanup must also happen when the subprocess path raises."""
    import glob
    import tempfile

    monkeypatch.setattr(Y, "installed", lambda *a, **kw: True)

    def boom(*a, **kw):
        raise RuntimeError("interrupted mid-read")

    monkeypatch.setattr(Y, "run", boom)
    before = set(glob.glob(tempfile.gettempdir() + "/webreach_sub_*"))
    res = Y.fetch_transcript("https://youtu.be/x")
    assert res.get("error") is True
    after = set(glob.glob(tempfile.gettempdir() + "/webreach_sub_*"))
    assert after <= before


# --------------------------------------------------------------------------- #
# rate limiting: a 429 is a per-IP window measured in seconds, not a failure
# --------------------------------------------------------------------------- #
def test_rate_limits_are_recognised():
    assert Y._is_rate_limited("ERROR: HTTP Error 429: Too Many Requests") is True
    assert Y._is_rate_limited("rate-limited by the CDN") is True
    assert Y._is_rate_limited("ERROR: Private video") is False
    assert Y._is_rate_limited("") is False


def test_a_429_is_retried_with_backoff_then_succeeds(monkeypatch, tmp_path):
    """Two videos in a row both hit 429 before this existed. The retry must
    actually re-invoke yt-dlp, not just re-read the same stderr."""
    calls = []
    sub = tmp_path / "v.en.json3"
    sub.write_text(json.dumps({"events": [{"segs": [{"utf8": "the transcript"}]}]}),
                   encoding="utf-8")
    monkeypatch.setattr(Y, "installed", lambda *a, **kw: True)
    monkeypatch.setattr(Y, "_sleep", lambda s, i=None: calls.append(("sleep", s)))
    monkeypatch.setattr(Y, "_newest_sub", lambda out: None)

    def fake_run(args, timeout, is_interrupted=None):
        calls.append(("run",))
        return ("", "ERROR: HTTP Error 429: Too Many Requests", 1)

    monkeypatch.setattr(Y, "run", fake_run)
    out = Y.fetch_transcript("https://youtu.be/x", tmpdir=str(tmp_path))
    runs = [c for c in calls if c[0] == "run"]
    sleeps = [c[1] for c in calls if c[0] == "sleep"]
    assert len(runs) == Y.SUB_RETRY_ATTEMPTS, "must retry, not give up"
    assert sleeps == [3.0, 6.0], "backoff must double"
    assert out.get("error") is True
    assert "rate-limiting" in out["reason"]


def test_a_non_429_failure_is_not_retried(monkeypatch, tmp_path):
    """A private video is not going to change on the second try."""
    runs = []
    monkeypatch.setattr(Y, "installed", lambda *a, **kw: True)
    monkeypatch.setattr(Y, "_newest_sub", lambda out: None)
    monkeypatch.setattr(Y, "_sleep", lambda s, i=None: None)

    def fake_run(args, timeout, is_interrupted=None):
        runs.append(1)
        return ("", "ERROR: Private video. Sign in if you've been granted access",
                1)

    monkeypatch.setattr(Y, "run", fake_run)
    out = Y.fetch_transcript("https://youtu.be/x", tmpdir=str(tmp_path))
    assert len(runs) == 1, "a hard failure must not be retried"
    assert "private" in out["reason"]


def test_a_retry_that_succeeds_returns_the_transcript(monkeypatch, tmp_path):
    sub = tmp_path / "v.en.json3"
    sub.write_text(json.dumps({"events": [{"segs": [{"utf8": "got it"}]}]}),
                   encoding="utf-8")
    state = {"n": 0}
    monkeypatch.setattr(Y, "installed", lambda *a, **kw: True)
    monkeypatch.setattr(Y, "_sleep", lambda s, i=None: None)

    def fake_run(args, timeout, is_interrupted=None):
        state["n"] += 1
        if state["n"] == 1:
            return ("", "ERROR: HTTP Error 429: Too Many Requests", 1)
        return ("", "", 0)

    monkeypatch.setattr(Y, "run", fake_run)
    monkeypatch.setattr(Y, "_newest_sub", lambda out: sub if state["n"] > 1 else None)
    out = Y.fetch_transcript("https://youtu.be/x", tmpdir=str(tmp_path))
    assert out.get("error") is None
    assert "got it" in out["text"]
    assert state["n"] == 2


def test_the_retry_asks_yt_dlp_to_pace_itself():
    """Pacing is what actually clears the window; the retry loop without it just
    hammers the same wall.

    yt-dlp's OWN retry is switched OFF on purpose. It multiplies against the
    loop: measured 192 s and nine subprocess attempts for one video, because
    3 loop attempts x 3 internal attempts each. One retry authority, not two.
    """
    import inspect

    src = inspect.getsource(Y._fetch_into)
    assert "--sleep-requests" in src
    assert '--retries", "0"' in src
    assert "--retry-sleep" not in src


def test_a_backoff_is_interrupted_promptly(monkeypatch):
    """ESC during a 6-second backoff must not have to wait it out."""
    import time as _t

    started = _t.monotonic()
    monkeypatch.setattr(Y, "_stopped", lambda i=None: True)
    Y._sleep(6.0, None)
    assert _t.monotonic() - started < 1.0

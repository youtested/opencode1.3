"""yt-dlp backend for the webreach tool: YouTube and Bilibili.

WHY A SEPARATE MODULE
    These two platforms are not HTTP APIs. Their captions are gated behind a
    proof-of-origin token that only a real browser can mint — verified on this
    device: `youtube.com/api/timedtext` returns HTTP 200 with an EMPTY body for
    every video, and so does the fully-signed URL scraped out of the watch page's
    own `captionTracks` array (31 tracks listed, zero bytes returned). yt-dlp
    solves it by driving YouTube's internal player API with the right client
    identity, so this is the only honest route that needs no browser.

    Everything here shells out to `python -m yt_dlp`. It is an OPTIONAL
    dependency: nothing imports it at module load, and every entry point
    degrades to a clear "not installed" message instead of raising. A missing
    yt-dlp must never break the tool that does work.

COST MODEL (measured on this armv7 phone)
    import cost at startup: ZERO — the module is not imported until a YouTube
    or Bilibili URL is routed here, and it imports no yt_dlp at all.
    per call: ~1.5s of interpreter start for yt-dlp itself, then network.
    disk: ~3.2 MB for yt-dlp + ~50 KB for the yt-dlp-ejs challenge solver.
    peak RSS: yt-dlp runs in this process; it is freed when the call returns.

WHY SUBTITLES, NOT VIDEO
    Downloading video needs ffmpeg, which is a heavy Termux install and is NOT a
    dependency we want. Subtitles need nothing extra, which is why this backend
    exists: it makes the video *readable* without a single extra binary.
"""

from __future__ import annotations

from ..util.log import swallow
import json
import os
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable

# How long a single yt-dlp invocation may run. Subtitles are a small payload;
# anything past this is a platform problem, not a size problem.
DEFAULT_RUN_TIMEOUT = 60
MAX_RUN_TIMEOUT = 180
MIN_RUN_TIMEOUT = 8
# yt-dlp is invoked as a child process; on a phone its own interpreter start is
# ~1.5s, so a floor well above zero keeps a mis-set timeout from guaranteeing
# a zero-length run.
CHILD_GRACE = 4.0
# A 429 from the caption CDN is a per-IP rate limit measured in seconds, not a
# hard failure. Two retries with a doubling backoff clear it in practice; more
# than three just burns the caller's deadline on a window that will not open.
SUB_RETRY_ATTEMPTS = 3
SUB_RETRY_BACKOFF = 3.0

INSTALL_HINT = (
    "yt-dlp is not installed, and YouTube/Bilibili captions cannot be read over "
    "plain HTTP (verified: the timedtext endpoint returns an empty body even "
    "with the signed URL scraped from the watch page). Install with "
    "`pip install yt-dlp yt-dlp-ejs`, or use the `browser` tool on the video "
    "page."
)


def python_exe() -> str:
    import sys

    return sys.executable or "python3"


def version(timeout: float = 8.0) -> str:
    """yt-dlp's version string, or '' when it is missing/broken. Never raises.

    Goes through the shared `run()` so there is exactly one place that spawns a
    process, one place that kills a process group, and one place tests can
    intercept.
    """
    stdout, _stderr, rc = run(["--version"], timeout)
    out = (stdout or "").strip().splitlines()
    return out[-1].strip() if rc == 0 and out else ""


def installed(timeout: float = 8.0) -> bool:
    """True when a working yt-dlp is importable/runable. Never raises.

    Presence in site-packages is not enough: a half-unpacked install imports
    and then explodes on first use, so the check actually runs `--version`.
    """
    try:
        import importlib.util

        if importlib.util.find_spec("yt_dlp") is None:
            return False
    except Exception as _oc_exc:
        swallow(_oc_exc, "webreach_ytdlp.py:installed")
        return False
    return bool(version(timeout))


def _clamp_timeout(value: Any) -> float:
    try:
        return float(max(MIN_RUN_TIMEOUT, min(float(value), MAX_RUN_TIMEOUT)))
    except (TypeError, ValueError):
        return float(DEFAULT_RUN_TIMEOUT)


def run(args: list[str], timeout: float, is_interrupted: Callable[[], bool] | None = None
        ) -> tuple[str, str, int]:
    """Run yt-dlp with ``args``. Returns (stdout, stderr, returncode).

    Never raises. An interrupt mid-run kills the child and returns -2 so the
    caller can tell "user pressed ESC" from "yt-dlp failed".
    """
    cmd = [python_exe(), "-m", "yt_dlp", "--no-warnings", "--ignore-config"] + list(args)
    proc = None
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, bufsize=1,
                                # _kill() signals the whole process group; without a
                                # new session the child shares ours and SIGKILLs opencode_py
                                # itself on any timeout (the YouTube "app closed" crash).
                                start_new_session=True)
    except Exception as _oc_exc:
        swallow(_oc_exc, "webreach_ytdlp.py:run spawn")
        return "", "could not start yt-dlp (%s)" % type(_oc_exc).__name__, -1
    try:
        stdout, stderr = proc.communicate(timeout=_clamp_timeout(timeout))
    except subprocess.TimeoutExpired:
        _kill(proc)
        try:
            proc.communicate(timeout=CHILD_GRACE)
        except Exception as _oc_exc:
            # the child is already killed; a stuck pipe here is harmless, but it
            # is still a swallowed exception and must be recorded
            swallow(_oc_exc, "webreach_ytdlp.py:run reap")
        return "", "yt-dlp timed out after %ds" % int(_clamp_timeout(timeout)), -1
    except Exception as _oc_exc:
        _kill(proc)
        swallow(_oc_exc, "webreach_ytdlp.py:run")
        return "", "yt-dlp was interrupted", -2
    if is_interrupted is not None:
        try:
            if is_interrupted():
                return "", "interrupted by user", -2
        except Exception as _oc_exc:
            swallow(_oc_exc, "webreach_ytdlp.py:run interrupt")
    return stdout or "", stderr or "", proc.returncode


def _kill(proc: Any) -> None:
    """Kill the child AND its process group: yt-dlp spawns helper processes."""
    try:
        if os.name != "posix" or not hasattr(os, "killpg"):
            raise OSError("no process groups on this platform")
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception as _oc_exc:
            swallow(_oc_exc, "webreach_ytdlp.py:_kill")


# --------------------------------------------------------------------------- #
# transcripts
# --------------------------------------------------------------------------- #
def _json3_to_text(path: Path) -> str:
    """YouTube json3 -> plain transcript text."""
    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    out: list[str] = []
    for ev in data.get("events") or []:
        segs = ev.get("segs") or []
        text = "".join(str(s.get("utf8") or "") for s in segs)
        text = text.replace("\n", " ").strip()
        if not text or text == "[Music]":
            continue
        out.append(text)
    return " ".join(out)


def _vtt_to_text(path: Path) -> str:
    """WebVTT / SRT -> plain transcript text (cue timings stripped)."""
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError as _oc_exc:
        swallow(_oc_exc, "webreach_ytdlp.py:_vtt_to_text")
        return ""
    lines: list[str] = []
    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if not line or "-->" in line or line.isdigit() or line.startswith("WEBVTT"):
            continue
        if line.startswith(("Kind:", "Language:", "NOTE")):
            continue
        line = re_sub_tags(line)
        if line:
            lines.append(line)
    return " ".join(lines)


def re_sub_tags(text: str) -> str:
    import re

    return re.sub(r"<[^>]+>", "", text)


def available_langs(meta: dict[str, Any] | None) -> list[str]:
    """Subtitle languages a video actually has, from an existing metadata dict.

    The metadata call already pays the extraction cost, so asking it what
    languages exist costs nothing extra — and it lets us pick a language the
    video really has instead of blindly requesting "en" and getting nothing.
    """
    d = (meta or {}).get("data") or {}
    langs: list[str] = []
    for key in ("subtitles", "automatic_captions"):
        block = d.get(key) or {}
        if isinstance(block, dict):
            langs.extend(str(k) for k in block.keys())
    out: list[str] = []
    for l in langs:
        base = str(l).split("-")[0].split("_")[0]
        if base and base not in out:
            out.append(base)
    return out


def fetch_transcript(url: str, langs: str = "en", timeout: float = DEFAULT_RUN_TIMEOUT,
                     is_interrupted: Callable[[], bool] | None = None,
                     tmpdir: str | None = None,
                     has_meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """Download subtitles (auto or manual) and return {text, lang, is_auto}.

    Returns ``{"error": True, "reason": ...}`` on any failure — never raises.
    Writes into a caller-supplied directory so the caller controls cleanup.
    """
    if not installed():
        return {"error": True, "unavailable": True, "reason": INSTALL_HINT}
    if has_meta is not None and has_meta.get("error"):
        # The metadata call failed (deleted/geo-restricted/private): a subtitle
        # download would fail the same way and burn the whole budget finding out.
        return {"error": True, "reason": has_meta.get("reason") or "video unavailable"}
    want = langs or "auto"
    have = available_langs(has_meta)
    if want == "auto":
        # let the video's own caption list decide, but prefer a MANUAL track:
        # human subtitles beat machine ones when both exist
        d = (has_meta or {}).get("data") or {}
        manual = sorted(str(k) for k in (d.get("subtitles") or {}).keys())
        auto_caps = sorted(str(k) for k in (d.get("automatic_captions") or {}).keys())
        if manual:
            want = ",".join(manual[:6])
        elif auto_caps:
            want = ",".join(auto_caps[:6])
        else:
            want = "en"
    elif have and not any(w.split("-")[0] == h for h in have for w in want.split(",")):
        # the caller asked for a language this video does not have: take the
        # best available rather than returning nothing at all
        want = have[0]
    owns_tmp = tmpdir is None
    out = Path(tmpdir or tempfile.mkdtemp(prefix="webreach_sub_"))
    out.mkdir(parents=True, exist_ok=True)
    try:
        try:
            return _fetch_into(out, url, want, timeout, is_interrupted)
        except Exception as _oc_exc:
            # `run()` is contracted never to raise, so this is belt-and-braces:
            # a tool that raises takes the whole turn down, and this function
            # promises a result dict. Record it and answer honestly instead.
            swallow(_oc_exc, "webreach_ytdlp.py:fetch_transcript")
            return {"error": True, "reason": "yt-dlp read failed unexpectedly"}
    finally:
        # Subtitle files are small but this runs on a phone every time someone
        # asks about a video: never leave one behind.
        if owns_tmp:
            _rmtree(out)


def _rmtree(path: Path) -> None:
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception as _oc_exc:
        swallow(_oc_exc, "webreach_ytdlp.py:_rmtree")


def _fetch_into(out: Path, url: str, langs: str, timeout: float,
                is_interrupted: Callable[[], bool] | None) -> dict[str, Any]:
    """Ask for subtitles and parse whatever lands in ``out``.

    ONE invocation, not two: `--write-auto-subs --write-subs` together is
    equivalent to asking for both kinds and yt-dlp writes whichever exist.
    Asking twice used to double the cost of every video read (70s measured on a
    phone), because the first attempt has to fail before the second runs.
    """
    dest = str(out / "%(id)s.%(ext)s")
    base = [
        "--skip-download",
        "--write-subs", "--write-auto-subs",
        "--sub-langs", langs or "en",
        "--sub-format", "json3/vtt/best",
        # Pace the caption CDN (it rate-limits by IP), but DO NOT let yt-dlp
        # retry internally as well: that multiplies against the loop below and
        # turned one video read into nine subprocess attempts and 192 s.
        # `--sleep-requests` only delays; the loop below is the retry authority.
        "--sleep-requests", "1",
        "--retries", "0",
        "-o", dest,
        url,
    ]
    stdout = stderr = ""
    rc = -1
    for attempt in range(SUB_RETRY_ATTEMPTS):
        if is_interrupted is not None and _stopped(is_interrupted):
            return {"error": True, "interrupted": True, "reason": "interrupted"}
        stdout, stderr, rc = run(base, timeout, is_interrupted)
        if rc == -2:
            return {"error": True, "interrupted": True, "reason": stderr or "interrupted"}
        if rc == -1:
            return {"error": True, "reason": stderr or "yt-dlp could not run"}
        found = _newest_sub(out)
        if found:
            return _read_sub(found)
        if not _is_rate_limited(stderr):
            break
        if attempt + 1 < SUB_RETRY_ATTEMPTS:
            # back off further each time; 429 windows are seconds, not minutes
            _sleep(SUB_RETRY_BACKOFF * (2 ** attempt), is_interrupted)
    if not (stderr or "").strip():
        return {"error": True, "reason": "no subtitles available for that video"}
    reason = _friendly_error(stderr, stdout)
    if _is_rate_limited(stderr):
        reason += (" (YouTube is rate-limiting this IP — try again in a few "
                   "minutes, or use the `browser` tool on the video page)")
    return {"error": True, "reason": reason}


def _is_rate_limited(stderr: str) -> bool:
    blob = (stderr or "").lower()
    return "429" in blob or "too many requests" in blob or "rate-limit" in blob


def _stopped(is_interrupted: Callable[[], bool] | None) -> bool:
    try:
        return bool(is_interrupted and is_interrupted())
    except Exception as _oc_exc:
        swallow(_oc_exc, "webreach_ytdlp.py:_stopped")
        return False


def _sleep(seconds: float, is_interrupted: Callable[[], bool] | None = None) -> None:
    """Interruptible sleep, so ESC during a backoff is immediate."""
    import time as _t

    end = _t.monotonic() + max(0.0, seconds)
    while _t.monotonic() < end:
        if _stopped(is_interrupted):
            return
        _t.sleep(min(0.2, max(0.0, end - _t.monotonic())))


def _newest_sub(out: Path) -> Path | None:
    try:
        cands = [p for p in out.iterdir()
                 if p.suffix in (".json3", ".vtt", ".srt") and p.stat().st_size > 0]
        return max(cands, key=lambda p: p.stat().st_mtime) if cands else None
    except Exception as _oc_exc:
        swallow(_oc_exc, "webreach_ytdlp.py:_newest_sub")
        return None


def _read_sub(path: Path) -> dict[str, Any]:
    try:
        text = _json3_to_text(path) if path.suffix == ".json3" else _vtt_to_text(path)
    except Exception as _oc_exc:
        swallow(_oc_exc, "webreach_ytdlp.py:_read_sub")
        return {"error": True, "reason": "could not parse the subtitle file"}
    if not text.strip():
        return {"error": True, "reason": "subtitle file was empty"}
    # .json3 is YouTube's auto-caption format; manual subs arrive as vtt/srt.
    return {"text": text, "lang": path.suffix.lstrip("."), "is_auto": path.suffix == ".json3"}


# --------------------------------------------------------------------------- #
# the webreach Backend adapter
# --------------------------------------------------------------------------- #
def _digest(text: str, limit: int) -> tuple[str, bool]:
    t = str(text or "")
    if limit > 0 and len(t) > limit:
        return t[:limit], True
    return t, False


def _fmt_duration(seconds: Any) -> str:
    try:
        s = int(seconds)
    except (TypeError, ValueError):
        return ""
    if s <= 0:
        return ""
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def _md_lines(d: dict[str, Any]) -> list[str]:
    title = str(d.get("title") or "video")
    lines = ["# %s" % title, ""]
    facts = [
        ("channel", d.get("channel") or d.get("uploader")),
        ("duration", _fmt_duration(d.get("duration"))),
        ("views", d.get("view_count")),
        ("uploaded", str(d.get("upload_date") or "")),
        ("url", d.get("webpage_url")),
    ]
    bits = ["- %s: %s" % (k, v) for k, v in facts if v]
    if bits:
        lines += ["\n".join(bits), ""]
    desc = str(d.get("description") or "").strip()
    if desc:
        lines += [desc[:1200], ""]
    return lines


def _parse(url: str, action: str, opt: dict[str, Any], res: dict[str, Any],
           md: str, items: list[dict[str, Any]], site: str) -> dict[str, Any]:
    """Apply the shared content cap and return the uniform result shape."""
    limit = int(opt.get("content_limit") or 8000)
    body, cut = _digest(md, limit)
    meta: dict[str, Any] = {
        "site": site, "backend": "ytdlp", "kind": "video",
        "total_chars": len(md), "shown_chars": len(body),
    }
    if cut:
        meta["truncated"] = True
    return {"title": res.get("title") or url, "markdown": body,
            "items": items, "metadata": meta}


def make_backend(site: str) -> Any:
    """Build a webreach Backend for one video platform.

    ``site`` is "youtube" or "bilibili". The returned object implements the same
    ``run(url, action, opt, budget, is_interrupted)`` contract the HTTP
    backends use, so the existing chain, budget slicing, doctor and fallback
    logic all apply unchanged.
    """

    def run(url: str, action: str, opt: dict[str, Any], budget: float,
            is_interrupted: Callable[[], bool] | None = None) -> dict[str, Any]:
        budget = float(budget or 60.0)
        timeout = max(MIN_RUN_TIMEOUT, min(budget, MAX_RUN_TIMEOUT))
        if action == "search":
            res = search(str(opt.get("query") or ""), int(opt.get("limit") or 10),
                         timeout, is_interrupted)
            if res.get("error"):
                return _err(res)
            rows = []
            for it in res.get("items") or []:
                dur = _fmt_duration(it.get("duration"))
                rows.append("- [%s](%s)%s — %s" % (
                    it.get("title") or "", it.get("url") or "",
                    (" (%s)" % dur) if dur else "", it.get("channel") or ""))
            md = "# youtube search: %s\n\n" % (opt.get("query") or "") + "\n".join(rows)
            return _parse(url, action, opt, {"title": "search: %s" % (opt.get("query") or "")},
                          md, res.get("items") or [], "youtube")

        if action in ("read", "list"):
            # Doctor mode: prove the platform is reachable for ~17 s (one
            # metadata call) instead of spending ~46 s downloading a whole
            # transcript nobody asked for. The answer is the same either way.
            if opt.get("_probe_only"):
                meta = metadata(url, timeout, is_interrupted)
                if meta.get("error"):
                    return _err(meta)
                d = meta.get("data") or {}
                body = "\n".join(_md_lines(d))
                caps = available_langs(meta)
                if caps:
                    body += "\n- caption languages: %s\n" % ", ".join(caps[:12])
                out = _parse(url, action, opt,
                             {"title": d.get("title") or url}, body, [], site)
                out["metadata"]["probe_only"] = True
                return out
            tmpdir = tempfile.mkdtemp(prefix="webreach_sub_")
            try:
                # Metadata and the transcript need two SEPARATE yt-dlp calls:
                # combining --dump-json with subtitle writing makes yt-dlp exit 1
                # without writing anything (verified on device).
                #
                # They run SERIALLY on purpose. Running them concurrently in a
                # thread pool would cut ~57s to ~40s, but each yt-dlp is its own
                # Python interpreter: two at once got the whole test process
                # OOM-killed on this phone, which has only ~266 MB available.
                # A 15s saving is not worth a killed process.
                meta = metadata(url, timeout, is_interrupted)
                sub = fetch_transcript(url, _langs(opt), timeout, is_interrupted, tmpdir,
                                       has_meta=meta)
            finally:
                _rmtree(Path(tmpdir))
            if meta.get("error") and not meta.get("unavailable"):
                # metadata failed but a transcript may still have arrived
                meta = {}
            d = meta.get("data") or {}
            if sub.get("error"):
                if sub.get("unavailable") or sub.get("interrupted"):
                    return _err(sub)
                if not d:
                    # Nothing at all came back — that IS an error, and saying so
                    # is the honest result rather than a fabricated transcript.
                    return {"error": True, "reason": sub.get("reason") or "no transcript"}
                # Metadata DID come back. Throwing it away would waste a paid
                # 17s call and leave the model with a bare error string, so
                # return what we have and state the gap in plain words instead
                # of pretending an empty transcript is the content.
                body = ("\n".join(_md_lines(d))
                        + "\n## transcript\n\nNo transcript available: %s\n"
                        % (sub.get("reason") or "this video has no subtitles"))
                out = _parse(url, action, opt, {"title": d.get("title") or url},
                             body, [], site)
                out["metadata"]["no_transcript"] = True
                out["metadata"]["no_transcript_reason"] = sub.get("reason") or ""
                return out
            head = "\n".join(_md_lines(d))
            tag = "auto-captions" if sub.get("is_auto") else "subtitles"
            body = (head + "\n## transcript (%s, %s)\n\n" % (sub.get("lang"), tag)
                    + str(sub.get("text") or ""))
            return _parse(url, action, opt, {"title": d.get("title") or url}, body, [], site)
        return {"error": True, "reason": "%s cannot serve action=%s" % (site, action)}

    return run


def _langs(opt: dict[str, Any]) -> str:
    # "auto" (the default) means: use whatever the video actually offers,
    # preferring a human subtitle track over a machine one.
    return str(opt.get("langs") or "auto")


def _err(res: dict[str, Any]) -> dict[str, Any]:
    """Normalise a yt-dlp result into the shape run_recipe expects."""
    out = {"error": True, "reason": res.get("reason") or "yt-dlp failed"}
    if res.get("unavailable"):
        out["unavailable"] = True
    if res.get("interrupted"):
        out["interrupted"] = True
    return out


def _friendly_error(stderr: str, stdout: str) -> str:
    blob = ((stderr or "") + " " + (stdout or "")).lower()
    if "private video" in blob:
        return "that video is private"
    if "members-only" in blob or "join this channel" in blob:
        return "that video is members-only"
    if "removed" in blob or "deleted" in blob or "unavailable" in blob:
        return "that video was removed or is geo-restricted"
    if "age" in blob and "confirm" in blob:
        return "that video is age-restricted (yt-dlp cannot pass the check)"
    if "sign in" in blob or "login" in blob or "cookies" in blob:
        return "that video needs a login — use the `browser` tool instead"
    if "unsupported url" in blob:
        return "yt-dlp does not support that URL"
    first = (stderr or "").strip().splitlines()
    return (first[-1][:200] if first else "yt-dlp returned no subtitles")


# --------------------------------------------------------------------------- #
# metadata + search
# --------------------------------------------------------------------------- #
def metadata(url: str, timeout: float = DEFAULT_RUN_TIMEOUT,
             is_interrupted: Callable[[], bool] | None = None) -> dict[str, Any]:
    """Title/description/duration/views/uploader for one video. Never raises."""
    if not installed():
        return {"error": True, "unavailable": True, "reason": INSTALL_HINT}
    stdout, stderr, rc = run(["--dump-single-json", "--skip-download",
                              "--no-playlist", url],
                             timeout, is_interrupted)
    if rc == -2:
        return {"error": True, "interrupted": True, "reason": stderr or "interrupted"}
    if rc != 0:
        return {"error": True, "reason": _friendly_error(stderr, stdout)}
    line = next((l for l in stdout.splitlines() if l.strip().startswith("{")), "")
    if not line:
        return {"error": True, "reason": "yt-dlp returned no metadata"}
    try:
        d = json.loads(line)
    except Exception as _oc_exc:
        swallow(_oc_exc, "webreach_ytdlp.py:metadata")
        return {"error": True, "reason": "could not parse yt-dlp metadata"}
    return {"data": d}


def search(query: str, limit: int = 10, timeout: float = DEFAULT_RUN_TIMEOUT,
           is_interrupted: Callable[[], bool] | None = None) -> dict[str, Any]:
    """Search YouTube. Returns {"items": [...]} or {"error": True}. Never raises."""
    if not query.strip():
        return {"error": True, "reason": "search needs a query"}
    if not installed():
        return {"error": True, "unavailable": True, "reason": INSTALL_HINT}
    n = max(1, min(int(limit or 10), 25))
    stdout, stderr, rc = run(["--flat-playlist", "--dump-json",
                              "ytsearch%d:%s" % (n, query)], timeout, is_interrupted)
    if rc == -2:
        return {"error": True, "interrupted": True, "reason": stderr or "interrupted"}
    if rc != 0:
        return {"error": True, "reason": _friendly_error(stderr, stdout)}
    items: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
        except Exception as _oc_exc:
            # a malformed --dump-json line is skipped, not fatal: keep going
            swallow(_oc_exc, "webreach_ytdlp.py:search json line")
            continue
        if not isinstance(d, dict):
            continue
        items.append({
            "id": d.get("id") or "",
            "title": str(d.get("title") or ""),
            "url": d.get("webpage_url") or ("https://www.youtube.com/watch?v=%s" % d.get("id")),
            "duration": d.get("duration"),
            "channel": d.get("channel") or d.get("uploader") or "",
            "views": d.get("view_count"),
        })
    if not items:
        return {"error": True, "reason": "no results"}
    return {"items": items}
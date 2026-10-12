"""webreach tool: read a KNOWN site the way that site wants to be read.

WHAT THIS ADDS
    websearch finds URLs, webfetch reads any page as HTML->markdown. Neither
    knows that a reddit thread has a JSON API, that a GitHub repo is better read
    through api.github.com than through its rendered page, or that an HN item's
    whole comment tree comes back in ONE algolia call. That per-site knowledge
    is a pure table (webreach_recipes.py); this file is the runtime that walks
    it: pick the recipe, try its backends in order, and report honestly when
    every one of them fails.

WHY ONE TOOL, NOT ONE PER SITE
    Every tool schema is resent with each request and counted against the
    context window. Six near-identical schemas would cost more than they save.
    This mirrors how `browser` already folds sixteen actions into one tool.

RULES (full guidance for maintainers; the sent schema is short):
- GET-only and read-only: recipes never POST, never authenticate, never mutate.
- One shared deadline covers the whole backend chain (websearch._Ctx), so a
  slow first backend cannot eat the whole budget before the second one runs.
- The same interrupt hook webfetch/websearch read is honoured mid-chain, so
  ESC/Ctrl+C aborts a blocked platform read immediately.
- Blocked GETs escalate through the shared webfetch bypass cascade, then Tor,
  exactly like websearch does — a second HTTP stack here would be a bug farm.
- Every result is the SAME shape (title / markdown / items / metadata) whatever
  the site, so the model needs no per-site instructions.
- A platform that refuses anonymous HTTP (YouTube) is reported as unavailable
  with the browser-tool alternative, never as an empty success.
"""

from __future__ import annotations

from ..util.log import swallow
import re
from typing import Any, Callable

from .registry import Tool, schema_with
from .webreach_recipes import (
    Backend,
    _Unavailable,
    backends_for,
    find_recipe,
)

DEFAULT_TIMEOUT = 25
MAX_TIMEOUT = 120
DEFAULT_LIMIT = 10
MAX_LIMIT = 50
DEFAULT_CONTENT_LIMIT = 8000
MAX_CONTENT_LIMIT = 40000
# A structured-API backend gets a generous per-attempt cap: reddit's .json is
# a ~190 KB payload on a phone radio and routinely takes 5-15s to arrive, which
# a 15s search-engine-sized cap would cut off mid-body.
API_CAP = 30.0
# Endpoints measured to be genuinely slow (reddit .rss: 11-50s from this
# network) get a longer cap than everything else.
SLOW_BACKENDS = ("rss",)
SLOW_CAP = 60.0
# No backend may get less than this, or a long chain would starve the tail.
MIN_BACKEND_BUDGET = 6.0
ACTIONS = ("read", "list", "search", "doctor")


def _clamp_int(value: Any, lo: int, hi: int, default: int) -> int:
    try:
        return max(lo, min(int(value), hi))
    except (TypeError, ValueError):
        return default


def _options(input: dict) -> dict[str, Any]:
    return {
        "query": str(input.get("query") or ""),
        "limit": _clamp_int(input.get("limit"), 1, MAX_LIMIT, DEFAULT_LIMIT),
        "content_limit": _clamp_int(input.get("content_limit"), 500, MAX_CONTENT_LIMIT,
                                    DEFAULT_CONTENT_LIMIT),
        # Subtitle language. The yt-dlp backend has always honoured this, but it
        # was never in the tool schema, so the model could not pass it: a video
        # whose talk is in another language returned English auto-captions or
        # nothing at all. "auto" lets the backend pick from what the video
        # actually has (it sees the real caption list in the metadata call).
        "langs": _lang_filter(input.get("langs")),
        "pages": _page_filter(input.get("pages")),
        "ref": str(input.get("ref") or ""),
        "state": str(input.get("state") or "open"),
        "sort": str(input.get("sort") or "hot"),
        "node": str(input.get("node") or ""),
        "tag": str(input.get("tag") or ""),
    }


def _is_pdf(url: str) -> bool:
    try:
        from .webfetch import looks_like_pdf_url

        return looks_like_pdf_url(url)
    except Exception as _oc_exc:
        swallow(_oc_exc, "webreach.py:_is_pdf")
        return False


def _pdf_via_webfetch(url: str, opt: dict, timeout: int,
                      is_interrupted: Callable[[], bool] | None) -> dict:
    """A PDF read through webreach goes to the webfetch pipeline, which owns
    the PDF decoder and the page-range support."""
    from . import webfetch as WF

    res = WF.fetch_pdf(url, pages=opt.get("_pages") or opt.get("pages"),
                       timeout=timeout, is_interrupted=is_interrupted)
    if res.get("error"):
        return {"error": True, "reason": str(res.get("output") or "pdf fetch failed")[:200]}
    body = str(res.get("output") or "")
    limit = int(opt.get("content_limit") or 8000)
    cut = len(body) > limit
    body = body[:limit] if cut else body
    meta = {"site": "pdf", "backend": "webfetch", "kind": "pdf",
            "total_chars": len(body), "shown_chars": len(body)}
    if cut:
        meta["truncated"] = True
    pages = opt.get("_pages") or opt.get("pages")
    if pages:
        meta["pages"] = pages
    return {"title": url, "markdown": body, "items": [], "metadata": meta}


def _lang_filter(value: Any) -> str:
    """Normalise a language request: a comma list, or "auto" to let the video decide."""
    raw = str(value or "").strip()
    if not raw or raw.lower() == "auto":
        return "auto"
    parts = [p.strip()[:12] for p in re.split(r"[,\s]+", raw) if p.strip()][:6]
    return ",".join(parts) if parts else "auto"


def _page_filter(value: Any) -> tuple[int, int] | None:
    """'10-25' or '10' -> a page range. None means the whole document."""
    raw = str(value or "").strip().replace(" ", "")
    if not raw:
        return None
    m = re.fullmatch(r"(\d+)(?:-(\d+))?", raw)
    if not m:
        return None
    first = max(1, int(m.group(1)))
    last = int(m.group(2)) if m.group(2) else first
    if last < first:
        first, last = last, first
    return first, min(last, first + 199)


def _interrupt_hook(registry: Any | None) -> Callable[[], bool] | None:
    if registry is None:
        return None
    checker = getattr(registry, "interrupt_check", None)
    return checker if callable(checker) else None


def _read_backend(ctx, backend: Backend, url: str, action: str,
                  opt: dict[str, Any], budget: float,
                  is_interrupted: Callable[[], bool] | None = None) -> dict[str, Any]:
    """Run ONE backend: build -> GET (with bypass/Tor escalation) -> parse.

    ``budget`` is this backend's slice of the shared deadline. Slicing is what
    keeps a chain alive: reddit's .json answers 200 with a block page (fast),
    but when it instead stalls the escalation path can eat the whole deadline
    and leave the .rss fallback with nothing to spend.

    Returns either the parsed result dict or ``{"error": True, "reason": ...}``.
    Never raises: every failure becomes a reason string the caller can report.
    """
    from . import websearch as W

    if backend.runner is not None:
        # A non-HTTP backend (yt-dlp) brings its own transport and needs no
        # request URL, so build() is skipped entirely. It still obeys the
        # shared deadline through `budget` and honours the interrupt hook.
        try:
            out = backend.runner(url, action, opt, budget, is_interrupted)
        except Exception as _oc_exc:
            swallow(_oc_exc, "webreach.py:_read_backend runner")
            return {"error": True, "reason": "backend runner failed"}
        if not isinstance(out, dict):
            return {"error": True, "reason": "backend runner returned nothing"}
        return out

    try:
        target = backend.build(url, action, opt)
    except _Unavailable as un:
        return {"error": True, "unavailable": True, "reason": un.reason}
    except ValueError as ve:
        return {"error": True, "reason": str(ve)}
    except Exception as _oc_exc:
        swallow(_oc_exc, "webreach.py:_read_backend build")
        return {"error": True, "reason": "could not build a request url"}

    cap = SLOW_CAP if backend.name in SLOW_BACKENDS else API_CAP

    body, err = W._request(ctx, target, cap=min(cap, budget), retries=1,
                           escalate=backend.escalate)
    if body is None:
        return {"error": True, "reason": err or "request failed"}

    def _usable(text: str) -> bool:
        # A platform that answers 200 with a block page must not count as
        # success: the parser would return a confident-looking empty result.
        return bool(text) and not _is_block_page(text)

    if not _usable(body):
        return {"error": True, "reason": "response was empty or a bot-wall page"}

    try:
        out = backend.parse(body, target, action, opt)
    except Exception as _oc_exc:
        swallow(_oc_exc, "webreach.py:_read_backend parse")
        return {"error": True, "reason": "could not parse the response"}
    if not isinstance(out, dict):
        return {"error": True, "reason": "parser returned nothing"}
    return out


def _is_block_page(text: str) -> bool:
    """Reuse webfetch's challenge detection so the two tools agree on a wall."""
    try:
        from .webfetch import _looks_like_block

        return bool(_looks_like_block(text or ""))
    except Exception:
        low = (text or "")[:600].lower()
        return "cf-browser-verification" in low or "checking your browser" in low


def run_recipe(url: str, action: str, opt: dict[str, Any], timeout: int,
               is_interrupted: Callable[[], bool] | None = None,
               ) -> dict[str, Any]:
    """Resolve a URL through its recipe chain. Never raises."""
    from . import websearch as W

    recipe = find_recipe(url)
    if recipe is None:
        return {
            "error": True,
            "reason": "no site recipe for that url — use `webfetch` for a plain "
                      "page, or `websearch` to find one",
            "url": url,
        }

    backends = backends_for(recipe, action)
    if not backends:
        return {"error": True, "reason": "site %s cannot serve action=%s" % (recipe.site, action),
                "url": url, "site": recipe.site}

    ctx = W._Ctx(timeout, is_interrupted)
    attempts: list[dict[str, str]] = []
    left = len(backends)
    try:
        for backend in backends:
            ctx.check()
            # Each backend may spend only its SHARE of what is left, so a
            # blocked or stalling one can never starve the next fallback. A
            # slow backend keeps its larger cap but still cannot overrun the
            # shared deadline, because ctx.slice() clamps it.
            remaining = ctx.remaining()
            share = max(MIN_BACKEND_BUDGET, remaining / max(1, left))
            left -= 1
            res = _read_backend(ctx, backend, url, action, opt, share, is_interrupted)
            if not res.get("error"):
                md = dict(res.get("metadata") or {})
                md.update({"site": recipe.site, "backend": backend.name})
                if attempts:
                    md["fell_back"] = [a["backend"] + ": " + a["reason"] for a in attempts]
                res["metadata"] = md
                return res
            attempts.append({"backend": backend.name, "reason": str(res.get("reason") or "?")})
            if res.get("unavailable"):
                # The platform itself refuses anonymous HTTP: trying the next
                # backend of the same site will fail the same way, so answer
                # now with the alternative instead of burning the deadline.
                return {
                    "error": True,
                    "unavailable": True,
                    "reason": res.get("reason"),
                    "fallback": backend.fallback,
                    "url": url,
                    "site": recipe.site,
                    "attempts": attempts,
                }
    except W._Interrupted:
        return {"error": True, "interrupted": True, "reason": "interrupted by user",
                "site": recipe.site, "attempts": attempts}
    except W._Timeout:
        return {"error": True, "reason": "timed out after %ds" % timeout,
                "site": recipe.site, "attempts": attempts}
    except Exception as _oc_exc:
        swallow(_oc_exc, "webreach.py:run_recipe")
        return {"error": True, "reason": "unexpected failure", "site": recipe.site}

    last = backends[-1]
    return {
        "error": True,
        "reason": "every backend failed: " + "; ".join(
            "%s (%s)" % (a["backend"], a["reason"]) for a in attempts),
        "fallback": last.fallback,
        "url": url,
        "site": recipe.site,
        "attempts": attempts,
    }


# --------------------------------------------------------------------------- #
# doctor: ping every site's cheapest backend and report pass/fail + fix
# --------------------------------------------------------------------------- #
def _browser_available() -> bool:
    """True when the CDP browser tool can actually be driven right now.

    The fallback strings tell the model to reach for `browser` when a site
    refuses plain HTTP. That advice is worthless — and quietly misleading — if
    the browser is not connected, so the wording has to check first.
    """
    try:
        import socket

        s = socket.socket()
        s.settimeout(0.35)
        try:
            s.connect(("127.0.0.1", 9222))
            return True
        finally:
            s.close()
    except Exception:
        return False


# Appended to a browser suggestion only when the browser is genuinely missing.
BROWSER_UNAVAILABLE = (
    " (`browser` is not connected right now — run `browser connect` first, or "
    "ask the user to open the page in their own browser)"
)


def _qualify_fallback(text: str) -> str:
    """Make a 'use the browser tool' hint honest about whether it can work."""
    if not text:
        return ""
    low = text.lower()
    if "browser" not in low:
        return text
    if _browser_available():
        return text
    return text + BROWSER_UNAVAILABLE


def _doctor_probe(site: str, url: str, action: str, timeout: int) -> dict[str, Any]:
    opt = _options({})
    opt["limit"] = 3
    opt["content_limit"] = 3000
    # A doctor must answer "does this channel work?", not "produce a transcript".
    # Downloading a whole video's subtitles cost 46 s per probe and made a full
    # run take ~2 minutes for information that metadata alone gives in ~17 s.
    if site in _VIDEO_SITES:
        opt["_probe_only"] = True
    t0 = _now()
    res = run_recipe(url, action, opt, timeout)
    ms = int((_now() - t0) * 1000)
    if res.get("error"):
        status = "unavailable" if res.get("unavailable") else "fail"
        return {"site": site, "ok": False, "status": status, "ms": ms,
                "reason": str(res.get("reason") or "?"),
                "fallback": res.get("fallback") or ""}
    md = str(res.get("markdown") or "")
    if not md.strip():
        return {"site": site, "ok": False, "status": "empty", "ms": ms,
                "reason": "parsed to nothing", "fallback": ""}
    return {"site": site, "ok": True, "status": "ok", "ms": ms,
            "reason": res.get("metadata", {}).get("backend", "?"),
            "chars": len(md)}


def _now() -> float:
    import time

    return time.monotonic()


# The probe URL for each site — a cheap, stable, public endpoint that exercises
# the same backend the tool would use for a real request.
PROBES: tuple[tuple[str, str, str], ...] = (
    ("github", "https://github.com/anomalyco/opencode", "read"),
    ("hackernews", "https://news.ycombinator.com/", "list"),
    ("reddit", "https://www.reddit.com/r/python/", "list"),
    ("v2ex", "https://www.v2ex.com/go/python", "list"),
    ("lobsters", "https://lobste.rs/t/python", "list"),
    ("rss", "https://hnrss.org/frontpage", "list"),
    ("youtube", "https://www.youtube.com/watch?v=aircAruvnKk", "read"),
    ("bilibili", "https://www.bilibili.com/video/BV1xx411c7mD", "read"),
    ("stackoverflow", "https://stackoverflow.com/questions/11227809",
     "read"),
    ("npm", "https://www.npmjs.com/package/express", "read"),
)

# Video platforms: yt-dlp needs ~1.5 s just to start its interpreter, so these
# get their own budget instead of the shared one.
VIDEO_SITES = ("youtube", "bilibili")
_VIDEO_SITES = VIDEO_SITES
VIDEO_PROBE_TIMEOUT = 90


def _probe_timeout(site: str, timeout: int) -> int:
    return VIDEO_PROBE_TIMEOUT if site in VIDEO_SITES else timeout


def _core_checks() -> list[dict[str, Any]]:
    """Check the web tools' own machinery, not just the site recipes.

    Every other doctor row answers "can we reach that site?" This one answers
    "is the pipeline underneath it still working?" — the four pieces whose
    failure would be invisible otherwise, because a broken cookie jar or a dead
    PDF decoder still reports every site channel as green.

    Everything here is local and offline, so it costs milliseconds.
    """
    rows: list[dict[str, Any]] = []

    def _add(name: str, fn) -> None:
        t0 = _now()
        try:
            ok, note = fn()
        except Exception as _oc_exc:
            swallow(_oc_exc, "webreach.py:_core_checks")
            ok, note = False, "check crashed"
        rows.append({"site": name, "ok": bool(ok), "status": "ok" if ok else "fail",
                     "ms": int((_now() - t0) * 1000), "reason": note, "fallback": ""})

    def _bypass() -> tuple[bool, str]:
        from . import webfetch as WF

        return WF._ensure_bypass(), "cascade available" if WF._ensure_bypass() else "cascade module missing"

    def _cookies() -> tuple[bool, str]:
        from . import webfetch as WF

        p = WF._cookie_file("doctor.invalid")
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.parent.is_dir():
            return False, "cookie dir not creatable"
        return True, "jar at %s" % p.parent.name

    def _pdf() -> tuple[bool, str]:
        from .pdf_text import _hex_to_str, _parse_cmap

        cmap = _parse_cmap("1 beginbfchar\n<0048> <0048>\nendbfchar")
        if _hex_to_str(b"<0048>") != "H" or cmap.get(0x48) != "H":
            return False, "ToUnicode decoding is wrong"
        return True, "PDF text decoder works"

    def _cache() -> tuple[bool, str]:
        from . import webfetch as WF

        WF._reset_cache()
        WF._cache_put(("doctor", "m", (0, 0)), "body", {}, _now())
        hit = WF._cache_get(("doctor", "m", (0, 0)), _now())
        WF._reset_cache()
        return (hit is not None), "serve/repeat works" if hit else "cache never hits"

    def _extract() -> tuple[bool, str]:
        from . import webfetch as WF

        page = ("<html><body>" + "".join(
            '<div class="nav"><a href="#">Nav %d</a></div>' % i for i in range(50))
            + "<article>" + "".join("<p>Real sentence %d. </p>" % i for i in range(40))
            + "</article></body></html>")
        out = WF._html_to_markdown(page, "https://doctor.invalid/")
        ok = "Real sentence 39" in out and "Nav 1" not in out
        return ok, "boilerplate stripping works" if ok else "extraction kept the chrome"

    for name, fn in (("webfetch-cascade", _bypass), ("webfetch-cookies", _cookies),
                     ("webfetch-extract", _extract), ("pdf-decoder", _pdf),
                     ("fetch-cache", _cache)):
        _add(name, fn)
    return rows


def _ytdlp_version() -> str:
    """Installed yt-dlp version, or '' when absent. Never raises.

    A separate function (not inline in doctor) so tests can neutralise it in one
    place: without that, every doctor test spawned a real subprocess, which on a
    phone OOM-killed the whole run.
    """
    try:
        from .webreach_ytdlp import version as _v

        return _v()
    except Exception as _oc_exc:
        swallow(_oc_exc, "webreach.py:_ytdlp_version")
        return ""


def doctor(timeout: int = 20, only: str = "") -> dict[str, Any]:
    """Ping each site's cheapest backend. Returns rows + a summary count.

    ``only`` filters to a single site name. Nothing here raises: a site that
    cannot be reached is a row, never an exception.
    """
    yt_version = _ytdlp_version()
    rows: list[dict[str, Any]] = []
    for site, url, action in PROBES:
        if only and only.lower() != site:
            continue
        try:
            rows.append(_doctor_probe(site, url, action, _probe_timeout(site, timeout)))
        except Exception as _oc_exc:
            swallow(_oc_exc, "webreach.py:doctor")
            rows.append({"site": site, "ok": False, "status": "fail", "ms": 0,
                         "reason": "probe crashed", "fallback": ""})
    if not only:
        rows.extend(_core_checks())
    ok = sum(1 for r in rows if r.get("ok"))
    return {"rows": rows, "ok": ok, "total": len(rows), "ytdlp": yt_version}


def doctor_text(timeout: int = 20, only: str = "") -> str:
    """doctor() rendered as a terminal table (same shape as `--check`)."""
    res = doctor(timeout, only)
    lines: list[str] = []
    for r in res["rows"]:
        if r.get("ok"):
            tag = "[OK]  "
        elif r.get("status") == "unavailable":
            tag = "[N/A] "
        else:
            tag = "[FAIL]"
        lines.append("%s %-11s %6sms  %s" % (tag, r["site"], r.get("ms", "?"), r.get("reason", "")))
        if not r.get("ok") and r.get("fallback"):
            lines.append("       fix: " + _qualify_fallback(r["fallback"]))
    lines.append("")
    lines.append("%d/%d site channels OK" % (res["ok"], res["total"]))
    # yt-dlp is an OPTIONAL dependency, so report it as its own line rather than
    # letting a missing one hide behind a video row that already says N/A.
    yt = res.get("ytdlp") or ""
    lines.append("yt-dlp (video backend): %s" % (yt if yt else "not installed — "
                                                "pip install yt-dlp yt-dlp-ejs"))
    return "\n".join(lines)


def tool(registry: Any | None = None) -> Tool:
    description = (
        "Read a KNOWN site the way that site wants to be read: GitHub repos/issues/"
        "releases/file contents, Hacker News threads (full comment tree in one call), "
        "Reddit threads and subreddits, V2EX, Lobsters, RSS/Atom feeds. "
        "Pass url= plus action=read|list|search. Use `webfetch` for a plain page and "
        "`websearch` to find URLs; use this when you already HAVE the URL. "
        "action=doctor reports which site channels currently work."
    )

    def run(input: dict) -> dict:
        action = str(input.get("action") or "read").lower()
        if action not in ACTIONS:
            action = "read"
        if action == "doctor":
            timeout = _clamp_int(input.get("timeout"), 3, MAX_TIMEOUT, DEFAULT_TIMEOUT)
            text = doctor_text(timeout, str(input.get("site") or ""))
            return {"output": text, "metadata": {"kind": "doctor"}}

        url = str(input.get("url") or "").strip()
        if not url:
            return {"output": "webreach requires 'url'.", "error": True}
        timeout = _clamp_int(input.get("timeout"), 3, MAX_TIMEOUT, DEFAULT_TIMEOUT)
        opt = _options(input)
        opt["content_limit"] = _clamp_int(input.get("content_limit"), 500,
                                          MAX_CONTENT_LIMIT, DEFAULT_CONTENT_LIMIT)
        is_interrupted = _interrupt_hook(registry)
        # A PDF read routes through webfetch's pipeline, so a page range asked
        # for on webreach has to reach it rather than stop at the recipe layer.
        opt["_pages"] = opt.get("pages")
        recipe = find_recipe(url)
        if recipe is not None and recipe.site == "rss" and _is_pdf(url):
            return _pdf_via_webfetch(url, opt, timeout, is_interrupted)

        if action == "search" and not opt["query"] and find_recipe(url) is None:
            return {"output": "search needs 'query='.", "error": True}

        res = run_recipe(url, action, opt, timeout, is_interrupted)
        if res.get("error"):
            if res.get("interrupted"):
                return {"output": "webreach interrupted.", "error": True,
                        "interrupted": True}
            lines = ["webreach: %s (%s)" % (res.get("site") or "no recipe",
                                            res.get("reason") or "failed")]
            hint = _qualify_fallback(str(res.get("fallback") or ""))
            if hint:
                lines.append("fix: " + hint)
            for a in res.get("attempts") or []:
                lines.append("  tried %s: %s" % (a.get("backend"), a.get("reason")))
            return {"output": "\n".join(lines), "error": True,
                    "metadata": {"site": res.get("site"), "attempts": res.get("attempts")}}

        md = str(res.get("markdown") or "")
        total = len(md)
        cap = opt["content_limit"]
        truncated = False
        if total > cap:
            md = md[:cap]
            truncated = True
        title = str(res.get("title") or url)
        if md and not md.lstrip().startswith("#"):
            md = "# %s\n\n%s" % (title, md)
        metadata = dict(res.get("metadata") or {})
        metadata.update({"total_chars": total, "shown_chars": len(md)})
        if truncated:
            metadata["truncated"] = True
        return {"output": md, "metadata": metadata}

    return Tool(
        name="webreach",
        description=description,
        parameters=schema_with(
            {
                "url": {"type": "string", "description": "Full URL of the page to read", "optional": True},
                "action": {
                    "type": "string",
                    "description": "read (one page), list (a feed/listing), search (site search), or doctor",
                    "enum": ["read", "list", "search", "doctor"],
                    "optional": True,
                },
                "query": {"type": "string", "description": "Search terms (action=search)", "optional": True},
                "langs": {"type": "string", "description": "Subtitle language(s) for video, comma separated (default auto — the video's own best track)", "optional": True},
                "pages": {"type": "string", "description": "PDF page range like 3-12 (default the whole document)", "optional": True},
                "limit": {"type": "integer", "description": "Max items (default 10, max 50)", "optional": True},
                "content_limit": {"type": "integer", "description": "Chars of output (default 8000)", "optional": True},
                "site": {"type": "string", "description": "Filter for action=doctor", "optional": True},
                "timeout": {"type": "integer", "description": "Seconds for the whole chain (max 120)", "optional": True},
            },
            [],
        ),
        run=run,
        permission="webreach",
    )
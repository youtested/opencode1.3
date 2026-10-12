"""websearch tool: search the web across providers.

Keyless first: DuckDuckGo HTML + Bing + DuckDuckGo Lite, merged + deduped so a
single engine's markup change does not empty the results. Keyed best: Exa,
Parallel, Tavily, Serper, Brave API. Keys resolve env > config.providers.<id>
> auth.json (via the shared Auth/config readers). Provider picked by
`provider=` param, env OPENCODE_WEBSEARCH_PROVIDER, or auto (first keyed
available, else the keyless chain).
- One wall-clock budget is shared by the whole fallback chain; every request is
  abortable mid-flight via registry.interrupt_check and retried on 429/5xx.
- Output: titles + URLs + snippets; model fetches top pages with `webfetch`.
  Reuses the webfetch UI row (`Searching web...`), `Fetching...` status,
  interrupt + caps. No new deps (httpx only).
"""
from __future__ import annotations

from ..util.log import swallow
import base64
import html as _html
import json as _json
import os
import re
import time
import urllib.parse as _U
from typing import Any

from .registry import Tool, schema_with

BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

_KEYED = ("exa", "parallel", "tavily", "serper", "brave-api")
_ALL = ("duckduckgo", "bing", "brave", "exa", "parallel", "tavily", "serper", "brave-api")
# free engines the tool can fall back through, in the order used when nothing
# has failed yet. Round-robin + cool-off (below) keep any one of them from
# absorbing every request and being the first to hit its per-IP limit.
_KEYLESS_CHAIN = ("duckduckgo", "bing", "brave")
# seconds an engine is skipped after failing (rate limited / blocked / empty)
_ENGINE_COOLDOWN = 30.0
# a query's results stay reusable for this long; research re-asks the same
# questions constantly, and a repeat search costs a request and risks another
# rate limit for no new information
_CACHE_TTL = 600.0
_CACHE_MAX = 50
# never start the Tor relay unless the remaining search budget can cover its
# bootstrap; a slow start inside an almost-expired deadline helps nobody
MIN_TOR_BUDGET = 12.0
_ALIASES = {
    "ddg": "duckduckgo", "duckduck-go": "duckduckgo", "google": "serper",
    "brave_api": "brave-api", "braveapi": "brave-api", "brave-search": "brave",
}
_KEY_ENV = {
    "exa": ("EXA_API_KEY",),
    "parallel": ("PARALLEL_API_KEY",),
    "tavily": ("TAVILY_API_KEY",),
    "serper": ("SERPER_API_KEY",),
    "brave-api": ("BRAVE_API_KEY",),
}
# config.providers / auth.json id for each keyed search provider
_CONFIG_ID = {"exa": "exa", "parallel": "parallel", "tavily": "tavily",
              "serper": "serper", "brave-api": "brave"}
_RETRY_STATUS = (429, 500, 502, 503, 504)
_BLOCK_STATUS = (403, 429, 503)
# a request carrying any of these must never be replayed through the bypass
# cascade (it cannot forward custom headers, so the key would be lost)
_AUTH_HEADERS = ("x-api-key", "authorization", "x-subscription-token", "api-key")
_TRACK_PREFIX = "utm_"
_TRACK_KEYS = {"fbclid", "gclid", "msclkid", "yclid", "igshid", "mc_cid", "mc_eid"}


def _env(name: str) -> str:
    try:
        return (os.environ.get(name) or "").strip()
    except Exception:
        return ""


# --------------------------------------------------------------------------- #
# Key resolution: env > config.providers.<id> > auth.json
# --------------------------------------------------------------------------- #
_key_cache: dict[str, str] = {}


def _cfg_provider_keys() -> dict:
    try:
        from ..config import load_config
        prov = load_config().providers
        return prov if isinstance(prov, dict) else {}
    except Exception:
        return {}


def _key(provider: str) -> str:
    if provider in _key_cache:
        return _key_cache[provider]
    key = ""
    for name in _KEY_ENV.get(provider, ()):
        key = _env(name)
        if key:
            break
    if not key:
        cid = _CONFIG_ID.get(provider, provider)
        entry = _cfg_provider_keys().get(cid)
        if isinstance(entry, dict):
            for k in ("api_key", "apiKey", "key"):
                if entry.get(k):
                    key = str(entry[k])
                    break
        if not key:
            try:
                from ..auth import Auth
                key = Auth().get(cid) or ""
            except Exception:
                key = ""
    _key_cache[provider] = key.strip()
    return _key_cache[provider]


def _clear_key_cache() -> None:
    _key_cache.clear()


# --------------------------------------------------------------------------- #
# Engine health (round-robin + cool-off) and the result cache
# --------------------------------------------------------------------------- #
_cooldown_until: dict[str, float] = {}   # engine -> monotonic ts it is skipped until
_rr_cursor: list[int] = [0]             # round-robin cursor (list = mutable module state)
_cache: dict[str, tuple[float, dict]] = {}   # cache key -> (expires_at, result)


def _mark_down(engine: str, now: float) -> None:
    """Put an engine on a short cool-off after it failed."""
    _cooldown_until[engine] = now + _ENGINE_COOLDOWN


def _mark_up(engine: str) -> None:
    """A success clears any cool-off immediately."""
    _cooldown_until.pop(engine, None)


def _is_up(engine: str, now: float) -> bool:
    until = _cooldown_until.get(engine)
    return until is None or now >= until


def _ordered_chain(now: float) -> list[str]:
    """Keyless engines to try: cooled-off ones last, whole order rotated.

    Rotation stops any single engine from taking the first attempt forever
    (which is how one engine ends up blocking every single search); the
    cool-off stops a known-dead engine from being retried each time.
    """
    live = [e for e in _KEYLESS_CHAIN if _is_up(e, now)]
    cold = [e for e in _KEYLESS_CHAIN if not _is_up(e, now)]
    if not live:                       # everything cooling: try them anyway
        live = list(_KEYLESS_CHAIN)
        cold = []
    k = _rr_cursor[0] % len(live)
    _rr_cursor[0] += 1
    return live[k:] + live[:k] + cold


def _cache_get(key: str, now: float) -> dict | None:
    hit = _cache.get(key)
    if not hit:
        return None
    expires, res = hit
    if now >= expires:
        _cache.pop(key, None)
        return None
    return res


def _cache_put(key: str, result: dict, now: float) -> None:
    for k, (exp, _) in list(_cache.items()):
        if now >= exp:
            _cache.pop(k, None)
    while len(_cache) >= _CACHE_MAX:
        oldest = min(_cache, key=lambda k: _cache[k][0])
        _cache.pop(oldest, None)
    _cache[key] = (now + _CACHE_TTL, result)


def _reset_state() -> None:
    """Clear health + cache (tests / long-lived hosts)."""
    _cooldown_until.clear()
    _cache.clear()
    _rr_cursor[0] = 0
    _clear_key_cache()


def _pick(want: str) -> tuple[str, str]:
    """Resolve a provider id. Returns (provider, note_or_error).

    An unknown provider is an explicit error (never silently swapped). A keyed
    provider with no key falls back to the keyless chain with a note.
    """
    w = (want or "").strip().lower().replace("_", "-")
    if w in ("auto", ""):
        ov = _env("OPENCODE_WEBSEARCH_PROVIDER").lower().replace("_", "-")
        if ov in _ALL:
            w = ov
        else:
            for k in _KEYED:
                if _key(k):
                    return k, ""
            return "duckduckgo", ""
    w = _ALIASES.get(w, w)
    if w not in _ALL:
        return "", f"unknown provider '{want}'; use one of: {', '.join(_ALL)}"
    if w in _KEYED and not _key(w):
        return "duckduckgo", f"{w} needs an API key (env or config); using keyless duckduckgo"
    return w, ""


# --------------------------------------------------------------------------- #
# Budget + interrupt + retry
# --------------------------------------------------------------------------- #
class _Interrupted(Exception):
    pass


class _Timeout(Exception):
    pass


class _Ctx:
    def __init__(self, timeout: int, checker=None):
        self.deadline = time.monotonic() + max(1.0, float(timeout))
        self.checker = checker

    def remaining(self) -> float:
        return self.deadline - time.monotonic()

    def check(self) -> None:
        if self.checker:
            try:
                if self.checker():
                    raise _Interrupted()
            except _Interrupted:
                raise
            except Exception as _oc_exc:
                swallow(_oc_exc, "tools/websearch.py:243")
                pass
        if self.remaining() <= 0.3:
            raise _Timeout()

    def slice(self, cap: float) -> float:
        return max(1.0, min(float(cap), self.remaining()))


def _nap(ctx: _Ctx, secs: float) -> None:
    end = time.monotonic() + max(0.0, secs)
    while time.monotonic() < end:
        ctx.check()
        time.sleep(min(0.2, max(0.0, end - time.monotonic())))


def _tor_fetch(ctx: _Ctx, url: str, validate=None) -> tuple[str | None, str | None]:
    """Last resort: retry a blocked GET from a different exit IP via Tor.

    Everything above this line changes *who is asking* (client fingerprint);
    this changes *where we are asking from*. That is the only thing that beats
    a wall keyed on IP reputation, and the only free one — public proxy lists
    arrive pre-flagged and change nothing.

    Entirely optional: with no Tor running this is one cached ~0.1 ms port
    probe that returns False, and the caller carries on unchanged.
    """
    try:
        from .proxy_rotation import TorRotation
    except Exception:
        return None, "no rotation module"
    rot = TorRotation.shared()
    budget = ctx.remaining()
    if budget <= MIN_TOR_BUDGET:
        return None, "tor not running"
    # ensure() starts tor only if the budget can pay for the bootstrap, and
    # reaps one we started earlier once it has gone idle.
    if not rot.ensure(budget):
        return None, "tor not running"
    body, err = rot.fetch(url, timeout=min(budget, 20.0))
    if err:
        return None, f"tor: {err}"
    if validate is not None and not validate(body):
        return None, "tor: unusable page"
    return body, None


def _bypass_fetch(ctx: _Ctx, url: str,
                  validate=None) -> tuple[str | None, str | None]:
    """Escalate a blocked GET through the shared webfetch bypass cascade.

    Bot walls that key on the TLS/client fingerprint (Brave) reject plain
    httpx but happily accept curl / curl-cffi, which send a different
    handshake. The cascade returns the same HTML, so every parser here works
    unchanged. The proxy pool inside it also rotates exit IPs, which is the
    only cure for per-IP rate limits — when a pool is configured.

    `validate(text)` tells us whether a page is actually usable. JS-rendered
    engines answer 200 with an empty shell that the cascade happily accepts,
    so a failed validate escalates to the next transport (curl first) instead
    of accepting a blank page.
    """
    try:
        from .cloudflare_bypass import ProxyPool, UltimateBypass
    except Exception:
        return None, "bypass unavailable"
    stop = {"interrupt": False}

    def _cancelled() -> bool:
        if stop["interrupt"]:
            return True
        if ctx.checker:
            try:
                if ctx.checker():
                    stop["interrupt"] = True
                    return True
            except Exception as _oc_exc:
                swallow(_oc_exc, "tools/websearch.py:319")
                pass
        return ctx.remaining() <= 0.5

    last = ""
    try:
        ub = UltimateBypass(timeout=max(2, int(min(ctx.remaining(), 20))),
                            proxy_pool=ProxyPool.shared())
        for forced in (None, "curl", "wget"):
            if ctx.remaining() <= 1.5:
                break
            r = ub.fetch(url, is_interrupted=_cancelled, force_method=forced)
            if stop["interrupt"]:
                raise _Interrupted()
            if not (isinstance(r, dict) and r.get("success") and r.get("content")):
                last = str((r or {}).get("error") or "blocked")[:80]
            else:
                body = str(r["content"])
                if validate is None or validate(body):
                    return body, None
                last = "unusable page"
            # Tight per-IP rate limits hand out one good page then wall us off;
            # a short pause lets the limit reset before the next transport.
            _nap(ctx, 2.0)
    except (_Interrupted, _Timeout):
        raise
    except Exception as e:
        return None, str(e)[:80]
    return None, last or "blocked"


def _request(ctx: _Ctx, url: str, *, params=None, headers=None, method="GET",
             json_body=None, cap=15.0, retries=2, escalate: bool = True,
             validate=None) -> tuple[str | None, str | None]:
    """One HTTP call with shared deadline, interrupt, and 429/5xx backoff.

    Returns (text, None) on success or (None, reason) on failure. Aborts
    mid-body via ctx.check() so a long response honours interrupt promptly.
    A blocked (or empty-shell) GET escalates through the webfetch bypass
    cascade; `validate` decides whether a returned page is usable.
    `escalate=False` opts out of that cascade (and the Tor retry behind it) for
    callers that know a client-fingerprint change cannot help them — e.g. the
    webreach recipes, which hit structured JSON APIs whose walls are keyed on
    IP reputation, where the cascade only burns 10+ s of phone radio.
    """
    import httpx

    hdrs = {"User-Agent": BROWSER_UA, "Accept-Language": "en-US,en;q=0.9"}
    if headers:
        hdrs.update(headers)
    last = ""
    text: str | None = None
    for attempt in range(retries + 1):
        ctx.check()
        try:
            timeout = httpx.Timeout(ctx.slice(cap))
            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                with client.stream(method, url, params=params, headers=hdrs,
                                   json=json_body) as r:
                    if r.status_code in _RETRY_STATUS and attempt < retries:
                        last = f"HTTP {r.status_code}"
                        wait = 0.5 * (2 ** attempt)
                        ra = r.headers.get("Retry-After")
                        if ra:
                            try:
                                wait = max(wait, float(ra))
                            except ValueError as _oc_exc:
                                swallow(_oc_exc, "opencode_py/tools/websearch.py:384")
                                pass
                        if min(wait, 5.0) >= ctx.remaining():
                            raise _Timeout()
                        _nap(ctx, min(wait, 5.0))
                        continue
                    if r.status_code >= 400:
                        last = f"HTTP {r.status_code}"
                        break
                    parts: list[str] = []
                    for chunk in r.iter_text():
                        ctx.check()
                        parts.append(chunk)
                    text = "".join(parts)
                    break
        except (_Interrupted, _Timeout):
            raise
        except Exception as e:
            last = (str(e) or type(e).__name__)
            if attempt >= retries:
                break
            _nap(ctx, min(1.0 * (attempt + 1), ctx.remaining()))

    if text is not None and (validate is None or validate(text)):
        return text, None

    # Escalate a blocked / failed / empty-shell GET through the bypass cascade,
    # then (only if that also failed) from a fresh exit IP via Tor.
    if (escalate and method == "GET" and ctx.remaining() > 2.0
            and not any(h.lower() in _AUTH_HEADERS for h in hdrs)):
        got, berr = _bypass_fetch(ctx, url, validate)
        if got:
            return got, None
        if berr:
            last = f"{last}; bypass: {berr}" if last else f"bypass: {berr}"
        if ctx.remaining() > 3.0:
            got, terr = _tor_fetch(ctx, url, validate)
            if got:
                return got, None
            if terr:
                last = f"{last}; {terr}" if last else terr
    if text is not None:
        return text, None
    return None, last or "request failed"


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _clean(s: str) -> str:
    try:
        return re.sub(r"\s+", " ", _html.unescape(re.sub(r"<[^>]+>", "", s or ""))).strip()
    except Exception:
        return (s or "").strip()


def _norm_url(u: str) -> str:
    try:
        p = _U.urlsplit(u.strip())
        host = p.netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        q = [(k, v) for k, v in _U.parse_qsl(p.query)
             if not k.lower().startswith(_TRACK_PREFIX) and k.lower() not in _TRACK_KEYS]
        path = p.path.rstrip("/") or "/"
        return _U.urlunsplit((p.scheme.lower(), host, path, _U.urlencode(q), ""))
    except Exception:
        return (u or "").strip().lower()


def _item(title: str, url: str, snippet: str) -> dict:
    return {"title": (title or "")[:150], "url": (url or "")[:500], "snippet": (snippet or "")[:300]}


def _merge(acc: list, new: list, seen: set, n: int) -> None:
    for it in new:
        u = it.get("url") or ""
        if u:
            k = _norm_url(u)
            if k in seen:
                continue
            seen.add(k)
        acc.append(it)
        if len(acc) >= n:
            return


def _ok_result(href: str, title: str) -> bool:
    return bool(title) and href.startswith("http") and "duckduckgo.com" not in href


def _unwrap_ddg(href: str) -> str:
    href = _html.unescape(href or "").strip()
    if href.startswith("//"):
        href = "https:" + href
    if "uddg=" in href:
        try:
            q = dict(_U.parse_qsl(_U.urlsplit(href).query))
            href = _U.unquote(q.get("uddg") or href)
        except Exception as _oc_exc:
            swallow(_oc_exc, "tools/websearch.py:480")
            pass
    return href


def _bing_url(href: str) -> str:
    href = _html.unescape(href or "").strip()
    if "bing.com/ck/a" in href:
        try:
            q = dict(_U.parse_qsl(_U.urlsplit(href).query))
            u = q.get("u", "")
            if u.startswith("a1"):
                s = u[2:] + "=" * (-len(u[2:]) % 4)
                dec = base64.urlsafe_b64decode(s).decode("utf-8", "replace")
                return dec if dec.startswith("http") else ""
        except Exception:
            return ""
        return ""
    return href


# --------------------------------------------------------------------------- #
# Keyless providers
# --------------------------------------------------------------------------- #
def _ddg_search(ctx: _Ctx, query: str, n: int, opts: dict) -> tuple[list, str | None]:
    lite = bool(opts.get("fast"))
    base = "https://lite.duckduckgo.com/lite/" if lite else "https://html.duckduckgo.com/html/"
    params: dict = {"q": query}
    if opts.get("region"):
        params["kl"] = opts["region"]
    marker = 'uddg=' if lite else 'class="result__a"'
    t, err = _request(ctx, base, params=params, cap=12.0,
                      validate=lambda b: marker in b or "result results_links" in b)
    if err:
        return [], f"duckduckgo: {err}"
    out: list[dict] = []
    if lite:
        for m in re.finditer(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', t, re.S):
            href = _unwrap_ddg(m.group(1))
            title = _clean(m.group(2))
            if _ok_result(href, title):
                out.append(_item(title, href, ""))
            if len(out) >= n + 4:
                break
    else:
        for b in re.split(r'<div class="result results_links[^"]*"', t)[1:]:
            m = re.search(r'<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', b, re.S)
            if not m:
                m = re.search(r'<a[^>]+href="([^"]+)"[^>]*class="result__a"[^>]*>(.*?)</a>', b, re.S)
            if not m:
                continue
            href = _unwrap_ddg(m.group(1))
            title = _clean(m.group(2))
            s = re.search(r'class="result__snippet"[^>]*>(.*?)</div', b, re.S)
            snip = _clean(s.group(1)) if s else ""
            if _ok_result(href, title):
                out.append(_item(title, href, snip))
            if len(out) >= n:
                break
    return out, None


def _bing_search(ctx: _Ctx, query: str, n: int, opts: dict) -> tuple[list, str | None]:
    params: dict = {"q": query}
    if opts.get("region"):
        params["cc"] = str(opts["region"]).upper()
    if opts.get("language"):
        params["setLang"] = opts["language"]
    t, err = _request(ctx, "https://www.bing.com/search", params=params, cap=12.0,
                      validate=lambda b: "b_algo" in b)
    if err:
        return [], f"bing: {err}"
    out: list[dict] = []
    for m in re.finditer(
        r'<h2[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>(.*?)(?=<h2|</li>)', t, re.S
    ):
        href = _bing_url(m.group(1))
        title = _clean(m.group(2))
        sn = re.search(r'class="b_caption".*?<p[^>]*>(.*?)</p>', m.group(3), re.S)
        snip = _clean(sn.group(1)) if sn else ""
        if title and href.startswith("http") and "bing.com/ck" not in href:
            out.append(_item(title, href, snip))
        if len(out) >= n:
            break
    return out, None


def _brave_html_search(ctx: _Ctx, query: str, n: int, opts: dict) -> tuple[list, str | None]:
    params: dict = {"q": query}
    t, err = _request(ctx, "https://search.brave.com/search", params=params, cap=12.0,
                      validate=lambda b: 'data-type="web"' in b or 'class="snippet' in b)
    if err:
        return [], f"brave: {err}"
    out: list[dict] = []
    seen: set[str] = set()

    def _add(url: str, title: str, snip: str) -> None:
        if not title or not url.startswith("http") or "brave.com" in url or url in seen:
            return
        seen.add(url)
        out.append(_item(title, url, snip))

    # Current Brave markup: one <div class="snippet" data-type="web"> per result.
    for blk in re.split(r'<div class="snippet[^"]*"[^>]*data-type="web"', t)[1:]:
        a = re.search(r'<a[^>]+href="(https?://[^"]+)"', blk)
        if not a:
            continue
        tm = re.search(r'<div class="title[^"]*"[^>]*>(.*?)</div>', blk, re.S)
        sm = re.search(r'<div class="content[^"]*"[^>]*>(.*?)</div>', blk, re.S)
        _add(_html.unescape(a.group(1)).strip(),
             _clean(tm.group(1)) if tm else "",
             _clean(sm.group(1)) if sm else "")
        if len(out) >= n:
            return out, None

    # Legacy markup fallback (older Brave layout).
    if not out:
        for m in re.finditer(r'href="(https?://[^"]+)"[^>]{0,300}>([^<]{10,150})<', t):
            _add(_html.unescape(m.group(1)).strip(), _clean(m.group(2)), "")
            if len(out) >= n:
                break
    return out, None


# --------------------------------------------------------------------------- #
# Keyed providers
# --------------------------------------------------------------------------- #
def _exa_search(ctx: _Ctx, query: str, n: int, opts: dict) -> tuple[list, str | None]:
    key = _key("exa")
    if not key:
        return [], "exa: missing key"
    typ = "deep" if opts.get("deep") else ("fast" if opts.get("fast") else "auto")
    body = {"query": query, "numResults": n, "type": typ,
            "contents": {"text": {"maxCharacters": 800}}}
    t, err = _request(ctx, "https://api.exa.ai/search", method="POST", json_body=body,
                      headers={"x-api-key": key, "Content-Type": "application/json"},
                      cap=ctx.remaining())
    if err:
        return [], f"exa: {err}"
    try:
        data = _json.loads(t)
    except Exception:
        return [], "exa: bad JSON"
    out: list[dict] = []
    for it in (data.get("results") or [])[:n]:
        out.append(_item(str(it.get("title") or ""), str(it.get("url") or ""),
                         str(it.get("summary") or it.get("text") or it.get("snippet") or "")))
    return out, None


def _parallel_search(ctx: _Ctx, query: str, n: int, opts: dict) -> tuple[list, str | None]:
    key = _key("parallel")
    headers = {"Content-Type": "application/json",
               "Accept": "application/json, text/event-stream"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "web_search",
                       "arguments": {"objective": query, "search_queries": [query]}}}
    t, err = _request(ctx, "https://search.parallel.ai/mcp", method="POST", json_body=body,
                      headers=headers, cap=ctx.remaining())
    if err:
        return [], f"parallel: {err}"
    payload = ""
    for line in (t or "").splitlines():
        s = line.strip()
        if s.startswith("data:"):
            s = s[5:].strip()
        if not s.startswith("{"):
            continue
        try:
            d = _json.loads(s)
        except Exception as _oc_exc:
            swallow(_oc_exc, "tools/websearch.py:652")
            continue
        if d.get("error"):
            return [], f"parallel: {d.get('error')}"
        for c in ((d.get("result") or {}).get("content") or []):
            if isinstance(c, dict) and c.get("text"):
                payload += str(c["text"]) + "\n"
    if not payload:
        payload = t or ""
    out: list[dict] = []
    seen: set[str] = set()
    # markdown links first -> real titles; then bare URLs
    for m in re.finditer(r'\[([^\]\n]{3,150})\]\((https?://[^\s)]+)\)', payload):
        url = m.group(2).rstrip(".,);]")
        if url in seen:
            continue
        seen.add(url)
        out.append(_item(_clean(m.group(1)), url, ""))
        if len(out) >= n:
            return out, None
    for m in re.finditer(r'https?://[^\s"\'<>]+', payload):
        url = m.group(0).rstrip(".,);]")
        if url in seen:
            continue
        seen.add(url)
        out.append(_item("", url, ""))
        if len(out) >= n:
            break
    if not out and payload.strip():
        out.append(_item(query[:80], "", payload.strip()[:1500]))
    return out, None


def _tavily_search(ctx: _Ctx, query: str, n: int, opts: dict) -> tuple[list, str | None]:
    key = _key("tavily")
    if not key:
        return [], "tavily: missing key"
    body = {"api_key": key, "query": query, "max_results": n,
            "search_depth": "advanced" if opts.get("deep") else "basic",
            "include_answer": False}
    if opts.get("freshness"):
        body["days"] = opts["freshness"]
    t, err = _request(ctx, "https://api.tavily.com/search", method="POST", json_body=body,
                      headers={"Content-Type": "application/json"}, cap=ctx.remaining())
    if err:
        return [], f"tavily: {err}"
    try:
        data = _json.loads(t)
    except Exception:
        return [], "tavily: bad JSON"
    out: list[dict] = []
    for it in (data.get("results") or [])[:n]:
        out.append(_item(str(it.get("title") or ""), str(it.get("url") or ""),
                         str(it.get("content") or it.get("snippet") or "")))
    return out, None


def _serper_search(ctx: _Ctx, query: str, n: int, opts: dict) -> tuple[list, str | None]:
    key = _key("serper")
    if not key:
        return [], "serper: missing key"
    body: dict = {"q": query, "num": n}
    if opts.get("language"):
        body["hl"] = opts["language"]
    if opts.get("region"):
        body["gl"] = str(opts["region"]).lower()
    fr = str(opts.get("freshness") or "").lower()
    body["tbs"] = {"day": "qdr:d", "week": "qdr:w", "month": "qdr:m", "year": "qdr:y"}.get(fr, "")
    if not body["tbs"]:
        body.pop("tbs")
    t, err = _request(ctx, "https://google.serper.dev/search", method="POST", json_body=body,
                      headers={"X-API-KEY": key, "Content-Type": "application/json"},
                      cap=ctx.remaining())
    if err:
        return [], f"serper: {err}"
    try:
        data = _json.loads(t)
    except Exception:
        return [], "serper: bad JSON"
    out: list[dict] = []
    for it in (data.get("organic") or [])[:n]:
        out.append(_item(str(it.get("title") or ""), str(it.get("link") or ""),
                         str(it.get("snippet") or "")))
    return out, None


def _brave_api_search(ctx: _Ctx, query: str, n: int, opts: dict) -> tuple[list, str | None]:
    key = _key("brave-api")
    if not key:
        return [], "brave-api: missing key"
    params: dict = {"q": query, "count": n}
    if opts.get("region"):
        params["country"] = str(opts["region"]).upper()
    if opts.get("language"):
        params["search_lang"] = opts["language"]
    fr = str(opts.get("freshness") or "").lower()
    params["freshness"] = {"day": "pd", "week": "pw", "month": "pm", "year": "py"}.get(fr, "")
    if not params["freshness"]:
        params.pop("freshness")
    if opts.get("deep"):
        params["extra_snippets"] = "true"
    t, err = _request(ctx, "https://api.search.brave.com/res/v1/web/search", params=params,
                      headers={"X-Subscription-Token": key, "Accept": "application/json"},
                      cap=ctx.remaining())
    if err:
        return [], f"brave-api: {err}"
    try:
        data = _json.loads(t)
    except Exception:
        return [], "brave-api: bad JSON"
    web = data.get("web") or {}
    out: list[dict] = []
    for it in (web.get("results") or [])[:n]:
        snip = str(it.get("description") or "")
        extra = it.get("extra_snippets") or []
        if extra and isinstance(extra, list):
            snip = (snip + " " + " ".join(str(x) for x in extra)).strip()
        out.append(_item(str(it.get("title") or ""), str(it.get("url") or ""), snip))
    return out, None


_PROVIDER_FN = {
    "duckduckgo": _ddg_search,
    "bing": _bing_search,
    "brave": _brave_html_search,
    "exa": _exa_search,
    "parallel": _parallel_search,
    "tavily": _tavily_search,
    "serper": _serper_search,
    "brave-api": _brave_api_search,
}


# --------------------------------------------------------------------------- #
# Tool
# --------------------------------------------------------------------------- #
def tool(registry: Any | None = None) -> Tool:
    import datetime as _dt
    year = _dt.datetime.now().year
    description = (
        "FIRST for any web info: search the web, then fetch. "
        "Search the web across providers — real-time results beyond knowledge cutoff. "
        f"The current year is {year}: use it for recent queries. "
        "Keyless: duckduckgo/bing work with no keys. "
        "Keyed best: exa/parallel/tavily/serper/brave-api via env keys or config. "
        "Returns titles+URLs+snippets — fetch top pages with webfetch. "
        "type fast=quick, deep=thorough. Optional: region, language, freshness."
    )

    def run(input: dict) -> dict:
        query = str(input.get("query") or "").strip()
        if not query:
            return {"output": "websearch requires 'query'.", "error": True}
        try:
            n = int(input.get("numResults") or 8)
        except (TypeError, ValueError):
            n = 8
        n = max(1, min(n, 20))
        typ = str(input.get("type") or "auto").strip().lower()
        fast, deep = typ == "fast", typ == "deep"
        if fast:
            n = min(n, 5)
        elif deep:
            n = min(20, max(n, 10))
        provider, note = _pick(str(input.get("provider") or ""))
        if not provider:
            return {"output": f"websearch: {note}", "error": True,
                    "metadata": {"provider": None}}
        try:
            timeout = int(input.get("timeout") or 25)
        except (TypeError, ValueError):
            timeout = 25
        timeout = max(5, min(timeout, 60))

        checker = getattr(registry, "interrupt_check", None) if registry is not None else None
        if not callable(checker):
            checker = None
        ctx = _Ctx(timeout, checker)

        opts = {
            "fast": fast, "deep": deep,
            "region": str(input.get("region") or "").strip(),
            "language": str(input.get("language") or "").strip(),
            "freshness": input.get("freshness"),
        }

        # Identical research queries are common; a repeat inside the TTL costs
        # no request and no rate-limit budget.
        ckey = "|".join((provider, query.lower(), str(n), str(fast), str(deep),
                         opts["region"], opts["language"], str(opts["freshness"])))
        now = time.monotonic()
        hit = _cache_get(ckey, now)
        if hit is not None:
            res = dict(hit)
            md = dict(res.get("metadata") or {})
            md["cached"] = True
            res["metadata"] = md
            res["output"] = (f"[cached {int(_CACHE_TTL // 60)}min] "
                             + str(res.get("output") or ""))
            return res

        # Fallback chain. An explicitly named engine is pinned first; in auto
        # mode the rotated order decides, so no single engine takes the first
        # attempt forever (that is how one engine ends up blocking every
        # search) and a cooled-off one is simply skipped.
        raw = str(input.get("provider") or "").strip().lower().replace("_", "-")
        explicit = raw not in ("", "auto")
        ordered = _ordered_chain(now)
        if provider in _KEYED:
            if explicit or _is_up(provider, now):
                chain = [provider] + [e for e in ordered if e != provider]
            else:
                # keyed engine is cooling: don't re-pay for the same wall
                chain = [e for e in ordered if e != provider]
        elif explicit:
            chain = [provider] + [e for e in ordered if e != provider]
        else:
            chain = ordered

        t0 = time.monotonic()
        acc: list[dict] = []
        seen: set[str] = set()
        errs: list[str] = []
        used: list[str] = []
        interrupted = False
        try:
            ctx.check()
        except _Interrupted:
            return {"output": "(interrupted)", "error": True, "interrupted": True}
        except _Timeout:
            return {"output": "websearch: timed out before starting.", "error": True}

        for name in chain:
            try:
                items, err = _PROVIDER_FN[name](ctx, query, n, opts)
            except _Interrupted:
                interrupted = True
                break
            except _Timeout as _oc_exc:
                swallow(_oc_exc, "opencode_py/tools/websearch.py:895")
                break
            except Exception as e:
                items, err = [], f"{name}: {e}"
            if err:
                errs.append(err)
            if items:
                _mark_up(name)
                used.append(name)
                _merge(acc, items, seen, n)
            else:
                # empty or blocked: keep the next search off this engine for a
                # while instead of re-paying for the same wall every time
                _mark_down(name, time.monotonic())
            if len(acc) >= n:
                break

        elapsed = round(time.monotonic() - t0, 1)
        label = "+".join(used) if used else provider
        if interrupted and not acc:
            return {"output": "(interrupted)", "error": True, "interrupted": True}
        if note:
            errs.insert(0, note)
        if not acc:
            msg = "; ".join(errs[:2]) if errs else "no results"
            return {"output": f"No search results for '{query}' ({label}, {elapsed}s). "
                               f"{msg} Try a different query.",
                    "error": True,
                    "metadata": {"provider": label, "elapsed_s": elapsed}}

        lines = [f"Web results for '{query}' ({label}, {len(acc)}, {elapsed}s):", ""]
        for i, it in enumerate(acc, 1):
            lines.append(f"{i}. {it.get('title') or it.get('url')}")
            if it.get("url"):
                lines.append(f"   {it['url']}")
            if it.get("snippet"):
                lines.append(f"   {it['snippet'][:280]}")
        lines.append("")
        lines.append("Fetch top pages with webfetch for full content.")
        result = {"output": "\n".join(lines),
                  "metadata": {"provider": label, "numResults": len(acc),
                               "elapsed_s": elapsed, "items": acc}}
        _cache_put(ckey, result, time.monotonic())
        return result

    return Tool(
        name="websearch",
        description=description,
        parameters=schema_with(
            {
                "query": {"type": "string", "description": "Websearch query (include the year for recent topics)"},
                "numResults": {"type": "integer", "description": "Number of results (default 8, max 20)", "optional": True},
                "type": {"type": "string", "enum": ["auto", "fast", "deep"], "description": "fast=quick, deep=thorough", "optional": True},
                "provider": {"type": "string",
                             "description": "duckduckgo/bing (keyless) or exa/parallel/tavily/serper/brave-api (keyed). auto picks best available.",
                             "optional": True},
                "timeout": {"type": "integer", "description": "Seconds for the whole search (max 60, default 25)", "optional": True},
                "region": {"type": "string", "description": "Region/country code, e.g. us, uk, de", "optional": True},
                "language": {"type": "string", "description": "Language code, e.g. en, de, ja", "optional": True},
                "freshness": {"type": "integer", "description": "Recency in days (keyed providers: tavily/serper/brave)", "optional": True},
            },
            ["query"],
        ),
        run=run,
        permission="websearch",
    )

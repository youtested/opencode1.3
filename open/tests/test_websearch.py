"""Tests for the websearch tool: provider selection, key resolution, parser
robustness (Bing/DDG/Brave markup), dedupe/merge, interrupt + deadline, the
region/language/freshness parameter mapping, and escalation to the shared
webfetch bypass cascade.

Fully offline: every network call goes through a monkeypatched ``_request`` or a
fake ``UltimateBypass``.
"""
from __future__ import annotations

import base64
import json
import types

import pytest

from opencode_py.tools import websearch as W


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #
def _bing_link(real: str) -> str:
    u = base64.urlsafe_b64encode(real.encode()).decode().rstrip("=")
    return f"https://www.bing.com/ck/a?!&amp;p=abc&amp;u=a1{u}"


BING_HTML = (
    '<li class="b_algo" data-id iid=SERP.1>'
    '<h2 class=""><a target="_blank" href="' + _bing_link("https://www.python.org/") + '">'
    "<strong>Welcome to</strong> <strong>Python.org</strong></a></h2>"
    '<div class="b_caption"><p>Experienced programmers can pick up Python quickly.</p></div>'
    "</li>"
    '<li class="b_algo"><h2 class=""><a href="' + _bing_link("https://www.w3schools.com/python/") + '">'
    "Python Tutorial</a></h2>"
    '<div class="b_caption"><p>Learn Python the easy way.</p></div></li>'
)

DDG_HTML = (
    '<div class="result results_links results_links_deep web-result">'
    '<a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Frealpython.com%2Fasync-io%2F">'
    "Python asyncio</a>"
    '<a class="result__snippet">Learn async with asyncio.</a></div>'
)

DDG_LITE_HTML = (
    '<a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Frealpython.com%2Fasync-io%2F">'
    "Python asyncio</a>"
)


@pytest.fixture(autouse=True)
def clean_keys(monkeypatch):
    for name in list(W._KEY_ENV):
        for env in W._KEY_ENV[name]:
            monkeypatch.delenv(env, raising=False)
    monkeypatch.delenv("OPENCODE_WEBSEARCH_PROVIDER", raising=False)
    monkeypatch.setattr(W, "_cfg_provider_keys", lambda: {})
    W._reset_state()
    yield
    W._reset_state()


def _ctx(timeout=25):
    return W._Ctx(timeout)


# --------------------------------------------------------------------------- #
# #1/#2 Bing parser
# --------------------------------------------------------------------------- #
def test_bing_decodes_ck_a_url():
    assert W._bing_url(_bing_link("https://www.python.org/")) == "https://www.python.org/"


def test_bing_ck_a_undecodable_is_dropped():
    assert W._bing_url("https://www.bing.com/ck/a?u=a1!!!bad!!!") == ""


def test_bing_parses_classed_h2_with_snippets(monkeypatch):
    monkeypatch.setattr(W, "_request", lambda ctx, url, **kw: (BING_HTML, None))
    items, err = W._bing_search(_ctx(), "python", 5, {})
    assert err is None
    assert len(items) == 2
    assert items[0]["url"] == "https://www.python.org/"
    assert "Python.org" in items[0]["title"]
    assert "pick up Python" in items[0]["snippet"]
    assert not any("bing.com/ck" in it["url"] for it in items)


# --------------------------------------------------------------------------- #
# DDG parser (html + lite)
# --------------------------------------------------------------------------- #
def test_ddg_html_unwraps_uddg_and_snippet(monkeypatch):
    monkeypatch.setattr(W, "_request", lambda ctx, url, **kw: (DDG_HTML, None))
    items, _ = W._ddg_search(_ctx(), "python", 5, {"fast": False})
    assert items[0]["url"] == "https://realpython.com/async-io/"
    assert items[0]["snippet"] == "Learn async with asyncio."


def test_ddg_lite_fast_path(monkeypatch):
    monkeypatch.setattr(W, "_request", lambda ctx, url, **kw: (DDG_LITE_HTML, None))
    items, _ = W._ddg_search(_ctx(), "python", 5, {"fast": True})
    assert items[0]["url"] == "https://realpython.com/async-io/"


# --------------------------------------------------------------------------- #
# dedupe / merge
# --------------------------------------------------------------------------- #
def test_norm_url_strips_www_trailing_slash_and_tracking():
    a = "https://www.Example.com/path/?utm_source=x&fbclid=1"
    b = "https://example.com/path"
    assert W._norm_url(a) == W._norm_url(b)


def test_merge_dedupes_across_providers():
    acc, seen = [], set()
    W._merge(acc, [{"title": "a", "url": "https://x.com/", "snippet": ""}], seen, 5)
    W._merge(acc, [{"title": "dup", "url": "https://x.com", "snippet": ""},
                   {"title": "b", "url": "https://y.com", "snippet": ""}], seen, 5)
    assert [i["title"] for i in acc] == ["a", "b"]


# --------------------------------------------------------------------------- #
# #4/#5 provider selection
# --------------------------------------------------------------------------- #
def test_unknown_provider_is_explicit_error():
    provider, note = W._pick("googIe")
    assert provider == ""
    assert "unknown provider" in note


def test_keyed_without_key_falls_back_to_keyless():
    provider, note = W._pick("exa")
    assert provider == "duckduckgo"
    assert "needs an API key" in note


def test_keyed_with_env_key_selected(monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "k")
    W._clear_key_cache()
    assert W._pick("exa") == ("exa", "")


def test_alias_normalisation(monkeypatch):
    provider, note = W._pick("brave_api")  # -> brave-api, no key -> keyless
    assert provider == "duckduckgo"
    assert "brave-api" in note


def test_auto_prefers_keyed_then_keyless(monkeypatch):
    assert W._pick("") == ("duckduckgo", "")
    monkeypatch.setenv("TAVILY_API_KEY", "k")
    W._clear_key_cache()
    assert W._pick("") == ("tavily", "")


# --------------------------------------------------------------------------- #
# #9 config key resolution
# --------------------------------------------------------------------------- #
def test_key_reads_config_providers(monkeypatch):
    monkeypatch.setattr(W, "_cfg_provider_keys", lambda: {"tavily": {"api_key": "cfg"}})
    W._clear_key_cache()
    assert W._key("tavily") == "cfg"


def test_env_beats_config(monkeypatch):
    monkeypatch.setenv("SERPER_API_KEY", "fromenv")
    monkeypatch.setattr(W, "_cfg_provider_keys", lambda: {"serper": {"api_key": "cfg"}})
    W._clear_key_cache()
    assert W._key("serper") == "fromenv"


# --------------------------------------------------------------------------- #
# #6/#7 interrupt + deadline
# --------------------------------------------------------------------------- #
def test_interrupt_returns_immediately():
    class Reg:
        def interrupt_check(self):
            return True

    r = W.tool(Reg()).run({"query": "x", "numResults": 5})
    assert r.get("interrupted") is True
    assert r["error"] is True


def test_deadline_is_shared_across_chain(monkeypatch):
    seen = []

    def slow(ctx, url, **kw):
        seen.append(url)
        return None, "HTTP 500"

    monkeypatch.setattr(W, "_request", slow)
    r = W.tool().run({"query": "x", "numResults": 20, "timeout": 5})
    assert r["error"] is True  # nothing found
    # the whole chain must share the budget and not loop forever
    assert len(seen) <= 4


# --------------------------------------------------------------------------- #
# #10 merge fallback in the tool
# --------------------------------------------------------------------------- #
def test_tool_merges_fallback_results(monkeypatch):
    def fake_ddg(ctx, q, n, opts):
        return [{"title": "ddg1", "url": "https://a.com", "snippet": ""}], None

    def fake_bing(ctx, q, n, opts):
        return [{"title": "bing1", "url": "https://b.com", "snippet": ""},
                {"title": "dup", "url": "https://a.com", "snippet": ""}], None

    monkeypatch.setitem(W._PROVIDER_FN, "duckduckgo", fake_ddg)
    monkeypatch.setitem(W._PROVIDER_FN, "bing", fake_bing)
    r = W.tool().run({"query": "x", "numResults": 5})
    titles = [i["title"] for i in r["metadata"]["items"]]
    assert titles == ["ddg1", "bing1"]  # merged, deduped, bing first-result dup removed


# --------------------------------------------------------------------------- #
# #12 region/language/freshness mapping
# --------------------------------------------------------------------------- #
def test_serper_maps_region_language_freshness(monkeypatch):
    monkeypatch.setenv("SERPER_API_KEY", "k")
    W._clear_key_cache()
    seen = {}

    def fake(ctx, url, **kw):
        seen.update(kw.get("json_body") or {})
        return json.dumps({"organic": []}), None

    monkeypatch.setattr(W, "_request", fake)
    W._serper_search(_ctx(), "q", 5,
                     {"region": "DE", "language": "de", "freshness": "week"})
    assert seen["gl"] == "de"
    assert seen["hl"] == "de"
    assert seen["tbs"] == "qdr:w"


def test_brave_api_maps_country_lang_freshness_deep(monkeypatch):
    monkeypatch.setenv("BRAVE_API_KEY", "k")
    W._clear_key_cache()
    seen = {}

    def fake(ctx, url, **kw):
        seen.update(kw.get("params") or {})
        return json.dumps({"web": {"results": []}}), None

    monkeypatch.setattr(W, "_request", fake)
    W._brave_api_search(_ctx(), "q", 5,
                        {"region": "de", "language": "de", "freshness": "day", "deep": True})
    assert seen["country"] == "DE"
    assert seen["search_lang"] == "de"
    assert seen["freshness"] == "pd"
    assert seen["extra_snippets"] == "true"


def test_exa_maps_type_fast_deep(monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "k")
    W._clear_key_cache()
    seen = {}

    def fake(ctx, url, **kw):
        seen.update(kw.get("json_body") or {})
        return json.dumps({"results": []}), None

    monkeypatch.setattr(W, "_request", fake)
    W._exa_search(_ctx(), "q", 5, {"fast": True})
    assert seen["type"] == "fast"
    W._exa_search(_ctx(), "q", 5, {"deep": True})
    assert seen["type"] == "deep"
    W._exa_search(_ctx(), "q", 5, {})
    assert seen["type"] == "auto"


# --------------------------------------------------------------------------- #
# #11 parallel parser
# --------------------------------------------------------------------------- #
def test_parallel_extracts_markdown_titles_and_dedupes(monkeypatch):
    body = (
        'event: message\ndata: {"result":{"content":[{"text":"'
        '[Real Python](https://realpython.com/x) and https://realpython.com/x '
        'plus https://other.com/y"}]}}\n'
    )

    def fake(ctx, url, **kw):
        return body, None

    monkeypatch.setattr(W, "_request", fake)
    items, err = W._parallel_search(_ctx(), "q", 5, {})
    urls = [i["url"] for i in items]
    assert "https://realpython.com/x" in urls
    assert "https://other.com/y" in urls
    assert len(urls) == len(set(urls))  # deduped
    assert items[0]["title"] == "Real Python"
    assert err is None


def test_parallel_surfaces_jsonrpc_error(monkeypatch):
    body = 'data: {"error":{"message":"bad key"}}\n'
    monkeypatch.setattr(W, "_request", lambda ctx, url, **kw: (body, None))
    items, err = W._parallel_search(_ctx(), "q", 5, {})
    assert items == []
    assert "bad key" in err


# --------------------------------------------------------------------------- #
# #13 fast/deep + schema
# --------------------------------------------------------------------------- #
def test_fast_caps_results_and_deep_raises(monkeypatch):
    def fake(ctx, q, n, opts):
        return [{"title": "t", "url": f"https://x{i}.com", "snippet": ""}
                for i in range(20)], None

    monkeypatch.setitem(W._PROVIDER_FN, "duckduckgo", fake)
    monkeypatch.setitem(W._PROVIDER_FN, "bing", fake)
    r = W.tool().run({"query": "x", "type": "fast", "numResults": 20})
    assert r["metadata"]["numResults"] <= 5


def test_schema_exposes_new_filters():
    props = W.tool().parameters["properties"]
    for key in ("region", "language", "freshness", "timeout", "provider", "type"):
        assert key in props


def test_requires_query():
    r = W.tool().run({})
    assert r["error"] is True
    assert "query" in r["output"]


# --------------------------------------------------------------------------- #
# Brave current markup
# --------------------------------------------------------------------------- #
BRAVE_HTML = (
    '<div class="snippet svelte-x" data-type="web">'
    '<div class="result-body"><a href="https://realpython.com/async-io-python/" '
    'target="_self" class="svelte-y">'
    '<div class="title search-snippet-title svelte-y">asyncio walkthrough</div>'
    '</a><div class="content svelte-z">Learn async with asyncio.</div></div></div>'
    '<div class="snippet svelte-x" data-type="web">'
    '<a href="https://search.brave.com/search?q=x" class="q">Brave</a>'
    '<div class="title">Brave</div></div>'
)


def test_brave_parses_current_snippet_markup(monkeypatch):
    monkeypatch.setattr(W, "_request", lambda ctx, url, **kw: (BRAVE_HTML, None))
    items, err = W._brave_html_search(_ctx(), "python", 5, {})
    assert err is None
    assert len(items) == 1  # the brave.com self-link is dropped
    assert items[0]["url"] == "https://realpython.com/async-io-python/"
    assert items[0]["title"] == "asyncio walkthrough"
    assert items[0]["snippet"] == "Learn async with asyncio."


# --------------------------------------------------------------------------- #
# bypass escalation
# --------------------------------------------------------------------------- #
def _fake_bypass_module(pages, poll_interrupt=False):
    """Install a fake .cloudflare_bypass exposing pages per force_method."""
    calls = []

    class FakeUB:
        def __init__(self, timeout=20, proxy_pool=None):
            self.timeout = timeout

        def fetch(self, url, is_interrupted=None, force_method=None):
            calls.append(force_method)
            if poll_interrupt and is_interrupted is not None:
                if is_interrupted():
                    return {"success": False, "error": "cancelled"}
            body = pages.get(force_method, "")
            if not body:
                return {"success": False, "error": "blocked"}
            return {"success": True, "method": force_method or "auto",
                    "content": body}

    class FakePool:
        @staticmethod
        def shared():
            return None

    mod = types.SimpleNamespace(UltimateBypass=FakeUB, ProxyPool=FakePool)
    return mod, calls


def test_bypass_escalates_to_curl_when_first_page_is_a_shell(monkeypatch):
    """A 200 JS shell must not be accepted; escalate to the next transport."""
    good = '<div class="snippet" data-type="web">results</div>'
    mod, calls = _fake_bypass_module({None: "<html>shell</html>", "curl": good})
    monkeypatch.setitem(
        __import__("sys").modules, "opencode_py.tools.cloudflare_bypass", mod)
    body, err = W._bypass_fetch(_ctx(30), "https://search.brave.com/search",
                                validate=lambda b: "data-type=" in b)
    assert body == good
    assert err is None
    assert calls[0] is None and "curl" in calls  # escalated after the shell


def test_bypass_returns_none_when_every_transport_fails(monkeypatch):
    mod, _ = _fake_bypass_module({})
    monkeypatch.setitem(
        __import__("sys").modules, "opencode_py.tools.cloudflare_bypass", mod)
    body, err = W._bypass_fetch(_ctx(30), "https://search.brave.com/search",
                                validate=lambda b: "x" in b)
    assert body is None
    assert err


def test_request_escalates_a_blocked_get(monkeypatch):
    """A plain-path 429 must fall through to the bypass cascade."""
    monkeypatch.setattr(W, "_bypass_fetch", lambda ctx, url, validate=None: ("BYPASSED", None))
    got, err = W._request(_ctx(30), "https://search.brave.com/search",
                          params={"q": "x"}, retries=0,
                          validate=lambda b: b == "BYPASSED")
    assert got == "BYPASSED"
    assert err is None


def test_request_escalate_false_skips_the_cascade(monkeypatch):
    """Opt-out added for the webreach recipes: a JSON API walled on IP
    reputation cannot be un-walled by a client-fingerprint change, so running
    the six-transport cascade only burns radio. Default stays True."""
    monkeypatch.setattr(W, "_bypass_fetch",
                        lambda ctx, url, validate=None: pytest.fail("must not bypass"))
    monkeypatch.setattr(W, "_tor_fetch",
                        lambda ctx, url, validate=None: pytest.fail("must not use tor"))
    got, err = W._request(_ctx(30), "https://api.example.com/x", retries=0, escalate=False)
    assert got is None
    assert err


def test_request_never_escalates_a_keyed_request(monkeypatch):
    """A request carrying an API key must not be replayed without its header."""
    called = []

    def boom(ctx, url, validate=None):
        called.append(url)
        return "SHOULD NOT HAPPEN", None

    monkeypatch.setattr(W, "_bypass_fetch", boom)
    got, err = W._request(_ctx(30), "https://api.exa.ai/search", method="POST",
                          json_body={}, headers={"x-api-key": "secret"},
                          retries=0)
    assert called == []  # no escalation
    assert got is None
    assert err


def test_request_accepts_a_valid_plain_page_without_bypassing(monkeypatch):
    """A usable plain page short-circuits: the bypass is never consulted."""
    import httpx

    class FakeResp:
        status_code = 200
        headers: dict = {}

        def iter_text(self):
            yield "a page with ok inside"

    class FakeStream:
        def __init__(self, *a, **k):
            self.r = FakeResp()

        def __enter__(self):
            return self.r

        def __exit__(self, *a):
            return False

    class FakeClient:
        def __init__(self, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def stream(self, *a, **k):
            return FakeStream()

    monkeypatch.setattr(httpx, "Client", FakeClient)
    monkeypatch.setattr(W, "_bypass_fetch",
                        lambda ctx, url, validate=None: pytest.fail("should not bypass"))
    got, err = W._request(_ctx(30), "https://example.com", retries=0,
                          validate=lambda b: "ok" in b)
    assert err is None
    assert "ok inside" in got


def test_bypass_honours_interrupt(monkeypatch):
    class Reg:
        def interrupt_check(self):
            return True

    mod, _ = _fake_bypass_module({None: "page"}, poll_interrupt=True)
    monkeypatch.setitem(
        __import__("sys").modules, "opencode_py.tools.cloudflare_bypass", mod)
    with pytest.raises(W._Interrupted):
        W._bypass_fetch(W._Ctx(30, Reg().interrupt_check), "https://x.com",
                        validate=lambda b: True)


def test_bypass_stops_on_deadline_without_raising_interrupt(monkeypatch):
    """An exhausted budget ends the cascade quietly (no _Interrupted)."""
    mod, _ = _fake_bypass_module({None: "page"}, poll_interrupt=True)
    monkeypatch.setitem(
        __import__("sys").modules, "opencode_py.tools.cloudflare_bypass", mod)
    ctx = W._Ctx(30)
    ctx.deadline = __import__("time").monotonic() - 1  # already expired
    body, err = W._bypass_fetch(ctx, "https://x.com", validate=lambda b: True)
    assert body is None
    assert err


# --------------------------------------------------------------------------- #
# 1. round-robin
# --------------------------------------------------------------------------- #
def test_chain_rotates_so_no_engine_always_goes_first():
    W._reset_state()
    now = 1000.0
    firsts = [W._ordered_chain(now)[0] for _ in range(len(W._KEYLESS_CHAIN))]
    assert set(firsts) == set(W._KEYLESS_CHAIN)
    # and every order is a permutation of the full chain
    for _ in range(6):
        assert sorted(W._ordered_chain(now)) == sorted(W._KEYLESS_CHAIN)


# --------------------------------------------------------------------------- #
# 2. circuit breaker
# --------------------------------------------------------------------------- #
def test_failed_engine_is_skipped_while_cooling():
    W._reset_state()
    now = 1000.0
    W._mark_down("duckduckgo", now)
    assert not W._is_up("duckduckgo", now)
    assert W._is_up("duckduckgo", now + W._ENGINE_COOLDOWN + 0.1)
    # a cooled-off engine sinks behind the healthy ones
    order = W._ordered_chain(now)
    assert order.index("duckduckgo") == len(order) - 1


def test_success_clears_the_cool_off_immediately():
    W._reset_state()
    now = 1000.0
    W._mark_down("bing", now)
    W._mark_up("bing")
    assert W._is_up("bing", now)


def test_all_cooling_still_tries_everything():
    W._reset_state()
    now = 1000.0
    for e in W._KEYLESS_CHAIN:
        W._mark_down(e, now)
    assert sorted(W._ordered_chain(now)) == sorted(W._KEYLESS_CHAIN)


def test_auto_mode_actually_rotates_the_first_engine():
    """Regression: the auto path must not always lead with duckduckgo.

    Pinning the auto-resolved engine first defeated the rotation, so one dead
    engine absorbed every search.
    """
    W._reset_state()
    firsts = []
    for i in range(len(W._KEYLESS_CHAIN)):
        seen = []

        def make(name):
            def fn(ctx, q, n, opts):
                seen.append(name)
                return [{"title": name, "url": f"https://{name}{len(seen)}.com",
                         "snippet": ""}], None
            return fn

        for e in W._KEYLESS_CHAIN:
            monkey = W._PROVIDER_FN[e]
            W._PROVIDER_FN[e] = make(e)
        try:
            W.tool().run({"query": f"q{i}", "numResults": 1})
        finally:
            for e in W._KEYLESS_CHAIN:
                W._PROVIDER_FN[e] = monkey
        firsts.append(seen[0])
    assert set(firsts) == set(W._KEYLESS_CHAIN), firsts


def test_explicit_provider_is_pinned_first():
    W._reset_state()
    seen = []

    def make(name):
        def fn(ctx, q, n, opts):
            seen.append(name)
            return [], f"{name}: blocked"
        return fn

    saved = {e: W._PROVIDER_FN[e] for e in W._KEYLESS_CHAIN}
    for e in W._KEYLESS_CHAIN:
        W._PROVIDER_FN[e] = make(e)
    try:
        W.tool().run({"query": "q", "provider": "bing", "numResults": 2})
    finally:
        for e in W._KEYLESS_CHAIN:
            W._PROVIDER_FN[e] = saved[e]
    assert seen[0] == "bing"


def test_a_cooling_keyed_provider_is_skipped_in_auto():
    W._reset_state()
    W._key_cache["tavily"] = "k"  # pretend a key exists
    now = __import__("time").monotonic()
    W._mark_down("tavily", now)
    seen = []

    def make(name):
        def fn(ctx, q, n, opts):
            seen.append(name)
            return [], f"{name}: blocked"
        return fn

    saved = {e: W._PROVIDER_FN[e] for e in W._KEYLESS_CHAIN}
    for e in W._KEYLESS_CHAIN:
        W._PROVIDER_FN[e] = make(e)
    saved_tav = W._PROVIDER_FN["tavily"]
    W._PROVIDER_FN["tavily"] = make("tavily")
    try:
        W.tool().run({"query": "q", "numResults": 2})
    finally:
        for e in W._KEYLESS_CHAIN:
            W._PROVIDER_FN[e] = saved[e]
        W._PROVIDER_FN["tavily"] = saved_tav
    assert "tavily" not in seen


def test_tool_uses_a_cooled_off_engine_last(monkeypatch):
    seen = []

    def make(name, ok):
        def fn(ctx, q, n, opts):
            seen.append(name)
            if ok:
                return [{"title": name, "url": f"https://{name}.com", "snippet": ""}], None
            return [], f"{name}: blocked"
        return fn

    for e in W._KEYLESS_CHAIN:
        monkeypatch.setitem(W._PROVIDER_FN, e, make(e, e == "bing"))
    W._mark_down("bing", __import__("time").monotonic())
    W.tool().run({"query": "x", "numResults": 2, "timeout": 20})
    # bing is cooling: the other two must be tried before it, bing last
    assert seen[-1] == "bing"
    assert set(seen[:-1]) == set(W._KEYLESS_CHAIN) - {"bing"}


# --------------------------------------------------------------------------- #
# 3. result cache
# --------------------------------------------------------------------------- #
def test_cache_hit_avoids_a_second_request(monkeypatch):
    calls = []

    def fake(ctx, q, n, opts):
        calls.append(q)
        return [{"title": f"t{len(calls)}", "url": f"https://a{len(calls)}.com",
                 "snippet": ""}], None

    monkeypatch.setitem(W._PROVIDER_FN, "duckduckgo", fake)
    monkeypatch.setitem(W._PROVIDER_FN, "bing", fake)
    monkeypatch.setitem(W._PROVIDER_FN, "brave", fake)
    t = W.tool()
    a = t.run({"query": "asyncio", "numResults": 2})
    after_first = len(calls)
    b = t.run({"query": "asyncio", "numResults": 2})
    assert after_first >= 1            # first search did real work
    assert len(calls) == after_first   # second search issued zero requests
    assert a["metadata"]["numResults"] == b["metadata"]["numResults"]
    assert b["metadata"].get("cached") is True
    assert "cached" in b["output"]


def test_cache_is_keyed_on_the_filters():
    W._reset_state()
    now = 100.0
    W._cache_put("a", {"output": "A"}, now)
    W._cache_put("b", {"output": "B"}, now)
    assert W._cache_get("a", now)["output"] == "A"
    assert W._cache_get("b", now)["output"] == "B"
    assert W._cache_get("c", now) is None


def test_cache_expires():
    W._reset_state()
    now = 100.0
    W._cache_put("k", {"output": "x"}, now)
    assert W._cache_get("k", now + W._CACHE_TTL - 1) is not None
    assert W._cache_get("k", now + W._CACHE_TTL + 1) is None


def test_cache_is_bounded():
    W._reset_state()
    now = 100.0
    for i in range(W._CACHE_MAX * 2):
        W._cache_put(f"k{i}", {"output": str(i)}, now + i)
    assert len(W._cache) <= W._CACHE_MAX


def test_failures_are_not_cached(monkeypatch):
    calls = []

    def bad(ctx, q, n, opts):
        calls.append(q)
        return [], "blocked"

    for e in W._KEYLESS_CHAIN:
        monkeypatch.setitem(W._PROVIDER_FN, e, bad)
    t = W.tool()
    t.run({"query": "nope", "numResults": 2, "timeout": 20})
    t.run({"query": "nope", "numResults": 2, "timeout": 20})
    assert len(calls) == 2 * len(W._KEYLESS_CHAIN)  # retried, not cached

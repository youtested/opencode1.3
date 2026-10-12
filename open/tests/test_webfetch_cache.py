"""Tests for webfetch's in-memory response cache and its bounded extraction.

Why the cache exists: the agent re-reads the same page while writing its answer,
and a repeat costs a full download (CNN measured 1.27 M characters at 39 s).
Why it is sized by BYTES: one CNN page is ~1.3 MB of str, so an entry-count cap
would be tens of MB on a phone with ~275 MB free.

Fully offline: httpx.Client is faked, so no socket is opened.
"""
from __future__ import annotations

import time

import pytest

from opencode_py.tools import webfetch as F


class FakeResp:
    def __init__(self, text, status=200, ctype="text/html; charset=utf-8"):
        self.status_code = status
        self.headers = {"content-type": ctype}
        self.cookies = None
        self._chunks = [text.encode("utf-8")]

    def iter_bytes(self):
        for c in self._chunks:
            yield c

    def close(self):
        pass


def _install(monkeypatch, body, status=200, ctype="text/html; charset=utf-8"):
    """Make every httpx.Client.stream yield ``body``. Returns a hit counter."""
    import contextlib

    import httpx

    calls = {"n": 0}

    class FakeStream:
        def __init__(self, *a, **kw):
            calls["n"] += 1
            self.r = FakeResp(body, status, ctype)

        def __enter__(self):
            return self.r

        def __exit__(self, *a):
            return False

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        def stream(self, *a, **kw):
            return FakeStream()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(httpx, "Client", FakeClient)
    calls["ctx"] = contextlib
    return calls


@pytest.fixture(autouse=True)
def clean_cache(monkeypatch):
    F._reset_cache()
    monkeypatch.setattr(F, "_cookie_file", lambda d: _NoFile(), raising=False)
    yield
    F._reset_cache()


class _NoFile:
    def __fspath__(self):
        raise OSError("no cookie file")


PAGE = "<html><head><title>T</title></head><body><p>" + ("hello world " * 200) + "</p></body></html>"


# --------------------------------------------------------------------------- #
# cache basics
# --------------------------------------------------------------------------- #
def test_second_fetch_of_the_same_url_is_served_from_memory(monkeypatch):
    calls = _install(monkeypatch, PAGE)
    a = F._webfetch("https://example.com/x", "markdown", 20)
    b = F._webfetch("https://example.com/x", "markdown", 20)
    assert calls["n"] == 1, "the network was hit twice"
    assert b["metadata"]["cached"] is True
    assert "[cached]" in b["output"]
    # the cached copy must carry the same content, minus the marker line
    assert "hello world" in b["output"]
    assert a["metadata"].get("cached") is None


def test_cache_is_keyed_on_the_requested_format(monkeypatch):
    calls = _install(monkeypatch, PAGE)
    F._webfetch("https://example.com/x", "markdown", 20)
    b = F._webfetch("https://example.com/x", "text", 20)
    assert calls["n"] == 2, "a different format must not reuse the markdown body"
    assert b["metadata"].get("cached") is None


def test_cache_is_keyed_on_the_url(monkeypatch):
    calls = _install(monkeypatch, PAGE)
    F._webfetch("https://example.com/a", "markdown", 20)
    F._webfetch("https://example.com/b", "markdown", 20)
    assert calls["n"] == 2


def test_a_changed_cookie_jar_invalidates_the_entry(monkeypatch):
    """A login/logout must not keep serving the pre-login page from memory."""
    calls = _install(monkeypatch, PAGE)
    stamps = {"v": (1, 1)}
    monkeypatch.setattr(F, "_cookie_stamp", lambda d: stamps["v"])
    F._webfetch("https://example.com/x", "markdown", 20)
    assert F._webfetch("https://example.com/x", "markdown",
                       20)["metadata"]["cached"] is True
    stamps["v"] = (2, 2)  # the jar changed
    assert F._webfetch("https://example.com/x", "markdown",
                       20)["metadata"].get("cached") is None
    assert calls["n"] == 2


def test_expired_entries_are_not_reused(monkeypatch):
    calls = _install(monkeypatch, PAGE)
    F._webfetch("https://example.com/x", "markdown", 20)
    for k in list(F._CACHE):
        F._CACHE[k] = (time.monotonic() - 1.0, *F._CACHE[k][1:])
    assert F._webfetch("https://example.com/x", "markdown",
                       20)["metadata"].get("cached") is None
    assert calls["n"] == 2


def test_a_failed_fetch_is_not_cached(monkeypatch):
    _install(monkeypatch, "nope", status=500)
    F._reset_cache()
    F._webfetch("https://example.com/x", "markdown", 20)
    assert not F._CACHE, "a 500 must not poison the cache"


# --------------------------------------------------------------------------- #
# byte budget — the reason this cache is not entry-capped
# --------------------------------------------------------------------------- #
def test_an_oversized_page_is_not_cached(monkeypatch):
    _install(monkeypatch, "<p>" + ("x" * (F._CACHE_MAX_ENTRY + 10)) + "</p>")
    F._reset_cache()
    F._webfetch("https://example.com/big", "markdown", 20)
    assert F._CACHE == {}, "a page over the per-entry cap must not be held"


def test_the_cache_respects_its_total_byte_budget():
    """One CNN page is ~1.3 M chars; the budget must cap total memory."""
    F._reset_cache()
    chunk = "y" * 400_000
    now = time.monotonic()
    for i in range(20):
        F._cache_put(("u%d" % i, "markdown", (0, 0)), chunk, {"i": i}, now + i)
        total = sum(len(v[1]) for v in F._CACHE.values())
        assert total <= F._CACHE_MAX_BYTES, "cache grew to %d bytes" % total
    assert len(F._CACHE) < 20, "entries should have been evicted"


def test_reset_cache_empties_everything():
    F._reset_cache()
    F._cache_put(("u", "markdown", (0, 0)), "body", {}, time.monotonic())
    assert F._CACHE
    F._reset_cache()
    assert F._CACHE == {}


def test_cache_helpers_never_raise_on_junk():
    F._reset_cache()
    assert F._cache_get(("missing",), time.monotonic()) is None
    F._cache_put(("k",), "b", {}, time.monotonic())  # unhashable metadata is fine
    assert F._cookie_stamp("nonexistent.invalid") == (0, 0)

def test_the_page_range_is_part_of_the_cache_key(monkeypatch):
    """A cached whole-document read must never be served for a pages=1-2
    request: the reader asked for two pages and would silently get all of them."""
    calls = _install(monkeypatch, PAGE)
    whole = F._webfetch("https://example.com/doc.pdf", "markdown", 20)
    ranged = F._webfetch("https://example.com/doc.pdf", "markdown", 20, pages="1-2")
    assert calls["n"] == 2, "the page range reused the whole-document entry"
    assert ranged["metadata"].get("cached") is None
    assert whole["metadata"].get("cached") is None


def test_the_same_page_range_does_reuse_the_cache(monkeypatch):
    calls = _install(monkeypatch, PAGE)
    F._webfetch("https://example.com/doc.pdf", "markdown", 20, pages="1-2")
    again = F._webfetch("https://example.com/doc.pdf", "markdown", 20, pages="1-2")
    assert calls["n"] == 1
    assert again["metadata"]["cached"] is True


def test_different_ranges_do_not_share_an_entry(monkeypatch):
    calls = _install(monkeypatch, PAGE)
    F._webfetch("https://example.com/doc.pdf", "markdown", 20, pages="1-2")
    F._webfetch("https://example.com/doc.pdf", "markdown", 20, pages="3-4")
    assert calls["n"] == 2

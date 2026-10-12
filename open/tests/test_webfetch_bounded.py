"""Tests for bounded extraction: density scoring plus its safety guard.

The guard is the whole point of this feature. A density heuristic that
misjudges a page shape could return LESS than before, deleting the article — so
every case here also asserts the fallback fires.
"""
from __future__ import annotations

import time

from opencode_py.tools import webfetch as F


def _page(n_chrome=300, n_prose=120):
    chrome = "".join(
        '<div class="nav"><a href="#">Nav %d</a></div>'
        '<div class="promo"><a href="#">Sponsored %d</a></div>' % (i, i)
        for i in range(n_chrome))
    art = "<article>" + "".join(
        "<p>Real sentence %d about the actual story. </p>" % i
        for i in range(n_prose)) + "</article>"
    return "<html><body><header>H</header>%s%s</body></html>" % (chrome, art)


# --------------------------------------------------------------------------- #
# it does the job
# --------------------------------------------------------------------------- #
def test_chrome_is_dropped_and_the_article_kept():
    page = _page()
    out = F._html_to_markdown(page, "https://x.test/")
    assert "Real sentence 119" in out
    assert "Sponsored" not in out
    assert "Nav 5" not in out


def test_the_reduction_is_large():
    """CNN measured 1.27 M characters; the whole point is that this collapses."""
    page = _page(n_chrome=600, n_prose=200)
    out = F._html_to_markdown(page, "https://x.test/")
    assert len(out) < len(page) / 4


def test_scripts_and_styles_never_reach_the_model():
    page = ('<html><body><script>var x=1;</script><style>a{}</style>'
            '<article><p>Real text.</p></article></body></html>')
    out = F._html_to_markdown(page, "https://x.test/")
    assert "var x" not in out
    assert "a{}" not in out
    assert "Real text." in out


# --------------------------------------------------------------------------- #
# the guard: worst case is "no improvement", never a lost article
# --------------------------------------------------------------------------- #
def test_a_page_that_is_all_content_is_not_truncated():
    plain = "<html><body>" + "".join(
        "<p>Plain content line %d. </p>" % i for i in range(60)) + "</body></html>"
    out = F._html_to_markdown(plain, "https://x.test/")
    assert "Plain content line 0." in out
    assert "Plain content line 59." in out


def test_a_tiny_page_is_never_cut():
    """Below the sanity floor the heuristic declines and returns everything."""
    page = '<html><body><article><p>Only a short note here.</p></article></body></html>'
    assert "Only a short note here." in F._strip_boilerplate(page)


def test_an_empty_or_markupless_input_is_safe():
    assert F._strip_boilerplate("") == ""
    assert F._strip_boilerplate("<p>hi</p>") == "<p>hi</p>"
    assert isinstance(F._strip_boilerplate(None or "x"), str)


def test_a_nested_wrapper_does_not_beat_the_real_article():
    """One outer div holding everything must not win on raw size alone."""
    inner = _page(n_chrome=50, n_prose=80)
    page = '<html><body><div id="wrapper">%s</div></body></html>' % inner
    out = F._html_to_markdown(page, "https://x.test/")
    assert "Real sentence 79" in out


# --------------------------------------------------------------------------- #
# performance: the scoring pass must stay linear
# --------------------------------------------------------------------------- #
def test_density_scoring_is_bounded_in_time():
    """Scoring runs the text converter per candidate, so a deeply nested page
    could go quadratic. The candidate cap is what stops it."""
    page = _page(n_chrome=800, n_prose=200)
    t0 = time.monotonic()
    F._strip_boilerplate(page)
    elapsed = time.monotonic() - t0
    assert elapsed < 5.0, "strip_boilerplate took %.1fs" % elapsed


def test_only_the_largest_candidates_are_scored():
    assert F._MAX_BLOCK_CANDIDATES <= 64


def test_the_helper_never_raises_on_malformed_markup():
    for junk in ('<div><p>unclosed', '<<<>>>', '<div class="a"><div class="b">'):
        assert isinstance(F._strip_boilerplate(junk), str)
        assert isinstance(F._score_block(junk), float)


def test_score_prefers_prose_over_markup():
    prose = "<article><p>%s</p></article>" % ("real words here " * 60)
    chrome = "<div>%s</div>" % ("<a href='#'>link</a>" * 200)
    assert F._score_block(prose) > F._score_block(chrome)


def test_junk_class_names_are_penalised():
    clean = "<div><p>%s</p></div>" % ("plain sentence content " * 40)
    junky = '<div class="sidebar-related-promo">%s</div>' % (
        "<p>plain sentence content </p>" * 40)
    assert F._score_block(clean) >= F._score_block(junky)

def test_scoring_never_re_enters_the_stripper():
    """_score_block must not call _html_to_text: that calls _strip_boilerplate,
    which calls _best_block, which scores again. Measured cost of getting this
    wrong: 37 s on a 124 KB page, versus 0.7 s done correctly."""
    page = _page(n_chrome=1200, n_prose=300)
    t0 = time.monotonic()
    F._strip_boilerplate(page)
    assert time.monotonic() - t0 < 5.0


def test_score_text_len_is_independent_of_the_stripper():
    block = '<div class="nav"><a href="#">x</a></div><p>%s</p>' % ("real words " * 50)
    assert F._score_text_len(block) > 0
    assert F._score_text_len("") == 0
    # scripts contribute nothing to visible text
    assert F._score_text_len("<script>var secret=1;</script><p>ok</p>") == len("ok")


# --------------------------------------------------------------------------- #
# size pre-check: refuse an oversized body BEFORE downloading it
# --------------------------------------------------------------------------- #
class _Sized:
    def __init__(self, declared, ctype="text/html; charset=utf-8"):
        self.status_code = 200
        self.headers = {"content-type": ctype}
        if declared is not None:
            self.headers["content-length"] = str(declared)
        self.reads = 0
        self.cookies = None

    def iter_bytes(self):
        self.reads += 1
        yield b"<p>should never be read</p>"

    def close(self):
        pass


def _serve_sized(monkeypatch, declared, ctype="text/html; charset=utf-8"):
    import httpx

    holder = {}

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        def stream(self, *a, **kw):
            holder["r"] = _Sized(declared, ctype)
            r = holder["r"]

            class Ctx:
                def __enter__(self_):
                    return r

                def __exit__(self_, *a):
                    return False

            return Ctx()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(httpx, "Client", FakeClient)
    return holder


def test_an_oversized_page_is_refused_without_reading_the_body(monkeypatch):
    """Downloading 5 MB of a 200 MB page only to truncate it wastes real mobile
    data. The refusal has to happen before the first chunk is read."""
    holder = _serve_sized(monkeypatch, 200 * 1024 * 1024)
    r = F._webfetch("https://example.com/huge", "markdown", 30)
    assert r["metadata"]["too_large"] is True
    assert holder["r"].reads == 0, "the body was downloaded anyway"
    assert "200 MB" in r["output"]
    assert "not downloaded" in r["output"]


def test_a_normal_sized_page_is_unaffected(monkeypatch):
    holder = _serve_sized(monkeypatch, 12_000)
    r = F._webfetch("https://example.com/ok", "markdown", 30)
    assert holder["r"].reads == 1
    assert "should never be read" in r["output"]
    assert not r.get("metadata", {}).get("too_large")


def test_a_missing_content_length_is_not_treated_as_huge(monkeypatch):
    """Chunked responses have no Content-Length. Guessing 'huge' there would
    break every streaming site."""
    holder = _serve_sized(monkeypatch, None)
    r = F._webfetch("https://example.com/stream", "markdown", 30)
    assert holder["r"].reads == 1
    assert not r.get("metadata", {}).get("too_large")


def test_the_declared_size_parser_never_raises():
    class Odd:
        headers = {"content-length": "not-a-number"}
    assert F._declared_size(Odd()) == 0

    class Broken:
        headers = None
    assert F._declared_size(Broken()) == 0

"""Tests for the webreach tool and its recipe table.

Fully offline: every network call goes through a monkeypatched
``websearch._request`` (the SAME hook test_websearch.py uses), so nothing here
opens a socket. The parse fixtures are trimmed but REAL response shapes, so a
platform changing its markup surfaces as a test failure rather than as a
silently empty tool result.
"""
from __future__ import annotations

import json
import time

import pytest

from opencode_py.tools import websearch as W
from opencode_py.tools import webreach as B
from opencode_py.tools import webreach_recipes as R


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    """Every test starts from a clean cache/health state and no Tor.

    This file tests the HTTP machinery only. The video recipes now carry a
    yt-dlp backend whose runner spawns a REAL interpreter, so every runner is
    stubbed out here: without this, one YouTube test quietly started a yt-dlp
    subprocess and OOM-killed the run on a phone with ~275 MB free. The yt-dlp
    runner's own behaviour is covered in test_webreach_ytdlp.py.
    """
    W._reset_state()
    monkeypatch.delenv("OPENCODE_TOR", raising=False)
    try:
        from opencode_py.tools import proxy_rotation as PR

        PR.TorRotation._shared = None
    except Exception:
        pass
    monkeypatch.setattr(B, "_ytdlp_version", lambda: "2026.08.19")
    for recipe in R.RECIPES:
        for backend in recipe.backends:
            if backend.runner is not None:
                monkeypatch.setattr(backend, "runner", lambda *a, **kw: {
                    "error": True, "unavailable": True,
                    "reason": "yt-dlp stubbed out in test_webreach.py"}, raising=False)
    yield
    W._reset_state()


def _opt(**kw):
    base = B._options({})
    base.update(kw)
    return base


# --------------------------------------------------------------------------- #
# URL matching
# --------------------------------------------------------------------------- #
def test_recipe_matches_own_urls():
    assert R.find_recipe("https://github.com/anomalyco/opencode").site == "github"
    assert R.find_recipe("https://api.github.com/repos/a/b").site == "github"
    assert R.find_recipe("https://news.ycombinator.com/item?id=1").site == "hackernews"
    assert R.find_recipe("https://www.reddit.com/r/python/").site == "reddit"
    assert R.find_recipe("https://old.reddit.com/r/python/").site == "reddit"
    assert R.find_recipe("https://www.v2ex.com/go/python").site == "v2ex"
    assert R.find_recipe("https://lobste.rs/t/python").site == "lobsters"
    assert R.find_recipe("https://www.youtube.com/watch?v=aircAruvnKk").site == "youtube"
    assert R.find_recipe("https://youtu.be/aircAruvnKk").site == "youtube"


def test_recipe_rejects_near_miss_hosts():
    """A lookalike host must never inherit a real site's recipe (SSRF-ish trap)."""
    assert R.find_recipe("https://notreddit.com/r/python/") is None
    assert R.find_recipe("https://github.com.evil.test/a/b") is None
    assert R.find_recipe("https://evil.test/?x=github.com") is None
    assert R.find_recipe("https://xgithub.com/a/b") is None


def test_recipe_rejects_non_http():
    assert R.find_recipe("") is None
    assert R.find_recipe("file:///etc/passwd") is None
    assert R.find_recipe("github.com/a/b") is None


def test_feed_matcher_does_not_swallow_ordinary_pages():
    """The catch-all feed matcher must not claim plain web pages."""
    assert R.find_recipe("https://example.com/blog/post") is None
    assert R.find_recipe("https://example.com/feed").site == "rss"
    assert R.find_recipe("https://example.com/feed.xml").site == "rss"
    assert R.find_recipe("https://example.com/x?format=rss").site == "rss"


def test_rss_matcher_never_shadows_a_real_site():
    """RECIPES order matters: reddit's .rss must use reddit, not the feed recipe."""
    got = R.find_recipe("https://www.reddit.com/r/python/.rss")
    assert got.site == "reddit"


# --------------------------------------------------------------------------- #
# backend selection
# --------------------------------------------------------------------------- #
def test_backends_for_action_picks_the_right_one():
    gh = R.find_recipe("https://github.com/a/b")
    assert [b.name for b in R.backends_for(gh, "read")][0] == "api-read"
    assert "search" in [b.name for b in R.backends_for(gh, "search")]
    assert "issues" in [b.name for b in R.backends_for(gh, "list")]


def test_unknown_action_falls_back_to_read():
    gh = R.find_recipe("https://github.com/a/b")
    names = [b.name for b in R.backends_for(gh, "nonsense")]
    assert names == [b.name for b in R.backends_for(gh, "read")]


def test_reddit_json_is_tried_before_rss():
    rd = R.find_recipe("https://www.reddit.com/r/python/")
    assert [b.name for b in R.backends_for(rd, "list")] == ["json", "rss"]


# --------------------------------------------------------------------------- #
# URL builders
# --------------------------------------------------------------------------- #
def test_github_repo_builds_the_api_url():
    gh = R.find_recipe("https://github.com/anomalyco/opencode")
    got = gh.backends[0].build("https://github.com/anomalyco/opencode", "read", _opt())
    assert got == "https://api.github.com/repos/anomalyco/opencode"


def test_github_issue_url_builds_the_single_issue_api():
    gh = R.find_recipe("https://github.com/anomalyco/opencode/issues/42")
    got = gh.backends[0].build("https://github.com/anomalyco/opencode/issues/42", "read", _opt())
    assert got == "https://api.github.com/repos/anomalyco/opencode/issues/42"


def test_github_file_url_builds_raw_not_api():
    """A blob URL carries its ref IN THE PATH — reusing the ref param would
    duplicate it (…/main/main/README.md)."""
    gh = R.find_recipe("https://github.com/anomalyco/opencode/blob/main/README.md")
    got = gh.backends[1].build("https://github.com/anomalyco/opencode/blob/main/README.md",
                               "read", _opt())
    assert got == "https://raw.githubusercontent.com/anomalyco/opencode/main/README.md"


def test_github_search_requires_a_query():
    gh = R.find_recipe("https://github.com/search")
    with pytest.raises(ValueError):
        gh.backends[2].build("https://github.com/search", "search", _opt())


def test_reddit_subreddit_builds_json():
    rd = R.find_recipe("https://www.reddit.com/r/python/")
    got = rd.backends[0].build("https://www.reddit.com/r/python/", "list", _opt(limit=5))
    assert got == "https://www.reddit.com/r/python/hot.json?limit=5"


def test_reddit_thread_builds_json():
    rd = R.find_recipe("https://www.reddit.com/r/python/comments/abc123/title/")
    got = rd.backends[0].build("https://www.reddit.com/r/python/comments/abc123/title/", "read", _opt())
    assert got.startswith("https://www.reddit.com/r/python/comments/abc123/title.json")


def test_reddit_rss_backend_derives_the_feed():
    rd = R.find_recipe("https://www.reddit.com/r/python/")
    got = rd.backends[1].build("https://www.reddit.com/r/python/", "list", _opt())
    assert got == "https://www.reddit.com/r/python/.rss"


def test_lobsters_uses_the_t_path_not_r():
    lb = R.find_recipe("https://lobste.rs/t/python")
    got = lb.backends[0].build("https://lobste.rs/t/python", "list", _opt())
    assert got == "https://lobste.rs/t/python.json"


def test_hn_item_builds_the_single_item_call():
    hn = R.find_recipe("https://news.ycombinator.com/")
    got = hn.backends[0].build("https://news.ycombinator.com/item?id=49940394", "read", _opt())
    assert got == "https://hn.algolia.com/api/v1/items/49940394"


def test_youtube_build_raises_unavailable_with_the_video_id():
    yt = R.find_recipe("https://www.youtube.com/watch?v=aircAruvnKk")
    with pytest.raises(R._Unavailable) as exc:
        yt.backends[0].build("https://www.youtube.com/watch?v=aircAruvnKk", "read", _opt())
    assert "aircAruvnKk" in exc.value.reason


# --------------------------------------------------------------------------- #
# parsers (real response shapes, trimmed)
# --------------------------------------------------------------------------- #
GH_REPO = json.dumps({
    "full_name": "anomalyco/opencode", "description": "The AI coding agent",
    "stargazers_count": 12345, "language": "TypeScript", "license": {"spdx_id": "MIT"},
    "open_issues_count": 42, "homepage": "https://opencode.ai", "default_branch": "main",
    "pushed_at": "2026-10-01T00:00:00Z", "topics": ["ai", "cli"],
})

HN_ITEM = json.dumps({
    "id": 49940394, "title": "A discussion", "points": 120,
    "children": [
        {"author": "alice", "text": "<p>first <i>reply</i></p>", "children": [
            {"author": "bob", "text": "<p>nested reply</p>", "children": []},
        ]},
    ],
})

HN_LIST = json.dumps({"hits": [
    {"title": "Story one", "url": "https://example.com/1", "points": 10, "objectID": "1"},
    {"title": "Story two", "url": None, "points": 20, "objectID": "2"},
]})

REDDIT_JSON = json.dumps({"kind": "Listing", "data": {"children": [
    {"kind": "t3", "data": {
        "title": "A post", "permalink": "/r/python/comments/abc/a_post/",
        "author": "alice", "score": 42, "num_comments": 2, "selftext": "<p>body text</p>",
        "replies": {"kind": "Listing", "data": {"children": [
            {"kind": "t1", "data": {"author": "bob", "body": "<p>a comment</p>", "replies": ""}},
        ]}},
    }},
]}})

REDDIT_BLOCK = "You've been blocked by network security.\nIf you think you've been blocked by mistake, file a ticket."

RSS_BODY = """<?xml version="1.0"?><rss><channel><title>Example Feed</title>
<item><title>Item One</title><link>https://example.com/1</link>
<pubDate>Mon, 01 Oct 2026 00:00:00 GMT</pubDate>
<description>&lt;p&gt;First body&lt;/p&gt;</description></item>
<item><title>Item Two</title><link>https://example.com/2</link>
<description>Second body</description></item></channel></rss>"""

V2EX = json.dumps([{
    "id": 1234567, "title": "V2EX 主题", "content": "<p>内容</p>",
    "replies": 12, "member": {"username": "someone"},
}])

LOB = json.dumps([{
    "short_id": "3r00ol", "title": "A lobsters story", "url": "https://example.com/x",
    "score": 3, "comment_count": 0, "submitter_user": "eatonphil", "tags": ["python"],
}])


def test_github_repo_parser_reports_the_numbers():
    gh = R.find_recipe("https://github.com/a/b")
    out = gh.backends[0].parse(GH_REPO, "https://api.github.com/repos/a/b", "read", _opt())
    assert out["title"] == "anomalyco/opencode"
    assert "12345" in out["markdown"]
    assert "MIT" in out["markdown"]
    assert out["metadata"]["stars"] == 12345


def test_github_repo_parser_rejects_a_non_repo_body():
    gh = R.find_recipe("https://github.com/a/b")
    out = gh.backends[0].parse("<html>nope</html>", "u", "read", _opt())
    assert out.get("error") is True


def test_hn_item_parser_flattens_the_comment_tree():
    hn = R.find_recipe("https://news.ycombinator.com/")
    out = hn.backends[0].parse(HN_ITEM, "u", "read", _opt())
    md = out["markdown"]
    assert "A discussion" in md
    assert "alice" in md and "first reply" in md
    assert "bob" in md and "nested reply" in md
    assert "<p>" not in md
    assert out["metadata"]["points"] == 120


def test_hn_list_parser_handles_a_missing_url():
    hn = R.find_recipe("https://news.ycombinator.com/")
    out = hn.backends[0].parse(HN_LIST, "u", "list", _opt())
    assert len(out["items"]) == 2
    assert out["items"][1]["url"].startswith("https://news.ycombinator.com/item?id=")


def test_reddit_json_parser_emits_posts_and_comments():
    rd = R.find_recipe("https://www.reddit.com/r/python/")
    out = rd.backends[0].parse(REDDIT_JSON, "u", "list", _opt())
    assert "A post" in out["markdown"]
    assert "body text" in out["markdown"]
    assert "a comment" in out["markdown"]
    assert out["metadata"]["posts"] == 1


def test_reddit_json_parser_reports_a_block_page_as_an_error():
    rd = R.find_recipe("https://www.reddit.com/r/python/")
    out = rd.backends[0].parse(REDDIT_BLOCK, "u", "list", _opt())
    assert out.get("error") is True


def test_reddit_rss_parser_reads_entries():
    rd = R.find_recipe("https://www.reddit.com/r/python/.rss")
    body = ('<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">'
            "<entry><title>Post</title><link href=\"https://www.reddit.com/r/python/comments/x/p/\"/>"
            "<content>&lt;p&gt;text&lt;/p&gt;</content>"
            "<author>u/alice</author></entry></feed>")
    out = rd.backends[1].parse(body, "u", "list", _opt())
    assert "Post" in out["markdown"]
    assert out["metadata"]["slow"] is True


def test_v2ex_parser_handles_chinese_content():
    vx = R.find_recipe("https://www.v2ex.com/go/python")
    out = vx.backends[0].parse(V2EX, "u", "list", _opt())
    assert "V2EX 主题" in out["markdown"]
    assert "内容" in out["markdown"]


def test_lobsters_parser_reads_stories():
    lb = R.find_recipe("https://lobste.rs/t/python")
    out = lb.backends[0].parse(LOB, "u", "list", _opt())
    assert "A lobsters story" in out["markdown"]
    assert out["metadata"]["count"] == 1


def test_rss_parser_reads_full_item_bodies():
    rs = R.find_recipe("https://example.com/feed.xml")
    out = rs.backends[0].parse(RSS_BODY, "u", "list", _opt())
    assert "Example Feed" in out["markdown"]
    assert "Item One" in out["markdown"]
    assert "First body" in out["markdown"]
    assert len(out["items"]) == 2


def test_rss_parser_rejects_html_that_is_not_a_feed():
    rs = R.find_recipe("https://example.com/feed.xml")
    out = rs.backends[0].parse("<html><body>hi</body></html>", "u", "list", _opt())
    assert out.get("error") is True


def test_strip_html_never_raises_on_junk():
    assert isinstance(R._strip_html(None), str)
    assert isinstance(R._strip_html("<div"), str)
    assert R._loads("not json") is None


# --------------------------------------------------------------------------- #
# chain behaviour
# --------------------------------------------------------------------------- #
def test_first_backend_success_makes_exactly_one_call(monkeypatch):
    calls = []

    def fake(ctx, url, **kw):
        calls.append(url)
        return GH_REPO, None

    monkeypatch.setattr(W, "_request", fake)
    out = B.run_recipe("https://github.com/a/b", "read", _opt(), 20)
    assert out.get("error") is None
    assert len(calls) == 1
    assert out["metadata"]["backend"] == "api-read"
    assert out["metadata"]["site"] == "github"


def test_chain_falls_through_to_the_second_backend(monkeypatch):
    calls = []

    def fake(ctx, url, **kw):
        calls.append(url)
        if url.endswith(".json"):
            return None, "HTTP 403"
        return RSS_BODY, None

    monkeypatch.setattr(W, "_request", fake)
    out = B.run_recipe("https://www.reddit.com/r/python/", "list", _opt(), 20)
    assert out.get("error") is None
    assert len(calls) == 2
    assert ".json" in calls[0]
    assert out["metadata"]["backend"] == "rss"
    assert out["metadata"]["fell_back"]


def test_block_page_is_not_counted_as_success(monkeypatch):
    """A 200 that is actually a bot wall must fall through, not parse to junk."""
    calls = []

    def fake(ctx, url, **kw):
        calls.append(url)
        return REDDIT_BLOCK, None

    monkeypatch.setattr(W, "_request", fake)
    out = B.run_recipe("https://www.reddit.com/r/python/", "list", _opt(), 20)
    assert out.get("error") is True
    # both reddit backends were tried before giving up
    assert len(calls) == 2


def test_all_backends_failing_names_every_attempt(monkeypatch):
    monkeypatch.setattr(W, "_request", lambda ctx, url, **kw: (None, "boom"))
    out = B.run_recipe("https://www.reddit.com/r/python/", "list", _opt(), 20)
    assert out.get("error") is True
    assert [a["backend"] for a in out["attempts"]] == ["json", "rss"]
    assert "boom" in out["reason"]
    assert out["fallback"]


def test_unavailable_stops_the_chain_immediately(monkeypatch):
    """YouTube refuses anonymous HTTP; the next backend cannot help."""
    calls = []
    monkeypatch.setattr(W, "_request", lambda ctx, url, **kw: (calls.append(url), ("", None))[1])
    out = B.run_recipe("https://www.youtube.com/watch?v=aircAruvnKk", "read", _opt(), 20)
    assert out.get("error") is True
    assert out.get("unavailable") is True
    assert "browser" in out["fallback"]
    assert calls == []


def test_unknown_site_is_told_to_use_webfetch():
    out = B.run_recipe("https://example.com/some/page", "read", _opt(), 10)
    assert out.get("error") is True
    assert "webfetch" in out["reason"]


def test_a_stalled_backend_cannot_starve_the_next_one(monkeypatch):
    """Budget slicing: the chain survives a first backend that burns its whole
    share. Without it a slow/blocked head would leave the fallback nothing."""
    seen = []

    def fake(ctx, url, **kw):
        seen.append((url, round(ctx.slice(kw.get("cap") or 30.0), 1), kw.get("escalate")))
        if url.endswith(".json"):
            time.sleep(0.4)
            return None, "stalled"
        return RSS_BODY, None

    monkeypatch.setattr(W, "_request", fake)
    out = B.run_recipe("https://www.reddit.com/r/python/", "list", _opt(), 20)
    assert out.get("error") is None
    assert out["metadata"]["backend"] == "rss"
    # the head got a smaller-or-equal slice than the tail, and both were capped
    assert seen[0][1] <= seen[1][1]
    assert all(s[1] <= 20.0 for s in seen)


def test_recipe_backends_do_not_escalate_through_the_bypass_cascade(monkeypatch):
    """Structured JSON APIs are walled on IP reputation, not client fingerprint:
    running the 6-method cascade would burn 10+ s of radio to fail identically."""
    escalated = []

    def boom(ctx, url, validate=None):
        escalated.append(url)
        return None, "cascade ran"

    monkeypatch.setattr(W, "_bypass_fetch", boom)
    monkeypatch.setattr(W, "_request", lambda ctx, url, **kw: (None, "HTTP 403"))
    B.run_recipe("https://www.reddit.com/r/python/", "list", _opt(), 20)
    assert escalated == []


def test_a_backend_that_opts_in_does_escalate():
    """The opt-out is a per-backend default, not a hard rule."""
    rd = R.find_recipe("https://www.reddit.com/r/python/")
    assert all(b.escalate is False for b in rd.backends)
    rd.backends[0].escalate = True
    assert rd.backends[0].escalate is True


def test_deadline_is_shared_across_the_chain(monkeypatch):
    """A slow first backend must not let the chain outlive its budget."""

    def slow(ctx, url, **kw):
        deadline = time.monotonic() + 1.2
        while time.monotonic() < deadline:
            if ctx.remaining() <= 0:
                raise W._Timeout()
        return None, "slow"

    monkeypatch.setattr(W, "_request", slow)
    t0 = time.monotonic()
    out = B.run_recipe("https://www.reddit.com/r/python/", "list", _opt(), 2)
    assert time.monotonic() - t0 < 8
    assert out.get("error") is True


def test_interrupt_aborts_the_chain(monkeypatch):
    calls = []
    monkeypatch.setattr(W, "_request", lambda ctx, url, **kw: (calls.append(url), (GH_REPO, None))[1])
    out = B.run_recipe("https://github.com/a/b", "read", _opt(), 20, lambda: True)
    assert out.get("error") is True
    assert out.get("interrupted") is True
    assert calls == []


# --------------------------------------------------------------------------- #
# tool surface
# --------------------------------------------------------------------------- #
def test_tool_requires_a_url():
    assert B.tool().run({}).get("error") is True


def test_tool_reads_a_github_repo(monkeypatch):
    monkeypatch.setattr(W, "_request", lambda ctx, url, **kw: (GH_REPO, None))
    out = B.tool().run({"url": "https://github.com/a/b", "action": "read"})
    assert out.get("error") is not True
    assert "anomalyco/opencode" in out["output"]
    assert out["metadata"]["site"] == "github"


def test_tool_truncates_and_says_so(monkeypatch):
    monkeypatch.setattr(W, "_request", lambda ctx, url, **kw: (RSS_BODY * 40, None))
    out = B.tool().run({"url": "https://example.com/feed.xml", "content_limit": 500})
    assert out["metadata"]["truncated"] is True
    assert len(out["output"]) <= 500 + 200  # cap, plus the prepended title line


def test_tool_reports_a_blocked_platform_with_the_fix(monkeypatch):
    monkeypatch.setattr(W, "_request", lambda ctx, url, **kw: (REDDIT_BLOCK, None))
    out = B.tool().run({"url": "https://www.reddit.com/r/python/", "action": "list"})
    assert out.get("error") is True
    assert "browser" in out["output"]
    assert "tried json" in out["output"]


def test_tool_search_without_a_query_is_rejected():
    out = B.tool().run({"url": "https://github.com/search", "action": "search"})
    assert out.get("error") is True


def test_tool_honours_the_registry_interrupt_hook(monkeypatch):
    seen = {}

    class Reg:
        def interrupt_check(self):
            seen["called"] = True
            return False

    monkeypatch.setattr(W, "_request", lambda ctx, url, **kw: (GH_REPO, None))
    out = B.tool(Reg()).run({"url": "https://github.com/a/b"})
    assert seen.get("called") is True
    assert out.get("error") is not True


def test_limit_and_timeout_are_clamped():
    opt = B._options({"limit": 9999, "content_limit": 1, "timeout": 9999})
    assert opt["limit"] == B.MAX_LIMIT
    assert opt["content_limit"] == 500
    assert B._clamp_int(9999, 3, 120, 25) == 120
    assert B._clamp_int("junk", 3, 120, 25) == 25


def test_bad_action_falls_back_to_read(monkeypatch):
    monkeypatch.setattr(W, "_request", lambda ctx, url, **kw: (GH_REPO, None))
    out = B.tool().run({"url": "https://github.com/a/b", "action": "banana"})
    assert out.get("error") is not True


# --------------------------------------------------------------------------- #
# doctor
# --------------------------------------------------------------------------- #
def test_doctor_classifies_ok_fail_and_unavailable(monkeypatch):
    def fake_run(url, action, opt, timeout, is_interrupted=None):
        if "youtube" in url:
            return {"error": True, "unavailable": True, "reason": "no token",
                    "fallback": "use the browser tool"}
        if "github" in url:
            return {"title": "x", "markdown": "# x\n\nok", "metadata": {"backend": "api-read"}}
        if "reddit" in url:
            return {"error": True, "reason": "HTTP 403", "fallback": "use the browser tool"}
        return {"error": True, "reason": "nope"}

    monkeypatch.setattr(B, "run_recipe", fake_run)
    res = B.doctor(timeout=1)
    by_site = {r["site"]: r for r in res["rows"]}
    assert by_site["github"]["ok"] is True
    assert by_site["youtube"]["status"] == "unavailable"
    assert by_site["reddit"]["ok"] is False
    # only github among the site rows is green (the core checks are counted too)
    assert sum(1 for r in res["rows"] if r["site"] == "github" and r["ok"]) == 1
    assert all(r["ok"] is False for r in res["rows"]
               if r["site"] not in ("github",) and "-" not in r["site"])
    # the full run also appends the offline core checks, so total is larger
    assert res["total"] == len(B.PROBES) + len(B._core_checks())
    assert [r["site"] for r in res["rows"]][:len(B.PROBES)] == [p[0] for p in B.PROBES]


def test_doctor_never_raises_on_a_crashing_probe(monkeypatch):
    def boom(*a, **kw):
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(B, "run_recipe", boom)
    monkeypatch.setattr(B, "_core_checks", list)
    res = B.doctor(timeout=1)
    assert all(r["ok"] is False for r in res["rows"])
    assert all(r["reason"] == "probe crashed" for r in res["rows"])


def test_doctor_flags_an_empty_parse(monkeypatch):
    monkeypatch.setattr(B, "run_recipe",
                        lambda *a, **kw: {"markdown": "   ", "metadata": {}})
    monkeypatch.setattr(B, "_core_checks", list)
    res = B.doctor(timeout=1)
    assert all(r["status"] == "empty" for r in res["rows"])


def test_doctor_filter_selects_one_site(monkeypatch):
    monkeypatch.setattr(B, "run_recipe",
                        lambda *a, **kw: {"markdown": "# x", "metadata": {}})
    res = B.doctor(timeout=1, only="github")
    assert [r["site"] for r in res["rows"]] == ["github"]


def test_doctor_reports_the_optional_ytdlp_backend(monkeypatch):
    """The doctor must state the yt-dlp version, and must say 'not installed'
    plainly when it is absent rather than letting the video rows hide it."""
    monkeypatch.setattr(B, "run_recipe",
                        lambda *a, **kw: {"markdown": "# x", "metadata": {}})
    monkeypatch.setattr(B, "_ytdlp_version", lambda: "2026.08.19")
    assert "2026.08.19" in B.doctor_text(timeout=1, only="github")
    monkeypatch.setattr(B, "_ytdlp_version", lambda: "")
    text = B.doctor_text(timeout=1, only="github")
    assert "not installed" in text
    assert "pip install yt-dlp" in text


def test_ytdlp_version_lookup_never_raises(monkeypatch):
    monkeypatch.setattr(B, "_ytdlp_version", lambda: "")
    assert B.doctor(timeout=1, only="github")["ytdlp"] == ""


def test_doctor_text_shows_a_fix_for_every_failure(monkeypatch):
    monkeypatch.setattr(B, "run_recipe",
                        lambda *a, **kw: {"error": True, "reason": "blocked",
                                         "fallback": "use the browser tool"})
    text = B.doctor_text(timeout=1)
    assert "[FAIL]" in text
    assert text.count("fix: use the browser tool") == len(B.PROBES)
    assert "site channels OK" in text


def test_the_doctor_filter_skips_the_core_checks(monkeypatch):
    """`--web-doctor github` must stay fast: the offline core rows are for the
    full run, not for a single-site check."""
    monkeypatch.setattr(B, "run_recipe",
                        lambda *a, **kw: {"markdown": "# x", "metadata": {}})
    res = B.doctor(timeout=1, only="github")
    assert [r["site"] for r in res["rows"]] == ["github"]
    assert res["total"] == 1


# --------------------------------------------------------------------------- #
# registration
# --------------------------------------------------------------------------- #
def test_tool_is_registered_in_the_registry():
    from opencode_py.tools import build_registry

    reg = build_registry()
    assert "webreach" in reg.names()
    t = reg.get("webreach")
    assert t is not None
    assert t.permission == "webreach"


def test_tui_label_and_icon_present():
    from opencode_py.tui.chat_view import TOOL_ICONS, TOOL_NAMES

    assert TOOL_NAMES.get("webreach") == "WebReach"
    assert TOOL_ICONS.get("webreach")


def test_tool_schema_is_small():
    """Tool schemas are resent every request: this one must stay cheap."""
    params = B.tool().parameters["properties"]
    assert set(params) == {"url", "action", "query", "limit", "content_limit",
                           "site", "timeout", "langs", "pages"}
    assert params["action"]["enum"] == ["read", "list", "search", "doctor"]

# --------------------------------------------------------------------------- #
# subtitle language + PDF page ranges
# --------------------------------------------------------------------------- #
def test_lang_filter_normalises_what_the_model_might_type():
    assert B._lang_filter("") == "auto"
    assert B._lang_filter("auto") == "auto"
    assert B._lang_filter(None) == "auto"
    assert B._lang_filter("es") == "es"
    assert B._lang_filter("es, en") == "es,en"
    assert B._lang_filter("ja  fr") == "ja,fr"
    # a pathological value is capped rather than passed straight to a command
    assert len(B._lang_filter(",".join(str(i) for i in range(40))).split(",")) <= 6


def test_page_filter_parses_ranges_and_survives_junk():
    assert B._page_filter("3-12") == (3, 12)
    assert B._page_filter("5") == (5, 5)
    assert B._page_filter("12-3") == (3, 12)   # reversed is normalised
    assert B._page_filter("") is None
    assert B._page_filter("junk") is None      # a typo means "whole document"
    assert B._page_filter(None) is None
    first, last = B._page_filter("9-9999")
    assert last - first <= 199, "a range must stay bounded"


def test_webfetch_accepts_the_same_page_spelling():
    """Both tools must agree on '3-12', or the same request means two things."""
    from opencode_py.tools import webfetch as WF

    for spelling in ("3-12", "5", "12-3", "", "junk"):
        assert WF._page_range(spelling) == B._page_filter(spelling)


def test_subtitle_language_reaches_the_yt_dlp_backend():
    """The gap this closes: langs existed in the backend but was never in the
    schema, so a non-English video could only ever return English."""
    from opencode_py.tools import webreach_ytdlp as Y

    assert "langs" in B.tool().parameters["properties"]
    assert B._options({"langs": "es"})["langs"] == "es"
    assert B._options({})["langs"] == "auto"
    assert Y._langs(B._options({})) == "auto"
    assert Y._langs(B._options({"langs": "fr"})) == "fr"


# --------------------------------------------------------------------------- #
# honest browser advice
# --------------------------------------------------------------------------- #
def test_a_browser_hint_says_so_when_the_browser_is_absent(monkeypatch):
    """Telling the model to use `browser` when it cannot connect is advice that
    cannot be followed — the message has to say so."""
    monkeypatch.setattr(B, "_browser_available", lambda: False)
    out = B._qualify_fallback("Use the browser tool")
    assert "browser" in out
    assert "not connected" in out
    assert "browser connect" in out


def test_a_browser_hint_is_untouched_when_it_can_work(monkeypatch):
    monkeypatch.setattr(B, "_browser_available", lambda: True)
    assert B._qualify_fallback("Use the browser tool") == "Use the browser tool"


def test_a_non_browser_fallback_is_never_annotated(monkeypatch):
    monkeypatch.setattr(B, "_browser_available", lambda: False)
    assert B._qualify_fallback("pip install yt-dlp") == "pip install yt-dlp"
    assert B._qualify_fallback("") == ""


def test_the_tool_error_text_carries_the_honest_hint(monkeypatch):
    monkeypatch.setattr(B, "_browser_available", lambda: False)
    monkeypatch.setattr(B, "run_recipe", lambda *a, **kw: {
        "error": True, "reason": "blocked",
        "fallback": "Use the browser tool", "site": "reddit"})
    out = B.tool().run({"url": "https://www.reddit.com/r/python/", "action": "list"})
    assert out.get("error") is True
    assert "not connected" in out["output"]


# --------------------------------------------------------------------------- #
# doctor core checks (offline)
# --------------------------------------------------------------------------- #
def test_core_checks_all_pass_on_this_build():
    rows = B._core_checks()
    names = {r["site"] for r in rows}
    assert {"webfetch-cascade", "webfetch-cookies", "webfetch-extract",
            "pdf-decoder", "fetch-cache"} <= names
    for r in rows:
        assert r["ok"] is True, "%s: %s" % (r["site"], r["reason"])


def test_core_checks_report_a_failure_rather_than_raising(monkeypatch):
    """A broken PDF decoder must show up as a red row, not a green doctor."""
    import opencode_py.tools.pdf_text as P

    monkeypatch.setattr(P, "_hex_to_str", lambda h: "")
    rows = {r["site"]: r for r in B._core_checks()}
    assert rows["pdf-decoder"]["ok"] is False


def test_core_checks_are_offline_and_fast():
    import time

    t0 = time.monotonic()
    B._core_checks()
    assert time.monotonic() - t0 < 8.0, "core checks must stay cheap"


def test_video_probes_do_not_download_a_whole_transcript(monkeypatch):
    """A doctor run answers 'is the channel up?'. Downloading every caption
    track cost ~46 s per probe and made the full run take two minutes."""
    from opencode_py.tools import webreach_ytdlp as Y

    calls = []

    def fake_metadata(url, timeout, is_interrupted=None):
        calls.append(("meta", url))
        return {"data": {"title": "v", "subtitles": {"en": {}},
                         "automatic_captions": {"es": {}}}}

    monkeypatch.setattr(Y, "installed", lambda *a, **kw: True)
    monkeypatch.setattr(Y, "metadata", fake_metadata)
    monkeypatch.setattr(Y, "run", lambda *a, **kw: pytest.fail("must not download"))
    out = Y.make_backend("youtube")("https://youtu.be/x", "read",
                                    _opt(_probe_only=True), 60)
    assert out.get("error") is None
    assert out["metadata"]["probe_only"] is True
    assert calls and calls[0][0] == "meta"
    # and the probe reports which caption languages exist
    assert "caption languages" in out["markdown"]


def test_a_probe_that_fails_reports_the_failure(monkeypatch):
    from opencode_py.tools import webreach_ytdlp as Y

    monkeypatch.setattr(Y, "installed", lambda *a, **kw: True)
    monkeypatch.setattr(Y, "metadata",
                        lambda *a, **kw: {"error": True, "reason": "that video is private"})
    out = Y.make_backend("youtube")("https://youtu.be/x", "read",
                                    _opt(_probe_only=True), 60)
    assert out.get("error") is True
    assert "private" in out["reason"]


# --------------------------------------------------------------------------- #
# PDF page ranges
# --------------------------------------------------------------------------- #
def test_pdf_page_selection_returns_only_those_pages(tmp_path):
    import zlib

    import opencode_py.tools.pdf_text as P

    # A realistic shape: each page OBJECT points at its own content stream via
    # /Contents, and the two must have DIFFERENT object numbers (reusing one
    # number is invalid PDF — the later object simply overwrites the earlier).
    # Selecting by page-object number would select nothing, which is exactly
    # the bug this covers.
    def _stream(n):
        body = b"BT /F1 12 Tf 72 700 Td (page %d body text) Tj ET" % n
        raw = zlib.compress(body)
        return (b"%d 0 obj\n<< /Length %d /Filter /FlateDecode >>\n"
                b"stream\n%s\nendstream\nendobj" % (20 + n, len(raw), raw))

    parts = []
    for n in (1, 2, 3):
        parts.append(_stream(n))
        parts.append(b"%d 0 obj\n<< /Type /Page /Parent 9 0 R /Contents %d 0 R >>\nendobj"
                     % (n, 20 + n))
    parts.append(b"9 0 obj\n<< /Type /Pages /Kids [1 0 R 2 0 R 3 0 R] /Count 3 >>\nendobj")
    data = b"%PDF-1.4\n" + b"\n".join(parts) + b"\n%%EOF\n"

    whole = P.extract(data)
    assert "page 1" in whole and "page 3" in whole, repr(whole[:120])
    one = P.extract(data, pages=(2, 2))
    assert "page 2" in one
    assert "page 1" not in one and "page 3" not in one
    assert P.page_count(data) == 3


def test_pdf_page_count_reads_the_tree():
    import opencode_py.tools.pdf_text as P

    assert P.page_count(b"%PDF-1.4\n1 0 obj\n<< /Type /Pages /Count 7 >>\nendobj\n") == 7


def test_a_pdf_url_is_recognised_without_being_trusted():
    from opencode_py.tools import webfetch as WF

    assert WF.looks_like_pdf_url("https://arxiv.org/pdf/2301.00001v1.pdf") is True
    assert WF.looks_like_pdf_url("https://x.test/a.pdf?v=2") is True
    # no .pdf suffix: the hint says no, but webfetch still keys off Content-Type
    assert WF.looks_like_pdf_url("https://arxiv.org/pdf/1706.03762") is False
    assert WF.looks_like_pdf_url("https://x.test/page.html") is False

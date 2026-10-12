"""Tests for the Stack Overflow and npm recipes.

The point of both recipes: the HTML page 403s from most IPs while the public API
does not (measured on device — stackoverflow.com and www.npmjs.com both failed
the same day their APIs returned full data). So these tests pin the URL shapes
those sites ACTUALLY use, and the parsers against trimmed-but-real bodies, so a
platform change fails a test instead of silently returning nothing.

Fully offline: every request is a monkeypatched ``websearch._request``.
"""
from __future__ import annotations

import json

import pytest

from opencode_py.tools import websearch as W
from opencode_py.tools import webreach as B
from opencode_py.tools import webreach_recipes as R


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    W._reset_state()
    monkeypatch.delenv("OPENCODE_TOR", raising=False)
    for recipe in R.RECIPES:
        for backend in recipe.backends:
            if backend.runner is not None:
                monkeypatch.setattr(backend, "runner", lambda *a, **kw: {
                    "error": True, "unavailable": True, "reason": "stubbed"},
                    raising=False)
    monkeypatch.setattr(B, "_ytdlp_version", lambda: "stub")
    yield
    W._reset_state()


def _opt(**kw):
    base = B._options({})
    base.update(kw)
    return base


# --------------------------------------------------------------------------- #
# trimmed but REAL response shapes
# --------------------------------------------------------------------------- #
SO_QUESTION = json.dumps({"quota_remaining": 295, "items": [{
    "question_id": 11227809,
    "title": "Why is conditional processing of a sorted array faster?",
    "link": "https://stackoverflow.com/questions/11227809/why-x",
    "score": 27547, "view_count": 5000000, "answer_count": 25,
    "tags": ["java", "c++", "performance"],
    "owner": {"display_name": "GManNickG"},
    "body": "<p>Sorting the data <em>before</em> the timed region makes the primary loop fast.</p>",
}]})

SO_ANSWERS = json.dumps({"quota_remaining": 294, "items": [
    {"answer_id": 11227811, "score": 35297, "is_accepted": True,
     "owner": {"display_name": "Mysticial"},
     "body": "<p>You are a victim of the <strong>branch predictor</strong>.</p>"},
    {"answer_id": 11227899, "score": 900, "is_accepted": False,
     "owner": {"display_name": "someone"}, "body": "<p>Also memory layout.</p>"},
]})

SO_LIST = json.dumps({"quota_remaining": 293, "items": [
    {"question_id": 231767, "title": "What does the yield keyword do?",
     "link": "https://stackoverflow.com/questions/231767/x", "score": 5000,
     "view_count": 900000, "answer_count": 4, "tags": ["python"],
     "owner": {"display_name": "dev"}, "body": ""},
]})

NPM_MANIFEST = json.dumps({
    "name": "express", "version": "5.2.1",
    "description": "Fast, unopinionated, minimalist web framework",
    "license": "MIT", "homepage": "https://expressjs.com",
    "repository": {"url": "git+https://github.com/expressjs/express.git", "type": "git"},
    "keywords": ["express", "framework"],
    "maintainers": [{"name": "dougwilson"}, {"name": "domenic"}],
    "dependencies": {"body-parser": "2.2.0", "cookie": "1.0.2", "express": "5.2.1"},
    "date": "2025-03-01T00:00:00.000Z",
})

NPM_SEARCH = json.dumps({"total": 4200, "time": "now", "objects": [
    {"package": {"name": "http-server", "version": "14.1.1",
                 "description": "A simple zero-configuration command-line http server",
                 "links": {"npm": "https://www.npmjs.com/package/http-server"}},
     "score": {"final": 2297.29}, "downloads": {"weekly": 900000}},
    {"package": {"name": "http-server-alt", "version": "1.0.0",
                 "description": "Alternative", "links": {"npm": ""}},
     "score": {"final": 900.1}},
]})


# --------------------------------------------------------------------------- #
# URL matching
# --------------------------------------------------------------------------- #
def test_stackexchange_recipe_matches_every_stack_site():
    for url in ("https://stackoverflow.com/questions/1",
                "https://superuser.com/questions/1",
                "https://serverfault.com/questions/1",
                "https://askubuntu.com/questions/1",
                "https://mathoverflow.net/questions/1"):
        got = R.find_recipe(url)
        assert got is not None and got.site == "stackoverflow", url


def test_npm_recipe_matches_the_package_url_shape():
    for url in ("https://www.npmjs.com/package/express",
                "https://npmjs.com/package/express",
                "https://www.npmjs.com/package/@types/node",
                "https://www.npmjs.com/package/express/v/4.18.0"):
        got = R.find_recipe(url)
        assert got is not None and got.site == "npm", url


def test_new_recipes_reject_near_miss_hosts():
    assert R.find_recipe("https://notstackoverflow.com/questions/1") is None
    assert R.find_recipe("https://stackoverflow.com.evil.test/q/1") is None
    assert R.find_recipe("https://notnpmjs.com/package/x") is None
    assert R.find_recipe("https://evil.test/?x=npmjs.com") is None


# --------------------------------------------------------------------------- #
# URL builders — every shape these sites really use
# --------------------------------------------------------------------------- #
def test_so_question_url_uses_the_api_with_the_body_filter():
    got = R._so_build("https://stackoverflow.com/questions/11227809/why-x",
                      "read", _opt())
    assert got.startswith(
        "https://api.stackexchange.com/2.3/questions/11227809?site=stackoverflow")
    assert "filter=withbody" in got


def test_so_answer_url_routes_through_the_slug():
    """The answer id sits AFTER the slug, at a varying index. Matching a fixed
    position silently returned the question instead of the answers."""
    got = R._so_build(
        "https://stackoverflow.com/questions/11227809/why-x/11227811/the-answer",
        "read", _opt())
    assert "/questions/11227809/answers?" in got
    assert "sort=votes" in got


def test_so_short_answer_url_also_routes_to_answers():
    got = R._so_build("https://stackoverflow.com/a/11227811", "read", _opt())
    assert "/questions/11227811/answers?" in got


def test_so_bare_questions_url_is_a_listing():
    got = R._so_build("https://stackoverflow.com/questions", "list", _opt())
    assert "/2.3/questions?site=stackoverflow" in got
    assert "sort=votes" in got


def test_so_tagged_url_carries_the_tag():
    got = R._so_build("https://stackoverflow.com/questions/tagged/python",
                      "list", _opt())
    assert "tagged=python" in got


def test_so_search_needs_a_query():
    so = R.find_recipe("https://stackoverflow.com/questions")
    with pytest.raises(ValueError):
        so.backends[0].build("https://stackoverflow.com/questions", "search", _opt())


def test_so_search_builds_the_advanced_endpoint():
    got = R._so_build("https://stackoverflow.com/questions", "search",
                      _opt(query="branch prediction"))
    assert "/2.3/search/advanced?site=stackoverflow" in got
    assert "q=branch%20prediction" in got


def test_so_maps_each_host_to_its_own_api_site():
    """serverfault.com must ask for site=serverfault, not site=stackoverflow."""
    assert "site=superuser" in R._so_build("https://superuser.com/questions/1",
                                          "read", _opt())
    assert "site=serverfault" in R._so_build("https://serverfault.com/questions/1",
                                             "read", _opt())


def test_npm_builds_the_registry_not_the_page():
    got = R._npm_build("https://www.npmjs.com/package/express", "read", _opt())
    assert got == "https://registry.npmjs.org/express/latest"


def test_npm_versioned_url_pins_the_version():
    got = R._npm_build("https://www.npmjs.com/package/express/v/4.18.0",
                       "read", _opt())
    assert got == "https://registry.npmjs.org/express/4.18.0"


def test_npm_scoped_name_keeps_its_slash_literal():
    """quote() must not encode the '/', or the registry sees one opaque segment."""
    got = R._npm_build("https://www.npmjs.com/package/@types/node", "read", _opt())
    assert got == "https://registry.npmjs.org/@types/node/latest"


def test_npm_rejects_a_url_with_no_package_name():
    with pytest.raises(ValueError):
        R._npm_build("https://www.npmjs.com/", "read", _opt())


def test_npm_search_needs_a_query():
    with pytest.raises(ValueError):
        R._npm_build("https://www.npmjs.com/", "search", _opt())


# --------------------------------------------------------------------------- #
# parsers
# --------------------------------------------------------------------------- #
def test_so_question_parser_keeps_the_body_and_the_stats():
    so = R.find_recipe("https://stackoverflow.com/questions/1")
    out = so.backends[0].parse(SO_QUESTION, "u", "read", _opt())
    md = out["markdown"]
    assert "Why is conditional processing" in md
    assert "27547" in md and "25 answers" in md
    assert "Sorting the data" in md
    assert "<p>" not in md and "<em>" not in md
    assert out["metadata"]["count"] == 1
    assert out["metadata"]["quota_remaining"] == 295


def test_so_answer_parser_labels_answers_without_titles():
    so = R.find_recipe("https://stackoverflow.com/questions/1")
    out = so.backends[0].parse(SO_ANSWERS, "u", "read", _opt())
    md = out["markdown"]
    assert "Mysticial" in md and "branch predictor" in md
    assert "ACCEPTED" in md
    # a second, unaccepted answer must NOT be labelled accepted
    assert md.count("ACCEPTED") == 1


def test_so_list_parser_emits_one_row_per_question():
    so = R.find_recipe("https://stackoverflow.com/questions")
    out = so.backends[0].parse(SO_LIST, "u", "list", _opt())
    assert out["metadata"]["count"] == 1
    assert "yield" in out["markdown"]


def test_so_parser_rejects_a_non_api_body():
    so = R.find_recipe("https://stackoverflow.com/questions/1")
    out = so.backends[0].parse("<html>403 Forbidden</html>", "u", "read", _opt())
    assert out.get("error") is True


def test_so_parser_surfaces_an_api_error_message():
    so = R.find_recipe("https://stackoverflow.com/questions/1")
    out = so.backends[0].parse(json.dumps({"error_message": {"message": "quota exceeded"}}),
                               "u", "read", _opt())
    assert out.get("error") is True
    assert "quota exceeded" in out["reason"]


def test_npm_parser_reads_the_manifest():
    npm = R.find_recipe("https://www.npmjs.com/package/express")
    out = npm.backends[0].parse(NPM_MANIFEST, "u", "read", _opt())
    md = out["markdown"]
    assert "express 5.2.1" in md
    assert "MIT" in md
    # the git+ prefix and .git suffix are noise for a human
    assert "github.com/expressjs/express" in md
    assert "git+" not in md and "repository: https://github.com/expressjs/express\n" in md
    assert out["metadata"]["deps"] == 3


def test_npm_parser_lists_dependencies():
    npm = R.find_recipe("https://www.npmjs.com/package/express")
    md = npm.backends[0].parse(NPM_MANIFEST, "u", "read", _opt())["markdown"]
    assert "## dependencies" in md
    assert "body-parser" in md


def test_npm_search_parser_reads_the_search_shape():
    npm = R.find_recipe("https://www.npmjs.com/package/x")
    out = npm.backends[0].parse(NPM_SEARCH, "u", "search", _opt())
    assert out["metadata"]["kind"] == "npm_search"
    assert out["metadata"]["total"] == 4200
    assert "http-server" in out["markdown"]
    assert "14.1.1" in out["markdown"]
    assert len(out["items"]) == 2


def test_npm_parser_rejects_a_non_manifest_body():
    npm = R.find_recipe("https://www.npmjs.com/package/x")
    out = npm.backends[0].parse("<html>nope</html>", "u", "read", _opt())
    assert out.get("error") is True


# --------------------------------------------------------------------------- #
# end to end through the chain
# --------------------------------------------------------------------------- #
def test_stackoverflow_reads_through_the_full_chain(monkeypatch):
    calls = []

    def fake(ctx, url, **kw):
        calls.append(url)
        return SO_QUESTION, None

    monkeypatch.setattr(W, "_request", fake)
    out = B.run_recipe("https://stackoverflow.com/questions/11227809/why-x",
                       "read", _opt(), 30)
    assert out.get("error") is None
    assert len(calls) == 1
    assert calls[0].startswith("https://api.stackexchange.com/")
    assert out["metadata"]["site"] == "stackoverflow"
    assert out["metadata"]["backend"] == "se-api"


def test_npm_reads_through_the_full_chain(monkeypatch):
    monkeypatch.setattr(W, "_request",
                        lambda ctx, url, **kw: (NPM_MANIFEST, None))
    out = B.run_recipe("https://www.npmjs.com/package/express", "read",
                       _opt(), 30)
    assert out.get("error") is None
    assert "express 5.2.1" in out["markdown"]
    assert out["metadata"]["site"] == "npm"


def test_a_block_page_is_not_counted_as_success(monkeypatch):
    """Both APIs are behind Cloudflare-ish walls in the wild; a 200 wall page
    must fall through to the error path, not parse into a confident empty row."""
    monkeypatch.setattr(W, "_request",
                        lambda ctx, url, **kw: ("blocked by network security", None))
    out = B.run_recipe("https://www.npmjs.com/package/express", "read",
                       _opt(), 30)
    assert out.get("error") is True


def test_tool_route_returns_the_package_to_the_model(monkeypatch):
    monkeypatch.setattr(W, "_request",
                        lambda ctx, url, **kw: (NPM_MANIFEST, None))
    out = B.tool().run({"url": "https://www.npmjs.com/package/express",
                        "action": "read"})
    assert out.get("error") is not True
    assert "express 5.2.1" in out["output"]


def test_both_recipes_are_visible_to_doctor():
    """A channel the doctor never probes is a channel nobody trusts."""
    sites = {p[0] for p in B.PROBES}
    assert "stackoverflow" in sites
    assert "npm" in sites
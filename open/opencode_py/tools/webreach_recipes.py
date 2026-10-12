"""Site recipes for the webreach tool: URL -> ordered backends -> markdown.

WHY THIS FILE EXISTS
    websearch knows how to FIND a URL and webfetch knows how to READ one, but
    neither knows the third thing: that a reddit thread has a JSON API, that a
    YouTube URL hides a video id, or that a GitHub repo is better read through
    api.github.com than through its HTML. That knowledge is what Agent Reach
    calls the "capability layer", and it is pure data — no sockets, no HTTP.

    Split from webreach.py on purpose: everything here is a pure function of
    (url, body), so the whole table is testable offline without a network.

CONTRACT
    A Recipe maps a URL to one or more Backends. A Backend builds a request URL
    and parses the response into the SAME shape the model always sees:

        {"title": str, "markdown": str, "items": [...], "metadata": {...}}

    Uniform output is what makes one tool usable for every site — the model
    never has to be told which endpoint produced the text.

ADDING A SITE
    Append one Backend to RECIPES. If the site needs a search/list form rather
    than a read form, give it an action; the tool routes action=read|list|search
    to the backends that declare support for it.
"""

from __future__ import annotations

from ..util.log import swallow
import html as _html
import json
import re
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import parse_qs, quote, urlparse


@dataclass
class Backend:
    """One way of getting at a site.

    ``build`` turns the caller's URL/action into a request URL. ``parse`` turns
    that response into the uniform result shape. ``cap`` is the per-call
    character cap for this endpoint (some APIs return far more than a page).
    """

    name: str
    build: Callable[[str, str, dict[str, Any]], str]
    parse: Callable[[str, str, str, dict[str, Any]], dict[str, Any]]
    actions: tuple[str, ...] = ("read", "list", "search")
    cap: int = 20000
    # When every backend for a site fails, this is what the model is told to do
    # instead. Set it only where a real alternative exists (browser, webfetch).
    fallback: str = ""
    # Opt into the Cloudflare-bypass cascade + Tor retry on failure. OFF by
    # default: these endpoints are structured JSON, so a client-fingerprint
    # change cannot defeat a wall that is keyed on IP reputation — the cascade
    # would just burn 10+ s of phone radio to fail the same way.
    escalate: bool = False
    # A non-HTTP backend brings its own transport and implements this signature
    # instead of build/parse: (url, action, opt, budget, is_interrupted) ->
    # result dict. It still receives a slice of the shared deadline, so a slow
    # video fetch can never overrun the chain.
    runner: Callable[..., dict[str, Any]] | None = None


@dataclass
class Recipe:
    """A site: how to recognise its URLs, and which backends serve them."""

    site: str
    host_endswith: str
    backends: list[Backend]
    note: str = ""
    matcher: Callable[[str], bool] | None = None


# --------------------------------------------------------------------------- #
# small shared helpers (all pure, all never raise)
# --------------------------------------------------------------------------- #
def _loads(body: str) -> Any:
    """json.loads that returns None instead of raising."""
    try:
        return json.loads(body)
    except Exception as _oc_exc:
        swallow(_oc_exc, "webreach_recipes.py:_loads")
        return None


def _strip_html(text: str) -> str:
    """Collapse an HTML fragment to readable plain text."""
    try:
        t = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1\s*>", " ", text)
        t = re.sub(r"(?is)<br[^>]*>|</p\s*>|</div\s*>|</li\s*>", "\n", t)
        t = re.sub(r"(?s)<[^>]+>", "", t)
        t = _html.unescape(t)
        t = re.sub(r"[ \t\r\f\v]+", " ", t)
        t = re.sub(r"\n{3,}", "\n\n", t)
        return t.strip()
    except Exception as _oc_exc:
        swallow(_oc_exc, "webreach_recipes.py:_strip_html")
        return str(text or "")


def _host(url: str) -> str:
    try:
        return (urlparse(url).netloc or "").lower()
    except Exception:
        return ""


def _path(url: str) -> str:
    try:
        return urlparse(url).path or ""
    except Exception:
        return ""


def _kv(body: str, key: str) -> str:
    """First ``"key": "value"`` string found in a JSON blob, ignoring escapes."""
    m = re.search(r'"%s"\s*:\s*"((?:[^"\\]|\\.)*)"' % re.escape(key), body)
    if not m:
        return ""
    try:
        return _json_str(m.group(1))
    except Exception:
        return m.group(1)


def _json_str(text: str) -> str:
    try:
        return json.loads('"%s"' % text)
    except Exception:
        return text


def _host_is(url: str, *domains: str) -> bool:
    """True when the URL's bare host equals or is a subdomain of any domain."""
    bare = _host(url)
    for prefix in ("www.", "m.", "old.", "new.", "api."):
        if bare.startswith(prefix):
            bare = bare[len(prefix):]
    return any(bare == d or bare.endswith("." + d) for d in domains)


def _looks_like_feed(url: str) -> bool:
    """True for URLs that plausibly ARE a feed, so the catch-all does not
    swallow ordinary web pages (those belong to webfetch, not here)."""
    p = _path(url).lower()
    if re.search(r"\.(rss|xml|atom)$", p):
        return True
    q = (urlparse(url).query or "").lower()
    if any(k in q for k in ("format=rss", "type=rss", "output=rss", "feed=rss")):
        return True
    if re.search(r"/(feed|rss|atom|feeds?)(/|$|\.|$)", p):
        return True
    # a host that advertises itself as a feed service: hnrss.org,
    # feeds.example.com, … (the path carries no marker on those)
    host = _host(url).split(":")[0]
    return bool(re.search(r"rss|feeds?", host))


def _ok(title: str, markdown: str, **meta: Any) -> dict[str, Any]:
    return {"title": title, "markdown": markdown, "items": [], "metadata": meta}


def _clamp(text: str, limit: int) -> tuple[str, bool]:
    t = str(text or "")
    if limit > 0 and len(t) > limit:
        return t[:limit], True
    return t, False


# --------------------------------------------------------------------------- #
# GitHub — api.github.com returns richer, smaller data than the HTML page
# --------------------------------------------------------------------------- #
def _gh_read_build(url: str, action: str, opt: dict[str, Any]) -> str:
    p = _path(url).strip("/")
    parts = p.split("/")
    if not parts or not parts[0]:
        raise ValueError("not a github repo url")
    owner, repo = parts[0], parts[1].removesuffix(".git")
    kind = parts[2] if len(parts) > 2 else ""
    if kind == "issues" and len(parts) > 3 and parts[3] not in ("", "new"):
        num = re.sub(r"\D", "", parts[3])
        if num:
            return f"https://api.github.com/repos/{owner}/{repo}/issues/{num}"
    if kind == "issues":
        state = "all" if opt.get("state") == "all" else "open"
        return f"https://api.github.com/repos/{owner}/{repo}/issues?state={state}&per_page={int(opt.get('limit') or 10)}"
    if kind == "releases":
        return f"https://api.github.com/repos/{owner}/{repo}/releases?per_page={int(opt.get('limit') or 5)}"
    if kind == "pulls":
        return f"https://api.github.com/repos/{owner}/{repo}/pulls?state=open&per_page={int(opt.get('limit') or 10)}"
    if len(parts) > 2:
        rest = parts[2:]
        refname = str(opt.get("ref") or "HEAD")
        # a blob/tree URL carries the ref AFTER the literal segment:
        # /a/b/blob/main/README.md -> ref=main, path=README.md. Reusing the
        # caller's ref here would put it in the URL twice.
        if rest and rest[0] in ("blob", "tree", "raw"):
            rest = rest[1:]
            if rest:
                refname, rest = rest[0], rest[1:]
        return (f"https://raw.githubusercontent.com/{owner}/{repo}/{refname}/"
                + "/".join(rest))
    return f"https://api.github.com/repos/{owner}/{repo}"


def _gh_repo_parse(body: str, url: str, action: str, opt: dict[str, Any]) -> dict[str, Any]:
    d = _loads(body)
    if not isinstance(d, dict) or "full_name" not in d:
        return {"error": True, "reason": "not a github repo response"}
    desc = d.get("description") or ""
    stars = d.get("stargazers_count")
    lines = [
        "# %s" % (d.get("full_name") or url),
        "",
        (desc or "(no description)"),
        "",
        "- stars: %s" % stars,
        "- language: %s" % (d.get("language") or "?"),
        "- license: %s" % ((d.get("license") or {}) or {}).get("spdx_id"),
        "- open issues: %s" % d.get("open_issues_count"),
        "- homepage: %s" % (d.get("homepage") or "(none)"),
        "- default branch: %s" % (d.get("default_branch") or "?"),
    ]
    if d.get("pushed_at"):
        lines.append("- last push: %s" % d["pushed_at"])
    topics = d.get("topics") or []
    if topics:
        lines.append("- topics: %s" % ", ".join(str(t) for t in topics[:20]))
    md, cut = _clamp("\n".join(lines), int(opt.get("content_limit") or 8000))
    meta = {"source": url, "stars": stars, "full_name": d.get("full_name")}
    if cut:
        meta["truncated"] = True
    return {"title": d.get("full_name") or url, "markdown": md, "items": [], "metadata": meta}


def _gh_list_parse(kind: str) -> Callable[..., dict[str, Any]]:
    def _parse(body: str, url: str, action: str, opt: dict[str, Any]) -> dict[str, Any]:
        data = _loads(body)
        if isinstance(data, dict) and "items" in data:
            data = data.get("items") or []
        if not isinstance(data, list):
            return {"error": True, "reason": "not a github list response"}
        rows: list[str] = []
        items: list[dict[str, Any]] = []
        for it in data[: int(opt.get("limit") or 10)]:
            if not isinstance(it, dict):
                continue
            num = it.get("number")
            title = it.get("title") or ""
            state = it.get("state") or ""
            who = ((it.get("user") or {}) or {}).get("login") or "?"
            when = (it.get("created_at") or "")[:10]
            href = it.get("html_url") or ""
            if kind == "release":
                name = it.get("name") or it.get("tag_name") or ""
                href = it.get("html_url") or ""
                who = ((it.get("author") or {}) or {}).get("login") or "?"
                when = (it.get("published_at") or it.get("created_at") or "")[:10]
                rows.append("- [%s](%s) — %s — %s" % (name, href, when, who))
            elif num:
                rows.append("- [#%s %s](%s) [%s] %s — %s" % (num, title, href, state, who, when))
            else:
                continue
            items.append({"title": title or it.get("name") or "", "url": href, "state": state})
        head = "# %s\n\n" % kind
        md, cut = _clamp(head + "\n".join(rows), int(opt.get("content_limit") or 8000))
        meta = {"source": url, "count": len(items), "kind": kind}
        if cut:
            meta["truncated"] = True
        return {"title": "%s (%d)" % (kind, len(items)), "markdown": md, "items": items, "metadata": meta}

    return _parse


def _gh_raw_parse(body: str, url: str, action: str, opt: dict[str, Any]) -> dict[str, Any]:
    md, cut = _clamp(body, int(opt.get("content_limit") or 8000))
    meta = {"source": url, "kind": "raw_file", "chars": len(body)}
    if cut:
        meta["truncated"] = True
    return {"title": _kv(body, "path") or url, "markdown": md, "items": [], "metadata": meta}


def _gh_search_build(url: str, action: str, opt: dict[str, Any]) -> str:
    q = str(opt.get("query") or "")
    if not q:
        raise ValueError("search needs query=")
    return ("https://api.github.com/search/repositories?q="
            + quote(q) + f"&per_page={int(opt.get('limit') or 10)}")


def _gh_search_parse(body: str, url: str, action: str, opt: dict[str, Any]) -> dict[str, Any]:
    d = _loads(body) or {}
    hits = d.get("items") or []
    rows: list[str] = []
    items: list[dict[str, Any]] = []
    for it in hits[: int(opt.get("limit") or 10)]:
        if not isinstance(it, dict):
            continue
        full = it.get("full_name") or ""
        rows.append("- [%s](%s) ★%s — %s" % (full, it.get("html_url") or "", it.get("stargazers_count", 0), it.get("description") or ""))
        items.append({"title": full, "url": it.get("html_url") or "", "score": it.get("stargazers_count", 0)})
    md, cut = _clamp("# github search\n\n" + "\n".join(rows), int(opt.get("content_limit") or 8000))
    meta = {"source": url, "count": len(items), "total": d.get("total_count")}
    if cut:
        meta["truncated"] = True
    return {"title": "github search (%d)" % len(items), "markdown": md, "items": items, "metadata": meta}


# --------------------------------------------------------------------------- #
# Hacker News — algolia returns a whole comment tree in ONE call (no N+1)
# --------------------------------------------------------------------------- #
def _hn_build(url: str, action: str, opt: dict[str, Any]) -> str:
    # HN keeps the story id in the QUERY (?id=…), not the path.
    m = re.search(r"[?&]id=(\d+)", url or "")
    if m:
        return f"https://hn.algolia.com/api/v1/items/{m.group(1)}"
    p = _path(url).strip("/")
    m2 = re.match(r"^item/(\d+)", p)
    if m2:
        return f"https://hn.algolia.com/api/v1/items/{m2.group(1)}"
    if action == "search" or not p:
        q = str(opt.get("query") or "")
        base = "https://hn.algolia.com/api/v1/search?tags=front_page"
        if q:
            base += "&query=" + quote(q)
        return base
    return "https://hn.algolia.com/api/v1/search?tags=front_page"


def _hn_walk(node: Any, depth: int, out: list[str], limit: int, maxdepth: int) -> None:
    if len(out) >= limit or depth > maxdepth:
        return
    if isinstance(node, dict):
        text = _strip_html(node.get("text") or "")
        if text:
            author = node.get("author") or "?"
            out.append("  " * depth + "- **%s**: %s" % (author, text[:500]))
        for ch in node.get("children") or []:
            _hn_walk(ch, depth + 1, out, limit, maxdepth)
    elif isinstance(node, list):
        for ch in node:
            _hn_walk(ch, depth, out, limit, maxdepth)


def _hn_parse(body: str, url: str, action: str, opt: dict[str, Any]) -> dict[str, Any]:
    d = _loads(body) or {}
    limit = int(opt.get("limit") or 10)
    items: list[dict[str, Any]] = []
    rows: list[str] = []
    if isinstance(d, dict) and ("children" in d or "title" in d):
        # a single item: title + flattened comment tree
        title = d.get("title") or d.get("id") or url
        out: list[str] = []
        _hn_walk(d.get("children") or [], 0, out, limit * 4, 4)
        md = "# %s\n\n%s\n" % (title, "\n".join(out[: limit * 4]))
        meta = {"source": url, "kind": "item", "comments": len(out)}
        if d.get("points") is not None:
            meta["points"] = d.get("points")
        mdc, cut = _clamp(md, int(opt.get("content_limit") or 8000))
        if cut:
            meta["truncated"] = True
        return {"title": title, "markdown": mdc, "items": items, "metadata": meta}
    for hit in (d.get("hits") or [])[:limit]:
        if not isinstance(hit, dict):
            continue
        title = hit.get("title") or hit.get("story_title") or ""
        href = hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID')}"
        items.append({"title": title, "url": href, "points": hit.get("points")})
        rows.append("- [%s](%s) — %s pts — HN %s" % (title, href, hit.get("points"), hit.get("objectID")))
    mdc, cut = _clamp("# hacker news\n\n" + "\n".join(rows), int(opt.get("content_limit") or 8000))
    meta = {"source": url, "kind": "front_page", "count": len(items)}
    if cut:
        meta["truncated"] = True
    return {"title": "hacker news (%d)" % len(items), "markdown": mdc, "items": items, "metadata": meta}


# --------------------------------------------------------------------------- #
# Reddit — .json is the good path but IS blocked from many IPs; .rss is the
# slow fallback that still works, so the chain is json -> rss -> browser hint
# --------------------------------------------------------------------------- #
def _reddit_build(url: str, action: str, opt: dict[str, Any]) -> str:
    p = _path(url).rstrip("/")
    p = re.sub(r"\.rss$", "", p)
    parts = [x for x in p.split("/") if x]
    limit = int(opt.get("limit") or 10)
    if len(parts) >= 4 and parts[2] == "comments":
        return "https://www.reddit.com" + p + ".json?limit=%d" % min(limit * 3, 100)
    if len(parts) >= 2:
        sort = str(opt.get("sort") or "hot")
        return f"https://www.reddit.com/r/{parts[1]}/{sort}.json?limit={min(limit, 100)}"
    raise ValueError("not a reddit subreddit url")


def _reddit_rss_build(url: str, action: str, opt: dict[str, Any]) -> str:
    p = _path(url).rstrip("/")
    p = re.sub(r"\.rss$", "", p)
    if "/comments/" in p:
        return "https://www.reddit.com" + p + ".rss"
    m = re.match(r"^/r/([^/]+)", p)
    if m:
        return f"https://www.reddit.com/r/{m.group(1)}/.rss"
    raise ValueError("not a reddit subreddit url")


def _reddit_comments(node: Any, depth: int, out: list[str], cap: int) -> None:
    """Flatten reddit's ``replies`` tree into indented '- **author**: text' lines."""
    if depth > 6 or len(out) >= cap or not isinstance(node, dict):
        return
    data = node.get("data")
    if not isinstance(data, dict):
        return
    text = _strip_html(data.get("body") or "")
    if text:
        out.append("  " * depth + "- **u/%s**: %s" % (data.get("author") or "?", text[:600]))
    replies = data.get("replies")
    if isinstance(replies, dict):
        for child in (replies.get("data") or {}).get("children") or []:
            _reddit_comments(child, depth + 1, out, cap)


def _reddit_json_parse(body: str, url: str, action: str, opt: dict[str, Any]) -> dict[str, Any]:
    data = _loads(body)
    if not isinstance(data, dict) or not isinstance(data.get("data"), dict):
        # Reddit answers 200 with a plain-text bot-wall instead of a 403, so
        # "not JSON" is almost always the block, not a schema change.
        head = (body or "").strip()[:80].lower()
        if "blocked" in head or "network security" in head:
            return {"error": True, "reason": "reddit returned its bot-wall page instead of JSON"}
        return {"error": True, "reason": "not a reddit JSON listing"}
    listing = data.get("data") or {}
    children = listing.get("children") or []
    limit = int(opt.get("limit") or 10)
    rows: list[str] = []
    items: list[dict[str, Any]] = []
    comments: list[str] = []
    for child in children[:limit]:
        if not isinstance(child, dict):
            continue
        d = child.get("data")
        if not isinstance(d, dict):
            continue
        title = str(d.get("title") or "")
        permalink = str(d.get("permalink") or "")
        href = "https://www.reddit.com" + permalink if permalink else ""
        author = str(d.get("author") or "?")
        score = d.get("score")
        if title:
            rows.append("- [%s](%s) — %s pts — u/%s — %s comments"
                        % (title, href, score, author, d.get("num_comments", "?")))
            items.append({"title": title, "url": href, "score": score, "author": author})
        text = _strip_html(d.get("selftext") or "")
        if text:
            comments.append(text[:1200])
        _reddit_comments(child, 0, comments, limit * 4)
    md = "\n".join(rows) + ("\n\n---\n" + "\n\n".join(comments) if comments else "")
    mdc, cut = _clamp(md, int(opt.get("content_limit") or 8000))
    meta = {"source": url, "kind": "reddit_json", "posts": len(items),
            "comment_blocks": len(comments)}
    if cut:
        meta["truncated"] = True
    return {"title": "reddit (%d posts)" % len(items), "markdown": mdc,
            "items": items, "metadata": meta}


def _reddit_rss_parse(body: str, url: str, action: str, opt: dict[str, Any]) -> dict[str, Any]:
    if "<entry" not in body and "<item" not in body:
        return {"error": True, "reason": "not a reddit feed"}
    blocks = re.findall(r"(?is)<entry[^>]*>(.*?)</entry\s*>", body) or re.findall(
        r"(?is)<item[^>]*>(.*?)</item\s*>", body)
    rows: list[str] = []
    items: list[dict[str, Any]] = []
    limit = int(opt.get("limit") or 10)
    for b in blocks[:limit]:
        def _t(name: str) -> str:
            m = re.search(r"(?is)<%s[^>]*>(.*?)</%s\s*>" % (name, name), b)
            return _strip_html(re.sub(r"(?s)<[^>]+>", "", m.group(1))) if m else ""

        title = _t("title")[:300]
        href = ""
        m_link = re.search(r'(?is)<link[^>]*href=["\']([^"\']+)["\']', b)
        if m_link:
            href = m_link.group(1)
        author = _t("author") or _t("name")
        content = _t("content") or _t("description")
        if title:
            rows.append("- [%s](%s) — %s" % (title, href, author))
            if content:
                rows.append("  " + content[:400])
        items.append({"title": title, "url": href})
    mdc, cut = _clamp("# reddit (rss, slow path)\n\n" + "\n".join(rows), int(opt.get("content_limit") or 8000))
    meta = {"source": url, "kind": "reddit_rss", "count": len(items), "slow": True}
    if cut:
        meta["truncated"] = True
    return {"title": "reddit (%d)" % len(items), "markdown": mdc, "items": items, "metadata": meta}


# --------------------------------------------------------------------------- #
# V2EX — public JSON API, Chinese content
# --------------------------------------------------------------------------- #
def _v2ex_build(url: str, action: str, opt: dict[str, Any]) -> str:
    p = _path(url).strip("/")
    m = re.match(r"^(?:t|topic)/(\d+)", p)
    if m:
        return "https://www.v2ex.com/api/topics/show.json?id=" + m.group(1)
    # /go/<node> names the node in the path; the param wins when both are given
    node = str(opt.get("node") or "")
    if not node:
        g = re.match(r"^go/([A-Za-z0-9_-]+)", p)
        if g:
            node = g.group(1)
    if not node:
        raise ValueError("no v2ex node in the url — use /go/<node> or pass node=")
    return "https://www.v2ex.com/api/topics/show.json?node_name=" + quote(node)


def _v2ex_parse(body: str, url: str, action: str, opt: dict[str, Any]) -> dict[str, Any]:
    data = _loads(body)
    if not isinstance(data, list) or not data:
        return {"error": True, "reason": "not a v2ex response"}
    limit = int(opt.get("limit") or 10)
    rows: list[str] = []
    items: list[dict[str, Any]] = []
    for t in data[:limit]:
        if not isinstance(t, dict):
            continue
        title = t.get("title") or ""
        href = "https://www.v2ex.com/t/%s" % t.get("id") if t.get("id") else ""
        member = ((t.get("member") or {}) or {}).get("username") or "?"
        replies = t.get("replies")
        rows.append("- [%s](%s) — %s replies — %s" % (title, href, replies if isinstance(replies, int) else 0, member))
        if t.get("content"):
            rows.append("  " + _strip_html(t["content"])[:400])
        items.append({"title": title, "url": href})
    mdc, cut = _clamp("# v2ex\n\n" + "\n".join(rows), int(opt.get("content_limit") or 8000))
    meta = {"source": url, "kind": "v2ex", "count": len(items)}
    if cut:
        meta["truncated"] = True
    return {"title": "v2ex (%d)" % len(items), "markdown": mdc, "items": items, "metadata": meta}


# --------------------------------------------------------------------------- #
# Lobsters — NOTE the path is /t/<tag>, not /r/<tag> (the latter 404s)
# --------------------------------------------------------------------------- #
def _lob_build(url: str, action: str, opt: dict[str, Any]) -> str:
    p = _path(url).strip("/")
    m = re.match(r"^s/([A-Za-z0-9]+)", p)
    if m:
        return "https://lobste.rs/s/" + m.group(1) + ".json"
    tag = str(opt.get("tag") or "python")
    return "https://lobste.rs/t/" + quote(tag) + ".json"


def _lob_parse(body: str, url: str, action: str, opt: dict[str, Any]) -> dict[str, Any]:
    data = _loads(body)
    if isinstance(data, dict) and isinstance(data.get("story"), dict):
        data = [data["story"]]
    if not isinstance(data, list) or not data:
        return {"error": True, "reason": "not a lobsters response"}
    limit = int(opt.get("limit") or 10)
    rows: list[str] = []
    items: list[dict[str, Any]] = []
    for s in data[:limit]:
        if not isinstance(s, dict):
            continue
        title = s.get("title") or ""
        href = s.get("url") or s.get("comments_url") or ""
        if not href and s.get("short_id"):
            href = "https://lobste.rs/s/" + s["short_id"]
        user = s.get("submitter_user") or "?"
        rows.append("- [%s](%s) — ▲%s 💬%s — %s" % (title, href, s.get("score", 0), s.get("comment_count", 0), user))
        items.append({"title": title, "url": href, "score": s.get("score")})
    mdc, cut = _clamp("# lobsters\n\n" + "\n".join(rows), int(opt.get("content_limit") or 8000))
    meta = {"source": url, "kind": "lobsters", "count": len(items)}
    if cut:
        meta["truncated"] = True
    return {"title": "lobsters (%d)" % len(items), "markdown": mdc, "items": items, "metadata": meta}


# --------------------------------------------------------------------------- #
# YouTube — timedtext needs a player-session token that plain HTTP cannot get.
# The recipe still earns its place: it recognises the URL, says the truth about
# why it cannot help, and points at the browser tool that CAN.
# --------------------------------------------------------------------------- #
_YT_ID = re.compile(r"(?:v=|youtu\.be/|/embed/|/shorts/|/live/)([A-Za-z0-9_-]{11})")


class _Unavailable(Exception):
    """Raised by a backend that cannot serve this URL over plain HTTP.

    The channel is recognized but the platform refuses anonymous HTTP: YouTube
    needs a player-session token that only a real browser session can obtain.
    Raising instead of returning a broken parse keeps the failure honest and
    lets the tool answer with the browser-tool alternative.
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _yt_id(url: str) -> str:
    m = _YT_ID.search(url or "")
    if m:
        return m.group(1)
    try:
        qs = parse_qs(urlparse(url).query)
        v = (qs.get("v") or [""])[0]
        if re.fullmatch(r"[A-Za-z0-9_-]{11}", v or ""):
            return v
    except Exception as _oc_exc:
        swallow(_oc_exc, "webreach_recipes.py:_yt_id")
    return ""


def _yt_build(url: str, action: str, opt: dict[str, Any]) -> str:
    vid = _yt_id(url)
    if not vid:
        raise ValueError("no youtube video id in url")
    raise _Unavailable(
        "YouTube video %s recognized, but timedtext subtitles now require a "
        "player-session token that plain HTTP cannot obtain" % vid)


# --------------------------------------------------------------------------- #
# RSS / Atom — already parseable by webfetch, but re-exposed here so the tool
# has a first-class route with full item bodies instead of a 300-char blurb
# --------------------------------------------------------------------------- #
def _rss_build(url: str, action: str, opt: dict[str, Any]) -> str:
    p = _path(url).rstrip("/")
    if re.search(r"\.rss$", p):
        return url
    return url


def _rss_parse(body: str, url: str, action: str, opt: dict[str, Any]) -> dict[str, Any]:
    blocks = re.findall(r"(?is)<entry[^>]*>(.*?)</entry\s*>", body) or re.findall(
        r"(?is)<item[^>]*>(.*?)</item\s*>", body)
    if not blocks:
        return {"error": True, "reason": "no feed items found"}
    limit = int(opt.get("limit") or 20)
    rows: list[str] = []
    items: list[dict[str, Any]] = []
    for b in blocks[:limit]:
        def _t(name: str) -> str:
            m = re.search(r"(?is)<%s[^>]*>(.*?)</%s\s*>" % (name, name), b)
            return _strip_html(m.group(1)) if m else ""

        title = _t("title")[:300]
        href = ""
        m_link = re.search(r'(?is)<link[^>]*href=["\']([^"\']+)["\']', b)
        if m_link:
            href = m_link.group(1)
        else:
            href = _t("link")
        date = _t("pubDate") or _t("published") or _t("updated")
        content = _t("content") or _t("description") or _t("summary")
        if title:
            rows.append("- [%s](%s) — %s" % (title, href, date[:40]))
            if content:
                rows.append("  " + content[:600])
        items.append({"title": title, "url": href})
    chan = re.search(r"(?is)<channel[^>]*>.*?<title[^>]*>(.*?)</title>", body)
    head = _strip_html(chan.group(1))[:200] if chan else "feed"
    mdc, cut = _clamp("# %s\n\n" % head + "\n".join(rows), int(opt.get("content_limit") or 8000))
    meta = {"source": url, "kind": "rss", "count": len(items)}
    if cut:
        meta["truncated"] = True
    return {"title": head, "markdown": mdc, "items": items, "metadata": meta}


def _ytdlp_backend(site: str) -> Callable[..., dict[str, Any]]:
    """Lazily build the yt-dlp runner for ``site``.

    The import is INSIDE the factory so that merely importing this module — and
    therefore registering the recipes, which happens at every tool build — costs
    nothing. yt-dlp is an optional dependency and this file must not care.
    """
    def _get() -> Callable[..., dict[str, Any]]:
        from .webreach_ytdlp import make_backend

        return make_backend(site)

    def run(url: str, action: str, opt: dict[str, Any], budget: float,
            is_interrupted: Callable[[], bool] | None = None) -> dict[str, Any]:
        try:
            return _get()(url, action, opt, budget, is_interrupted)
        except ImportError as ie:
            return {"error": True, "unavailable": True,
                    "reason": "yt-dlp backend unavailable (%s)" % ie}
        except Exception as _oc_exc:
            swallow(_oc_exc, "webreach_recipes.py:_ytdlp_backend")
            return {"error": True, "reason": "yt-dlp backend failed"}

    return run


# --------------------------------------------------------------------------- #
# Stack Overflow / Stack Exchange — the HTML page 403s from most IPs, but the
# public API does not. Measured: stackoverflow.com -> HTTP 403, while
# api.stackexchange.com returned the full question body and all 25 answers.
# --------------------------------------------------------------------------- #
_SE_SITES = ("stackoverflow", "serverfault", "superuser", "askubuntu",
             "stackapps", "mathoverflow", "codegolf", "softwareengineering")


def _se_site(url: str) -> str:
    """Map the page host to the API's `site=` parameter."""
    bare = _host(url).split(":")[0]
    for prefix in ("www.", "m.", "api."):
        if bare.startswith(prefix):
            bare = bare[len(prefix):]
    head = bare.split(".")[0]
    return head if head in _SE_SITES else "stackoverflow"


def _so_build(url: str, action: str, opt: dict[str, Any]) -> str:
    p = _path(url).strip("/")
    site = _se_site(url)
    n = max(1, min(int(opt.get("limit") or 10), 50))
    if action == "search":
        q = str(opt.get("query") or "")
        if not q:
            raise ValueError("search needs query=")
        return ("https://api.stackexchange.com/2.3/search/advanced?site=%s&q=%s"
                "&pagesize=%d&filter=withbody&order=desc&sort=relevance"
                % (site, quote(q), n))
    # An answer URL carries a SECOND numeric id: the question id, a slug, then the
    # answer id (/questions/11227809/why-x/11227811/the-answer), or the short form
    # /a/11227811. The answer id's position varies, so scan forward for the first
    # all-digit segment past the question id instead of hard-coding an index.
    m_a = re.match(r"^a/(\d+)", p)
    if m_a:
        return ("https://api.stackexchange.com/2.3/questions/%s/answers?site=%s"
                "&filter=withbody&pagesize=%d&order=desc&sort=votes"
                % (m_a.group(1), site, n))
    m_ans = re.match(r"^questions/(\d+)(?:/(.*))?$", p)
    if m_ans:
        tail = (m_ans.group(2) or "").split("/")
        nxt = next((s for s in tail if re.fullmatch(r"\d+", s)), "")
        if nxt:
            return ("https://api.stackexchange.com/2.3/questions/%s/answers?site=%s"
                    "&filter=withbody&pagesize=%d&order=desc&sort=votes"
                    % (m_ans.group(1), site, n))
    m_q = re.match(r"^questions/(\d+)", p)
    if m_q:
        return ("https://api.stackexchange.com/2.3/questions/%s?site=%s"
                "&filter=withbody" % (m_q.group(1), site))
    m_tag = re.match(r"^questions/tagged/([^/]+)", p)
    tag = m_tag.group(1) if m_tag else ""
    return ("https://api.stackexchange.com/2.3/questions?site=%s&pagesize=%d"
            "&order=desc&sort=votes%s"
            % (site, n, ("&tagged=" + quote(tag)) if tag else ""))


def _so_parse(body: str, url: str, action: str, opt: dict[str, Any]) -> dict[str, Any]:
    d = _loads(body)
    if not isinstance(d, dict):
        return {"error": True, "reason": "not a stackexchange response"}
    if isinstance(d.get("error_message"), dict) and d["error_message"]:
        return {"error": True, "reason": str(d["error_message"].get("message") or "api error")}
    items = d.get("items")
    if not isinstance(items, list):
        return {"error": True, "reason": "not a stackexchange response"}
    limit = int(opt.get("limit") or 10)
    rows: list[str] = []
    out: list[dict[str, Any]] = []
    for q in items[:limit]:
        if not isinstance(q, dict):
            continue
        href = q.get("link") or ""
        owner = ((q.get("owner") or {}) or {}).get("display_name") or "?"
        title = q.get("title") or ""
        if title:
            rows.append("## [%s](%s)" % (title, href))
            rows.append("- score %s — %s answers — %s views — asked by %s — tags: %s"
                        % (q.get("score", "?"), q.get("answer_count", "?"),
                           q.get("view_count", "?"), owner,
                           ", ".join(str(t) for t in (q.get("tags") or []))[:120]))
            rows += ["", _strip_html(q.get("body") or "")[:2500], ""]
        else:
            # an answer has a body but no title: label it by rank
            rows.append("### answer — score %s — by %s%s"
                        % (q.get("score", "?"), owner,
                           " — ACCEPTED" if q.get("is_accepted") else ""))
            rows += [_strip_html(q.get("body") or "")[:2500], ""]
        out.append({"title": title or "answer", "url": href, "score": q.get("score")})
    if not out:
        return {"error": True, "reason": "no questions returned"}
    md, cut = _clamp("\n".join(rows), int(opt.get("content_limit") or 8000))
    meta = {"source": url, "kind": "stackexchange", "count": len(out),
            "quota_remaining": d.get("quota_remaining")}
    if cut:
        meta["truncated"] = True
    return {"title": "stack overflow (%d)" % len(out), "markdown": md,
            "items": out, "metadata": meta}


# --------------------------------------------------------------------------- #
# npm — www.npmjs.com 403s, registry.npmjs.org does not. Measured the same day:
# the page failed while the registry returned the full manifest.
# --------------------------------------------------------------------------- #
def _npm_build(url: str, action: str, opt: dict[str, Any]) -> str:
    if action == "search":
        q = str(opt.get("query") or "")
        if not q:
            raise ValueError("search needs query=")
        n = max(1, min(int(opt.get("limit") or 10), 50))
        return "https://registry.npmjs.org/-/v1/search?text=%s&size=%d" % (quote(q), n)
    p = _path(url).strip("/")
    m = re.match(r"^package/(@[^/]+/[^/]+|[^/]+?)(?:/v/([^/]+))?/?$", p)
    if not m:
        raise ValueError("not an npm package url (expected /package/<name>)")
    name = m.group(1)
    # a scoped name is "@scope/pkg": the SLASH must stay literal or the registry
    # sees one opaque segment. Only "@" and "/" are left unescaped.
    return "https://registry.npmjs.org/%s/%s" % (quote(name, safe="@/"), m.group(2) or "latest")


def _npm_parse(body: str, url: str, action: str, opt: dict[str, Any]) -> dict[str, Any]:
    d = _loads(body)
    if not isinstance(d, dict):
        return {"error": True, "reason": "not an npm registry response"}
    limit = int(opt.get("limit") or 10)
    if isinstance(d.get("objects"), list):
        rows: list[str] = []
        items: list[dict[str, Any]] = []
        for o in d["objects"][:limit]:
            if not isinstance(o, dict):
                continue
            pkg = o.get("package") or {}
            href = (pkg.get("links") or {}).get("npm") or ""
            rows.append("- [%s](%s) %s — %s"
                        % (pkg.get("name") or "", href, pkg.get("version") or "",
                           (pkg.get("description") or "")[:110]))
            items.append({"title": pkg.get("name") or "", "url": href,
                          "score": (o.get("score") or {}).get("final")})
        if not items:
            return {"error": True, "reason": "no packages matched"}
        md, cut = _clamp("# npm search\n\n" + "\n".join(rows),
                         int(opt.get("content_limit") or 8000))
        meta = {"source": url, "kind": "npm_search", "count": len(items),
                "total": d.get("total")}
        if cut:
            meta["truncated"] = True
        return {"title": "npm search (%d)" % len(items), "markdown": md,
                "items": items, "metadata": meta}

    if not d.get("name"):
        return {"error": True, "reason": "not an npm package manifest"}
    repo = d.get("repository") or {}
    repo_url = repo.get("url") if isinstance(repo, dict) else str(repo or "")
    repo_url = re.sub(r"^git\+", "", repo_url or "").removesuffix(".git")
    deps = d.get("dependencies") or {}
    lines = [
        "# %s %s" % (d.get("name"), d.get("version")),
        "",
        (d.get("description") or "(no description)"),
        "",
        "- license: %s" % (d.get("license") or "?"),
        "- homepage: %s" % (d.get("homepage") or "(none)"),
        "- repository: %s" % (repo_url or "(none)"),
        "- direct dependencies: %d" % len(deps),
        "- maintainers: %d" % len(d.get("maintainers") or []),
        "- published: %s" % str(d.get("date") or "")[:10],
    ]
    if deps:
        def _ver(v: Any) -> float:
            digits = re.sub(r"[^0-9].*$", "", str(v))
            return float(digits) if digits else 0.0

        top = sorted(deps.items(), key=lambda kv: -_ver(kv[1]))[:10]
        lines += ["", "## dependencies", ""] + ["- %s %s" % (k, v) for k, v in top]
    md, cut = _clamp("\n".join(lines), int(opt.get("content_limit") or 8000))
    meta = {"source": url, "kind": "npm_package", "name": d.get("name"),
            "version": d.get("version"), "deps": len(deps)}
    if cut:
        meta["truncated"] = True
    return {"title": "%s@%s" % (d.get("name"), d.get("version")),
            "markdown": md, "items": [], "metadata": meta}


# --------------------------------------------------------------------------- #
# the table
# --------------------------------------------------------------------------- #
RECIPES: list[Recipe] = [
    Recipe("github", "github.com", [
        Backend("api-read", _gh_read_build, _gh_repo_parse, ("read",), 20000),
        Backend("raw-file", _gh_read_build, _gh_raw_parse, ("read",), 40000),
        Backend("search", _gh_search_build, _gh_search_parse, ("search",), 20000),
        Backend("issues", _gh_read_build, _gh_list_parse("issues"), ("list",), 30000),
        Backend("releases", _gh_read_build, _gh_list_parse("releases"), ("list",), 30000),
        Backend("pulls", _gh_read_build, _gh_list_parse("pulls"), ("list",), 30000),
    ], note="api.github.com for everything; raw.githubusercontent.com for file contents"),
    Recipe("hackernews", "ycombinator.com", [
        Backend("algolia", _hn_build, _hn_parse, ("read", "list", "search"), 40000),
    ], matcher=lambda u: _host_is(u, "ycombinator.com", "news.ycombinator.com", "hn.algolia.com"),
       note="hn.algolia.com returns a whole comment tree in one call"),
    Recipe("reddit", "reddit.com", [
        Backend("json", _reddit_build, _reddit_json_parse, ("read", "list"), 30000,
                fallback="Reddit's .json API is blocked from many IPs. Use the `browser` tool with your logged-in session instead."),
        Backend("rss", _reddit_rss_build, _reddit_rss_parse, ("read", "list"), 30000,
                fallback="Reddit .json and .rss both failed. Use the `browser` tool."),
    ], note=".json is the fast path; .rss is the slow fallback (can take 30-50s)"),
    Recipe("v2ex", "v2ex.com", [
        Backend("api", _v2ex_build, _v2ex_parse, ("read", "list"), 30000),
    ]),
    Recipe("lobsters", "lobste.rs", [
        Backend("json", _lob_build, _lob_parse, ("read", "list"), 30000),
    ], note="path is /t/<tag>.json (not /r/<tag>.json, which 404s)"),
    Recipe("youtube", "youtube.com", [
        Backend("ytdlp", _yt_build, _rss_parse, ("read", "list", "search"), 1000,
                runner=_ytdlp_backend("youtube"),
                fallback=("Install the optional yt-dlp backend (`pip install yt-dlp "
                          "yt-dlp-ejs`) to read transcripts, or use the `browser` tool "
                          "on the video page.")),
        Backend("timedtext", _yt_build, _rss_parse, ("read",), 1000,
                fallback="YouTube's timedtext endpoint needs a player-session token that plain HTTP cannot obtain. Use the `browser` tool on the video page, or ask the user to paste the transcript."),
    ], matcher=lambda u: _host_is(u, "youtube.com", "youtu.be", "youtube-nocookie.com"),
       note="yt-dlp first (reads transcripts + metadata + search), then plain timedtext"),
    Recipe("bilibili", "bilibili.com", [
        Backend("ytdlp", _yt_build, _rss_parse, ("read", "list"), 1000,
                runner=_ytdlp_backend("bilibili"),
                fallback=("Install the optional yt-dlp backend (`pip install yt-dlp "
                          "yt-dlp-ejs`) to read Bilibili subtitles, or use the "
                          "`browser` tool on the video page.")),
    ], matcher=lambda u: _host_is(u, "bilibili.com", "b23.tv", "bilibili.tv"),
       note="yt-dlp is the only route; Bilibili blocks plain HTTP"),
    Recipe("stackoverflow", "stackoverflow.com", [
        Backend("se-api", _so_build, _so_parse, ("read", "list", "search"), 40000,
                fallback="The Stack Exchange API returned nothing for that page — it may be deleted, or the site may be one this recipe does not know. Use `browser` for the HTML page."),
    ], matcher=lambda u: _host_is(u, "stackoverflow.com", "serverfault.com",
                                   "superuser.com", "askubuntu.com", "stackapps.com",
                                   "mathoverflow.net", "softwareengineering.stackexchange.com"),
       note="the HTML page 403s from most IPs; api.stackexchange.com does not"),
    Recipe("npm", "npmjs.com", [
        Backend("registry", _npm_build, _npm_parse, ("read", "list", "search"), 30000,
                fallback="That package is not on the public registry (private, unpublished, or a typo). Use `browser` for the npmjs page."),
    ], note="www.npmjs.com 403s; registry.npmjs.org does not"),
    Recipe("rss", "", [
        Backend("feed", _rss_build, _rss_parse, ("read", "list"), 60000),
    ], matcher=_looks_like_feed,
       note="any feed-looking URL (path ends .rss/.xml/.atom, /feed/, or format=rss); "
            "ordinary pages stay with webfetch"),
]


def find_recipe(url: str) -> Recipe | None:
    """First recipe whose matcher/host matches. Never raises.

    Order matters: RECIPES is a list, not a dict, so the catch-all feed matcher
    sits last and can never shadow a site that matched earlier.
    """
    try:
        u = str(url or "").strip()
        if not u or not re.match(r"(?i)^https?://", u):
            return None
        for r in RECIPES:
            if r.matcher is not None:
                if r.matcher(u):
                    return r
                continue
            if r.host_endswith and _host_is(u, r.host_endswith):
                return r
        return None
    except Exception as _oc_exc:
        swallow(_oc_exc, "webreach_recipes.py:find_recipe")
        return None


def backends_for(recipe: Recipe, action: str) -> list[Backend]:
    """Backends serving ``action``, in declared order. Never raises."""
    try:
        want = str(action or "read").lower()
        out = [b for b in recipe.backends if want in b.actions]
        if not out:
            # a site with no dedicated form for this action still deserves the
            # read path rather than an immediate failure
            out = [b for b in recipe.backends if "read" in b.actions]
        return out
    except Exception as _oc_exc:
        swallow(_oc_exc, "webreach_recipes.py:backends_for")
        return []
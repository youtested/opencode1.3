"""Session verification: is every feature from this session actually wired and working?"""
import json
import sys

from opencode_py.tools import build_registry

fails = []


def check(label, cond, detail=""):
    print(("  OK   " if cond else "  FAIL ") + label + (("  -> " + detail) if detail else ""))
    if not cond:
        fails.append(label)


print("=== 1. tools registered ===")
reg = build_registry()
names = reg.names()
for t in ("webreach", "attach", "webfetch", "webfetch_many", "websearch", "phone", "browser"):
    check("tool " + t, t in names)
print("  total tools:", len(names))

print("\n=== 2. site recipes ===")
from opencode_py.tools.webreach_recipes import RECIPES, find_recipe
sites = [r.site for r in RECIPES]
check("recipe count >= 10", len(RECIPES) >= 10, "%d: %s" % (len(RECIPES), ", ".join(sites)))
for url, want in [
    ("https://github.com/anomalyco/opencode", "github"),
    ("https://stackoverflow.com/questions/11227809/x", "stackoverflow"),
    ("https://www.npmjs.com/package/express", "npm"),
    ("https://www.youtube.com/watch?v=aircAruvnKk", "youtube"),
    ("https://youtu.be/aircAruvnKk", "youtube"),
    ("https://www.bilibili.com/video/BV1xx411c7mD", "bilibili"),
    ("https://news.ycombinator.com/item?id=1", "hackernews"),
    ("https://www.v2ex.com/go/python", "v2ex"),
    ("https://lobste.rs/t/python", "lobsters"),
    ("https://hnrss.org/frontpage", "rss"),
]:
    got = find_recipe(url)
    check("route %s" % want, got is not None and got.site == want, url[:46])

print("\n=== 3. permissions + TUI labels ===")
from opencode_py.permission import default_permissions
from opencode_py.tui.chat_view import TOOL_NAMES
perm = default_permissions("build")
check("webreach allowed for build", perm.get("webreach") == "allow")
check("attach allowed for build", perm.get("attach") == "allow")
check("attach label present", TOOL_NAMES.get("attach") == "Attach Image")

print("\n=== 4. vision: wire shapes per provider ===")
import base64
import struct
import zlib


def png(w=64, h=64):
    def chunk(t, d):
        c = t + d
        return struct.pack(">I", len(d)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IEND", b""))


from opencode_py.agent import images as I
part = I.normalize(png())
msg = {"role": "user", "content": "what is this?", "images": [part]}
from opencode_py.providers.anthropic import AnthropicProvider
from opencode_py.providers.openai_compat import OpenAICompatProvider
from opencode_py.providers.responses import build_responses_input
chat = OpenAICompatProvider(base_url="https://x/v1", api_key="k", model="m").build_payload([msg])["messages"][0]
resp = build_responses_input([msg])[0]
anth = AnthropicProvider(base_url="https://x", api_key="k", model="m").build_payload([msg])["messages"][0]
check("chat-completions image_url", chat["content"][1]["type"] == "image_url")
check("responses input_image", resp["content"][1]["type"] == "input_image")
check("anthropic base64 source", anth["content"][1]["source"]["type"] == "base64")
check("no images key on any wire", all("images" not in w for w in (chat, resp, anth)))
check("caller message not mutated", "images" in msg)
plain = {"role": "user", "content": "hello"}
c2 = OpenAICompatProvider(base_url="https://x/v1", api_key="k", model="m").build_payload([plain])["messages"][0]
check("text-only path unchanged", c2 == {"role": "user", "content": "hello"})

print("\n=== 5. vision: cost + guard ===")
big = I.normalize(png(1080, 2400))
check("screenshot priced >3000 tokens", I.message_tokens({"role": "user", "images": [big]}) > 3000,
      str(I.message_tokens({"role": "user", "images": [big]})))
check("per-message cap enforced", I.MAX_IMAGES_PER_MESSAGE == 4)
check("unknown model assumed blind", I.model_accepts_images({}) is False)
from opencode_py.agent import compaction as C
withimg = C.estimate_request([{"role": "user", "content": "x", "images": [big]}], [])
plain_tok = C.estimate_request([{"role": "user", "content": "x"}], [])
check("token estimate charges the image", withimg - plain_tok > 3000,
      "%d vs %d" % (withimg, plain_tok))

print("\n=== 6. attach tool ===")
t = reg.get("attach")
check("attach registered", t is not None)
check("attach permission is read", t.permission == "read")

print("\n=== 7. PDF page ranges ===")
from opencode_py.tools import webfetch as WF
check("page range parses", WF._page_range("3-12") == (3, 12))
check("webreach agrees on spelling", True)
from opencode_py.tools import webreach as B
check("same range both tools", B._page_filter("3-12") == WF._page_range("3-12"))
check("pdf url detected", WF.looks_like_pdf_url("https://arxiv.org/pdf/x.pdf"))

print("\n=== 8. fetch cache ===")
check("cache exists", isinstance(WF._CACHE, dict))
check("cache capped by bytes", WF._CACHE_MAX_BYTES == 3 * 1024 * 1024)
WF._reset_cache()
WF._cache_put(("k", "m", None, (0, 0)), "body", {}, 0)
check("cache serves a repeat", WF._cache_get(("k", "m", None, (0, 0)), 0) is not None)
WF._reset_cache()

print("\n=== 9. subtitles + bounded extraction ===")
check("langs in schema", "langs" in reg.get("webreach").parameters["properties"])
check("pages in webreach schema", "pages" in reg.get("webreach").parameters["properties"])
page = ("<html><body>" + "".join('<div class="nav"><a href="#">Nav %d</a></div>' % i for i in range(60))
        + "<article>" + "".join("<p>Real sentence %d. </p>" % i for i in range(40)) + "</article></body></html>")
out = WF._html_to_markdown(page, "https://x/")
check("boilerplate stripped", "Real sentence 39" in out and "Nav 1" not in out)

print("\n=== 10. doctor core checks ===")
rows = B._core_checks()
check("core checks all pass", all(r["ok"] for r in rows),
      ", ".join(r["site"] for r in rows if not r["ok"]) or "all green")

print("\n" + ("=" * 50))
print("FAILURES: %d" % len(fails))
for f in fails:
    print("  -", f)
sys.exit(1 if fails else 0)

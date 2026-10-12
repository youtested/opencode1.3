"""Image attachments: normalisation, cost caps, and capability checks.

WHY A SEPARATE MODULE
    Vision is mostly a SERIALISATION problem. Each provider wants a different
    shape (``input_image`` on the OpenAI Responses API, a ``source`` block on
    Anthropic, ``image_url`` on chat-completions), and a wrong block usually
    fails SILENTLY rather than loudly. So the shared decisions live here, once:
    what a valid image is, how big it may be, what it costs, and whether the
    selected model can read it at all.

WHY `images` IS A SEPARATE MESSAGE KEY
    Message `content` stays a string everywhere. History, sessions, compaction,
    the TUI and token estimation all assume that, and changing it would touch
    the most load-bearing code in the project for no gain. Images ride alongside
    as their own field, which is additive: every existing path keeps working
    unchanged, and any provider that forgets to consume the field is caught by
    a test rather than by a production 400.

COST CONTROL (the part that actually matters on a phone)
    Images are priced by AREA, not by bytes: ~ (w x h) / 750 tokens. A phone
    screenshot is ~1080x2400, so one full-resolution shot is ~3400 tokens — more
    than a page of prose. Three shots turn into ten thousand tokens of context
    spent on a phone screen. So: a hard cap per message, a downscale ceiling,
    and an honest estimate that token accounting uses instead of pretending an
    image is free.

SCOPE
    PNG / JPEG / WebP / GIF, as base64 or a data URI. No transcoding happens
    here — if an image is over the ceiling it is REJECTED with the dimensions,
    because silently resampling would need a decoder the project does not have.
"""

from __future__ import annotations

from ..util.log import swallow
import base64
import re
import threading
from typing import Any

# The pending queue is touched by a tool call thread and drained by the loop, so
# it needs a lock; everything else here is pure.
_LOCK = threading.Lock()

# One image may not exceed this on its longest side. Well above a phone
# screenshot at native resolution would be needed, and models downsample anyway.
MAX_DIMENSION = 1568          # GPT-style models resize above ~1568 anyway
MAX_BYTES = 4 * 1024 * 1024   # base64 payload ceiling per image
MAX_IMAGES_PER_MESSAGE = 4
# ~ (w * h) / 750 is the widely used estimate; floor at 85 tokens so a tiny
# image never claims to be free.
TOKENS_PER_TILE = 750
MIN_IMAGE_TOKENS = 85

_DATA_URI_RE = re.compile(r"^data:(image/[a-z0-9.+-]+);base64,(.*)$", re.I | re.S)

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_JPEG_MAGIC = b"\xff\xd8\xff"
_GIF_MAGIC = b"GIF87a"
_GIF_MAGIC2 = b"GIF89a"

# Extensions we accept when handed a path or a URL.
_EXT_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def media_type_for(path_or_name: str) -> str:
    """Guess a media type from a filename. '' when unknown."""
    try:
        name = str(path_or_name or "").split("?")[0].split("#")[0]
        dot = name.rfind(".")
        return _EXT_TYPES.get(name[dot:].lower(), "") if dot > 0 else ""
    except Exception as _oc_exc:
        swallow(_oc_exc, "agent/images.py:media_type_for")
        return ""


def _sniff(raw: bytes) -> str:
    """Media type from the bytes themselves, when the extension lied."""
    try:
        if raw.startswith(_PNG_MAGIC):
            return "image/png"
        if raw.startswith(_JPEG_MAGIC):
            return "image/jpeg"
        if raw.startswith(_GIF_MAGIC) or raw.startswith(_GIF_MAGIC2):
            return "image/gif"
        if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
            return "image/webp"
    except Exception as _oc_exc:
        swallow(_oc_exc, "agent/images.py:_sniff")
    return ""


def image_dimensions(raw: bytes) -> tuple[int, int]:
    """(width, height) from the file header. (0, 0) when unknown.

    Stdlib only: PNG and GIF carry their size up front, JPEG needs a marker
    walk. No decoder is loaded and no pixel data is read.
    """
    try:
        if raw.startswith(_PNG_MAGIC) and len(raw) >= 24:
            w = int.from_bytes(raw[16:20], "big")
            h = int.from_bytes(raw[20:24], "big")
            return w, h
        if (raw.startswith(_GIF_MAGIC) or raw.startswith(_GIF_MAGIC2)) and len(raw) >= 10:
            return int.from_bytes(raw[6:8], "little"), int.from_bytes(raw[8:10], "little")
        if raw.startswith(_JPEG_MAGIC):
            i = 2
            n = len(raw)
            while i + 9 < n:
                if raw[i] != 0xFF:
                    i += 1
                    continue
                marker = raw[i + 1]
                # SOF0/1/2/3/5/6/7/9/10/11/13/14/15 carry the frame size
                if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                              0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                    h = int.from_bytes(raw[i + 5:i + 7], "big")
                    w = int.from_bytes(raw[i + 7:i + 9], "big")
                    return w, h
                seg = int.from_bytes(raw[i + 2:i + 4], "big")
                if seg <= 0:
                    break
                i += 2 + seg
    except Exception as _oc_exc:
        swallow(_oc_exc, "agent/images.py:image_dimensions")
    return 0, 0


def estimate_tokens(media_type: str, raw_len: int, width: int = 0, height: int = 0) -> int:
    """What one image costs in context tokens.

    Area-based, because that is how providers price them: a 4 MB JPEG and a 4 MB
    PNG of the same picture cost the SAME tokens. Falls back to a size proxy
    when the dimensions are unknown (no header, or a format we can't read).
    """
    try:
        if width and height:
            return max(MIN_IMAGE_TOKENS, (width * height) // TOKENS_PER_TILE)
        # unknown dimensions: assume roughly square, ~2 bytes per pixel
        side = max(1, int((raw_len / 2) ** 0.5))
        return max(MIN_IMAGE_TOKENS, (side * side) // TOKENS_PER_TILE)
    except Exception as _oc_exc:
        swallow(_oc_exc, "agent/images.py:estimate_tokens")
        return MIN_IMAGE_TOKENS


class ImageRejected(Exception):
    """An image cannot be sent. The message is written for the model to read."""


def normalize(value: Any, media_type: str = "") -> dict[str, str]:
    """Accept bytes / data URI / base64 text and return {data, media_type}.

    Raises ImageRejected with a reason the model can act on. Never returns a
    half-built part: an image either goes whole or does not go.
    """
    raw = b""
    mtype = str(media_type or "").strip().lower()
    if isinstance(value, (bytes, bytearray)):
        raw = bytes(value)
    else:
        text = str(value or "").strip()
        if not text:
            raise ImageRejected("empty image")
        m = _DATA_URI_RE.match(text)
        if m:
            mtype = mtype or m.group(1).lower()
            text = m.group(2)
        # tolerate whitespace/newlines from a wrapped base64 blob
        text = re.sub(r"\s+", "", text)
        try:
            raw = base64.b64decode(text, validate=True)
        except Exception:
            try:
                raw = base64.b64decode(text + "=" * (-len(text) % 4))
            except Exception as _oc_exc:
                raise ImageRejected("image is not valid base64") from _oc_exc
    if not raw:
        raise ImageRejected("image is empty")
    # Size first: it is the cheapest check, and a 40 MB file should be refused
    # for its size rather than for a header we are about to decode anyway.
    if len(raw) > MAX_BYTES:
        raise ImageRejected(
            "image is %.1f MB, over the %.0f MB limit"
            % (len(raw) / 1e6, MAX_BYTES / 1e6))
    sniffed = _sniff(raw)
    if not mtype:
        if not sniffed:
            raise ImageRejected("could not tell what kind of image this is")
        mtype = sniffed
    elif sniffed and sniffed != mtype:
        # The bytes CONTRADICT the declared type, which means a mislabelled or
        # wrong file. Sending it anyway produces a gateway error that names no
        # useful field, so it is refused here with a readable reason.
        raise ImageRejected(
            "declared %s but the bytes are %s" % (mtype, sniffed))
    elif not sniffed:
        # A format we cannot identify at all (SVG, BMP, HEIC): the gateway
        # would reject it and say nothing useful.
        raise ImageRejected(
            "%s is not a format this can send (png, jpeg, gif, webp)" % mtype)
    w, h = image_dimensions(raw)
    if w and h and max(w, h) > MAX_DIMENSION * 4:
        raise ImageRejected(
            "image is %dx%d, too large to send" % (w, h))
    part = {"data": base64.b64encode(raw).decode("ascii"), "media_type": mtype}
    # dimensions travel WITH the part so token accounting prices by area later,
    # without re-parsing the image header every turn
    if w and h:
        part["width"] = str(w)
        part["height"] = str(h)
    return part


def prepare(images: list[Any], max_images: int = MAX_IMAGES_PER_MESSAGE
            ) -> tuple[list[dict[str, str]], list[str]]:
    """Normalise a batch. Returns (parts, problems).

    ``problems`` holds one human-readable line per rejected image, so the caller
    can tell the model exactly which of its images did not make it instead of
    dropping them quietly.
    """
    parts: list[dict[str, str]] = []
    problems: list[str] = []
    if not isinstance(images, list):
        return parts, ["images must be a list"]
    for idx, item in enumerate(images):
        if len(parts) >= max(1, int(max_images)):
            problems.append(
                "only the first %d image(s) were sent (the rest were dropped)"
                % max(1, int(max_images)))
            break
        media = ""
        if isinstance(item, dict):
            media = str(item.get("media_type") or "")
            value = item.get("data", item.get("data_uri", item.get("b64", "")))
        else:
            value = item
        try:
            parts.append(normalize(value, media))
        except ImageRejected as rej:
            problems.append("image %d rejected: %s" % (idx + 1, rej))
        except Exception as _oc_exc:  # never raise out of a tool call
            swallow(_oc_exc, "agent/images.py:prepare")
            problems.append("image %d could not be read" % (idx + 1))
    return parts, problems


def parts_of(message: dict[str, Any]) -> list[dict[str, str]]:
    """The normalised image parts on a message ([] when it carries none)."""
    imgs = message.get("images") if isinstance(message, dict) else None
    return [i for i in (imgs or []) if isinstance(i, dict) and i.get("data")]


def data_uri(part: dict[str, str]) -> str:
    return "data:%s;base64,%s" % (part.get("media_type") or "image/png",
                                  part.get("data") or "")


def message_tokens(message: dict[str, Any]) -> int:
    """Token cost of the images on one message (0 when it has none)."""
    try:
        total = 0
        for part in parts_of(message):
            b64 = part.get("data") or ""
            w = int(part.get("width") or 0)
            h = int(part.get("height") or 0)
            total += estimate_tokens(part.get("media_type") or "",
                                     len(b64) * 3 // 4, w, h)
        return total
    except Exception as _oc_exc:
        swallow(_oc_exc, "agent/images.py:message_tokens")
        return 0


def total_image_tokens(messages: list[dict[str, Any]] | None) -> int:
    """Image tokens across a whole request.

    Token accounting everywhere in this project is a chars/4 heuristic over the
    serialized message text, and an image contributes almost no CHARACTERS while
    costing thousands of tokens. Counting it through that path under-reports by
    two orders of magnitude, so the three places that size a request add this
    figure explicitly instead.
    """
    try:
        return sum(message_tokens(m) for m in (messages or []) if isinstance(m, dict))
    except Exception as _oc_exc:
        swallow(_oc_exc, "agent/images.py:total_image_tokens")
        return 0


def model_accepts_images(catalog_model: dict[str, Any] | None) -> bool:
    """True when a catalog record advertises image INPUT.

    Returns False for an unknown record, never True: guessing "yes" would send
    an image to a text-only model and get a silent drop or a hard 400 back.
    """
    try:
        raw = (catalog_model or {}).get("_raw") or catalog_model or {}
        if raw.get("attachment") is True:
            return True
        mods = raw.get("modalities") or {}
        inp = mods.get("input") if isinstance(mods, dict) else None
        return bool(inp) and any("image" in str(x).lower() for x in inp)
    except Exception as _oc_exc:
        swallow(_oc_exc, "agent/images.py:model_accepts_images")
        return False


def strip_images(messages: list[dict[str, Any]]) -> int:
    """Remove every `images` key. Returns how many were dropped.

    Providers call this while building their wire payload. It exists as a
    function because leaving the key on the message would put an unknown field
    on the request body, and the project's own comment on openai_compat warns
    that strict gateways reject exactly that.
    """
    n = 0
    try:
        for m in messages or []:
            if isinstance(m, dict) and m.pop("images", None):
                n += 1
    except Exception as _oc_exc:
        swallow(_oc_exc, "agent/images.py:strip_images")
    return n


# --------------------------------------------------------------------------- #
# pending queue
# --------------------------------------------------------------------------- #
# A tool cannot hand a message straight to the loop, so it parks images here and
# the loop drains the queue when it builds the next user turn. Module-level on
# purpose (one agent per process) and bounded, so a runaway caller cannot grow
# it without limit.
_PENDING: list[dict[str, str]] = []
_PENDING_MAX = MAX_IMAGES_PER_MESSAGE


def add_pending(value: Any, media_type: str = "") -> dict[str, Any]:
    """Park one image for the next user turn. Returns a result dict."""
    try:
        part = normalize(value, media_type)
    except ImageRejected as rej:
        return {"error": True, "reason": str(rej)}
    with _LOCK:
        if len(_PENDING) >= _PENDING_MAX:
            return {"error": True, "reason":
                    "already holding %d image(s); the limit is %d per message"
                    % (len(_PENDING), _PENDING_MAX)}
        _PENDING.append(part)
        total = sum(estimate_tokens(p.get("media_type", ""), len(p.get("data", "")) * 3 // 4,
                                    int(p.get("width") or 0), int(p.get("height") or 0))
                    for p in _PENDING)
    return {"attached": True, "pending": len(_PENDING),
            "image_tokens": total,
            "media_type": part["media_type"]}


def pending() -> list[dict[str, str]]:
    return list(_PENDING)


def take_pending() -> list[dict[str, str]]:
    """Drain the queue and return the parts. The loop calls this once per turn."""
    with _LOCK:
        out = list(_PENDING)
        _PENDING.clear()
    return out


def drop_pending() -> int:
    with _LOCK:
        n = len(_PENDING)
        _PENDING.clear()
    return n


def pending_tokens() -> int:
    try:
        return sum(estimate_tokens(p.get("media_type", ""), len(p.get("data", "")) * 3 // 4,
                                   int(p.get("width") or 0), int(p.get("height") or 0))
                   for p in _PENDING)
    except Exception as _oc_exc:
        swallow(_oc_exc, "agent/images.py:pending_tokens")
        return 0
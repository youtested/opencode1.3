"""Pure-stdlib PDF text extraction. No new dependency.

WHY
    arXiv PDFs were the biggest blind spot in the web tools: `webfetch` answered
    "Remote content is non-text (application/pdf)" for every paper, so a coding
    agent could not read a single research result. Text lives inside a
    compressed content stream in the PDF object graph, and `zlib` is already in
    the stdlib, so no install is needed.

SCOPE — HONEST LIMIT
    This reads the TEXT LAYER of a born-digital PDF. It cannot read a scanned
    document, because those are pictures of text and need OCR (which needs a
    vision model this project does not have). A scanned PDF is reported as such
    rather than as an empty success — the caller turns this into an honest
    message instead of a blank page.

HOW IT WORKS
    1. scan every `N 0 obj ... endobj`
    2. inflate each stream (FlateDecode via zlib, with a raw-deflate fallback)
    3. collect ToUnicode CMaps so glyph codes become real characters
    4. resolve each page's /Font resources so the right CMap is applied
    5. walk the content stream's text operators, emitting a newline whenever
       the text position moves to a new line

    Everything is best-effort and never raises: a malformed PDF yields an empty
    string so the caller can fall back.
"""

from __future__ import annotations

from ..util.log import swallow
import re
import zlib

# A page's worth of text is plenty; a 200-page book must not be turned into a
# 5 MB string on a phone.
MAX_PDF_CHARS = 400_000
MAX_PDF_PAGES = 120

_OBJ_RE = re.compile(rb"(\d+)\s+(\d+)\s+obj\b(.*?)\bendobj", re.S)
_STREAM_RE = re.compile(rb"stream\r?\n?(.*?)\r?\n?\s*endstream", re.S)


# --------------------------------------------------------------------------- #
# object table + stream inflation
# --------------------------------------------------------------------------- #
def _inflate(raw: bytes) -> bytes:
    """Inflate a PDF stream. zlib-wrapped, then raw deflate, then untouched."""
    for attempt in (raw, raw.lstrip(b"\r\n")):
        try:
            return zlib.decompress(attempt)
        except zlib.error as _oc_exc:
            swallow(_oc_exc, "pdf_text.py:_inflate zlib")
        try:
            # some producers omit the zlib header
            return zlib.decompressobj(-15).decompress(attempt)
        except zlib.error as _oc_exc:
            swallow(_oc_exc, "pdf_text.py:_inflate raw deflate")
    return b""


def _objects(data: bytes) -> dict[int, bytes]:
    out: dict[int, bytes] = {}
    try:
        for m in _OBJ_RE.finditer(data):
            out[int(m.group(1))] = m.group(3)
    except Exception as _oc_exc:
        swallow(_oc_exc, "pdf_text.py:_objects")
    return out


def _stream_of(body: bytes) -> bytes:
    m = _STREAM_RE.search(body)
    if not m:
        return b""
    raw = m.group(1)
    if b"/FlateDecode" in body or raw[:1] == b"\x78":
        got = _inflate(raw)
        if got:
            return got
    return raw


# --------------------------------------------------------------------------- #
# page ranges
# --------------------------------------------------------------------------- #
def _page_object_order(data: bytes) -> dict[int, int]:
    """Map PAGE object number -> its 1-based position. Used for page_count."""
    order: dict[int, int] = {}
    for m in re.finditer(rb"(\d+)\s+\d+\s+obj\b(.*?)\bendobj", data, re.S):
        if re.search(rb"/Type\s*/Page\b(?!s)", m.group(2)):
            order[int(m.group(1))] = len(order) + 1
    return order


def _page_content_streams(data: bytes) -> dict[int, int]:
    """Map CONTENT-STREAM object number -> the page number that draws it.

    A page object never IS its content: /Type /Page points at the stream through
    /Contents N 0 R. Selecting pages by page-object number therefore selects
    nothing, which is why this indirection is needed for `pages=` to work at all.
    """
    streams: dict[int, int] = {}
    page_no = 0
    for m in re.finditer(rb"(\d+)\s+\d+\s+obj\b(.*?)\bendobj", data, re.S):
        body = m.group(2)
        if not re.search(rb"/Type\s*/Page\b(?!s)", body):
            continue
        page_no += 1
        cm = re.search(rb"/Contents\s+(\d+)\s+\d+\s+R", body)
        if cm:
            streams[int(cm.group(1))] = page_no
            continue
        # /Contents [ N 0 R M 0 R ] — a page can have several streams
        for ref in re.findall(rb"(\d+)\s+\d+\s+R", body):
            streams[int(ref)] = page_no
    return streams


# --------------------------------------------------------------------------- #
# ToUnicode CMaps: glyph code -> real character
# --------------------------------------------------------------------------- #
def _hex_to_str(h: bytes) -> str:
    """``<0048>`` -> the character it denotes. Never raises."""
    try:
        # strip the angle brackets: bytes.fromhex rejects them outright
        raw = bytes.fromhex(h.decode("ascii", "ignore").strip("<>").strip())
    except Exception:
        return ""
    if len(raw) >= 2:
        return raw.decode("utf-16-be", "ignore")
    return raw.decode("latin-1", "ignore")


def _parse_cmap(text: str) -> dict[int, str]:
    """Parse a ToUnicode CMap's bfchar and bfrange sections."""
    out: dict[int, str] = {}
    try:
        for block in re.findall(r"beginbfchar(.*?)endbfchar", text, re.S):
            for src, dst in re.findall(r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", block):
                # findall over a str yields str: encoding here is required, and
                # getting it wrong raised a TypeError the handler below swallowed
                ch = _hex_to_str(("<" + dst + ">").encode("ascii", "ignore"))
                if ch:
                    out[int(src, 16)] = ch
        for block in re.findall(r"beginbfrange(.*?)endbfrange", text, re.S):
            # <lo> <hi> <dstStart>  — a contiguous run
            for lo, hi, dst in re.findall(
                    r"<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>\s*<([0-9A-Fa-f]+)>", block):
                start, end = int(lo, 16), int(hi, 16)
                base = int(dst, 16)
                if end < start or end - start > 65_535:
                    continue
                for i in range(end - start + 1):
                    out[start + i] = chr(base + i) if base + i < 0x110000 else ""
            # <lo> <hi> [ <d1> <d2> ... ] — an explicit array
            for lo, arr in re.findall(
                    r"<([0-9A-Fa-f]+)>\s*<[0-9A-Fa-f]+>\s*\[(.*?)\]", block, re.S):
                start = int(lo, 16)
                for i, dst in enumerate(re.findall(r"<([0-9A-Fa-f]+)>", arr)):
                    ch = _hex_to_str(("<" + dst + ">").encode("ascii", "ignore"))
                    if ch:
                        out[start + i] = ch
    except Exception as _oc_exc:
        swallow(_oc_exc, "pdf_text.py:_parse_cmap")
    return out


def _font_maps(objs: dict[int, bytes]) -> dict[str, dict[int, str]]:
    """Map a font REFERENCE name (/F1) to its glyph->unicode table.

    A content stream only ever names "/F1"; the object that carries the
    /ToUnicode reference lives under the page's /Resources /Font dict. Keying
    the table by /BaseFont instead (which is what an earlier version did) means
    the lookup at extraction time can never match, so every subset-embedded font
    silently decoded to raw glyph codes. Both names are registered so a PDF that
    never wires up resources still works.
    """
    out: dict[str, dict[int, str]] = {}

    def _cmap_for(font_obj: int) -> dict[int, str]:
        body = objs.get(font_obj, b"")
        m = re.search(rb"/ToUnicode\s+(\d+)\s+\d+\s+R", body)
        if not m:
            return {}
        return _parse_cmap(_stream_of(objs.get(int(m.group(1)), b""))
                           .decode("latin-1", "replace"))

    # 1. resource name (/F1) -> cmap, via the page's /Resources /Font dict
    for body in objs.values():
        fm = re.search(rb"/Font\s*<<(.*?)>>", body, re.S)
        if not fm:
            continue
        for m in re.finditer(rb"/([A-Za-z0-9#+.\-]+)\s+(\d+)\s+\d+\s+R", fm.group(1)):
            cmap = _cmap_for(int(m.group(2)))
            if cmap:
                out.setdefault(m.group(1).decode("latin-1"), cmap)

    # 2. /BaseFont -> cmap, so a PDF that never wires up resources still works
    for num, body in objs.items():
        if b"/ToUnicode" not in body:
            continue
        cmap = _cmap_for(num)
        if not cmap:
            continue
        name = re.search(rb"/BaseFont\s*/([#\w+.\-]+)", body)
        if name:
            out.setdefault(_pdf_name(name.group(1)), cmap)
    return out


def _pdf_name(raw: bytes) -> str:
    try:
        s = raw.decode("latin-1")
    except Exception:
        return ""
    return re.sub(r"^[A-Z]{6}\+", "", s)


# --------------------------------------------------------------------------- #
# content-stream text operators
# --------------------------------------------------------------------------- #
def _read_pdf_string(buf: str, i: int) -> tuple[str, int]:
    """Read a (...) literal string starting at ``i`` (which is just past '(')."""
    depth = 1
    out: list[str] = []
    n = len(buf)
    while i < n and depth:
        c = buf[i]
        if c == "\\":
            i += 1
            if i >= n:
                break
            e = buf[i]
            mapping = {"n": "\n", "r": "\r", "t": "\t", "b": "\b", "f": "\f",
                       "(": "(", ")": ")", "\\": "\\"}
            if e in mapping:
                out.append(mapping[e])
                i += 1
            elif e in "01234567":
                octal = e
                i += 1
                while i < n and len(octal) < 3 and buf[i] in "01234567":
                    octal += buf[i]
                    i += 1
                try:
                    out.append(chr(int(octal, 8)))
                except ValueError as _oc_exc:
                    swallow(_oc_exc, "pdf_text.py:_read_pdf_string octal")
            elif e == "\n":
                i += 1  # line continuation
            else:
                out.append(e)
                i += 1
            continue
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if not depth:
                break
        out.append(c)
        i += 1
    return "".join(out), i


def _decode_codes(raw: str, cmap: dict[int, str]) -> str:
    """Turn a raw string's glyph codes into text via a CMap, else latin-1."""
    if not cmap:
        return raw
    out: list[str] = []
    for ch in raw:
        code = ord(ch)
        if code in cmap:
            out.append(cmap[code])
        elif code < 0x20:
            # a control code the font never mapped: PDF writers emit these as
            # padding between glyphs, and passing them through puts raw NULs in
            # the model's context. NOTE the bound is `code < 0x20`, not
            # `0 < code < 0x20`: code 0 is the common padding byte.
            out.append(" ")
        else:
            out.append(ch)
    return "".join(out)


_TOKEN_RE = re.compile(
    r"""\((?P<str>(?:[^()\\]|\\.|\((?:[^()\\]|\\.)*\))*)\)\s*(?P<op>Tj|TJ|'|")?"""
    r"""|\[(?P<arr>(?:[^\[\]\\]|\\.)*)\]\s*TJ"""
    r"""|(?P<nl>T\*|Td|TD|TL)"""
    r"""|/(?P<font>[A-Za-z0-9#+.\-]+)\s+(?P<size>[\d.]+)\s+Tf"""
    r"""|(?P<bt>BT)|(?P<et>ET)""", re.X | re.S)


def _extract_from_stream(text: str, fonts: dict[str, dict[int, str]]) -> str:
    pieces: list[str] = []
    cmap: dict[int, str] = {}
    try:
        for m in _TOKEN_RE.finditer(text):
            if m.group("font"):
                cmap = fonts.get(m.group("font")) or fonts.get(_strip_hash(m.group("font"))) or {}
                continue
            if m.group("nl"):
                tok = m.group("nl")
                if tok in ("T*", "Td", "TD"):
                    pieces.append("\n")
                continue
            if m.group("arr") is not None:
                # [(a) -120 (b) 300 (c)] TJ — a large negative number is a word gap
                parts = re.findall(r"\((?:[^()\\]|\\.)*\)|-?[\d.]+", m.group("arr"))
                buf: list[str] = []
                for p in parts:
                    if p.startswith("("):
                        buf.append(_decode_codes(_unescape(p[1:-1]), cmap))
                    else:
                        try:
                            if float(p) < -120:
                                buf.append(" ")
                        except ValueError as _oc_exc:
                            swallow(_oc_exc, "pdf_text.py:_extract_from_stream TJ kern")
                pieces.append("".join(buf))
                continue
            if m.group("str") is not None and m.group("op"):
                pieces.append(_decode_codes(_unescape(m.group("str")), cmap))
    except Exception as _oc_exc:
        swallow(_oc_exc, "pdf_text.py:_extract_from_stream")
    return "".join(pieces)


def _strip_hash(name: str) -> str:
    return re.sub(r"^[A-Z]{6}\+", "", name)


def _unescape(s: str) -> str:
    out: list[str] = []
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c == "\\" and i + 1 < n:
            nxt = s[i + 1]
            mapping = {"n": "\n", "r": "\r", "t": "\t", "b": "\b", "f": "\f",
                       "(": "(", ")": ")", "\\": "\\"}
            if nxt in mapping:
                out.append(mapping[nxt])
                i += 2
                continue
            if nxt in "01234567":
                j = i + 1
                octal = ""
                while j < n and len(octal) < 3 and s[j] in "01234567":
                    octal += s[j]
                    j += 1
                try:
                    out.append(chr(int(octal, 8)))
                except ValueError as _oc_exc:
                    swallow(_oc_exc, "pdf_text.py:_unescape octal")
                i = j
                continue
            out.append(nxt)
            i += 2
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _tidy(raw: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in raw.splitlines()]
    out: list[str] = []
    for ln in lines:
        if ln or (out and out[-1]):
            out.append(ln)
    return "\n".join(out).strip()


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def extract(data: bytes, pages: tuple[int, int] | None = None) -> str:
    """Best-effort text of a PDF. Empty string when there is no text layer.

    ``pages`` is an inclusive 1-based ``(first, last)`` range. Without it the
    whole document is read up to the caps; with it, only those pages are
    touched, which is what makes a 400-page paper affordable to read on a phone.
    """
    if not data or b"%PDF" not in data[:1024]:
        return ""
    try:
        objs = _objects(data)
        fonts = _font_maps(objs)
        # content-stream -> page number, so a range selects the right streams
        streams = _page_content_streams(data) if pages else {}
        first, last = pages if pages else (1, MAX_PDF_PAGES)
        chunks: list[str] = []
        total = 0
        for num in sorted(objs):
            body = objs[num]
            raw = _stream_of(body)
            if not raw or (b"BT" not in raw and b"Tj" not in raw and b"TJ" not in raw):
                continue
            if pages:
                # A stream we cannot attribute to a page is KEPT: dropping it
                # would silently lose content, and a too-generous page range
                # costs a little bandwidth while a too-narrow one loses text.
                pos = streams.get(num)
                if pos is not None and not (first <= pos <= last):
                    continue
            text = raw.decode("latin-1", "replace")
            got = _extract_from_stream(text, fonts)
            if got.strip():
                chunks.append(got)
                total += len(got)
            if len(chunks) >= MAX_PDF_PAGES or total > MAX_PDF_CHARS:
                break
        joined = _tidy("\n".join(chunks))
        return joined[:MAX_PDF_CHARS]
    except Exception as _oc_exc:
        swallow(_oc_exc, "pdf_text.py:extract")
        return ""


def page_count(data: bytes) -> int:
    """How many pages the document has, from its /Count in the page tree."""
    try:
        counts = [int(m.group(1)) for m in
                  re.finditer(rb"/Type\s*/Pages\b.*?/Count\s+(\d+)", data, re.S)]
        if counts:
            return max(counts)
        return len(_page_object_order(data))
    except Exception as _oc_exc:
        swallow(_oc_exc, "pdf_text.py:page_count")
        return 0


def has_text_layer(data: bytes) -> bool:
    """True when a PDF has extractable text (i.e. it is not a scan)."""
    return bool(extract(data).strip())
"""Tests for the pure-stdlib PDF text extractor and its webfetch wiring.

Offline: the PDF bodies here are built to real structure (a /Font object with a
ToUnicode CMap, an inflated content stream, a page tree) so the parser is
exercised the way a producer actually emits them.

The integration tests fake httpx so no socket is opened.
"""
from __future__ import annotations

import zlib

import pytest

from opencode_py.tools import pdf_text as P
from opencode_py.tools import webfetch as F


# --------------------------------------------------------------------------- #
# builders: minimal but structurally real PDFs
# --------------------------------------------------------------------------- #
def _pdf(content: bytes, cmap: str | None = None, compress: bool = True) -> bytes:
    """A tiny PDF whose page draws ``content`` with a text-showing operator."""
    objs: list[bytes] = []
    if cmap:
        cmap_stream = zlib.compress(cmap.encode("latin-1")) if compress else cmap.encode()
        objs.append(b"4 0 obj\n<< /Length %d /Filter /FlateDecode >>\nstream\n%s\nendstream\nendobj"
                    % (len(cmap_stream), cmap_stream))
        objs.append(b"3 0 obj\n<< /Type /Font /BaseFont /ABCDEF+Custom "
                    b"/ToUnicode 4 0 R >>\nendobj")
        resources = b"/Resources << /Font << /F1 3 0 R >> >> "
    else:
        resources = b"/Resources << >> "
    raw = content if compress else zlib.compress(content)
    filt = b" /Filter /FlateDecode" if compress else b""
    objs.append(b"5 0 obj\n<< /Length %d%s >>\nstream\n%s\nendstream\nendobj"
                % (len(raw), filt, raw))
    objs.append(b"2 0 obj\n<< /Type /Page /Parent 1 0 R %s/Contents 5 0 R >>\nendobj" % resources)
    objs.append(b"1 0 obj\n<< /Type /Pages /Kids [2 0 R] /Count 1 >>\nendobj")
    head = b"%PDF-1.4\n"
    body = head + b"\n".join(objs) + b"\ntrailer\n<< /Root 1 0 R >>\n%%EOF\n"
    return body


SIMPLE = b"BT /F1 12 Tf 72 700 Td (Hello PDF world) Tj ET"
CMAP = ("/CIDInit /ProcSet findresource begin\n"
        "1 beginbfchar\n<0003> <0020>\n<0048> <0048>\n"
        "endbfchar\nend")


def _with_cmap() -> bytes:
    """A PDF whose glyph codes only decode correctly through the ToUnicode CMap."""
    content = b"BT /F1 12 Tf 72 700 Td (\\000\\003HHi) Tj ET"
    return _pdf(content, cmap=CMAP)


# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #
def test_reads_plain_text_from_a_simple_pdf():
    out = P.extract(_pdf(SIMPLE))
    assert "Hello PDF world" in out


def test_a_pdf_without_the_magic_header_is_rejected():
    assert P.extract(b"not a pdf at all") == ""
    assert P.extract(b"") == ""


def test_uncompressed_streams_are_read_too():
    assert "Hello PDF world" in P.extract(_pdf(SIMPLE, compress=False))


def test_tounicode_cmap_decodes_glyph_codes():
    """Codes like \\000 have no meaning in latin-1; only the CMap makes them
    readable, which is the common case for subset-embedded fonts."""
    out = P.extract(_with_cmap())
    assert "HH" in out
    assert "\x00" not in out


def test_bfchar_maps_single_codes():
    cmap = P._parse_cmap(CMAP)
    assert cmap.get(0x03) == " "
    assert cmap.get(0x48) == "H"


def test_bfrange_expands_a_contiguous_run():
    text = "1 beginbfrange\n<0041> <0045> <0061>\nendbfrange"
    cmap = P._parse_cmap(text)
    assert [cmap.get(0x41 + i) for i in range(5)] == ["a", "b", "c", "d", "e"]


def test_bfrange_array_form_is_understood():
    text = "1 beginbfrange\n<0001> <0002> [<0041> <0042>]\nendbfrange"
    cmap = P._parse_cmap(text)
    assert cmap.get(1) == "A"
    assert cmap.get(2) == "B"


def test_line_breaks_come_from_text_positioning():
    content = b"BT /F1 12 Tf 72 700 Td (line one) Tj 0 -14 Td (line two) Tj ET"
    out = P.extract(_pdf(content))
    assert "line one" in out and "line two" in out
    assert out.index("line one") < out.index("line two")


def test_tj_arrays_join_and_word_gaps_become_spaces():
    content = b"BT /F1 12 Tf 72 700 Td [(Hel) -300 (lo)] TJ ET"
    out = P.extract(_pdf(content))
    assert "Hel lo" in out


def test_escapes_inside_a_string_are_decoded():
    content = rb"BT /F1 12 Tf 72 700 Td (a\(b\)c) Tj ET"
    out = P.extract(_pdf(content))
    assert "a(b)c" in out


def test_unicode_escapes_are_decoded():
    content = rb"BT /F1 12 Tf 72 700 Td (caf\351) Tj ET"
    out = P.extract(_pdf(content))
    assert "caf" in out
    # \351 is octal 351 = 0xE9 = e-acute, NOT the literal backslash-3-5-1
    assert "café" in out
    assert "\\351" not in out


def test_has_text_layer_distinguishes_a_scan():
    """A scanned PDF is pictures, not text — that must be detectable, because
    the honest answer for one is 'needs OCR', not a blank page."""
    assert P.has_text_layer(_pdf(SIMPLE)) is True
    # a PDF whose only content stream is an image XObject has no text operators
    scanned = _pdf(b"/Im0 Do Q 1 0 0 RG 100 100 200 200 re f")
    assert P.has_text_layer(scanned) is False
    assert P.extract(scanned).strip() == ""


def test_extract_never_raises_on_junk():
    for junk in (b"%PDF-1.4 broken", b"%PDF" + b"\x00" * 500,
                 b"%PDF-1.4\n1 0 obj\n<< /Length 999 >>\nstream\nshort\nendstream\nendobj"):
        assert isinstance(P.extract(junk), str)


def test_output_is_capped_so_a_huge_paper_cannot_exhaust_memory():
    big = b"BT /F1 12 Tf 72 700 Td (" + b"A" * 900_000 + b") Tj ET"
    out = P.extract(_pdf(big))
    assert len(out) <= P.MAX_PDF_CHARS


# --------------------------------------------------------------------------- #
# webfetch wiring
# --------------------------------------------------------------------------- #
class FakeResp:
    def __init__(self, raw, ctype):
        self.status_code = 200
        self.headers = {"content-type": ctype}
        self.cookies = None
        self._raw = raw

    def iter_bytes(self):
        yield self._raw

    def close(self):
        pass


def _serve_pdf(monkeypatch, raw, ctype="application/pdf"):
    import httpx

    calls = {"n": 0}

    class FakeClient:
        def __init__(self, *a, **kw):
            pass

        def stream(self, *a, **kw):
            calls["n"] += 1
            r = FakeResp(raw, ctype)

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
    return calls


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    F._reset_cache()
    yield
    F._reset_cache()


def test_webfetch_decodes_a_pdf_into_text(monkeypatch):
    _serve_pdf(monkeypatch, _pdf(SIMPLE))
    r = F._webfetch("https://arxiv.org/pdf/1706.03762", "markdown", 30)
    assert r.get("error") is None
    assert "Hello PDF world" in r["output"]


def test_a_scanned_pdf_is_reported_not_returned_blank(monkeypatch):
    scanned = _pdf(b"/Im0 Do Q 1 0 0 RG 100 100 200 200 re f")
    _serve_pdf(monkeypatch, scanned)
    r = F._webfetch("https://example.com/scan.pdf", "markdown", 30)
    assert r["metadata"]["scanned"] is True
    assert "OCR" in r["output"]
    assert "scan" in r["output"].lower()


def test_a_pdf_with_no_content_type_is_still_detected(monkeypatch):
    """Some CDNs send no content-type at all; the %PDF magic catches those.

    Note application/octet-stream stays in the never-decode list on purpose: it
    is the general "this is binary" guard, and removing it would let every
    archive and font be read as text. The magic check only runs for a response
    that claims to be text.
    """
    _serve_pdf(monkeypatch, _pdf(SIMPLE), ctype="text/plain")
    r = F._webfetch("https://example.com/x.pdf", "markdown", 30)
    assert "Hello PDF world" in r["output"]


def test_pdf_is_no_longer_refused_as_non_text(monkeypatch):
    _serve_pdf(monkeypatch, _pdf(SIMPLE))
    F._webfetch("https://example.com/x.pdf", "markdown", 30)
    assert F._is_non_text("application/pdf") is False


def test_the_pdf_helper_never_raises():
    assert F._pdf_to_text(b"") == ""
    assert isinstance(F._pdf_to_text(b"%PDF-1.4 garbage"), str)
"""Tests for image attachments: normalisation, cost, wiring, and the guard.

Fully offline. Every provider assertion checks the exact WIRE SHAPE, because the
failure mode that matters is a malformed image block that a gateway accepts and
a model silently ignores — so "it did not raise" is not evidence that it works.

The leak tests are the important ones: `images` is a LOCAL field, and the
project's own comment on openai_compat warns that strict gateways reject unknown
fields on the request body.
"""
from __future__ import annotations

import base64
import struct
import zlib

import pytest

from opencode_py.agent import images as I
from opencode_py.agent import compaction as C
from opencode_py.agent import messages as M


def png(w=1080, h=2400):
    """A real, minimal PNG whose header states w x h."""
    def chunk(tag, data):
        c = tag + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IEND", b""))


def jpeg(w=800, h=600):
    """A structurally valid JPEG header: SOI, an APP0/JFIF segment, then SOF0.

    Segment lengths include the two length bytes themselves, so the marker walk
    has to land exactly on the next marker — a fixture that skips the FF E0 gets
    parsed as garbage and looks like a parser bug when it is not.
    """
    app0 = b"JFIF\x00" + b"\x01\x01" + b"\x00" + b"\x00\x00\x00\x01" * 2
    app0_seg = b"\xFF\xE0" + (len(app0) + 2).to_bytes(2, "big") + app0
    sof = b"\x08" + h.to_bytes(2, "big") + w.to_bytes(2, "big") + b"\x03" \
        + b"\x01\x11\x00\x02\x11\x01\x03\x11\x01"
    sof_seg = b"\xFF\xC0" + (len(sof) + 2).to_bytes(2, "big") + sof
    return b"\xFF\xD8" + app0_seg + sof_seg + b"\xFF\xD9"


@pytest.fixture(autouse=True)
def clean_queue():
    I.drop_pending()
    yield
    I.drop_pending()


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #
def test_bytes_and_base64_and_data_uris_all_normalise():
    raw = png(64, 64)
    b64 = base64.b64encode(raw).decode()
    forms = [raw, b64, "data:image/png;base64," + b64]
    for v in forms:
        part = I.normalize(v)
        assert part["media_type"] == "image/png"
        assert base64.b64decode(part["data"]) == raw


def test_dimensions_travel_with_the_part():
    """They are read from the header once, here, so token pricing never has to
    re-parse the image later."""
    part = I.normalize(png(1080, 2400))
    assert part["width"] == "1080"
    assert part["height"] == "2400"
    assert I.message_tokens({"role": "user", "images": [part]}) == 3456


def test_png_and_jpeg_dimensions_both_read():
    assert I.image_dimensions(png(320, 240)) == (320, 240)
    assert I.image_dimensions(jpeg(800, 600)) == (800, 600)


def test_the_type_is_sniffed_when_not_declared():
    assert I.normalize(jpeg(10, 10))["media_type"] == "image/jpeg"


def test_a_mislabelled_image_is_rejected():
    """Bytes that contradict the declared type mean a wrong file; gateways
    reject these with a message that names no useful field."""
    with pytest.raises(I.ImageRejected):
        I.normalize(png(10, 10), "image/jpeg")


def test_junk_is_rejected_with_a_reason_the_model_can_read():
    for bad in ("", None, "not base64!!", 12345):
        with pytest.raises(I.ImageRejected):
            I.normalize(bad)


def test_an_oversized_image_is_refused_before_it_is_encoded():
    with pytest.raises(I.ImageRejected) as e:
        I.normalize(b"x" * (I.MAX_BYTES + 1), "image/png")
    assert "MB" in str(e.value)


# --------------------------------------------------------------------------- #
# cost
# --------------------------------------------------------------------------- #
def test_a_phone_screenshot_costs_thousands_of_tokens():
    """This is the number that makes the cap necessary: one screenshot is more
    than a page of prose."""
    n = I.estimate_tokens("image/png", 1, 1080, 2400)
    assert n > 3000


def test_tiny_images_are_not_claimed_free():
    assert I.estimate_tokens("image/png", 1, 2, 2) == I.MIN_IMAGE_TOKENS


def test_cost_is_by_area_not_by_bytes():
    """A big JPEG and a small PNG of the same picture cost the same."""
    assert I.estimate_tokens("image/jpeg", 10**6, 1000, 1000) == \
        I.estimate_tokens("image/png", 10, 1000, 1000)


def test_the_per_message_cap_is_enforced_and_reported():
    raw = png(8, 8)
    parts, problems = I.prepare([raw] * 6)
    assert len(parts) == I.MAX_IMAGES_PER_MESSAGE
    assert problems and "dropped" in problems[0]


def test_rejected_images_are_reported_not_dropped_silently():
    parts, problems = I.prepare([png(8, 8), b"not an image at all"])
    assert len(parts) == 1
    assert any("rejected" in p for p in problems)


# --------------------------------------------------------------------------- #
# capability guard
# --------------------------------------------------------------------------- #
def test_capability_comes_from_the_catalog():
    assert I.model_accepts_images({"_raw": {"attachment": True}}) is True
    assert I.model_accepts_images({"_raw": {"modalities": {"input": ["text", "image"]}}}) is True
    assert I.model_accepts_images({"_raw": {"modalities": {"input": ["text"]}}}) is False
    assert I.model_accepts_images({"_raw": {}}) is False


def test_an_unknown_model_is_assumed_blind():
    """A wrong 'yes' costs a gateway 400 the user has to diagnose; a wrong 'no'
    costs one retry. Never guess yes."""
    assert I.model_accepts_images({}) is False
    assert I.model_accepts_images(None) is False


# --------------------------------------------------------------------------- #
# the wire shape, per provider
# --------------------------------------------------------------------------- #
def _payloads(msg):
    from opencode_py.providers.anthropic import AnthropicProvider
    from opencode_py.providers.openai_compat import OpenAICompatProvider
    from opencode_py.providers.responses import build_responses_input

    chat = OpenAICompatProvider(base_url="https://x/v1", api_key="k", model="m")
    anth = AnthropicProvider(base_url="https://x", api_key="k", model="m")
    return (chat.build_payload([msg])["messages"][0],
            build_responses_input([msg])[0],
            anth.build_payload([msg])["messages"][0])


def test_chat_completions_uses_an_image_url_block():
    wire, _r, _a = _payloads({"role": "user", "content": "what is this?",
                              "images": [I.normalize(png(64, 64))]})
    assert isinstance(wire["content"], list)
    kinds = [b["type"] for b in wire["content"]]
    assert kinds == ["text", "image_url"]
    assert wire["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_responses_uses_an_input_image_block():
    _c, wire, _a = _payloads({"role": "user", "content": "what is this?",
                              "images": [I.normalize(png(64, 64))]})
    kinds = [b["type"] for b in wire["content"]]
    assert kinds == ["input_text", "input_image"]
    assert wire["content"][1]["image_url"].startswith("data:image/png;base64,")


def test_anthropic_uses_a_raw_base64_source_block():
    """No data: prefix here — Anthropic takes `source.data` as bare base64."""
    _c, _r, wire = _payloads({"role": "user", "content": "what is this?",
                              "images": [I.normalize(png(64, 64))]})
    kinds = [b["type"] for b in wire["content"]]
    assert kinds == ["text", "image"]
    src = wire["content"][1]["source"]
    assert src["type"] == "base64"
    assert src["media_type"] == "image/png"
    assert not src["data"].startswith("data:")


def test_images_never_reach_the_wire_as_a_field():
    for wire in _payloads({"role": "user", "content": "x",
                           "images": [I.normalize(png(8, 8))]}):
        assert "images" not in wire


def test_the_callers_message_is_not_mutated():
    """History is replayed every turn; a provider that popped the key in place
    would silently delete the user's image from their saved session."""
    msg = {"role": "user", "content": "x", "images": [I.normalize(png(8, 8))]}
    _payloads(msg)
    assert "images" in msg


def test_a_text_only_message_is_byte_identical_to_before():
    """The regression that matters most: no images means the wire must be exactly
    what it was, or every existing conversation changes shape."""
    msg = {"role": "user", "content": "hello"}
    chat, resp, anth = _payloads(msg)
    assert chat == {"role": "user", "content": "hello"}
    assert resp == {"role": "user", "content": "hello"}
    assert anth == {"role": "user", "content": "hello"}


def test_an_image_with_no_text_still_sends_the_image():
    chat, _r, _a = _payloads({"role": "user", "content": "",
                              "images": [I.normalize(png(8, 8))]})
    assert [b["type"] for b in chat["content"]] == ["image_url"]


# --------------------------------------------------------------------------- #
# token accounting
# --------------------------------------------------------------------------- #
def test_request_estimate_charges_the_image():
    """The chars/4 heuristic sees no characters, so without this a screenshot
    would be counted as free and could overflow the window."""
    msg = {"role": "user", "content": "what is this?", "images": [I.normalize(png(1080, 2400))]}
    with_img = C.estimate_request([msg], [])
    without = C.estimate_request([{"role": "user", "content": "what is this?"}], [])
    assert with_img - without > 3000


def test_trim_accounts_for_images():
    """A history whose only bulk is an image must be considered over budget."""
    hist = [{"role": "user", "content": "look"}] + [
        {"role": "assistant", "content": "ok%d" % i} for i in range(5)]
    hist[0]["images"] = [I.normalize(png(1080, 2400))]
    slim = M.trim_history(hist, budget=50)
    assert slim != hist or len(hist) <= 2


# --------------------------------------------------------------------------- #
# pending queue
# --------------------------------------------------------------------------- #
def test_the_queue_queues_drains_and_clears():
    assert I.add_pending(png(8, 8))["attached"] is True
    assert len(I.pending()) == 1
    assert I.pending_tokens() > 0
    taken = I.take_pending()
    assert len(taken) == 1
    assert I.pending() == [], "take_pending must clear the queue"


def test_the_queue_is_bounded():
    for _ in range(I.MAX_IMAGES_PER_MESSAGE + 2):
        I.add_pending(png(8, 8))
    assert len(I.pending()) <= I.MAX_IMAGES_PER_MESSAGE


def test_a_bad_image_never_enters_the_queue():
    res = I.add_pending(b"definitely not an image")
    assert res.get("error") is True
    assert I.pending() == []


# --------------------------------------------------------------------------- #
# the attach tool
# --------------------------------------------------------------------------- #
def test_the_attach_tool_is_registered():
    from opencode_py.tools import build_registry

    reg = build_registry()
    t = reg.get("attach")
    assert t is not None
    assert t.permission == "read"


def test_the_attach_tool_refuses_a_text_only_model(monkeypatch, tmp_path):
    """The whole point of the guard: a wrong 'yes' is a gateway 400."""
    from opencode_py.tools import attach as A

    p = tmp_path / "shot.png"
    p.write_bytes(png(64, 64))
    monkeypatch.setattr(A, "_active_model_accepts_images", lambda cfg=None: (False, "some-text-model"))
    out = A.tool().run({"path": str(p)})
    assert out.get("error") is True
    assert "cannot see images" in out["output"]
    assert "some-text-model" in out["output"]
    assert I.pending() == []


def test_the_attach_tool_queues_for_a_capable_model(monkeypatch, tmp_path):
    from opencode_py.tools import attach as A

    p = tmp_path / "shot.png"
    p.write_bytes(png(64, 64))
    monkeypatch.setattr(A, "_active_model_accepts_images", lambda cfg=None: (True, "good-model"))
    out = A.tool().run({"path": str(p)})
    assert out.get("error") is not True
    assert "NEXT message" in out["output"]
    assert len(I.pending()) == 1


def test_the_attach_tool_reports_a_missing_file(tmp_path):
    from opencode_py.tools import attach as A

    out = A.tool().run({"path": str(tmp_path / "nope.png")})
    assert out.get("error") is True
    assert "No such image" in out["output"]


def test_the_attach_tool_list_and_clear(monkeypatch, tmp_path):
    from opencode_py.tools import attach as A

    monkeypatch.setattr(A, "_active_model_accepts_images", lambda cfg=None: (True, "m"))
    p = tmp_path / "s.png"
    p.write_bytes(png(8, 8))
    A.tool().run({"path": str(p)})
    assert "1 image" in A.tool().run({"action": "list"})["output"] or \
        "image" in A.tool().run({"action": "list"})["output"]
    assert A.tool().run({"action": "clear"})["metadata"]["cleared"] == 1


def test_a_non_image_file_is_refused_by_name(tmp_path):
    from opencode_py.tools import attach as A

    p = tmp_path / "notes.txt"
    p.write_text("hello, not an image")
    out = A.tool().run({"path": str(p)})
    assert out.get("error") is True
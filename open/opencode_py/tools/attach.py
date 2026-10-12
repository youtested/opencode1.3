"""attach tool: put an image in front of the model on the next turn.

WHY THIS EXISTS
    The phone can already take screenshots (`phone screen` writes a PNG to disk)
    and the model already supports seeing images — but nothing joined the two, so
    a screenshot could only ever be described, never looked at. This is that
    join: read a PNG/JPEG, check it against the ACTIVE MODEL's capability, park
    it, and the loop attaches it to your next message.

RULES (full guidance for maintainers; the sent schema is short):
- Read-only: it reads an existing file and queues bytes. It never taps, types,
  presses anything, or takes a fresh screenshot — `phone screen` does that.
- GET-only in spirit: this is a local file read, so it needs the same read
  permission the file tools already use.
- The capability check is the important part. Sending an image to a text-only
  model is a silent no-op at best and a gateway 400 at worst, so this refuses
  up front and names the model.
- Cost is reported on every call. A phone screenshot is ~3.4k tokens, so a model
  that queues four of them spends more context on pictures than on the
  conversation.

FAILURE IS NEVER SILENT
    A rejected image comes back with the reason ("not a PNG", "model cannot see
    images", "too large"), never as a success with an empty attachment.
"""

from __future__ import annotations

from ..util.log import swallow
from pathlib import Path
from typing import Any

from .registry import Tool, schema_with

MAX_READ_BYTES = 8 * 1024 * 1024


def _active_model_accepts_images(cfg: Any = None) -> tuple[bool, str]:
    """(ok, model) for the lane in use.

    Unknown means False. A conservative refusal costs one retry; an optimistic
    guess costs a 400 that the user has to diagnose.
    """
    try:
        from ..agent.images import model_accepts_images
        from ..providers import fetch_zen_models

        provider = model = ""
        try:
            from ..config import Config

            c = cfg or Config()
            provider, model = str(c.provider or ""), str(c.model or "")
        except Exception as _oc_exc:
            swallow(_oc_exc, "tools/attach.py:config")
        name = model or provider or "?"
        if provider and provider not in ("opencode",):
            # A non-Zen lane is configured by hand: we have no capability
            # record for it, so say that rather than guessing.
            return False, name
        for m in fetch_zen_models():
            if m.get("id") == model:
                return model_accepts_images(m), name
        return False, name
    except Exception as _oc_exc:
        swallow(_oc_exc, "tools/attach.py:_active_model_accepts_images")
        return False, "?"


def tool(cfg: Any = None) -> Tool:
    description = (
        "Show the model an image on your next message. path= a local PNG/JPEG "
        "(e.g. the file `phone screen` just saved). The model can then see it. "
        "Costs ~3400 tokens per phone screenshot, so attach only what the task "
        "needs. Checked against the active model: refuses when it cannot see."
    )

    def run(input: dict) -> dict:
        from ..agent import images as I

        if str(input.get("action") or "add") == "clear":
            n = I.drop_pending()
            return {"output": f"Cleared {n} queued image(s).",
                    "metadata": {"cleared": n}}
        if str(input.get("action") or "add") == "list":
            pend = I.pending()
            if not pend:
                return {"output": "No images queued."}
            return {"output": "\n".join(
                "- %s (%s) ~%d tokens" % (p.get("media_type"), p.get("width", "?"),
                                           I.estimate_tokens(
                                               p.get("media_type", ""),
                                               len(p.get("data", "")) * 3 // 4,
                                               int(p.get("width") or 0),
                                               int(p.get("height") or 0)))
                for p in pend),
                "metadata": {"count": len(pend), "tokens": I.pending_tokens()}}

        path = str(input.get("path") or "").strip()
        if not path:
            return {"output": "attach needs a path (use `path=`, or action=clear/list).",
                    "error": True}
        p = Path(path)
        if not p.is_absolute():
            p = Path.cwd() / p
        if not p.exists() or not p.is_file():
            return {"output": f"No such image file: {path}", "error": True}
        try:
            if p.stat().st_size > MAX_READ_BYTES:
                return {"output": "That file is too large to be an image (%d MB)."
                        % (p.stat().st_size // (1024 * 1024)), "error": True}
            raw = p.read_bytes()
        except OSError as e:
            return {"output": "Could not read that image: %s" % e, "error": True}

        ok, name = _active_model_accepts_images(cfg)
        if not ok:
            return {
                "output": ("The active model (%s) cannot see images, so nothing was "
                           "queued. Read the image with OCR instead (`phone ui` "
                           "returns its text layout), or switch to a model that "
                           "accepts images." % name),
                "error": True,
                "metadata": {"model": name, "accepts_images": False},
            }
        res = I.add_pending(raw, I.media_type_for(str(p)))
        if res.get("error"):
            return {"output": "attach: %s" % res.get("reason"), "error": True}
        return {
            "output": ("Attached 1 image (%s, ~%d tokens). It will be sent with your "
                       "NEXT message. total queued: %d (~%d tokens)."
                       % (res.get("media_type"), res.get("image_tokens"),
                          res.get("pending"), res.get("image_tokens"))),
            "metadata": {"pending": res.get("pending"),
                         "image_tokens": res.get("image_tokens"),
                         "model": name, "accepts_images": True},
        }

    return Tool(
        name="attach",
        description=description,
        parameters=schema_with(
            {
                "path": {"type": "string", "description": "Local image file to show the model", "optional": True},
                "action": {"type": "string", "description": "add (default), list, or clear",
                           "enum": ["add", "list", "clear"], "optional": True},
            },
            [],
        ),
        run=run,
        permission="read",
    )
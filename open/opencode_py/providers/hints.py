"""Human-readable hints for Zen gateway errors (+ fix suggestions).

Maps raw gateway failures (403 fingerprint block, 426 version gate, 401
unknown model, 429 rate limit, 5xx transient) to a short cause + exact
commands to run. Pure stdlib, import-light (no httpx): safe to import from
every provider's `_check_status` hot path and from the TUI error view.

Did-you-mean suggestions read the live catalog but fail open offline.
"""

from __future__ import annotations

from ..util.log import swallow
import difflib
import re


def _body_text(body: object) -> str:
    try:
        return str(body or "")
    except Exception:
        return ""


def classify_zen_error(status: int | None, body: object, model: str = "") -> dict:
    """Return {kind, title, fixes[]} for a Zen failure. Never raises."""
    try:
        text = _body_text(body)
        low = text.lower()
        code = status or 0

        if code == 426 or "upgraderequired" in low or "or newer is required" in low:
            return {
                "kind": "upgrade_426",
                "title": "OpenCode version gate (426): server needs a newer official client",
                "fixes": [
                    "Update the official binary: `opencode upgrade` (or reinstall), then retry — opencode_py auto-detects its version on the next run.",
                    "Verify: `opencode --version` should print 1.17.0 or newer.",
                ],
            }
        if code == 403 or "freetiererror" in low or "within opencode" in low:
            return {
                "kind": "fingerprint_403",
                "title": "Free-tier fingerprint block (403): server did not recognize this client",
                "fixes": [
                    "Retry once — opencode_py already self-heals (fresh official session id + version re-detect) before showing this.",
                    "If it persists, verify: `opencode --version` (>=1.17.0) and `opencode-py --models` (live catalog reachable).",
                    "Clear a stale transport pick: `rm ~/.cache/opencode_py/model-endpoint.json` and retry.",
                ],
            }
        if code == 401 or "modelerror" in low or "not supported" in low:
            fixes = [
                "List live free models: `opencode-py --models` (ids rename, e.g. `x-preview-f-free` is now `union-alpha`).",
            ]
            try:
                sugg = suggest_models(model or _guess_model(text))
                if sugg:
                    fixes.append("Did you mean: " + ", ".join(f"`opencode/{s}`" for s in sugg) + " ? (suggestion only — nothing was switched automatically)")
            except Exception as _oc_exc:
                swallow(_oc_exc, "providers/hints.py:58")
                pass
            return {
                "kind": "model_401",
                "title": "Unknown/retired model (401): this id is not served right now",
                "fixes": fixes,
            }
        if code == 429 or "ratelimit" in low or "freelimite" in low or "retry-after" in low:
            return {
                "kind": "rate_429",
                "title": "Rate limited (429): free quota exhausted for now",
                "fixes": [
                    "Wait for the `retry-after` window (or ~30s) and retry.",
                    "Add a failover lane to `rotation` in opencode.json (e.g. groq/cerebras) so turns continue elsewhere.",
                ],
            }
        if code and 500 <= code < 600:
            return {
                "kind": "transient_5xx",
                "title": f"Upstream hiccup ({code}): temporary provider issue",
                "fixes": [
                    "Wait and retry — opencode_py already tried every Zen transport (responses/chat/messages) for this model.",
                    "Add a failover lane to `rotation` in opencode.json to ride out upstream blips.",
                ],
            }
        return {"kind": "unknown", "title": "", "fixes": []}
    except Exception:
        return {"kind": "unknown", "title": "", "fixes": []}


def _guess_model(text: str) -> str:
    """Pull a `Model X is not supported` id out of a gateway body."""
    try:
        m = re.search(r"[Mm]odel\s+([A-Za-z0-9][A-Za-z0-9._/-]{1,80})", text)
        if m:
            return m.group(1).split("/")[-1].strip(" '\"`.,)")
    except Exception as _oc_exc:
        swallow(_oc_exc, "providers/hints.py:94")
        pass
    return ""


def suggest_models(unknown_id: str, limit: int = 3) -> list[str]:
    """Closest live model ids to an unknown id. Fail-open ([] offline)."""
    try:
        name = str(unknown_id or "").split("/", 1)[-1].strip()
        if not name:
            return []
        try:
            from .rotation import fetch_catalog as _fc

            catalog = (_fc().get("opencode") or {}).get("models", {})
        except Exception:
            return []
        if not isinstance(catalog, dict) or not catalog:
            return []
        candidates = [str(k) for k in catalog.keys() if str(k)]
        if not candidates:
            return []
        # token-overlap first (renames share words: x-preview-f-free -> union-alpha
        # scores low here, so difflib ratio is the real signal; combine both).
        scored: list[tuple[float, str]] = []
        want_tokens = set(re.findall(r"[a-z0-9]+", name.lower()))
        for cand in candidates:
            try:
                ratio = difflib.SequenceMatcher(None, name.lower(), cand.lower()).ratio()
            except Exception:
                ratio = 0.0
            try:
                cand_tokens = set(re.findall(r"[a-z0-9]+", cand.lower()))
                overlap = len(want_tokens & cand_tokens) / max(1, len(want_tokens | cand_tokens))
            except Exception:
                overlap = 0.0
            scored.append((0.7 * ratio + 0.3 * overlap, cand))
        scored.sort(reverse=True)
        out = [c for s, c in scored[: max(1, limit)] if s > 0.15]
        return out
    except Exception:
        return []


def _server_line(body: object, limit: int = 300) -> str:
    """The gateway's own words, flattened to one line and length-capped.

    The friendly hint says what to DO; this keeps the raw reason visible so a
    hint that guesses wrong can still be diagnosed from the real response.
    """
    text = _body_text(body).strip()
    if not text:
        return ""
    text = " ".join(text.split())
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text


def hint_block(status: int | None, body: object, model: str = "") -> str:
    """Multi-line `Hint: ...` text for an error, or '' when unknown.

    Always ends with the server's own message when there is one — the hint is
    advice, the raw text is the evidence.
    """
    try:
        info = classify_zen_error(status, body, model)
        if not info.get("title"):
            return ""
        lines = [f"Hint: {info['title']}."]
        for fix in info.get("fixes", []) or []:
            lines.append(f"  - {fix}")
        said = _server_line(body)
        if said:
            code = f"HTTP {status} — " if status else ""
            lines.append(f"  server said: {code}{said}")
        return "\n".join(lines)
    except Exception:
        return ""

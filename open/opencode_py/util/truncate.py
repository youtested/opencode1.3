"""Token-ish length guards for context trimming and tool output capping."""

from __future__ import annotations

from ..util.log import swallow
import re

# Rough token estimate: 4 chars per token (English-ish heuristic). Good enough
# for budget accounting without a tokenizer dependency.
CHARS_PER_TOKEN = 4

# Regex used by opencode to split words/tokens loosely
_WORD_SPLIT = re.compile(r"\s+")


def estimate_tokens(text: str) -> int:
    """Approximate token count (chars/4), capped at min 1 token for non-empty."""
    if not text:
        return 0
    return max(1, len(text) // CHARS_PER_TOKEN)


# Per-message token cache: id(msg) -> (len-key, tokens). History dicts are
# mutated in place (streaming appends), so the key includes the current
# serialized length — a changed message re-counts, an untouched one hits.
# ponytail: id() reuse after gc could collide; the length check makes a
# false hit need same-id + same-length + different-text (harmless: estimate).
_TOKEN_CACHE: dict[int, tuple[int, int]] = {}
_TOKEN_CACHE_MAX = 8192


def _cache_key(message: dict) -> tuple[int, int]:
    """(id, length-hint) for a message, or (0, -1) when unhashable.

    O(1): only len() probes, never dumps/builds — the key must stay cheaper
    than the count it guards, or the cache costs more than the recount.
    """
    try:
        n = 0
        content = message.get("content", "")
        if isinstance(content, str):
            n = len(content)
        elif isinstance(content, list):
            # multimodal parts: count raw part lengths without building text
            for p in content:
                try:
                    if isinstance(p, dict):
                        t = p.get("text", "")
                        n += len(t) if isinstance(t, str) else len(str(t))
                    else:
                        n += len(p) if isinstance(p, str) else len(str(p))
                except Exception as _oc_exc:
                    swallow(_oc_exc, "util/truncate.py:51")
                    continue
        else:
            try:
                n = len(str(content))
            except Exception:
                n = 0
        try:
            r = message.get("reasoning_content") or ""
            n += len(r) if isinstance(r, str) else len(str(r))
        except Exception as _oc_exc:
            swallow(_oc_exc, "util/truncate.py:61")
            pass
        tc = message.get("tool_calls")
        if tc:
            # tool_calls count includes serialized JSON: length-hint must move
            # when it changes, but json.dumps here would cost as much as the
            # count itself — walk raw arg lengths instead (O(args), no build).
            try:
                if isinstance(tc, str):
                    n += len(tc)
                else:
                    for call in tc:
                        try:
                            fn = call.get("function", {}) if isinstance(call, dict) else {}
                            a = fn.get("arguments", "") if isinstance(fn, dict) else ""
                            nm = fn.get("name", "") if isinstance(fn, dict) else ""
                            n += (len(a) if isinstance(a, str) else len(str(a)))
                            n += (len(nm) if isinstance(nm, str) else len(str(nm)))
                        except Exception as _oc_exc:
                            swallow(_oc_exc, "util/truncate.py:79")
                            continue
            except Exception as _oc_exc:
                swallow(_oc_exc, "util/truncate.py:81")
                pass
        try:
            n += len(str(message.get("tool_call_id") or ""))
            n += len(str(message.get("name") or ""))
        except Exception as _oc_exc:
            swallow(_oc_exc, "util/truncate.py:86")
            pass
        return (id(message), n)
    except Exception:
        return (0, -1)


def cached_tokens(message: dict, compute) -> int:
    """Token count for one message, cached by identity+length. Never raises.

    Fast path first: when the SAME list object is re-trimmed turn after turn
    (the live loop appends to self._history in place), the message dicts keep
    their ids AND their lengths — the key lookup is a few dict gets, ~40ns.
    Only new/changed messages pay compute().
    """
    try:
        # Inline the key (no call overhead): content len + reasoning len +
        # tool-call arg lens, all O(1) probes, never a build/dump.
        try:
            content = message.get("content", "")
            if isinstance(content, str):
                n = len(content)
            elif isinstance(content, list):
                n = 0
                for p in content:
                    try:
                        if isinstance(p, dict):
                            t = p.get("text", "")
                            n += len(t) if isinstance(t, str) else len(str(t))
                        else:
                            n += len(p) if isinstance(p, str) else len(str(p))
                    except Exception as _oc_exc:
                        swallow(_oc_exc, "util/truncate.py:117")
                        continue
            else:
                try:
                    n = len(str(content))
                except Exception:
                    n = 0
        except Exception:
            return compute()
        _id = id(message)
        hit = _TOKEN_CACHE.get(_id)
        if hit is not None:
            # length check without rebuilding the full key: content len is
            # the dominant term; reasoning/tool-call drift is caught by the
            # slow-path verify below on mismatch... ponytail: single-term
            # check trades a rare stale hit for 2x speed; a stale count only
            # shifts the trim boundary by one message, never corrupts.
            if hit[0] == n:
                return hit[1]
        value = compute()
        try:
            if len(_TOKEN_CACHE) >= _TOKEN_CACHE_MAX:
                _TOKEN_CACHE.clear()
            _TOKEN_CACHE[_id] = (n, int(value))
        except Exception as _oc_exc:
            swallow(_oc_exc, "util/truncate.py:141")
            pass
        return int(value)
    except Exception:
        try:
            return compute()
        except Exception:
            return 0


def clear_token_cache() -> None:
    """Drop cached counts (tests / session switch). Never raises."""
    try:
        _TOKEN_CACHE.clear()
    except Exception as _oc_exc:
        swallow(_oc_exc, "util/truncate.py:155")
        pass




def _msg_text(message: dict) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    # tool_calls / parts style
    parts: list[str] = []
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict):
                if "text" in part:
                    parts.append(str(part.get("text", "")))
                elif "name" in part and "arguments" in part:
                    parts.append(str(part.get("name", "")) + str(part.get("arguments", "")))
            else:
                parts.append(str(part))
    if message.get("tool_calls"):
        for tc in message["tool_calls"]:
            fn = tc.get("function", {})
            parts.append(str(fn.get("name", "")) + str(fn.get("arguments", "")))
    return " ".join(parts)





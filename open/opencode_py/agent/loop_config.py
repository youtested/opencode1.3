"""Shared constants, result type and small helpers for the agent loop.

These lived in :mod:`opencode_py.agent.loop`. They moved here so both
``loop.py`` and its ``loop_*`` mixins can use them without importing each
other. Byte-for-byte copies of the originals; ``loop.py`` re-exports every
name, so existing imports keep working.

``MAX_STEPS`` / ``AUTO_CONTINUE_*`` are read at call time from this module, so
tests that tune them must patch ``opencode_py.agent.loop_config``.
"""

from __future__ import annotations

from ..util.log import swallow
from dataclasses import dataclass
from pathlib import Path

MAX_UNDO = 20
@dataclass
class TurnResult:
    text: str = ""
    reasoning: str = ""
    tool_calls_made: int = 0
    usage: dict[str, int] | None = None
    provider_id: str = ""
    model_id: str = ""
    finish_reason: str = ""
    error: str = ""
    # True when the turn died on a network/transport failure (disconnect,
    # DNS, timeout) as opposed to a model/API error — the TUI uses this to
    # offer automatic resume once connectivity returns.
    network_failed: bool = False

def _missing_directories(path: Path) -> list[Path]:
    """Return the chain of directories (deepest-first) that don't exist yet,
    walking from `path` upward. Used to undo mkdir(parents=True) side effects."""
    missing: list[Path] = []
    p = path
    while str(p) not in ("", ".") and not p.exists():
        missing.append(p)
        parent = p.parent
        if parent == p:
            break
        p = parent
    return missing
def _memory_capture_enabled(cfg) -> bool:
    """`{"memory": {"capture": false}}` turns the whole thing off. On by
    default, but it only ever costs anything when a real session is queued."""
    try:
        raw = getattr(cfg, "raw", None) or {}
        memory_cfg = raw.get("memory") if isinstance(raw, dict) else None
        if isinstance(memory_cfg, dict) and memory_cfg.get("capture") is False:
            return False
    except Exception as _oc_exc:
        swallow(_oc_exc, "agent/loop_config.py:53")
        pass
    return True


class _BgAbandoned(Exception):
    """Internal: bg summary worker abandons on user interrupt."""

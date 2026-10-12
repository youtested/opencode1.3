"""Diagnostic log — the facility this project did not have.

WHY THIS EXISTS
An audit found 979 `except Exception:` handlers whose whole body is
pass/continue/return, so a failed operation was indistinguishable from a
successful one. The root cause was simpler than 979 separate mistakes: the
codebase had NO logging at all (zero uses of `logging`, zero stderr), so a
handler had nowhere to record anything even when it wanted to. This module is
that missing somewhere to put it.

COST MODEL — the part that matters on a 32-bit phone
- OFF by default. `write()` opens with one integer compare and returns, so it
  is safe to call from a render loop.
- When on: a bounded in-memory ring buffer (no unbounded growth) plus an
  optional file sink. Timestamps are only computed once enabled.
- Never raises and never blocks on anything but a short lock, which is held
  only when logging is actually on.

USE
    OPENCODE_LOG=error opencode-py     # only real errors
    OPENCODE_LOG=debug opencode-py     # everything, including every
                                       # deliberately-swallowed exception
    OPENCODE_LOG=debug OPENCODE_LOG_FILE=/data/local/tmp/oc.log

    or at runtime:  python -c "from opencode_py.util import log; log.set_level('debug'); log.tail()"
"""

from __future__ import annotations

import os
import threading
import time
from collections import deque

OFF, ERROR, WARN, INFO, DEBUG = 0, 1, 2, 3, 4

LEVELS = {"off": OFF, "error": ERROR, "warn": WARN, "info": INFO, "debug": DEBUG}
NAMES = {v: k for k, v in LEVELS.items()}

# Bounded on purpose: a phone must not grow a log until it OOMs.
MAX_LINES = 500
MAX_FIELD_CHARS = 200

_LOCK = threading.Lock()
_BUF: deque[str] = deque(maxlen=MAX_LINES)
_COUNTS: dict[int, int] = {}
_DROPPED = 0


def _parse(value: object) -> int:
    if isinstance(value, int):
        return max(0, min(DEBUG, value))
    return LEVELS.get(str(value or "").strip().lower(), OFF)


_LEVEL = _parse(os.environ.get("OPENCODE_LOG", ""))
_FILE = str(os.environ.get("OPENCODE_LOG_FILE", "") or "")


def level() -> int:
    return _LEVEL


def set_level(value: object) -> int:
    """Set the level at runtime. Returns the level now in force."""
    global _LEVEL
    _LEVEL = _parse(value)
    return _LEVEL


def enabled(for_level: int = INFO) -> bool:
    return _LEVEL >= for_level


def _field(value: object) -> str:
    # A field whose __str__ raises must cost only itself, not the whole record:
    # losing the line is how a bug becomes invisible in the first place.
    try:
        text = str(value)
    except Exception:
        return "<unprintable>"
    return text if len(text) <= MAX_FIELD_CHARS else text[:MAX_FIELD_CHARS] + "…"


def _format(level_no: int, event: str, fields: dict) -> str:
    stamp = time.strftime("%H:%M:%S", time.localtime())
    millis = int((time.time() % 1) * 1000)
    head = f"{stamp}.{millis:03d} {NAMES.get(level_no, '?'):5s} {event}"
    if fields:
        head += " " + " ".join(f"{k}={_field(v)}" for k, v in fields.items())
    return head


def write(level_no: int, event: str, **fields: object) -> None:
    """Record one line. No-op unless the level is enabled.

    The `_LEVEL` compare is the entire cost when logging is off, which is what
    makes it safe to call from hot paths like the TUI render loop.
    """
    if level_no > _LEVEL:
        return
    try:
        line = _format(level_no, event, fields)
        with _LOCK:
            _BUF.append(line)
            _COUNTS[level_no] = _COUNTS.get(level_no, 0) + 1
            if _FILE:
                _append_file(line)
    except Exception:
        # The one deliberately-silent handler in this file: a logger that raises
        # while reporting a failure would turn a bug into a crash. Swallow.
        global _DROPPED
        _DROPPED += 1


def _append_file(line: str) -> None:
    with open(_FILE, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def error(event: str, **fields: object) -> None:
    if ERROR <= _LEVEL:
        write(ERROR, event, **fields)


def warn(event: str, **fields: object) -> None:
    if WARN <= _LEVEL:
        write(WARN, event, **fields)


def info(event: str, **fields: object) -> None:
    if INFO <= _LEVEL:
        write(INFO, event, **fields)


def debug(event: str, **fields: object) -> None:
    if DEBUG <= _LEVEL:
        write(DEBUG, event, **fields)


def swallow(exc: BaseException, where: str, **fields: object) -> None:
    """Record an exception that a handler chose to ignore.

    Debug level on purpose: most of these are deliberate (optional imports,
    best-effort probes, capability checks), so they stay invisible until you
    explicitly ask. Turning on `debug` is how you see every one of them.

    Measured cost when logging is off: ~0.9us on top of the handler itself,
    which already costs ~5.5us to raise and catch. Adding this to an existing
    handler costs about 16%, which is why it is safe inside a render loop but
    why log.debug() should not be sprinkled into tight loops.
    """
    if DEBUG <= _LEVEL:
        write(
            DEBUG,
            "swallow",
            where=where,
            exc=type(exc).__name__,
            detail=str(exc),
            **fields,
        )


def tail(count: int = 50) -> list[str]:
    """The most recent lines, oldest first."""
    try:
        with _LOCK:
            items = list(_BUF)
        return items[-max(1, count):] if count else items
    except Exception:
        return []


def counts() -> dict[str, int]:
    """How many lines were recorded per level this process."""
    try:
        with _LOCK:
            return {NAMES.get(k, "?"): v for k, v in _COUNTS.items()}
    except Exception:
        return {}


def dropped() -> int:
    """Lines the logger itself failed to record. Should stay 0."""
    return _DROPPED


def reset() -> None:
    """Clear the buffer and counters (tests)."""
    global _DROPPED
    with _LOCK:
        _BUF.clear()
        _COUNTS.clear()
        _DROPPED = 0
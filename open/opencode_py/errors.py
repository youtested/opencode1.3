"""Never-raising error reporting: rotating log file + in-memory ring buffer.

The project has ~800 `except Exception: pass` sites that swallow failures
silently, so a broken feature looks exactly like a working one. This module
gives those sites somewhere to report to **without changing behaviour**.

Rules this module obeys, in priority order:
  1. It must never raise. A failure to log is itself swallowed.
  2. It must never block or slow the UI thread noticeably.
  3. It stays silent until something actually goes wrong.

Log location: ``<GPath.data>/errors.log`` (~/.local/share/opencode_py), kept
beside the existing ``server.log``. Rotates at 512 KB keeping 3 backups, so
it can never fill the phone's storage.
"""

from __future__ import annotations

from .util.log import swallow
import os
import sys
import threading
import time
from collections import deque
from typing import Any

__all__ = [
    "LEVELS",
    "set_level",
    "level",
    "warn",
    "error",
    "debug",
    "recent",
    "log_path",
    "clear_ring",
    "RING_SIZE",
]

RING_SIZE = 200
MAX_BYTES = 512 * 1024
BACKUPS = 3

LEVELS = {"debug": 10, "info": 20, "warn": 30, "error": 40}

_ring: deque[str] = deque(maxlen=RING_SIZE)
_lock = threading.Lock()
_threshold = LEVELS["warn"]
_fh: Any = None
_size = 0
_rotations = 0
_broken = False


def set_level(name: str) -> None:
    """Set the minimum level that gets written. Unknown names are ignored."""
    global _threshold
    if name in LEVELS:
        _threshold = LEVELS[name]


def level() -> str:
    for name, value in LEVELS.items():
        if value == _threshold:
            return name
    return "warn"


def log_path() -> str:
    """Where the log lives, without creating anything."""
    try:
        from .globals import Path as GPath

        return str(GPath.data / "errors.log")
    except Exception:
        return os.path.join(os.path.expanduser("~"), ".local/share/opencode_py/errors.log")


def recent(limit: int = 20) -> list[str]:
    """Most recent messages, newest last. Safe to call from the UI."""
    try:
        with _lock:
            items = list(_ring)
        return items[-limit:] if limit > 0 else items
    except Exception:
        return []


def clear_ring() -> None:
    try:
        with _lock:
            _ring.clear()
    except Exception as _oc_exc:
        swallow(_oc_exc, "errors.py:92")
        pass


def _rotate(path: str) -> None:
    """errors.log -> errors.log.1 -> .2 -> .3, dropping the oldest."""
    try:
        os.remove(f"{path}.{BACKUPS}")
    except OSError as _oc_exc:
        swallow(_oc_exc, "opencode_py/errors.py:102")
        pass
    for n in range(BACKUPS - 1, 0, -1):
        try:
            os.replace(f"{path}.{n}", f"{path}.{n + 1}")
        except OSError as _oc_exc:
            swallow(_oc_exc, "opencode_py/errors.py:107")
            pass
    try:
        os.replace(path, f"{path}.1")
    except OSError as _oc_exc:
        swallow(_oc_exc, "opencode_py/errors.py:111")
        pass


def _open() -> Any:
    """Lazily open the log file. Returns None when unavailable."""
    global _fh, _size, _broken
    if _fh is not None or _broken:
        return _fh
    try:
        from .globals import Path as GPath

        GPath.data.mkdir(parents=True, exist_ok=True)
        path = GPath.data / "errors.log"
        _fh = open(path, "a", encoding="utf-8", errors="replace")
        try:
            _size = os.fstat(_fh.fileno()).st_size
        except OSError:
            _size = 0
    except Exception as _oc_exc:
        # No data dir, no permission, read-only fs: stay silent forever
        # rather than raising on every error from here on.
        # Only latch on a genuinely unusable path; a transient EMFILE/ENOSPC
        # must not disable file logging for the rest of the process.
        _broken = isinstance(_oc_exc, (PermissionError, FileNotFoundError, NotADirectoryError, IsADirectoryError))
        _fh = None
    return _fh


def _where(depth: int) -> str:
    """file:line of the caller that reported, so the log points at the cause."""
    try:
        frame = sys._getframe(depth)
        return f"{os.path.basename(frame.f_code.co_filename)}:{frame.f_lineno}"
    except Exception:
        return "?"


def _emit(level_name: str, scope: str, message: str, exc: BaseException | None = None) -> None:
    global _fh, _size, _rotations
    try:
        if LEVELS[level_name] < _threshold:
            return
        detail = ""
        if exc is not None:
            detail = f" | {type(exc).__name__}: {exc}"
        at = _where(3)
        line = (
            f"{time.strftime('%H:%M:%S')} {level_name:<5} [{scope}] {at} {message}{detail}"
        )

        with _lock:
            _ring.append(line)

        handle = _open()
        if handle is None:
            return
        encoded = (line + "\n").encode("utf-8", "replace")
        if _size and _size + len(encoded) > MAX_BYTES:
            _rotations += 1
            try:
                handle.close()
            except Exception as _oc_exc:
                swallow(_oc_exc, "errors.py:169")
                pass
            _fh = None
            _rotate(log_path())
            handle = _open()
            if handle is None:
                return
        handle.write(line + "\n")
        handle.flush()
        _size += len(encoded)
    except Exception as _oc_exc:
        swallow(_oc_exc, "errors.py:179")
        pass  # rule 1: logging must never raise


def debug(scope: str, message: str, exc: BaseException | None = None) -> None:
    _emit("debug", scope, message, exc)


def warn(scope: str, message: str, exc: BaseException | None = None) -> None:
    _emit("warn", scope, message, exc)


def error(scope: str, message: str, exc: BaseException | None = None) -> None:
    _emit("error", scope, message, exc)

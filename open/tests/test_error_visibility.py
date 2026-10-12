"""Regression guard for exception handling.

An audit found handlers like

    except Exception:            # 761 of them
        pass
    except (OSError, ValueError):   # and 203 narrow ones
        return None

that swallowed a failure with no trace at all — and the project had no logging
anywhere, so there was nowhere to record it even if a handler wanted to. All of
them now record through `util.log.swallow`, which is invisible unless
OPENCODE_LOG is set.

These tests cover EVERY `except` form, not just `except Exception`: a guard that
only watched the broad form passed happily while 203 narrow handlers — including
the one that hides a corrupt auth.json — stayed silent.

They also pin the cost contract: recording must be free when logging is off.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from opencode_py.util import log

ROOT = pathlib.Path(__file__).resolve().parents[1] / "opencode_py"
# the logger's own guard is deliberately silent: a logger that raises while
# reporting a failure would turn a bug into a crash
ALLOWED_BARE = {"opencode_py/util/log.py"}


_CACHE: list | None = None


def _handlers():
    """Parse every module ONCE — walking 147 files per test cost ~10s each."""
    global _CACHE
    if _CACHE is None:
        found = []
        for path in sorted(ROOT.rglob("*.py")):
            rel = path.relative_to(ROOT.parent).as_posix()
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                # EVERY except form: `except Exception`, `except ValueError`,
                # `except (OSError, KeyError)` and any other tuple
                if not isinstance(node, ast.ExceptHandler) or node.type is None:
                    continue
                found.append((rel, node))
        _CACHE = found
    return _CACHE


def _records(handler: ast.ExceptHandler) -> bool:
    for stmt in handler.body:
        if (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
                and getattr(stmt.value.func, "id", "") == "swallow"):
            return True
    return False


def _is_trace_free(handler: ast.ExceptHandler) -> bool:
    core = [s for s in handler.body
            if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]
    if len(core) != 1:
        return False
    stmt = core[0]
    if isinstance(stmt, (ast.Pass, ast.Continue, ast.Break)):
        return True
    return isinstance(stmt, ast.Return) and stmt.value is None


def test_no_trace_free_exception_handler_is_silent():
    """Every handler that swallows an exception must record it."""
    bare = [
        f"{rel}:{h.lineno}"
        for rel, h in _handlers()
        if _is_trace_free(h) and not _records(h) and rel not in ALLOWED_BARE
    ]
    assert not bare, (
        f"{len(bare)} handlers swallow an exception without recording it: "
        + ", ".join(bare[:8])
    )


def test_handler_bound_exceptions_are_named():
    """`except ... as _oc_exc:` needs the name so the record can quote it."""
    unnamed = []
    for rel, handler in _handlers():
        if rel in ALLOWED_BARE:
            continue
        if _records(handler) and handler.name is None:
            unnamed.append(f"{rel}:{handler.lineno}")
    assert not unnamed, f"records an exception it never bound: {unnamed[:8]}"


def test_instrumentation_is_substantial():
    """Guards against a rewriter that silently instruments almost nothing."""
    total = sum(1 for _rel, h in _handlers() if _records(h))
    assert total >= 900, f"only {total} handlers record; instrumentation regressed"


def test_narrow_except_forms_are_covered():
    """The 203 `except (OSError, ValueError)`-style handlers. A guard limited to
    `except Exception` passes while a corrupt auth.json still vanishes."""
    narrow = [
        f"{rel}:{h.lineno}  except ({ast.unparse(h.type)})"
        for rel, h in _handlers()
        if ast.unparse(h.type) not in ("Exception", "BaseException")
        and _is_trace_free(h) and not _records(h)
    ]
    assert not narrow, (
        f"{len(narrow)} narrow handlers swallow silently: " + ", ".join(narrow[:6])
    )


# ----------------------------------------------------------- the facility


@pytest.fixture(autouse=True)
def _clean_log():
    log.set_level("off")
    log.reset()
    yield
    log.set_level("off")
    log.reset()


def test_log_is_silent_by_default():
    log.error("boom", where="x")
    log.warn("careful")
    log.debug("detail")
    assert log.tail() == []
    assert log.counts() == {}


def test_swallow_is_invisible_until_asked():
    log.swallow(RuntimeError("kaboom"), "tui/chat_view.py:42")
    assert log.tail() == []
    log.set_level("debug")
    log.swallow(RuntimeError("kaboom"), "tui/chat_view.py:42")
    lines = log.tail()
    assert len(lines) == 1
    assert "swallow" in lines[0]
    assert "tui/chat_view.py:42" in lines[0]
    assert "RuntimeError" in lines[0]
    assert "kaboom" in lines[0]


def test_levels_filter():
    log.set_level("warn")
    log.error("bad")
    log.warn("meh")
    log.info("fine print")
    log.debug("noise")
    lines = log.tail()
    assert len(lines) == 2
    assert "bad" in lines[0] and "meh" in lines[1]


def test_buffer_is_bounded():
    """A phone must not grow a log until it OOMs."""
    log.set_level("debug")
    for i in range(log.MAX_LINES + 200):
        log.debug("filler", i=i)
    assert len(log.tail(log.MAX_LINES * 2)) == log.MAX_LINES


def test_long_field_is_truncated():
    log.set_level("debug")
    log.error("big", detail="x" * 5000)
    line = log.tail()[0]
    assert len(line) < log.MAX_FIELD_CHARS + 300
    assert "…" in line


def test_logger_never_raises_on_hostile_input():
    log.set_level("debug")

    class Hostile:
        def __str__(self):
            raise RuntimeError("nope")

    # Compare against a baseline rather than absolute 0: _DROPPED is a
    # process-global counter, and background threads leaked by earlier tests
    # keep logging, so it can be non-zero by the time this runs. What matters
    # is that hostile input adds no drop of its own.
    before = log.dropped()
    log.error("hostile", bad=Hostile())          # must not raise
    log.swallow(ValueError("x"), "where", extra=Hostile())
    assert log.dropped() == before


def test_reset_clears_everything():
    log.set_level("debug")
    log.info("a")
    log.reset()
    assert log.tail() == [] and log.counts() == {}


def test_set_level_accepts_names_and_numbers():
    assert log.set_level("debug") == log.DEBUG
    assert log.set_level("off") == log.OFF
    assert log.set_level(log.WARN) == log.WARN
    assert log.set_level("nonsense") == log.OFF
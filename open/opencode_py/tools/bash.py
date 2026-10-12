"""bash tool: run a shell command in a persistent shell session with timeout.

RULES (full guidance for maintainers; the sent schema is short):
- Terminal ops only (git, npm, docker…). NEVER file ops (read/write/edit/
  search) — dedicated tools exist for those.
- Runs in cwd by default; use `workdir`, never `cd <dir> && <cmd>`.
- ESC aborts via the engine interrupt hook; output truncated to caps.
"""

from __future__ import annotations

from ..util.log import swallow
import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path
from io import UnsupportedOperation
from typing import Callable

from .registry import Registry, Tool, schema_with

READ_CHUNK = 65536


def _kill_proc_tree(proc: subprocess.Popen, pgid: int | None = None) -> None:
    """Kill the shell AND every process it spawned.

    The shell is started as a new session/process-group leader, so background
    grandchildren (`sleep 1000 &`) live in the same group as the shell; a bare
    `proc.kill()` only signals the shell PID and leaks the children, which keep
    holding the stdout pipe and the cwd hostage. Killing the group reaps them.

    ``pgid`` should be the shell's process-group id captured while the shell
    was still alive. Passing it by value matters: by the time we clean up, the
    shell PID is often already gone (the command finished), but its group id
    still exists as long as a stray background child keeps a membership — and
    `os.getpgid(proc.pid)` would then fail, losing the handle on that group.
    Fall back to resolving it live only when no pgid was given.

    Children that daemonize into their own session (`setsid`, ...) escape the
    shell's group. On Linux we catch those that are still direct children of
    the live shell (the timeout/interrupt paths) by scanning /proc by
    parentage; a fully double-forked daemon already reparented to init is out
    of scope. That sweep is skipped when the shell has already exited — its
    children have been reparented and nothing is left to catch, and killing the
    process group already handled any non-daemonized strays. Scanning /proc
    opens a file per process and is measurable overhead on every command, so
    the common (clean-exit) path must not pay it.
    """
    if sys.platform == "win32" or not hasattr(os, "killpg"):
        try:
            proc.kill()
        except Exception as _oc_exc:
            swallow(_oc_exc, "tools/bash.py:55")
            pass
        return
    shell_alive = proc.poll() is None
    group_killed = False
    if pgid:
        try:
            os.killpg(pgid, signal.SIGKILL)
            group_killed = True
        except (ProcessLookupError, PermissionError, OSError) as _oc_exc:
            swallow(_oc_exc, "opencode_py/tools/bash.py:66")
            pass
    if not group_killed:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            group_killed = True
        except (ProcessLookupError, PermissionError, OSError) as _oc_exc:
            swallow(_oc_exc, "opencode_py/tools/bash.py:72")
            pass
    # Catch daemonized (`setsid`) children that escaped the group while their
    # parent shell is still alive (timeout/interrupt paths). Once the shell has
    # exited those children have reparented, so the /proc parentage scan finds
    # nothing — skip it then.
    if shell_alive and not group_killed and os.path.isdir("/proc"):
        ppid = proc.pid
        try:
            for entry in os.listdir("/proc"):
                if not entry.isdigit():
                    continue
                try:
                    with open("/proc/%s/stat" % entry, "rb") as fh:
                        stat = fh.read()
                    # format: pid (comm) state ppid pgrp ...
                    rparen = stat.rfind(b")")
                    if rparen < 0 or rparen + 2 >= len(stat):
                        continue
                    rest = stat[rparen + 2 :].split()
                    if len(rest) >= 2 and rest[1].isdigit() and int(rest[1]) == ppid:
                        os.kill(int(entry), signal.SIGKILL)
                except Exception as _oc_exc:
                    swallow(_oc_exc, "tools/bash.py:92")
                    continue
        except Exception as _oc_exc:
            swallow(_oc_exc, "tools/bash.py:94")
            pass
    if not group_killed:
        try:
            proc.kill()
        except Exception as _oc_exc:
            swallow(_oc_exc, "tools/bash.py:99")
            pass


def _stream_output(
    proc: subprocess.Popen,
    deadline: float,
    max_bytes: int,
    is_interrupted: Callable[[], bool] | None = None,
) -> tuple[bytes, bool, str, bytes, int]:
    """Read the process output to a natural end, capping the captured bytes.

    Returns ``(raw, capped, end)`` where ``end`` is one of:

    - ``"shell_exited"`` — the main shell terminated (poll() non-None).
      This is the honest end of the command. The stdout pipe may still be open
      because a stray background child inherited it; we do NOT wait for that
      pipe's EOF, and we drain only what is already buffered. It is the
      caller's job to reap the group so the stray child dies too.
    - ``"eof"`` — the pipe reached EOF (every writer closed it).
    - ``"timeout"`` — the deadline was hit before the command finished.
    - ``"interrupted"`` — ``is_interrupted()`` became true mid-command.

    Uses the raw fd (not the buffered pipe object, whose internal buffer can
    hold data `select` doesn't see) so nothing gets stuck. The fd is set
    non-blocking so EOF (``os.read`` -> ``b""``) is detectable even while a
    background child keeps the pipe open; bytes beyond the cap are still
    drained (discarded) so the child never blocks on a full pipe and memory
    stays bounded regardless of output size.
    """
    chunks: list[bytes] = []
    captured = 0
    capped = False
    eof = False
    total = 0  # every byte the child produced, even past the cap
    tail_keep = bytearray()  # last max_bytes of the FULL stream (errors live here)
    stdout = proc.stdout
    if stdout is None:
        return b"", False, "shell_exited", b"", 0
    try:
        fd = stdout.fileno()
    except (OSError, ValueError, UnsupportedOperation):
        return b"", False, "shell_exited", b"", 0
    try:
        os.set_blocking(fd, False)
    except OSError as _oc_exc:
        swallow(_oc_exc, "opencode_py/tools/bash.py:149")
        pass

    def append(data: bytes) -> None:
        nonlocal captured, capped, total
        total += len(data)
        # tail ring: ALWAYS keep the last max_bytes of the FULL stream, so the
        # process exit code / traceback / test failure (which prints LAST) is
        # never lost even when the head cap cuts it off.
        tail_keep.extend(data)
        if len(tail_keep) > max_bytes:
            del tail_keep[:len(tail_keep) - max_bytes]
        if not capped:
            room = max_bytes - captured
            if room > 0:
                chunks.append(data[:room])
                captured += len(data[:room])
            if len(data) > room:
                capped = True

    while True:
        # Honor a user interrupt the moment it lands (ESC / ctrl+C flips
        # the shared flag the app wires into the engine's interrupt callback).
        if is_interrupted is not None:
            try:
                if is_interrupted():
                    return b"".join(chunks), capped, "interrupted", bytes(tail_keep), total
            except Exception as _oc_exc:
                swallow(_oc_exc, "tools/bash.py:171")
                pass
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return b"".join(chunks), capped, "timeout", bytes(tail_keep), total
        if proc.poll() is not None:
            # The shell is done. The pipe may still be open because a
            # background grandchild inherited it: do NOT wait for that pipe's
            # EOF, or a finished command would block to the deadline and fake
            # a timeout. Drain what is already buffered and let the caller
            # reap the group (killing the stray child outright).
            if not eof:
                while True:
                    try:
                        data = os.read(fd, READ_CHUNK)
                    except BlockingIOError as _oc_exc:
                        swallow(_oc_exc, "opencode_py/tools/bash.py:192")
                        break
                    except OSError as _oc_exc:
                        swallow(_oc_exc, "opencode_py/tools/bash.py:194")
                        break
                    if not data:
                        eof = True
                        break
                    append(data)
            return b"".join(chunks), capped, "shell_exited", bytes(tail_keep), total
        if eof:
            # Every writer closed the pipe, but the shell itself is still
            # alive (e.g. it is waiting on a child that redirected stdout).
            # Wait for the shell in small slices instead of re-selecting on a
            # permanently-readable fd (busy spin).
            try:
                proc.wait(timeout=min(remaining, 0.25))
            except subprocess.TimeoutExpired as _oc_exc:
                swallow(_oc_exc, "opencode_py/tools/bash.py:208")
                continue
            return b"".join(chunks), capped, "shell_exited", bytes(tail_keep), total
        # Wait for data (in short slices so the deadline stays live). This runs
        # whether or not the main shell has exited: a background child that
        # keeps the pipe open can still write after the shell is done. Status
        # changes are detected at the top of the loop, not here.
        ready, _, _ = select.select([fd], [], [], min(remaining, 0.25))
        if not ready:
            continue
        try:
            data = os.read(fd, READ_CHUNK)
        except BlockingIOError as _oc_exc:
            swallow(_oc_exc, "opencode_py/tools/bash.py:220")
            # select reported ready but the read raced an empty pipe; retry
            continue
        except OSError:
            return b"".join(chunks), capped, "eof", bytes(tail_keep), total
        if not data:
            eof = True
            continue
        append(data)


def _bash(
    command: str,
    timeout: int = 120,
    workdir: str | None = None,
    max_lines: int = 2000,
    max_bytes: int = 51200,
    is_interrupted: Callable[[], bool] | None = None,
) -> dict:
    if len(command) > 200_000:
        return {"output": "Command too long (max 200KB).", "error": True, "exit_code": 1}
    cwd = Path(workdir).resolve() if workdir else Path.cwd()
    if not cwd.exists():
        # "(no output)" is indistinguishable from a command that ran fine and
        # printed nothing, so the caller could not tell a mistyped path from
        # genuine silence. Name the directory instead.
        return {
            "output": f"working directory does not exist: {cwd}",
            "error": True,
            "exit_code": 1,
        }
    if not cwd.is_dir():
        return {
            "output": f"working directory is not a directory: {cwd}",
            "error": True,
            "exit_code": 1,
        }

    shell = os.environ.get("SHELL", "/bin/sh")
    proc = None
    try:
        # start_new_session makes the shell the leader of its own process group
        # so a timeout can SIGKILL the whole group (background children too),
        # not just the shell PID.
        proc = subprocess.Popen(
            command,
            shell=True,
            executable=shell,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env={k: v for k, v in os.environ.items() if not k.startswith("OPENCODE_") or k in ("OPENCODE_CONFIG",)},
            start_new_session=sys.platform != "win32",
        )
    except Exception as e:
        return {"output": f"failed to run command: {e}", "error": True, "exit_code": 127}

    deadline = time.monotonic() + max(float(timeout), 0.1)
    # Capture the shell's process-group id now, while the shell PID is still
    # alive. When the command finishes the shell dies, but stray background
    # children keep this pgid; a later getpgid(proc.pid) would fail and lose
    # the handle on them, so we kill by the captured id instead.
    pgid: int | None = None
    if sys.platform != "win32" and hasattr(os, "getpgid"):
        try:
            pgid = os.getpgid(proc.pid)
        except (ProcessLookupError, PermissionError, OSError) as _oc_exc:
            swallow(_oc_exc, "opencode_py/tools/bash.py:273")
            pass
    raw: bytes = b""
    capped = False
    end = "shell_exited"
    tail_full: bytes = b""
    total = 0
    adopted: dict | None = None
    try:
        raw, capped, end, tail_full, total = _stream_output(proc, deadline, max_bytes, is_interrupted)
        if end == "timeout":
            try:
                from . import background as _bg

                adopted = _bg.adopt(proc, pgid, command, str(cwd), seed=raw)
            except Exception:
                adopted = None
            if adopted is not None and not adopted.get("error"):
                text = raw.decode("utf-8", errors="replace")
                tid = (adopted.get("metadata") or {}).get("task_id", "")
                result = {
                    "output": text,
                    "exit_code": -1,
                    "error": True,
                    "metadata": {
                        "exit_code": -1,
                        "truncated": capped,
                        "timeout": True,
                        "timed_out": True,
                        "continued": True,
                        "task_id": tid,
                    },
                }
                if tid:
                    prefix = (
                        f"Timed out after {timeout:.1f}s — job CONTINUES as {tid} "
                        "(no re-run needed).\n"
                        f"Poll with background_task action=read task_id={tid}.\n"
                        f"Partial output so far ({len(raw)} bytes):\n---\n"
                    )
                    result["output"] = prefix + text
                    result["metadata"]["hint"] = (
                        f"background_task action=read task_id={tid}"
                    )
                return _apply_caps(result, max_lines, max_bytes)
    finally:
        if adopted is None or adopted.get("error"):
            _kill_proc_tree(proc, pgid)
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired as _oc_exc:
                swallow(_oc_exc, "opencode_py/tools/bash.py:323")
                pass

    text = raw.decode("utf-8", errors="replace")
    tail_text = tail_full.decode("utf-8", errors="replace")
    if end in ("timeout", "interrupted"):
        result: dict = {
            "output": text,
            "exit_code": -1,
            "error": True,
            "metadata": {"exit_code": -1, "truncated": capped,
                         "tail": tail_text if capped else "",
                         "total_bytes": total if capped else len(raw)},
        }
        if end == "timeout":
            result["metadata"]["timeout"] = True
            result["metadata"]["timed_out"] = True
        else:
            result["metadata"]["stopped"] = True
            result["metadata"]["interrupted"] = True
            result["stopped"] = True
            result["interrupted"] = True
        return _apply_caps(result, max_lines, max_bytes)

    code = proc.returncode if proc.returncode is not None else 0
    result = {
        "output": text,
        "exit_code": code,
        "metadata": {"exit_code": code, "truncated": capped,
                     "tail": tail_text if capped else "",
                     "total_bytes": total if capped else len(raw)},
    }
    return _apply_caps(result, max_lines, max_bytes)


def _apply_caps(result: dict, max_lines: int, max_bytes: int) -> dict:
    output = result.get("output", "")
    truncated = bool(result.get("metadata", {}).get("truncated"))
    tail_text = str(result.get("metadata", {}).pop("tail", "") or "")
    total_bytes = result.get("metadata", {}).pop("total_bytes", None)
    raw = output.encode("utf-8", errors="replace")
    raw_len = len(raw)
    stream_cut = bool(truncated and tail_text and total_bytes and int(total_bytes) > max_bytes)
    if stream_cut or raw_len > max_bytes:
        # head+tail: errors/failures live at the TAIL, old code kept head only
        # and lost them forever. _stream_output rescues the true tail bytes,
        # so splice them in and cut the dull middle. Note: on a stream-cap hit
        # raw is head-only at exactly max_bytes, so raw_len is NOT > max_bytes
        # — the `stream_cut` arm fires regardless of raw_len.
        if tail_text and total_bytes and int(total_bytes or 0) > 0:
            head = raw[: max_bytes // 2].decode("utf-8", errors="ignore")
            tail_b = tail_text.encode("utf-8", errors="replace")[-max_bytes // 2:].decode("utf-8", errors="ignore")
            omitted = max(0, int(total_bytes) - max_bytes) if total_bytes else max(0, raw_len - max_bytes)
            output = (head + f"\n... [{omitted} middle bytes omitted — head+tail kept] ...\n" + tail_b)
            if total_bytes:
                result.get("metadata", {})["total_bytes"] = int(total_bytes)
        else:
            half = max_bytes // 2
            head = raw[:half].decode("utf-8", errors="ignore")
            tail = raw[-half:].decode("utf-8", errors="ignore")
            omitted = raw_len - max_bytes
            output = (head + f"\n... [{omitted} middle bytes omitted — head+tail kept] ...\n" + tail)
        truncated = True
    lines = output.splitlines()
    if len(lines) > max_lines:
        # Trim the MIDDLE, never the tail. Errors and test failures print LAST,
        # so `lines[:max_lines]` silently deleted exactly the output this whole
        # head+tail design exists to preserve -- a short-line but very long
        # output (minified JS, dense JSON, seq/log dumps with few newlines)
        # hits the line cap before the byte cap and loses its failure.
        head_n = max_lines // 2
        tail_n = max_lines - head_n
        omitted_lines = len(lines) - max_lines
        output = "\n".join(lines[:head_n])
        output += f"\n... plus {omitted_lines} more lines omitted — head+tail kept ..."
        output += "\n" + "\n".join(lines[-tail_n:])
        truncated = True
    result["output"] = output
    result["metadata"]["truncated"] = truncated
    return result


def tool(
    max_lines: int = 2000,
    max_bytes: int = 51200,
    default_timeout: int = 120,
    registry: Registry | None = None,
    max_timeout_ms: int = 1800_000,
) -> Tool:
    description = """Run a shell command (git/npm/docker…). No file ops — use the file tools. Runs in cwd; pass workdir instead of cd. On timeout the job CONTINUES as a background_task (no re-run, poll with action=read)."""

    def run(input: dict) -> dict:
        # The loop wraps every tool call in try/except, so a bare KeyError or
        # TypeError surfaced as "bash failed: 'command'" — a raw crash string
        # for what is really a missing or mistyped argument.
        command = input.get("command")
        if not isinstance(command, str) or not command.strip():
            return {
                "output": "bash needs a command to run (a non-empty string). "
                "Nothing was run.",
                "error": True,
                "exit_code": 1,
            }
        # `input.get("timeout") or default` swallowed an explicit timeout of 0
        # (0 is falsy), so asking for the shortest possible wait silently got
        # the 120s default. Distinguish "absent" from "zero", and bound the top
        # end: nothing capped it, so timeout=1e9 meant ~11 days of hanging.
        raw_timeout = input.get("timeout")
        if raw_timeout in (None, ""):
            timeout_ms = default_timeout * 1000
        else:
            try:
                timeout_ms = int(raw_timeout)
            except (TypeError, ValueError):
                return {
                    "output": f"timeout must be a number of milliseconds, got "
                    f"{raw_timeout!r}. Nothing was run.",
                    "error": True,
                    "exit_code": 1,
                }
        clamped = None
        if timeout_ms < 0:
            timeout_ms = 0
        elif timeout_ms > max_timeout_ms:
            clamped = timeout_ms
            timeout_ms = max_timeout_ms
        timeout = max(0.1, timeout_ms / 1000.0)
        workdir = input.get("workdir")
        # Read the engine's interrupt callback at call time (the registry hook
        # is installed by AgentLoop.__init__) so ESC/ctrl+C aborts a running
        # command instead of letting it run to its timeout. Each engine (main
        # + sub-agents) owns its own registry, so workers always see their own
        # turn's interrupt state.
        is_interrupted: Callable[[], bool] | None = None
        if registry is not None:
            checker = getattr(registry, "interrupt_check", None)
            if callable(checker):
                is_interrupted = checker
        result = _bash(
            command,
            timeout=timeout,
            workdir=workdir,
            max_lines=max_lines,
            max_bytes=max_bytes,
            is_interrupted=is_interrupted,
        )
        if clamped is not None:
            # Say so rather than silently running for less than asked, so a
            # capped long job is never mistaken for one that finished early.
            result["output"] = (
                f"(timeout capped at {max_timeout_ms} ms — you asked for "
                f"{clamped} ms)\n" + result.get("output", "")
            )
        return result

    return Tool(
        name="bash",
        description=description,
        parameters=schema_with(
            {
                "command": {
                    "type": "string",
                    "description": "Shell command",
                },
                "timeout": {
                    "type": "number",
                    "description": "Ms (default 120000)",
                    "optional": True,
                },
                "workdir": {
                    "type": "string",
                    "description": "Working directory",
                    "optional": True,
                },
            },
            ["command"],
        ),
        run=run,
        permission="bash",
    )

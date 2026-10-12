"""Start and stop the local OpenCode server as a managed background process."""
from __future__ import annotations

from .util.log import swallow
import argparse
import json
import os
import secrets
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .globals import Path as GPath
from .globals import resolve_worktree

STATE_VERSION = 1
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 4096


def _state_path() -> Path:
    GPath.init()
    return GPath.data / "server.json"


def _log_path() -> Path:
    GPath.init()
    return GPath.data / "server.log"


def _read_state() -> dict[str, Any] | None:
    try:
        value = json.loads(_state_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _write_state(state: dict[str, Any]) -> None:
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
    try:
        os.chmod(temporary, 0o600)
    except OSError as _oc_exc:
        swallow(_oc_exc, "opencode_py/server_control.py:50")
        pass
    os.replace(temporary, path)
    try:
        os.chmod(path, 0o600)
    except OSError as _oc_exc:
        swallow(_oc_exc, "opencode_py/server_control.py:55")
        pass


def _remove_state() -> None:
    try:
        _state_path().unlink()
    except FileNotFoundError as _oc_exc:
        swallow(_oc_exc, "opencode_py/server_control.py:62")
        pass


def _pid_alive(pid: int) -> bool:
    try:
        status = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        state = status.rsplit(") ", 1)[1][:1]
        if state == "Z":
            return False
    except OSError as _oc_exc:
        swallow(_oc_exc, "opencode_py/server_control.py:72")
        pass
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def _pid_matches_server(pid: int, directory: str) -> bool:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return False
    command = raw.replace(b"\0", b" ").decode("utf-8", errors="replace")
    return "opencode_py.main" in command and "--serve" in command and directory in command


def _url(host: str, port: int) -> str:
    display_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
    return f"http://{display_host}:{port}"


def _request(url: str, token: str, method: str = "GET", timeout: float = 0.7) -> dict[str, Any]:
    request = Request(
        f"{url}/api/v1/{'info' if method == 'GET' else 'shutdown'}",
        headers={"Authorization": f"Bearer {token}"},
        method=method,
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"server returned HTTP {exc.code}: {raw}") from exc
    except URLError as exc:
        raise RuntimeError(str(exc.reason)) from exc
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _healthy(url: str, token: str) -> bool:
    try:
        return _request(url, token).get("version") is not None
    except RuntimeError:
        return False


def start_server(
    directory: str | Path = ".",
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    token: str | None = None,
    cors: str | None = None,
) -> dict[str, Any]:
    GPath.init()
    worktree = str(Path(resolve_worktree(Path(directory))).resolve())
    host = host or DEFAULT_HOST
    if not 1 <= int(port) <= 65535:
        raise ValueError("server port must be between 1 and 65535")
    port = int(port)
    existing = _read_state()
    previous = ""
    if existing:
        old_url = str(existing.get("url", ""))
        old_token = str(existing.get("token", ""))
        if old_url and _healthy(old_url, old_token):
            return existing
        previous = old_token
        _remove_state()
    token = token or os.environ.get("OPENCODE_SERVER_TOKEN") or previous or secrets.token_urlsafe(32)
    if len(token) < 16:
        raise ValueError("server token must be at least 16 characters")
    if cors is None:
        cors = "*"
    command = [
        sys.executable,
        "-m",
        "opencode_py.main",
        worktree,
        "--serve",
        "--server-host",
        host,
        "--server-port",
        str(port),
        "--server-token",
        token,
    ]
    if cors:
        command.extend(["--server-cors", cors])
    log_file = _log_path()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    output = log_file.open("ab", buffering=0)
    try:
        process = subprocess.Popen(
            command,
            cwd=worktree,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            close_fds=True,
            start_new_session=True,
        )
    finally:
        output.close()
    url = _url(host, port)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited during startup; see {log_file}")
        if _healthy(url, token):
            state = {
                "version": STATE_VERSION,
                "pid": process.pid,
                "directory": worktree,
                "host": host,
                "port": port,
                "url": url,
                "token": token,
                "started_at": time.time(),
            }
            _write_state(state)
            return state
        time.sleep(0.15)
    try:
        os.kill(process.pid, signal.SIGTERM)
    except OSError as _oc_exc:
        swallow(_oc_exc, "opencode_py/server_control.py:202")
        pass
    raise RuntimeError(f"server did not become ready; see {log_file}")


def stop_server() -> dict[str, Any]:
    state = _read_state()
    if not state:
        return {"running": False, "stopped": True}
    url = str(state.get("url", ""))
    token = str(state.get("token", ""))
    pid = int(state.get("pid", 0) or 0)
    if pid <= 0:
        _remove_state()
        return {"running": False, "stopped": True}
    if url and token:
        try:
            _request(url, token, method="POST", timeout=1)
        except RuntimeError as _oc_exc:
            swallow(_oc_exc, "opencode_py/server_control.py:220")
            pass
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and _pid_alive(pid):
        time.sleep(0.1)
    if _pid_alive(pid) and _pid_matches_server(pid, str(state.get("directory", ""))):
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError as _oc_exc:
            swallow(_oc_exc, "opencode_py/server_control.py:228")
            pass
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and _pid_alive(pid):
        time.sleep(0.1)
    if _pid_alive(pid):
        raise RuntimeError("server did not stop")
    _remove_state()
    return {"running": False, "stopped": True}


def server_status() -> dict[str, Any]:
    state = _read_state()
    if not state:
        return {"running": False, "url": ""}
    running = _healthy(str(state.get("url", "")), str(state.get("token", "")))
    result = dict(state)
    result["running"] = running
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="opencode-server")
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser("start", help="start the background server")
    start.add_argument("directory", nargs="?", default=".")
    start.add_argument("--host", default=DEFAULT_HOST)
    start.add_argument("--port", type=int, default=DEFAULT_PORT)
    start.add_argument("--token", default=None)
    start.add_argument("--cors", default=None)
    commands.add_parser("stop", help="stop the managed background server")
    commands.add_parser("status", help="show the managed server status")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "start":
            result = start_server(args.directory, host=args.host, port=args.port, token=args.token, cors=args.cors)
            print(f"Server started: {result['url']}")
            print(f"Server token: {result['token']}")
        elif args.command == "stop":
            result = stop_server()
            print("Server stopped" if result.get("stopped") else "Server is not running")
        else:
            result = server_status()
            if result.get("running"):
                print(f"Server running: {result.get('url', '')}")
            else:
                print("Server stopped")
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "server_status", "start_server", "stop_server"]

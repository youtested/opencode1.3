"""Shared local control service for the OpenCode coding server."""
from __future__ import annotations

from .util.log import swallow
import argparse
import json
import os
import secrets
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

from . import __version__, server_control
from .globals import Path as GPath

MAX_BODY_BYTES = 64 * 1024
DEFAULT_CONTROL_PORT = 4097
DEFAULT_CODING_PORT = 4096


def _config_path() -> Path:
    GPath.init()
    return GPath.data / "control.json"


def _read_config() -> dict[str, Any]:
    try:
        value = json.loads(_config_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_config(value: dict[str, Any]) -> None:
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    try:
        os.chmod(temporary, 0o600)
    except OSError as _oc_exc:
        swallow(_oc_exc, "opencode_py/control.py:45")
        pass
    os.replace(temporary, path)
    try:
        os.chmod(path, 0o600)
    except OSError as _oc_exc:
        swallow(_oc_exc, "opencode_py/control.py:50")
        pass


def _cors_values(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return [str(item).strip() for item in (value or ()) if str(item).strip()]


class ControlState:
    def __init__(
        self,
        *,
        token: str,
        host: str = "127.0.0.1",
        port: int = DEFAULT_CONTROL_PORT,
        cors: str | list[str] | tuple[str, ...] | None = "*",
        coding_host: str = "127.0.0.1",
        coding_port: int = DEFAULT_CODING_PORT,
        coding_directory: str | Path = ".",
    ):
        self.token = token
        self.host = host
        self.port = port
        self.cors = tuple(_cors_values(cors))
        self.coding_host = coding_host
        self.coding_port = coding_port
        self.coding_directory = str(Path(coding_directory).resolve())
        self.httpd: Any = None
        self.started_at = time.time()

    @property
    def url(self) -> str:
        display_host = "127.0.0.1" if self.host in ("0.0.0.0", "::") else self.host
        return f"http://{display_host}:{self.port}"

    def status(self) -> dict[str, Any]:
        result = dict(server_control.server_status())
        result.update(
            {
                "control_url": self.url,
                "control_version": __version__,
                "control_started_at": self.started_at,
                "coding_directory": self.coding_directory,
            }
        )
        return result

    def start(self, body: dict[str, Any]) -> dict[str, Any]:
        port = body.get("coding_port", self.coding_port)
        try:
            port = int(port)
        except (TypeError, ValueError) as exc:
            raise ValueError("coding_port must be a number") from exc
        result = server_control.start_server(
            self.coding_directory,
            host=str(body.get("coding_host") or self.coding_host),
            port=port,
            token=str(body.get("token") or "") or None,
            cors=str(body.get("cors") or "") or None,
        )
        result = dict(result)
        result["running"] = True
        return result

    def stop(self) -> dict[str, Any]:
        result = dict(server_control.stop_server())
        result["running"] = False
        return result

    def toggle(self, body: dict[str, Any]) -> dict[str, Any]:
        if self.status().get("running"):
            return self.stop()
        return self.start(body)


class _ControlHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = f"opencode-control/{__version__}"

    @property
    def state(self) -> ControlState:
        return self.server.state  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: Any) -> None:
        return

    def _origin(self) -> str | None:
        origin = self.headers.get("Origin")
        if not origin or not self.state.cors:
            return None
        if "*" in self.state.cors:
            return "*"
        return origin if origin in self.state.cors else None

    def _headers(self, content_type: str, length: int | None = None) -> None:
        self.send_header("Content-Type", content_type)
        if length is not None:
            self.send_header("Content-Length", str(length))
        origin = self._origin()
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")

    def _send(self, status: int, body: bytes = b"", content_type: str = "application/json") -> None:
        self.send_response(status)
        self._headers(content_type, len(body))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _json(self, status: int, value: Any) -> None:
        self._send(status, json.dumps(value, ensure_ascii=False).encode("utf-8"))

    def _error(self, status: int, message: str) -> None:
        self._json(status, {"error": message})

    def _authorized(self) -> bool:
        header = self.headers.get("Authorization", "")
        token = header[7:] if header.lower().startswith("bearer ") else ""
        token = token or self.headers.get("X-OpenCode-Control-Token", "")
        return bool(token) and secrets.compare_digest(token, self.state.token)

    def _path(self) -> tuple[str, dict[str, list[str]]]:
        parsed = urlsplit(self.path)
        return parsed.path.rstrip("/") or "/", parse_qs(parsed.query)

    def _body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise ValueError("invalid Content-Length") from exc
        if length < 0 or length > MAX_BODY_BYTES:
            raise ValueError("request body is too large")
        raw = self.rfile.read(length) if length else b""
        if not raw:
            return {}
        try:
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("request body must be valid JSON") from exc
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def _page(self) -> None:
        try:
            body = (Path(__file__).with_name("web") / "control.html").read_bytes()
        except OSError:
            self._error(HTTPStatus.NOT_FOUND, "control page is not installed")
            return
        self._send(HTTPStatus.OK, body, "text/html; charset=utf-8")

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self._headers("text/plain", 0)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type, X-OpenCode-Control-Token")
        origin = self._origin()
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
        self.end_headers()

    def do_GET(self) -> None:
        path, _query = self._path()
        if path in ("/", "/control.html"):
            self._page()
            return
        if path == "/api/v1/control/health":
            self._json(HTTPStatus.OK, {"status": "ok", "version": __version__})
            return
        if not self._authorized():
            self._error(HTTPStatus.UNAUTHORIZED, "missing or invalid control token")
            return
        if path == "/api/v1/control/status":
            self._json(HTTPStatus.OK, self.state.status())
            return
        self._error(HTTPStatus.NOT_FOUND, "endpoint not found")

    def do_POST(self) -> None:
        if not self._authorized():
            self._error(HTTPStatus.UNAUTHORIZED, "missing or invalid control token")
            return
        path, _query = self._path()
        try:
            body = self._body()
        except ValueError as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        try:
            if path == "/api/v1/control/start":
                result = self.state.start(body)
            elif path == "/api/v1/control/stop":
                result = self.state.stop()
            elif path == "/api/v1/control/toggle":
                result = self.state.toggle(body)
            else:
                self._error(HTTPStatus.NOT_FOUND, "endpoint not found")
                return
        except (OSError, RuntimeError, ValueError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        self._json(HTTPStatus.OK, result)


class _ControlServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def make_control_server(state: ControlState, host: str = "127.0.0.1", port: int = 0) -> _ControlServer:
    httpd = _ControlServer((host, port), _ControlHandler)
    httpd.state = state  # type: ignore[attr-defined]
    state.httpd = httpd
    bound_host, bound_port = httpd.server_address[:2]
    state.host = str(bound_host)
    state.port = int(bound_port)
    return httpd


def _control_token(token: str | None) -> str:
    token = token or os.environ.get("OPENCODE_CONTROL_TOKEN")
    if token:
        if len(token) < 16:
            raise ValueError("control token must be at least 16 characters")
        return token
    existing = _read_config().get("token")
    if isinstance(existing, str) and len(existing) >= 16:
        return existing
    token = secrets.token_urlsafe(32)
    _write_config({"token": token, "version": 1})
    return token


def serve_control(
    *,
    host: str = "127.0.0.1",
    port: int = DEFAULT_CONTROL_PORT,
    token: str | None = None,
    cors: str | list[str] | tuple[str, ...] | None = "*",
    coding_host: str = "127.0.0.1",
    coding_port: int = DEFAULT_CODING_PORT,
    coding_directory: str | Path = ".",
    open_browser: bool = False,
) -> ControlState:
    GPath.init()
    if not 0 <= int(port) <= 65535:
        raise ValueError("control port must be between 0 and 65535")
    state = ControlState(
        token=_control_token(token),
        host=host,
        port=int(port),
        cors=cors,
        coding_host=coding_host,
        coding_port=coding_port,
        coding_directory=coding_directory,
    )
    httpd = make_control_server(state, host, int(port))
    url = state.url
    print(f"OpenCode control service listening on {url}/", flush=True)
    print(f"Control token: {state.token}", flush=True)
    if open_browser:
        browser_url = f"{url}/#token={state.token}"
        threading.Timer(0.2, lambda: webbrowser.open(browser_url)).start()
    try:
        httpd.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt as _oc_exc:
        swallow(_oc_exc, "opencode_py/control.py:320")
        pass
    finally:
        httpd.server_close()
    return state


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="opencode-control")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_CONTROL_PORT)
    parser.add_argument("--token", default=None)
    parser.add_argument("--cors", default="*")
    parser.add_argument("--coding-host", default="127.0.0.1")
    parser.add_argument("--coding-port", type=int, default=DEFAULT_CODING_PORT)
    parser.add_argument("--coding-directory", default=".")
    parser.add_argument("--open-browser", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        serve_control(
            host=args.host,
            port=args.port,
            token=args.token,
            cors=args.cors,
            coding_host=args.coding_host,
            coding_port=args.coding_port,
            coding_directory=args.coding_directory,
            open_browser=args.open_browser,
        )
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ControlState", "main", "make_control_server", "serve_control"]

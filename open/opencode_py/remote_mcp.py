"""Optional remote MCP client for HTTP and streamable-HTTP servers."""
from __future__ import annotations

from .util.log import swallow
import json
import os
import threading
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


class RemoteMCPError(RuntimeError):
    pass


class RemoteMCPServer:
    is_remote = True

    def __init__(
        self,
        name: str,
        url: str,
        *,
        transport: str = "streamable-http",
        headers: dict[str, str] | None = None,
        auth: dict[str, Any] | None = None,
        timeout: float = 20.0,
        tool_timeout: float | None = None,
    ):
        self.name = name
        self.url = url
        self.transport = transport or "streamable-http"
        self.headers = dict(headers or {})
        self.auth = dict(auth or {})
        self.timeout = max(1.0, float(timeout or 20.0))
        self.tool_timeout = max(self.timeout, float(tool_timeout or 60.0))
        self._counter = 0
        self._lock = threading.RLock()
        self.server_info: dict[str, Any] = {}
        self._started = False
        self._aborted = False
        self._users = 0

    def _token(self) -> str:
        token = self.auth.get("token")
        env_name = self.auth.get("tokenEnv")
        if not token and env_name:
            token = os.environ.get(str(env_name))
        return str(token or "")

    def _auth_headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "User-Agent": "opencode_py",
            **self.headers,
        }
        token = self._token()
        if token:
            headers.setdefault("Authorization", "Bearer " + token)
        return headers

    def _refresh_token(self) -> bool:
        refresh = self.auth.get("refresh_token")
        token_url = self.auth.get("token_url")
        if not refresh or not token_url:
            return False
        data = {
            "grant_type": "refresh_token",
            "refresh_token": str(refresh),
        }
        for key in ("client_id", "client_secret", "scope"):
            if self.auth.get(key):
                data[key] = str(self.auth[key])
        request = Request(
            str(token_url),
            data=urlencode(data).encode("utf-8"),
            headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, OSError, ValueError):
            return False
        token = payload.get("access_token") if isinstance(payload, dict) else None
        if not token:
            return False
        self.auth["token"] = str(token)
        if payload.get("refresh_token"):
            self.auth["refresh_token"] = str(payload["refresh_token"])
        return True

    @staticmethod
    def _parse_response(raw: bytes, content_type: str) -> dict[str, Any]:
        text = raw.decode("utf-8", errors="replace").strip()
        if "text/event-stream" in (content_type or ""):
            messages = []
            for line in text.splitlines():
                if line.startswith("data:"):
                    try:
                        messages.append(json.loads(line[5:].strip()))
                    except json.JSONDecodeError as _oc_exc:
                        swallow(_oc_exc, "opencode_py/remote_mcp.py:104")
                        continue
            if not messages:
                raise RemoteMCPError("remote MCP returned an empty event stream")
            return messages[-1]
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RemoteMCPError(f"remote MCP returned invalid JSON: {exc}") from exc
        return value if isinstance(value, dict) else {"result": value}

    def _post(self, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        request = Request(
            self.url,
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers=self._auth_headers(),
            method="POST",
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                return self._parse_response(response.read(), response.headers.get("Content-Type", ""))
        except HTTPError as exc:
            if exc.code == 401 and self._refresh_token():
                return self._post(payload, timeout)
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise RemoteMCPError(f"remote MCP HTTP {exc.code}: {detail}") from exc
        except (URLError, OSError) as exc:
            raise RemoteMCPError(f"remote MCP connection failed: {exc}") from exc

    def _ensure_started(self) -> None:
        if self._started:
            return
        with self._lock:
            if self._started:
                return
            self._aborted = False
            result = self._post(
                {
                    "jsonrpc": "2.0",
                    "id": self._next_id(),
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {},
                        "clientInfo": {"name": "opencode_py", "version": "0.1.0"},
                    },
                },
                self.timeout,
            )
            if result.get("error"):
                raise RemoteMCPError(str(result["error"]))
            self._server_info = result.get("serverInfo") or result
            self._started = True
            try:
                self._post(
                    {"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}},
                    self.timeout,
                )
            except RemoteMCPError as _oc_exc:
                swallow(_oc_exc, "opencode_py/remote_mcp.py:162")
                pass

    def _next_id(self) -> int:
        with self._lock:
            self._counter += 1
            return self._counter

    def call(self, method: str, params: dict[str, Any] | None = None, timeout: float | None = None) -> dict[str, Any]:
        self._ensure_started()
        if self._aborted:
            raise RemoteMCPError("remote MCP request interrupted")
        result = self._post(
            {"jsonrpc": "2.0", "id": self._next_id(), "method": method, "params": params or {}},
            timeout or self.tool_timeout,
        )
        if result.get("error"):
            raise RemoteMCPError(str(result["error"].get("message") or result["error"]))
        return result.get("result") or {}

    def list_tools(self) -> list[dict[str, Any]]:
        return list(self.call("tools/list").get("tools") or [])

    def list_prompts(self) -> list[dict[str, Any]]:
        return list(self.call("prompts/list").get("prompts") or [])

    def list_resources(self) -> list[dict[str, Any]]:
        return list(self.call("resources/list").get("resources") or [])

    def get_prompt(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        return self.call("prompts/get", {"name": name, "arguments": arguments or {}})

    def read_resource(self, uri: str) -> dict[str, Any]:
        return self.call("resources/read", {"uri": uri})

    def run_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        result = self.call("tools/call", {"name": name, "arguments": arguments or {}})
        text = result.get("content")
        if text is None:
            text = json.dumps(result, ensure_ascii=False)
        if isinstance(text, list):
            parts = []
            for item in text:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            text = "\n".join(p for p in parts if p)
        return {"output": text, "error": bool(result.get("isError"))}

    def acquire(self) -> None:
        with self._lock:
            self._users += 1

    def release(self) -> None:
        with self._lock:
            self._users = max(0, self._users - 1)

    def abort(self) -> None:
        self._aborted = True

    def close(self) -> None:
        self._aborted = True
        self._started = False


__all__ = ["RemoteMCPError", "RemoteMCPServer"]

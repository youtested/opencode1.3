"""Small synchronous client for the OpenCode local server."""
from __future__ import annotations

import json
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


class OpenCodeError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, body: str = ""):
        super().__init__(message)
        self.status = status
        self.body = body


class OpenCodeClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:4096",
        token: str = "",
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def _url(self, path: str, query: dict[str, Any] | None = None) -> str:
        url = f"{self.base_url}/api/v1/{path.lstrip('/')}"
        if query:
            url = f"{url}?{urlencode(query)}"
        return url

    def _headers(self, *, accept: str = "application/json") -> dict[str, str]:
        headers = {"Accept": accept}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        *,
        query: dict[str, Any] | None = None,
        accept: str = "application/json",
    ) -> Any:
        body = None
        headers = self._headers(accept=accept)
        if payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = Request(
            self._url(path, query),
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
        except HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                message = json.loads(raw).get("error", raw)
            except (json.JSONDecodeError, AttributeError):
                message = raw or str(exc)
            raise OpenCodeError(str(message), status=exc.code, body=raw) from exc
        except URLError as exc:
            raise OpenCodeError(str(exc.reason)) from exc
        if not raw:
            return {}
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OpenCodeError("server returned invalid JSON", body=raw.decode("utf-8", errors="replace")) from exc

    def info(self) -> dict[str, Any]:
        return self._request("GET", "info")

    def health(self) -> dict[str, Any]:
        return self._request("GET", "health")

    def config(self) -> dict[str, Any]:
        return self._request("GET", "config")

    def sessions(self) -> list[dict[str, Any]]:
        return list(self._request("GET", "sessions").get("sessions", []))

    def create_session(
        self,
        *,
        title: str = "",
        agent: str | None = None,
        model: str | None = None,
        provider: str | None = None,
    ) -> dict[str, Any]:
        payload = {"title": title}
        if agent is not None:
            payload["agent"] = agent
        if model is not None:
            payload["model"] = model
        if provider is not None:
            payload["provider"] = provider
        return self._request("POST", "sessions", payload)

    def get_session(self, session_id: str) -> dict[str, Any]:
        return self._request("GET", f"sessions/{quote(session_id, safe='')}")

    def send_message(self, session_id: str, content: str) -> dict[str, Any]:
        return self._request(
            "POST",
            f"sessions/{quote(session_id, safe='')}/messages",
            {"content": content},
        )

    def abort(self, session_id: str) -> dict[str, Any]:
        return self._request("POST", f"sessions/{quote(session_id, safe='')}/abort", {})

    def events(
        self,
        session_id: str,
        *,
        after: int = 0,
        timeout: float | None = None,
    ) -> Iterator[dict[str, Any]]:
        path = f"sessions/{quote(session_id, safe='')}/events"
        request = Request(
            self._url(path, {"after": max(0, int(after))}),
            headers=self._headers(accept="text/event-stream"),
            method="GET",
        )
        try:
            response = urlopen(request, timeout=timeout or self.timeout)
        except HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            raise OpenCodeError(raw or str(exc), status=exc.code, body=raw) from exc
        except URLError as exc:
            raise OpenCodeError(str(exc.reason)) from exc
        try:
            data_lines: list[str] = []
            while True:
                line = response.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip("\r\n")
                if text.startswith("data:"):
                    data_lines.append(text[5:].lstrip())
                elif not text and data_lines:
                    raw_data = "\n".join(data_lines)
                    data_lines.clear()
                    try:
                        yield json.loads(raw_data)
                    except json.JSONDecodeError as exc:
                        raise OpenCodeError("server sent an invalid event") from exc
        finally:
            response.close()


class OpenCodeControlClient(OpenCodeClient):
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:4097",
        token: str = "",
        timeout: float = 30.0,
    ):
        super().__init__(base_url, token, timeout)

    def health(self) -> dict[str, Any]:
        return self._request("GET", "control/health")

    def status(self) -> dict[str, Any]:
        return self._request("GET", "control/status")

    def start(self, **options: Any) -> dict[str, Any]:
        return self._request("POST", "control/start", options)

    def stop(self) -> dict[str, Any]:
        return self._request("POST", "control/stop", {})

    def toggle(self, **options: Any) -> dict[str, Any]:
        return self._request("POST", "control/toggle", options)


__all__ = ["OpenCodeClient", "OpenCodeControlClient", "OpenCodeError"]

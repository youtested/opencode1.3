"""IDE bridge tools: read the editor's active file, open a file, or stage an edit.

These tools talk to the local server's /api/v1/ide endpoints. The Acode plugin
pushes the active file into the server and polls for actions, so the agent can
work on the file the user is actually looking at.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from ..globals import Path as GPath
from .registry import Tool, schema_with

DEFAULT_URL = "http://127.0.0.1:4096"


def _server_state() -> dict[str, Any]:
    try:
        value = json.loads((GPath.data / "server.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _base() -> str:
    url = os.environ.get("OPENCODE_SERVER_URL") or str(_server_state().get("url") or "")
    return (url or DEFAULT_URL).rstrip("/")


def _token() -> str:
    return os.environ.get("OPENCODE_SERVER_TOKEN") or str(_server_state().get("token") or "")


def _request(path: str, payload: dict[str, Any] | None = None, method: str = "GET") -> dict[str, Any]:
    url = f"{_base()}/api/v1/{path.lstrip('/')}"
    data = None
    headers = {"Accept": "application/json"}
    token = _token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        if exc.code == 401:
            detail += " (token is stale - run /server start to refresh it)"
        raise RuntimeError(f"ide server HTTP {exc.code}: {detail}") from exc
    except Exception as exc:
        raise RuntimeError(f"ide server unreachable at {_base()}: {exc}") from exc
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise RuntimeError("ide server returned invalid JSON") from exc
    return value if isinstance(value, dict) else {}


def tool() -> Tool:
    description = (
        "IDE integration: read the file the user currently has open in Acode, "
        "open a file at a line, or stage an edit for the editor. Use this when "
        "the user says 'this file', 'the open file', or asks to edit what they see."
    )

    def run(input: dict) -> dict:
        operation = str(input.get("operation") or "context").strip()
        try:
            if operation == "context":
                context = _request("ide/context").get("context") or {}
                if not context or not context.get("path"):
                    return {"output": "No editor file is active. Open a file in Acode first.", "error": True}
                lines = context.get("content", "").splitlines()
                line_no = int(context.get("line") or 0)
                excerpt = ""
                if 0 < line_no <= len(lines):
                    excerpt = "\n".join(lines[max(0, line_no - 4):line_no + 3])
                path = context.get("path")
                output = f"Active file: {path} ({context.get('language') or 'unknown'}, line {line_no})"
                if excerpt:
                    output += f"\n\n{excerpt}"
                return {"output": output, "metadata": {"path": path, "language": context.get("language")}}
            if operation == "open":
                path = str(input.get("filePath") or input.get("path") or "").strip()
                if not path:
                    return {"output": "filePath is required.", "error": True}
                _request("ide/action", {
                    "id": str(input.get("id") or "open"),
                    "type": "open",
                    "path": path,
                    "line": int(input.get("line") or 1),
                    "character": int(input.get("character") or 0),
                }, method="POST")
                return {"output": f"Asked Acode to open {path}."}
            if operation == "apply":
                path = str(input.get("filePath") or "").strip()
                content = str(input.get("content") or "")
                if not path:
                    return {"output": "filePath is required.", "error": True}
                _request("ide/action", {
                    "id": str(input.get("id") or "apply"),
                    "type": "apply",
                    "path": path,
                    "content": content,
                }, method="POST")
                return {"output": f"Staged an edit for {path}. The editor will ask you to Apply or Cancel."}
            return {"output": f"Unknown IDE operation {operation!r} (want context, open, or apply).", "error": True}
        except RuntimeError as exc:
            return {"output": str(exc), "error": True}

    return Tool(
        name="ide",
        description=description,
        parameters=schema_with(
            {
                "operation": {
                    "type": "string",
                    "description": "context (read the active file), open, or apply",
                    "enum": ["context", "open", "apply"],
                },
                "filePath": {"type": "string", "description": "File to open or edit"},
                "line": {"type": "integer", "description": "Line number"},
                "character": {"type": "integer", "description": "Column"},
                "content": {"type": "string", "description": "New full file content for apply"},
            },
            required=[],
        ),
        run=run,
    )

"""Optional, lazy JSON-RPC language-server support for the LSP tool."""
from __future__ import annotations

from .util.log import swallow
import json
import os
import subprocess
import threading
from pathlib import Path
from typing import Any


class LspError(RuntimeError):
    pass


_LANGUAGE_IDS = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascriptreact",
    ".ts": "typescript",
    ".tsx": "typescriptreact",
    ".rs": "rust",
    ".go": "go",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".java": "java",
    ".cs": "csharp",
    ".php": "php",
    ".rb": "ruby",
    ".swift": "swift",
    ".kt": "kotlin",
    ".scala": "scala",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".sh": "shellscript",
}


def language_for(path: str | Path) -> str:
    return _LANGUAGE_IDS.get(Path(path).suffix.lower(), "plaintext")


def path_uri(path: str | Path) -> str:
    return Path(path).resolve().as_uri()


class LspConnection:
    def __init__(self, name: str, command: str, args: list[str], cwd: Path, timeout: float = 5.0):
        self.name = name
        self.command = command
        self.args = list(args or [])
        self.cwd = Path(cwd)
        self.timeout = max(0.5, float(timeout or 5))
        self.process: subprocess.Popen[bytes] | None = None
        self._pending: dict[int, tuple[threading.Event, dict[str, Any]]] = {}
        self._documents: set[str] = set()
        self._diagnostics: dict[str, list[dict[str, Any]]] = {}
        self._request_id = 0
        self._lock = threading.Lock()
        self._reader: threading.Thread | None = None
        self._initialized = False

    def start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        try:
            self.process = subprocess.Popen(
                [self.command, *self.args],
                cwd=str(self.cwd),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=os.environ.copy(),
            )
        except OSError as exc:
            raise LspError(f"Could not start {self.name}: {exc}") from exc
        self._reader = threading.Thread(target=self._read_loop, name=f"lsp-{self.name}", daemon=True)
        self._reader.start()
        result = self.request(
            "initialize",
            {
                "processId": os.getpid(),
                "rootUri": path_uri(self.cwd),
                "workspaceFolders": [{"uri": path_uri(self.cwd), "name": self.cwd.name}],
                "capabilities": {
                    "textDocument": {
                        "definition": {"linkSupport": True},
                        "references": {},
                        "hover": {"contentFormat": ["plaintext", "markdown"]},
                        "documentSymbol": {"hierarchicalDocumentSymbolSupport": True},
                        "publishDiagnostics": {"relatedInformation": True},
                    },
                    "workspace": {"symbol": {}},
                },
                "clientInfo": {"name": "opencode_py"},
            },
        )
        self._initialized = True
        self.notify("initialized", {})
        self._server_result = result

    def _read_loop(self) -> None:
        stream = self.process.stdout if self.process else None
        if stream is None:
            return
        try:
            while True:
                headers: dict[str, str] = {}
                while True:
                    line = stream.readline()
                    if not line:
                        return
                    text = line.decode("utf-8", errors="replace").strip()
                    if not text:
                        break
                    if ":" in text:
                        key, value = text.split(":", 1)
                        headers[key.lower().strip()] = value.strip()
                length = int(headers.get("content-length", "0"))
                raw = stream.read(length)
                if not raw:
                    return
                message = json.loads(raw.decode("utf-8", errors="replace"))
                message_id = message.get("id")
                if isinstance(message_id, int):
                    with self._lock:
                        pending = self._pending.get(message_id)
                    if pending:
                        event, slot = pending
                        slot.update(message)
                        event.set()
                elif message.get("method") == "textDocument/publishDiagnostics":
                    params = message.get("params") or {}
                    uri = str(params.get("uri") or "")
                    if uri:
                        self._diagnostics[uri] = list(params.get("diagnostics") or [])
        except (OSError, ValueError, json.JSONDecodeError) as _oc_exc:
            swallow(_oc_exc, "opencode_py/lsp_runtime.py:142")
            return

    def _send(self, message: dict[str, Any]) -> None:
        process = self.process
        if process is None or process.stdin is None:
            raise LspError("Language server is not running")
        body = json.dumps(message, separators=(",", ":")).encode("utf-8")
        with self._lock:
            process.stdin.write(f"Content-Length: {len(body)}\r\n\r\n".encode("ascii") + body)
            process.stdin.flush()

    def request(self, method: str, params: dict[str, Any]) -> Any:
        if not self._initialized and method != "initialize":
            self.start()
        with self._lock:
            self._request_id += 1
            request_id = self._request_id
            event = threading.Event()
            slot: dict[str, Any] = {}
            self._pending[request_id] = (event, slot)
        try:
            self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
            if not event.wait(self.timeout):
                raise LspError(f"{self.name} timed out")
            if slot.get("error"):
                error = slot["error"]
                raise LspError(str(error.get("message") or error))
            return slot.get("result")
        finally:
            with self._lock:
                self._pending.pop(request_id, None)

    def notify(self, method: str, params: dict[str, Any]) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def ensure_document(self, path: Path) -> None:
        if not self._initialized:
            self.start()
        uri = path_uri(path)
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise LspError(f"Could not read {path}: {exc}") from exc
        params = {
            "textDocument": {
                "uri": uri,
                "languageId": language_for(path),
                "version": 1,
                "text": text,
            }
        }
        if uri not in self._documents:
            self.notify("textDocument/didOpen", params)
            self._documents.add(uri)
        else:
            self.notify(
                "textDocument/didChange",
                {"textDocument": {"uri": uri, "version": 1}, "contentChanges": [{"text": text}]},
            )

    def close(self) -> None:
        process = self.process
        if process is None:
            return
        try:
            if process.stdin:
                process.stdin.close()
        except OSError as _oc_exc:
            swallow(_oc_exc, "opencode_py/lsp_runtime.py:210")
            pass
        try:
            process.terminate()
            process.wait(timeout=1)
        except (OSError, subprocess.TimeoutExpired):
            try:
                process.kill()
            except OSError as _oc_exc:
                swallow(_oc_exc, "opencode_py/lsp_runtime.py:218")
                pass
        self.process = None
        self._initialized = False


class LspManager:
    def __init__(self, root: Path, config: dict[str, Any]):
        self.root = Path(root).resolve()
        self.config = config or {}
        self.timeout = float(self.config.get("timeout", 5) or 5)
        self.servers = self.config.get("servers") or {}
        self._connections: dict[str, LspConnection] = {}

    def _spec(self, language: str, path: Path) -> tuple[str, dict[str, Any]] | None:
        if not isinstance(self.servers, dict):
            return None
        for name, value in self.servers.items():
            if not isinstance(value, dict) or value.get("enabled") is False:
                continue
            languages = value.get("languages")
            extensions = value.get("extensions")
            if language in (languages or [name]):
                return str(name), value
            if any(path.suffix.lower() == str(ext).lower() for ext in (extensions or [])):
                return str(name), value
        return None

    def _connection(self, language: str, path: Path) -> tuple[str, LspConnection] | None:
        selected = self._spec(language, path)
        if selected is None:
            return None
        name, spec = selected
        connection = self._connections.get(name)
        if connection is None:
            command = str(spec.get("command") or "")
            if not command:
                return None
            connection = LspConnection(name, command, list(spec.get("args") or []), self.root, self.timeout)
            self._connections[name] = connection
        return name, connection

    @staticmethod
    def _location_text(item: Any) -> str:
        if not isinstance(item, dict):
            return str(item)
        uri = str(item.get("uri") or item.get("targetUri") or "")
        start = item.get("range") or item.get("targetSelectionRange") or item.get("targetRange") or {}
        start = start.get("start") if isinstance(start, dict) else {}
        line = int((start or {}).get("line", 0)) + 1 if isinstance(start, dict) else 1
        character = int((start or {}).get("character", 0)) + 1 if isinstance(start, dict) else 1
        path = uri.replace("file://", "")
        return f"{path}:{line}:{character}"

    @staticmethod
    def _contents(value: Any) -> str:
        if isinstance(value, dict):
            value = value.get("value") or value.get("contents") or ""
        if isinstance(value, list):
            return "\n".join(LspManager._contents(item) for item in value)
        return str(value or "").strip()

    def request(self, operation: str, file_path: str, line: int, character: int, query: str, limit: int) -> dict[str, Any] | None:
        path = Path(file_path).resolve()
        language = language_for(path)
        selected = self._connection(language, path)
        if selected is None:
            return None
        name, connection = selected
        try:
            connection.ensure_document(path)
            uri = path_uri(path)
            params: dict[str, Any]
            if operation == "workspaceSymbol":
                method, params = "workspace/symbol", {"query": query}
            elif operation == "documentSymbol":
                method, params = "textDocument/documentSymbol", {"textDocument": {"uri": uri}}
            elif operation in ("goToDefinition", "goToImplementation"):
                method = "textDocument/definition" if operation == "goToDefinition" else "textDocument/implementation"
                params = {"textDocument": {"uri": uri}, "position": {"line": max(0, line - 1), "character": max(0, character - 1)}}
            elif operation == "findReferences":
                method, params = "textDocument/references", {"textDocument": {"uri": uri}, "position": {"line": max(0, line - 1), "character": max(0, character - 1)}, "context": {"includeDeclaration": True}}
            elif operation == "hover":
                method, params = "textDocument/hover", {"textDocument": {"uri": uri}, "position": {"line": max(0, line - 1), "character": max(0, character - 1)}}
            elif operation == "diagnostics":
                method, params = "textDocument/diagnostic", {"textDocument": {"uri": uri}}
            else:
                return None
            result = connection.request(method, params)
            output = self._format(operation, result)
            if not output and operation == "diagnostics":
                output = "No diagnostics reported by the language server."
            if not output:
                return None
            return {"output": output, "metadata": {"source": "lsp", "server": name, "operation": operation}}
        except LspError:
            return None

    def _format(self, operation: str, result: Any) -> str:
        if operation == "hover":
            return self._contents(result)
        if operation == "diagnostics":
            items = result.get("items", []) if isinstance(result, dict) else result or []
            rows = []
            for item in items:
                if not isinstance(item, dict):
                    continue
                where = self._location_text({"uri": (item.get("location") or {}).get("uri", ""), "range": item.get("range", {})})
                rows.append(f"{where}: {item.get('message', '')}")
            return "\n".join(rows)
        if operation == "documentSymbol":
            rows = []
            for item in result or []:
                if not isinstance(item, dict):
                    continue
                where = self._location_text(item.get("location") or {"uri": "", "range": item.get("selectionRange") or item.get("range") or {}})
                rows.append(f"{where}: {item.get('name', '')}")
            return "\n".join(rows)
        if operation == "workspaceSymbol":
            rows = []
            for item in result or []:
                if isinstance(item, dict):
                    rows.append(f"{item.get('name', '')}: {item.get('containerName', '')}")
            return "\n".join(rows)
        if isinstance(result, list):
            return "\n".join(self._location_text(item) for item in result)
        return self._location_text(result)

    def close(self) -> None:
        for connection in self._connections.values():
            connection.close()
        self._connections.clear()


__all__ = ["LspConnection", "LspError", "LspManager", "language_for", "path_uri"]

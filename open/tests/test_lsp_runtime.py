import sys

from opencode_py.lsp_runtime import LspManager


FAKE_SERVER = r'''
import json
import sys

def read_message():
    headers = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            return None
        line = line.decode().strip()
        if not line:
            break
        key, value = line.split(":", 1)
        headers[key.lower()] = value.strip()
    length = int(headers["content-length"])
    return json.loads(sys.stdin.buffer.read(length).decode())

def send(value):
    body = json.dumps(value).encode()
    sys.stdout.buffer.write(f"Content-Length: {len(body)}\r\n\r\n".encode() + body)
    sys.stdout.buffer.flush()

while True:
    message = read_message()
    if message is None:
        break
    method = message.get("method")
    if method == "initialize":
        send({"jsonrpc": "2.0", "id": message["id"], "result": {"capabilities": {}}})
    elif method == "shutdown":
        send({"jsonrpc": "2.0", "id": message["id"], "result": None})
        break
    elif method == "textDocument/definition":
        send({"jsonrpc": "2.0", "id": message["id"], "result": [{"uri": "file:///tmp/example.py", "range": {"start": {"line": 0, "character": 4}}}]})
    elif method == "textDocument/hover":
        send({"jsonrpc": "2.0", "id": message["id"], "result": {"contents": {"value": "def greet(name):"}}})
    elif method == "textDocument/diagnostic":
        send({"jsonrpc": "2.0", "id": message["id"], "result": {"items": []}})
    elif method == "textDocument/documentSymbol":
        send({"jsonrpc": "2.0", "id": message["id"], "result": [{"name": "greet", "kind": 12, "range": {"start": {"line": 0, "character": 0}, "end": {"line": 1, "character": 0}}}]})
    elif "id" in message:
        send({"jsonrpc": "2.0", "id": message["id"], "result": None})
'''


def test_real_lsp_client_uses_configured_server(tmp_path):
    server = tmp_path / "server.py"
    server.write_text(FAKE_SERVER, encoding="utf-8")
    source = tmp_path / "example.py"
    source.write_text("def greet(name):\n    return name\n", encoding="utf-8")
    config = {
        "enabled": True,
        "timeout": 2,
        "servers": {
            "python": {
                "command": sys.executable,
                "args": [str(server)],
            }
        },
    }
    manager = LspManager(tmp_path, config)
    try:
        result = manager.request("goToDefinition", str(source), 1, 5, "", 20)
        assert result is not None
        assert result["metadata"]["source"] == "lsp"
        assert "example.py" in result["output"]
        hover = manager.request("hover", str(source), 1, 5, "", 20)
        assert hover is not None and "greet" in hover["output"]
        diagnostics = manager.request("diagnostics", str(source), 0, 0, "", 20)
        assert diagnostics is not None and "No diagnostics" in diagnostics["output"]
    finally:
        manager.close()


def test_lsp_config_roundtrip():
    from opencode_py.config import Config

    cfg = Config.from_dict(
        {
            "lsp": {
                "enabled": True,
                "servers": {"python": {"command": "pylsp"}},
                "timeout": 3,
            }
        }
    )
    assert cfg.lsp["enabled"] is True
    assert cfg.lsp["servers"]["python"]["command"] == "pylsp"
    assert cfg.as_dict()["lsp"]["timeout"] == 3

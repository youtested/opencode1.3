import json
import threading

from opencode_py.config import Config
from opencode_py.server import ServerState, make_server


def test_ide_context_endpoint_and_tool(tmp_path):
    cfg = Config()
    cfg.raw = {}
    state = _state(cfg, tmp_path)
    server = make_server(state, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    import urllib.request
    try:
        payload = {
            "path": "/tmp/demo.py",
            "language": "python",
            "line": 2,
            "content": "def demo():\n    return 1\n",
        }
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/v1/ide/context",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + state.token},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            assert response.status == 200
        assert state.ide_context["path"] == "/tmp/demo.py"
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def _state(cfg, tmp_path):
    return ServerState(cfg, tmp_path, token="test-token-1234567890")


def test_ide_tool_authenticates_from_server_state_file(tmp_path, monkeypatch):
    """The tool must find the running server's url+token with no env vars set."""
    import urllib.request

    from opencode_py.tools import ide

    cfg = Config()
    cfg.raw = {}
    state = _state(cfg, tmp_path)
    server = make_server(state, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "server.json").write_text(
        json.dumps({"url": f"http://127.0.0.1:{port}", "token": state.token}),
        encoding="utf-8",
    )
    monkeypatch.setattr(ide.GPath, "data", data_dir)
    monkeypatch.delenv("OPENCODE_SERVER_URL", raising=False)
    monkeypatch.delenv("OPENCODE_SERVER_TOKEN", raising=False)
    try:
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/v1/ide/context",
            data=json.dumps({"path": "/tmp/bridged.py", "language": "python", "line": 1,
                             "content": "x = 1\n"}).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer " + state.token},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            assert response.status == 200
        result = ide.tool().run({"operation": "context"})
        assert not result.get("error"), result
        assert "bridged.py" in result["output"]
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_ide_tool_reports_unreachable_server(tmp_path, monkeypatch):
    from opencode_py.tools import ide

    monkeypatch.setattr(ide.GPath, "data", tmp_path)
    monkeypatch.delenv("OPENCODE_SERVER_URL", raising=False)
    monkeypatch.delenv("OPENCODE_SERVER_TOKEN", raising=False)
    result = ide.tool().run({"operation": "context"})
    assert result.get("error"), result

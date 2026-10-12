import json
import socket
from pathlib import Path
from urllib.request import Request, urlopen

from opencode_py import server_control


def free_port():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def test_managed_server_start_status_and_stop(tmp_path, monkeypatch):
    state_file = tmp_path / "server.json"
    log_file = tmp_path / "server.log"
    monkeypatch.setattr(server_control, "_state_path", lambda: state_file)
    monkeypatch.setattr(server_control, "_log_path", lambda: log_file)
    state = None
    try:
        state = server_control.start_server(
            Path.cwd(), port=free_port(), token="test-token-123456789"
        )
        with urlopen(
            Request(
                f"{state['url']}/api/v1/info",
                headers={"Origin": "null", "Authorization": "Bearer test-token-123456789"},
            ),
            timeout=3,
        ) as response:
            assert response.headers.get("Access-Control-Allow-Origin") == "*"
        assert state_file.exists()
        assert server_control.server_status()["running"] is True
        assert server_control.start_server(Path.cwd(), port=state["port"])["pid"] == state["pid"]
        assert server_control.stop_server()["stopped"] is True
        assert not state_file.exists()
    finally:
        if state is not None:
            try:
                server_control.stop_server()
            except RuntimeError:
                pass


def test_token_survives_restart(tmp_path, monkeypatch):
    """A dead server must not invalidate the token clients (Acode plugin) already saved."""
    state_file = tmp_path / "server.json"
    log_file = tmp_path / "server.log"
    monkeypatch.setattr(server_control, "_state_path", lambda: state_file)
    monkeypatch.setattr(server_control, "_log_path", lambda: log_file)
    monkeypatch.delenv("OPENCODE_SERVER_TOKEN", raising=False)
    state_file.write_text(
        json.dumps(
            {
                "url": f"http://127.0.0.1:{free_port()}",
                "token": "kept-token-1234567890",
                "pid": 0,
            }
        ),
        encoding="utf-8",
    )
    started = None
    try:
        started = server_control.start_server(Path.cwd(), port=free_port())
        assert started["token"] == "kept-token-1234567890"
    finally:
        if started is not None:
            try:
                server_control.stop_server()
            except RuntimeError:
                pass


def test_stop_clears_malformed_state(tmp_path, monkeypatch):
    state_file = tmp_path / "server.json"
    state_file.write_text(json.dumps({"pid": 0}), encoding="utf-8")
    monkeypatch.setattr(server_control, "_state_path", lambda: state_file)
    assert server_control.stop_server() == {"running": False, "stopped": True}
    assert not state_file.exists()


def test_server_shutdown_endpoint_is_authenticated(tmp_path, monkeypatch):
    from test_server import TOKEN, request, running_server

    with running_server(tmp_path) as (state, address):
        status, _raw = request(address, "POST", "/api/v1/shutdown", {}, token=None)
        assert status == 401
        status, raw = request(address, "POST", "/api/v1/shutdown", {}, token=TOKEN)
        assert status == 202
        assert json.loads(raw)["stopping"] is True

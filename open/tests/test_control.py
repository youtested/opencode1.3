import http.client
import json
import socket
import threading
from pathlib import Path

from opencode_py import control, server_control
from opencode_py.sdk import OpenCodeControlClient


TOKEN = "control-token-123456789"


def free_port():
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def request(address, method, path, body=None, token=TOKEN):
    connection = http.client.HTTPConnection(address[0], address[1], timeout=10)
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    payload = None
    if body is not None:
        payload = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    connection.request(method, path, body=payload, headers=headers)
    response = connection.getresponse()
    raw = response.read()
    status = response.status
    connection.close()
    return status, raw


def test_shared_control_start_status_stop(tmp_path, monkeypatch):
    state_file = tmp_path / "server.json"
    log_file = tmp_path / "server.log"
    monkeypatch.setattr(server_control, "_state_path", lambda: state_file)
    monkeypatch.setattr(server_control, "_log_path", lambda: log_file)
    state = control.ControlState(
        token=TOKEN,
        host="127.0.0.1",
        port=0,
        coding_port=free_port(),
        coding_directory=Path.cwd(),
    )
    httpd = control.make_control_server(state, "127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        status, _body = request(httpd.server_address, "GET", "/api/v1/control/status", token=None)
        assert status == 401
        status, raw = request(httpd.server_address, "POST", "/api/v1/control/start", {})
        assert status == 200
        started = json.loads(raw)
        assert started["running"] is True
        assert started["url"].startswith("http://127.0.0.1:")
        status, raw = request(httpd.server_address, "GET", "/api/v1/control/status")
        assert status == 200
        assert json.loads(raw)["running"] is True
        client = OpenCodeControlClient(
            f"http://127.0.0.1:{httpd.server_address[1]}", TOKEN
        )
        assert client.status()["running"] is True
        status, raw = request(httpd.server_address, "POST", "/api/v1/control/stop", {})
        assert status == 200
        assert json.loads(raw)["running"] is False
        assert server_control.server_status()["running"] is False
    finally:
        try:
            server_control.stop_server()
        except RuntimeError:
            pass
        httpd.shutdown()
        thread.join(timeout=5)
        httpd.server_close()

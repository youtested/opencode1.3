import http.client
import json
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from opencode_py.config import Config
from opencode_py.sdk import OpenCodeClient
from opencode_py.server import ServerState, make_server
from opencode_py.session import Session


TOKEN = "test-token-123456789"


class MemoryStore:
    def __init__(self):
        self.saved = {}

    def new_session(self, **kwargs):
        session = Session(
            {
                "id": f"session-{len(self.saved) + 1}",
                "title": kwargs.get("title", ""),
                "created": time.time(),
                "directory": kwargs.get("directory", ""),
                "provider": kwargs.get("provider", ""),
                "model": kwargs.get("model", ""),
                "agent": kwargs.get("agent", "build"),
                "messages": [],
            }
        )
        self.saved[session.id] = session
        return session

    def save_session(self, session):
        self.saved[session.id] = session
        return Path(f"{session.id}.json")

    def list_sessions(self, directory):
        return [session for session in self.saved.values() if session.directory == directory]

    def load_session(self, session_id):
        return self.saved.get(session_id)


class FakeEngine:
    def __init__(self, callback):
        self.callback = callback
        self.interrupt_check = lambda: False
        self.history = []
        self.queued = []
        self.aborted = False

    @property
    def interrupt(self):
        return self.interrupt_check

    @interrupt.setter
    def interrupt(self, value):
        self.interrupt_check = value

    def run_turn(self, prompt):
        self.callback({"kind": "text_delta", "text": f"answer:{prompt}"})
        self.history = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"answer:{prompt}"},
        ]
        return SimpleNamespace(
            text=f"answer:{prompt}",
            reasoning="",
            tool_calls_made=0,
            usage=None,
            provider_id="test",
            model_id="test-model",
            finish_reason="stop",
            error="",
            network_failed=False,
        )

    def get_history(self):
        return list(self.history)

    def queue_prompt(self, prompt):
        self.queued.append(prompt)

    def abort(self):
        self.aborted = True

    def close(self):
        return None


@contextmanager
def running_server(tmp_path):
    cfg = Config()
    store = MemoryStore()
    state = ServerState(
        cfg,
        tmp_path,
        token=TOKEN,
        engine_factory=lambda session, callback: FakeEngine(callback),
        session_store=store,
    )
    httpd = make_server(state, "127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield state, httpd.server_address
    finally:
        httpd.shutdown()
        thread.join(timeout=5)
        httpd.server_close()
        state.close()


def request(address, method, path, body=None, token=TOKEN):
    connection = http.client.HTTPConnection(address[0], address[1], timeout=5)
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
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


def wait_for_session(address, session_id, token=TOKEN):
    for _ in range(50):
        status, raw = request(address, "GET", f"/api/v1/sessions/{session_id}", token=token)
        if status == 200:
            data = json.loads(raw)
            if not data.get("runtime", {}).get("active"):
                return data
        time.sleep(0.01)
    raise AssertionError("session did not finish")


def test_health_and_web_are_public_but_api_requires_token(tmp_path):
    with running_server(tmp_path) as (_state, address):
        status, _body = request(address, "GET", "/api/v1/health", token=None)
        assert status == 200
        status, body = request(address, "GET", "/")
        assert status == 200
        assert b"OpenCode" in body
        status, _body = request(address, "GET", "/api/v1/info", token=None)
        assert status == 401


def test_session_message_stream_and_history(tmp_path):
    with running_server(tmp_path) as (_state, address):
        status, raw = request(address, "POST", "/api/v1/sessions", {"title": "test"})
        assert status == 201
        session_id = json.loads(raw)["id"]
        status, raw = request(
            address,
            "POST",
            f"/api/v1/sessions/{session_id}/messages",
            {"content": "hello"},
        )
        assert status == 202
        data = wait_for_session(address, session_id)
        assert data["messages"][-1]["content"] == "answer:hello"
        assert data["runtime"]["active"] is False
        status, raw = request(address, "GET", "/api/v1/sessions")
        assert status == 200
        assert any(item["id"] == session_id for item in json.loads(raw)["sessions"])


def test_sse_replays_bounded_event_history(tmp_path):
    with running_server(tmp_path) as (_state, address):
        _status, raw = request(address, "POST", "/api/v1/sessions", {})
        session_id = json.loads(raw)["id"]
        request(address, "POST", f"/api/v1/sessions/{session_id}/messages", {"content": "stream"})
        wait_for_session(address, session_id)
        connection = http.client.HTTPConnection(address[0], address[1], timeout=5)
        connection.request(
            "GET",
            f"/api/v1/sessions/{session_id}/events?after=0",
            headers={"Authorization": f"Bearer {TOKEN}"},
        )
        response = connection.getresponse()
        kinds = []
        for _ in range(30):
            line = response.readline()
            if not line:
                break
            text = line.decode("utf-8", errors="replace").strip()
            if text.startswith("data:"):
                kinds.append(json.loads(text[5:].strip())["kind"])
            if "turn_complete" in kinds:
                break
        connection.close()
        assert "text_delta" in kinds
        assert "turn_complete" in kinds


def test_sdk_uses_the_same_api(tmp_path):
    with running_server(tmp_path) as (_state, address):
        client = OpenCodeClient(f"http://{address[0]}:{address[1]}", TOKEN)
        assert client.health()["status"] == "ok"
        session = client.create_session(title="sdk")
        client.send_message(session["id"], "from sdk")
        deadline = time.time() + 3
        while time.time() < deadline:
            current = client.get_session(session["id"])
            if not current["runtime"]["active"]:
                break
            time.sleep(0.01)
        assert current["messages"][-1]["content"] == "answer:from sdk"


def test_server_config_does_not_persist_token(tmp_path):
    cfg = Config.from_dict(
        {
            "server": {
                "host": "0.0.0.0",
                "port": 4321,
                "cors": ["https://example.test"],
                "token": "must-not-leak",
            }
        }
    )
    assert cfg.server == {"host": "0.0.0.0", "port": 4321, "cors": ["https://example.test"]}
    assert "token" not in json.dumps(cfg.as_dict())

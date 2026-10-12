"""Tests for the optional Tor rotation module and its use by websearch.

Fully offline: the port probe is injected and curl is faked, so nothing here
opens a socket or runs a real request.
"""
from __future__ import annotations

import types

import pytest

from opencode_py.tools import proxy_rotation as PR
from opencode_py.tools import websearch as W


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.delenv("OPENCODE_TOR", raising=False)
    PR.TorRotation._shared = None
    yield
    PR.TorRotation._shared = None


# --------------------------------------------------------------------------- #
# detection
# --------------------------------------------------------------------------- #
def test_not_available_when_nothing_is_listening(monkeypatch):
    monkeypatch.setattr(PR, "_port_open", lambda *a, **k: False)
    r = PR.TorRotation()
    assert r.available() is False
    assert r.port is None
    assert r.proxy_url() is None
    assert r.curl_args() == []


def test_detects_a_listening_port(monkeypatch):
    monkeypatch.setattr(PR, "_port_open", lambda port, **k: port == 9150)
    r = PR.TorRotation()
    assert r.available() is True
    assert r.port == 9150
    assert r.proxy_url() == "socks5://127.0.0.1:9150"


def test_probe_is_cached_while_down(monkeypatch):
    calls = []

    def probe(port, **k):
        calls.append(port)
        return False

    monkeypatch.setattr(PR, "_port_open", probe)
    r = PR.TorRotation(ttl_down=90.0)
    for _ in range(10):
        r.available()
    assert len(calls) == len(PR.DEFAULT_PORTS)  # one probe pass only


def test_probe_is_cached_while_up(monkeypatch):
    calls = []

    def probe(port, **k):
        calls.append(port)
        return True

    monkeypatch.setattr(PR, "_port_open", probe)
    r = PR.TorRotation(ttl_up=600.0)
    for _ in range(10):
        r.available()
    assert len(calls) == 1  # stopped at the first open port, then cached


def test_env_disable_wins(monkeypatch):
    monkeypatch.setenv("OPENCODE_TOR", "0")
    monkeypatch.setattr(PR, "_port_open", lambda *a, **k: True)
    r = PR.TorRotation()
    assert r.available() is False


def test_shared_is_a_singleton(monkeypatch):
    monkeypatch.setattr(PR, "_port_open", lambda *a, **k: False)
    assert PR.TorRotation.shared() is PR.TorRotation.shared()


# --------------------------------------------------------------------------- #
# curl args
# --------------------------------------------------------------------------- #
def test_curl_args_use_socks5_hostname_so_dns_stays_inside_tor(monkeypatch):
    """--socks5 would leak DNS lookups to the local resolver."""
    monkeypatch.setattr(PR, "_port_open", lambda *a, **k: True)
    args = PR.TorRotation().curl_args()
    assert args == ["--socks5-hostname", "127.0.0.1:9050"]


def test_curl_args_empty_without_tor(monkeypatch):
    monkeypatch.setattr(PR, "_port_open", lambda *a, **k: False)
    assert PR.TorRotation().curl_args() == []


# --------------------------------------------------------------------------- #
# failure handling
# --------------------------------------------------------------------------- #
def test_note_failure_retires_tor_immediately(monkeypatch):
    monkeypatch.setattr(PR, "_port_open", lambda *a, **k: True)
    r = PR.TorRotation()
    assert r.available() is True
    r.note_failure()
    assert r.available() is False


def test_fetch_without_tor_is_a_clean_no_op(monkeypatch):
    monkeypatch.setattr(PR, "_port_open", lambda *a, **k: False)
    body, err = PR.TorRotation().fetch("https://example.com")
    assert body is None
    assert err == "tor unavailable"


def test_fetch_uses_curl_with_socks_and_returns_body(monkeypatch):
    monkeypatch.setattr(PR, "_port_open", lambda *a, **k: True)
    monkeypatch.setattr(PR.shutil, "which", lambda name: "/usr/bin/curl")
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return types.SimpleNamespace(stdout="hello", stderr="", returncode=0)

    monkeypatch.setattr(PR.subprocess, "run", fake_run)
    body, err = PR.TorRotation().fetch("https://example.com")
    assert body == "hello"
    assert err is None
    assert "--socks5-hostname" in seen["cmd"]
    assert seen["cmd"][-1] == "https://example.com"


def test_fetch_failure_retires_tor(monkeypatch):
    monkeypatch.setattr(PR, "_port_open", lambda *a, **k: True)
    monkeypatch.setattr(PR.shutil, "which", lambda name: "/usr/bin/curl")
    monkeypatch.setattr(
        PR.subprocess, "run",
        lambda cmd, **kw: types.SimpleNamespace(stdout="", stderr="boom", returncode=7))
    r = PR.TorRotation()
    body, err = r.fetch("https://example.com")
    assert body is None
    assert err
    assert r.available() is False


# --------------------------------------------------------------------------- #
# websearch integration
# --------------------------------------------------------------------------- #
def _install_fake_rotation(monkeypatch, available, body="PAGE"):
    calls = []

    class FakeRot:
        @staticmethod
        def shared():
            return FakeRot()

        def ensure(self, budget):
            return available

        def available(self):
            return available

        def fetch(self, url, timeout=20.0, extra_args=None):
            calls.append(url)
            return (body, None) if body else (None, "tor: dead")

    mod = types.SimpleNamespace(TorRotation=FakeRot)
    monkeypatch.setitem(__import__("sys").modules,
                        "opencode_py.tools.proxy_rotation", mod)
    return calls


def test_tor_ignored_when_not_running(monkeypatch):
    calls = _install_fake_rotation(monkeypatch, available=False)
    body, err = W._tor_fetch(W._Ctx(20), "https://x.com", lambda b: True)
    assert body is None
    assert err == "tor not running"
    assert calls == []          # never even attempted a fetch


def test_tor_fetch_used_when_running(monkeypatch):
    calls = _install_fake_rotation(monkeypatch, available=True, body="RESULTS")
    body, err = W._tor_fetch(W._Ctx(30), "https://x.com", lambda b: "RESULT" in b)
    assert body == "RESULTS"
    assert err is None
    assert calls == ["https://x.com"]


def test_tor_rejects_an_unusable_page(monkeypatch):
    _install_fake_rotation(monkeypatch, available=True, body="<html>shell</html>")
    body, err = W._tor_fetch(W._Ctx(30), "https://x.com", lambda b: "RESULT" in b)
    assert body is None
    assert "unusable" in err


def test_tor_skipped_when_budget_is_gone(monkeypatch):
    calls = _install_fake_rotation(monkeypatch, available=True)
    ctx = W._Ctx(20)
    ctx.deadline = __import__("time").monotonic() + 1  # < 3s left
    body, err = W._tor_fetch(ctx, "https://x.com", lambda b: True)
    assert body is None
    assert calls == []


def test_tor_is_last_resort_after_the_bypass_cascade(monkeypatch):
    """Order matters: cheap client-fingerprint fix first, Tor only after."""
    order = []

    def bypass(ctx, url, validate=None):
        order.append("bypass")
        return None, "blocked"

    def tor(ctx, url, validate=None):
        order.append("tor")
        return "FROM_TOR", None

    monkeypatch.setattr(W, "_bypass_fetch", bypass)
    monkeypatch.setattr(W, "_tor_fetch", tor)

    class Resp:
        status_code = 429
        headers: dict = {}

        def iter_text(self):
            yield ""

    import httpx

    class S:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return Resp()

        def __exit__(self, *a):
            return False

    class C:
        def __init__(self, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def stream(self, *a, **k):
            return S()

    monkeypatch.setattr(httpx, "Client", C)
    body, err = W._request(W._Ctx(30), "https://x.com", retries=0,
                           validate=lambda b: b == "FROM_TOR")
    assert body == "FROM_TOR"
    assert order == ["bypass", "tor"]


def test_tor_never_receives_a_keyed_request(monkeypatch):
    """A request carrying an API key must not be replayed through Tor."""
    called = []

    monkeypatch.setattr(W, "_bypass_fetch",
                        lambda ctx, url, validate=None: pytest.fail("no bypass either"))
    monkeypatch.setattr(W, "_tor_fetch",
                        lambda ctx, url, validate=None: pytest.fail("no tor"))
    got, err = W._request(W._Ctx(30), "https://api.exa.ai/search", method="POST",
                          json_body={}, headers={"x-api-key": "secret"}, retries=0)
    assert got is None
    assert err
    assert called == []


# --------------------------------------------------------------------------- #
# on-demand lifecycle
# --------------------------------------------------------------------------- #
def test_ensure_does_not_start_when_budget_is_too_small(monkeypatch):
    """A slow bootstrap inside an expiring deadline helps nobody."""
    started = []
    monkeypatch.setattr(PR.TorRotation, "start", lambda self, budget=25.0: started.append(1))
    r = PR.TorRotation()
    assert r.ensure(PR.MIN_START_BUDGET - 1) is False
    assert started == []


def test_ensure_returns_false_when_tor_not_installed(monkeypatch):
    monkeypatch.setattr(PR, "_port_open", lambda *a, **k: False)
    monkeypatch.setattr(PR.TorRotation, "binary", staticmethod(lambda: None))
    assert PR.TorRotation().ensure(60.0) is False


def test_ensure_does_not_start_when_autostart_disabled(monkeypatch):
    monkeypatch.delenv("OPENCODE_TOR_AUTOSTART", raising=False)
    monkeypatch.setenv("OPENCODE_TOR_AUTOSTART", "0")
    monkeypatch.setattr(PR, "_port_open", lambda *a, **k: False)
    monkeypatch.setattr(PR.TorRotation, "binary", staticmethod(lambda: "/x/tor"))
    started = []
    monkeypatch.setattr(PR.TorRotation, "start", lambda self, budget=25.0: started.append(1))
    assert PR.TorRotation().ensure(60.0) is False
    assert started == []


def test_ensure_reuses_an_already_running_proxy(monkeypatch):
    """A tor the user started themselves is used, never restarted."""
    monkeypatch.setattr(PR, "_port_open", lambda *a, **k: True)
    r = PR.TorRotation()
    started = []
    monkeypatch.setattr(r, "start", lambda budget=25.0: started.append(1))
    assert r.ensure(60.0) is True
    assert started == []


def test_reap_never_touches_a_tor_we_do_not_own(monkeypatch):
    """The rule that matters: only our own process is ever stopped."""
    r = PR.TorRotation()
    assert r.reap() is False
    assert r.stop() is False


def test_reap_stops_an_owned_tor_after_idle(monkeypatch):
    r = PR.TorRotation()
    stopped = []
    monkeypatch.setattr(r, "stop", lambda: stopped.append(1) or True)
    r._owned_pid = 999
    r._last_used = __import__("time").monotonic()
    assert r.reap() is False          # not idle yet
    r._last_used -= PR.AUTO_STOP_AFTER + 1
    assert r.reap() is True          # idle long enough
    assert stopped == [1]


def test_reap_respects_autostop_off(monkeypatch):
    monkeypatch.setenv("OPENCODE_TOR_AUTOSTOP", "0")
    r = PR.TorRotation()
    stopped = []
    monkeypatch.setattr(r, "stop", lambda: stopped.append(1) or True)
    r._owned_pid = 999
    r._last_used -= PR.AUTO_STOP_AFTER + 1
    assert r.reap() is False
    assert stopped == []


def test_start_waits_for_the_port_and_claims_the_pid(monkeypatch):
    monkeypatch.setattr(PR.TorRotation, "binary", staticmethod(lambda: "/x/tor"))
    monkeypatch.setattr(PR.TorRotation, "data_dir", staticmethod(lambda: "/tmp/tor"))
    states = iter([False, False, True])

    monkeypatch.setattr(PR, "_port_open", lambda *a, **k: next(states))
    monkeypatch.setattr(PR.time, "sleep", lambda s: None)

    class FakeProc:
        pid = 4242

        def poll(self):
            return None

    proc = FakeProc()
    monkeypatch.setattr(PR.subprocess, "Popen", lambda *a, **k: proc)
    r = PR.TorRotation()
    assert r.start(budget=5.0) is True
    assert r._owned_pid == 4242
    assert r.available() is True


def test_start_gives_up_when_tor_dies_during_bootstrap(monkeypatch):
    monkeypatch.setattr(PR.TorRotation, "binary", staticmethod(lambda: "/x/tor"))
    monkeypatch.setattr(PR.TorRotation, "data_dir", staticmethod(lambda: "/tmp/tor"))
    monkeypatch.setattr(PR, "_port_open", lambda *a, **k: False)

    class DeadProc:
        pid = 1

        def poll(self):
            return 1

    monkeypatch.setattr(PR.subprocess, "Popen", lambda *a, **k: DeadProc())
    r = PR.TorRotation()
    assert r.start(budget=5.0) is False
    assert r._owned_pid is None


def test_start_is_a_no_op_without_the_binary(monkeypatch):
    monkeypatch.setattr(PR.TorRotation, "binary", staticmethod(lambda: None))
    assert PR.TorRotation().start() is False


def test_websearch_budget_gate_prevents_a_start(monkeypatch):
    """websearch must not spin up a relay it cannot wait for."""
    _install_fake_rotation(monkeypatch, available=False)
    ctx = W._Ctx(20)
    ctx.deadline = __import__("time").monotonic() + 5   # < MIN_TOR_BUDGET
    body, err = W._tor_fetch(ctx, "https://x.com", lambda b: True)
    assert body is None
    assert err == "tor not running"

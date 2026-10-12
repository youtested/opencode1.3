from opencode_py import server_control
from opencode_py.config import Config
from opencode_py.tui.settings_screen import SettingsScreen


def test_settings_server_row_uses_shared_controller(tmp_path, monkeypatch):
    calls = []

    def status():
        return {"running": False, "url": "off", "token": ""}

    def start(directory, **kwargs):
        calls.append(("start", directory, kwargs))
        return {"url": "http://127.0.0.1:4096", "token": "test-token"}

    def stop():
        calls.append(("stop",))
        return {"stopped": True}

    monkeypatch.setattr(server_control, "server_status", status)
    monkeypatch.setattr(server_control, "start_server", start)
    monkeypatch.setattr(server_control, "stop_server", stop)
    screen = SettingsScreen(
        cfg=Config(),
        engine=None,
        auth=None,
        directory=str(tmp_path),
    )
    rows = screen._build_rows()
    row = next(item for item in rows if item.label == "server on/off")
    assert row.kind == "bool"
    assert row.get() == "no"
    row.apply("yes")
    assert calls[0][0] == "start"
    assert calls[0][1] == str(tmp_path)
    row.apply("no")
    assert calls[-1] == ("stop",)
    assert any(item.label == "server address" for item in rows)
    assert any(item.label == "server token" for item in rows)


def test_ctrl_s_still_opens_settings():
    from opencode_py.tui.app import OpenCodeTUI

    bindings = getattr(OpenCodeTUI, "BINDINGS", [])
    ctrl_s = [binding for binding in bindings if getattr(binding, "key", "") == "ctrl+s"]
    assert ctrl_s
    assert getattr(ctrl_s[0], "action", "") == "settings"

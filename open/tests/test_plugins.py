import json
import time

from opencode_py.plugins import PluginDispatcher, reset_dispatcher


def test_empty_dispatcher_is_free():
    d = PluginDispatcher(None)
    assert d.count == 0
    assert d.names() == []
    t = time.perf_counter()
    for _ in range(20000):
        d.has("tool.before")
    per_call_us = (time.perf_counter() - t) / 20000 * 1e6
    assert per_call_us < 50
    assert d.emit("tool.before", {"tool": "x"}) == []


def test_python_hook_file(tmp_path):
    hook = tmp_path / "h.py"
    seen = tmp_path / "seen.json"
    hook.write_text(
        "import json, pathlib\n"
        "def hook(event, payload):\n"
        f"    pathlib.Path({str(seen)!r}).write_text(json.dumps({{'event': event, 'tool': payload.get('tool')}}))\n"
        "    return {'note': 'ok-' + event}\n",
        encoding="utf-8",
    )
    d = PluginDispatcher([{"name": "h", "target": str(hook), "events": ["file.changed"], "timeout": 2}])
    assert d.has("file.changed")
    assert not d.has("tool.before")
    out = d.emit("file.changed", {"tool": "write", "filePath": "/tmp/a.py"})
    assert out and out[0].get("note") == "ok-file.changed"
    assert json.loads(seen.read_text())["event"] == "file.changed"


def test_python_hooks_map(tmp_path):
    hook = tmp_path / "hooks.py"
    hook.write_text(
        "def only_after(event, payload):\n"
        "    return {'hit': 'after'}\n"
        "HOOKS = {'tool.after': only_after}\n",
        encoding="utf-8",
    )
    d = PluginDispatcher([{"name": "h", "target": str(hook), "events": ["tool.after"]}])
    assert d.emit("tool.after", {"tool": "edit"})[0]["hit"] == "after"
    assert d.emit("tool.before", {"tool": "edit"}) == []


def test_command_hook(tmp_path):
    out = tmp_path / "out.json"
    d = PluginDispatcher([
        {"name": "c", "target": f"printf '%s' \"$OPENCODE_HOOK_JSON\" > {out}", "events": ["session.end"], "timeout": 2}
    ])
    d.emit("session.end", {"sessionId": "s1"})
    assert json.loads(out.read_text())["sessionId"] == "s1"


def test_js_hook_if_node_present(tmp_path):
    node = None
    for name in ("node", "nodejs"):
        from shutil import which
        node = which(name)
        if node:
            break
    if not node:
        return
    script = tmp_path / "h.mjs"
    script.write_text(
        "let raw='';process.stdin.on('data',d=>raw+=d);"
        "process.stdin.on('end',()=>{const p=JSON.parse(raw||'{}');"
        "process.stdout.write(JSON.stringify({ok:true,tool:p.tool}));});\n",
        encoding="utf-8",
    )
    d = PluginDispatcher([{"name": "js", "target": str(script), "events": ["file.changed"], "timeout": 5}])
    out = d.emit("file.changed", {"tool": "edit"})
    assert out and out[0].get("ok") is True
    assert out[0].get("tool") == "edit"


def test_broken_hook_is_isolated(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("raise RuntimeError('boom')\n", encoding="utf-8")
    d = PluginDispatcher([{"name": "bad", "target": str(bad), "events": ["tool.before"], "timeout": 1}])
    out = d.emit("tool.before", {"tool": "x"})
    assert out and out[0].get("error") is True
    assert "boom" in out[0]["output"]


def test_event_filter_and_timeout_cap(tmp_path):
    d = PluginDispatcher([
        {"name": "a", "target": "./a.py", "events": ["tool.after"], "timeout": 999},
        {"name": "b", "target": "./b.py", "events": ["nope"], "timeout": 0},
    ])
    assert d.names() == ["a"]
    assert d.has("tool.after")
    assert not d.has("file.changed")


def test_reset_dispatcher_returns_clean_state():
    reset_dispatcher()
    from opencode_py.plugins import get_dispatcher
    assert get_dispatcher(None).count == 0

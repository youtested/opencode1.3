import json
from pathlib import Path

from opencode_py.agent.loop import AgentLoop
from opencode_py.config import Config
from opencode_py.plugins import PluginDispatcher, reset_dispatcher
from opencode_py.tools.registry import Registry, Tool


def _loop(registry):
    cfg = Config()
    return AgentLoop(cfg=cfg, registry=registry, directory=Path("."), provider=None)


def test_tool_hooks_fire_around_a_real_tool_run(tmp_path):
    log = tmp_path / "log.json"
    hook = tmp_path / "h.py"
    hook.write_text(
        "import json, pathlib\n"
        "LOG = pathlib.Path(" + repr(str(log)) + ")\n"
        "def hook(event, payload):\n"
        "    rows = json.loads(LOG.read_text()) if LOG.exists() else []\n"
        "    rows.append({'event': event, 'tool': payload.get('tool')})\n"
        "    LOG.write_text(json.dumps(rows))\n"
        "    return None\n",
        encoding="utf-8",
    )
    reset_dispatcher()
    from opencode_py import plugins as plugins_mod
    plugins_mod._GLOBAL = PluginDispatcher([
        {"name": "h", "target": str(hook), "events": ["tool.before", "tool.after", "file.changed"], "timeout": 2}
    ])

    registry = Registry()
    registry.register(Tool(
        name="write",
        description="test writer",
        parameters={"type": "object", "properties": {"filePath": {"type": "string"}, "content": {"type": "string"}}},
        run=lambda args: {"output": "wrote " + str(args.get("filePath"))},
    ))
    target = tmp_path / "out.txt"
    loop = _loop(registry)
    try:
        loop.run_tool("write", {"filePath": str(target), "content": "hello"})
    finally:
        try:
            loop.close()
        except Exception:
            pass
        reset_dispatcher()
    rows = json.loads(log.read_text())
    events = [r["event"] for r in rows]
    assert "tool.before" in events
    assert "tool.after" in events
    assert all(r["tool"] == "write" for r in rows)

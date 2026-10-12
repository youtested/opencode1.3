"""Regression: custom slash commands must submit a real turn (no UnboundLocalError)."""
import os
import json

from opencode_py.commands import build_registry, CUSTOM_COMMANDS


def test_custom_command_handler_submits_prompt_with_extra(monkeypatch):
    os.environ["OPENCODE_CONFIG_CONTENT"] = json.dumps({
        "commands": {"hi": {"description": "say hi", "prompt": "say hellow"}}
    })
    CUSTOM_COMMANDS.clear()
    reg = build_registry()
    cmd = reg.get("hi")
    assert cmd is not None

    sent = []

    class Ctx:
        pass

    ctx = Ctx()
    ctx.on_run_prompt = lambda text: sent.append(text)
    cmd.handler(ctx, "so")
    assert sent == ["say hellow\n\nso"]


def test_custom_command_handler_without_extra(monkeypatch):
    os.environ["OPENCODE_CONFIG_CONTENT"] = json.dumps({
        "commands": {"hi": {"description": "say hi", "prompt": "say hellow"}}
    })
    CUSTOM_COMMANDS.clear()
    reg = build_registry()
    cmd = reg.get("hi")
    sent = []

    class Ctx:
        pass

    ctx = Ctx()
    ctx.on_run_prompt = lambda text: sent.append(text)
    cmd.handler(ctx, "")
    assert sent == ["say hellow"]

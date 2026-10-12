"""Custom slash commands must submit directly, never open the centered popup."""
import json
import os

import pytest
from textual.app import App, ComposeResult

from opencode_py.commands import build_registry, CUSTOM_COMMANDS
from opencode_py.tui.input_bar import InputBar


@pytest.mark.asyncio
async def test_custom_command_is_marked_in_dropdown():
    os.environ["OPENCODE_CONFIG_CONTENT"] = json.dumps({
        "commands": {"hi": {"description": "say hi", "prompt": "say hellow"}}
    })
    CUSTOM_COMMANDS.clear()
    build_registry()
    bar = InputBar(commands=[{"name": "hi", "description": "say hi", "custom": True}])
    assert bar._is_custom("hi") is True
    assert bar._is_custom("mcp") is False


@pytest.mark.asyncio
async def test_custom_command_submits_without_popup():
    os.environ["OPENCODE_CONFIG_CONTENT"] = json.dumps({
        "commands": {"hi": {"description": "say hi", "prompt": "say hellow"}}
    })
    CUSTOM_COMMANDS.clear()
    build_registry()

    posted = []

    class Harness(App):
        def compose(self) -> ComposeResult:
            yield InputBar(commands=[{"name": "hi", "description": "say hi", "custom": True}])

    app = Harness()
    async with app.run_test() as pilot:
        bar = app.query_one(InputBar)
        posted.append(bar)
        bar.on_prompt_submitted(
            type("E", (), {"value": "/hi", "stop": lambda self: None})()
        )
        await pilot.pause()
        # popup must NOT have been triggered; the bar cleared and let the event pass
        assert bar.input.value == ""

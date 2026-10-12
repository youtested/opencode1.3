import json

import pytest
from textual.app import App
from textual.widgets import Button

from opencode_py.config import Config
from opencode_py.plugin_manager import (
    list_disabled,
    list_hooks,
    list_plugins,
    remove,
    save_hook,
    save_plugin,
    set_enabled,
)
from opencode_py.tui.plugin_picker import PluginAddDialog, PluginPicker


def test_save_list_remove_plugin(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    save_plugin("tools", "./tools/demo.py", str(tmp_path), False)
    cfg = Config(); cfg.raw = json.loads((tmp_path / "opencode.json").read_text())
    assert list_plugins(cfg) == ["./tools/demo.py"]
    assert remove("tools", str(tmp_path), False) is True
    cfg.raw = json.loads((tmp_path / "opencode.json").read_text())
    assert list_plugins(cfg) == []


def test_save_and_toggle_hook(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    save_hook("fmt", "./hooks/f.py", ["file.changed"], str(tmp_path), False)
    cfg = Config(); cfg.raw = json.loads((tmp_path / "opencode.json").read_text())
    hooks = list_hooks(cfg)
    assert hooks and hooks[0]["name"] == "fmt"
    assert hooks[0]["events"] == ["file.changed"]
    assert set_enabled("fmt", False, str(tmp_path), False) is True
    cfg.raw = json.loads((tmp_path / "opencode.json").read_text())
    assert "hook:fmt" in list_disabled(cfg)
    assert set_enabled("fmt", True, str(tmp_path), False) is True


class PickerHarness(App):
    def on_mount(self) -> None:
        self.push_screen(PluginPicker(
            plugins=["./tools/demo.py"],
            hooks=[{"name": "fmt", "target": "./hooks/f.py", "events": ["file.changed"]}],
            disabled=set(),
            on_result=lambda a: None,
        ))


@pytest.mark.asyncio
async def test_plugin_picker_lists_tools_and_hooks():
    app = PickerHarness()
    async with app.run_test() as pilot:
        await pilot.pause()
        rows = [str(o.prompt) for o in app.screen.query_one("#plugin-picker-list").options]
        assert any("demo.py" in r for r in rows)
        assert any("fmt" in r for r in rows)
        assert any("Add tool plugin" in r for r in rows)
        assert any("Add event hook" in r for r in rows)


@pytest.mark.asyncio
async def test_plugin_picker_buttons_fit_one_row():
    app = PickerHarness()
    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        row = [app.screen.query_one("#" + i, Button).region
                for i in ("plugin-add-tool", "plugin-add-hook", "plugin-test", "plugin-toggle", "plugin-close")]
        assert all(r.y == row[0].y for r in row)
        for left, right in zip(row, row[1:]):
            assert left.x + left.width <= right.x + 1


@pytest.mark.asyncio
async def test_plugin_add_dialog_hook_has_events_field():
    class H(App):
        def on_mount(self) -> None:
            self.push_screen(PluginAddDialog(kind="hook"))

    app = H()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert len(app.screen.query("#plugin-add-events")) == 1
        assert len(app.screen.query("#plugin-add-target")) == 1

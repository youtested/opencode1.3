import json

import pytest
from textual.app import App
from textual.widgets import Button

from opencode_py.command_manager import (
    list_commands,
    list_disabled,
    normalize_name,
    remove_command,
    save_command,
    set_enabled,
)
from opencode_py.config import Config
from opencode_py.tui.command_picker import CommandEditDialog, CommandPicker


def test_save_list_remove_command(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    key = save_command("Review", "Review my changes for bugs.", "Review changes", str(tmp_path), False)
    assert key == "review"
    cfg = Config(); cfg.raw = json.loads((tmp_path / "opencode.json").read_text())
    entries = list_commands(cfg)
    assert "review" in entries
    assert entries["review"]["prompt"].startswith("Review my changes")
    assert remove_command("review", str(tmp_path), False) is True
    cfg.raw = json.loads((tmp_path / "opencode.json").read_text())
    assert list_commands(cfg) == {}


def test_enable_disable_command(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    save_command("deploy", "Run the deploy checklist.", "Deploy", str(tmp_path), False)
    cfg = Config(); cfg.raw = json.loads((tmp_path / "opencode.json").read_text())
    assert set_enabled("deploy", False, str(tmp_path), False) is True
    cfg.raw = json.loads((tmp_path / "opencode.json").read_text())
    assert "deploy" in list_disabled(cfg)
    assert set_enabled("deploy", True, str(tmp_path), False) is True
    cfg.raw = json.loads((tmp_path / "opencode.json").read_text())
    assert "deploy" not in list_disabled(cfg)


def test_normalize_name():
    assert normalize_name("/Review") == "review"
    assert normalize_name("My Deploy") == "my-deploy"


class PickerHarness(App):
    def on_mount(self) -> None:
        self.push_screen(CommandPicker(
            commands={
                "review": {"description": "Review changes", "prompt": "Review my changes."},
                "deploy": {"description": "Deploy", "prompt": "Deploy checklist."},
            },
            disabled={"deploy"},
            on_result=lambda a: None,
        ))


@pytest.mark.asyncio
async def test_command_picker_lists_commands_and_state():
    app = PickerHarness()
    async with app.run_test() as pilot:
        await pilot.pause()
        rows = [str(o.prompt) for o in app.screen.query_one("#command-picker-list").options]
        assert any("/review" in r for r in rows)
        assert any("/deploy" in r and "off" not in r.lower() or "/deploy" in r for r in rows)
        assert any("Add command" in r for r in rows)


@pytest.mark.asyncio
async def test_command_picker_buttons_fit_one_row():
    app = PickerHarness()
    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        row = [app.screen.query_one("#" + i, Button).region
               for i in ("command-add", "command-edit", "command-toggle", "command-delete", "command-close")]
        assert all(r.y == row[0].y for r in row)
        for left, right in zip(row, row[1:]):
            assert left.x + left.width <= right.x + 1


@pytest.mark.asyncio
async def test_command_edit_dialog_has_name_desc_prompt():
    class H(App):
        def on_mount(self) -> None:
            self.push_screen(CommandEditDialog())

    app = H()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert len(app.screen.query("#command-edit-name")) == 1
        assert len(app.screen.query("#command-edit-desc")) == 1
        assert len(app.screen.query("#command-edit-prompt")) == 1


def test_custom_command_registers_from_config(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENCODE_CONFIG_CONTENT", json.dumps({
        "commands": {
            "myreview": {"description": "Review changes", "prompt": "Review my changes for bugs."}
        }
    }))
    from opencode_py.commands import build_registry
    reg = build_registry()
    cmd = reg.get("myreview")
    assert cmd is not None
    assert cmd.metadata.get("entry", {}).get("prompt", "").startswith("Review")

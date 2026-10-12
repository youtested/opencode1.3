"""Custom slash-command manager popup in /agent picker style."""
from __future__ import annotations

from ..util.log import swallow
from typing import Any

from rich.markup import escape
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.events import Key
from textual.screen import ModalScreen
from textual.widgets import Button, Input, OptionList, Static
from textual.widgets.option_list import Option

_COMMAND_PICKER_CSS = """
CommandPicker {
    align: center middle;
}
#command-picker-box {
    width: 84;
    max-width: 99%;
    max-height: 90%;
    height: auto;
    background: $surface;
    border: round $accent;
    padding: 1 2;
}
#command-picker-title {
    text-style: bold;
    color: $accent;
    height: 1;
    margin-bottom: 1;
}
#command-picker-hint {
    color: $text-muted;
    height: 1;
    margin-top: 1;
}
#command-picker-list {
    height: auto;
    max-height: 18;
    background: transparent;
    border: none;
    padding: 0;
    scrollbar-size-vertical: 1;
    scrollbar-size-horizontal: 0;
}
#command-picker-actions {
    height: 3;
    width: 100%;
    align: center middle;
    background: $surface;
    padding: 0 1;
    margin-top: 1;
}
#command-picker-actions Button {
    height: 3;
    min-width: 8;
    width: 1fr;
    max-width: 22;
    padding: 0 1;
    margin: 0 1;
    border: heavy $accent;
    background: transparent;
    color: $text;
    text-align: center;
}
"""


class CommandEditDialog(ModalScreen[tuple[str, str, str] | None]):
    """Add/edit a custom command: name, description, prompt."""

    CSS = """
    CommandEditDialog {
        align: center middle;
    }
    #command-edit-box {
        width: 66;
        max-width: 96%;
        height: auto;
        background: $surface;
        border: round $accent;
        padding: 1 2;
    }
    #command-edit-title {
        text-style: bold;
        color: $accent;
        height: 1;
        margin-bottom: 1;
    }
    #command-edit-box Input, #command-edit-box TextArea {
        margin-bottom: 1;
        border: none;
        background: $background;
        padding: 0 1;
    }
    #command-edit-actions {
        height: 3;
        align: center middle;
        background: $surface;
    }
    #command-edit-actions Button {
        height: 3;
        min-width: 12;
        padding: 0 2;
        margin: 0 1;
        border: heavy $accent;
        background: transparent;
        color: $text;
    }
    """

    def __init__(self, name: str = "", description: str = "", prompt: str = "", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._name = name
        self._description = description
        self._prompt = prompt
        self._editing = bool(name)

    def compose(self) -> ComposeResult:
        title = "Edit command" if self._editing else "Add command"
        with Vertical(id="command-edit-box"):
            yield Static(f"  {title}  ", id="command-edit-title")
            yield Input(
                value=self._name,
                placeholder="Name, e.g. review",
                id="command-edit-name",
                disabled=self._editing,
            )
            yield Input(
                value=self._description,
                placeholder="Short description, e.g. Review my changes",
                id="command-edit-desc",
            )
            yield Input(
                value=self._prompt,
                placeholder="What should this command do?",
                id="command-edit-prompt",
            )
            with Horizontal(id="command-edit-actions"):
                yield Button("Save", id="command-edit-save", variant="default")
                yield Button("Cancel", id="command-edit-cancel", variant="default")

    def on_mount(self) -> None:
        try:
            if not self._editing:
                self.query_one("#command-edit-name", Input).focus()
            else:
                self.query_one("#command-edit-prompt", Input).focus()
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/command_picker.py:151")
            pass

    def _done(self) -> None:
        try:
            name = self.query_one("#command-edit-name", Input).value.strip()
            desc = self.query_one("#command-edit-desc", Input).value.strip()
            prompt = self.query_one("#command-edit-prompt", Input).value.strip()
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/command_picker.py:159")
            return
        if not name or not prompt:
            self.app.notify("Name and prompt are required.", severity="warning")
            return
        self.dismiss((name, desc, prompt))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "command-edit-save":
            self._done()
        else:
            self.dismiss(None)
        event.stop()

    def on_input_submitted(self, event: Any) -> None:
        try:
            if not str(getattr(event.input, "id", "")).startswith("command-edit-"):
                return
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/command_picker.py:177")
            return
        event.stop()
        self._done()

    def on_key(self, event: Key) -> None:
        if event.key == "escape":
            self.dismiss(None)
            event.stop()


class CommandPicker(ModalScreen[str | None]):
    """List custom commands; add/edit/enable/disable/delete."""

    CSS = _COMMAND_PICKER_CSS
    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("ctrl+d", "delete", "Delete", show=False),
        Binding("ctrl+e", "toggle", "Enable/disable", show=False),
        Binding("ctrl+n", "edit", "Edit", show=False),
    ]

    def __init__(self, commands: dict | None = None, disabled: set[str] | None = None,
                 on_result: Any = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.commands = dict(commands or {})
        self.disabled = set(disabled or ())
        self._on_result = on_result

    def _describe(self, name: str) -> str:
        entry = self.commands.get(name) or {}
        desc = str(entry.get("description") or "").strip()
        if not desc:
            desc = " ".join(str(entry.get("prompt") or "").split()[:12])
        return desc or "(no description)"

    def _options(self) -> list[Option]:
        opts: list[Option] = []
        for name in sorted(self.commands):
            mark = "○" if name in self.disabled else "●"
            opts.append(Option(
                f"[{mark}] [green]▣[/] [b]/{escape(name)}[/]  [dim]{escape(self._describe(name))}[/]",
                id=f"__cmd__{name}",
            ))
        if not opts:
            opts.append(Option("[dim](none configured)[/]", disabled=True))
        opts.append(Option("＋ Add command", id="__add__"))
        return opts

    def reload(self, commands: dict, disabled: set[str]) -> None:
        self.commands = dict(commands or {})
        self.disabled = set(disabled or ())
        try:
            lst = self.query_one("#command-picker-list", OptionList)
            lst.clear_options()
            lst.add_options(self._options())
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/command_picker.py:233")
            pass

    def compose(self) -> ComposeResult:
        with Vertical(id="command-picker-box"):
            yield Static("  Commands — Enter info · Esc close  ", id="command-picker-title")
            yield OptionList(*self._options(), id="command-picker-list")
            yield Static(
                "↑/↓ move · Enter info · ＋ add · Ctrl+N edit · Ctrl+E on/off · Ctrl+D delete",
                id="command-picker-hint",
            )
            with Horizontal(id="command-picker-actions"):
                yield Button("Add", id="command-add", variant="default")
                yield Button("Edit", id="command-edit", variant="default")
                yield Button("On/Off", id="command-toggle", variant="default")
                yield Button("Delete", id="command-delete", variant="default")
                yield Button("Close", id="command-close", variant="default")

    def on_mount(self) -> None:
        try:
            self.query_one("#command-picker-list", OptionList).focus()
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/command_picker.py:254")
            pass

    def _highlighted(self) -> str | None:
        try:
            opt = self.query_one("#command-picker-list", OptionList).highlighted_option
        except Exception:
            return None
        if opt is None or opt.id is None:
            return None
        oid = str(opt.id)
        return oid[len("__cmd__"):] if oid.startswith("__cmd__") else None

    def _refocus(self) -> None:
        try:
            self.query_one("#command-picker-list", OptionList).focus()
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/command_picker.py:270")
            pass

    def _emit(self, action: Any) -> None:
        if self._on_result is None:
            return
        try:
            self._on_result(action)
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/command_picker.py:278")
            pass

    def on_option_list_option_selected(self, event: Any) -> None:
        try:
            oid = str(getattr(event.option, "id", "") or "")
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/command_picker.py:284")
            return
        event.stop()
        if oid == "__add__":
            self._open_edit(None)
        elif oid.startswith("__cmd__"):
            self.dismiss(oid)

    def _open_edit(self, name: str | None) -> None:
        entry = self.commands.get(name or "", {})
        try:
            self.app.push_screen(
                CommandEditDialog(
                    name=name or "",
                    description=str(entry.get("description") or ""),
                    prompt=str(entry.get("prompt") or ""),
                ),
                lambda result: self._after_edit(name, result),
            )
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/command_picker.py:303")
            pass

    def _after_edit(self, old: str | None, result: tuple[str, str, str] | None) -> None:
        self._refocus()
        if not result:
            return
        name, desc, prompt = result
        self._emit(("edit" if old else "add", name, desc, prompt))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "command-close":
            self.dismiss(None)
        elif bid == "command-add":
            self._open_edit(None)
        elif bid == "command-edit":
            name = self._highlighted()
            if name:
                self._open_edit(name)
        elif bid in ("command-toggle", "command-delete"):
            name = self._highlighted()
            if name:
                self._emit(("toggle" if bid == "command-toggle" else "delete", name))
            self._refocus()
        event.stop()

    def action_delete(self) -> None:
        name = self._highlighted()
        if name:
            self._emit(("delete", name))
        self._refocus()

    def action_toggle(self) -> None:
        name = self._highlighted()
        if name:
            self._emit(("toggle", name))
        self._refocus()

    def action_edit(self) -> None:
        name = self._highlighted()
        if name:
            self._open_edit(name)

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_key(self, event: Key) -> None:
        if event.key == "escape":
            self.dismiss(None)
            event.stop()

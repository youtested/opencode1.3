"""Plugin manager popup in /agent picker style: tools + hooks, one row each."""
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

from ..plugin_manager import HOOK_EVENTS

_PLUGIN_PICKER_CSS = """
PluginPicker {
    align: center middle;
}
#plugin-picker-box {
    width: 84;
    max-width: 99%;
    max-height: 90%;
    height: auto;
    background: $surface;
    border: round $accent;
    padding: 1 2;
}
#plugin-picker-title {
    text-style: bold;
    color: $accent;
    height: 1;
    margin-bottom: 1;
}
#plugin-picker-hint {
    color: $text-muted;
    height: 1;
    margin-top: 1;
}
#plugin-picker-list {
    height: auto;
    max-height: 18;
    background: transparent;
    border: none;
    padding: 0;
    scrollbar-size-vertical: 1;
    scrollbar-size-horizontal: 0;
}
#plugin-picker-actions {
    height: 3;
    width: 100%;
    align: center middle;
    background: $surface;
    padding: 0 1;
    margin-top: 1;
}
#plugin-picker-actions Button {
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


class PluginAddDialog(ModalScreen[tuple[str, str, str, str] | None]):
    """Add dialog: name + target + kind (tool/hook) + events (hooks only)."""

    CSS = """
    PluginAddDialog {
        align: center middle;
    }
    #plugin-add-box {
        width: 64;
        max-width: 96%;
        height: auto;
        background: $surface;
        border: round $accent;
        padding: 1 2;
    }
    #plugin-add-title {
        text-style: bold;
        color: $accent;
        height: 1;
        margin-bottom: 1;
    }
    #plugin-add-box Input {
        margin-bottom: 1;
        border: none;
        background: $background;
        padding: 0 1;
    }
    #plugin-add-actions {
        height: 3;
        align: center middle;
        background: $surface;
    }
    #plugin-add-actions Button {
        height: 3;
        min-width: 12;
        padding: 0 2;
        margin: 0 1;
        border: heavy $accent;
        background: transparent;
        color: $text;
    }
    """

    def __init__(self, kind: str = "tool", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._kind = kind

    def compose(self) -> ComposeResult:
        title = {"tool": "Add tool plugin", "hook": "Add event hook", "npm": "Add npm plugin"}.get(self._kind, "Add plugin")
        target_hint = {
            "tool": "Module, e.g. my_tools.plugin or ./tools/demo.py",
            "hook": "File or command, e.g. ./hooks/format.py",
            "npm": "npm package, e.g. npm:my-mcp-plugin or @scope/plugin",
        }.get(self._kind, "Target")
        with Vertical(id="plugin-add-box"):
            yield Static(f"  {title}  ", id="plugin-add-title")
            yield Input(placeholder="Name, e.g. formatter", id="plugin-add-name")
            yield Input(placeholder=target_hint, id="plugin-add-target")
            if self._kind == "hook":
                yield Input(
                    placeholder="Events, comma separated: " + ", ".join(HOOK_EVENTS[:3]),
                    id="plugin-add-events",
                )
            with Horizontal(id="plugin-add-actions"):
                yield Button("Save", id="plugin-add-save", variant="default")
                yield Button("Cancel", id="plugin-add-cancel", variant="default")

    def on_mount(self) -> None:
        try:
            self.query_one("#plugin-add-name", Input).focus()
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/plugin_picker.py:143")
            pass

    def _done(self) -> None:
        try:
            name = self.query_one("#plugin-add-name", Input).value.strip()
            target = self.query_one("#plugin-add-target", Input).value.strip()
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/plugin_picker.py:150")
            return
        events = ""
        if self._kind == "hook":
            try:
                events = self.query_one("#plugin-add-events", Input).value.strip()
            except Exception:
                events = ""
        if not name or not target:
            self.app.notify("Name and target are required.", severity="warning")
            return
        self.dismiss((name, target, self._kind, events))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "plugin-add-save":
            self._done()
        else:
            self.dismiss(None)
        event.stop()

    def on_input_submitted(self, event: Any) -> None:
        try:
            if not str(getattr(event.input, "id", "")).startswith("plugin-add-"):
                return
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/plugin_picker.py:174")
            return
        event.stop()
        self._done()

    def on_key(self, event: Key) -> None:
        if event.key == "escape":
            self.dismiss(None)
            event.stop()


class PluginPicker(ModalScreen[str | None]):
    """List tool plugins + hooks, add/enable/disable/test/delete."""

    CSS = _PLUGIN_PICKER_CSS
    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("ctrl+d", "delete", "Delete", show=False),
        Binding("ctrl+e", "toggle", "Enable/disable", show=False),
        Binding("ctrl+t", "test", "Test", show=False),
    ]

    def __init__(self, plugins: list[str] | None = None, hooks: list[dict] | None = None,
                 disabled: set[str] | None = None, on_result: Any = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.plugins = list(plugins or [])
        self.hooks = [dict(h) for h in (hooks or [])]
        self.disabled = set(disabled or ())
        self._on_result = on_result

    def _hook_state(self, hook: dict) -> bool:
        name = str(hook.get("name") or "")
        return not (hook.get("disabled") is True or f"hook:{name}" in self.disabled)

    def _options(self) -> list[Option]:
        opts: list[Option] = []
        for name in self.plugins:
            mark = "○" if name in self.disabled else "●"
            opts.append(Option(
                f"[{mark}] [green]▣[/] [b]{escape(name)}[/]  [dim]tool plugin[/]",
                id=f"__plugin__{name}",
            ))
        for hook in self.hooks:
            name = str(hook.get("name") or "?")
            events = ",".join(str(e) for e in (hook.get("events") or []))
            target = str(hook.get("target") or "?")
            mark = "○" if not self._hook_state(hook) else "●"
            opts.append(Option(
                f"[{mark}] [cyan]⇄[/] [b]{escape(name)}[/]  [dim]{escape(target)} · {escape(events)}[/]",
                id=f"__plugin__hook:{name}",
            ))
        if not opts:
            opts.append(Option("[dim](none configured)[/]", disabled=True))
        opts.append(Option("＋ Add tool plugin", id="__add_tool__"))
        opts.append(Option("＋ Add event hook", id="__add_hook__"))
        opts.append(Option("＋ Add npm package", id="__add_npm__"))
        return opts

    def reload(self, plugins: list[str], hooks: list[dict], disabled: set[str]) -> None:
        self.plugins = list(plugins or [])
        self.hooks = [dict(h) for h in (hooks or [])]
        self.disabled = set(disabled or ())
        try:
            lst = self.query_one("#plugin-picker-list", OptionList)
            lst.clear_options()
            lst.add_options(self._options())
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/plugin_picker.py:240")
            pass

    def compose(self) -> ComposeResult:
        with Vertical(id="plugin-picker-box"):
            yield Static("  Plugins — Enter info · Esc close  ", id="plugin-picker-title")
            yield OptionList(*self._options(), id="plugin-picker-list")
            yield Static(
                "↑/↓ move · Enter info · ＋ add · Ctrl+E on/off · Ctrl+T test · Ctrl+D delete",
                id="plugin-picker-hint",
            )
            with Horizontal(id="plugin-picker-actions"):
                yield Button("Add Tool", id="plugin-add-tool", variant="default")
                yield Button("Add Hook", id="plugin-add-hook", variant="default")
                yield Button("Add npm", id="plugin-add-npm", variant="default")
                yield Button("Test", id="plugin-test", variant="default")
                yield Button("On/Off", id="plugin-toggle", variant="default")
                yield Button("Close", id="plugin-close", variant="default")

    def on_mount(self) -> None:
        try:
            self.query_one("#plugin-picker-list", OptionList).focus()
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/plugin_picker.py:262")
            pass

    def _highlighted(self) -> str | None:
        try:
            opt = self.query_one("#plugin-picker-list", OptionList).highlighted_option
        except Exception:
            return None
        if opt is None or opt.id is None:
            return None
        oid = str(opt.id)
        return oid[len("__plugin__"):] if oid.startswith("__plugin__") else None

    def _refocus(self) -> None:
        try:
            self.query_one("#plugin-picker-list", OptionList).focus()
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/plugin_picker.py:278")
            pass

    def _emit(self, action: Any) -> None:
        if self._on_result is None:
            return
        try:
            self._on_result(action)
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/plugin_picker.py:286")
            pass

    def on_option_list_option_selected(self, event: Any) -> None:
        try:
            oid = str(getattr(event.option, "id", "") or "")
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/plugin_picker.py:292")
            return
        event.stop()
        if oid == "__add_tool__":
            self._open_add("tool")
        elif oid == "__add_hook__":
            self._open_add("hook")
        elif oid == "__add_npm__":
            self._open_add("npm")
        elif oid.startswith("__plugin__"):
            self.dismiss(oid)

    def _open_add(self, kind: str) -> None:
        try:
            self.app.push_screen(PluginAddDialog(kind=kind), lambda r: self._after_add(r))
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/plugin_picker.py:307")
            pass

    def _after_add(self, result: tuple[str, str, str, str] | None) -> None:
        self._refocus()
        if not result:
            return
        name, target, kind, events = (list(result) + ["", "", "", ""])[:4]
        event_list = [e.strip() for e in events.split(",") if e.strip()]
        self._emit(("add", name, target, kind, event_list))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "plugin-close":
            self.dismiss(None)
        elif bid == "plugin-add-tool":
            self._open_add("tool")
        elif bid == "plugin-add-hook":
            self._open_add("hook")
        elif bid == "plugin-add-npm":
            self._open_add("npm")
        elif bid in ("plugin-test", "plugin-toggle"):
            name = self._highlighted()
            if name:
                self._emit(("test" if bid == "plugin-test" else "toggle", name))
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

    def action_test(self) -> None:
        name = self._highlighted()
        if name:
            self._emit(("test", name))
        self._refocus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_key(self, event: Key) -> None:
        if event.key == "escape":
            self.dismiss(None)
            event.stop()

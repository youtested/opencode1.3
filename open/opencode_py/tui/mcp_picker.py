"""MCP manager popup in /agent picker style: one row per server + add/test/enable/remove."""
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


_MCP_PICKER_CSS = """
McpPicker {
    align: center middle;
}
#mcp-picker-box {
    width: 96;
    max-width: 99%;
    max-height: 90%;
    height: auto;
    background: $surface;
    border: round $accent;
    padding: 1 2;
}
#mcp-picker-title {
    text-style: bold;
    color: $accent;
    height: 1;
    margin-bottom: 1;
}
#mcp-picker-hint {
    color: $text-muted;
    height: 1;
    margin-top: 1;
}
#mcp-picker-list {
    height: auto;
    max-height: 20;
    background: transparent;
    border: none;
    padding: 0;
    scrollbar-size-vertical: 1;
    scrollbar-size-horizontal: 0;
}
#mcp-picker-name, #mcp-picker-cmd {
    height: 3;
    margin: 0 2;
}
#mcp-picker-actions {
    height: 3;
    width: 100%;
    align: center middle;
    background: $surface;
    padding: 0 1;
    margin-top: 1;
}
#mcp-picker-actions Button {
    height: 3;
    min-width: 8;
    width: 1fr;
    max-width: 18;
    padding: 0 1;
    margin: 0 1;
    border: heavy $accent;
    background: transparent;
    color: $text;
    text-align: center;
}
"""


class McpAddDialog(ModalScreen[tuple[str, str, str, str] | None]):
    """Add dialog: name + endpoint (command line or URL) + optional token."""

    CSS = """
    McpAddDialog {
        align: center middle;
    }
    #mcp-add-box {
        width: 60;
        max-width: 92%;
        height: auto;
        background: $surface;
        border: round $accent;
        padding: 1 2;
    }
    #mcp-add-title {
        text-style: bold;
        color: $accent;
        height: 1;
        margin-bottom: 1;
    }
    #mcp-add-box Input {
        margin-bottom: 1;
        border: none;
        background: $background;
        padding: 0 1;
    }
    #mcp-add-actions {
        height: 3;
        align: center middle;
        background: $surface;
    }
    #mcp-add-actions Button {
        height: 3;
        min-width: 12;
        padding: 0 2;
        margin: 0 1;
        border: heavy $accent;
        background: transparent;
        color: $text;
    }
    """

    def __init__(self, mode: str = "local", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._mode = mode

    def compose(self) -> ComposeResult:
        title = "Add local server" if self._mode == "local" else "Add remote server"
        endpoint_label = (
            "Run line, e.g. python -m my_server  (npx -y ... needs Node)"
            if self._mode == "local"
            else "URL, e.g. https://mcp.example.com/mcp"
        )
        with Vertical(id="mcp-add-box"):
            yield Static(f"  {title}  ", id="mcp-add-title")
            yield Input(placeholder="Name, e.g. files", id="mcp-add-name")
            yield Input(placeholder=endpoint_label, id="mcp-add-endpoint")
            if self._mode == "remote":
                yield Input(placeholder="Token (optional, stored in opencode.json)", id="mcp-add-token")
            with Horizontal(id="mcp-add-actions"):
                yield Button("Save", id="mcp-add-save", variant="default")
                yield Button("Cancel", id="mcp-add-cancel", variant="default")

    def on_mount(self) -> None:
        try:
            inp = self.query_one("#mcp-add-name", Input)
            inp.focus()
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/mcp_picker.py:144")
            pass

    def _done(self) -> None:
        try:
            name = self.query_one("#mcp-add-name", Input).value.strip()
            endpoint = self.query_one("#mcp-add-endpoint", Input).value.strip()
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/mcp_picker.py:151")
            return
        token = ""
        if self._mode == "remote":
            try:
                token = self.query_one("#mcp-add-token", Input).value.strip()
            except Exception:
                token = ""
        if not name:
            self.app.notify("Server name can't be empty.", severity="warning")
            return
        if not endpoint:
            self.app.notify("Endpoint can't be empty.", severity="warning")
            return
        self.dismiss((name, endpoint, self._mode, token))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "mcp-add-save":
            self._done()
        else:
            self.dismiss(None)
        event.stop()

    def on_input_submitted(self, event: Any) -> None:
        try:
            if getattr(event.input, "id", "") not in ("mcp-add-name", "mcp-add-endpoint", "mcp-add-token"):
                return
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/mcp_picker.py:178")
            return
        event.stop()
        self._done()

    def on_key(self, event: Key) -> None:
        if event.key == "escape":
            self.dismiss(None)
            event.stop()


class McpPicker(ModalScreen[str | None]):
    """List servers, add local/remote, test, enable/disable, remove."""

    CSS = _MCP_PICKER_CSS
    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("ctrl+d", "delete", "Delete", show=False),
        Binding("ctrl+e", "toggle", "Enable/disable", show=False),
        Binding("ctrl+t", "test", "Test", show=False),
    ]

    def __init__(self, servers: dict | None = None, on_result: Any = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.servers: dict[str, Any] = dict(servers or {})
        self._on_result = on_result

    def _describe(self, name: str, spec: Any) -> str:
        if not isinstance(spec, dict):
            return "(bad entry)"
        state = "off" if spec.get("disabled") is True else "on"
        if spec.get("url"):
            return f"remote · {spec.get('url')} · {state}"
        cmd = str(spec.get("command") or "?")
        args = " ".join(str(a) for a in (spec.get("args") or []))
        return f"local · {(cmd + ' ' + args).strip()} · {state}"

    def _options(self) -> list[Option]:
        opts: list[Option] = []
        for name, spec in self.servers.items():
            mark = "○" if isinstance(spec, dict) and spec.get("disabled") is True else "●"
            badge = "[cyan]⇄[/]" if isinstance(spec, dict) and spec.get("url") else "[green]▣[/]"
            opts.append(
                Option(
                    f"[{mark}] {badge} {escape(name)}  [dim]{escape(self._describe(name, spec))}[/]",
                    id=f"__srv__{name}",
                )
            )
        opts.append(Option("＋ Add local server", id="__add_local__"))
        opts.append(Option("＋ Add remote server", id="__add_remote__"))
        return opts

    def reload_servers(self, servers: dict) -> None:
        self.servers = dict(servers or {})
        try:
            lst = self.query_one("#mcp-picker-list", OptionList)
            lst.clear_options()
            lst.add_options(self._options())
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/mcp_picker.py:236")
            pass

    def compose(self) -> ComposeResult:
        with Vertical(id="mcp-picker-box"):
            yield Static("  MCP — Enter info · Esc close  ", id="mcp-picker-title")
            yield OptionList(*self._options(), id="mcp-picker-list")
            yield Static(
                "↑/↓ move · Enter info · ＋ add · Ctrl+E on/off · Ctrl+T test · Ctrl+D delete",
                id="mcp-picker-hint",
            )
            with Horizontal(id="mcp-picker-actions"):
                yield Button("Add Local", id="mcp-add-local", variant="default")
                yield Button("Add Remote", id="mcp-add-remote", variant="default")
                yield Button("Test", id="mcp-test", variant="default")
                yield Button("On/Off", id="mcp-toggle", variant="default")
                yield Button("Close", id="mcp-close", variant="default")

    def on_mount(self) -> None:
        try:
            self.query_one("#mcp-picker-list", OptionList).focus()
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/mcp_picker.py:257")
            pass

    def _highlighted_server(self) -> str | None:
        try:
            lst = self.query_one("#mcp-picker-list", OptionList)
            opt = lst.highlighted_option
        except Exception:
            return None
        if opt is None or opt.id is None:
            return None
        oid = str(opt.id)
        return oid[len("__srv__"):] if oid.startswith("__srv__") else None

    def _refocus(self) -> None:
        try:
            self.query_one("#mcp-picker-list", OptionList).focus()
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/mcp_picker.py:274")
            pass

    def on_option_list_option_selected(self, event: Any) -> None:
        try:
            opt = getattr(event, "option", None)
            oid = str(getattr(opt, "id", "") or "")
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/mcp_picker.py:281")
            return
        event.stop()
        if oid == "__add_local__":
            self._open_add("local")
        elif oid == "__add_remote__":
            self._open_add("remote")
        elif oid.startswith("__srv__"):
            self.dismiss(oid)

    def _open_add(self, mode: str) -> None:
        try:
            self.app.push_screen(McpAddDialog(mode=mode), lambda result: self._after_add(result))
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/mcp_picker.py:294")
            pass

    def _after_add(self, result: tuple[str, str, str, str] | None) -> None:
        self._refocus()
        if not result or self._result_cb is None:
            return
        try:
            self._result_cb(("add",) + tuple(result))
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/mcp_picker.py:303")
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "mcp-close":
            self.dismiss(None)
        elif bid == "mcp-add-local":
            self._open_add("local")
        elif bid == "mcp-add-remote":
            self._open_add("remote")
        elif bid in ("mcp-test", "mcp-toggle"):
            name = self._highlighted_server()
            if name is not None and self._result_cb is not None:
                try:
                    self._result_cb(("test" if bid == "mcp-test" else "toggle", name))
                except Exception as _oc_exc:
                    swallow(_oc_exc, "tui/mcp_picker.py:319")
                    pass
            self._refocus()
        event.stop()

    def action_delete(self) -> None:
        name = self._highlighted_server()
        if name is None or self._result_cb is None:
            return
        try:
            self._result_cb(("delete", name))
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/mcp_picker.py:330")
            pass
        self._refocus()

    def action_toggle(self) -> None:
        name = self._highlighted_server()
        if name is None or self._result_cb is None:
            return
        try:
            self._result_cb(("toggle", name))
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/mcp_picker.py:340")
            pass
        self._refocus()

    def action_test(self) -> None:
        name = self._highlighted_server()
        if name is None or self._result_cb is None:
            return
        try:
            self._result_cb(("test", name))
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/mcp_picker.py:350")
            pass
        self._refocus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_key(self, event: Key) -> None:
        if event.key == "escape":
            self.dismiss(None)
            event.stop()

    @property
    def _result_cb(self):
        return self._on_result

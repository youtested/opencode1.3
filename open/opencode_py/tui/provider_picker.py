"""Provider manager popup in the same style as /mcp and /plugins."""
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

_CSS = """
ProviderPicker { align: center middle; }
#provider-box { width: 88; max-width: 99%; max-height: 90%; height: auto; background: $surface; border: round $accent; padding: 1 2; }
#provider-title { text-style: bold; color: $accent; height: 1; margin-bottom: 1; }
#provider-hint { color: $text-muted; height: 1; margin-top: 1; }
#provider-list { height: auto; max-height: 20; background: transparent; border: none; padding: 0; }
#provider-actions { height: 3; width: 100%; align: center middle; background: $surface; padding: 0 1; margin-top: 1; }
#provider-actions Button { height: 3; min-width: 10; width: 1fr; max-width: 22; margin: 0 1; padding: 0 1; border: heavy $accent; background: transparent; color: $text; }
"""

class ProviderKeyDialog(ModalScreen[tuple[str, str] | None]):
    CSS = """
    ProviderKeyDialog { align: center middle; }
    #provider-key-box { width: 58; max-width: 94%; height: auto; background: $surface; border: round $accent; padding: 1 2; }
    #provider-key-title { text-style: bold; color: $accent; height: 1; margin-bottom: 1; }
    #provider-key-box Input { margin-bottom: 1; border: none; background: $background; padding: 0 1; }
    #provider-key-actions { height: 3; align: center middle; }
    #provider-key-actions Button { height: 3; min-width: 12; margin: 0 1; border: heavy $accent; background: transparent; color: $text; }
    """
    def __init__(self, provider: str, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.provider = provider
    def compose(self) -> ComposeResult:
        with Vertical(id="provider-key-box"):
            yield Static(f"  Add key for {self.provider}  ", id="provider-key-title")
            yield Input(placeholder="Paste API key", id="provider-key-input")
            with Horizontal(id="provider-key-actions"):
                yield Button("Save", id="provider-key-save", variant="default")
                yield Button("Cancel", id="provider-key-cancel", variant="default")
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "provider-key-save":
            try:
                value = self.query_one("#provider-key-input", Input).value.strip()
            except Exception:
                value = ""
            self.dismiss((self.provider, value)) if value else self.app.notify("Key cannot be empty", severity="warning")
        else:
            self.dismiss(None)
        event.stop()
    def on_input_submitted(self, event: Any) -> None:
        self.on_button_pressed(type("B", (), {"button": type("I", (), {"id": "provider-key-save"})(), "stop": lambda self: None})())
    def on_key(self, event: Key) -> None:
        if event.key == "escape":
            self.dismiss(None)
            event.stop()

class ProviderAddDialog(ModalScreen[tuple[str, str, str] | None]):
    CSS = """
    ProviderAddDialog { align: center middle; width: 100%; height: 100%; }
    #provider-key-box { width: 58; max-width: 94%; height: auto; background: $surface; border: round $accent; padding: 1 2; }
    #provider-key-title { text-style: bold; color: $accent; height: 1; margin-bottom: 1; }
    #provider-key-box Input { margin-bottom: 1; border: none; background: $background; padding: 0 1; }
    #provider-key-actions { height: 3; align: center middle; }
    #provider-key-actions Button { height: 3; min-width: 12; margin: 0 1; border: heavy $accent; background: transparent; color: $text; }
    """

    def compose(self) -> ComposeResult:
        with Vertical(id="provider-key-box"):
            yield Static("  Add provider  ", id="provider-key-title")
            yield Input(placeholder="Name, e.g. my-provider", id="provider-add-name")
            yield Input(placeholder="Base URL, e.g. http://127.0.0.1:11434/v1", id="provider-add-url")
            yield Input(placeholder="API key (optional)", id="provider-add-key")
            with Horizontal(id="provider-key-actions"):
                yield Button("Save", id="provider-add-save", variant="default")
                yield Button("Cancel", id="provider-add-cancel", variant="default")
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "provider-add-save":
            name = self.query_one("#provider-add-name", Input).value.strip()
            url = self.query_one("#provider-add-url", Input).value.strip()
            key = self.query_one("#provider-add-key", Input).value.strip()
            if not name or not url: self.app.notify("Name and URL are required", severity="warning")
            else: self.dismiss((name, url, key))
        else: self.dismiss(None)
        event.stop()
    def on_key(self, event: Key) -> None:
        if event.key == "escape": self.dismiss(None); event.stop()

class ProviderPicker(ModalScreen[str | None]):
    CSS = _CSS
    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("ctrl+e", "key", "Add key", show=False),
        Binding("ctrl+r", "refresh", "Refresh", show=False),
        Binding("ctrl+d", "remove", "Remove", show=False),
    ]
    def __init__(self, rows: list[dict] | None = None, on_result: Any = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.rows = list(rows or [])
        self.selected_id = ""
        self._on_result = on_result

    def set_status(self, text) -> None:
        try:
            self.query_one("#provider-hint").update(str(text or ""))
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/provider_picker.py:109")
            pass
    def _options(self) -> list[Option]:
        out = [Option("[b]Providers — Ctrl+E add key · Ctrl+R refresh · Ctrl+D remove[/]", disabled=True)]
        for row in self.rows:
            mark = "[green]●[/]" if row["status"] == "Connected" else "[yellow]○[/]"
            model = (" · " + row["model"]) if row.get("model") else ""
            out.append(Option(f"{mark} [b]{escape(row['name'])}[/]  [dim]{escape(row['status'])}{escape(model)}[/]", id="__provider__" + row["id"]))
        out.append(Option("[cyan]＋ Add provider[/]", id="__provider_add__"))
        return out
    def reload(self, rows: list[dict]) -> None:
        self.rows = list(rows or [])
        try:
            lst = self.query_one("#provider-list", OptionList)
            options = self._options()
            lst.clear_options(); lst.add_options(options)
            if self.selected_id:
                for index, option in enumerate(options):
                    if option.id == self.selected_id:
                        lst.highlighted = index
                        break
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/provider_picker.py:132")
            pass
    def compose(self) -> ComposeResult:
        with Vertical(id="provider-box"):
            yield Static("  Providers — API keys and models  ", id="provider-title")
            yield OptionList(*self._options(), id="provider-list")
            yield Static("Enter a provider to view it · Ctrl+E add key · Ctrl+R refresh · Ctrl+D remove", id="provider-hint")
            with Horizontal(id="provider-actions"):
                yield Button("+ Add", id="provider-new", variant="default")
                yield Button("Add Key", id="provider-add", variant="default")
                yield Button("Login URL", id="provider-login", variant="default")
                yield Button("Refresh", id="provider-refresh", variant="default")
                yield Button("Remove", id="provider-remove", variant="default")
                yield Button("Close", id="provider-close", variant="default")
    def on_mount(self) -> None:
        try: self.query_one("#provider-list", OptionList).focus()
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/provider_picker.py:147")
            pass
    def _selected(self) -> dict | None:
        try:
            opt = self.query_one("#provider-list", OptionList).highlighted_option
            oid = str(getattr(opt, "id", "") or "")
            if oid == "__provider_add__":
                return None
            if oid.startswith("__provider__"):
                self.selected_id = oid
                return next((r for r in self.rows if r["id"] == oid[len("__provider__"):]), None)
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/provider_picker.py:157")
            pass
        return None

    def _restore_highlight(self) -> None:
        try:
            lst = self.query_one("#provider-list", OptionList)
            if self.selected_id:
                for index, option in enumerate(self._options()):
                    if option.id == self.selected_id:
                        lst.highlighted = index
                        return
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/provider_picker.py:166")
            pass
    def _emit(self, action: Any) -> None:
        if self._on_result:
            try: self._on_result(action)
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/provider_picker.py:174")
                pass
    def on_option_list_option_selected(self, event: Any) -> None:
        try:
            oid = str(getattr(event.option, "id", "") or "")
        except Exception:
            oid = ""
        if oid == "__provider_add__":
            self._emit(("add",))
        else:
            row = self._selected()
            if row: self.dismiss("__provider__" + row["id"])
        event.stop()
    def on_button_pressed(self, event: Button.Pressed) -> None:
        row = self._selected()
        bid = event.button.id
        if bid == "provider-close": self.dismiss(None)
        elif bid == "provider-new": self._emit(("add",))
        elif bid == "provider-add" and row: self._emit(("key", row["id"]))
        elif bid == "provider-login" and row: self._emit(("login", row["id"], row.get("url", "")))
        elif bid == "provider-refresh": self._emit(("refresh",))
        elif bid == "provider-remove" and row: self._emit(("remove", row["id"]))
        else:
            self._restore_highlight()
        event.stop()
    def action_key(self) -> None:
        row = self._selected()
        if row: self._emit(("key", row["id"]))
    def action_refresh(self) -> None: self._emit(("refresh",))
    def action_remove(self) -> None:
        row = self._selected()
        if row: self._emit(("remove", row["id"]))
    def action_cancel(self) -> None: self.dismiss(None)
    def on_key(self, event: Key) -> None:
        if event.key == "escape": self.dismiss(None); event.stop()

__all__ = ["ProviderAddDialog", "ProviderKeyDialog", "ProviderPicker"]

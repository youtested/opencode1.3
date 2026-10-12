"""Theme picker screen (bare /theme): arrow-navigable, custom themes first.

Each row shows three live color swatches sampled from that theme's own
palette (background / primary / accent), the command-style name, and a dim
one-line hint. Enter applies the highlighted theme immediately; Esc cancels.

Footer buttons manage custom themes: + New creates one (copies the
highlighted theme), Edit opens it in the editor, Rename asks for a new
name, Delete asks for confirmation. Built-ins can't be edited.
"""

from __future__ import annotations

from ..util.log import swallow
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, OptionList, Static
from textual.widgets.option_list import Option

from .theme import LIGHT_THEMES, THEMES, custom_names, get_theme, is_custom

# Short per-theme briefing shown dimmed after the name.
HINTS: dict[str, str] = {
    "opencode": "opencode default dark",
    "dark": "black · orange & gray",
    "tokyo-night": "deep indigo · neon blue/purple",
    "blackout": "true-black OLED · amber/violet",
    "catppuccin-mocha": "soft pastel on dark cocoa",
    "dracula": "classic purple/pink night",
    "nord": "cool arctic blues",
    "gruvbox-dark": "warm retro amber/olive",
    "one-dark": "atom-style muted slate",
    "solarized": "classic low-contrast teal",
    "solarized-light": "paper-warm light",
    "catppuccin-latte": "pastel light",
    "github-light": "clean github white",
}

_THEME_PICKER_CSS = """
ThemePicker {
    align: center middle;
}
#theme-picker-box {
    width: 64;
    max-height: 88%;
    height: auto;
    background: $surface;
    border: round #666;
    padding: 0 1;
}
#theme-picker-title {
    text-style: bold;
    margin-bottom: 1;
}
#theme-picker-list {
    height: auto;
    max-height: 20;
    background: transparent;
}
#theme-picker-actions {
    height: 3;
    align: center middle;
    background: $surface;
    margin-top: 1;
}
#theme-picker-actions Button {
    height: 3;
    min-width: 10;
    padding: 0 1;
    margin: 0 1;
    border: heavy $accent;
    background: transparent;
    color: $text;
}
#theme-picker-actions Button:disabled {
    color: $text-muted;
    border: none;
}
"""


def _swatches(name: str) -> str:
    """Three blocks painted with the theme's own bg/primary/accent colors."""
    t = get_theme(name)
    return (
        f"[on {t.c('background')}]  [/]"
        f"[on {t.c('primary')}]  [/]"
        f"[on {t.c('accent')}]  [/]"
    )


def _row(name: str, current: str) -> str:
    marker = "●" if name == current else " "
    tag = " (custom)" if is_custom(name) else ""
    hint = HINTS.get(name, "your custom theme" if is_custom(name) else "")
    return f"{_swatches(name)} [{marker}] /{name}{tag}  [dim]{hint}[/]"


# Picker result: ("apply", name) | ("new", base) | ("edit", name) |
# ("rename", name) | ("delete", name) | None on Esc.
PickerResult = tuple[str, str] | None


class ThemePicker(ModalScreen[PickerResult]):
    """Pick a theme with the arrow keys; Enter applies it live."""

    CSS = _THEME_PICKER_CSS
    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    def __init__(self, current: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self.current_theme = current or "opencode"

    def compose(self) -> ComposeResult:
        with Vertical(id="theme-picker-box"):
            yield Static(
                f"  Theme — arrows to move · Enter to apply · Esc to cancel"
                f"  [dim](current: {self.current_theme})[/]",
                id="theme-picker-title",
            )
            options: list[Option] = []
            customs = custom_names()
            if customs:
                options.append(Option("[b]Custom themes[/]", disabled=True))
                for name in customs:
                    options.append(Option(_row(name, self.current_theme), id=name))
                options.append(Option("", disabled=True))
            options.append(Option("[b]Dark themes[/]", disabled=True))
            for name in THEMES:
                if name == "opencode-dark":
                    continue  # alias — `opencode` is the canonical entry
                if name == LIGHT_THEMES[0]:
                    options.append(
                        Option("", disabled=True)  # visual gap between sections
                    )
                    options.append(Option("[b]Light themes[/]", disabled=True))
                options.append(Option(_row(name, self.current_theme), id=name))
            yield OptionList(*options, id="theme-picker-list")
            with Horizontal(id="theme-picker-actions"):
                yield Button("+ New", id="theme-new", variant="default")
                yield Button("Edit", id="theme-edit", variant="default")
                yield Button("Rename", id="theme-rename", variant="default")
                yield Button("Delete", id="theme-delete", variant="default")

    def on_mount(self) -> None:
        lst = self.query_one("#theme-picker-list", OptionList)
        lst.focus()
        # highlight the current theme so Enter re-applies / arrows move from it
        for i in range(lst.option_count):
            if lst.get_option_at_index(i).id == self.current_theme:
                lst.highlighted = i
                break
        else:
            lst.highlighted = 0
        self._refresh_buttons()

    def _highlighted_name(self) -> str | None:
        try:
            lst = self.query_one("#theme-picker-list", OptionList)
            opt = lst.get_option_at_index(lst.highlighted or 0)
            oid = getattr(opt, "id", None)
            if oid and not getattr(opt, "disabled", False):
                return str(oid)
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/theme_picker.py:165")
            pass
        return None

    def _refresh_buttons(self) -> None:
        name = self._highlighted_name()
        custom = bool(name and is_custom(name))
        for bid in ("theme-edit", "theme-rename", "theme-delete"):
            try:
                self.query_one(f"#{bid}", Button).disabled = not custom
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/theme_picker.py:175")
                pass

    def on_option_list_option_highlighted(self, event: OptionList.OptionHighlighted) -> None:
        self._refresh_buttons()
        event.stop()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        option_id = getattr(event.option, "id", None)
        if not option_id or getattr(event.option, "disabled", False):
            return  # section header / gap row: nothing to apply
        self.dismiss(("apply", option_id))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        base = self._highlighted_name() or self.current_theme
        if bid == "theme-new":
            self.dismiss(("new", base))
            event.stop()
            return
        if bid in ("theme-edit", "theme-rename", "theme-delete"):
            name = self._highlighted_name()
            if name and is_custom(name):
                self.dismiss((bid[len("theme-"):], name))
                event.stop()
                return
            try:
                self.app.notify(
                    "Select a custom theme first — + New creates one; "
                    "built-ins can't be edited.",
                    severity="warning",
                )
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/theme_picker.py:207")
                pass
            return
        event.stop()

    def action_cancel(self) -> None:
        self.dismiss(None)

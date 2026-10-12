"""Custom theme editor: name + hex inputs with a live preview.

Type a #rrggbb code into any row (bare `fab283`, `#fab283`, or short
`#fab` all work). The swatch next to the row updates as you type and the
preview pane below re-renders a full-TUI mock (chat, tools, syntax,
markdown, diff, status signals, agent badges), so you see the theme before
saving.

All color groups sit side by side in one scrollable popup — main, syntax,
markdown, diff, UX, and per-agent badge colors (`agent.build` …
`agent.test`, overriding the primary/secondary/accent roles). The preview
is its own box attached under the list (same popup, visually joined).
Agent badges come from Theme.agent_color(), so they follow your palette
automatically.
"""

from __future__ import annotations

from ..util.log import swallow
from typing import Any

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, ScrollableContainer, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static

from .theme import (
    OPENSE_DARK,
    Theme,
    get_theme,
    is_valid_hex,
    normalize_hex,
    valid_theme_name,
)

CURATED_KEYS: list[str] = [
    "background", "background_panel", "background_element", "border",
    "primary", "secondary", "accent",
    "text", "text_muted",
    "error", "warning", "success", "info",
    "user_bubble",
    "syntax_keyword", "syntax_function", "syntax_string",
    "markdown_heading", "markdown_link",
    "diff_added", "diff_removed",
]

# Two panes, not six columns: Textual layout cost explodes with sibling
# columns (7 cols ≈ 45s on-device, 2 cols ≈ 4s for the same 67 rows).
# Groups are merged into left (40 rows) / right (27 rows + preview) panes;
# group title rows keep every family labeled and findable.
# ponytail: fixed group lists (not derived) — a new OPENSE_DARK key lands in
# "extra" via _grouped_keys() fallback; add it to a pane when it needs one.
_PANES: list[list[tuple[str, list[str]]]] = [
    [
        ("main", [
            "background", "background_panel", "background_element",
            "background_menu", "border", "border_active", "border_subtle",
            "primary", "secondary", "accent",
            "text", "text_muted",
            "error", "warning", "success", "info",
            "user_bubble", "streaming_cursor",
        ]),
        ("syntax", [
            "syntax_keyword", "syntax_function", "syntax_variable",
            "syntax_string", "syntax_number", "syntax_type",
            "syntax_operator", "syntax_punctuation", "syntax_comment",
        ]),
        ("markdown", [
            "markdown_heading", "markdown_link", "markdown_link_text",
            "markdown_code", "markdown_quote", "markdown_strong",
            "markdown_hr", "markdown_list_item", "markdown_list_enumeration",
            "markdown_strike", "markdown_todo_done", "markdown_todo_open",
            "markdown_code_block",
        ]),
    ],
    [
        ("diff", [
            "diff_added", "diff_removed", "diff_context", "diff_hunk_header",
            "diff_highlight_added", "diff_highlight_removed",
            "diff_added_bg", "diff_removed_bg", "diff_context_bg",
            "diff_line_number", "diff_added_line_number_bg",
            "diff_removed_line_number_bg", "diff_gutter",
        ]),
        ("ux", [
            "tool_running", "tool_success", "tool_error", "tool_denied",
            "mention_file", "search_hit", "thinking_time", "queue_badge",
            "footer_active",
        ]),
        ("agents", [
            "agent.build", "agent.plan", "agent.general", "agent.explore",
            "agent.test",
        ]),
    ],
]

# Per-agent badge overrides, stored as `agent.<name>` palette keys. Shows as
# its own Advanced section so users can pin e.g. build=orange without
# touching the core primary/secondary/accent roles.
AGENT_KEYS: list[str] = [
    "agent.build", "agent.plan", "agent.general", "agent.explore", "agent.test",
]

AGENT_DEFAULTS: dict[str, str] = {
    "agent.build": "primary",
    "agent.plan": "secondary",
    "agent.general": "accent",
    "agent.explore": "info",
    "agent.test": "success",
}


def _agent_default(key: str, colors: dict[str, str]) -> str:
    role = AGENT_DEFAULTS.get(key, "primary")
    return colors.get(role, OPENSE_DARK[role])


def _wid(key: str) -> str:
    """Widget-id-safe form of a palette key (`agent.build` → `agent__build`)."""
    return key.replace(".", "__")


def _unwid(suffix: str) -> str:
    return suffix.replace("__", ".")


def advanced_keys() -> list[str]:
    return [k for k in OPENSE_DARK if k not in CURATED_KEYS]


def _grouped_keys() -> list[tuple[str, list[str]]]:
    """All rows: left pane groups, then right pane groups. Any key missing
    from _PANES (a future OPENSE_DARK addition) falls into a trailing
    "extra" group, never lost."""
    seen: set[str] = set()
    out: list[tuple[str, list[str]]] = []
    for pane in _PANES:
        for title, keys in pane:
            present = [k for k in keys if k in OPENSE_DARK or k in AGENT_KEYS]
            seen.update(present)
            if present:
                out.append((title, present))
    extra = [k for k in OPENSE_DARK if k not in seen]
    if extra:
        out.append(("extra", extra))
    return out




_EDITOR_CSS = """
ThemeEditor {
    align: center middle;
}
#theme-editor-box {
    width: 110;
    max-width: 98%;
    max-height: 96%;
    height: auto;
    background: $surface;
    border: round $accent;
    padding: 0 1;
}
#theme-editor-title {
    text-style: bold;
    height: 1;
    margin-bottom: 1;
}
#theme-editor-name {
    border: none;
    background: $background;
    margin-bottom: 1;
    padding: 0 1;
}
#theme-editor-rows {
    height: auto;
    max-height: 34;
    overflow-x: auto;
    overflow-y: auto;
    background: transparent;
    scrollbar-size-vertical: 1;
    scrollbar-size-horizontal: 1;
}
#theme-editor-list {
    height: auto;
    width: 100%;
}
.theme-editor-col {
    height: auto;
    width: 100%;
    background: transparent;
}
.theme-editor-group {
    text-style: bold;
    color: $accent;
    height: 1;
    margin: 1 0 0 0;
}
.theme-editor-group-first {
    margin: 0 0 0 0;
}
.theme-editor-row {
    height: 1;
    background: transparent;
}
.theme-editor-key {
    width: 22;
    color: $text-muted;
}
.theme-editor-hex {
    width: 12;
    height: 1;
    border: none;
    background: $background;
    padding: 0 1;
}
.theme-editor-hex--bad {
    color: $error;
    text-style: bold;
}
.theme-editor-swatch {
    width: 5;
}
#theme-editor-section {
    text-style: bold;
    height: 1;
    margin: 1 0 0 0;
}
#theme-editor-preview-box {
    background: $surface;
    border: round $accent;
    padding: 0 1;
    margin-bottom: 1;
}
#theme-editor-preview-title {
    color: $text-muted;
    height: 1;
}
#theme-editor-preview {
    height: auto;
    max-height: 14;
    overflow-x: auto;
    overflow-y: auto;
    background: $background;
    padding: 0 1;
    scrollbar-size-vertical: 1;
    scrollbar-size-horizontal: 1;
}
#theme-editor-error {
    height: 1;
    color: $error;
}
#theme-editor-actions {
    height: 3;
    align: center middle;
    background: $surface;
}
#theme-editor-actions Button {
    height: 3;
    min-width: 12;
    padding: 0 2;
    margin: 0 1;
    border: heavy $accent;
    background: transparent;
    color: $text;
}
"""


def _preview_text(colors: dict[str, str]) -> str:
    """Full-TUI mock: every color family rendered the way the app uses it.

    Chat (user bubble, assistant row, reasoning, tool running/ok/err/denied),
    status/inline signals (error/warning/info/success, search hit, queue
    badge, footer blink, streaming cursor), full syntax + markdown spread,
    and a diff block — plus one badge row per agent so `agent.*` overrides
    preview live as you type.
    """
    t = Theme(name="preview", colors=colors)
    sw = lambda k: t.c(k)  # noqa: E731
    agents = ("build", "plan", "general", "explore", "test")
    agent_badges = " ".join(f"[{t.agent_color(a)}]▣ {a}[/]" for a in agents)
    return (
        f"[on {sw('background_panel')}] [{sw('user_bubble')}]you:[/] [{sw('text')}]hello, theme me up[/] [/]\n"
        f"[{t.agent_color('build')}]▣ Build[/] [{sw('text')}]sure — patching now[/] [{sw('streaming_cursor')}]▌[/]\n"
        f"[{sw('thinking_time')}]↳ reasoning 12s · reading files…[/]\n"
        f"[{sw('tool_running')}]◷ edit …running[/]  [{sw('tool_success')}]✓ read ok[/]  "
        f"[{sw('tool_error')}]✗ tests failed[/]  [{sw('tool_denied')}]⊘ deploy denied[/]\n"
        f"[{sw('syntax_keyword')}]def[/] [{sw('syntax_function')}]run[/]([{sw('syntax_variable')}]self[/], "
        f"[{sw('syntax_string')}]\"hi\"[/], [{sw('syntax_number')}]42[/]) [{sw('syntax_operator')}]→[/] "
        f"[{sw('syntax_type')}]Theme[/][{sw('syntax_punctuation')}],[/] [{sw('syntax_comment')}]# note[/]\n"
        f"[{sw('markdown_heading')}]# heading[/]  [{sw('markdown_link')}]link[/] "
        f"[{sw('markdown_link_text')}]link-text[/]  [{sw('markdown_code')}]`code`[/]  "
        f"[{sw('markdown_quote')}] Préface[/]\n"
        f"[{sw('markdown_strong')}]**bold**[/] [{sw('markdown_strike')}]~~gone~~[/] "
        f"[{sw('markdown_list_item')}]• item[/] [{sw('markdown_list_enumeration')}]1. one[/] "
        f"[{sw('markdown_todo_done')}]☑ done[/] [{sw('markdown_todo_open')}]☐ todo[/]\n"
        f"[on {sw('diff_added')}] + added line [/]  [on {sw('diff_removed')}] - removed line [/]  "
        f"[{sw('diff_context')}] context [/] [{sw('diff_hunk_header')}]@@ hunk @@[/]\n"
        f"[on {sw('background')}]"
        f"[{sw('diff_gutter')}]│[/][{sw('diff_line_number')}] 12 [/]"
        f"[on {sw('diff_added_bg')}][{sw('diff_highlight_added')}]++new++[/][/] "
        f"[{sw('diff_gutter')}]│[/][{sw('diff_line_number')}] 13 [/]"
        f"[{sw('diff_context')}] ctx [/]"
        f"[on {sw('diff_removed_bg')}][{sw('diff_highlight_removed')}]-—old—-[/][/] "
        f"[{sw('diff_gutter')}]│[/][on {sw('diff_added_line_number_bg')}] + [/][on {sw('diff_removed_line_number_bg')}] - [/]"
        f"[/]\n"
        f"[{sw('error')}]error[/] [{sw('warning')}]warning[/] [{sw('info')}]info[/] [{sw('success')}]ok[/]  "
        f"[on {sw('search_hit')}] hit [/] [{sw('queue_badge')}]queued 2[/] [{sw('footer_active')}]● live[/] "
        f"[{sw('mention_file')}]@file[/]\n"
        f"[on {sw('background_element')}] status · {sw('border')}border[/] "
        f"[{sw('border_active')}]■[/][{sw('border_subtle')}]■[/] "
        f"[on {sw('primary')}] Save [/] [on {sw('accent')}] ✕ [/]\n"
        f"{agent_badges}"
    )


class ThemeEditor(ModalScreen[tuple[str | None, str, dict[str, str]] | None]):
    """Edit a palette. Dismisses with (old_name|None, name, palette) or None."""

    CSS = _EDITOR_CSS
    BINDINGS = [Binding("escape", "cancel", "Cancel", show=False)]

    def __init__(
        self,
        name: str = "",
        base: str = "opencode",
        palette: dict[str, str] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._old_name = name or ""
        start = dict(get_theme(base).colors)
        if palette:
            start.update(palette)
        self._values: dict[str, str] = {k: start.get(k, OPENSE_DARK[k]) for k in OPENSE_DARK}
        for key in AGENT_KEYS:
            self._values.setdefault(key, _agent_default(key, start))

    def compose(self) -> ComposeResult:
        # ponytail: LAZY inputs — rows open as statics (~2.4s for 67), each
        # row swaps its value Static for a real Input on focus/tab/click, so
        # typing still works everywhere. 67 live Inputs cost ≈ 4-7s on-device
        # (each Input pays cursor-blink timers + validators on mount).
        # Single full list (main, syntax, markdown, diff, ux, agents) with the
        # live preview pinned at the top — everything scrolls in one column.
        with Vertical(id="theme-editor-box"):
            title = "Edit theme" if self._old_name else "New custom theme"
            yield Static(f"  {title} — type #rrggbb codes · Tab moves · Esc cancels  ", id="theme-editor-title")
            yield Input(
                value=self._old_name,
                placeholder="name (lowercase letters, digits, dashes)",
                id="theme-editor-name",
            )
            with Vertical(id="theme-editor-preview-box"):
                yield Static("  live preview — follows your typing  ", id="theme-editor-preview-title")
                yield Static("  opening…  ", id="theme-editor-preview")
                yield Static("", id="theme-editor-error")
            with ScrollableContainer(id="theme-editor-rows"):
                with Vertical(id="theme-editor-list"):
                    for gtitle, gkeys in _grouped_keys():
                        yield Static(f"  {gtitle}  ", classes="theme-editor-group")
                        for key in gkeys:
                            with Horizontal(classes="theme-editor-row"):
                                yield Static(key, classes="theme-editor-key")
                                yield Static(self._row_value(key), id=f"theme-val-{_wid(key)}", classes="theme-editor-hex")
                                yield Static("    ", classes="theme-editor-swatch", id=f"theme-sw-{_wid(key)}")
            with Horizontal(id="theme-editor-actions"):
                yield Button("Save", id="theme-editor-save", variant="default")
                yield Button("Cancel", id="theme-editor-cancel", variant="default")

    def _swap_to_input(self, key: str) -> None:
        """Replace a row's value Static with a real editable Input."""
        try:
            old = self.query_one(f"#theme-val-{_wid(key)}", Static)
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/theme_editor.py:375")
            return
        try:
            row = old.parent
            inp = Input(value=self._row_value(key), id=f"theme-hex-{_wid(key)}", classes="theme-editor-hex")
            row.mount(inp, before=old)
            old.remove()
            inp.focus()
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/theme_editor.py:383")
            pass

    def on_click(self, event: Any) -> None:
        try:
            w = getattr(event, "widget", None) or getattr(event, "control", None)
            wid = getattr(w, "id", "") or ""
            if wid.startswith("theme-val-"):
                self._swap_to_input(_unwid(wid[len("theme-val-"):]))
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/theme_editor.py:392")
            pass

    async def on_key(self, event: Any) -> None:
        try:
            if getattr(event, "key", "") == "tab":
                focused = self.focused
                fid = getattr(focused, "id", "") or ""
                if fid.startswith("theme-hex-"):
                    event.stop()
                    cur = _unwid(fid[len("theme-hex-"):])
                    self._swap_back(cur)
                    nxt = self._next_key(cur)
                    if nxt:
                        self._swap_to_input(nxt)
                    return
                if fid == "theme-editor-name":
                    first = self._all_row_keys()
                    if first:
                        event.stop()
                        self._swap_to_input(first[0])
                    return
            elif getattr(event, "key", "") == "escape":
                focused = self.focused
                fid = getattr(focused, "id", "") or ""
                if fid.startswith("theme-hex-"):
                    self._swap_back(_unwid(fid[len("theme-hex-"):]))
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/theme_editor.py:419")
            pass

    def _next_key(self, cur: str) -> str | None:
        keys = self._all_row_keys()
        try:
            i = keys.index(cur)
            if i + 1 < len(keys):
                return keys[i + 1]
        except ValueError as _oc_exc:
            swallow(_oc_exc, "opencode_py/tui/theme_editor.py:433")
            pass
        return None

    def on_mount(self) -> None:
        self._paint_swatches()
        try:
            self.query_one("#theme-editor-name", Input).focus()
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/theme_editor.py:436")
            pass
        # Preview markup (~1300 chars of nested styles) parses slowly — fill
        # it one frame after open so rows appear first, preview streams in.
        try:
            self.set_timer(0.05, self._refresh_preview)
        except Exception:
            self._refresh_preview()

    def _all_row_keys(self) -> list[str]:
        return [k for _, gkeys in _grouped_keys() for k in gkeys]

    def _row_value(self, key: str) -> str:
        if key in self._values:
            return self._values[key]
        if key in AGENT_KEYS:
            return _agent_default(key, self._values)
        return OPENSE_DARK.get(key, "#ffffff")

    def _paint_one(self, key: str) -> None:
        try:
            sw = self.query_one(f"#theme-sw-{_wid(key)}", Static)
            val = self._row_value(key)
            sw.styles.background = val if is_valid_hex(val) else "#000000"
            sw.update("    ")
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/theme_editor.py:461")
            pass

    def _paint_swatches(self) -> None:
        for key in self._all_row_keys():
            self._paint_one(key)

    def _preview_colors(self) -> dict[str, str]:
        colors = dict(OPENSE_DARK)
        for k, v in self._values.items():
            if k.startswith("agent."):
                continue  # agent overrides resolve after role colors below
            if is_valid_hex(v):
                try:
                    colors[k] = normalize_hex(v)
                except ValueError as _oc_exc:
                    swallow(_oc_exc, "opencode_py/tui/theme_editor.py:483")
                    pass
        for key in AGENT_KEYS:
            raw = self._values.get(key, "")
            if is_valid_hex(raw):
                try:
                    colors[key] = normalize_hex(raw)
                except ValueError as _oc_exc:
                    swallow(_oc_exc, "opencode_py/tui/theme_editor.py:490")
                    pass
            else:
                colors.pop(key, None)  # fall back to the role default
        return colors

    def _refresh_preview(self) -> None:
        try:
            self.query_one("#theme-editor-preview", Static).update(_preview_text(self._preview_colors()))
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/theme_editor.py:492")
            pass

    def _set_error(self, msg: str) -> None:
        try:
            self.query_one("#theme-editor-error", Static).update(msg)
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/theme_editor.py:498")
            pass

    def on_input_changed(self, event: Input.Changed) -> None:
        eid = getattr(event.input, "id", "") or ""
        if eid == "theme-editor-name":
            event.stop()
            return
        if eid.startswith("theme-hex-"):
            key = _unwid(eid[len("theme-hex-"):])
            self._values[key] = event.value
            try:
                event.input.set_class(not is_valid_hex(event.value), "theme-editor-hex--bad")
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/theme_editor.py:511")
                pass
            self._paint_one(key)
            self._refresh_preview()
        event.stop()

    def _swap_back(self, key: str) -> None:
        """Turn a row's Input back into a Static after its value is stored."""
        try:
            inp = self.query_one(f"#theme-hex-{_wid(key)}", Input)
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/theme_editor.py:521")
            return
        try:
            val = self._values.get(key, self._row_value(key))
            row = inp.parent
            st = Static(val, id=f"theme-val-{_wid(key)}", classes="theme-editor-hex")
            row.mount(st, before=inp)
            inp.remove()
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/theme_editor.py:529")
            pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id or ""
        if bid == "theme-editor-save":
            self._save()
        elif bid == "theme-editor-cancel":
            self.dismiss(None)
        event.stop()

    def on_input_submitted(self, event: Any) -> None:
        try:
            if getattr(event.input, "id", "") != "theme-editor-name":
                return
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/theme_editor.py:544")
            return
        event.stop()
        self._save()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def _save(self) -> None:
        try:
            name = self.query_one("#theme-editor-name", Input).value.strip().lower()
        except Exception:
            name = ""
        if not valid_theme_name(name):
            self._set_error("Name: lowercase letters, digits, dashes.")
            return
        # flush any open row Input back into values before validating
        try:
            for inp in self.query(Input):
                eid = getattr(inp, "id", "") or ""
                if eid.startswith("theme-hex-"):
                    self._values[_unwid(eid[len("theme-hex-"):])] = inp.value
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/theme_editor.py:566")
            pass
        bad = [k for k, v in self._values.items() if not is_valid_hex(v)]
        if bad:
            self._set_error(f"Bad hex in: {', '.join(bad[:5])}" + (" …" if len(bad) > 5 else ""))
            return
        palette = {k: normalize_hex(v) for k, v in self._values.items()}
        self.dismiss((self._old_name or None, name, palette))

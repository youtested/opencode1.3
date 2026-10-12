"""Skill picker popup: list skills, add/rename/edit/delete project skills.

Dead-centered, keyboard-driven (arrows + Enter, Esc closes) — mirrors the
agent picker so muscle memory carries over:

- One row per skill: `name  description` (invalid dirs never list: the
  scanner only returns valid SKILL.md files).
- Enter on a skill row opens the editor (description + body, Save writes
  SKILL.md in place, Esc back).
- `＋ Add skill` row (or the Add button) opens a name+description dialog;
  Save creates `.opencode/skills/<name>/SKILL.md` with a starter body.
- `Ctrl+N` renames the highlighted skill (dir + frontmatter name move).
- `Ctrl+D` (or Delete button) asks for confirmation, then removes the dir.
- Buttons: Add / Rename / Delete / Edit / Close (two rows, never clipped).

Never touches the filesystem itself: all mutations go through callbacks
the app wires (`on_add`, `on_rename`, `on_delete`, `on_edit`, `on_load`).
Dismisses with the picked skill name (or None)."""

from __future__ import annotations

from ..util.log import swallow
from typing import Callable

from rich.markup import escape
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.events import Key
from textual.screen import ModalScreen
from textual.widgets import Button, Input, OptionList, Static
from textual.widgets.option_list import Option

_SKILL_PICKER_CSS = """
SkillPicker {
    align: center middle;
}
#skill-picker-box {
    width: 84;
    max-width: 96%;
    min-width: 68;
    max-height: 88%;
    height: auto;
    background: $surface;
    border: round $accent;
    padding: 1 2;
}
#skill-picker-title {
    text-style: bold;
    color: $accent;
    height: 1;
    margin-bottom: 1;
}
#skill-picker-hint {
    color: $text-muted;
    height: 1;
    margin-top: 1;
}
#skill-picker-list {
    height: auto;
    max-height: 20;
    background: transparent;
    border: none;
    padding: 0;
    scrollbar-size-vertical: 1;
    scrollbar-size-horizontal: 0;
}
#skill-picker-actions {
    height: 3;
    align: center middle;
    background: $surface;
    padding: 0;
    margin-top: 1;
}
#skill-picker-actions Button {
    height: 3;
    min-width: 8;
    padding: 0 1;
    margin: 0;
    border: heavy $accent;
    background: transparent;
    color: $text;
}
"""


class SkillNameDialog(ModalScreen[tuple[str, str] | None]):
    """Name + description dialog for add/rename. Dismisses with
    (name, description), or None on Esc/Cancel."""

    CSS = """
    SkillNameDialog {
        align: center middle;
    }
    #skill-name-box {
        width: 56;
        max-width: 92%;
        height: auto;
        background: $surface;
        border: round $accent;
        padding: 1 2;
    }
    #skill-name-title {
        text-style: bold;
        color: $accent;
        height: 1;
        margin-bottom: 1;
    }
    #skill-name-input, #skill-desc-input {
        margin-bottom: 1;
        border: none;
        background: $background;
        padding: 0 1;
    }
    #skill-name-actions {
        height: 3;
        align: center middle;
        background: $surface;
    }
    #skill-name-actions Button {
        height: 3;
        min-width: 12;
        padding: 0 2;
        margin: 0 1;
        border: heavy $accent;
        background: transparent;
        color: $text;
    }
    """

    def __init__(
        self,
        old: str | None = None,
        description: str = "",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._old = old
        self._desc_value = description or ""

    def compose(self) -> ComposeResult:
        title = "Rename skill" if self._old else "Add skill"
        with Vertical(id="skill-name-box"):
            yield Static(f"  {title}  ", id="skill-name-title")
            yield Input(
                value=self._old or "",
                placeholder="Name (lowercase, dashes)",
                id="skill-name-input",
            )
            yield Input(
                value=self._desc_value,
                placeholder="Short description",
                id="skill-desc-input",
            )
            with Horizontal(id="skill-name-actions"):
                yield Button("Save", id="skill-name-save", variant="default")
                yield Button("Cancel", id="skill-name-cancel", variant="default")

    def on_mount(self) -> None:
        try:
            inp = self.query_one("#skill-name-input", Input)
            inp.focus()
            inp.cursor_position = len(inp.value)
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/skill_picker.py:163")
            pass

    def _done(self) -> None:
        try:
            name = self.query_one("#skill-name-input", Input).value.strip().lower()
        except Exception:
            name = ""
        try:
            desc = self.query_one("#skill-desc-input", Input).value.strip()
        except Exception:
            desc = ""
        if not name:
            try:
                self.app.notify("Skill name can't be empty.", severity="warning")
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/skill_picker.py:178")
                pass
            return
        self.dismiss((name, desc))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "skill-name-save":
            self._done()
        else:
            self.dismiss(None)
        event.stop()

    def on_input_submitted(self, event: object) -> None:
        try:
            if getattr(event, "id", "") not in ("skill-name-input", "skill-desc-input"):
                eid = getattr(getattr(event, "input", None), "id", "")
                if eid not in ("skill-name-input", "skill-desc-input"):
                    return
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/skill_picker.py:196")
            return
        event.stop()  # type: ignore[attr-defined]
        self._done()

    def on_key(self, event: Key) -> None:
        if event.key == "escape":
            self.dismiss(None)
            event.stop()


class SkillEditScreen(ModalScreen[tuple[str, str] | None]):
    """Edit a skill's description + body. Dismisses with (description, body),
    or None on Esc/Cancel — the app writes SKILL.md via on_edit."""

    CSS = """
    SkillEditScreen {
        align: center middle;
    }
    #skill-edit-box {
        width: 76;
        max-width: 94%;
        max-height: 90%;
        height: auto;
        background: $surface;
        border: round $accent;
        padding: 1 2;
    }
    #skill-edit-title {
        text-style: bold;
        color: $accent;
        height: 1;
        margin-bottom: 1;
    }
    #skill-edit-desc {
        margin-bottom: 1;
        border: none;
        background: $background;
        padding: 0 1;
    }
    #skill-edit-body {
        height: 16;
        max-height: 40%;
        border: none;
        background: $background;
        padding: 0 1;
    }
    #skill-edit-actions {
        height: 3;
        align: center middle;
        background: $surface;
        margin-top: 1;
    }
    #skill-edit-actions Button {
        height: 3;
        min-width: 12;
        padding: 0 2;
        margin: 0 1;
        border: heavy $accent;
        background: transparent;
        color: $text;
    }
    """

    def __init__(
        self,
        name: str = "",
        description: str = "",
        body: str = "",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._name = name
        self._desc_value = description or ""
        self._body_value = body or ""

    def compose(self) -> ComposeResult:
        from textual.widgets import TextArea

        with Vertical(id="skill-edit-box"):
            yield Static(f"  Edit skill: {escape(self._name)}  ", id="skill-edit-title")
            yield Input(
                value=self._desc_value,
                placeholder="Short description",
                id="skill-edit-desc",
            )
            yield TextArea(
                self._body_value,
                id="skill-edit-body",
            )
            with Horizontal(id="skill-edit-actions"):
                yield Button("Save", id="skill-edit-save", variant="default")
                yield Button("Cancel", id="skill-edit-cancel", variant="default")

    def on_mount(self) -> None:
        try:
            self.query_one("#skill-edit-desc", Input).focus()
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/skill_picker.py:293")
            pass

    def _done(self) -> None:
        try:
            desc = self.query_one("#skill-edit-desc", Input).value.strip()
        except Exception:
            desc = ""
        try:
            from textual.widgets import TextArea

            body = self.query_one("#skill-edit-body", TextArea).text
        except Exception:
            body = ""
        if not (body or "").strip():
            try:
                self.app.notify("Skill body can't be empty.", severity="warning")
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/skill_picker.py:310")
                pass
            return
        self.dismiss((desc, body))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "skill-edit-save":
            self._done()
        else:
            self.dismiss(None)
        event.stop()

    def on_key(self, event: Key) -> None:
        if event.key == "escape":
            self.dismiss(None)
            event.stop()


class SkillBodyView(ModalScreen[None]):
    """Read-only skill body viewer. Esc closes."""

    CSS = """
    SkillBodyView {
        align: center middle;
    }
    #skill-body-box {
        width: 76;
        max-width: 94%;
        max-height: 86%;
        height: auto;
        background: $surface;
        border: round $accent;
        padding: 1 2;
    }
    #skill-body-title {
        text-style: bold;
        color: $accent;
        height: 1;
        margin-bottom: 1;
    }
    #skill-body-scroll {
        height: auto;
        max-height: 30;
        background: transparent;
    }
    """

    def __init__(self, name: str = "", body: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self._name = name
        self._body = body

    def compose(self) -> ComposeResult:
        from textual.containers import VerticalScroll

        with Vertical(id="skill-body-box"):
            yield Static(f"  {escape(self._name)} — Esc closes  ", id="skill-body-title")
            with VerticalScroll(id="skill-body-scroll"):
                yield Static(escape(self._body) or "  (empty)  ")
            with Horizontal(id="skill-picker-actions"):
                yield Button("Close", id="skill-body-close", variant="default")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(None)
        event.stop()

    def on_key(self, event: Key) -> None:
        if event.key == "escape":
            self.dismiss(None)
            event.stop()


class SkillPicker(ModalScreen[str | None]):
    """List / add / rename / edit / delete skills. Dismisses with the picked
    skill name (Enter edits it) or None on Esc/Close."""

    CSS = _SKILL_PICKER_CSS
    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=False),
        Binding("ctrl+d", "delete", "Delete", show=False),
        Binding("ctrl+n", "rename", "Rename", show=False),
    ]

    def __init__(
        self,
        skills: list[tuple[str, str]] | None = None,
        on_add: Callable[[str, str], str | None] | None = None,
        on_delete: Callable[[str], bool] | None = None,
        on_rename: Callable[[str, str], str | None] | None = None,
        on_edit: Callable[[str, str, str], str | None] | None = None,
        on_load: Callable[[str], tuple[str, str] | None] | None = None,
        on_view: Callable[[str], str | None] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._skills = list(skills or [])
        self.on_add = on_add
        self.on_delete = on_delete
        self.on_rename = on_rename
        self.on_edit = on_edit
        self.on_load = on_load
        self.on_view = on_view  # legacy read-only viewer (unused by default)

    # -- rows ------------------------------------------------------------
    def _options(self) -> list[Option]:
        opts: list[Option] = []
        for name, desc in self._skills:
            opts.append(
                Option(
                    f"{escape(name)}  [dim]{escape(desc)}[/]",
                    id=f"__skill__{name}",
                )
            )
        opts.append(Option("＋ Add skill", id="__add__"))
        return opts

    def reload_skills(self, skills: list[tuple[str, str]]) -> None:
        """Reload rows (after add/delete/rename) keeping the popup open."""
        self._skills = list(skills)
        try:
            lst = self.query_one("#skill-picker-list", OptionList)
            lst.clear_options()
            lst.add_options(self._options())
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/skill_picker.py:433")
            pass

    def compose(self) -> ComposeResult:
        with Vertical(id="skill-picker-box"):
            yield Static("  Skills — Enter edits · Esc closes  ", id="skill-picker-title")
            yield OptionList(*self._options(), id="skill-picker-list")
            yield Static(
                "↑/↓ move · Enter edit · ＋ add · Ctrl+N rename · Ctrl+D delete",
                id="skill-picker-hint",
            )
            with Horizontal(id="skill-picker-actions"):
                yield Button("Add", id="skill-add", variant="default")
                yield Button("Rename", id="skill-rename", variant="default")
                yield Button("Delete", id="skill-delete", variant="default")
                yield Button("Edit", id="skill-edit", variant="default")
                yield Button("Close", id="skill-close", variant="default")

    def on_mount(self) -> None:
        try:
            self.query_one("#skill-picker-list", OptionList).focus()
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/skill_picker.py:454")
            pass

    # -- add/rename dialog -----------------------------------------------
    def _open_name_dialog(self, old: str | None = None) -> None:
        desc = ""
        if old is not None:
            desc = next((d for n, d in self._skills if n == old), "")
        try:
            self.app.push_screen(
                SkillNameDialog(old=old, description=desc),
                lambda result: self._after_name_dialog(old, result),
            )
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/skill_picker.py:467")
            pass

    def _after_name_dialog(
        self, old: str | None, result: tuple[str, str] | None
    ) -> None:
        self._refocus_list()
        if not result:
            return
        name, desc = result
        name = (name or "").strip().lower().replace(" ", "-")
        if old is None:
            if not name:
                return
            if any(n == name for n, _d in self._skills):
                try:
                    self.app.notify(f"Skill '{name}' already exists.", severity="warning")
                except Exception as _oc_exc:
                    swallow(_oc_exc, "tui/skill_picker.py:484")
                    pass
                return
            if self.on_add is not None:
                try:
                    self.on_add(name, (desc or "").strip())
                except Exception as _oc_exc:
                    swallow(_oc_exc, "tui/skill_picker.py:490")
                    pass
            return
        if name and name != old and self.on_rename is not None:
            try:
                self.on_rename(old, name)
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/skill_picker.py:496")
                pass

    def _highlighted_skill(self) -> str | None:
        try:
            lst = self.query_one("#skill-picker-list", OptionList)
            opt = lst.highlighted_option
        except Exception:
            return None
        if opt is None or opt.id is None:
            return None
        oid = str(opt.id)
        if oid == "__add__":
            return None
        if oid.startswith("__skill__"):
            return oid[len("__skill__"):]
        return None

    def _refocus_list(self) -> None:
        try:
            self.query_one("#skill-picker-list", OptionList).focus()
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/skill_picker.py:517")
            pass

    def _edit_current(self) -> None:
        name = self._highlighted_skill()
        if name is None:
            return
        desc = next((d for n, d in self._skills if n == name), "")
        body: str | None = None
        if self.on_load is not None:
            try:
                loaded = self.on_load(name)
            except Exception:
                loaded = None
            if loaded is not None:
                desc, body = loaded
        if body is None and self.on_view is not None:
            try:
                body = self.on_view(name)
            except Exception:
                body = None
        try:
            self.app.push_screen(
                SkillEditScreen(name=name, description=desc, body=body or ""),
                lambda result: self._after_edit(name, result),
            )
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/skill_picker.py:543")
            pass

    def _after_edit(self, name: str, result: tuple[str, str] | None) -> None:
        self._refocus_list()
        if not result or self.on_edit is None:
            return
        desc, body = result
        try:
            self.on_edit(name, (desc or "").strip(), body)
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/skill_picker.py:553")
            pass

    def _view_current(self) -> None:
        # Enter edits now (was: read-only viewer). Kept as a fallback path
        # for callers that only wire on_view.
        if self.on_edit is not None or self.on_load is not None:
            self._edit_current()
            return
        name = self._highlighted_skill()
        if name is None or self.on_view is None:
            return
        try:
            body = self.on_view(name)
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/skill_picker.py:567")
            return
        if body is None:
            return
        try:
            self.app.push_screen(SkillBodyView(name=name, body=body))
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/skill_picker.py:573")
            pass

    # -- selection --------------------------------------------------------
    def on_option_list_option_selected(self, event: object) -> None:
        opt = getattr(event, "option", None)
        oid = str(getattr(opt, "id", "") or "")
        try:
            event.stop()  # type: ignore[attr-defined]
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/skill_picker.py:582")
            pass
        if oid == "__add__":
            self._open_name_dialog(None)
            return
        if oid.startswith("__skill__"):
            self._view_current()
            return

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "skill-close":
            self.dismiss(None)
        elif bid == "skill-add":
            self._open_name_dialog(None)
        elif bid == "skill-rename":
            self.action_rename()
        elif bid == "skill-delete":
            self.action_delete()
        elif bid == "skill-edit":
            self._edit_current()
        event.stop()

    def on_key(self, event: Key) -> None:
        if event.key == "escape":
            self.dismiss(None)
            event.stop()

    def action_delete(self) -> None:
        name = self._highlighted_skill()
        if name is None or self.on_delete is None:
            return
        try:
            from .confirm_dialog import ConfirmDialog
        except Exception:
            try:
                self.on_delete(name)
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/skill_picker.py:619")
                pass
            return

        def _after(ok: bool | None) -> None:
            self._refocus_list()
            if not ok:
                return
            try:
                self.on_delete(name)
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/skill_picker.py:629")
                pass

        try:
            self.app.push_screen(
                ConfirmDialog(
                    message=f"Delete skill '{name}' and its SKILL.md?",
                    title="Delete skill",
                ),
                _after,
            )
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/skill_picker.py:640")
            pass

    def action_rename(self) -> None:
        name = self._highlighted_skill()
        if name is None:
            return
        self._open_name_dialog(name)

    def action_cancel(self) -> None:
        self.dismiss(None)

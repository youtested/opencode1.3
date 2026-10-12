"""Skill manager for the Textual TUI.

Split out of :mod:`opencode_py.tui.app` as a mixin; method bodies are unchanged.
"""
from __future__ import annotations

from ..util.log import swallow
from .input_bar import InputBar

class SkillsMixin:

    def _open_skill_picker(self) -> None:
        """Dead-centered skill manager: view/add/rename/delete SKILL.md skills."""
        from ..tools.skill import list_skills
        from .skill_picker import SkillPicker

        try:
            skills = [(s.name, s.description) for s in list_skills(fresh=True)]
        except Exception:
            skills = []
        picker = SkillPicker(
            skills=skills,
            on_add=self._skill_add,
            on_delete=self._skill_delete,
            on_rename=self._skill_rename,
            on_edit=self._skill_edit,
            on_load=self._skill_load,
            on_view=self._skill_view,
        )
        self._skill_picker = picker
        self.push_screen(picker, self._on_skill_picked)


    def _on_skill_picked(self, result: str | None) -> None:
        self._skill_picker = None
        try:
            self.query_one(InputBar).input.focus()
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/skills.py:37")
            pass


    def _refresh_skill_picker(self) -> None:
        from ..tools.skill import list_skills

        picker = getattr(self, "_skill_picker", None)
        if picker is None:
            return
        try:
            picker.reload_skills([(s.name, s.description) for s in list_skills(fresh=True)])
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/skills.py:49")
            pass


    def _skill_add(self, name: str, description: str) -> str | None:
        from ..tools.skill import create_skill

        try:
            skill = create_skill(name, description)
        except ValueError as e:
            self.notify(f"Can't add skill: {e}", severity="warning")
            return str(e)
        self._refresh_skill_picker()
        self.notify(f"Skill '{skill.name}' added.")
        return None


    def _skill_delete(self, name: str) -> bool:
        from ..tools.skill import delete_skill

        try:
            delete_skill(name)
        except KeyError:
            return False
        except OSError as e:
            self.notify(str(e), severity="error")
            return False
        self._refresh_skill_picker()
        self.notify(f"Skill '{name}' deleted.")
        return True


    def _skill_rename(self, old: str, new: str) -> str | None:
        from ..tools.skill import rename_skill

        try:
            skill = rename_skill(old, new)
        except (ValueError, KeyError) as e:
            self.notify(f"Can't rename: {e}", severity="warning")
            return str(e)
        self._refresh_skill_picker()
        self.notify(f"Renamed '{old}' → '{skill.name}'.")
        return None


    def _skill_view(self, name: str) -> str | None:
        from ..tools.skill import _load

        try:
            skill = _load(name)
        except Exception:
            return None
        if skill is None:
            return None
        return skill.body


    def _skill_load(self, name: str) -> tuple[str, str] | None:
        from ..tools.skill import _load

        try:
            skill = _load(name)
        except Exception:
            return None
        if skill is None:
            return None
        return skill.description, skill.body


    def _skill_edit(self, name: str, description: str, body: str) -> str | None:
        from ..tools.skill import update_skill

        try:
            skill = update_skill(name, description, body)
        except (ValueError, KeyError, OSError) as e:
            self.notify(f"Can't save skill: {e}", severity="warning")
            return str(e)
        self._refresh_skill_picker()
        self.notify(f"Skill '{skill.name}' saved.")
        return None

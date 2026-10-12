"""AGENTS.md manager for the Textual TUI.

Split out of :mod:`opencode_py.tui.app` as a mixin; method bodies are unchanged.
"""
from __future__ import annotations

from ..util.log import swallow
from typing import Any
from pathlib import Path

class AgentMdMixin:

    # -- agent.md manager ------------------------------------------------
    def _open_agent_md(self) -> None:
        """Agent.md files manager (dead-centered popup)."""
        from ..permission import agent_md_groups as _groups
        from .agent_md_popup import AgentMdPopup

        popup = AgentMdPopup(
            groups=_groups(self.cfg),
            on_open=self._open_agent_md_file,
            on_edit=self._edit_agent_md,
            on_add=self._add_agent_md,
            on_delete=self._delete_agent_md,
            on_save_all=self._save_agent_md_all,
            on_rename=self._rename_agent_md,
            on_preview=self._preview_agent_prompt,
        )
        self._agent_md_popup = popup

        def _back_from_md(_result: Any) -> None:
            self._agent_md_popup = None
            # Esc from the manager left focus on the Agent.md button, so
            # ↑/↓ in the picker went nowhere (frozen highlight). Hand it
            # back to the agent list, like the permission-editor return.
            try:
                from textual.widgets import OptionList as _OL

                picker = getattr(self, "_agent_picker", None)
                if picker is not None:
                    picker.query_one("#agent-picker-list", _OL).focus()
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/agent_md.py:41")
                pass

        self.push_screen(popup, _back_from_md)


    def _preview_agent_prompt(self, agent: str) -> None:
        """Show the BYTE-EXACT system prompt `agent` will send.

        Built by the SAME labeled_prompt_parts() the sender uses — every
        block (base, environment, AGENTS.md files, shared+own .md, memory,
        skills, config override) in send order with its source label, plus
        chars + ~tokens. Read-only; Esc/Close goes back to the manager."""
        from ..agent.system import labeled_prompt_parts as _parts
        from ..globals import resolve_worktree
        from .agent_md_popup import AgentPromptPreview

        try:
            directory = Path(str(self.directory))
        except Exception:
            from pathlib import Path as _P
            directory = _P(".")
        try:
            worktree = resolve_worktree(directory)
        except Exception:
            worktree = directory
        try:
            engine = self._active_engine()
            provider_id = str(getattr(engine, "provider_id", "") or self.cfg.provider)
            model_id = str(getattr(engine, "model_id", "") or self.cfg.model)
        except Exception:
            provider_id, model_id = str(self.cfg.provider), str(self.cfg.model)
        try:
            blocks = _parts(
                directory=directory, worktree=worktree,
                provider_id=provider_id, model_id=model_id,
                cfg=self.cfg, agent=agent,
            )
        except Exception as e:
            self.notify(f"Preview failed: {e}", severity="error")
            return

        def _after_preview(_result: Any) -> None:
            try:
                from textual.widgets import OptionList as _OL
                popup = getattr(self, "_agent_md_popup", None)
                if popup is not None:
                    popup.query_one("#agent-md-list", _OL).focus()
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/agent_md.py:89")
                pass

        self.push_screen(AgentPromptPreview(agent=agent, blocks=blocks), _after_preview)


    def _refresh_agent_md(self) -> None:
        from ..permission import agent_md_groups as _groups

        popup = getattr(self, "_agent_md_popup", None)
        if popup is None:
            return
        try:
            popup.reload(_groups(self.cfg))
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/agent_md.py:103")
            pass


    @staticmethod
    def _is_shared(agent: str) -> bool:
        try:
            from ..permission import SHARED_AGENT as _SA
            return str(agent) == str(_SA)
        except Exception:
            return str(agent) == "shared"


    def _write_agent_md(self, agent: str, name: str, content: str) -> None:
        """Store one .md file for an agent as a REAL file:
        `<config>/agents/<agent>/<name>.md` (`shared` routes to the shared
        `<config>/agents/AGENT.md`, no config marker). Otherwise the config
        keeps only a marker so the agent lists the file (content on disk)."""
        if self._is_shared(agent):
            from ..permission import shared_md_path as _spath
            try:
                sp = _spath()
                sp.parent.mkdir(parents=True, exist_ok=True)
                sp.write_text(content, encoding="utf-8")
            except OSError:
                self.notify("Can't write 'AGENT.md'.", severity="warning")
                return
            self._refresh_agent_md()
            return
        from ..permission import agent_md_path as _mpath

        try:
            p = _mpath(agent, name)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
        except OSError:
            self.notify(f"Can't write '{name}'.", severity="warning")
            return
        # marker in the spec so customs survive reloads even when empty.
        # bug 3 fix: key by the NORMALIZED filename (p.name, with .md
        # suffix) — the raw input ("guide" vs "guide.md") used to create
        # duplicate rows while the file lived at one path.
        agents = dict(getattr(self.cfg, "agents", None) or {})
        spec = dict(agents.get(agent) or {})
        md = spec.get("md")
        if not isinstance(md, dict):
            # legacy list shape or missing: normalize to a name set
            names = set()
            if isinstance(md, list):
                for item in md:
                    if isinstance(item, dict) and item.get("name"):
                        names.add(str(item["name"]))
                    elif isinstance(item, str):
                        names.add(item)
            normed: dict[str, str] = {}
            for n in names:
                try:
                    normed[str(_mpath(agent, n).name)] = ""
                except Exception:
                    normed[str(n)] = ""
            md = normed
        md[str(p.name)] = ""
        spec["md"] = md
        agents[agent] = spec
        self.cfg.agents = agents
        self._persist_agents()
        self._refresh_agent_md()


    def _open_agent_md_file(self, agent: str, name: str) -> None:
        """Enter on a .md file: read-only viewer (cat) showing the real
        file path plus the content. Edit (button or `e`) opens the content
        editor; Esc/Close goes back to the manager."""
        from .agent_md_popup import AgentMdViewer

        if self._is_shared(agent):
            from ..permission import shared_md_path as _spath
            from ..permission import shared_md_text as _stext
            try:
                path = _spath()
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/agent_md.py:183")
                return
            try:
                current = _stext(self.cfg)
                if path.is_file():
                    current = path.read_text(encoding="utf-8", errors="replace")[:100_000]
            except OSError:
                current = ""
        else:
            from ..permission import agent_md_path as _mpath
            from ..permission import md_entries_for_agent_cfg as _entries
            try:
                path = _mpath(agent, name)
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/agent_md.py:196")
                return
            current = dict(_entries(self.cfg, agent)).get(path.name, dict(_entries(self.cfg, agent)).get(name, ""))
            try:
                if path.exists():
                    current = path.read_text(encoding="utf-8", errors="replace")[:100_000]
            except OSError as _oc_exc:
                swallow(_oc_exc, "opencode_py/tui/agent_md.py:208")
                pass

        def after(result: Any) -> None:
            if result == "edit":
                self._edit_agent_md(agent, path.name)
                return
            try:
                from textual.widgets import OptionList as _OL

                popup = getattr(self, "_agent_md_popup", None)
                if popup is not None:
                    popup.query_one("#agent-md-list", _OL).focus()
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/agent_md.py:215")
                pass

        self.push_screen(
            AgentMdViewer(agent=agent, name=path.name, path=str(path), content=current),
            after,
        )


    def _edit_agent_md(self, agent: str, name: str) -> None:
        """Edit an EXISTING .md file's content (was missing: Enter only
        viewed). Name locked, content editable; Save rewrites the real
        file, Cancel drops."""
        from .agent_md_popup import AgentMdEditor

        if self._is_shared(agent):
            from ..permission import shared_md_path as _spath
            from ..permission import shared_md_text as _stext
            try:
                path = _spath()
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/agent_md.py:235")
                return
            try:
                current = _stext(self.cfg)
                if path.is_file():
                    current = path.read_text(encoding="utf-8", errors="replace")[:100_000]
            except OSError:
                current = ""
        else:
            from ..permission import agent_md_path as _mpath
            from ..permission import md_entries_for_agent_cfg as _entries
            try:
                path = _mpath(agent, name)
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/agent_md.py:248")
                return
            current = dict(_entries(self.cfg, agent)).get(path.name, dict(_entries(self.cfg, agent)).get(name, ""))
            try:
                if path.exists():
                    current = path.read_text(encoding="utf-8", errors="replace")[:100_000]
            except OSError as _oc_exc:
                swallow(_oc_exc, "opencode_py/tui/agent_md.py:263")
                pass

        def on_load_path(path_str: str) -> str | None:
            try:
                from pathlib import Path as _P

                lp = _P(path_str).expanduser()
                if not lp.is_absolute():
                    lp = _P(str(self.directory)) / lp
                return lp.read_text(encoding="utf-8")[:100_000]
            except Exception:
                return None

        def after_edit(result: tuple[str, str] | None) -> None:
            if result:
                _name, content = result
                self._write_agent_md(agent, _name or path.name, content)
                self.notify(f"Saved '{path.name}' for {agent}.")
            try:
                from textual.widgets import OptionList as _OL

                popup = getattr(self, "_agent_md_popup", None)
                if popup is not None:
                    popup.query_one("#agent-md-list", _OL).focus()
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/agent_md.py:279")
                pass

        self.push_screen(
            AgentMdEditor(agent=agent, name=path.name, content=current, on_load_path=on_load_path),
            after_edit,
        )


    def _rename_agent_md(self, agent: str, old: str) -> None:
        """Rename one stored .md file (content moves to the new name)."""
        if self._is_shared(agent):
            self.notify("Shared AGENT.md can't be renamed (only edited).", severity="warning")
            return
        from ..permission import agent_md_path as _mpath
        from ..permission import md_entries_for_agent_cfg as _entries
        from .agent_picker import AgentNameDialog

        def after(result: tuple[str, str] | None) -> None:
            try:
                from textual.widgets import OptionList as _OL

                popup = getattr(self, "_agent_md_popup", None)
                if popup is not None:
                    popup.query_one("#agent-md-list", _OL).focus()
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/agent_md.py:304")
                pass
            if not result:
                return
            new_name = (result[0] or "").strip()
            if not new_name:
                return
            try:
                new_key = str(_mpath(agent, new_name).name)
                old_key = str(_mpath(agent, old).name)
            except Exception:
                new_key, old_key = new_name, old
            if new_key == old_key:
                return
            existing = dict(_entries(self.cfg, agent))
            if new_key in existing:
                self.notify(f"'{new_key}' already exists for {agent}.", severity="warning")
                return
            # rename the REAL file first (file-authoritative); the marker
            # follows only if the rename succeeded
            try:
                old_p = _mpath(agent, old)
                new_p = _mpath(agent, new_name)
                if old_p.exists():
                    new_p.parent.mkdir(parents=True, exist_ok=True)
                    old_p.replace(new_p)
            except OSError as e:
                self.notify(f"Rename failed on disk — {e}.", severity="error")
                return
            agents = dict(getattr(self.cfg, "agents", None) or {})
            spec = dict(agents.get(agent) or {})
            md = dict(spec.get("md") or {})
            md.pop(old, None)
            md.pop(old_p.name if hasattr(old_p, "name") else old, None)
            md[new_p.name if hasattr(new_p, "name") else new_name] = ""
            spec["md"] = md
            agents[agent] = spec
            self.cfg.agents = agents
            err = self._persist_agents()
            self._refresh_agent_md()
            if err:
                self.notify(f"Renamed on disk, but NOT saved — {err}.", severity="error")
                return
            self.notify(f"Renamed '{old}' → '{new_name}'.")

        # bug 2 fix: name-only dialog with a FILE title (was "Rename agent")
        self.push_screen(
            AgentNameDialog(
                old=old,
                title="Rename file",
                hide_description=True,
                name_placeholder="File name (e.g. guide.md)",
            ),
            after,
        )


    def _add_agent_md(self) -> None:
        """＋ Add: pick the owning agent, then the file editor."""
        from ..permission import SHARED_AGENT as _SA
        from ..permission import list_agents as _list_agents
        from .agent_md_popup import AgentMdEditor, AgentPickPopup

        names = [_SA] + [n for n, _d, _c in _list_agents(self.cfg) if n != _SA]

        def on_load_path(path: str) -> str | None:
            try:
                from pathlib import Path as _P

                p = _P(path).expanduser()
                if not p.is_absolute():
                    p = (_P(str(self.directory)) / p)
                return p.read_text(encoding="utf-8")[:100_000]
            except Exception:
                return None

        def after_pick(picked: str | None) -> None:
            if not picked:
                try:
                    from textual.widgets import OptionList as _OL

                    popup = getattr(self, "_agent_md_popup", None)
                    if popup is not None:
                        popup.query_one("#agent-md-list", _OL).focus()
                except Exception as _oc_exc:
                    swallow(_oc_exc, "tui/agent_md.py:388")
                    pass
                return

            def after_edit(result: tuple[str, str] | None) -> None:
                if result:
                    _name, content = result
                    self._write_agent_md(picked, _name, content)
                    self.notify(f"Saved '{_name}' for {picked}.")
                try:
                    from textual.widgets import OptionList as _OL

                    popup = getattr(self, "_agent_md_popup", None)
                    if popup is not None:
                        popup.query_one("#agent-md-list", _OL).focus()
                except Exception as _oc_exc:
                    swallow(_oc_exc, "tui/agent_md.py:403")
                    pass

            self.push_screen(
                AgentMdEditor(agent=picked, on_load_path=on_load_path),
                after_edit,
            )

        self.push_screen(AgentPickPopup(agents=names), after_pick)


    def _delete_agent_md(self, agent: str, name: str) -> bool:
        if self._is_shared(agent):
            self.notify("Shared AGENT.md can't be deleted (only edited).", severity="warning")
            return False
        from ..permission import agent_md_path as _mpath

        try:
            p = _mpath(agent, name)
            if p.exists():
                p.unlink()
                # confirm the file is really gone before touching the index
                if p.exists():
                    self.notify(f"Couldn't delete '{name}' from disk.", severity="error")
                    return False
        except OSError as e:
            self.notify(f"Couldn't delete '{name}' — {e}.", severity="error")
            return False
        # drop the marker too; prune empty specs (but never builtins'
        # identity — only the md key)
        agents = dict(getattr(self.cfg, "agents", None) or {})
        spec = dict(agents.get(agent) or {})
        md = spec.get("md")
        if isinstance(md, dict):
            md = dict(md)
            md.pop(name, None)
            md.pop(p.name if hasattr(p, "name") else name, None)
            if md:
                spec["md"] = md
            else:
                spec.pop("md", None)
        if spec:
            agents[agent] = spec
        else:
            agents.pop(agent, None)
        self.cfg.agents = agents
        err = self._persist_agents()
        self._refresh_agent_md()
        if err:
            self.notify(f"Deleted from disk, but NOT saved — {err}.", severity="error")
            return False
        self.notify(f"Deleted '{name}' from {agent}.")
        return True


    def _save_agent_md_all(self) -> None:
        """Save button: everything already persists per edit; this locks
        in the current picks too (same guarantee as the Agents Save)."""
        self._save_agents_everything()


    def _on_agent_picker_done(self, result: str | None) -> None:
        self._agent_picker = None

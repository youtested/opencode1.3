"""Custom-agent management for the Textual TUI.

Split out of :mod:`opencode_py.tui.app` as a mixin; method bodies are unchanged.
"""
from __future__ import annotations

from ..util.log import swallow
from typing import Any

class AgentsMixin:

    def action_toggle_agent(self) -> None:
        self._open_agent_picker()


    # -- agent picker (custom agents) ------------------------------------
    def _open_agent_picker(self) -> None:
        """Dead-centered agent manager: switch/add/delete/rename/edit rules."""
        from ..permission import list_agents as _list_agents
        from .agent_picker import AgentPicker

        engine = self._active_engine()
        current = str(getattr(engine, "agent", None) or "build")
        picker = AgentPicker(
            agents=_list_agents(self.cfg),
            current=current,
            on_switch=self._set_agent,
            on_add=self._agent_add,
            on_delete=self._agent_delete,
            on_rename=self._agent_rename,
            on_permissions=self._open_agent_permissions,
            on_save=self._save_agents_everything,
            on_agent_md=self._open_agent_md,
            on_edit_description=self._agent_edit_description,
        )
        self._agent_picker = picker
        self.push_screen(picker, self._on_agent_picker_done)


    def _refresh_agent_picker(self) -> None:
        from ..permission import list_agents as _list_agents

        picker = getattr(self, "_agent_picker", None)
        if picker is None:
            return
        try:
            current = str(getattr(self._active_engine(), "agent", None) or "build")
            picker.reload_agents(_list_agents(self.cfg), current)
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/agents.py:48")
            pass


    def _persist_agents(self) -> str | None:
        """Write the agents dict back to opencode.json (durable).

        LOUD: returns an error string when the save fails OR the
        read-back verify disagrees — callers surface it instead of
        claiming success. Returns None on a verified save.
        """
        try:
            from ..config import save_config

            raw = dict(getattr(self.cfg, "raw", None) or {})
            raw["agents"] = dict(getattr(self.cfg, "agents", None) or {})
            self.cfg.raw = raw
            # merge_disk_agents=False: the manager owns cfg.agents here —
            # unioning would resurrect just-deleted agents
            save_config(self.cfg, merge_disk_agents=False)
        except Exception as e:
            return f"Save failed: {e}"
        # read-back verify: reload the file and confirm our agents survived
        try:
            from ..config import load_config as _load
            from ..permission import list_agents as _la

            fresh = _load()
            have = {n for n, _d, _c in _la(fresh)}
            want = {n for n, _d, _c in _la(self.cfg)}
            missing = want - have
            if missing:
                return f"Save unverified: {', '.join(sorted(missing))} not on disk"
        except Exception as e:
            return f"Save unverified: {e}"
        return None


    def _save_agents_everything(self) -> None:
        """Save button: persist the agent list permanently.

        Only agents (adds/renames/deletes and every custom's rules) are written.
        This used to also copy the LIVE engine's provider/model/agent into cfg,
        so pressing Save here replaced the model chosen in Settings with whatever
        the running session happened to use. Provider, model and default agent
        belong to Settings (or /model) and are never touched from here.
        """
        self._persist_agents()
        self.notify("Agents saved.")


    def _agent_add(self, name: str, description: str) -> str | None:
        from ..permission import BUILTIN_AGENTS

        name = (name or "").strip().lower().replace(" ", "-")
        if not name or name in BUILTIN_AGENTS:
            self.notify(f"Bad agent name '{name}'.", severity="warning")
            return "bad name"
        agents = dict(getattr(self.cfg, "agents", None) or {})
        if name in agents:
            self.notify(f"Agent '{name}' already exists.", severity="warning")
            return "exists"
        agents[name] = {"description": description or "Custom agent.", "tools": {}}
        self.cfg.agents = agents
        err = self._persist_agents()
        self._refresh_agent_picker()
        if err:
            self.notify(f"Agent '{name}' NOT saved — {err}.", severity="error")
            return err
        self.notify(f"Agent '{name}' added.")
        return None


    def _agent_delete(self, name: str) -> bool:
        from ..permission import BUILTIN_AGENTS

        if name in BUILTIN_AGENTS:
            self.notify("Built-in agents can't be deleted.", severity="warning")
            return False
        agents = dict(getattr(self.cfg, "agents", None) or {})
        if name not in agents:
            return False
        # leaving a deleted agent falls back to build (never stranded)
        try:
            if str(getattr(self._active_engine(), "agent", "")) == name:
                self._set_agent("build")
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/agents.py:140")
            pass
        agents.pop(name, None)
        self.cfg.agents = agents
        err = self._persist_agents()
        self._refresh_agent_picker()
        if err:
            self.notify(f"Delete NOT saved — {err}.", severity="error")
            return False
        self.notify(f"Agent '{name}' deleted.")
        return True


    def _agent_edit_description(self, name: str, description: str) -> None:
        """Change a CUSTOM agent's description (builtins keep fixed text)."""
        from ..permission import BUILTIN_AGENTS

        if name in BUILTIN_AGENTS:
            self.notify("Built-in descriptions can't be edited.", severity="warning")
            return
        agents = dict(getattr(self.cfg, "agents", None) or {})
        if name not in agents:
            return
        spec = dict(agents.get(name) or {})
        spec["description"] = (description or "").strip() or "Custom agent."
        agents[name] = spec
        self.cfg.agents = agents
        err = self._persist_agents()
        self._refresh_agent_picker()
        if err:
            self.notify(f"Description NOT saved — {err}.", severity="error")
            return
        self.notify(f"Description updated for '{name}'.")


    def _agent_toggle_readonly(self, name: str) -> None:
        """Flip a CUSTOM agent's readonly flag (build stays writable)."""
        from ..permission import BUILTIN_AGENTS

        if name in BUILTIN_AGENTS:
            self.notify("Built-in agents keep their own mode.", severity="warning")
            return
        agents = dict(getattr(self.cfg, "agents", None) or {})
        if name not in agents:
            return
        spec = dict(agents.get(name) or {})
        spec["readonly"] = not bool(spec.get("readonly"))
        agents[name] = spec
        self.cfg.agents = agents
        err = self._persist_agents()
        if err:
            self.notify(f"Readonly NOT saved — {err}.", severity="error")
            return
        try:
            if str(getattr(self._active_engine(), "agent", "")) == name:
                self._set_agent(name, quiet=True)
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/agents.py:196")
            pass
        state = "read-only" if spec["readonly"] else "writable"
        self.notify(f"'{name}' is now {state}.")


    def _agent_rename(self, old: str, new: str) -> str | None:
        from ..permission import BUILTIN_AGENTS

        new = (new or "").strip().lower().replace(" ", "-")
        agents = dict(getattr(self.cfg, "agents", None) or {})
        if old in BUILTIN_AGENTS or old not in agents:
            self.notify("Built-in agents can't be renamed.", severity="warning")
            return "protected"
        if not new or new in BUILTIN_AGENTS or new in agents:
            self.notify(f"Bad agent name '{new}'.", severity="warning")
            return "bad name"
        agents[new] = agents.pop(old)
        self.cfg.agents = agents
        try:
            if str(getattr(self._active_engine(), "agent", "")) == old:
                self._set_agent(new)
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/agents.py:218")
            pass
        err = self._persist_agents()
        self._refresh_agent_picker()
        if err:
            self.notify(f"Rename NOT saved — {err}.", severity="error")
            return err
        self.notify(f"Renamed '{old}' → '{new}'.")
        return None


    def _agent_tool_action(self, agent: str, tool: str) -> str:
        """Effective allow/ask/deny for one tool of one agent.

        Always probed with mode="ask" so the CONFIGURED rule shows, not
        the mode-mangled result: under mode="auto" every "ask" evaluates
        to "allow", which made the editor display a permanent "allow"
        for the active agent (edits saved fine underneath — the display
        lied, and arrows looked dead).

        Out-of-scope tools (agent has a `scope` list not containing the
        tool) always show deny — visible but unusable, never hidden, for
        present and future agents alike.
        """
        try:
            from ..permission import agent_spec as _spec

            scope = (_spec(self.cfg, agent) or {}).get("scope")
            if isinstance(scope, list) and scope and tool not in {str(t) for t in scope}:
                return "deny"
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/agents.py:248")
            pass
        try:
            from ..permission import PermissionEngine, merge_permissions

            eng = PermissionEngine.from_config(
                merge_permissions(getattr(self.cfg, "permission", None) or {}, agent, self.cfg),
                mode="ask",
            )
            return str(eng.evaluate(tool, "") or "ask")
        except Exception:
            return "ask"


    def _agent_perm_state(self, agent: str) -> tuple[bool, bool, set[str] | None]:
        """(readonly, readonly_locked, scope-set) for the permission editor."""
        from ..permission import BUILTIN_AGENTS, agent_spec

        try:
            spec = agent_spec(self.cfg, agent) or {}
        except Exception:
            spec = {}
        try:
            from ..permission import agent_readonly as _ro
            readonly = bool(_ro(self.cfg, agent))
        except Exception:
            readonly = agent in ("plan", "explore")
        locked = agent in BUILTIN_AGENTS
        scope = spec.get("scope")
        scope_set = {str(x) for x in scope} if isinstance(scope, list) else None
        return readonly, locked, scope_set


    def _open_agent_permissions(self, agent: str) -> None:
        """Per-agent tool editor for the picker (live tool list)."""
        from ..permission import agent_effective_tools
        from .agent_permission_editor import AgentPermissionEditor

        try:
            tools = agent_effective_tools(self.cfg, agent, self._active_engine().registry)
        except Exception:
            tools = agent_effective_tools(self.cfg, agent, None)
        rows = [(t, self._agent_tool_action(agent, t)) for t in tools]
        readonly, locked, scope_set = self._agent_perm_state(agent)

        def on_set(tool: str, action: str) -> None:
            agents = dict(getattr(self.cfg, "agents", None) or {})
            spec = dict(agents.get(agent) or {})
            tools_cfg = dict(spec.get("tools") or {})
            tools_cfg[tool] = action
            spec["tools"] = tools_cfg
            # scope follows the editor: allowing an out-of-scope tool adds
            # it to scope (model actually receives it); denying a scoped
            # tool drops it from scope (shows deny, stays visible). Ask
            # keeps membership as-is. This keeps editor display and sent
            # schemas in agreement for present and future agents.
            try:
                scope = spec.get("scope")
                if isinstance(scope, list):
                    names = [str(t) for t in scope]
                    if action == "deny":
                        names = [t for t in names if t != tool]
                    elif tool not in names:
                        names.append(tool)
                    spec["scope"] = sorted(set(names))
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/agents.py:313")
                pass
            # read-only agents keep their walls: flipping a mutating tool
            # to allow clears the readonly flag (explicit choice wins)
            if action == "allow" and tool in ("edit", "write", "apply_patch", "bash"):
                spec["readonly"] = False
            agents[agent] = spec
            self.cfg.agents = agents
            self._persist_agents()
            try:
                if str(getattr(self._active_engine(), "agent", "")) == agent:
                    self._set_agent(agent, quiet=True)  # rebuild live rules silently
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/agents.py:325")
                pass
            editor = getattr(self, "_agent_perm_editor", None)
            if editor is not None:
                try:
                    tools2 = agent_effective_tools(
                        self.cfg, agent, self._active_engine().registry)
                    ro2, _locked2, scope2 = self._agent_perm_state(agent)
                    editor.reload_state(
                        tools=[(t, self._agent_tool_action(agent, t)) for t in tools2],
                        readonly=ro2,
                        scope=scope2,
                    )
                except Exception as _oc_exc:
                    swallow(_oc_exc, "tui/agents.py:338")
                    pass

        def on_toggle() -> None:
            self._agent_toggle_readonly(agent)
            editor2 = getattr(self, "_agent_perm_editor", None)
            if editor2 is not None:
                try:
                    ro3, _l3, _s3 = self._agent_perm_state(agent)
                    editor2.reload_state(readonly=ro3)
                except Exception as _oc_exc:
                    swallow(_oc_exc, "tui/agents.py:348")
                    pass

        editor = AgentPermissionEditor(
            agent=agent, tools=rows, on_set=on_set,
            readonly=readonly, readonly_locked=locked, scope=scope_set,
            on_toggle_readonly=on_toggle,
        )
        self._agent_perm_editor = editor

        def _back(_result: Any) -> None:
            self._agent_perm_editor = None
            # return focus to the picker's list (Esc from the editor left
            # focus on the Permissions button, so ↑/↓ went nowhere and the
            # highlight looked frozen).
            try:
                from textual.widgets import OptionList as _OL

                picker = getattr(self, "_agent_picker", None)
                if picker is not None:
                    picker.query_one("#agent-picker-list", _OL).focus()
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/agents.py:369")
                pass

        self.push_screen(editor, _back)
        # focus the editor list immediately (not only on mount): a fast
        # arrow press in the mount gap previously went nowhere.
        try:
            from textual.widgets import OptionList as _OL2

            editor.query_one("#agent-perm-list", _OL2).focus()
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/agents.py:379")
            pass


    def _set_agent(self, agent: str, quiet: bool = False) -> None:
        from ..permission import list_agents as _list_agents

        known = {n for n, _d, _c in _list_agents(self.cfg)}
        if agent not in known:
            self.notify(f"Unknown agent '{agent}'. Agents: {', '.join(sorted(known))}")
            return
        engine = self._active_engine()
        if engine.agent == agent:
            if not quiet:
                self.notify(f"Agent: {agent}")
            self._update_header()
            return
        engine.agent = agent
        # The PermissionEngine is built from the agent's permission defaults at
        # construction time (build vs plan differ: plan force-denies
        # bash/write/edit/apply_patch). Switching agents mid-session must
        # rebuild it, otherwise those deny rules never take effect and a plan
        # agent can still execute a mutating tool call the model emits (e.g. a
        # bash command). Sub-agents share this same engine (spawn passes
        # permission_engine=self.permission), so one rebuild covers them too.
        from ..permission import PermissionEngine, merge_permissions

        engine.permission = PermissionEngine.from_config(
            merge_permissions(self.cfg.permission, agent, self.cfg),
            mode=getattr(engine.permission, "mode", "auto"),
        )
        engine.permission.ask_callback = self._permission_ask
        sess = self._sessions.get(self._current_session_id)
        if sess is not None:
            sess.agent = agent
        if not quiet:
            self.notify(f"Agent: {agent}")
        self._update_header()

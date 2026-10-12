"""Slash-command handling and the picker launcher methods.

Split out of :mod:`opencode_py.tui.app` as a mixin; method bodies are unchanged.
"""
from __future__ import annotations

from ..util.log import swallow
from typing import Any
from ..config import save_config
from .app_meta import _COMMAND_USAGE, _SAFE_WHILE_BUSY
from .input_bar import CommandSelected, InputBar
from .status_bar import StatusBar

class CommandsMixin:

    # -- command handling -------------------------------------------------
    def _run_command(self, line: str) -> None:
        from ..commands import handle_command
        from ..commands import CommandContext

        # /models is a full-screen, live model list. The bare form is already
        # intercepted by the command popup; with arguments it would fall through
        # to the sync fetch_zen_models() in commands.py and freeze the UI thread,
        # so route it to the picker (which fetches off-thread) instead.
        # /sessions opens the same opencode-style picker as Ctrl+R (the plain
        # /sessions command only prints a text list).
        stripped = line.strip()
        name = (
            stripped[1:].split(maxsplit=1)[0]
            if stripped.startswith("/") and len(stripped) > 1
            else ""
        )
        if name == "models":
            self._open_model_picker()
            return
        if name == "agent":
            # bare /agent opens the manager (list/add/delete/rename/rules);
            # `/agent <name>` switches directly, customs included.
            rest = stripped[1 + len(name):].strip()
            if rest:
                self._set_agent(rest.split(maxsplit=1)[0].lower())
            else:
                self._open_agent_picker()
            return
        if name == "skills":
            # bare /skills opens the manager popup (view/add/rename/edit/
            # delete); reload/validate subcommands run inline as before.
            rest = stripped[1 + len(name):].strip()
            first = rest.split(maxsplit=1)[0].lower() if rest else ""
            if first in ("reload", "refresh", "clear", "validate", "check", "test"):
                pass  # fall through to the command handler
            else:
                self._open_skill_picker()
                return
        if name == "sessions":
            self.action_sessions()
            return
        if name == "connect":
            # skip the Run/Cancel confirm popup: go straight to the
            # dead-centered provider picker (like models/sessions/etc).
            # `/connect groq` preselects the provider; bare opens the list.
            rest = stripped[1 + len(name):].strip()
            self._open_connect(rest.split(maxsplit=1)[0] if rest else "")
            return
        if name == "permissions":
            # dedicated read-only popup: every effective rule in plain
            # words (mode + merged agent/user rules), not the raw config.
            self._open_permissions()
            return
        if name == "setting":
            self.action_settings()
            return
        if name == "commands":
            self._open_command_picker()
            return
        if name == "provider":
            self._open_provider_picker()
            return
        if name == "plugins":
            self._open_plugin_picker()
            return
        if name == "mcp":
            # bare /mcp (or list/test) opens the manager popup; add/remove
            # with arguments run inline through the command handler.
            rest = stripped[1 + len(name):].strip()
            first = rest.split(maxsplit=1)[0].lower() if rest else ""
            if not rest or first in ("list", "ls", "test", "check", "ping", "picker", "ui"):
                if not rest:
                    self._open_mcp_picker()
                    return
                if first in ("picker", "ui"):
                    self._open_mcp_picker()
                    return
                # fall through to the handler for list/test variants
            # else fall through to the command handler (add/remove path)
        if name == "thinking":
            # bare /thinking (or with show/hide/last) opens the popup; effort
            # levels passed inline (e.g. `/thinking high`) apply directly.
            rest = stripped[1 + len(name):].strip()
            if not rest or rest.lower() in ("show", "hide", "last", "toggle", "on", "off", "expand", "collapse", "all"):
                self._open_thinking_picker()
                return
            # else fall through to the command handler (effort level path)
        if name == "new":
            # commands._new is headless-shaped (no get_session callback here),
            # so it used to reply "New session." and do NOTHING. Do it for real.
            # BUT never mid-turn: this interception sat before the busy gate,
            # so /clear while streaming wiped the running engine's history out
            # from under its worker thread.
            if self._busy:
                self._chat_for(self._current_session_id).append_meta(
                    "⏳ still working on the previous request…"
                )
                self.notify("Busy — finish or interrupt the running request first (Ctrl+C)")
                return
            self._action_new()
            self._chat_for(self._current_session_id).append_meta(
                "Started a new session — the previous one stays in the picker (Ctrl+R)."
            )
            return
        # Mutating commands must not run mid-turn: they'd race the running
        # engine (e.g. /undo popping the undo stack the worker is appending to).
        if self._busy and name not in _SAFE_WHILE_BUSY:
            self._chat_for(self._current_session_id).append_meta(
                "⏳ still working on the previous request…"
            )
            self.notify("Busy — finish or interrupt the running request first (Ctrl+C)")
            return

        engine = self._active_engine()
        session = self._active_session()
        cmd = self.command_registry.get(name)
        custom = (getattr(cmd, "metadata", {}) or {}).get("entry") if cmd is not None else None
        if custom:
            prompt = str(custom.get("prompt") or "").strip()
            if not prompt:
                self.notify(f"/{name} has no prompt", severity="warning")
                return
            extra = stripped[len(name) + 1:].strip() if stripped.startswith("/" + name) else ""
            text = f"{prompt}\n\n{extra}" if extra else prompt
            self._start_turn(self._current_session_id, text, engine)
            return

        def reply(text: str) -> None:
            # persistent chat output + a short toast; /models & friends must
            # not vanish into a transient notification
            self._chat_for(self._current_session_id).append_meta(text)
            self.notify(text.splitlines()[0][:60] if text else "", timeout=3, markup=False)

        ctx = CommandContext(
            config=self.cfg,
            auth=self.auth,
            session=session,
            engine=engine,
            worktree=str(self.directory),
            reply=reply,
            get_session=lambda: self._active_session(),
            set_model=self._set_model,
            set_agent=self._set_agent,
            exit_app=self.exit,
            resume=self._resume_session,
            connect=self._open_connect,
            registry=self.command_registry,
        )
        handle_command(self.command_registry, ctx, line)
        self._update_header()


    def _preview_command(self, name: str) -> str:
        """Run a read-only command with a collecting reply and return its output."""
        from ..commands import CommandContext, handle_command

        collected: list[str] = []
        ctx = CommandContext(
            config=self.cfg,
            auth=self.auth,
            session=self._active_session(),
            engine=self._active_engine(),
            worktree=str(self.directory),
            reply=collected.append,
            get_session=lambda: self._active_session(),
            set_model=self._set_model,
            set_agent=self._set_agent,
            exit_app=self.exit,
            resume=self._resume_session,
            connect=self._open_connect,
            registry=self.command_registry,
            # preview pass: handlers must not mutate anything (/export used to
            # write the file here, then AGAIN when Run pressed it)
            preview_only=True,
        )
        handle_command(self.command_registry, ctx, f"/{name}")
        return "\n".join(collected)


    def on_command_selected(self, event: CommandSelected) -> None:
        # /models is the full-screen, live-updating provider model list
        if event.name == "models":
            self._open_model_picker()
            return
        if event.name == "agent":
            self._open_agent_picker()
            return
        if event.name in ("skills", "skill"):
            self._open_skill_picker()
            return
        if event.name == "sessions":
            self.action_sessions()
            return
        if event.name == "connect":
            self._open_connect("")
            return
        if event.name == "permissions":
            self._open_permissions()
            return
        if event.name == "setting":
            self.action_settings()
            return
        if event.name == "theme":
            self._open_theme_picker()
            return
        if event.name == "thinking":
            self._open_thinking_picker()
            return
        if event.name in ("commands", "cmds"):
            self._open_command_picker()
            return
        if event.name == "providers":
            self._open_provider_picker()
            return
        if event.name == "provider":
            self._open_provider_picker()
            return
        if event.name == "plugins":
            self._open_plugin_picker()
            return
        if event.name == "mcp":
            self._open_mcp_picker()
            return
        cmd = self.command_registry.get(event.name)
        content: str | None = None
        if cmd is not None and cmd.preview:
            content = self._preview_command(event.name).strip() or cmd.description

        def done(result: str | None) -> None:
            bar = self.query_one(InputBar)
            bar.input.focus()
            if result == "run":
                self._run_command(f"/{event.name}")
            elif result == "cancel":
                # Esc = back: put the command back in the input, cursor at the
                # end. ("close" leaves the input empty — the user READ the
                # output; stuffing "/name" back used to leave stray text.)
                bar.input.value = f"/{event.name}"
                bar.input.cursor_position = len(bar.input.value)

        from .command_popup import CommandPopup

        self.push_screen(
            CommandPopup(
                event.name,
                event.description,
                content=content,
                usage=_COMMAND_USAGE.get(event.name, ""),
            ),
            done,
        )


    def _open_model_picker(self) -> None:
        def on_picked(choice: str | None) -> None:
            try:
                self.query_one(InputBar).input.focus()
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/commands.py:273")
                pass
            if not choice:
                return
            provider, _, model = choice.partition("/")
            if not provider or not model:
                return
            # Deliberately NOT writing cfg.provider/cfg.model here: this pick is
            # session-scoped (stamped on the session record below), and leaving
            # it in cfg meant the next unrelated save_config() silently committed
            # it as the global default.
            engine = self._active_engine()
            engine.provider_id = provider
            engine.model_id = model
            # Network-touching rebuild runs at the next turn start (engine
            # thread), never inline here on the UI thread.
            engine.mark_rotation_dirty()
            # The pick belongs to THIS session: stamp provider/model on the
            # session record and save now, so a restart + reopen keeps it
            # (the typed /model path already stamps sess.model; the popup
            # never did, so its choice evaporated on exit).
            try:
                sid = self._current_session_id
                sess = self._sessions.get(sid)
                if sess is not None:
                    sess.provider = provider
                    sess.model = model
                    try:
                        hist = engine.get_history()
                        if hist:
                            sess.messages = hist
                    except Exception as _oc_exc:
                        swallow(_oc_exc, "tui/commands.py:302")
                        pass
                    if getattr(sess, "messages", None):
                        from ..session import save_session as _save_pick
                        _save_pick(sess)
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/commands.py:307")
                pass
            # Repaint the % NOW against the new model's window (same tokens,
            # new denominator) instead of showing the old model's math until
            # the next turn completes.
            try:
                from ..providers import model_context_size as _ctx_size
                sid = self._current_session_id
                usage = dict(self._usage.get(sid) or {})
                ctx = _ctx_size(provider, model, auth=self.auth)
                if ctx:
                    usage["context_size"] = ctx
                if usage:
                    self._usage[sid] = usage
                    self.query_one(StatusBar).set_usage(usage)
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/commands.py:322")
                pass
            self.notify(f"Model set to {provider}/{model} (this session)")
            self._update_header()
            # Deliberately NOT save_config() here: picking a model to TRY must
            # not silently rewrite the user's default in opencode.json. The
            # Settings screen persists explicit choices.

        from .model_picker import ModelPicker

        self.push_screen(
            ModelPicker(current=self.cfg.model, cfg=self.cfg, auth=self.auth),
            on_picked,
        )


    def _open_theme_picker(self) -> None:
        """Arrow-navigable theme list (custom first, then dark); Enter applies.

        Footer buttons loop back here: new/edit open the ThemeEditor, rename
        asks for a name, delete asks for confirmation — then the picker
        re-opens so you can see/apply the result."""
        from .theme import custom_themes_dict, set_active_theme
        from .theme_picker import ThemePicker

        def persist() -> bool:
            try:
                self.cfg.custom_themes = custom_themes_dict()
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/commands.py:350")
                pass
            try:
                save_config(self.cfg)
                return True
            except Exception as e:
                self.notify(f"Theme saved, but NOT written to disk ({e}).")
                return False

        def done(choice) -> None:
            if not choice:
                try:
                    self.query_one(InputBar).input.focus()
                except Exception as _oc_exc:
                    swallow(_oc_exc, "tui/commands.py:363")
                    pass
                return
            action, name = choice
            if action == "apply":
                try:
                    self.query_one(InputBar).input.focus()
                except Exception as _oc_exc:
                    swallow(_oc_exc, "tui/commands.py:370")
                    pass
                if not name:
                    return
                self.cfg.theme = name
                set_active_theme(name)
                if persist():
                    self.notify(f"Theme set to {name}")
                return
            if action == "new":
                self._open_theme_editor("", name or self.cfg.theme or "opencode")
                return
            if action == "edit":
                self._open_theme_editor(name, name)
                return
            if action == "rename":
                self._rename_custom_theme(name)
                return
            if action == "delete":
                self._delete_custom_theme(name)
                return

        self.push_screen(ThemePicker(current=self.cfg.theme), done)


    def _open_theme_editor(self, name: str, base: str) -> None:
        """Create/edit a custom theme; save persists + re-opens the picker."""
        from .theme import get_theme, set_active_theme, set_custom_theme
        from .theme_editor import ThemeEditor

        start_palette = None
        if name:
            try:
                start_palette = dict(get_theme(name).colors)
            except Exception:
                start_palette = None

        def done(result) -> None:
            if not result:
                self._open_theme_picker()
                return
            old, new_name, palette = result
            try:
                from .theme import delete_custom_theme as _del

                if old and old != new_name:
                    try:
                        _del(old)
                    except Exception as _oc_exc:
                        swallow(_oc_exc, "tui/commands.py:418")
                        pass
                slug = set_custom_theme(new_name, palette)
            except ValueError as e:
                self.notify(str(e), severity="warning")
                self._open_theme_picker()
                return
            if old and old != slug and getattr(self.cfg, "theme", "") == old:
                self.cfg.theme = slug
            self.cfg.theme = slug
            set_active_theme(slug)
            from .theme import custom_themes_dict as _dump

            try:
                self.cfg.custom_themes = _dump()
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/commands.py:433")
                pass
            try:
                save_config(self.cfg)
            except Exception as e:
                self.notify(f"Theme '{slug}' applied, but NOT saved ({e}).")
                self._open_theme_picker()
                return
            self.notify(f"Theme '{slug}' saved")
            self._open_theme_picker()

        self.push_screen(ThemeEditor(name=name, base=base, palette=start_palette), done)


    def _rename_custom_theme(self, name: str) -> None:
        from .agent_picker import AgentNameDialog
        from .theme import custom_themes_dict, is_custom, rename_custom_theme, set_active_theme

        if not is_custom(name):
            self.notify("Only custom themes can be renamed.", severity="warning")
            self._open_theme_picker()
            return

        def after(result) -> None:
            if not result:
                self._open_theme_picker()
                return
            new_name = (result[0] or "").strip().lower()
            try:
                slug = rename_custom_theme(name, new_name)
            except (ValueError, KeyError) as e:
                self.notify(f"Can't rename: {e}", severity="warning")
                self._open_theme_picker()
                return
            if getattr(self.cfg, "theme", "") == name:
                self.cfg.theme = slug
                set_active_theme(slug)
            try:
                self.cfg.custom_themes = custom_themes_dict()
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/commands.py:472")
                pass
            try:
                save_config(self.cfg)
            except Exception as e:
                self.notify(f"Renamed to '{slug}', but NOT saved ({e}).")
                self._open_theme_picker()
                return
            self.notify(f"Renamed '{name}' → '{slug}'.")
            self._open_theme_picker()

        self.push_screen(
            AgentNameDialog(
                old=name,
                title="Rename theme",
                hide_description=True,
                name_placeholder="theme-name (lowercase, dashes)",
            ),
            after,
        )


    def _delete_custom_theme(self, name: str) -> None:
        from .session_list import ConfirmDeleteDialog
        from .theme import custom_themes_dict, delete_custom_theme, is_custom, set_active_theme

        if not is_custom(name):
            self.notify("Only custom themes can be deleted.", severity="warning")
            self._open_theme_picker()
            return

        def after(result) -> None:
            confirmed = bool(result and result[1]) if isinstance(result, tuple) else False
            if not confirmed:
                self._open_theme_picker()
                return
            delete_custom_theme(name)
            if getattr(self.cfg, "theme", "") == name:
                self.cfg.theme = "opencode"
                set_active_theme("opencode")
            try:
                self.cfg.custom_themes = custom_themes_dict()
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/commands.py:514")
                pass
            try:
                save_config(self.cfg)
            except Exception as e:
                self.notify(f"Theme '{name}' deleted, but NOT saved ({e}).")
                self._open_theme_picker()
                return
            self.notify(f"Theme '{name}' deleted.")
            self._open_theme_picker()

        self.push_screen(ConfirmDeleteDialog(session_id=name, title=name), after)


    def _open_thinking_picker(self) -> None:
        """Thinking popup: current model + on/off + effort levels + show/hide.

        Enter applies the highlighted row immediately (effort persists and
        takes effect next turn; on/off flips bubble visibility live).
        """
        from .thinking_picker import ThinkingPicker

        engine = self._active_engine()
        try:
            model_id = getattr(engine, "model_id", "") or self.cfg.model
            provider_id = getattr(engine, "provider_id", "") or self.cfg.provider
        except Exception:
            model_id, provider_id = self.cfg.model, self.cfg.provider
        try:
            from ..providers.rotation import model_effort_levels
            levels = model_effort_levels(model_id, provider_id or "opencode")
        except Exception:
            levels = []
        try:
            chat = self._chat_for(self._current_session_id)
            n_thoughts = len(chat.reasoning_bubbles())
        except Exception:
            n_thoughts = 0

        def done(choice: str | None) -> None:
            try:
                self.query_one(InputBar).input.focus()
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/commands.py:556")
                pass
            if not choice:
                return
            if choice == "__on__":
                self.cfg.show_thoughts = True
                try:
                    save_config(self.cfg)
                except Exception as e:
                    self.notify(f"Thinking on, but NOT saved ({e}).")
                    return
                self._set_thoughts_visible(True)
                self.notify("Thinking on — thought bubbles show.")
            elif choice == "__off__":
                self.cfg.show_thoughts = False
                try:
                    save_config(self.cfg)
                except Exception as e:
                    self.notify(f"Thinking off, but NOT saved ({e}).")
                    return
                self._set_thoughts_visible(False)
                self.notify("Thinking off — thoughts hidden (kept in history).")
            elif choice == "__show__":
                self._expand_all_thoughts()
            elif choice == "__hide__":
                self._collapse_all_thoughts()
            elif choice.startswith("__effort__"):
                self._set_reasoning_effort(choice[len("__effort__"):])

        self.push_screen(
            ThinkingPicker(
                model=f"{provider_id}/{str(model_id).split('/', 1)[-1]}",
                thinking_on=bool(getattr(self.cfg, "show_thoughts", True)),
                effort=str(getattr(self.cfg, "reasoning_effort", "") or ""),
                levels=levels,
                current_thoughts=n_thoughts,
            ),
            done,
        )


    def _set_thoughts_visible(self, visible: bool) -> None:
        """Apply the Thinking on/off switch to every mounted chat now."""
        try:
            for chat in list(getattr(self, "_chats", {}).values()):
                try:
                    chat.set_thoughts_visible(visible)
                except Exception as _oc_exc:
                    swallow(_oc_exc, "tui/commands.py:603")
                    continue
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/commands.py:605")
            pass
        try:
            self._update_header()
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/commands.py:609")
            pass


    def _expand_all_thoughts(self) -> None:
        try:
            chat = self._chat_for(self._current_session_id)
            n = chat.set_all_reasoning(True)
            self.notify(f"Expanded {n} thought(s).")
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/commands.py:618")
            pass


    def _collapse_all_thoughts(self) -> None:
        try:
            chat = self._chat_for(self._current_session_id)
            n = chat.set_all_reasoning(False)
            self.notify(f"Collapsed {n} thought(s).")
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/commands.py:627")
            pass


    def _set_reasoning_effort(self, level: str) -> None:
        """Validate an effort level against the CURRENT model and apply it."""
        level = str(level or "").strip().lower()
        try:
            engine = self._active_engine()
            model_id = getattr(engine, "model_id", "") or self.cfg.model
            provider_id = getattr(engine, "provider_id", "") or self.cfg.provider
            from ..providers.rotation import model_effort_levels
            levels = model_effort_levels(model_id, provider_id or "opencode")
        except Exception:
            levels, model_id = [], ""
        if not levels:
            self.notify("This model has no effort levels — fixed thinking.", severity="warning")
            return
        if level not in [str(v).lower() for v in levels]:
            self.notify(f"Effort must be one of: {', '.join(levels)}", severity="warning")
            return
        self.cfg.reasoning_effort = level
        try:
            save_config(self.cfg)
        except Exception as e:
            self.notify(f"Effort set, but NOT saved ({e}).")
            return
        try:
            self._apply_runtime_settings()
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/commands.py:656")
            pass
        self.notify(f"Reasoning effort → {level} (next turn).")


    def _open_connect(self, provider: str = "") -> None:
        from .connect_screen import ConnectScreen

        self.app.push_screen(
            ConnectScreen(auth=self.auth, on_connected=self._on_connected, initial=provider),
            self._on_connect_dismissed,
        )


    def _on_connected(self, provider_id: str) -> None:
        self.notify(f"Saved API key for {provider_id}.")
        self._update_header()


    def _on_connect_dismissed(self, result: str | None) -> None:
        if result:
            self.notify(f"Connected {result}.")


    def _open_mcp_picker(self) -> None:
        from .. import mcp_manager as _mm
        from .input_bar import InputBar
        from .mcp_picker import McpPicker

        holder: dict[str, Any] = {}

        def refresh_rows() -> None:
            picker = holder.get("picker")
            if picker is None:
                return
            try:
                picker.reload_servers(_mm.list_servers(self.cfg))
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/commands.py:693")
                pass

        def on_result(action: Any) -> None:
            chat = self._chat_for(self._current_session_id)
            if isinstance(action, str):
                if action.startswith("__srv__"):
                    name = action[len("__srv__"):]
                    spec = _mm.list_servers(self.cfg).get(name, {})
                    if isinstance(spec, dict) and spec.get("url"):
                        chat.append_meta(f"MCP '{name}': {spec.get('url')} — Ctrl+T test · Ctrl+E on/off · Ctrl+D delete")
                    else:
                        cmd = spec.get("command", "?") if isinstance(spec, dict) else "?"
                        chat.append_meta(f"MCP '{name}': {cmd} — Ctrl+T test · Ctrl+E on/off · Ctrl+D delete")
                return
            if not isinstance(action, tuple) or not action:
                return
            kind = action[0]
            if kind == "add":
                _k, name, endpoint, mode, token = (list(action) + ["", "", "", ""])[:5]
                if mode == "remote":
                    chat.append_meta(f"Adding remote MCP '{name}': {endpoint}…")
                    self.run_worker(lambda: self._mcp_add_remote(chat, name, endpoint, token, refresh_rows), thread=True)
                else:
                    import shlex as _shlex

                    try:
                        parts = _shlex.split(endpoint)
                    except ValueError as e:
                        self.notify(f"Bad run line: {e}", severity="error")
                        return
                    if not parts:
                        return
                    chat.append_meta(f"Adding MCP '{name}': {' '.join(parts)}…")
                    self.run_worker(lambda: self._mcp_add(chat, name, parts[0], parts[1:], refresh_rows), thread=True)
            elif kind == "toggle":
                name = action[1]
                spec = _mm.list_servers(self.cfg).get(name, {})
                enabled = not (isinstance(spec, dict) and spec.get("disabled") is True)
                _mm.set_server_enabled(name, not enabled, str(self.directory), False)
                raw = getattr(self.cfg, "raw", None)
                if isinstance(raw, dict) and isinstance(raw.get("mcpServers"), dict) and name in raw["mcpServers"]:
                    entry = raw["mcpServers"][name]
                    if isinstance(entry, dict):
                        if enabled:
                            entry["disabled"] = True
                        else:
                            entry.pop("disabled", None)
                try:
                    engine = self._active_engine()
                except Exception:
                    engine = None
                note = _mm.refresh_engine_mcp(engine, self.cfg)
                self.notify(f"'{name}' {'disabled' if enabled else 'enabled'}. {note}")
                refresh_rows()
            elif kind == "test":
                name = action[1]
                chat.append_meta(f"Testing MCP '{name}'…")
                self.run_worker(lambda: self._mcp_test_one(chat, name, refresh_rows), thread=True)
            elif kind == "delete":
                name = action[1]
                removed = _mm.remove_server_everywhere(name, str(self.directory))
                if not removed:
                    self.notify(f"No MCP server '{name}'.", severity="warning")
                    return
                raw = getattr(self.cfg, "raw", None)
                if isinstance(raw, dict) and isinstance(raw.get("mcpServers"), dict):
                    raw["mcpServers"].pop(name, None)
                try:
                    engine = self._active_engine()
                except Exception:
                    engine = None
                note = _mm.refresh_engine_mcp(engine, self.cfg)
                self.notify(f"Removed '{name}'. {note}")
                chat.append_meta(f"Removed MCP '{name}'. {note}")
                refresh_rows()

        def done(_choice: Any) -> None:
            try:
                self.query_one(InputBar).input.focus()
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/commands.py:773")
                pass

        holder["picker"] = McpPicker(servers=_mm.list_servers(self.cfg), on_result=on_result)
        self.push_screen(holder["picker"], done)


    def _open_provider_picker(self) -> None:
        from .. import provider_manager as _pm
        from ..auth import Auth
        from .input_bar import InputBar
        from .provider_picker import ProviderAddDialog, ProviderKeyDialog, ProviderPicker

        auth = self.auth if isinstance(self.auth, Auth) else Auth()
        holder: dict[str, Any] = {}

        def refresh() -> None:
            picker = holder.get("picker")
            if picker is not None:
                try: picker.reload(_pm.rows(auth, self.cfg))
                except Exception as _oc_exc:
                    swallow(_oc_exc, "tui/commands.py:814")
                    pass

        def on_result(action: Any) -> None:
            if not isinstance(action, tuple) or not action: return
            kind = action[0]
            if kind == "key":
                provider = action[1]
                def saved(result: Any) -> None:
                    if not result: refresh(); return
                    try:
                        _pm.set_key(auth, result[0], result[1])
                        self.notify(f"Key saved for {result[0]}.")
                    except Exception as exc: self.notify(str(exc), severity="error")
                    refresh()
                try: self.push_screen(ProviderKeyDialog(provider), saved)
                except Exception as _oc_exc:
                    swallow(_oc_exc, "tui/commands.py:829")
                    pass
            elif kind == "remove":
                if _pm.remove_key(auth, action[1]):
                    self.notify(f"Removed key for {action[1]}.")
                refresh()
            elif kind == "add":
                def added(result: Any) -> None:
                    if not result: refresh(); return
                    try:
                        _pm.add_custom(auth, self.cfg, result[0], result[1], result[2])
                        self.notify(f"Added provider {result[0]}.")
                    except Exception as exc: self.notify(str(exc), severity="error")
                    refresh()
                try: self.push_screen(ProviderAddDialog(), added)
                except Exception as _oc_exc:
                    swallow(_oc_exc, "tui/commands.py:843")
                    pass
            elif kind == "login":
                picker = holder.get("picker")
                if picker is not None:
                    picker.set_status(f"Open {action[2]} to finish sign-in for {action[1]}.")
            elif kind == "refresh":
                picker = holder.get("picker")
                if picker is not None: picker.set_status("Provider list refreshed.")
                refresh()

        def done(_choice: Any) -> None:
            try: self.query_one(InputBar).input.focus()
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/commands.py:855")
                pass

        holder["picker"] = ProviderPicker(rows=_pm.rows(auth, self.cfg), on_result=on_result)
        self.push_screen(holder["picker"], done)


    def _open_command_picker(self) -> None:
        """Agent-style custom slash command manager."""
        from .. import command_manager as _cm
        from .input_bar import InputBar
        from .command_picker import CommandPicker

        holder: dict[str, Any] = {}

        def snapshot() -> tuple[dict, set[str]]:
            return _cm.list_commands(self.cfg), _cm.list_disabled(self.cfg)

        def reload_config_from_disk() -> None:
            try:
                from pathlib import Path as _P
                from ..config import load_config
                fresh = load_config(_P(self.directory))
                self.cfg.raw = fresh.raw
                # the typed copy feeds as_dict() on the next save; leaving it
                # stale is what made the next save_config resurrect deletions
                self.cfg.commands = fresh.commands
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/commands.py:857")
                pass

        def refresh() -> None:
            picker = holder.get("picker")
            if picker is not None:
                try:
                    picker.reload(*snapshot())
                except Exception as _oc_exc:
                    swallow(_oc_exc, "tui/commands.py:865")
                    pass

        def rebuild_registry() -> None:
            try:
                from ..commands import CUSTOM_COMMANDS, register_custom_commands
                CUSTOM_COMMANDS.clear()
                register_custom_commands(self.command_registry)
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/commands.py:873")
                pass

        def on_result(action: Any) -> None:
            chat = self._chat_for(self._current_session_id)
            if isinstance(action, str):
                if action.startswith("__cmd__"):
                    name = action[len("__cmd__"):]
                    chat.append_meta(f"/{name} — Ctrl+N edit · Ctrl+E on/off · Ctrl+D delete · type /{name} to run")
                return
            if not isinstance(action, tuple) or not action:
                return
            kind = action[0]
            try:
                if kind == "add":
                    _k, name, desc, prompt = action
                    key = _cm.save_command(name, prompt, desc, str(self.directory), False)
                    msg = f"Added /{key}"
                elif kind == "edit":
                    _k, name, desc, prompt = action
                    key = _cm.save_command(name, prompt, desc, str(self.directory), False)
                    msg = f"Updated /{key}"
                elif kind == "toggle":
                    name = action[1]
                    current = _cm.list_disabled(self.cfg)
                    key = _cm.normalize_name(name)
                    enabled = key in current
                    _cm.set_enabled(key, not enabled, str(self.directory), False)
                    msg = f"/{key} {'disabled' if enabled else 'enabled'}"
                elif kind == "delete":
                    key = _cm.normalize_name(action[1])
                    msg = f"Removed /{key}" if _cm.remove_command(key, str(self.directory), False) else f"No command /{key} found."
                else:
                    return
            except (OSError, ValueError) as exc:
                self.notify(str(exc), severity="error")
                return
            reload_config_from_disk()
            raw = getattr(self.cfg, "raw", None)
            if isinstance(raw, dict):
                raw["commands"] = {k: v for k, v in _cm.list_commands(self.cfg).items()}
                raw["commandDisabled"] = list(_cm.list_disabled(self.cfg))
            rebuild_registry()
            self.notify(msg)
            chat.append_meta(msg)
            refresh()

        def done(_choice: Any) -> None:
            try:
                self.query_one(InputBar).input.focus()
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/commands.py:923")
                pass

        commands, disabled = snapshot()
        holder["picker"] = CommandPicker(commands=commands, disabled=disabled, on_result=on_result)
        self.push_screen(holder["picker"], done)


    def _open_plugin_picker(self) -> None:
        """Agent-style plugin manager: tool plugins + event hooks."""
        from .. import plugin_manager as _pm
        from .input_bar import InputBar
        from .plugin_picker import PluginPicker

        holder: dict[str, Any] = {}

        def snapshot() -> tuple[list[str], list[dict], set[str]]:
            return (
                _pm.list_plugins(self.cfg),
                _pm.list_hooks(self.cfg),
                _pm.list_disabled(self.cfg),
            )

        def refresh() -> None:
            picker = holder.get("picker")
            if picker is None:
                return
            try:
                picker.reload(*snapshot())
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/commands.py:952")
                pass

        def on_result(action: Any) -> None:
            chat = self._chat_for(self._current_session_id)
            if isinstance(action, str):
                if action.startswith("__plugin__"):
                    name = action[len("__plugin__"):]
                    chat.append_meta(f"Plugin '{name}' — Ctrl+T test · Ctrl+E on/off · Ctrl+D delete")
                return
            if not isinstance(action, tuple) or not action:
                return
            kind = action[0]
            if kind == "add":
                _k, name, target, ptype, events = (list(action) + ["", "", "", "", ""])[:5]
                try:
                    if ptype == "npm":
                        _pm.save_hook(name, target, [e for e in (events or []) if e] or ["tool.after"], str(self.directory), False)
                        msg = f"Added npm plugin '{name}' → {target}"
                    elif ptype == "hook":
                        _pm.save_hook(name, target, [e for e in (events or []) if e], str(self.directory), False)
                        msg = f"Added hook '{name}' → {target}"
                    else:
                        _pm.save_plugin(name, target, str(self.directory), False)
                        msg = f"Added tool plugin '{name}' → {target}"
                    raw = getattr(self.cfg, "raw", None)
                    if isinstance(raw, dict):
                        if ptype in ("hook", "npm"):
                            raw["pluginHooks"] = _pm.list_hooks(self.cfg)
                        else:
                            raw["plugins"] = _pm.list_plugins(self.cfg)
                    self.notify(msg)
                    chat.append_meta(msg)
                except OSError as exc:
                    self.notify(f"Save failed: {exc}", severity="error")
            elif kind == "toggle":
                name = action[1]
                current = _pm.list_disabled(self.cfg)
                key = name
                enabled = key in current or f"hook:{name}" in current
                if _pm.set_enabled(name, not enabled, str(self.directory), False):
                    raw = getattr(self.cfg, "raw", None)
                    if isinstance(raw, dict):
                        raw["pluginDisabled"] = list(_pm.list_disabled(self.cfg))
                        raw["pluginHooks"] = _pm.list_hooks(self.cfg)
                    self.notify(f"'{name}' {'disabled' if enabled else 'enabled'}.")
            elif kind == "test":
                name = action[1]
                chat.append_meta(_pm.test_plugin(name, self.cfg, str(self.directory)))
            elif kind == "delete":
                name = action[1]
                if _pm.remove(name, str(self.directory), False):
                    raw = getattr(self.cfg, "raw", None)
                    if isinstance(raw, dict):
                        raw.pop("plugins", None) if not _pm.list_plugins(self.cfg) else None
                        raw["pluginHooks"] = _pm.list_hooks(self.cfg)
                        raw["pluginDisabled"] = list(_pm.list_disabled(self.cfg))
                    self.notify(f"Removed '{name}'.")
                    chat.append_meta(f"Removed plugin '{name}'.")
            refresh()

        def done(_choice: Any) -> None:
            try:
                self.query_one(InputBar).input.focus()
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/commands.py:1016")
                pass

        plugins, hooks, disabled = snapshot()
        holder["picker"] = PluginPicker(
            plugins=plugins, hooks=hooks, disabled=disabled, on_result=on_result
        )
        self.push_screen(holder["picker"], done)


    def _mcp_add(self, chat: Any, name: str, command: str, cmd_args: list, refresh: Any = None) -> None:
        from .. import mcp_manager as _mm

        line = _mm.test_server(name, command, list(cmd_args), timeout=15.0)
        try:
            path = _mm.save_server(name, command, list(cmd_args), str(self.directory), False)
        except OSError as e:
            self.call_from_thread(chat.append_meta, f"{line}\nSave FAILED: {e}")
            return
        raw = getattr(self.cfg, "raw", None)
        if isinstance(raw, dict):
            servers = raw.get("mcpServers")
            if not isinstance(servers, dict):
                servers = {}
                raw["mcpServers"] = servers
            servers[name] = {"command": command, "args": list(cmd_args)}
        try:
            engine = self._active_engine()
        except Exception:
            engine = None
        note = _mm.refresh_engine_mcp(engine, self.cfg)
        self.call_from_thread(
            chat.append_meta, f"{line}\nSaved '{name}' → {path}\n{note}"
        )
        self.call_from_thread(self.notify, f"MCP '{name}' added. {note}")
        try:
            if callable(refresh):
                self.call_from_thread(refresh)
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/commands.py:1054")
            pass


    def _mcp_add_remote(self, chat: Any, name: str, url: str, token: str, refresh: Any = None) -> None:
        from .. import mcp_manager as _mm

        line = _mm.test_server(name, None, None, timeout=15.0, url=url, auth={"token": token} if token else {})
        try:
            path = _mm.save_remote_server(name, url, str(self.directory), False, token=token)
        except OSError as e:
            self.call_from_thread(chat.append_meta, f"{line}\nSave FAILED: {e}")
            return
        raw = getattr(self.cfg, "raw", None)
        if isinstance(raw, dict):
            servers = raw.get("mcpServers")
            if not isinstance(servers, dict):
                servers = {}
                raw["mcpServers"] = servers
            spec: dict[str, Any] = {"url": url, "transport": "streamable-http", "allowRemote": True}
            if token:
                spec["auth"] = {"token": token}
            servers[name] = spec
        try:
            engine = self._active_engine()
        except Exception:
            engine = None
        note = _mm.refresh_engine_mcp(engine, self.cfg)
        self.call_from_thread(
            chat.append_meta, f"{line}\nSaved '{name}' → {path}\n{note}"
        )
        self.call_from_thread(self.notify, f"Remote MCP '{name}' added. {note}")
        try:
            if callable(refresh):
                self.call_from_thread(refresh)
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/commands.py:1089")
            pass


    def _mcp_test_one(self, chat: Any, name: str, refresh: Any = None) -> None:
        from .. import mcp_manager as _mm

        spec = _mm.list_servers(self.cfg).get(name, {})
        if isinstance(spec, dict) and spec.get("url"):
            line = _mm.test_server(
                name, None, None, timeout=15.0, url=spec.get("url"),
                transport=str(spec.get("transport") or "streamable-http"),
                headers=spec.get("headers") if isinstance(spec.get("headers"), dict) else None,
                auth=spec.get("auth") if isinstance(spec.get("auth"), dict) else None,
            )
        else:
            line = _mm.test_server(name, spec.get("command") if isinstance(spec, dict) else None, (spec.get("args") or []) if isinstance(spec, dict) else [], timeout=15.0)
        self.call_from_thread(chat.append_meta, line)
        try:
            if callable(refresh):
                self.call_from_thread(refresh)
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/commands.py:1110")
            pass

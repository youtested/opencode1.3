"""App lifecycle, header bar and focus chrome for the Textual TUI.

Split out of :mod:`opencode_py.tui.app` as a mixin; method bodies are unchanged.
"""
from __future__ import annotations

from ..util.log import swallow
from .input_bar import InputBar
from .status_bar import StatusBar

class ChromeMixin:

    def _save_all_live_sessions(self) -> None:
        """Final persist of every LIVE session that actually has a conversation.

        This is one of the ONLY durable-write paths (besides the in-flight
        crash-safety autosave tick). It runs when the user quits (ctrl+q /
        /exit), when the TUI tears down, and when Termux closes the app via
        SIGTERM/SIGHUP. Sessions with no conversation are skipped, so an
        untouched scratch session never leaves a file behind.
        """
        from ..session import save_session

        # Newer than any in-flight autosave: a worker that hasn't written yet
        # must not overwrite these final bodies with its (older) snapshot.
        self._autosave_generation += 1
        for sid, sess in list(self._sessions.items()):
            engine = self._engines.get(sid)
            history = None
            if engine is not None:
                try:
                    history = engine.get_history()
                except Exception:
                    history = None
            if not history:
                history = list(getattr(sess, "messages", None) or [])
            if not history:
                continue
            # Preserve child-session links stamped onto the in-memory bubbles
            # by the relinker: the engine's history copy predates them (it was
            # snapshotted at turn end), so a blind overwrite would ERASE the
            # clickability we just healed — the exact regression that wiped
            # this parent's links on exit.
            try:
                chat = self._chats.get(sid)
                if chat is not None:
                    live_links: dict[str, str] = {}
                    for b in chat.query("*"):
                        try:
                            if getattr(b, "role", "") != "tool":
                                continue
                            c = getattr(b, "content", None)
                            if not isinstance(c, dict) or c.get("tool") != "task":
                                continue
                            meta = c.get("metadata") or {}
                            lsid = meta.get("sessionId")
                            bid = c.get("id") or c.get("call_id")
                            if lsid and bid:
                                live_links[str(bid)] = str(lsid)
                        except Exception as _oc_exc:
                            swallow(_oc_exc, "tui/chrome.py:59")
                            continue
                    if live_links:
                        for m in history:
                            try:
                                if isinstance(m, dict) and m.get("role") == "tool" and m.get("name") == "task":
                                    cid = str(m.get("tool_call_id") or "")
                                    if cid in live_links and not (m.get("session_id") or m.get("sessionId")):
                                        m["session_id"] = live_links[cid]
                                        m["sessionId"] = live_links[cid]
                            except Exception as _oc_exc:
                                swallow(_oc_exc, "tui/chrome.py:69")
                                continue
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/chrome.py:71")
                pass
            sess.messages = history
            try:
                save_session(sess)
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/chrome.py:76")
                pass


    def on_exit_app(self) -> None:
        """Fired when the app quits (ctrl+q / /exit).

        Unblocks any engine thread waiting on a permission dialog and persists
        the final state of every live session that has a conversation.
        """
        self._exit_requested.set()
        try:
            self._stop_auto_voice()
            self._clear_voice_all()
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/chrome.py:90")
            pass
        try:
            self._save_all_live_sessions()
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/chrome.py:94")
            pass


    def on_unmount(self) -> None:
        """Final persist as the TUI tears down (same policy as on_exit_app)."""
        try:
            self._stop_auto_voice()
            self._clear_voice_all()
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/chrome.py:103")
            pass
        try:
            self._save_all_live_sessions()
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/chrome.py:107")
            pass


    def action_resume(self) -> None:
        """Ctrl+R: open the session picker to continue a past conversation."""
        self.action_sessions()


    def action_focus_input(self) -> None:
        self.query_one(InputBar).focus()


    def action_toggle_thought(self) -> None:
        chat = self._chat_for(self._current_session_id)
        chat.toggle_last_reasoning()
        self.query_one(InputBar).focus()


    def _update_header(self) -> None:
        status = self.query_one(StatusBar)
        # The engine may not be warmed yet (first paint happens before the
        # background engine build finishes) — fall back to cfg values rather
        # than forcing the ~0.4s import on the UI thread mid-mount.
        # A finished sub-agent's engine is popped to save RAM: fall back to
        # the SESSION's recorded agent (not the main engine) so viewing a
        # web-agent still shows Web-Agent instead of Build.
        engine = self._engines.get(self._current_session_id) or self._main_engine
        sess = self._sessions.get(self._current_session_id)
        if engine is None:
            header = {
                "agent": self.cfg.default_agent or "build",
                "model": self.cfg.model,
                "provider": self.cfg.provider,
                # This was hardcoded to "auto", so before an engine exists the
                # header showed "auto" regardless of the saved setting — a user
                # on "fully_auto" saw a lie for a frame. Every other field here
                # already falls back to cfg; this one now does too.
                "permission_mode": getattr(self.cfg, "permission_mode", "auto"),
                "rotation_locked": getattr(self.cfg, "rotation_lock", False),
                "reasoning_effort": getattr(self.cfg, "reasoning_effort", "") or "",
            }
        else:
            sess_agent = (getattr(sess, "agent", "") or "").strip()
            main_id = getattr(getattr(self, "_main_engine", None), "session_id", "")
            if engine is self._main_engine and sess_agent and self._current_session_id != main_id:
                agent_name = sess_agent
            else:
                agent_name = engine.agent
            header = {
                "agent": agent_name,
                # reflect the model/provider that actually answered: rotation can
                # fail over to a backup lane (e.g. deepseek -> nemotron) while
                # cfg.model keeps the user's configured base model.
                "model": getattr(engine, "model_id", "") or self.cfg.model,
                "provider": getattr(engine, "provider_id", "") or self.cfg.provider,
                "permission_mode": engine.permission.mode,
                "rotation_locked": getattr(engine, "rotation_locked", False),
                "reasoning_effort": getattr(self.cfg, "reasoning_effort", "") or "",
            }
        status.set_header(**header)
        try:
            bar = self.query_one(InputBar)
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/chrome.py:166")
            return
        if hasattr(bar, "set_header"):
            bar.set_header(**header)

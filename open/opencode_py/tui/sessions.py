"""Session routing, session-tree navigation and history rendering.

Split out of :mod:`opencode_py.tui.app` as a mixin; method bodies are unchanged.
"""
from __future__ import annotations

from ..util.log import swallow
import time
from typing import Any, Callable, TYPE_CHECKING
from .. import errors
from .chat_view import ChatView, MessageBubble
from .input_bar import InputBar, SessionNavRequested
from .status_bar import StatusBar
from .subagent_footer import NavRequested
from pathlib import Path
from textual.containers import Vertical

if TYPE_CHECKING:
    from ..agent.loop import AgentLoop

class SessionsMixin:

    def _chat_for(self, session_id: str) -> ChatView:
        """Chat view for a session, creating (hidden) one on first use so a
        spawned sub-agent has a live, switchable conversation.

        Bounded: at most _MAX_HIDDEN_CHATS hidden chats are kept; the
        least-recently-created finished one is unmounted first, so long
        sessions with many agents no longer leak widgets/memory.
        """
        chat = self._chats.get(session_id)
        if chat is not None:
            return chat
        self._prune_hidden_chats()
        chat = ChatView()
        self._chats[session_id] = chat
        try:
            self.query_one("#chat-stack", Vertical).mount(chat, after=self._main_chat)
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/sessions.py:38")
            pass
        chat.display = "none"
        return chat


    def _prune_hidden_chats(self) -> None:
        try:
            hidden = [sid for sid, c in self._chats.items() if c is not getattr(self, "_main_chat", None)]
            overflow = len(hidden) - self._MAX_HIDDEN_CHATS + 1
            if overflow <= 0:
                return
            busy = set(getattr(self, "_busy_sessions", set()) or set())
            current = getattr(self, "_current_session_id", None)
            for sid in hidden:
                if overflow <= 0:
                    break
                if sid == current or sid in busy:
                    continue
                sess = (getattr(self, "_sessions", {}) or {}).get(sid)
                if getattr(sess, "parent_id", None):
                    pass  # finished sub-agent chats are prunable
                elif sid in (getattr(self, "_engines", {}) or {}):
                    continue  # live engine — keep
                c = self._chats.pop(sid, None)
                if c is not None:
                    try:
                        c.remove()
                    except Exception as _oc_exc:
                        swallow(_oc_exc, "tui/sessions.py:66")
                        pass
                    overflow -= 1
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/sessions.py:69")
            pass


    def _active_engine(self) -> AgentLoop:
        return self._engines.get(self._current_session_id, self.engine)


    def _active_session(self) -> Any:
        return self._sessions.get(self._current_session_id, self.session)


    def _collect_picker_rows(self) -> list[dict[str, Any]]:
        """Build the picker's rows fresh: live sessions first (running
        sub-agents marked), then persisted sessions from disk. Called both
        when the popup opens and by its refresh timer, so the list is never a
        stale snapshot.

        Deliberately NOT scoped to this project's directory: sessions are
        shared across projects on purpose here, and an exact-path filter hid
        every session saved before the workspace folder was renamed
        (opencode_in_python -> opencode_python). `list_sessions(directory=...)`
        stays available for anyone who wants upstream-style scoping."""
        from ..session import list_sessions, suggested_title

        rows: list[dict[str, Any]] = []
        seen: set[str] = set()

        def _collect(sid: str, title: str, agent: str, created: float | None, status: str) -> None:
            if not sid or sid in seen:
                return
            seen.add(sid)
            rows.append(
                {
                    "id": sid,
                    "title": title,
                    "agent": agent,
                    "created": created,
                    "status": status,
                }
            )

        for sid, sess in self._sessions.items():
            if getattr(sess, "parent_id", None):
                continue  # launched agents live INSIDE their parent, not here
            status = "running" if sid in self._busy_sessions else ""
            _collect(sid, sess.title, sess.agent, getattr(sess, "created", None), status)
        self._rebuild_family_from_disk()
        seen_count = 0
        for sess in list_sessions():
            if not getattr(sess, "has_messages", True):
                continue  # opened but never chatted — not a session to resume
            if getattr(sess, "parent_id", None):
                continue  # launched agents live INSIDE their parent, not here
            _collect(sess.id, sess.title or suggested_title(sess), sess.agent, sess.created, "")
            seen_count += 1
            if seen_count >= 200:
                break
        return rows


    def _children_of(self, parent_id: str) -> list[dict[str, Any]]:
        """Launched agents of one parent session, oldest first.

        Merges the live tree with saved children from disk so the in-parent
        list is complete even after a restart. Each entry: id/title/agent/
        created/status. Never raises.
        """
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        try:
            for c in self._sibling_records(parent_id):
                cid = c.get("id")
                if cid and cid not in seen:
                    seen.add(cid)
                    out.append({
                        "id": cid,
                        "title": c.get("title") or "sub-agent",
                        "agent": c.get("agent") or "build",
                        "created": c.get("created") or 0.0,
                        "status": c.get("status") or ("running" if cid in self._busy_sessions else "completed"),
                    })
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/sessions.py:151")
            pass
        try:
            from ..session import list_sessions as _list_all
            for sess in _list_all():
                try:
                    if getattr(sess, "parent_id", None) != parent_id:
                        continue
                    sid = getattr(sess, "id", "")
                    if not sid or sid in seen:
                        continue
                    seen.add(sid)
                    out.append({
                        "id": sid,
                        "title": getattr(sess, "title", "") or "sub-agent",
                        "agent": getattr(sess, "agent", "") or "build",
                        "created": getattr(sess, "created", 0.0) or 0.0,
                        "status": "running" if sid in self._busy_sessions else "completed",
                    })
                except Exception as _oc_exc:
                    swallow(_oc_exc, "tui/sessions.py:170")
                    continue
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/sessions.py:172")
            pass
        out.sort(key=lambda c: c.get("created") or 0)
        return out


    def _rebuild_family_from_disk(self) -> None:
        """Rebuild the parent/child navigation maps from saved sessions.

        The live maps only know children spawned THIS run; after a restart
        (or a resume) the arrows died because _parent_of found nothing. Every
        saved session with a parent_id re-registers here, so ←/→/↑ keep
        working across restarts and resumed parents find their children.
        Never raises; never duplicates existing live entries.
        """
        try:
            from ..session import list_sessions as _list_all
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/sessions.py:189")
            return
        try:
            saved = _list_all()
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/sessions.py:193")
            return
        for sess in saved:
            try:
                sid = getattr(sess, "id", "")
                pid = getattr(sess, "parent_id", None)
                if not sid or not pid or sid == pid:
                    continue
                if sid not in self._child_parent:
                    self._child_parent[sid] = pid
                records = self._children.setdefault(pid, [])
                if not any(c.get("id") == sid for c in records):
                    records.append({
                        "id": sid,
                        "title": getattr(sess, "title", "") or "sub-agent",
                        "agent": getattr(sess, "agent", "") or "build",
                        "created": getattr(sess, "created", 0.0) or 0.0,
                        "status": "completed",
                    })
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/sessions.py:212")
                continue


    def action_sessions(self) -> None:
        """Ctrl+R / `/sessions`: open the opencode-style picker (Today /
        Yesterday / older sections)."""
        from .session_list import SessionList

        sl = SessionList(
            self._collect_picker_rows(),
            current=self._current_session_id,
            on_rename=self._rename_session,
            on_delete=self._delete_session,
            on_save=self._save_current_session,
            on_refresh=self._collect_picker_rows,
            on_agents=self._children_of,
        )
        sl.on_open_agents = self._open_agents_view
        self.push_screen(sl, self._on_session_picked)


    def _open_agents_view(self, parent_id: str) -> None:
        """Open the in-parent agents list for one session.

        Shows ONLY the models launched inside that parent — Enter resumes
        one, Esc returns to the sessions picker.
        """
        from .session_list import AgentsView

        title = ""
        try:
            sess = self._sessions.get(parent_id)
            title = str(getattr(sess, "title", "") or "")
            if not title:
                from ..session import load_session, suggested_title

                loaded = load_session(parent_id)
                if loaded is not None:
                    title = loaded.title or suggested_title(loaded)
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/sessions.py:252")
            pass
        self.push_screen(
            AgentsView(
                parent_id,
                parent_title=title,
                agents=self._children_of(parent_id),
                current=self._current_session_id,
                on_refresh=lambda pid=parent_id: self._children_of(pid),
            ),
            self._on_agent_picked,
        )


    def _on_agent_picked(self, choice: str | None) -> None:
        if choice:
            self._resume_session(choice)
        try:
            self.query_one(InputBar).input.focus()
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/sessions.py:271")
            pass


    def action_models(self) -> None:
        """Ctrl+O: launch the model picker (same as `/models`).

        Not Ctrl+M: that is the Enter byte (0x0D), so no terminal can send it
        on its own and a binding for it is unreachable.
        """
        self._open_model_picker()


    def _on_session_picked(self, choice: str | None) -> None:
        if choice:
            self._resume_session(choice)
        try:
            self.query_one(InputBar).input.focus()
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/sessions.py:289")
            pass


    def _switch_session(self, session_id: str) -> None:
        if session_id == self._current_session_id:
            return
        # leaving a talking chat silences it and drops its tail — the new
        # session starts quiet instead of inheriting stale speech.
        try:
            self._clear_voice_all()
            self._stop_auto_voice()
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/sessions.py:301")
            pass
        if session_id in self._pruned:
            # A pruned id whose file still exists on disk is a STALE marker
            # (failed delete / prune-then-resume race) — heal it and open the
            # session instead of refusing with "finished and closed". Only a
            # genuinely file-less id keeps the refusal.
            try:
                from ..session import load_session as _load_pruned
                revived = _load_pruned(session_id)
            except Exception:
                revived = None
            if revived is not None:
                try:
                    self._pruned.discard(session_id)
                except Exception as _oc_exc:
                    swallow(_oc_exc, "tui/sessions.py:316")
                    pass
                self._sessions[session_id] = revived
                try:
                    self._rebuild_family_from_disk()
                except Exception as _oc_exc:
                    swallow(_oc_exc, "tui/sessions.py:321")
                    pass
                chat = self._chat_for(session_id)
                try:
                    self._render_history(chat, revived.messages)
                except Exception as exc:
                    errors.error(
                        "tui.sessions",
                        f"revived session {session_id} will open EMPTY: history render failed",
                        exc,
                    )
            else:
                self.notify("That sub-agent session is finished and closed.")
                try:
                    self.query_one(InputBar).focus()
                except Exception as _oc_exc:
                    swallow(_oc_exc, "tui/sessions.py:336")
                    pass
                return
        old = self._chats.get(self._current_session_id)
        if old is None:
            # the current chat vanished (delete-flow edge): fall back to the
            # main chat instead of crashing the keypress mid-navigation
            old = self._main_chat
            self._chats.setdefault(self.session.id, self._main_chat)
            if session_id != self.session.id:
                self._current_session_id = self.session.id
                self._switch_session(session_id)
                return
        self._current_session_id = session_id
        new = self._chat_for(session_id)
        if not new.is_attached:
            # mount failed inside _chat_for: retry once now that we are on the
            # UI thread; an unmounted widget would leave a BLANK screen while
            # internal state believes the switch happened
            try:
                self.query_one("#chat-stack", Vertical).mount(new, after=self._main_chat)
            except Exception:
                self.notify("Could not open that session's view.", severity="error")
                return
        # Sibling-arrow switches land on chats that were created empty (never
        # rendered): fill from the session NOW, or ←/→ shows a blank screen
        # even though the transcript exists on disk/in memory.
        try:
            if not list(new.query("*")):
                sess = self._sessions.get(session_id)
                msgs = getattr(sess, "messages", None) if sess is not None else None
                if not msgs:
                    from ..session import load_session as _load_sw
                    loaded = _load_sw(session_id)
                    if loaded is not None:
                        self._sessions[session_id] = loaded
                        msgs = loaded.messages
                if msgs:
                    self._render_history(new, msgs)
        except Exception as exc:
            # Wraps the whole session load: a failure here means the switch
            # half-happened and the chat on screen may not match the
            # selected session, with no visible reason.
            errors.error(
                "tui.sessions",
                f"session switch to {session_id} did not complete cleanly",
                exc,
            )
        old.display = "none"
        new.display = "block"
        # Flush a deferred render (task-row click created this chat before it
        # was mounted): without this the child opens EMPTY and the click
        # looks dead.
        try:
            if self._pending_render.pop(session_id, False):
                sess = self._sessions.get(session_id)
                msgs = getattr(sess, "messages", None) if sess is not None else None
                if msgs:
                    self._render_history(new, msgs)
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/sessions.py:395")
            pass
        # While a sub-agent chat has focus, ↑/←/→ route to session navigation
        # (parent / siblings) instead of scrolling the message list.
        new._session_is_child = bool(self._parent_of(session_id))
        # Remember the last sub-agent viewed under its parent so returning and
        # pressing ctrl+down resumes it (official keeps the selection).
        parent_of = self._parent_of(session_id)
        if parent_of:
            self._last_selection[parent_of] = session_id
        sess = self._sessions.get(session_id)
        if sess and sess.title:
            self.notify(f"Session: {sess.title}", markup=False)
        self._update_header()
        self._update_footer()
        # The footer's context hint (`12,345 (6%)`) must follow the session
        # now on screen — otherwise the previous session's token counter stays
        # painted on the status bar after a switch/resume.
        try:
            self.query_one(StatusBar).set_usage(self._usage.get(session_id) or {})
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/sessions.py:415")
            pass
        # Streaming/busy indicators belong to the VIEWED session too: watching
        # an idle chat while another one streams used to keep the spinner (and
        # a locked input) on screen for no reason.
        self._sync_streaming_visuals()
        self.query_one(InputBar).focus()


    def _sync_streaming_visuals(self) -> None:
        """Reflect the CURRENTLY VIEWED session's busy state in the status bar
        and input bar. Busy/streaming visuals were global: viewing idle session
        B while A streamed showed A's spinner over B, and only A's turn end
        cleared it. Safe to call from anywhere on the UI thread."""
        here = self._current_session_id in self._busy_sessions
        try:
            self.query_one(StatusBar).set_streaming(here)
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/sessions.py:432")
            pass
        try:
            self.query_one(InputBar).set_busy(here)
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/sessions.py:436")
            pass


    def on_open_task_session(self, event: Any) -> None:
        sid = getattr(event, "sid", None)
        if not sid:
            return
        # A finished sub-agent from a previous run is only persisted on disk;
        # reopening its task row must load that history instead of switching
        # to an empty chat. Children deleted from disk report clearly instead
        # of silently doing nothing (the "click does nothing" symptom).
        if sid not in self._sessions and sid not in self._pruned:
            from ..session import load_session

            sess = load_session(sid)
            if sess is not None:
                self._sessions[sid] = sess
                try:
                    self._rebuild_family_from_disk()
                except Exception as _oc_exc:
                    swallow(_oc_exc, "tui/sessions.py:456")
                    pass
                chat = self._chat_for(sid)
                if chat.is_attached:
                    # Replay only into an EMPTY chat, the same rule
                    # _switch_session uses. _render_history appends, and a
                    # sub-agent that already streamed here has its bubbles
                    # mounted — replaying on top printed every message twice,
                    # the final report most obviously.
                    if not list(chat.query("*")):
                        self._render_history(chat, sess.messages)
                else:
                    # chat not mounted yet (fresh popup-less open): defer the
                    # render until after the switch mounts it, else mount()
                    # raises and the click appears dead.
                    try:
                        self._pending_render[sid] = True
                    except Exception as _oc_exc:
                        swallow(_oc_exc, "tui/sessions.py:473")
                        pass
            else:
                self.notify("Sub-agent session not found on disk.", severity="error")
                try:
                    self.query_one(InputBar).focus()
                except Exception as _oc_exc:
                    swallow(_oc_exc, "tui/sessions.py:479")
                    pass
                return
        self._switch_session(sid)


    # -- sub-agent navigation (official session.child.*) ------------------
    def _sibling_records(self, parent_id: str) -> list[dict[str, Any]]:
        """Every sub-agent the parent spawned, oldest first (official numbers
        ``(2 of 4)`` by creation time across ALL of the parent's children)."""
        return sorted(self._children.get(parent_id, []), key=lambda c: c.get("created") or 0)


    def _session_nav_active(self) -> bool:
        """Arrow-key session routing must not fight the prompt's cursor/history
        keys: while the user is typing a non-empty prompt the arrows belong to
        the input (official's input scope wins over the session scope)."""
        from .input_bar import PromptTextArea

        focused = self.focused
        if isinstance(focused, PromptTextArea):
            return not focused.text.strip()
        return True


    def _parent_of(self, session_id: str) -> str | None:
        """The parent session id of a session: from the live children registry
        (authoritative while a sub-agent is/was running) or the saved parent_id."""
        parent = self._child_parent.get(session_id)
        if parent:
            return parent
        sess = self._sessions.get(session_id)
        pid = getattr(sess, "parent_id", None)
        if pid:
            return pid
        # Reverse-index fallback: the registry entry may have been dropped
        # (restart, prune) while the parent's sibling list still holds us.
        # Heal the fast path so the next lookup doesn't rescan.
        for _pid, records in self._children.items():
            if any(c.get("id") == session_id for c in records):
                self._child_parent[session_id] = _pid
                return _pid
        return None


    def _go_parent(self) -> bool:
        parent_id = self._parent_of(self._current_session_id)
        if parent_id:
            self._switch_session(parent_id)
            return True
        return False


    def _go_prev(self) -> None:
        self._move_sibling(-1)


    def _go_next(self) -> None:
        self._move_sibling(1)


    def _move_sibling(self, direction: int) -> bool:
        parent_id = self._parent_of(self._current_session_id)
        if not parent_id:
            return False
        siblings = self._sibling_records(parent_id)
        if len(siblings) <= 1:
            # Single child: heal a stale registry (restart/prune wiped the
            # record) from the in-memory session's parent_id so the arrows
            # at least stay on this child instead of dying silently.
            sess = self._sessions.get(self._current_session_id)
            pid = getattr(sess, "parent_id", None) if sess else None
            if pid and parent_id == pid:
                records = self._children.setdefault(parent_id, [])
                if not any(c.get("id") == self._current_session_id for c in records):
                    records.append({"id": self._current_session_id, "created": time.time(), "status": "running"})
            return False
        try:
            index = next(i for i, c in enumerate(siblings) if c.get("id") == self._current_session_id)
        except StopIteration:
            # Registry drift: current child missing from the sibling list
            # (record evicted but parent link intact). Re-add ourselves so
            # the NEXT press can move instead of dying forever.
            self._children.setdefault(parent_id, []).append(
                {"id": self._current_session_id, "created": time.time(), "status": "running"})
            return False
        target = (index + direction) % len(siblings)
        target_sid = siblings[target]["id"]
        # Never land on a pruned (closed/deleted) child: skip straight to
        # the next live sibling. Landing on one hit the "finished and closed"
        # wall and LOOKED like ←/→ doing nothing.
        if target_sid in self._pruned:
            for step in range(1, len(siblings)):
                cand = siblings[(index + direction * step) % len(siblings)]["id"]
                if cand not in self._pruned:
                    target_sid = cand
                    break
            else:
                return False
        self._switch_session(target_sid)
        return True


    def _go_first_child(self) -> None:
        # open the in-parent agents list so the user touches and picks the
        # launched agent inside this session — except a single child, which
        # resumes directly (keeps the old single-agent flow instant).
        kids = self._children_of(self._current_session_id)
        if not kids:
            return
        if len(kids) == 1:
            self._switch_session(kids[0]["id"])
            return
        last = self._last_selection.get(self._current_session_id)
        if last and any(c.get("id") == last for c in kids):
            self._switch_session(last)
            return
        # no prior selection: resume the first child directly (old instant
        # flow); the agents popup stays one Enter away on the picker row.
        self._switch_session(kids[0]["id"])


    def action_fd_parent(self) -> None:
        if self._session_nav_active():
            self._go_parent()


    def action_fd_prev(self) -> None:
        if self._session_nav_active():
            self._go_prev()


    def action_fd_next(self) -> None:
        if self._session_nav_active():
            self._go_next()


    def action_fd_first(self) -> None:
        self._go_first_child()


    def _chat_scroll_target(self) -> ChatView | None:
        """The chat of the session currently on screen (used by HOME/END/PgUp/
        PgDn so they scroll the conversation no matter what widget has focus)."""
        return self._chats.get(self._current_session_id)


    def action_chat_home(self) -> None:
        chat = self._chat_scroll_target()
        if chat is not None:
            chat.scroll_home(animate=False)


    def action_chat_end(self) -> None:
        chat = self._chat_scroll_target()
        if chat is not None:
            chat.scroll_end(animate=False)


    def action_chat_page_up(self) -> None:
        chat = self._chat_scroll_target()
        if chat is not None:
            chat.scroll_page_up(animate=False)


    def action_chat_page_down(self) -> None:
        chat = self._chat_scroll_target()
        if chat is not None:
            chat.scroll_page_down(animate=False)


    def on_nav_requested(self, event: NavRequested) -> None:
        if event.action == "parent":
            self._go_parent()
        elif event.action == "prev":
            self._go_prev()
        elif event.action == "next":
            self._go_next()


    def on_session_nav_requested(self, event: SessionNavRequested) -> None:
        """Arrow keys pressed with an empty prompt (posted from the input bar).
        `↑` parent (or previous prompt when there is no parent), `←`/`→` cycle
        the parallel sub-agent siblings, `↓` recalls the next prompt / draft."""
        key = event.direction
        if key == "up":
            bar = self.query_one(InputBar)
            if (
                bar.input.value == ""
                and bar._hist_index == len(bar._history)
                and bar._draft
            ):
                # the final ↓ cleared the box: the next ↑ restores what we were
                # typing instead of session-navigating away
                bar.input.value = bar._draft
                bar.input.cursor_position = len(bar._draft)
                return
            if not self._go_parent():
                # No parent → fall back to prompt history (main session AND a
                # parent-of-agents: being a parent gives ↑ no navigation
                # target, so sent chats must still recall). Only a CHILD whose
                # parent record vanished stays silent (no popup, no pasted
                # prompts).
                if self._parent_of(self._current_session_id) is None:
                    bar.recall_history("up")
        elif key == "down":
            self.query_one(InputBar).recall_history("down")
        elif key == "left":
            self._go_prev()
        elif key == "right":
            self._go_next()


    def _update_footer(self) -> None:
        """Show the subagent footer only while viewing a child session
        (opencode's SubagentFooter: `Build (2 of 4)` + usage + parent/prev/next)."""
        footer = getattr(self, "_footer", None)
        if footer is None:
            return
        current = self._current_session_id
        self._mark_selected_task()
        parent_id = self._parent_of(current)
        if not parent_id or not getattr(self.cfg, "subagent_footer", False):
            # removed by default (cfg.subagent_footer=False): the bar cost a
            # screen line on phones and duplicated what ↑ / ← / → already do.
            # Task-row highlighting above still runs — it is independent.
            footer.hide()
            return
        sess = self._sessions.get(current)
        siblings = self._sibling_records(parent_id)
        if not siblings:
            footer.hide()
            return
        index = next(
            (i for i, c in enumerate(siblings) if c.get("id") == current),
            0,
        )
        footer.show(
            label=str(getattr(sess, "agent", "") or "build").title(),
            index=index + 1,
            total=len(siblings),
            usage=self._usage.get(current),
        )


    def _mark_selected_task(self) -> None:
        """Highlight which sub-agent is currently selected on the parent's task
        rows: the child being viewed, or — while sitting at the parent — the
        child you last opened (official opencode marks the active sub-agent).

        Only the PARENT chat's task bubbles are touched (task rows live there),
        and a no-op selection change returns early: this ran per footer update
        over EVERY chat x EVERY bubble, which visibly janked long sessions on
        armv7 mid-stream."""
        current = self._current_session_id
        parent_id = self._parent_of(current)
        if parent_id:
            target_sid, parent_chat_id = current, parent_id
        else:
            target_sid = self._last_selection.get(current) or ""
            parent_chat_id = current
        marked = getattr(self, "_marked_tasks", None)
        if marked == (parent_chat_id, target_sid):
            return
        self._marked_tasks = (parent_chat_id, target_sid)
        chat = self._chats.get(parent_chat_id)
        if chat is None:
            return
        try:
            for bubble in chat.query(MessageBubble):
                if bubble.role != "tool" or not isinstance(bubble.content, dict) or bubble.content.get("tool") != "task":
                    continue
                meta = bubble.content.get("metadata") or {}
                bubble.selected = str(meta.get("sessionId") or "") == target_sid
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/sessions.py:753")
            pass


    def _resume_session(self, session_id: str) -> None:
        """Switch to a live session or load a persisted one (engine + chat
        rebuilt around its saved history) so the conversation can continue."""
        if session_id in self._sessions:
            self._switch_session(session_id)
            return
        from ..session import load_session, suggested_title
        from ..agent.loop import AgentLoop
        from ..tools import build_registry as build_tool_registry

        sess = load_session(session_id)
        if sess is not None:
            # A prune marker with a live file on disk is STALE — it means the
            # in-memory chat was torn down, not that history vanished. The old
            # check order refused to open resurrected rows ("That sub-agent
            # session is finished and closed") after a failed delete left the
            # body behind. Disk presence wins; heal the marker. A resumed
            # CHILD specifically must never stay pruned, or every later click
            # on its task row silently does nothing.
            try:
                if session_id in self._pruned:
                    self._pruned.discard(session_id)
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/sessions.py:779")
                pass
        if sess is None:
            if session_id in self._pruned:
                self.notify(
                    "That session no longer exists (it was closed or deleted)."
                )
            else:
                self.notify("Session not found.")
            return
        if not sess.title:
            sess.title = suggested_title(sess)
        # a session saved under a custom agent that was since deleted or
        # renamed must not strand the engine: fall back to build loudly
        # instead of running under a ghost identity.
        resume_agent = sess.agent or "build"
        try:
            from ..permission import list_agents as _la

            known_agents = {n for n, _d, _c in _la(self.cfg)}
            if resume_agent not in known_agents:
                self.notify(
                    f"Session's agent '{resume_agent}' no longer exists — using build.",
                    severity="warning",
                )
                resume_agent = "build"
                sess.agent = "build"
        except Exception:
            resume_agent = sess.agent or "build"
        engine = AgentLoop(
            cfg=self.cfg,
            registry=build_tool_registry(self.cfg),
            directory=Path(sess.directory) if sess.directory else self.directory,
            auth=self.auth,
            agent=resume_agent,
            provider_id=sess.provider or self.cfg.provider,
            model_id=sess.model or self.cfg.model,
            session_id=session_id,
        )
        engine.on_event = self._on_engine_event
        engine.interrupt = self._session_interrupt_checker(session_id)
        engine.permission.ask_callback = self._permission_ask
        engine.question_service.ask_callback = self._question_ask
        # the pin is a workspace-wide choice: a resumed session must not
        # silently come back UNLOCKED and fail over on its first error
        engine.rotation_locked = bool(getattr(self.cfg, "rotation_lock", False))
        engine.set_history(sess.messages)
        self._engines[session_id] = engine
        self._sessions[session_id] = sess
        # a resumed child must rejoin the family tree so the arrows keep
        # working (the live maps were empty after a restart).
        try:
            self._rebuild_family_from_disk()
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/sessions.py:832")
            pass
        chat = self._chat_for(session_id)
        chat.display = "none"
        self._render_history(chat, sess.messages)
        # a resumed PARENT's task rows must be clickable too: re-link every
        # known child session onto its row once the bubbles exist. Rendering
        # mounts in 15-row chunks via call_later, so the rows may not exist
        # yet — retry until linked or the chat switches away.
        try:
            self._schedule_relink(session_id, attempts=40)
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/sessions.py:843")
            pass
        self._switch_session(session_id)
        # restore the context-usage hint immediately (it used to stay blank
        # until the next turn completed): same request-size estimate the
        # engine uses — system prompt + history + tool schemas — so a
        # reopened chat shows its true footprint instead of reading low.
        try:
            eng = self._engines.get(session_id)
            est = eng.estimate_history_tokens() if eng is not None else 0
            if est:
                usage = {"input_tokens": est, "output_tokens": 0, "total_tokens": est}
                ctx_size = 0
                from ..providers import model_context_size

                ctx_size = model_context_size(
                    sess.provider or self.cfg.provider,
                    sess.model or self.cfg.model,
                    auth=self.auth,
                )
                if ctx_size:
                    usage["context_size"] = ctx_size
                self._usage[session_id] = usage
                self.query_one(StatusBar).set_usage(usage)
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/sessions.py:867")
            pass
        self.notify(f"Resumed: {sess.title or session_id}", markup=False)


    def _rename_session(self, session_id: str, title: str) -> str | None:
        """Persist a rename; returns an error message or None on success."""
        from ..session import load_session, save_session

        try:
            sess = self._sessions.get(session_id)
            if sess is None:
                sess = load_session(session_id)
                if sess is None:
                    return "Session not found."
            sess.title = title
            # Renames only stick to disk when the session actually has a
            # conversation — a never-chatted scratch session must not be
            # materialised into a file just because it was titled. Say so,
            # instead of silently dropping the title on the next launch.
            if getattr(sess, "messages", None):
                save_session(sess)
            else:
                self.notify(
                    "Rename kept for this run — the title saves once the session has messages.",
                    severity="warning",
                    timeout=5,
                )
        except Exception as e:
            return f"Rename failed: {e}"
        return None


    def _save_current_session(self, session_id: str = "") -> bool:
        """Save the highlighted session right now (Ctrl+S in the picker). An
        empty id means "the session I'm in" (the old popup-wide Save). Includes
        the engine's live history so the durable copy matches what's on
        screen. Returns False when there was nothing to save / it failed."""
        from ..session import save_session

        sid = session_id or self._current_session_id
        sess = self._sessions.get(sid)
        if sess is None:
            from ..session import load_session

            sess = load_session(sid)
            if sess is None:
                self.notify("Session not found.", severity="error")
                return False
        engine = self._engines.get(sid)
        try:
            if engine is not None:
                history = engine.get_history()
                if history:
                    sess.messages = history
            if not getattr(sess, "messages", None):
                self.notify("Nothing to save — this session has no conversation.")
                return False
            save_session(sess)
        except Exception as e:
            self.notify(f"Save failed: {e}", severity="error", markup=False)
            return False
        self.notify(f"Session saved: {sess.title or '(untitled)'}", markup=False)
        return True


    def _action_new(self) -> None:
        """`/new` / `/clear`: start a brand-new session in place.

        The old conversation is durably saved first (it stays resumable from
        the picker), then the workspace resets exactly like the delete-main-
        session flow: same engine instance re-registered under a new session
        id with empty history, chat cleared, header/footer/usage refreshed."""
        from ..session import new_session

        try:
            self._save_all_live_sessions()
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/sessions.py:944")
            pass
        old_id = self.session.id
        old_chat = self._chats.get(old_id)
        self.session = new_session(
            directory=str(self.directory),
            provider=self.cfg.provider,
            model=self.cfg.model,
            agent=self.engine.agent,
        )
        self.engine.session_id = self.session.id
        try:
            self.engine.set_history([])
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/sessions.py:957")
            pass
        self.engine.clear_prompts()
        self._engines.pop(old_id, None)
        self._sessions.pop(old_id, None)
        self._turn.pop(old_id, None)
        if old_chat is not None and old_chat is not self._main_chat:
            try:
                old_chat.remove()
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/sessions.py:966")
                pass
            self._chats.pop(old_id, None)
        else:
            self._main_chat.clear()
        self._chats[self.session.id] = self._main_chat
        self._sessions[self.session.id] = self.session
        self._engines[self.session.id] = self.engine
        self._current_session_id = self.session.id
        self._active_turn_session_id = self.session.id
        self._usage.pop(old_id, None)
        try:
            self.query_one(StatusBar).set_usage({})
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/sessions.py:979")
            pass
        self._main_chat.show_logo()
        self._update_header()
        self._update_footer()
        self.query_one(InputBar).focus()


    def _delete_session(self, session_id: str) -> bool:
        """Delete a session from disk (and its live registrations). Any session
        is deletable — deleting the one you're in resets the workspace to a
        brand-new session so a later save doesn't resurrect the deleted file.
        A session mid-turn is still protected (it owns a running engine)."""
        from ..session import delete_session, new_session

        if session_id in self._busy_sessions:
            return False  # never drop a running turn out from under it
        if session_id == self.session.id:
            # deleting the workspace itself -> start a fresh session in place
            old_id = self.session.id
            viewing_old = self._current_session_id == old_id
            self.session = new_session(
                directory=str(self.directory),
                provider=self.cfg.provider,
                model=self.cfg.model,
                agent=self.engine.agent,
            )
            self.engine.session_id = self.session.id
            try:
                self.engine.set_history([])
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/sessions.py:1009")
                pass
            self._chats.pop(old_id, None)
            self._sessions.pop(old_id, None)
            self._engines.pop(old_id, None)
            self._turn.pop(old_id, None)
            self.engine.clear_prompts()
            self._chats[self.session.id] = self._main_chat
            self._sessions[self.session.id] = self.session
            self._engines[self.session.id] = self.engine
            self._pruned.discard(old_id)
            if self._active_turn_session_id == old_id:
                self._active_turn_session_id = self.session.id
            # Same cleanup /new does: drop the old token counter, reset the
            # status bar, clear + logo the main chat.
            self._usage.pop(old_id, None)
            try:
                self.query_one(StatusBar).set_usage({})
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/sessions.py:1027")
                pass
            self._main_chat.clear()
            if viewing_old or self._current_session_id not in self._sessions:
                # Land in the fresh workspace. (The old build repointed
                # _current_session_id unconditionally WITHOUT touching what was
                # on screen: watching another chat while deleting the main one
                # left prompts streaming into a display:none widget, and the
                # batch flow could blank the pane entirely.)
                self._current_session_id = self.session.id
                self._main_chat.display = "block"
                self._main_chat._session_is_child = False
                self._main_chat.show_logo()
            # else: the user is watching another LIVE chat — leave it exactly
            # where it is instead of yanking them into the empty workspace.
            self._update_header()
            self._update_footer()
            self._sync_streaming_visuals()
            # Do NOT persist the fresh replacement session here: saving an empty
            # session on every delete is how 0-message ghost files accumulate.
            # It only gets its own file once the user actually chats and one of
            # the save conditions fires (crash-safety autosave / exit / close).
            self.notify("Session deleted — starting a new one.")
            delete_session(old_id)
            return True
        # a non-main session (resumed, sub-agent, …)
        was_live = session_id in self._sessions
        was_current = session_id == self._current_session_id
        if was_current:
            self._switch_session(self.session.id)  # hop back to a live session
        # drop the deleted session from the sub-agent family tree
        self._children.pop(session_id, None)  # its own children first
        for pid, records in list(self._children.items()):
            self._children[pid] = [r for r in records if r.get("id") != session_id]
        self._child_parent.pop(session_id, None)
        self._task_start.pop(session_id, None)
        self._usage.pop(session_id, None)
        self._sessions.pop(session_id, None)
        self._engines.pop(session_id, None)
        chat = self._chats.pop(session_id, None)
        self._turn.pop(session_id, None)
        self._pruned.add(session_id)
        self._running_agents.pop(session_id, None)
        self._refresh_running_agents()
        if chat is not None:
            try:
                chat.remove()
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/sessions.py:1074")
                pass
        if was_current:
            self.notify("Session deleted.")
        # ALWAYS drop the durable copy too. The old `True if was_live else
        # delete_session(...)` skipped the disk delete for any session that was
        # live in RAM — but a RESUMED session was loaded from a file, so the
        # file survived: the picker's 2s refresh resurrected the row, the next
        # Ctrl+D was needed (and only then actually deleted it), and until then
        # clicking the row hit the _pruned wall ("finished and closed").
        deleted_on_disk = delete_session(session_id)
        return True if (was_live or deleted_on_disk) else False


    def _render_history(self, chat: ChatView, messages: list[dict[str, Any]]) -> None:
        """Replay saved messages into a ChatView, instant-first.

        Only the last ~80 message-tasks mount now (~0.4s); older tasks sit in
        chat._history_pending and stream in 2-chunk look-ahead as the user
        scrolls up — never the whole 4000 at once, never a loading spinner
        they can catch.
        """
        import json as _json

        def _args(raw: Any) -> dict[str, Any]:
            if isinstance(raw, dict):
                return raw
            if isinstance(raw, str):
                try:
                    parsed = _json.loads(raw)
                except _json.JSONDecodeError:
                    return {}
                return parsed if isinstance(parsed, dict) else {}
            return {}

        def _reasoning_text(raw: Any) -> str:
            if isinstance(raw, str):
                return raw
            if isinstance(raw, list):
                parts = []
                for p in raw:
                    if isinstance(p, dict):
                        parts.append(str(p.get("text") or p.get("content") or ""))
                    elif isinstance(p, str):
                        parts.append(p)
                return "\n".join(p for p in parts if p)
            return str(raw or "")

        tasks: list[Callable[[], None]] = []  # one deferred bubble-mount per message
        pending: list[tuple[str, str, dict[str, Any]]] = []  # (call_id, toolname, input)
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            if role == "system":
                continue
            if role == "compaction" or msg.get("compaction"):
                text = str(content or "")
                tasks.append(lambda _b=None, t=text, _c=chat: _c.append_compaction(t, before=_b, scroll=False))
                continue
            if role == "user":
                text = str(content or "")
                agent = str(msg.get("agent") or "build")
                tasks.append(lambda _b=None, t=text, a=agent, _c=chat: _c.append_user(t, agent=a, before=_b, scroll=False))
                continue
            if role == "assistant":
                reasoning = msg.get("reasoning_content")
                if reasoning:
                    rtext = _reasoning_text(reasoning)
                    tasks.append(lambda _b=None, t=rtext, _c=chat: _c.append_reasoning(t, before=_b, scroll=False))
                if content:
                    ctext = str(content)
                    tasks.append(lambda _b=None, t=ctext, _c=chat: _c.append_assistant(t, before=_b, scroll=False))
                for call in msg.get("tool_calls") or []:
                    fn = call.get("function") or {}
                    pending.append(
                        (call.get("id", ""), fn.get("name", "tool"), _args(fn.get("arguments")))
                    )
                continue
            if role == "tool":
                call_id = msg.get("tool_call_id") or ""
                tool = msg.get("name") or "tool"
                tool_input: dict[str, Any] = {}
                for j, (cid, name, args) in enumerate(pending):
                    if cid == call_id:
                        tool, tool_input = name, args
                        pending.pop(j)
                        break
                else:
                    if pending:
                        # call_id missing (out-of-order results / poisoned save):
                        # match the OLDEST pending call for the SAME tool name so
                        # parallel tool results don't get swapped between
                        # different tools; only fall back to the oldest call when
                        # no same-name call is still waiting. Use the pending
                        # call's own id/name/args so the row stays accurate.
                        idx = 0
                        for j, (cid2, name2, args2) in enumerate(pending):
                            if name2 == tool:
                                idx = j
                                break
                        call_id, tool, tool_input = pending.pop(idx)
                metadata: dict[str, Any] = {}
                # Re-attach a persisted child-session link (see run_turn):
                # resumed parents render task rows from saved history, and
                # the click handler reads metadata.sessionId.
                try:
                    psid = msg.get("session_id") or msg.get("sessionId")
                    if tool == "task" and psid:
                        metadata["sessionId"] = psid
                except Exception as _oc_exc:
                    swallow(_oc_exc, "tui/sessions.py:1183")
                    pass
                if tool == "todowrite":
                    # history stores the todos as JSON text; the renderer reads
                    # them from metadata.todos
                    try:
                        parsed = _json.loads(str(content or ""))
                        if isinstance(parsed, list):
                            metadata["todos"] = parsed
                    except _json.JSONDecodeError as _oc_exc:
                        swallow(_oc_exc, "opencode_py/tui/sessions.py:1229")
                        pass
                payload: dict[str, Any] = {
                    "id": call_id,
                    "tool": tool,
                    "input": dict(tool_input),
                    # "completed" (not "done"): tool renderers + the gray
                    # block frame key off this exact value
                    "status": "completed",
                    "done": True,
                    "output": content or "",
                    # copy: every lambda below closes over `payload` by NAME,
                    # so without this all rows share the LAST dict and every
                    # task row opens the same (last) child.
                    "metadata": dict(metadata),
                }
                tasks.append(lambda _b=None, p=dict(payload, metadata=dict(metadata), input=dict(tool_input)), _c=chat: _c.append_tool(p, before=_b, scroll=False))
                continue
            if content:
                text = str(content)
                tasks.append(lambda _b=None, t=text, _c=chat: _c.append_meta(t, before=_b, scroll=False))

        # A session cut off mid-tool-run (interrupt / sudden kill / old
        # poisoned save) ends with an assistant tool_calls message whose results
        # never arrived. Render those as interrupted rows so resuming shows
        # exactly where the turn stopped instead of silently dropping them.
        for call_id, name, tool_input in pending:
            payload = {
                "id": call_id,
                "tool": name,
                "input": dict(tool_input) if isinstance(tool_input, dict) else tool_input,
                "status": "interrupted",
                "done": False,
                "output": "",
                "metadata": {},
            }
            tasks.append(lambda _b=None, p=payload, _c=chat: _c.append_tool(dict(p), before=_b, scroll=False))

        sid = ""
        try:
            sid = str(getattr(getattr(chat, "app", None), "_current_session_id", "") or "")
        except Exception:
            sid = ""
        if len(tasks) <= 90:
            for t in tasks:
                try:
                    t()
                except Exception as _oc_exc:
                    swallow(_oc_exc, "tui/sessions.py:1239")
                    continue
            try:
                chat.set_history_pending([], session_id=sid)
                chat.scroll_end(animate=False)
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/sessions.py:1244")
                pass
            return
        split = max(0, len(tasks) - 80)
        older = tasks[:split]
        recent = tasks[split:]
        for t in recent:
            try:
                t()
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/sessions.py:1253")
                continue
        try:
            chat.set_history_pending(list(reversed(older)), session_id=sid, chunk=80)
            chat.scroll_end(animate=False)
            try:
                chat.call_after_refresh(lambda: self.call_later(chat._maybe_prefetch))
            except Exception:
                try:
                    self.call_later(chat._maybe_prefetch)
                except Exception as _oc_exc:
                    swallow(_oc_exc, "tui/sessions.py:1263")
                    pass
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/sessions.py:1265")
            pass

"""Engine event bridge, streaming deltas and task rows for the Textual TUI.

Split out of :mod:`opencode_py.tui.app` as a mixin; method bodies are unchanged.
"""
from __future__ import annotations

from ..util.log import swallow
import asyncio, sys, threading, time, traceback
from typing import Any
from .chat_view import ChatView, MessageBubble
from .input_bar import InputBar, format_duration
from .status_bar import StatusBar

class EventsMixin:

    # -- running agents ---------------------------------------------------
    def _refresh_running_agents(self) -> None:
        """Show the launched sub-agents in the status line above the prompt,
        like opencode's `Delegating...` indicator (transient, no sidebar)."""
        try:
            bar = self.query_one(InputBar)
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/events.py:21")
            return
        bar.set_running_agents(list(self._running_agents.values()))


    # -- engine event bridge ---------------------------------------------
    def _on_engine_event(self, event: dict[str, Any]) -> None:
        # Called from the engine thread; hop to the UI thread.
        if getattr(self, "_thread_id", None) == threading.get_ident():
            # Already on the UI thread (e.g. a /command handler that makes the
            # engine emit an event, like /undo): call_from_thread would raise
            # RuntimeError, so handle inline instead.
            self._handle_event(event)
            return
        # Never block engine (or sub-agent) worker threads on the UI render:
        # every event rides the async bridge (FIFO on the app loop, so ordering
        # with the text deltas is preserved) and the worker keeps streaming.
        # Before this, each tool/subagent event did a blocking call_from_thread
        # rendezvous — N parallel children meant an N-way UI-thread storm that
        # stalled all streams on every tool row.
        self._schedule_async(event)
        return


    def _schedule_async(self, event: dict[str, Any]) -> None:
        loop = self._loop
        if loop is None:
            # App not running / already tearing down (Textual resets its
            # _loop to None at teardown) — dropping deltas is correct here.
            return
        try:
            asyncio.run_coroutine_threadsafe(self._async_handle(event), loop)
        except RuntimeError as _oc_exc:
            swallow(_oc_exc, "opencode_py/tui/events.py:55")
            pass  # loop closed mid-shutdown — dropping is correct
        except Exception as e:
            # Anything else must stay visible: a silent drop looks exactly
            # like the model freezing mid-stream.
            traceback.print_exc(file=sys.stderr)
            sys.stderr.write(f"[tui] delta schedule failed: {e!r}\n")


    async def _async_handle(self, event: dict[str, Any]) -> None:
        try:
            with self._context():
                self._handle_event(event)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # A delta-render failure must never vanish into a dropped future
            # (``run_coroutine_threadsafe`` swallows coroutine exceptions and
            # would leave the screen silently frozen mid-stream). Log it so the
            # bug is actually visible in the console.
            traceback.print_exc(file=sys.stderr)
            sys.stderr.write(f"[tui] async event handler error: {e!r}\n")


    def _queue_delta(self, session_id: str, text: str, kind: str) -> None:
        if not text:
            # drop empty deltas entirely — an empty text chunk must not create
            # an assistant bubble that never gets filled
            return
        ui_thread = getattr(self, "_thread_id", None)
        if ui_thread is not None and threading.get_ident() != ui_thread:
            # Single-writer invariant: _pending and _delta_timer are only
            # touched by the UI thread (deltas arrive via _schedule_async on
            # the app loop; flushes run on the flush timer). If a future caller
            # ever queues off-thread, marshal to the UI thread instead of
            # racing the timer/swap instead of corrupting the buffer.
            try:
                self.call_from_thread(self._queue_delta, session_id, text, kind)
            except RuntimeError as _oc_exc:
                swallow(_oc_exc, "opencode_py/tui/events.py:93")
                pass
            return
        # Route to active buffer if this is the actively streaming session,
        # otherwise buffer in background to prevent cross-session bleeding.
        target = self._pending if session_id == self._active_turn_session_id else self._pending_bg
        buf = target.setdefault(session_id, {"text": [], "reasoning": []})
        buf[kind].append(text)
        if self._delta_timer is None:
            self._delta_timer = self.set_timer(0.03, self._flush_deltas)


    def _cancel_delta_timer(self) -> None:
        if self._delta_timer is not None:
            try:
                self._delta_timer.stop()
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/events.py:107")
                pass
            self._delta_timer = None


    def _flush_deltas(self) -> None:
        """Render any buffered text/reasoning deltas (one render per batch).

        Chunks are joined into a SINGLE string per session/kind: the old code
        called stream_delta per chunk (one full bubble re-render + scroll_end
        layout per token — the 32-bit stutter). Ordering preserved.
        Each flush also paces the busy spinner to real arrival speed.
        """
        self._cancel_delta_timer()
        # Flush active session buffer
        pending = self._pending
        self._pending = {}
        for session_id, buf in pending.items():
            chat = self._chat_for(session_id)
            reasoning = "".join(buf.get("reasoning") or [])
            if reasoning:
                chat.stream_reasoning_delta(reasoning)
            text = "".join(buf.get("text") or [])
            if text:
                chat.end_reasoning()
                chat.stream_delta(text)
            try:
                n = len(reasoning) + len(text)
                if n > 0:
                    self.query_one(InputBar).note_stream_activity(n)
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/events.py:137")
                pass
        # Flush background session buffers (non-active sessions)
        if self._pending_bg:
            bg_pending = self._pending_bg
            self._pending_bg = {}
            for session_id, buf in bg_pending.items():
                chat = self._chat_for(session_id)
                reasoning = "".join(buf.get("reasoning") or [])
                if reasoning:
                    chat.stream_reasoning_delta(reasoning)
                text = "".join(buf.get("text") or [])
                if text:
                    chat.end_reasoning()
                    chat.stream_delta(text)


    def _handle_event(self, event: dict[str, Any]) -> None:
        kind = event.get("kind")
        session_id = event.get("session_id") or self.session.id
        chat = self._chat_for(session_id)
        status = self.query_one(StatusBar)
        if kind == "step":
            pass
        elif kind == "prompt_promoted":
            # the engine folded the oldest queued prompt into the running turn
            # (opencode's Session Drain) — drop its ` QUEUED ` badge, finalize
            # the previous reply's bubble, and let the next text stream in as a
            # fresh response to the promoted prompt.
            # the previous reply's text may still be sitting in the delta buffer
            # (deltas are batched on a 30ms flush timer) — empty it BEFORE
            # remove_last_stream_bubble decides whether the stream bubble is
            # "empty", otherwise the trailing bubble can be dropped while its
            # text is still un-rendered.
            self._flush_deltas()
            chat.promote_next_queued()
            chat.end_reasoning()
            chat.remove_last_stream_bubble()
            # the promoted prompt starts a fresh answer: speak the queued
            # tail first so no sentence is lost across the boundary.
            try:
                self._flush_voice(session_id)
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/events.py:179")
                pass
        elif kind == "retry":
            status.set_retry_message(event.get("message", "↻ retrying…"))
            self._on_retry_event(session_id, event)
        elif kind == "error":
            status.set_retry_message("")
            self._clear_task_retry(session_id)
            self._show_error(event.get("error", "unknown error"), retryable=bool(event.get("retryable")), session_id=session_id)
            # a failed turn has nothing worth saying: drop its voice tail
            try:
                self._clear_voice(session_id)
                self._stop_auto_voice()
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/events.py:192")
                pass
        elif kind == "text_delta":
            # an empty trailing chunk (a failed summary) is real output only
            # if it actually carries text — otherwise it would light up the
            # runtime line and spawn a blank assistant bubble
            text = event.get("text") or ""
            status.set_retry_message("")
            self._clear_task_retry(session_id)
            if text:
                self._turn_state(session_id)["had_text"] = True
            self._queue_delta(session_id, text, "text")
            # stream-voice: talk each finished sentence while still writing
            try:
                self._stream_voice(session_id, text)
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/events.py:207")
                pass
        elif kind == "reasoning_delta":
            text = event.get("text") or ""
            status.set_retry_message("")
            self._clear_task_retry(session_id)
            if text:
                self._turn_state(session_id)["had_reasoning"] = True
            self._queue_delta(session_id, text, "reasoning")
        elif kind == "tool_call":
            # the model is now responding/acting — drop any stale retry hint
            status.set_retry_message("")
            self._clear_task_retry(session_id)
            # render any buffered text first so the tool row lands below it
            self._flush_deltas()
            # speak the sentence tail before the tool row lands, then pause
            # voice while the tool runs (next text resumes it)
            try:
                self._flush_voice(session_id)
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/events.py:226")
                pass
            tool = event.get("tool", "?")
            # the current thought/assistant stream is over — finalize it so a
            # multi-step tool loop doesn't merge every step's text into one
            # bubble or leave a stale ▍ cursor
            chat.end_reasoning()
            chat.remove_last_stream_bubble()
            self._turn_state(session_id)["had_tools"] = True
            chat.append_tool(
                {
                    "tool": tool,
                    "status": "running",
                    "input": event.get("arguments", {}),
                    "call_id": event.get("call_id", ""),
                }
            )
            try:
                if session_id == self._current_session_id:
                    inp = event.get("arguments", {}) or {}
                    if not isinstance(inp, dict):
                        inp = {}
                    t = str(tool or "").lower()
                    if t in ("read", "glob", "grep", "lsp"):
                        fp = str(inp.get("filePath", "") or "")
                        msg = f"Reading {fp}..." if fp else "Reading..."
                    elif t in ("bash", "execute", "shell"):
                        cmd = str(inp.get("command", "") or "")[:60]
                        msg = f"Testing {cmd}..." if cmd else "Testing..."
                    elif t in ("edit", "write", "apply_patch"):
                        fp = str(inp.get("filePath", "") or "")
                        msg = f"Editing {fp}..." if fp else "Editing..."
                    elif t in ("webfetch", "webfetch_many"):
                        urls = inp.get("urls") or []
                        u = str(inp.get("url", "") or (urls[0] if isinstance(urls, list) and urls else ""))[:60]
                        msg = f"Fetching {u}..." if u else "Fetching..."
                    elif t == "webreach":
                        u = str(inp.get("url", "") or inp.get("site", "") or "")[:60]
                        msg = f"Reading {u}..." if u else "Checking site channels..."
                    elif t == "websearch":
                        q = str(inp.get("query", "") or "")[:60]
                        msg = f"Searching web {q}..." if q else "Searching web..."
                    elif t == "task":
                        msg = "Delegating..."
                    else:
                        msg = f"Running {tool}..."
                    self.query_one(InputBar).set_status(msg)
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/events.py:270")
                pass
        elif kind == "tool_start":
            tool_run = {
                "tool": event.get("tool", "?"),
                "status": "running",
                "input": event.get("input", {}),
                "call_id": event.get("call_id", ""),
            }
            if not chat.update_tool_bubble(tool_run):
                chat.append_tool(tool_run)
            try:
                if session_id == self._current_session_id:
                    inp2 = event.get("input", {}) or {}
                    if not isinstance(inp2, dict):
                        inp2 = {}
                    t = str(event.get("tool", "") or "").lower()
                    if t in ("read", "glob", "grep", "lsp"):
                        fp = str(inp2.get("filePath", "") or "")
                        msg = f"Reading {fp}..." if fp else "Reading..."
                    elif t in ("bash", "execute", "shell"):
                        cmd = str(inp2.get("command", "") or "")[:60]
                        msg = f"Testing {cmd}..." if cmd else "Testing..."
                    elif t in ("edit", "write", "apply_patch"):
                        fp = str(inp2.get("filePath", "") or "")
                        msg = f"Editing {fp}..." if fp else "Editing..."
                    elif t in ("webfetch", "webfetch_many"):
                        urls = inp2.get("urls") or []
                        u = str(inp2.get("url", "") or (urls[0] if isinstance(urls, list) and urls else ""))[:60]
                        msg = f"Fetching {u}..." if u else "Fetching..."
                    elif t == "webreach":
                        u = str(inp2.get("url", "") or inp2.get("site", "") or "")[:60]
                        msg = f"Reading {u}..." if u else "Checking site channels..."
                    elif t == "websearch":
                        q = str(inp2.get("query", "") or "")[:60]
                        msg = f"Searching web {q}..." if q else "Searching web..."
                    elif t == "task":
                        msg = "Delegating..."
                    else:
                        msg = f"Running {event.get('tool', '?')}..."
                    self.query_one(InputBar).set_status(msg)
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/events.py:308")
                pass
        elif kind == "tool_progress":
            try:
                chat2 = self._chat_for(session_id)
                bubble = chat2.find_tool(event.get("tool", ""), event.get("call_id", ""))
                if bubble is not None:
                    if event.get("done") is not None:
                        bubble.set_tool_metadata("done", int(event.get("done")))
                    if event.get("total") is not None:
                        bubble.set_tool_metadata("total", int(event.get("total")))
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/events.py:319")
                pass
        elif kind == "tool_complete":
            run = {
                "tool": event.get("tool", "?"),
                "status": "error" if event.get("status") == "error" else "completed",
                "input": event.get("input", {}),
                "output": event.get("output", ""),
                "metadata": event.get("metadata", {}),
                "call_id": event.get("call_id", ""),
            }
            chat.update_tool_bubble(run)
            try:
                if session_id == self._current_session_id:
                    self.query_one(InputBar).set_status("working...")
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/events.py:334")
                pass
            if event.get("tool") == "task":
                # enrich the task row with the child's runtime + toolcall count
                # (official's `↳ 3 toolcalls · 12.5s` completion detail).
                self._finalize_task_row(chat, run)
        elif kind == "tool_denied":
            run = {
                "tool": event.get("tool", "?"),
                "status": "error",
                "input": event.get("input", {}),
                "output": event.get("reason") or "permission denied",
                "call_id": event.get("call_id", ""),
            }
            if not chat.update_tool_bubble(run):
                # No preceding tool_call (e.g. a permission denial) — show the
                # denied row so the rejected action is visible.
                chat.append_tool(run)
        elif kind == "interrupted":
            self._flush_deltas()
            self._turn_state(session_id)["interrupted"] = True
            # a killed turn stops talking immediately, tail dropped
            try:
                self._clear_voice(session_id)
                self._stop_auto_voice()
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/events.py:359")
                pass
            chat.end_reasoning()
            chat.remove_last_stream_bubble()
            chat.end_stream()
            chat.append_meta("⏹ Interrupted")
        elif kind == "usage":
            self._usage[session_id] = event.get("usage") or {}
            if session_id == self._current_session_id:
                status.set_usage(event.get("usage") or {})
        elif kind == "compaction_start":
            # official opencode: show `Compacting conversation…` with a spinner
            # while the session summarizes to recover/avoid context overflow.
            # Only when the compacting session is the one on screen: a
            # sub-agent (task tool) compacting in the background must NOT flash
            # the main bar's spinner — its summary bubble goes to the sub-agent's
            # own (hidden) chat, so the result would otherwise never appear and
            # just look like the current conversation vanished.
            if session_id == self._current_session_id:
                try:
                    self.query_one(InputBar).set_compacting(True)
                except Exception as _oc_exc:
                    swallow(_oc_exc, "tui/events.py:380")
                    pass
            # Render any buffered text/reasoning deltas FIRST (same invariant as
            # tool_call/interrupted/prompt_promoted): begin_compaction_stream
            # finalizes the Thought + stream bubbles, so skipping the flush would
            # freeze them mid-sentence (or drop an empty one) and re-emit the
            # buffered tail as a stray bubble below the compaction divider.
            self._flush_deltas()
            # Kick off the live `▸ Compacted summary` bubble (hidden chats too —
            # they get the final divider on switch).
            chat.begin_compaction_stream()
        elif kind == "summary_delta":
            # Stream the anchored summary into the compaction bubble live so the
            # user watches it being written rather than waiting on a spinner.
            text = event.get("text") or ""
            if text:
                chat.stream_compaction_delta(text)
        elif kind == "compacted":
            # opencode renders a ` Session compacted ` divider when the session
            # summarizes to recover/avoid context overflow. Mirror that: flush
            # pending deltas, end the current reasoning bubble, then finalize
            # the live summary bubble into the divider.
            self._flush_deltas()
            if session_id == self._current_session_id:
                try:
                    self.query_one(InputBar).set_compacting(False)
                except Exception as _oc_exc:
                    swallow(_oc_exc, "tui/events.py:406")
                    pass
            chat.end_compaction_stream(event.get("summary") or "")
            self._turn_state(session_id)["had_tools"] = True
        elif kind == "rotated":
            # Show the failover popup ONLY when rotation is unlocked. When the
            # selected model is locked the lane can't change, so a "model
            # changed" toast would be noise (or a lie) — suppress it.
            eng = self._engines.get(session_id) or self.engine
            provider = event.get("provider", "?")
            model = event.get("model", "?")
            # reflect the lane that actually answered in the mode line under
            # the input box (deepseek -> nemotron) as soon as it switches.
            eng.provider_id = provider
            eng.model_id = model
            self._update_header()
            if not getattr(eng, "rotation_locked", False):
                reason = event.get("reason") or "provider error"
                self.notify(
                    f"Now using {model} · {provider}\n{reason}",
                    title="Model changed",
                )
        elif kind == "subagent_start":
            self._on_subagent_start(event)
        elif kind == "subagent_done":
            self._on_subagent_done(event)
        elif kind == "auto_continue":
            # The engine nudged the model because it stopped mid-job. The web
            # client shows "Continuing automatically…"; without this the terminal
            # only showed the raw nudge text appearing in chat, which looks like
            # the user had typed it.
            try:
                self.notify("Continuing automatically…", timeout=3, markup=False)
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/events.py:auto_continue")
        elif kind == "bg_collected":
            self._on_bg_collected(event)
        elif kind == "notice":
            # small engine notices (e.g. fully_auto auto-answered a
            # question): one toast, no popup, nothing blocks.
            try:
                text = str(event.get("text") or "").strip()
            except Exception:
                text = ""
            if text:
                self.notify(text, timeout=5, markup=False)


    def _on_subagent_start(self, event: dict[str, Any]) -> None:
        from ..session import load_session

        sid = event.get("session_id") or ""
        if not sid:
            return
        if sid not in self._sessions:
            sess = load_session(sid)
            if sess is not None:
                self._sessions[sid] = sess
            else:
                # register a placeholder so the fallbacks
                # (`_sessions.get(sid) or self.session`) never route a sub-agent
                # turn's history onto the main session file. The placeholder
                # MUST carry the parent link + title now (not blank): a blank
                # placeholder saved at exit used to mint orphan files with
                # parent_id None that showed in /sessions like real parents.
                from ..session import Session
                parent_hint = self._active_turn_session_id or self._current_session_id
                self._sessions[sid] = Session(
                    {"id": sid,
                     "title": event.get("title") or "sub-agent",
                     "agent": event.get("agent") or "build",
                     "parent_id": parent_hint,
                     "provider": self.cfg.provider,
                     "model": self.cfg.model},
                    directory=str(self.directory))
        sub = self.engine.find_subagent(sid)
        if sub is not None:
            self._engines[sid] = sub
        self._busy_sessions.add(sid)
        chat = self._chat_for(sid)
        # Show the parent's instruction at the very top of the sub-agent's
        # chat, exactly like official opencode renders the task directive as
        # the session's first message.
        prompt = event.get("prompt") or ""
        if prompt and not chat.children:
            chat.append_directive(prompt, title=event.get("title") or "", agent=event.get("agent") or "")
        self._chats[sid] = chat
        title = event.get("title") or "sub-agent"
        agent = event.get("agent") or "build"
        sess = self._sessions.get(sid)
        parent_id = getattr(sess, "parent_id", None) or self._active_turn_session_id
        if parent_id:
            # register the child in the parent's family tree (numbered siblings
            # for the footer's `(2 of N)`), and attach its id to the parent
            # chat's running task row so it renders live state / is clickable.
            self._child_parent[sid] = parent_id
            records = self._children.setdefault(parent_id, [])
            if not any(c.get("id") == sid for c in records):
                records.append(
                    {
                        "id": sid,
                        "title": title,
                        "agent": agent,
                        "created": time.time(),
                        "status": "running",
                    }
                )
            self._link_task_row(sid, parent_id, call_id=event.get("call_id") or "")
        self._running_agents[sid] = f"{title} · {agent}"
        self._refresh_running_agents()
        self._update_footer()
        self.notify(f"Sub-agent started: {title}", markup=False)


    def _on_subagent_done(self, event: dict[str, Any]) -> None:
        sid = event.get("session_id") or ""
        sess = self._sessions.get(sid)
        if sess is not None:
            sess.completed = time.time()
            # Adopt the spawn-time identity (parent/title/provider/model):
            # placeholders created at subagent_start may predate the engine
            # save, and without this the file keeps blank identity fields.
            # Parent is looked up through EVERY source (live map → engine
            # session_id chain → active turn): a child saved without it
            # becomes a stray parent-looking row in /sessions, forever.
            try:
                if not getattr(sess, "parent_id", None):
                    sess.parent_id = (
                        self._child_parent.get(sid)
                        or self._active_turn_session_id
                        or self._current_session_id
                        or getattr(sess, "parent_id", None))
                if not getattr(sess, "title", ""):
                    sess.title = event.get("title") or "sub-agent"
                if not getattr(sess, "provider", ""):
                    sess.provider = self.cfg.provider
                if not getattr(sess, "model", ""):
                    sess.model = self.cfg.model
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/events.py:535")
                pass
            # Keep the in-memory session's messages in sync with what the
            # sub-agent engine actually produced (spawn_task saves to disk, but
            # the app-side object was last loaded with the empty placeholder).
            sub = self._engines.get(sid)
            if sub is not None:
                try:
                    sess.messages = sub.get_history()
                except Exception as _oc_exc:
                    swallow(_oc_exc, "tui/events.py:544")
                    pass
        self._busy_sessions.discard(sid)
        chat = self._chats.get(sid)
        if chat is not None:
            chat.end_reasoning()
            # drop an empty streaming cursor if the sub-agent replied with no text
            chat.remove_last_stream_bubble()
            chat.end_stream()
        title = event.get("title") or "sub-agent"
        ok = event.get("ok", True)
        self._running_agents.pop(sid, None)
        self._refresh_running_agents()
        # Keep the finished sub-agent session registered (like the official
        # store) so its task row stays clickable, its chat stays reviewable,
        # and the parent's `(2 of N)` footer count keeps them in the total.
        record = None
        for r in self._children.get(self._child_parent.get(sid, ""), []):
            if r.get("id") == sid:
                record = r
                break
        if record is not None:
            record["status"] = "completed" if ok else "failed"
            record["completed"] = time.time()
        # Clean up in-memory dicts for completed sub-agent to prevent memory leak
        self._engines.pop(sid, None)
        self._turn.pop(sid, None)
        # Background agents that nobody collected yet keep ticking: the
        # parent row still shows "running" until task_read. Popping the
        # start time here froze the live `↳ Xs` line at spawn. Instead
        # stamp the final duration + a bg_done flag on the PARENT row and
        # let it show "done, unread" (Fix C renders it). The timer skips
        # flagged rows. Detection uses the event's was_background flag
        # (set at worker-completion time) — NOT the live entry map, which
        # the immediate fold may already have popped (race that wrongly
        # tore down live rows).
        if event.get("was_background"):
            try:
                from .input_bar import format_duration as _fd
            except Exception:
                _fd = None
            start = self._task_start.get(sid)
            if start is not None and _fd is not None:
                try:
                    dur = _fd(time.monotonic() - start)
                except Exception:
                    dur = ""
            else:
                dur = str(event.get("duration") or "")
            try:
                parent_id = self._child_parent.get(sid, "")
                pchat = self._chats.get(parent_id) if parent_id else None
                bubble = pchat.find_task(sid) if pchat is not None else None
                if bubble is not None:
                    bubble.pop_tool_metadata("elapsed")
                    if dur:
                        bubble.set_tool_metadata("duration", dur)
                    tc = event.get("toolcalls")
                    if isinstance(tc, int) and tc >= 0:
                        bubble.set_tool_metadata("toolcalls", tc)
                    bubble.set_tool_metadata("bg_done", True)
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/events.py:605")
                pass
        else:
            self._task_start.pop(sid, None)
        self._children.pop(sid, None)
        # NOTE: _child_parent[sid] is intentionally KEPT (not popped): it is
        # the only map that answers "who is my parent" for ←/→/↑ after the
        # child finishes. Dropping it killed all arrow navigation the moment
        # a sub-agent completed. Memory cost is one dict entry per child.
        self._update_footer()
        if ok:
            self.notify(f"Sub-agent done: {title}", markup=False)
        else:
            self.notify(f"Sub-agent failed: {title}", severity="error", markup=False)


    def _on_bg_collected(self, event: dict[str, Any]) -> None:
        """A detached reply was collected (task_read or fold): flip its
        parent row from the "done, unread" state to the normal completed
        rendering — clear bg_done, drop the timer, keep duration/toolcalls."""
        sid = event.get("session_id") or ""
        if not sid:
            return
        try:
            parent_id = self._child_parent.get(sid, "")
            pchat = self._chats.get(parent_id) if parent_id else None
            bubble = pchat.find_task(sid) if pchat is not None else None
            if bubble is not None:
                bubble.pop_tool_metadata("bg_done")
                bubble.pop_tool_metadata("elapsed")
                try:
                    if isinstance(bubble.content, dict):
                        # status flip re-renders via the completed branch
                        # (↳ N toolcalls · dur) and stops the spinner
                        bubble.update_tool({**bubble.content, "status": "completed"})
                except Exception as _oc_exc:
                    swallow(_oc_exc, "tui/events.py:640")
                    pass
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/events.py:642")
            pass
        self._task_start.pop(sid, None)


    # -- task-row enrichment (official Subagent completion detail) --------
    def _schedule_relink(self, parent_id: str, attempts: int = 40) -> None:
        """Retry _relink_task_rows until every task row is linked AND backfilled.

        _render_history mounts bubbles in 15-row chunks via call_later, so a
        single relink right after render finds zero rows on long sessions
        (your 26-message parent renders its task rows seconds later). Each
        pass links what's mounted; passes continue while rows are missing
        links OR missing durations (backfill lands on a later pass than the
        link), as long as the user is still viewing this parent.
        """
        try:
            missing = self._relink_task_rows(parent_id)
        except Exception:
            missing = 1
        try:
            chat = self._chats.get(parent_id)
            unfilled = 0
            if chat is not None:
                for b in chat.query("*"):
                    try:
                        if getattr(b, "role", "") != "tool":
                            continue
                        c = getattr(b, "content", None)
                        if not isinstance(c, dict) or c.get("tool") != "task":
                            continue
                        meta = c.get("metadata") or {}
                        if not meta.get("sessionId") or not meta.get("duration"):
                            unfilled += 1
                    except Exception as _oc_exc:
                        swallow(_oc_exc, "tui/events.py:676")
                        continue
            pending = max(missing, unfilled)
        except Exception:
            pending = missing
        if pending > 0 and attempts > 0:
            try:
                # NOTE: no current-session gate here — the resume path calls
                # this BEFORE _switch_session, so _current_session_id still
                # points at the previous session on the first passes. Gating
                # on it killed the whole retry chain (rows mounted later never
                # got linked/backfilled). Retries are cheap and idempotent.
                self.call_later(lambda: self._schedule_relink(parent_id, attempts - 1))
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/events.py:689")
                pass


    def _relink_task_rows(self, parent_id: str) -> int:
        """Re-attach child session ids onto a resumed parent's task rows.

        Covers ALL subagents — old saves (transcript never stored the link)
        and new ones alike. Matching order: saved transcript link first,
        then live family records by title, then disk children oldest-first
        against still-unlinked task rows. Returns the count of still-unlinked
        task rows (so the scheduler knows whether to retry). Never raises.
        """
        chat = self._chats.get(parent_id)
        if chat is None:
            return 0
        try:
            from ..session import load_session as _load
            parent = self._sessions.get(parent_id) or _load(parent_id)
        except Exception:
            parent = None
        if parent is None:
            return 0
        try:
            messages = getattr(parent, "messages", None) or []
        except Exception:
            messages = []
        # assistant-level spawn stamps (interrupt-proof link source)
        stamped: dict[str, dict] = {}
        try:
            for m in messages:
                if isinstance(m, dict) and isinstance(m.get("task_children"), dict):
                    for k, v in m["task_children"].items():
                        stamped.setdefault(str(k), (v or {}))
        except Exception:
            stamped = {}
        # 1. transcript links (new saves): call_id -> child session id
        linked: dict[str, str] = {}
        order: list[str] = []
        for m in messages:
            try:
                if not isinstance(m, dict) or m.get("role") != "tool" or m.get("name") != "task":
                    continue
                sid = m.get("session_id") or m.get("sessionId")
                if sid and str(sid) not in order:
                    order.append(str(sid))
                cid = m.get("tool_call_id")
                if cid and sid:
                    linked[str(cid)] = str(sid)
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/events.py:738")
                continue
        # 2. live family records (this run's spawns, incl. title for matching)
        try:
            records = list(self._sibling_records(parent_id))
        except Exception:
            records = []
        # 3. disk children (old saves, restarts): oldest first
        try:
            kids = [dict(c) for c in (self._children_of(parent_id) or [])]
        except Exception:
            kids = []
        try:
            from ..session import load_session as _load2
            for k in kids:
                ksess = self._sessions.get(k["id"]) or _load2(k["id"])
                if ksess is not None and not k.get("title"):
                    k["title"] = getattr(ksess, "title", "") or k.get("title", "")
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/events.py:756")
            pass
        try:
            bubbles = [b for b in chat.query("*")
                       if getattr(b, "role", "") == "tool"
                       and isinstance(getattr(b, "content", None), dict)
                       and b.content.get("tool") == "task"]
        except Exception:
            return 0
        if not bubbles:
            return 1  # rows not mounted yet — scheduler retries
        used: set[str] = set()
        missing = 0

        def _backfill(bubble: Any, target: str) -> None:
            # Runtime + toolcall count from the child's saved file. Runs for
            # EVERY linked row (not just newly linked ones): resumed parents
            # never saw the live finalize event, so their rows would otherwise
            # show no `↳ N toolcalls · Ns` even with sessionId present.
            try:
                meta = bubble.content.get("metadata") or {}
                if meta.get("duration") and meta.get("toolcalls") is not None:
                    return
                from ..session import load_session as _load3
                ksess = self._sessions.get(target) or _load3(target)
                if ksess is None:
                    return
                created = float(getattr(ksess, "created", 0) or 0)
                completed = float(getattr(ksess, "completed", 0) or 0)
                if completed > created and not meta.get("duration"):
                    bubble.set_tool_metadata("duration", format_duration(completed - created))
                if meta.get("toolcalls") is None:
                    try:
                        n_tools = sum(
                            1 for m in (getattr(ksess, "messages", None) or [])
                            if isinstance(m, dict) and m.get("role") == "tool")
                        bubble.set_tool_metadata("toolcalls", n_tools)
                    except Exception as _oc_exc:
                        swallow(_oc_exc, "tui/events.py:793")
                        pass
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/events.py:795")
                pass

        for b in bubbles:
            try:
                meta = b.content.get("metadata") or {}
                if meta.get("sessionId"):
                    used.add(str(meta.get("sessionId")))
                    _backfill(b, str(meta.get("sessionId")))
                    continue
                bid = b.content.get("id") or b.content.get("call_id") or ""
                target = linked.get(str(bid)) if bid else None
                if target is None or target in used:
                    # title match against family records (description match)
                    inp = b.content.get("input") or {}
                    desc = str(inp.get("description") or meta.get("title") or "").strip().lower()
                    for r in records + kids:
                        rid = str(r.get("id") or "")
                        if not rid or rid in used:
                            continue
                        rt = str(r.get("title") or "").strip().lower()
                        if desc and rt and (desc == rt or desc in rt or rt in desc):
                            target = rid
                            break
                    if target is None:
                        # spawn stamps first (survive interrupts), then
                        # transcript order, then family records.
                        for rid in list(stamped) + order + [str(r.get("id")) for r in records + kids]:
                            if rid and rid not in used:
                                # prefer title agreement when a stamp exists
                                if rid in stamped and desc:
                                    rt = str((stamped[rid] or {}).get("title") or "").strip().lower()
                                    if rt and not (desc == rt or desc in rt or rt in desc):
                                        continue
                                target = rid
                                break
                if target:
                    b.set_tool_metadata("sessionId", target)
                    used.add(str(target))
                    _backfill(b, str(target))
                else:
                    missing += 1
            except Exception:
                missing += 1
                continue
        return missing


    def _link_task_row(self, sid: str, parent_id: str, call_id: str = "") -> None:
        """Attach a sub-agent's session id + start time to the exact task row
        that spawned it (the parent chat's row with the matching tool call id).

        With parallel sub-agents the events arrive concurrently, so "newest
        unattached running row" is wrong under the hood — the call id makes the
        mapping unambiguous (official opencode keys every sub-agent by its task
        call). A missing id keeps the old fallback for compatibility."""
        chat = self._chats.get(parent_id)
        if chat is None:
            return
        if call_id:
            bubble = chat.find_tool("task", call_id)
            if bubble is not None and bubble.content.get("status") == "running" and not (bubble.content.get("metadata") or {}).get("sessionId"):
                bubble.set_tool_metadata("sessionId", sid)
                self._task_start[sid] = time.monotonic()
                return
        for child in reversed(list(chat.query(MessageBubble))):
            if child.role != "tool" or child.content.get("tool") != "task":
                continue
            meta = child.content.get("metadata") or {}
            if child.content.get("status") == "running" and not meta.get("sessionId"):
                child.set_tool_metadata("sessionId", sid)
                self._task_start[sid] = time.monotonic()
                break


    def _finalize_task_row(self, chat: ChatView, run: dict[str, Any]) -> None:
        """Write the completed task row's runtime + toolcall count (the
        `↳ 3 toolcalls · 12.5s` line under a finished sub-agent).

        Prefers the engine-stamped truth in the task result metadata
        (`duration_s` + `toolcalls` from spawn_task): race-free and
        resume-proof. Falls back to the live start-time map + child chat
        widget for old sessions/results that predate the stamp.

        SKIPPED for background-agent spawn completes: that tool call only
        STARTED the worker (metadata.background=True) — finalizing here
        would steal the start time and paint a 0s completion while the
        agent is still running. The real finalize lands on task_read
        collect (or the bg-done update)."""
        meta = run.get("metadata") or {}
        sid = meta.get("sessionId")
        if not sid:
            return
        if meta.get("background"):
            # spawn acknowledgement, not a completion — leave the row live.
            return
        bubble = chat.find_task(sid)
        if bubble is None:
            return
        bubble.pop_tool_metadata("elapsed")
        stamped_s = meta.get("duration_s")
        stamped_n = meta.get("toolcalls")
        if isinstance(stamped_s, (int, float)) and stamped_s >= 0:
            bubble.set_tool_metadata("duration", format_duration(float(stamped_s)))
        else:
            start = self._task_start.pop(sid, None)
            if start is not None:
                bubble.set_tool_metadata("duration", format_duration(time.monotonic() - start))
        if isinstance(stamped_n, int) and stamped_n >= 0:
            bubble.set_tool_metadata("toolcalls", stamped_n)
        else:
            child = self._chats.get(sid)
            toolcalls = 0
            if child is not None:
                toolcalls = len([r for r in child.tool_runs() if r.get("status") == "completed"])
            bubble.set_tool_metadata("toolcalls", toolcalls)


    def _refresh_task_elapsed(self) -> None:
        """Tick live elapsed time onto every running sub-agent row.

        Fires every 1s on its own timer while the app lives (started with the
        first turn): each unfinished task row gets `elapsed` metadata
        (`12.5s`) rendered as `↳ 12.5s` under the row. Cleared on finalize
        so the completed `↳ N toolcalls · Xs` line takes over. Cheap metadata
        writes on the UI thread; never raises.

        Rows flagged bg_done (background worker finished, reply uncollected)
        are skipped: their final duration is already stamped (Fix B) and
        re-ticking would resurrect the live line over the "done" state."""
        if not self._task_start:
            return
        from .input_bar import format_duration

        now = time.monotonic()
        for chat in list(self._chats.values()):
            try:
                bubbles = list(chat.query(MessageBubble))
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/events.py:933")
                continue
            for bubble in bubbles:
                try:
                    if bubble.role != "tool":
                        continue
                    content = bubble.content
                    if not isinstance(content, dict) or content.get("tool") != "task":
                        continue
                    if content.get("status") != "running":
                        continue
                    meta = content.get("metadata") or {}
                    if meta.get("bg_done"):
                        continue
                    sid = meta.get("sessionId")
                    start = self._task_start.get(sid) if sid else None
                    if start is None:
                        continue
                    bubble.set_tool_metadata("elapsed", format_duration(now - start))
                except Exception as _oc_exc:
                    swallow(_oc_exc, "tui/events.py:952")
                    continue


    def _on_retry_event(self, session_id: str, event: dict[str, Any]) -> None:
        """A sub-agent's provider lane is retrying — paint its parent task row
        error-red with `↳ Retrying (attempt N) · …` (official Subagent retry)."""
        parent_id = self._child_parent.get(session_id)
        if not parent_id:
            return
        chat = self._chats.get(parent_id)
        if chat is None:
            return
        bubble = chat.find_task(session_id)
        if bubble is not None:
            bubble.set_tool_metadata(
                "retry",
                {"attempt": event.get("attempt", 1), "message": event.get("message", "")},
            )


    def _clear_task_retry(self, session_id: str) -> None:
        """The sub-agent made progress again — remove the retry decoration."""
        parent_id = self._child_parent.get(session_id)
        if not parent_id:
            return
        chat = self._chats.get(parent_id)
        if chat is None:
            return
        bubble = chat.find_task(session_id)
        if bubble is not None:
            bubble.pop_tool_metadata("retry")

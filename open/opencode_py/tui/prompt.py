"""Prompt submission, turn lifecycle and autosave for the Textual TUI.

Split out of :mod:`opencode_py.tui.app` as a mixin; method bodies are unchanged.
"""
from __future__ import annotations

from ..util.log import swallow
import copy, threading, time
from typing import Any, TYPE_CHECKING
from .. import errors
from .input_bar import InputBar, PromptSubmitted
from .status_bar import StatusBar

if TYPE_CHECKING:
    from ..agent.loop import AgentLoop

class PromptMixin:

    # -- prompt handling -------------------------------------------------
    def on_prompt_submitted(self, event: PromptSubmitted) -> None:
        value = event.value
        if not value.strip():
            return
        sid = self._current_session_id
        chat = self._chat_for(sid)
        engine = self._engines.get(sid) or self.engine
        session = self._sessions.get(sid) or self.session
        # always show what the user typed, then route it
        bubble = chat.append_user(value, agent=engine.agent)
        if value.lstrip().startswith("/"):
            self._run_command(value.lstrip())
            return
        # name the session from its first real message (opencode behaviour)
        if not getattr(session, "title", ""):
            session.title = value.strip()[:60]
        if self._busy:
            if sid != self._active_turn_session_id:
                # The running turn belongs to a DIFFERENT session (user resumed
                # an idle chat mid-stream). Queueing here parked the prompt in
                # an engine with no running turn — stuck "Queued" forever.
                chat.append_meta(
                    "⏳ Another session's request is still running — switch back "
                    "to it, or Ctrl+C to interrupt before sending here."
                )
                self.notify("Busy in another session", severity="warning")
                return
            # opencode's queue-and-promote: never drop a prompt typed while the
            # agent is busy. It goes into the ENGINE's FIFO (thread-safe, one
            # per session) and its bubble shows the ` QUEUED ` badge. The next
            # provider-turn boundary of the SAME running turn folds it in
            # (run_turn drains the queue), so there is no fresh-turn gap —
            # exactly how opencode keeps one Session Drain going.
            #
            # EXCEPTION: the parent is just parked in task_read, waiting on
            # a background agent while doing nothing itself. Then the new
            # message breaks the wait (wake_bg_wait): the old drain ends
            # after its current step, and the message starts its own turn
            # from _turn_done — promptly, with NO QUEUED badge. The bg
            # agent keeps running; its reply folds in at a later turn end.
            woken = False
            try:
                woken = bool(engine.wake_bg_wait())
            except Exception as exc:
                errors.warn("tui.prompt", "wake_bg_wait failed; treating as not woken", exc)
                woken = False
            if woken:
                bubble.queued = False
                try:
                    chat._queued_bubbles = [
                        b for b in chat._queued_bubbles if b is not bubble
                    ]
                except Exception as exc:
                    errors.warn("tui.prompt", "could not clear the QUEUED badge", exc)
                # stash on the ENGINE (not the turn slot: _turn_done pops
                # the slot in its finally before reading it)
                try:
                    engine._pending_fresh = value
                except Exception as exc:
                    errors.warn(
                        "tui.prompt",
                        "could not stash the pending reply; it will not be attached",
                        exc,
                    )
                self.notify("Reading your message now — background agent keeps running")
                return
            bubble.queued = True
            depth = engine.queue_prompt(value)
            self.notify(f"Queued ({depth}) — will run in the current turn")
            return
        self._start_turn(sid, value, engine)


    def _autosave_in_flight(self) -> None:
        """Periodic crash-safety save of the running turn's session.

        The engine keeps the in-flight assistant reply live in get_history()
        as the stream grows, so this captures the very last conversation. Runs
        on the app thread via a timer; a sudden process kill can only lose the
        tokens streamed since the previous tick.

        The durable write (temp file + fsync + atomic rename + .bak replica +
        index) can take 10-25ms on this phone's flash, so it is done on a
        background thread — never on the UI thread where it would hitch the
        render every 2s mid-stream. The thread never mutates the live Session
        (snapshot copy) and its write is guarded by an autosave generation so
        it can't race a newer turn/exit save.
        """
        if not self._busy:
            return
        if self._autosave_thread is not None and self._autosave_thread.is_alive():
            return  # a slower disk can't pile up saves
        try:
            sid = self._active_turn_session_id
            engine = self._engines.get(sid)
            sess = self._sessions.get(sid)
            if engine is not None and sess is not None:
                history = engine.get_history()
                # Only overwrite with a real conversation: the worker may not
                # have appended the prompt yet when the first tick fires, and
                # saving the empty history would destroy the durable copy that
                # _start_turn just wrote.
                if not history:
                    return
                generation = self._autosave_generation
                self._autosave_thread = threading.Thread(
                    target=self._autosave_write,
                    args=(sid, history, generation),
                    daemon=True,
                )
                self._autosave_thread.start()
        except Exception as exc:
            # Autosave is what protects the user's conversation. If it cannot
            # start, the session is not being written to disk at all.
            errors.error(
                "tui.prompt",
                f"autosave could not start for session {sid!r}; this turn is NOT saved",
                exc,
            )


    def _autosave_write(self, sid: str, history: list[dict[str, Any]], generation: int) -> None:
        """Worker thread body: persist a snapshot WITHOUT mutating the live
        Session shared with the UI thread. Skipped via `should_write` if the
        autosave generation moved on (turn ended / exit save-all started)."""
        try:
            from ..session import save_session

            sess = self._sessions.get(sid)
            if sess is None:
                return
            snapshot = copy.copy(sess)
            snapshot.messages = list(history)
            # Double-check generation at write time (not just at start) to avoid
            # a stale worker overwriting a newer durable copy mid-write.
            if self._autosave_generation != generation:
                return
            save_session(
                snapshot,
                should_write=lambda: self._autosave_generation == generation,
            )
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/prompt.py:160")
            pass


    def _cancel_autosave(self) -> None:
        self._autosave_generation += 1  # invalidate any in-flight streaming autosave
        if self._autosave_timer is not None:
            try:
                self._autosave_timer.stop()
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/prompt.py:169")
                pass
            self._autosave_timer = None


    def _turn_state(self, sid: str) -> dict[str, Any]:
        """The per-session turn bookkeeping slot (created on first touch)."""
        st = self._turn.get(sid)
        if st is None:
            st = {
                "had_text": False,
                "had_reasoning": False,
                "had_error": False,
                "had_tools": False,
                "interrupted": False,
                "started": None,
            }
            self._turn[sid] = st
        return st


    def _start_turn(self, sid: str, value: str, engine: AgentLoop, resume: bool = False) -> None:
        """Start a model turn in a worker thread for the initial prompt.

        Prompts submitted while this turn runs are queued on the engine and
        folded into the SAME turn (one Session Drain) at the next provider-turn
        boundary; this is only the drain's starting point.

        With ``resume=True`` the engine re-runs its last user prompt instead
        (auto-resume after a reconnect) — nothing is appended or duplicated.
        Any reconnect watcher for this session is stopped: a live turn (or a
        fresh user prompt) supersedes waiting.
        """
        self._stop_reconnect_watch(sid)
        # a new turn supersedes the old reply: stop any auto-voice still
        # talking so replies never overlap, and drop its queued sentences
        # (the generation bump makes anything queued stale).
        try:
            from ..tools.speak import stop_speech
            stop_speech()
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/prompt.py:209")
            pass
        try:
            self._clear_voice(sid)
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/prompt.py:213")
            pass
        chat = self._chat_for(sid)
        # show an eager Thinking… bubble immediately (before the first token
        # arrives) so the UI reacts to Enter instead of sitting silent
        chat.begin_thinking()
        st = self._turn_state(sid)
        st.update(
            had_text=False,
            had_reasoning=False,
            had_error=False,
            had_tools=False,
            interrupted=False,
        )
        st["started"] = time.monotonic()
        # the previous turn's runtime disappears the moment the model starts
        # working again (official opencode shows it only on the final report)
        self._clear_last_duration()
        self._busy = True
        self._busy_sessions.add(sid)
        self._active_turn_session_id = sid
        # Streaming indicators follow the VIEWED session (not a global flag):
        # if the user is watching some other idle chat, it must not inherit
        # this turn's spinner/locked input.
        self._sync_streaming_visuals()
        # Persistence policy: starting a turn NEVER writes a file. A session
        # is only durably stored once one of the save conditions fires —
        # (1) the 2s crash-safety autosave tick while a real conversation
        # streams (phone dies / reboot mid-turn -> at most the last 2 seconds
        # are lost), (2) the final save-all on exit (ctrl+q / /exit), or (3) the
        # SIGTERM/SIGHUP handler when Termux is closed. Merely starting a turn
        # must never mint an empty or half-built session file.
        if self._autosave_timer is None:
            self._autosave_timer = self.set_interval(2.0, self._autosave_in_flight)
        # Live `↳ Xs` elapsed behind running sub-agent rows ticks every 1s
        # on its own timer (separate from the 2s disk autosave above: display
        # stays snappy without doubling flash writes on old phones).
        if self._elapsed_timer is None:
            try:
                self._elapsed_timer = self.set_interval(1.0, self._refresh_task_elapsed)
            except Exception:
                self._elapsed_timer = None

        def run():
            result = None
            try:
                result = engine.resume_turn() if resume else engine.run_turn(value)
            except Exception as e:  # never let a worker crash silently
                self.call_from_thread(self._show_error, f"{type(e).__name__}: {e}", False, sid)
            finally:
                # Pass THIS worker's sid: the global _active_turn_session_id can
                # already point at a newer turn (another session's prompt or a
                # promoted queued prompt), and finalizing the wrong session's
                # chat would clear the wrong flags and promote the wrong queue.
                self.call_from_thread(self._turn_done, result, sid)

        self.run_worker(run, thread=True)


    def _show_error(self, message: str, retryable: bool = False, session_id: str | None = None) -> None:
        sid = session_id or self.session.id
        self._turn_state(sid)["had_error"] = True
        chat = self._chat_for(sid)
        self._flush_deltas()
        chat.end_reasoning()
        chat.remove_last_stream_bubble()
        chat.append_meta(f"⚠ {message}")
        hint = " Retry, or check /connect for a model/API key." if retryable else ""
        self.notify(f"error: {message}{hint}", severity="error")
        chat.end_stream()
        # Reset the visual streaming state here too, not only in _turn_done. The
        # worker's `finally` normally calls _turn_done and clears these, but if
        # that call_from_thread ever fails (e.g. app unmount mid-error) the UI
        # must not stay stuck on an "streaming" indicator on the error path.
        self._streaming_visual_reset()


    def _clear_last_duration(self) -> None:
        """Hide the previous turn's runtime on the mode line.

        Official opencode shows the runtime (`▣ Build · model · 1m 12s`) only
        while the final report is displayed; it disappears as soon as the model
        starts doing things again (a new turn begins working / running tools).
        """
        try:
            self.query_one(InputBar).set_last_duration("")
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/prompt.py:299")
            pass


    def _streaming_visual_reset(self) -> None:
        """Defensively clear the streaming-indicator UI state (status bar +
        input bar). Does NOT touch _busy/_busy_sessions: those belong to the
        worker thread and are owned by _turn_done's finally, so resetting them
        here could race an active turn on another session. Indicators follow
        the VIEWED session, not a global flag."""
        self._sync_streaming_visuals()


    def _turn_done(self, result: Any = None, sid: str | None = None) -> None:
        if sid is not None:
            self._interrupt_flags[sid] = False
        self._cancel_autosave()
        # Never trust the global _active_turn_session_id here: it can already
        # point at a NEWER turn (another session's prompt, or the promoted
        # queued prompt of a just-finished drain). This worker finalizes the
        # session it actually ran — passed in by _start_turn's run().
        sid = sid or self._active_turn_session_id
        st = self._turn_state(sid)
        engine = self._engines.get(sid) or self.engine
        session = self._sessions.get(sid) or self.session
        interrupted = False
        turn_failed = False
        promote_next = False
        try:
            if result is not None and result.provider_id:
                # reflect the lane/model that actually answered (e.g. openrouter)
                engine.provider_id = result.provider_id
                engine.model_id = result.model_id or engine.model_id
                # count the answered turn toward the most-used default
                try:
                    from ..config import record_model_use
                    record_model_use(result.provider_id, engine.model_id)
                except Exception as _oc_exc:
                    swallow(_oc_exc, "tui/prompt.py:336")
                    pass
                # rebuild_rotation() hits the network (model catalogs) — it
                # must NEVER run here on the UI thread (it froze the whole
                # screen at end of turn whenever the cache was stale). Flag
                # it; run_turn rebuilds on the engine thread before streaming.
                engine.mark_rotation_dirty()
                self._update_header()
            chat = self._chat_for(sid)
            status = self.query_one(StatusBar)
            status.set_retry_message("")
            self._flush_deltas()
            chat.end_reasoning()
            if not st["had_text"] and not st["had_reasoning"] and not st["had_error"] and not st["had_tools"] and not st["interrupted"]:
                # provider returned nothing (no text, no reasoning, no tool call,
                # no error) — drop the empty streaming cursor bubble before
                # end_stream clears its pointer.
                chat.remove_last_stream_bubble()
                chat.append_meta(
                    "(no reply from the model — check your connection and /connect "
                    "for a working model, or switch rotation in /config)"
                )
                self.notify("No reply from the model.", severity="warning")
            chat.end_stream()
            if st["had_text"] and not st["had_error"]:
                # the mode line lives fixed above the prompt box now, not in the chat
                self._update_header()
            # show the finished turn's runtime (`▣ Build · model · 1m 12s`) like
            # opencode's per-message footer, but ONLY on the final report. Official
            # opencode computes the duration when the message finished with a real
            # text answer (`finish` not tool-calls), so a tool-only, errored, or
            # interrupted turn shows no runtime — it appears again on the next
            # turn's last report.
            started = st["started"]
            st["started"] = None
            if started is not None:
                elapsed = time.monotonic() - started
                if st["had_text"] and not st["had_error"] and not st["interrupted"]:
                    try:
                        from .input_bar import format_duration

                        self.query_one(InputBar).set_last_duration(format_duration(elapsed))
                    except Exception as _oc_exc:
                        swallow(_oc_exc, "tui/prompt.py:378")
                        pass
            interrupted = st["interrupted"]
            turn_failed = bool(getattr(result, "error", "")) if result is not None else False
            # opencode's queue-and-promote fallback: normally the ENGINE folds
            # queued prompts into the running turn (one Session Drain) — but a
            # prompt submitted in the tiny window after the turn's last boundary
            # but before it returns sits in the engine queue. Promote it as the
            # start of the next drain so nothing is ever left stuck "queued".
            # An interrupted turn stops the drain for real: the remaining queue
            # stays queued (the ` QUEUED ` badges keep showing) and only runs when
            # the next prompt starts a fresh drain.
            promote_next = (
                not interrupted
                and not turn_failed
                and engine.prompt_pending() > 0
            )
        finally:
            # Guaranteed reset: ANY exception above (widget lookups while a
            # modal screen is up, render errors, engine bookkeeping) used to
            # abort this method BEFORE _busy was cleared — leaving the input
            # locked and every new prompt queued forever ("app stops
            # working"). The busy/streaming reset must survive any failure.
            self._busy = False
            self._busy_sessions.discard(sid)
            # A force-stop is spent once nothing is busy anymore: a stale set
            # flag would instantly reject the next turn's dialogs.
            try:
                if not self._busy_sessions:
                    self._force_stop.clear()
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/prompt.py:408")
                pass
            # Indicators reflect the VIEWED session: if the finished turn ran
            # in a background session, its spinner was never on screen — and
            # if another turn is still streaming elsewhere, keep that honest.
            self._sync_streaming_visuals()
            # Keep the in-memory snapshot fresh (a later exit/close save reads it)
            # but persist NOTHING at turn end: per the save policy a conversation
            # file is only written on exit / Termux close / a crash mid-turn (the
            # 2s autosave tick already snapshotted it while streaming).
            try:
                session.messages = engine.get_history()
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/prompt.py:420")
                pass
            # this turn's bookkeeping slot is fully consumed — drop it (each future
            # turn / session / sub-agent owns its own slot, so nothing leaks across)
            self._turn.pop(sid, None)
            self._disarm_interrupt_escape()
        if promote_next:
            value = engine.pop_prompt()
            try:
                self._chat_for(sid).promote_next_queued()
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/prompt.py:430")
                pass
            self.notify("Running queued request")
            if value:
                self._start_turn(sid, value, engine)
                return
        # A message that broke a task_read wait (pending_fresh) starts its
        # own turn now that the old drain ended — promptly, never QUEUED.
        try:
            fresh = getattr(engine, "_pending_fresh", None)
            engine._pending_fresh = None
        except Exception:
            fresh = None
        if fresh:
            self._start_turn(sid, fresh, engine)
            return
        if turn_failed and engine.prompt_pending() > 0:
            # the drain died mid-way: leave the remaining prompts QUEUED (badges
            # stay) instead of machine-gunning them into a failing provider
            self.notify(
                "Queued requests are waiting — press Enter to continue.",
                severity="warning",
            )
        if getattr(result, "network_failed", False) and not self._exit_requested.is_set():
            # The turn died on transport (disconnect/DNS/timeout), not on a
            # model error: watch connectivity and resume automatically.
            self._start_reconnect_watch(sid)
        self._auto_speak_result(sid, result, interrupted, turn_failed)
        try:
            self.query_one(InputBar).focus()
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/prompt.py:460")
            pass

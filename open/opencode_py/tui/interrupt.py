"""ESC-to-interrupt and force-stop behaviour for the Textual TUI.

Split out of :mod:`opencode_py.tui.app` as a mixin; method bodies are unchanged.
"""
from __future__ import annotations

from ..util.log import swallow
from .input_bar import InputBar
from .status_bar import StatusBar

class InterruptMixin:

    def action_interrupt(self) -> None:
        # Flipping the session's flag makes run_turn stop at its next iteration
        # check (loop.py). The worker thread's `finally` calls _turn_done,
        # which resets the flag and clears _busy — we must NOT call _turn_done
        # here, or the worker would finish concurrently and double-complete.
        self._interrupt_turn(self._active_turn_session_id or self._current_session_id)


    def _interrupt_engines(self, session_id: str) -> None:
        """Abort the engine behind `session_id` (and its sub-agents) NOW.

        The interrupt flag alone is only noticed at the engine's next step
        check, so a turn parked on a socket read (or an idle "thinking"
        gap with no incoming chunk) would keep going until that read
        finished. `abort()` force-closes the provider stream, in-flight
        webfetch responses and MCP pipes, so the stop lands immediately.
        Abort is idempotent and safe on an idle engine.
        """
        engines = list(getattr(self, "_engines", {}).values())
        main = getattr(self, "_main_engine", None)
        if main is not None:
            engines.append(main)
        seen: set[int] = set()
        for engine in engines:
            sid = getattr(engine, "session_id", None)
            if engine is None or not sid or id(engine) in seen:
                continue
            if not self._in_family(sid, session_id):
                continue
            seen.add(id(engine))
            try:
                engine.abort()
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/interrupt.py:44")
                pass


    def _in_family(self, sid: str, root: str) -> bool:
        """True when `sid` is `root` itself or one of its nested sub-agents."""
        seen: set[str] = set()
        cur: str | None = sid
        while cur and cur not in seen:
            if cur == root:
                return True
            seen.add(cur)
            cur = self._parent_of(cur)
        return False


    def _interrupt_turn(self, sid: str, disarm: bool = True) -> None:
        """Stop the active turn: flip its flag, abort its engine, hush the voice.

        `disarm=False` keeps the ESC-again arming alive, so ESC then ESC can
        still force-stop everything right after the 1st press stopped the turn.
        """
        if self._busy_sessions and sid in self._busy_sessions:
            self.notify("Interrupting...")
            self._interrupt_flags[sid] = True
            self._interrupt_engines(sid)
            if disarm:
                self._disarm_interrupt_escape()
        # Ctrl+C also silences a talking reply (auto voice runs detached).
        try:
            self._stop_auto_voice()
            self._clear_voice(sid)
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/interrupt.py:76")
            pass


    def action_interrupt_escape(self) -> None:
        """ESC stops the turn on first press, force-stops everything on second.

        First press (busy): interrupts the active turn right away (session
        flag + engine abort + silences a talking reply) and arms the
        `esc again` hint. Second press (still busy within 5s): force-stops
        ANYTHING running — all busy sessions' flags, every engine's
        streams/fetches/sub-agents, all background tasks, and any open
        modal. Nothing is ignored: the worker threads can't miss it
        (flags + closed sockets + dismissed dialogs + unblocked waits).
        Idle sessions just move focus back to the prompt.
        """
        self._cancel_esc_timer()
        if not self._busy_sessions:
            self._force_stop.clear()
            try:
                self.query_one(InputBar).focus()
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/interrupt.py:97")
                pass
            return
        self._esc_presses += 1
        self._esc_timer = self.set_timer(5.0, self._disarm_interrupt_escape)
        if self._esc_presses >= 2:
            self._force_stop_all()
            self._disarm_interrupt_escape()
            return
        # 1st press: stop the turn NOW, then keep the 2nd press armed. Arm
        # before interrupting so _turn_done (when the worker actually ends)
        # is what clears the hint, never a re-arm after the fact.
        self._arm_interrupt_escape(armed=True)
        self._interrupt_turn(self._active_turn_session_id or self._current_session_id, disarm=False)


    def _force_stop_all(self, pop_modal: bool = True) -> None:
        """Second ESC: stop ANYTHING still running, nothing ignored.

        Aborts every known engine (busy sessions first, then any leftovers —
        a sub-agent sid lookup miss must never leave a child running), plus
        fetches/MCP servers via engine.abort(), permission/question waits via
        _force_stop, background tasks, and the top modal. `pop_modal=False`
        when the caller is itself a dismissing dialog (it pops itself).
        """
        for sid in list(self._busy_sessions):
            self._interrupt_flags[sid] = True
            self._interrupt_engines(sid)
        # A force-stop silences a talking reply too.
        try:
            self._stop_auto_voice()
            self._clear_voice_all()
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/interrupt.py:129")
            pass
        try:
            seen: set[int] = set()
            for eng in list(getattr(self, "_engines", {}).values()):
                if id(eng) not in seen:
                    seen.add(id(eng))
                    try:
                        eng.abort()
                    except Exception as _oc_exc:
                        swallow(_oc_exc, "tui/interrupt.py:138")
                        pass
            main_eng = getattr(self, "engine", None)
            if main_eng is not None and id(main_eng) not in seen:
                try:
                    main_eng.abort()
                except Exception as _oc_exc:
                    swallow(_oc_exc, "tui/interrupt.py:144")
                    pass
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/interrupt.py:146")
            pass
        # unblock workers stuck in a permission/question modal wait
        self._force_stop.set()
        # dismiss whatever modal is on top (a stuck dialog must not trap the
        # worker after the user demanded a stop)
        if pop_modal:
            try:
                if not self._is_main_screen_active():
                    self.pop_screen()
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/interrupt.py:156")
                pass
        # background shell tasks are work too — stop them all
        stopped = 0
        try:
            from ..tools import background as _bg

            stopped = _bg.stop_all()
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/interrupt.py:164")
            pass
        self.notify(f"Force-stopped{f' ({stopped} background task(s))' if stopped else ''}.")


    def _arm_interrupt_escape(self, armed: bool) -> None:
        if armed:
            self._esc_presses = 1
        else:
            self._esc_presses = 0
            self._cancel_esc_timer()
        try:
            self.query_one(StatusBar).set_interrupt_armed(armed)
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/interrupt.py:177")
            pass


    def _cancel_esc_timer(self) -> None:
        if self._esc_timer is not None:
            self._esc_timer.stop()
            self._esc_timer = None


    def _disarm_interrupt_escape(self) -> None:
        self._arm_interrupt_escape(False)

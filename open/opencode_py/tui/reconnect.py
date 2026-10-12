"""Auto-resume after a network-killed turn, for the Textual TUI.

Split out of :mod:`opencode_py.tui.app` as a mixin; method bodies are unchanged.
"""
from __future__ import annotations

from ..util.log import swallow
import threading
from .app_meta import _probe_online

class ReconnectMixin:

    # -- auto-resume after reconnect --------------------------------------
    def _start_reconnect_watch(self, sid: str) -> None:
        """Watch connectivity in the background after a network-killed turn.

        Cheap and fast: short probes with backoff (3s → 60s), one tiny
        request per interval. When the route is back, the failed turn resumes
        by itself. Any new turn on this session (or app exit) stops the watch.
        """
        self._stop_reconnect_watch(sid)
        stop = threading.Event()
        self._reconnect_watchers[sid] = stop
        try:
            self.notify("Connection lost — will resume automatically when back online.")
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/reconnect.py:25")
            pass

        def _watch() -> None:
            intervals = (3.0, 3.0, 5.0, 5.0, 10.0, 15.0, 20.0, 30.0)
            i = 0
            while not stop.is_set() and not self._exit_requested.is_set():
                if _probe_online():
                    try:
                        self.call_from_thread(self._auto_resume_turn, sid)
                    except Exception as _oc_exc:
                        swallow(_oc_exc, "tui/reconnect.py:35")
                        pass
                    return
                wait = intervals[i] if i < len(intervals) else 60.0
                i += 1
                stop.wait(timeout=wait)
            try:
                self._reconnect_watchers.pop(sid, None)
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/reconnect.py:43")
                pass

        threading.Thread(target=_watch, name=f"reconnect-{sid[:8]}", daemon=True).start()


    def _stop_reconnect_watch(self, sid: str) -> None:
        stop = self._reconnect_watchers.pop(sid, None)
        if stop is not None:
            try:
                stop.set()
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/reconnect.py:54")
                pass


    def _auto_resume_turn(self, sid: str) -> None:
        """Connectivity is back: resume the network-killed turn by itself."""
        self._reconnect_watchers.pop(sid, None)
        if self._exit_requested.is_set():
            return
        engine = self._engines.get(sid)
        if engine is None:
            return
        if sid in self._busy_sessions or self._busy:
            # user already started something — don't double-run
            return
        if engine.prompt_pending() > 0:
            # user queued follow-ups meanwhile: run those normally instead
            value = engine.pop_prompt()
            if value:
                try:
                    self._chat_for(sid).promote_next_queued()
                except Exception as _oc_exc:
                    swallow(_oc_exc, "tui/reconnect.py:75")
                    pass
                self.notify("Back online — running your queued request.")
                self._start_turn(sid, value, engine)
            return
        if sid != self._current_session_id:
            # user moved to another chat: don't hijack it, just report back
            self.notify("Back online — switch back and press Enter to resume.")
            return
        self.notify("Back online — resuming…")
        self._start_turn(sid, "", engine, resume=True)

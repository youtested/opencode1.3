"""Auto voice (TTS) behaviour for the Textual TUI.

Split out of :mod:`opencode_py.tui.app` as a mixin so the TUI class stays
readable. Method bodies are unchanged; ``OpenCodeTUI`` inherits this.
"""

from __future__ import annotations

from ..util.log import swallow
from typing import Any


class VoiceMixin:
    # -- auto voice -------------------------------------------------------
    def _vstate(self) -> tuple:
        """Streaming-voice per-session state (unsaid text, generation, spoke).

        Created lazily so event paths never crash on a partially-built app.
        """
        buf = getattr(self, "_voice_buf", None)
        if buf is None:
            buf = {}
            self._voice_buf = buf
        gen = getattr(self, "_voice_gen", None)
        if gen is None:
            gen = {}
            self._voice_gen = gen
        spoke = getattr(self, "_voice_spoke", None)
        if spoke is None:
            spoke = {}
            self._voice_spoke = spoke
        return buf, gen, spoke

    def _ensure_voice_worker(self) -> None:
        """One FIFO daemon thread speaks queued sentences in order, never
        overlapping. The UI thread only enqueues; it never waits on speech."""
        try:
            if getattr(self, "_voice_worker_started", False):
                return
            import queue as _queue_mod
            import threading as _thread_mod
            self._voice_queue = _queue_mod.Queue()
            self._voice_worker_started = True

            def _run() -> None:
                while True:
                    try:
                        item = self._voice_queue.get()
                    except Exception as _oc_exc:
                        swallow(_oc_exc, "tui/voice.py:48")
                        return
                    try:
                        self._voice_step(*item)
                    except Exception as _oc_exc:
                        swallow(_oc_exc, "tui/voice.py:52")
                        pass
                    finally:
                        try:
                            self._voice_queue.task_done()
                        except Exception as _oc_exc:
                            swallow(_oc_exc, "tui/voice.py:57")
                            pass

            _thread_mod.Thread(target=_run, name="auto-voice", daemon=True).start()
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/voice.py:61")
            pass

    def _voice_step(self, gen: int, sid: str, text: str) -> bool:
        """Speak one queued sentence unless a newer turn made it stale or
        voice was switched off mid-stream. Never raises."""
        try:
            _buf, generations, _spoke = self._vstate()
            if gen != generations.get(sid, 0):
                return False
            from ..tools import speak as _speak_mod
            if not _speak_mod.auto_enabled(getattr(self, "cfg", None)):
                return False
            res = _speak_mod._action_speak({"text": text}, getattr(self, "cfg", None))
            # Cloud voices play async (media-player returns at once): pace
            # the queue by the spoken length so sentences never overlap.
            # Cancellable: waits in 0.2s slices against the generation, so a
            # stop/switch/quit wakes the worker instantly instead of blocking
            # the whole voice queue up to 120s (sentences piling, shutdown
            # hanging).
            try:
                if isinstance(res, dict) and not res.get("error"):
                    meta = res.get("metadata") or {}
                    if str(meta.get("engine") or "") == "elevenlabs":
                        import time as _time

                        budget = min(120.0, max(2.0, len(text) / 14.0))
                        waited = 0.0
                        while waited < budget:
                            try:
                                _buf2, generations2, _spoke2 = self._vstate()
                                if gen != generations2.get(sid, 0):
                                    break
                            except Exception as _oc_exc:
                                swallow(_oc_exc, "tui/voice.py:94")
                                pass
                            if getattr(self, "_exit_requested", None) is not None:
                                try:
                                    if self._exit_requested.is_set():
                                        break
                                except Exception as _oc_exc:
                                    swallow(_oc_exc, "tui/voice.py:100")
                                    pass
                            _time.sleep(min(0.2, budget - waited))
                            waited += 0.2
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/voice.py:104")
                pass
            return True
        except Exception:
            return False

    def _say(self, sid: str, text: str) -> bool:
        """Queue one sentence for speech. True when queued. Never raises."""
        try:
            text = str(text or "").strip()
            if not text:
                return False
            from ..tools.speak import auto_enabled
            if not auto_enabled(getattr(self, "cfg", None)):
                return False
            try:
                if sid != getattr(self, "_current_session_id", sid):
                    return False
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/voice.py:122")
                pass
            _buf, gen, spoke = self._vstate()
            self._ensure_voice_worker()
            queue = getattr(self, "_voice_queue", None)
            if queue is None:
                return False
            queue.put((gen.get(sid, 0), sid, text))
            spoke[sid] = True
            return True
        except Exception:
            return False

    def _stream_voice(self, sid: str, delta: str) -> None:
        """Talk each finished sentence while the reply is still streaming.

        Called on every text delta: finished sentences are queued for speech
        immediately, the unfinished tail stays buffered. Only the viewed
        session talks. Never raises.
        """
        try:
            from ..tools.speak import auto_enabled, split_stream_chunks
            if not auto_enabled(getattr(self, "cfg", None)):
                return
            try:
                if sid != getattr(self, "_current_session_id", sid):
                    return
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/voice.py:149")
                pass
            buf, _gen, _spoke = self._vstate()
            combined = str(buf.get(sid, "") or "") + str(delta or "")
            chunks, rest = split_stream_chunks(combined)
            buf[sid] = rest
            for chunk in chunks:
                self._say(sid, chunk)
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/voice.py:157")
            pass

    def _flush_voice(self, sid: str) -> None:
        """Speak whatever sentence tail is still buffered. Never raises."""
        try:
            buf, _gen, _spoke = self._vstate()
            tail = str(buf.get(sid) or "").strip()
            buf[sid] = ""
            if tail:
                self._say(sid, tail)
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/voice.py:168")
            pass

    def _clear_voice(self, sid: str) -> None:
        """Drop one session's voice tail; its queued sentences go stale via
        the generation bump so they never speak late. Never raises."""
        try:
            buf, gen, spoke = self._vstate()
            buf.pop(sid, None)
            spoke.pop(sid, None)
            gen[sid] = gen.get(sid, 0) + 1
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/voice.py:179")
            pass

    def _clear_voice_all(self) -> None:
        """Drop every session's voice state (session switch / force-stop)."""
        try:
            buf, gen, spoke = self._vstate()
            for sid in list(buf):
                buf.pop(sid, None)
            for sid in list(spoke):
                spoke.pop(sid, None)
            for sid in list(gen):
                gen[sid] = gen.get(sid, 0) + 1
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/voice.py:192")
            pass

    def _auto_speak_result(self, sid: str, result: Any, interrupted: bool, turn_failed: bool) -> None:
        """Finish the reply's speech when Settings > auto voice is on.

        Plain-English: streaming voice already spoke each sentence live —
        this only says the leftover tail. If nothing streamed (auto was
        toggled on mid-turn), the whole reply is spoken. Never raises.
        """
        try:
            if interrupted or turn_failed:
                return
            text = str(getattr(result, "text", "") or "").strip()
            if not text:
                return
            if getattr(result, "error", ""):
                return
            from ..tools.speak import auto_enabled
            if not auto_enabled(getattr(self, "cfg", None)):
                return
            try:
                if sid != getattr(self, "_current_session_id", sid):
                    return
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/voice.py:216")
                pass
            _buf, _gen, spoke = self._vstate()
            if spoke.get(sid):
                self._flush_voice(sid)
                return
            self._say(sid, text)
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/voice.py:223")
            pass

    def _stop_auto_voice(self) -> None:
        try:
            from ..tools.speak import stop_speech
            stop_speech()
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/voice.py:230")
            pass

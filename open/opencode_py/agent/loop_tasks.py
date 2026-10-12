"""Sub-agent spawning, background tasks and task control for the agent loop.

Split out of :mod:`opencode_py.agent.loop` as a mixin; method bodies are
byte-for-byte unchanged."""

from __future__ import annotations

from ..util.log import swallow
import threading, time

from typing import Any, Callable

from ..session import save_session


class LoopTasksMixin:
    def _subagent_bridge(self, sub_id: str) -> Callable[[dict], None]:
        """Forward a sub-agent's events to our own on_event, tagged with the
        sub-session id so the UI can route them to the right chat view.

        Nested sub-agents already carry their own `session_id` (tagged by the
        deeper bridge); keep that id so a grandchild's events reach its own chat
        instead of being re-tagged with the direct child's id.
        """

        def forward(event: dict[str, Any]) -> None:
            kind = event.get("kind", "")
            sid = event.get("session_id") or sub_id
            payload = {k: v for k, v in event.items() if k not in ("kind", "session_id")}
            self._emit(kind, session_id=sid, **payload)

        return forward

    @staticmethod
    def _child_rotation(provider_factory: Callable[[], Any], session_id: str) -> Any:
        """Build a child's rotation, rebound to the CHILD session id.

        The default factory closes over the parent session; without rebinding,
        every parallel child shares one Zen upstream lane. Rotations built by
        custom factories (tests) are rebound too when they expose session_id.
        Never raises: falls back to a fresh rotation on factory failure.
        """
        try:
            rotation = provider_factory()
        except Exception:
            rotation = None
        if rotation is not None and getattr(rotation, "session_id", None) is not None:
            try:
                rotation.session_id = session_id
                return rotation
            except Exception as _oc_exc:
                swallow(_oc_exc, "agent/loop_tasks.py:50")
                pass
            return rotation
        return rotation

    def spawn_task(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Run a sub-agent in its own session; returns the sub-agent's reply.

        The sub-agent gets its own history/session but shares the parent's
        config, auth, permission engine, directory, and interrupt flag. Events
        stream out tagged with the sub-session id. Nested `task` calls work:
        sub-agents build their own registry, which lazily resolves the same hook.

        With background=true the child runs detached: this returns a session
        id immediately and the parent keeps working; use task_status /
        task_read / task_stop (same verbs as background_task) to poll,
        collect, or kill it. At most ONE background agent at a time.
        """
        bg = arguments.get("background")
        if bg is True or str(bg or "").strip().lower() in ("1", "true", "yes", "on"):
            return self._spawn_task_background(arguments)
        bundle, error = self._prepare_task(arguments)
        if error is not None:
            return error
        assert bundle is not None
        return self._finish_task(bundle)

    def _release_task_slot(self, bundle: dict) -> None:
        """Free one _subagent_sem slot exactly once per bundle.

        Inline and background paths both end in _finish_task; a killed
        background worker still completes through _finish_task later.
        Without the flag, stop-then-finish would release twice and grow
        the semaphore past its cap, silently allowing extra agents.
        """
        try:
            if bundle.get("_slot_released"):
                return
            bundle["_slot_released"] = True
            self._subagent_sem.release()
        except Exception as _oc_exc:
            swallow(_oc_exc, "agent/loop_tasks.py:90")
            pass

    def _bg_live(self) -> dict[str, dict[str, Any]] | None:
        """The running detached entry, or None. Holds _bg_lock."""
        with self._bg_lock:
            for sid, entry in self._bg_agents.items():
                if entry.get("running"):
                    return entry
        return None

    def _spawn_task_background(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Start a child detached; return its session id immediately."""
        if self._bg_live() is not None:
            return {
                "output": (
                    "A background agent is already running. Collect it with "
                    "task_read (or stop it with task_stop) before starting another."
                ),
                "error": True,
            }
        bundle, error = self._prepare_task(arguments)
        if error is not None:
            return error
        assert bundle is not None
        sub = bundle["sub"]
        sub_session = bundle["session"]
        # A detached child needs its OWN interrupt flag: the parent's flag
        # would kill it on the next user Ctrl+C even though the user only
        # meant to stop the parent. task_stop flips this one instead.
        stop_flag = {"stopped": False}
        parent_interrupt = self.interrupt
        sub.interrupt = lambda: bool(stop_flag["stopped"]) or parent_interrupt()
        entry: dict[str, Any] = {
            "sid": sub_session.id,
            "bundle": bundle,
            "running": True,
            "result": None,
            "stop_flag": stop_flag,
            "started": time.time(),
            "thread": None,
        }
        with self._bg_lock:
            self._bg_agents[sub_session.id] = entry

        def _run() -> None:
            try:
                entry["result"] = self._finish_task(bundle)
            except Exception as e:  # never leave a zombie entry
                entry["result"] = {"output": f"background agent failed: {e}", "error": True}
            finally:
                entry["running"] = False

        thread = threading.Thread(target=_run, name=f"bg-agent-{sub_session.id[:8]}", daemon=True)
        entry["thread"] = thread
        thread.start()
        return {
            "output": (
                f"Started background agent {sub_session.id} ({bundle['description']}).\n"
                f"Keep working — poll with task_status, collect with task_read, "
                f"kill with task_stop."
            ),
            "metadata": {"sessionId": sub_session.id, "background": True},
        }

    def _bg_get(self, session_id: str) -> dict[str, Any] | None:
        with self._bg_lock:
            return self._bg_agents.get((session_id or "").strip())

    def _bg_collect(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Blocking collect: wait for the worker, fold reply into history
        as a tool result would, drop the entry. Returns the child result."""
        thread = entry.get("thread")
        if thread is not None:
            while thread.is_alive():
                if self.interrupt():
                    break
                # A newly typed user message breaks the wait (see
                # wake_bg_wait): the turn ends, the message starts its own
                # turn at once instead of QUEUED, the agent keeps running.
                wake = getattr(self, "_bg_wake", None)
                if wake is not None:
                    try:
                        if wake.is_set():
                            wake.clear()
                            return {
                                "output": (
                                    "Background agent still running — "
                                    "collect it later with task_read."
                                ),
                                "metadata": {
                                    "sessionId": entry.get("sid", ""),
                                    "background": True,
                                    "still_running": True,
                                },
                            }
                    except Exception as _oc_exc:
                        swallow(_oc_exc, "agent/loop_tasks.py:186")
                        pass
                thread.join(timeout=0.2)
        with self._bg_lock:
            self._bg_agents.pop(entry.get("sid", ""), None)
        result = entry.get("result") or {"output": "(no reply from sub-agent)", "error": True}
        return result

    def wake_bg_wait(self) -> bool:
        """Wake a parent parked in task_read so a fresh user message can
        start its own turn immediately. Returns True if a wait was broken
        (the bg agent keeps running; its reply folds in later).

        Also arms end-of-step: the current tool step completes, then the
        drain exits instead of starting new provider steps.
        """
        try:
            wake = getattr(self, "_bg_wake", None)
            if wake is None:
                import threading as _th

                wake = _th.Event()
                self._bg_wake = wake
            with self._bg_lock:
                parked = any(e.get("running") for e in self._bg_agents.values())
            if not parked:
                return False
            wake.set()
            try:
                self._bg_end_turn = True
            except Exception as _oc_exc:
                swallow(_oc_exc, "agent/loop_tasks.py:216")
                pass
            return True
        except Exception:
            return False

    def task_status(self, session_id: str = "") -> dict[str, Any]:
        """Peek at detached agent(s): running/finished, runtime, title."""
        with self._bg_lock:
            entries = list(self._bg_agents.values())
            if session_id:
                entries = [e for e in entries if e.get("sid") == session_id.strip()]
        if not entries:
            return {"output": "No background agents."}
        lines = []
        for e in entries:
            state = "RUNNING" if e.get("running") else "FINISHED"
            lines.append(
                f"  {e.get('sid', '?')}  {state:<9} {time.time() - e.get('started', time.time()):6.1f}s  "
                f"{str((e.get('bundle') or {}).get('description', ''))[:60]}"
            )
        return {"output": f"{len(entries)} background agent(s):\n" + "\n".join(lines)}

    def task_read(self, session_id: str = "") -> dict[str, Any]:
        """Collect a detached agent's reply (waits if still running)."""
        entry = None
        if session_id:
            entry = self._bg_get(session_id)
            if entry is None:
                return {"output": f"No such background agent {session_id!r}.", "error": True}
        else:
            entry = self._bg_live()
            if entry is None:
                with self._bg_lock:
                    leftovers = list(self._bg_agents.values())
                if not leftovers:
                    return {"output": "No background agents."}
                entry = leftovers[0]
        return self._bg_collect(entry)

    def task_stop(self, session_id: str = "") -> dict[str, Any]:
        """Kill a detached agent (or all) without collecting."""
        with self._bg_lock:
            if session_id:
                targets = [self._bg_agents.get(session_id.strip())]
                targets = [t for t in targets if t is not None]
            else:
                targets = list(self._bg_agents.values())
        if not targets:
            return {"output": "No background agents."}
        stopped = 0
        for entry in targets:
            try:
                entry.get("stop_flag", {})["stopped"] = True
            except Exception as _oc_exc:
                swallow(_oc_exc, "agent/loop_tasks.py:270")
                pass
            thread = entry.get("thread")
            if thread is not None:
                thread.join(timeout=5)
            with self._bg_lock:
                self._bg_agents.pop(entry.get("sid", ""), None)
            stopped += 1
        # NOTE: no _subagent_sem.release() here — the worker still owns its
        # slot until _finish_task runs (or _release_task_slot fires once via
        # the bundle flag). Releasing here too would over-release the cap.
        return {"output": f"Stopped {stopped} background agent(s)."}

    def _fold_finished_bg_agents(self) -> None:
        """Fold finished-but-uncollected detached replies into history.

        Runs at turn end: a background agent that completed while the
        parent kept working must not vanish silently. Its reply is
        appended as an assistant note (capped length) so the NEXT turn
        sees it even if the model never called task_read. Still-running
        agents are left alone.
        """
        with self._bg_lock:
            done = [e for e in self._bg_agents.values() if not e.get("running")]
        for entry in done:
            self._fold_one_bg_agent(entry)

    def _fold_one_bg_agent(self, entry: dict, result: dict | None = None) -> None:
        """Fold a single finished entry into history + emit bg_collected.

        `result` overrides entry["result"]: at worker-completion time the
        entry doesn't hold it yet (it's still being computed), so the
        caller passes the fresh dict. Turn-end fold calls without it.
        """
        result = result if result is not None else (entry.get("result") or {})
        text = str(result.get("output", "") or "")[:2000]
        title = str((entry.get("bundle") or {}).get("description", "sub-agent"))
        try:
            self._history.append({
                "role": "assistant",
                "content": f"[background agent finished: {title}]\n{text}",
            })
        except Exception as _oc_exc:
            swallow(_oc_exc, "agent/loop_tasks.py:312")
            pass
        with self._bg_lock:
            self._bg_agents.pop(entry.get("sid", ""), None)
        self._emit("bg_collected", session_id=entry.get("sid", ""), title=title)

    def _finish_task(self, bundle: dict) -> dict:
        """Run a prepared child NOW and return its reply (inline path)."""
        sub = bundle["sub"]
        sub_session = bundle["session"]
        prompt = bundle["prompt"]
        description = bundle["description"]
        subagent_type = bundle["agent"]
        task_started = bundle["started"]
        try:
            result = sub.run_turn(prompt)
        except BaseException as e:
            # never leave the sub-agent session dangling: persist and report done.
            # BaseException (not just Exception) so Ctrl+C/SystemExit also
            # releases the semaphore, pops the child, and closes server refs —
            # otherwise 4 interrupts exhaust the pool and every future task
            # fails with "too many sub-agents".
            try:
                sub_session.messages = sub.get_history()
            except Exception as _oc_exc:
                swallow(_oc_exc, "agent/loop_tasks.py:336")
                pass
            sub_session.completed = time.time()
            try:
                if sub_session.messages:
                    save_session(sub_session)
            except Exception as _oc_exc:
                swallow(_oc_exc, "agent/loop_tasks.py:342")
                pass
            try:
                sub.close()
            except Exception as _oc_exc:
                swallow(_oc_exc, "agent/loop_tasks.py:346")
                pass
            self._emit(
                "subagent_done",
                session_id=sub_session.id,
                agent=subagent_type,
                title=description,
                ok=False,
            )
            self._release_task_slot(bundle)
            self.subagents.pop(sub_session.id, None)
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            return {
                "output": f"sub-agent failed: {e}",
                "error": True,
                "metadata": {
                    "sessionId": sub_session.id,
                    "title": description,
                    "status": "error",
                    "agent": subagent_type,
                },
            }
        sub_session.messages = sub.get_history()
        sub_session.completed = time.time()
        try:
            if sub_session.messages:
                save_session(sub_session)
        except Exception as _oc_exc:
            swallow(_oc_exc, "agent/loop_tasks.py:374")
            pass
        sub.close()

        # Stamp the child link onto the pending assistant message NOW (not
        # only at history-append): an interrupt between spawn and append used
        # to leave the row permanently unlinked for all future resumes.
        try:
            for m in reversed(self._history):
                if isinstance(m, dict) and m.get("role") == "assistant":
                    m.setdefault("task_children", {})[str(sub_session.id)] = {
                        "title": description, "agent": subagent_type}
                    break
        except Exception as _oc_exc:
            swallow(_oc_exc, "agent/loop_tasks.py:387")
            pass
        # Completion detail travels WITH the task result (engine-level truth):
        # wall-clock runtime + the child's real toolcall count. The TUI used
        # to derive both from live widgets (start-time map + child chat) that
        # were already popped/unmounted when it read them — so finished rows
        # showed nothing but the `↓ ctrl+down` hint. Stamping here makes the
        # `↳ N toolcalls · Xs` footer race-free and resume-proof.
        try:
            child_calls = int(getattr(result, "tool_calls_made", 0) or 0)
        except Exception:
            child_calls = 0
        task_duration = max(0.0, time.time() - task_started)
        # Was this child detached? Snapshot BEFORE any fold/collect pops
        # the entry: the TUI needs it to decide row teardown vs "done,
        # unread" (Fix B). bg_collected mirrors whether the reply was
        # already folded/collected at emit time.
        try:
            with self._bg_lock:
                _bg_entry_now = self._bg_agents.get(sub_session.id)
            _was_bg = _bg_entry_now is not None
        except Exception:
            _was_bg = False
        self._emit(
            "subagent_done",
            session_id=sub_session.id,
            agent=subagent_type,
            title=description,
            ok=not result.error,
            duration=task_duration,
            toolcalls=child_calls,
            was_background=_was_bg,
        )
        text = result.text or result.error or "(no reply from sub-agent)"
        self._release_task_slot(bundle)
        # Free the finished child (history + registry + rotation): the TUI
        # keeps its own transcript; holding it here too doubles peak RAM per
        # fan-out on low-memory phones.
        self.subagents.pop(sub_session.id, None)
        # Background worker completion: if the parent has NO turn running
        # (the idle case — the whole point of background agents), fold the
        # reply into history NOW so the model learns it done immediately
        # instead of waiting for a turn end that may never come. When a
        # parent turn IS active, turn-end fold owns it (avoids mid-step
        # history mutation under the running loop). Pass the fresh result:
        # entry["result"] isn't set yet (we're still computing it).
        try:
            with self._bg_lock:
                bg_entry = self._bg_agents.get(sub_session.id)
            if bg_entry is not None and not getattr(self, "_turn_active", False):
                self._fold_one_bg_agent(bg_entry, {
                    "output": text,
                    "error": bool(result.error and not result.text),
                })
        except Exception as _oc_exc:
            swallow(_oc_exc, "agent/loop_tasks.py:441")
            pass
        return {
            "output": text,
            "error": bool(result.error and not result.text),
            "metadata": {
                "sessionId": sub_session.id,
                "title": description,
                "status": "error" if result.error else "completed",
                "duration_s": task_duration,
                "toolcalls": child_calls,
                # resolved agent actually launched (may differ from the
                # requested input after read-only forcing) so the row
                # renders what ran, not what was asked.
                "agent": subagent_type,
            },
        }

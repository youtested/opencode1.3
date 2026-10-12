"""ESC interrupts on the 1st press; a 2nd press force-stops everything.

Two regressions live here:

1. The 1st ESC used to only arm a footer hint, so the most natural "stop"
   key looked broken. It now flips the turn's interrupt flag and aborts the
   engine straight away; ESC ESC still escalates to the nuclear force-stop.
2. `_interrupt_engines` was CALLED (by Ctrl+C and by the force-stop) but never
   DEFINED, so every interrupt raised AttributeError inside the key handler and
   Textual tore the app down with a traceback. The tests drive the real methods
   and a real mounted app, so a missing attribute fails here, not on the phone.
"""

from pathlib import Path

from opencode_py.config import Config
from opencode_py.tui.app import OpenCodeTUI
from opencode_py.tools import background as bg


class _Engine:
    """Just enough AgentLoop surface for the abort path."""

    def __init__(self, sid: str) -> None:
        self.session_id = sid
        self.aborts = 0

    def abort(self) -> None:
        self.aborts += 1


class _Bar:
    def __init__(self) -> None:
        self.armed = False
        self.focus_calls = 0

    def set_interrupt_armed(self, armed: bool) -> None:
        self.armed = armed

    def focus(self) -> None:
        self.focus_calls += 1


class _Flag:
    def __init__(self) -> None:
        self._set = False

    def set(self) -> None:
        self._set = True

    def clear(self) -> None:
        self._set = False

    def is_set(self) -> bool:
        return self._set


class _App:
    """The real esc/interrupt code paths on a stub app (no network, no mount)."""

    def __init__(self, busy: bool = True, *sids: str, modal: bool = False) -> None:
        for name in (
            "action_interrupt",
            "action_interrupt_escape",
            "_interrupt_turn",
            "_interrupt_engines",
            "_in_family",
            "_parent_of",
            "_cancel_esc_timer",
            "_force_stop_all",
            "_arm_interrupt_escape",
            "_disarm_interrupt_escape",
        ):
            setattr(self, name, getattr(OpenCodeTUI, name).__get__(self))
        self.bar = _Bar()
        self.sid = sids[0] if sids else "s1"
        self._busy_sessions = set(sids or ("s1",)) if busy else set()
        self._interrupt_flags: dict[str, bool] = {}
        self._child_parent: dict[str, str] = {}
        self._children: dict[str, list[dict]] = {}
        self._sessions: dict[str, object] = {}
        self._engines = {s: _Engine(s) for s in sids or ("s1",)}
        self._main_engine = self._engines.get(self.sid)
        self.popped = 0
        self._stop_auto_voice = lambda: None
        self._clear_voice = lambda _sid: None
        self._clear_voice_all = lambda: None
        self._force_stop = _Flag()
        self._esc_presses = 0
        self._esc_timer = None
        self._active_turn_session_id = self.sid
        self._current_session_id = self.sid
        self.timers: list[float] = []
        self.set_timer = lambda secs, _cb: self.timers.append(secs)
        self.notify = lambda *_a, **_k: None
        self.query_one = lambda _kind: self.bar
        self.modal = modal
        self._is_main_screen_active = lambda: not self.modal
        self.pop_screen = lambda: setattr(self, "popped", self.popped + 1)

    def esc(self) -> None:
        self.action_interrupt_escape()

    def aborts(self) -> int:
        return sum(e.aborts for e in self._engines.values())


def test_first_esc_stops_the_turn_immediately():
    app = _App()
    app.esc()

    assert app._interrupt_flags.get(app.sid) is True, "1st ESC did not stop the turn"
    assert app._engines[app.sid].aborts == 1, "1st ESC did not abort the engine"
    assert app.bar.armed is True, "2nd ESC must stay armed after the 1st one"


def test_first_esc_is_not_the_force_stop():
    app = _App()
    app.esc()

    assert app._force_stop.is_set() is False
    assert app.popped == 0, "1st ESC must not nuke background tasks/modal"


def test_first_esc_also_aborts_sub_agent_engines():
    app = _App(True, "s1", "kid", "other")
    app._child_parent["kid"] = "s1"
    app._child_parent["other"] = "nobody"

    app.esc()

    assert app._engines["kid"].aborts == 1, "sub-agent kept running"
    assert app._engines["other"].aborts == 0, "another session was aborted"


def test_second_esc_force_stops_everything(monkeypatch):
    stopped: list[bool] = []
    monkeypatch.setattr(bg, "stop_all", lambda: stopped.append(True) or 7)
    app = _App(True, "s1", "s2", modal=True)
    app.esc()
    app.esc()

    assert app._force_stop.is_set() is True
    assert app._interrupt_flags.get("s2") is True, "2nd ESC skipped a busy session"
    assert stopped == [True], "2nd ESC did not stop background tasks"
    assert app.popped == 1, "2nd ESC did not dismiss the top modal"
    assert app._esc_presses == 0
    assert app.bar.armed is False


def test_esc_when_idle_only_refocuses_the_prompt():
    app = _App(busy=False)
    app.esc()

    assert app._interrupt_flags == {}, "idle ESC must not interrupt anything"
    assert app.bar.focus_calls == 1
    assert app._esc_presses == 0


def test_turn_end_disarms_the_second_press():
    app = _App()
    app.esc()
    # the worker finishing -> _turn_done clears the armed hint
    app._disarm_interrupt_escape()

    assert app.bar.armed is False
    assert app._esc_presses == 0


class _Ns:
    def __init__(self) -> None:
        self.ask_callback = None
        self.mode = "ask"


class _WiredEngine(_Engine):
    """The AgentLoop attributes the real app touches while wiring/mounting."""

    def __init__(self, sid: str) -> None:
        super().__init__(sid)
        self.permission = _Ns()
        self.question_service = _Ns()
        self.on_event = None
        self.interrupt = lambda: False
        self.agent = "build"
        self.model = "test-model"


async def test_escape_on_a_mounted_app_never_crashes(tmp_path: Path):
    """The regression that shipped: a real ESC key event on the real app.

    Every stub test above passes its own methods in, so a method that exists
    nowhere in the class slips through. This one presses the actual key on a
    mounted OpenCodeTUI — the app-level binding plus the real interrupt path.
    """
    engine = _WiredEngine("s1")
    app = OpenCodeTUI(cfg=Config(), engine=engine, directory=Path(tmp_path))
    async with app.run_test() as pilot:
        sid = app.session.id
        app._busy_sessions.add(sid)
        try:
            await pilot.press("escape")
            await pilot.pause()

            assert app._interrupt_flags.get(sid) is True, "ESC did not stop the turn"
            assert engine.aborts == 1, "ESC did not abort the engine"
        finally:
            app._busy_sessions.discard(sid)

"""Regression: opening a sub-agent's task row must not replay its transcript.

`_render_history()` only APPENDS bubbles, so a caller that replays into a chat
that already has messages mounted duplicates the whole conversation: the final
report showed once as it streamed, then again verbatim right underneath it.
`_switch_session` already guards this with "only replay into an empty chat";
this test pins the same rule for `on_open_task_session`.
"""

from pathlib import Path
from types import SimpleNamespace

MESSAGES = [
    {"role": "user", "content": "go"},
    {"role": "assistant", "content": "the final report"},
]


class _FakeChat:
    def __init__(self, attached: bool = True, populated: bool = True) -> None:
        self._attached = attached
        self._populated = populated

    @property
    def is_attached(self) -> bool:
        return self._attached

    def query(self, _selector: str) -> list:
        return ["already-rendered-bubble"] if self._populated else []


def _make_app(chat):
    from opencode_py.tui import app as appmod

    app = appmod.OpenCodeTUI.__new__(appmod.OpenCodeTUI)
    app.cfg = None
    app.auth = None
    app.directory = Path.cwd()
    app.notify = lambda *a, **k: None
    app._sessions: dict = {}
    app._engines: dict = {}
    app._pruned: set = set()
    app._pending_render: dict = {}
    app._rebuild_family_from_disk = lambda: None
    app._chat_for = lambda _sid: chat
    app._switch_session = lambda _sid: None
    app.rendered: list = []
    app._render_history = lambda c, msgs: app.rendered.append((c, list(msgs)))
    return app


def _open_task_row(app, monkeypatch, sid: str = "sub-1"):
    from opencode_py import session as sessmod

    sess = SimpleNamespace(
        id=sid,
        title="sub",
        agent="build",
        provider=None,
        model=None,
        directory=None,
        messages=list(MESSAGES),
    )
    monkeypatch.setattr(sessmod, "load_session", lambda _sid: sess)
    app.on_open_task_session(SimpleNamespace(sid=sid))
    return sess


def test_populated_chat_is_not_replayed_a_second_time(monkeypatch):
    """The bug: an attached chat already showing the report got the whole
    transcript appended again, so every message (the report most visibly)
    appeared twice."""
    app = _make_app(_FakeChat(attached=True, populated=True))
    _open_task_row(app, monkeypatch)

    assert app.rendered == [], "replayed history onto an already-rendered chat"


def test_empty_attached_chat_still_replays(monkeypatch):
    """A chat that is mounted but empty must still get its history, or opening
    a task row would show a blank conversation."""
    app = _make_app(_FakeChat(attached=True, populated=False))
    _open_task_row(app, monkeypatch)

    assert len(app.rendered) == 1, "empty chat was not replayed"
    assert app.rendered[0][1] == MESSAGES


def test_unmounted_chat_defers_instead_of_replaying(monkeypatch):
    app = _make_app(_FakeChat(attached=False, populated=False))
    _open_task_row(app, monkeypatch)

    assert app.rendered == [], "replayed into a chat that is not mounted"
    assert app._pending_render.get("sub-1") is True, "render was not deferred"

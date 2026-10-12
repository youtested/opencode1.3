"""opencode_py Textual TUI app.

Mirrors opencode's session screen: header status bar (agent/model/provider/
permission), scrollable chat with live tool blocks + diff rendering, and a
prompt input bar. The engine runs in a worker thread; events are bridged to
the UI via call_from_thread.
"""

from __future__ import annotations

from ..util.log import swallow
import sys
import threading
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical

from .app_meta import _prewarm_heavy_deps

from ..config import load_config
from .chat_view import ChatView
from .input_bar import InputBar
from .status_bar import StatusBar
from .subagent_footer import SubagentFooter

from .agent_md import AgentMdMixin
from .agents import AgentsMixin
from .capture import CaptureMixin
from .chrome import ChromeMixin
from .commands import CommandsMixin
from .dialogs import DialogsMixin
from .events import EventsMixin
from .interrupt import InterruptMixin
from .prompt import PromptMixin
from .reconnect import ReconnectMixin
from .sessions import SessionsMixin
from .settings import SettingsMixin
from .skills import SkillsMixin
from .voice import VoiceMixin

if TYPE_CHECKING:
    from ..agent.loop import AgentLoop
    from ..config import Config


class OpenCodeTUI(ChromeMixin, SettingsMixin, AgentsMixin, SkillsMixin, AgentMdMixin, CommandsMixin, SessionsMixin, EventsMixin, PromptMixin, ReconnectMixin, InterruptMixin, DialogsMixin, CaptureMixin, VoiceMixin, App):
    SUB_TITLE = "opencode_py"

    ENABLE_COMMAND_PALETTE = False  # ctrl+p is bound to Settings instead

    CSS = """
    Screen {
        background: $background;
        color: $text;
    }
    #root {
        layout: vertical;
        height: 1fr;
    }
    #chat-stack {
        layout: vertical;
        height: 1fr;
    }
    ChatView {
        width: 100%;
        height: 1fr;
        padding: 0 2;
        background: $background;
        scrollbar-size-vertical: 0;
        scrollbar-size-horizontal: 0;
    }
    .chat-welcome-logo {
        width: 100%;
        height: 100%;
        content-align: center middle;
    }
    SubagentFooter {
        width: 100%;
        height: auto;
        background: $panel;
        padding: 0 2;
    }
    #subagent-info {
        width: 1fr;
        height: 1;
        padding: 0 1 0 2;
    }
    #subagent-nav {
        height: 1;
        padding: 0 1;
        background: $panel;
    }
    .subagent-gap {
        height: 1;
    }
    InputBar {
        width: 100%;
        height: auto;
        padding: 0 1;
        background: $background;
    }
    .prompt-frame {
        height: auto;
        background: $background;
    }
    #prompt-accent {
        width: 1;
        height: 3;
        background: $primary;
    }
    .prompt-body {
        width: 1fr;
        height: auto;
        padding: 0 0 0 1;
    }
    #prompt-input {
        width: 1fr;
        background: $surface;
        color: $text;
        border: none;
        outline: none;
        padding: 0 1;
        height: 3;
        min-height: 3;
        content-align: left middle;
    }
    #prompt-input:focus {
        border: none;
        outline: none;
    }
    #prompt-title {
        width: 1fr;
        height: 1;
        margin-top: 1;
        margin-bottom: 1;
        padding: 0 1;
        color: $text-muted;
    }
    #prompt-meta {
        width: 1fr;
        height: 1;
        padding: 0 1;
        color: $text-muted;
    }
    #prompt-status {
        width: 1fr;
        height: 1;
        padding: 0 1;
        color: $text-muted;
    }
    #prompt-status.hidden {
        display: none;
    }
    #suggestions {
        height: auto;
        max-height: 10;
        overflow-y: auto;
        scrollbar-size-vertical: 1;
        padding: 0 1;
        background: $surface;
        color: $text;
        border: round $accent;
        margin: 0 1 1 2;
    }
    #suggestions.hidden {
        display: none;
    }
    StatusBar {
        width: 100%;
        height: 1;
        padding: 0 1;
        background: $background;
        color: $text-muted;
    }
    .cmd-popup {
        width: 74;
        height: auto;
        border: round $accent;
        background: $surface;
        padding: 1 2;
    }
    .cmd-popup.settings {
        width: 78;
        max-height: 92%;
        /* no bottom padding: the hint line sits flush on the border so
           there is no dead row between the buttons and the popup edge. */
        padding: 1 2 0 2;
    }
    .cmd-popup-title {
        height: 1;
        text-style: bold;
        color: $accent;
        background: $surface;
    }
    .settings-title {
        height: 1;
        text-style: bold;
        color: $accent;
        background: $surface;
    }
    .settings-count {
        height: 1;
        color: $text-muted;
        background: $surface;
        padding: 0 1;
    }
    .settings-scroll {
        height: 40;
        min-height: 4;
        max-height: 70%;
        border: none;
    }
    .settings-body {
        padding: 1 2;
        color: $text;
    }
    #settings-list {
        height: auto;
        max-height: 20;
        border: solid $accent-muted;
        background: $background;
        padding: 0 1;
        scrollbar-size-vertical: 1;
        scrollbar-size-horizontal: 0;
    }
    .settings-edit {
        margin: 0 2;
        height: 3;
    }
    .settings-hint {
        height: 1;
        padding: 0 2;
        color: $text-muted;
        background: $surface;
    }
    .cmd-popup-actions {
        height: 1;
        align: center middle;
        background: $surface;
        padding: 0 1;
    }
    .cmd-popup-actions Button {
        height: 1;
        min-width: 0;
        padding: 0 1;
        margin: 0 1;
        border: none;
    }
    /* settings keeps roomier buttons; its bar is taller to match,
       with no trailing gap before the hint/popup edge. Height 3 =
       1 text line + top/bottom border, so the label is never clipped.
       Visible `tall` borders per variant (the shape the fill expects —
       `round` made the fill spill outside the corners). */
    .cmd-popup.settings .cmd-popup-actions {
        height: 3;
        padding: 0 1;
    }
    .cmd-popup.settings .cmd-popup-actions Button {
        height: 3;
        min-width: 12;
        padding: 0 2;
        border: heavy $accent;
        background: transparent;
    }
    /* sessions buttons match the settings size: roomy height-3 pills
       with visible heavy borders, same bar height. One bottom padding
       row so the last row breathes just above the border. */
    .cmd-popup.session-popup {
        max-height: 90%;
        width: 88;
        max-width: 92%;
        margin: 2 4;
        padding: 1 2;
    }
    /* permission dialog: same centered popup shell; tight width for its
       four buttons in a row. No margin: the screen's center alignment
       places it exactly (margins skewed it by a cell). */
    .cmd-popup.perm-popup {
        width: 76;
        max-width: 94%;
        max-height: 90%;
        padding: 1 2;
    }
    /* the permission buttons row has no own CSS and stretched to fill
       the popup height; hug content instead. */
    .cmd-popup.perm-popup .dialog-buttons {
        height: auto;
        align: center middle;
    }
    /* delete/rename dialogs match the sessions button size: roomy
       height-3 pills, heavy accent borders, transparent fill (no
       colors anywhere). Bars match too. */
    .cmd-popup.session-popup .cmd-popup-actions,
    ConfirmDeleteDialog .cmd-popup-actions,
    RenameDialog .cmd-popup-actions {
        height: 3;
        padding: 0 1;
        align: center middle;
        background: $surface;
    }
    .cmd-popup.session-popup .cmd-popup-actions Button,
    ConfirmDeleteDialog .cmd-popup-actions Button,
    RenameDialog .cmd-popup-actions Button {
        height: 3;
        min-width: 12;
        padding: 0 2;
        margin: 0 1;
        border: heavy $accent;
        background: transparent;
        color: $text;
    }
    /* no focus/hover highlight on dialog buttons: the auto-focused
       Cancel looked "highlighted" next to Delete. Keyboard still
       works (Enter activates the focused one), just no visuals. */
    .cmd-popup.session-popup .cmd-popup-actions Button:hover,
    ConfirmDeleteDialog .cmd-popup-actions Button:hover,
    RenameDialog .cmd-popup-actions Button:hover,
    .cmd-popup.session-popup .cmd-popup-actions Button:focus,
    ConfirmDeleteDialog .cmd-popup-actions Button:focus,
    RenameDialog .cmd-popup-actions Button:focus {
        background: transparent;
        border: heavy $accent;
        color: $text;
        text-style: none;
        text-opacity: 1;
    }
    /* sessions search: same look as the models search (height-3 heavy
       accent border) so it is impossible to miss; width auto-fits the
       popup which is what keeps it "smaller". */
    #session-search {
        height: 3;
        border: heavy $accent;
        padding: 0 1;
        background: $surface;
        color: $text;
        margin-bottom: 1;
    }
    #session-search:focus {
        border: heavy $accent;
        background: $surface;
        background-tint: transparent;
    }
    #session-search > .input--cursor {
        background: $primary;
        color: $background;
        text-style: bold;
    }
    #session-search > .input--placeholder {
        color: $text-muted;
    }
    SessionList, AgentsView, SettingsScreen, ConfirmDeleteDialog, RenameDialog,
    PermissionDialog, QuestionDialog, PermissionsPopup {
        align: center middle;
    }
    OptionList > .option--highlighted {
        background: $block-cursor-background;
        color: $block-cursor-foreground;
        text-style: bold;
    }
    ListView > .list-item--highlighted {
        background: $block-cursor-background;
        color: $block-cursor-foreground;
    }
    #session-list,
    #permissions-list {
        height: auto;
        max-height: 20;
        border: none;
        padding: 0;
        scrollbar-size-vertical: 1;
        scrollbar-size-horizontal: 0;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "interrupt", "Interrupt"),
        Binding("ctrl+r", "resume", "Resume"),
        Binding("ctrl+t", "toggle_agent", "Switch agent"),
        Binding("ctrl+o", "models", "Models"),
        Binding("escape", "interrupt_escape", "Interrupt (again = force stop)"),
        Binding("ctrl+p", "settings", "Settings"),
        Binding("ctrl+s", "settings", "Settings"),
        Binding("ctrl+shift+e", "toggle_thought", "Expand/collapse thought"),
        # session routing between parallel sub-agents (opencode's
        # session.parent / session.child.next / session.child.previous /
        # session.child.first). Non-priority so the prompt keeps the arrow keys
        # for cursor movement while it is non-empty.
        Binding("up", "fd_parent", "Parent session", priority=False),
        Binding("left", "fd_prev", "Previous subagent", priority=False),
        Binding("right", "fd_next", "Next subagent", priority=False),
        Binding("ctrl+down", "fd_first", "View subagents"),
        # HOME/END/PgUp/PgDn always scroll the conversation, no matter where
        # focus is (the input box normally eats them for text editing). Prior
        # bindings win over the focused widget, so they work right out of the
        # gate — no need to click a message to hand focus to the chat first.
        Binding("home", "chat_home", "Scroll to top", priority=True),
        Binding("end", "chat_end", "Scroll to bottom", priority=True),
        Binding("pageup", "chat_page_up", "Scroll page up", priority=True),
        Binding("pagedown", "chat_page_down", "Scroll page down", priority=True),
    ]

    def __init__(
        self,
        cfg: Config | None = None,
        engine: AgentLoop | None = None,
        directory: Path | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.cfg = cfg or load_config()
        # Live theme selection: every widget resolves colors via active_theme()
        # at render time, so this one call styles the whole app.
        # Custom themes come from cfg (registered at load) — re-register
        # here too in case cfg was built by hand.
        from .theme import register_custom_themes, set_active_theme

        try:
            register_custom_themes(getattr(self.cfg, "custom_themes", None))
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/app.py:421")
            pass
        set_active_theme(getattr(self.cfg, "theme", "") or "opencode")
        self.directory = directory or Path.cwd()
        # Engine-chain imports (agent.loop, tools, commands, session, auth) are
        # deferred out of module scope so `import opencode_py.tui` (and thus
        # first paint) doesn't pay ~0.4s before the app even exists. The engine
        # is built on demand: a background thread warms it right after on_mount
        # for the real app, and any synchronous first access (tests, sub-agent
        # spawn) builds it inline. `engine=` still injects a prebuilt engine.
        self._engine: AgentLoop | None = engine
        self._engine_lock = threading.Lock()
        # Single dialog queue: permission AND question modals from every agent
        # (parent + parallel children) serialize here, so two engine threads
        # can never stack screens or steal each other's 30s answer window.
        # Each waiter gets its full window once its dialog is actually shown.
        self._dialog_lock = threading.Lock()
        # Force-stop flag: set by the 2nd ESC while work is still running.
        # Dialog waits watch it so a stuck modal can't trap the worker after
        # the user demanded a stop. Cleared once nothing is busy anymore.
        self._force_stop = threading.Event()
        # Reconnect watchers (auto-resume): sid -> stop Event for the
        # background thread watching connectivity after a network-killed turn.
        self._reconnect_watchers: dict[str, threading.Event] = {}
        if engine is not None:
            self._wire_engine(engine)
        from ..globals import Path as GPath
        from ..auth import Auth

        self.auth = Auth(auth_file=GPath.auth_file())
        # Per-session interrupt flags: each session has its own flag so interrupting
        # one session doesn't affect others. Keyed by session_id.
        self._interrupt_flags: dict[str, bool] = {}
        from ..commands import build_registry as build_command_registry

        self.command_registry = build_command_registry()
        from ..session import new_session

        self.session = new_session(
            directory=str(self.directory),
            provider=self.cfg.provider,
            model=self.cfg.model,
            agent=self.cfg.default_agent or "build",
        )
        if engine is not None:
            engine.session_id = self.session.id
        self._chats: dict[str, ChatView] = {}
        self._sessions: dict[str, Any] = {self.session.id: self.session}
        self._engines: dict[str, AgentLoop] = {}
        if engine is not None:
            self._engines[self.session.id] = engine
        self._main_engine: AgentLoop | None = engine
        self._current_session_id = self.session.id
        self._active_turn_session_id = self.session.id
        self._busy = False
        self._busy_sessions: set[str] = set()
        self._running_agents: dict[str, str] = {}
        # Per-session turn bookkeeping. A single global set of had_text/
        # had_reasoning/... flags is WRONG: a sub-agent's events (keyed by the
        # child session id) or a prompt submitted on another session while this
        # one streams would overwrite the active turn's flags, so _turn_done
        # could finalize the wrong bubble / show the wrong runtime. Every
        # event handler writes to the SESSION's slot; _turn_done reads the
        # slot of the turn that actually finished.
        self._turn: dict[str, dict[str, Any]] = {}
        self._esc_presses = 0
        self._esc_timer: Any = None
        # Auto-refocus timer: dragging the prompt cursor back after the focus
        # lands somewhere else (e.g. a tapped reasoning bubble), so typing keeps
        # working without having to tap the input box again.
        self._refocus_timer: Any = None
        self._main_screen: Any = None
        self._pruned: set[str] = set()
        # Deferred renders: child chats created before their widget is mounted
        # (task-row click on a fresh open) render right after the switch.
        self._pending_render: dict[str, bool] = {}
        # Live sub-agent family tree (mirrors the official store): parent_id ->
        # ordered list of child records {id, title, agent, created, status}.
        # Kept even after a child finishes so `(2 of N)` counts stay correct and
        # finished children stay reviewable.
        self._children: dict[str, list[dict[str, Any]]] = {}
        self._child_parent: dict[str, str] = {}
        # Last sub-agent viewed under each parent, so returning to a parent and
        # pressing ctrl+down resumes that same agent (official opencode keeps
        # your previous sub-agent selection per parent).
        self._last_selection: dict[str, str] = {}
        self._task_start: dict[str, float] = {}
        self._usage: dict[str, dict[str, int]] = {}
        # delta batching: text/reasoning deltas are queued (non-blocking) and
        # flushed on a short timer so a fast stream isn't re-rendered per token.
        self._pending: dict[str, dict[str, list[str]]] = {}
        self._pending_bg: dict[str, dict[str, list[str]]] = {}
        self._delta_timer: Any = None
        # Periodic autosave while a turn streams: the engine keeps the in-flight
        # assistant reply live in its history, so this persists the conversation
        # up to the very last token. If Termux/Android kills the app suddenly
        # (no graceful exit), the session file is at most a few seconds stale
        # and the picker resumes where the user left off.
        self._autosave_timer: Any = None
        self._exit_requested = threading.Event()
        # Invalidates any in-flight streaming autosave when its turn ends or the
        # app saves-all (exit/teardown), so a stale worker can never overwrite a
        # newer durable copy. See _autosave_in_flight.
        self._autosave_generation = 0
        self._autosave_thread: threading.Thread | None = None
        self._elapsed_timer: Any = None
        # Streaming auto-voice state: per-session unsaid text, turn generation
        # (stale queued sentences are skipped), spoken-this-turn flags, and a
        # single FIFO worker so sentences talk in order without overlapping.
        self._voice_buf: dict[str, str] = {}
        self._voice_gen: dict[str, int] = {}
        self._voice_spoke: dict[str, bool] = {}
        self._voice_queue: Any = None
        self._voice_worker_started: bool = False

    @property
    def engine(self) -> AgentLoop:
        """The main engine, built lazily on first access.

        Building happens synchronously here (so tests / synchronous handlers see
        a fully-wired engine), and a background thread warmed it after mount for
        the real app — whichever builds first, the lock ensures one instance.
        """
        self._ensure_engine()
        assert self._main_engine is not None
        return self._main_engine

    def _session_interrupt_checker(self, session_id: str) -> Callable[[], bool]:
        """Per-session interrupt hook: each engine reads only its own flag, so
        Ctrl+C on one session doesn't interrupt the others."""
        return lambda: self._interrupt_flags.get(session_id, False)

    def _wire_engine(self, engine: AgentLoop) -> None:
        engine.on_event = self._on_engine_event
        sid = engine.session_id
        engine.interrupt = self._session_interrupt_checker(sid)
        # Permission "ask" mode: bridge the engine thread to a modal dialog.
        # Sub-agents share the same PermissionEngine instance, so one hook works
        # for all sessions.
        engine.permission.ask_callback = self._permission_ask
        # Question "ask" mode: bridge the engine thread's question.ask to a
        # modal dialog, mirroring the official TUI's question popup. Sub-agents
        # share the same QuestionService instance, so one hook works for all.
        engine.question_service.ask_callback = self._question_ask

    def _ensure_engine(self) -> None:
        if self._main_engine is not None:
            return
        with self._engine_lock:
            if self._main_engine is not None:
                return
            from ..agent.loop import AgentLoop
            from ..tools import build_registry as build_tool_registry

            engine = AgentLoop(
                cfg=self.cfg,
                registry=build_tool_registry(self.cfg),
                directory=self.directory,
                auth=self.auth,
                agent=self.cfg.default_agent or "build",
            )
            self._wire_engine(engine)
            engine.session_id = self.session.id
            self._main_engine = engine
            self._engines.setdefault(self.session.id, engine)

    def _warm_engine(self) -> None:
        """Background engine build launched after mount: keeps first paint fast
        while the (~0.4s) engine-chain import runs off the UI thread."""
        # rotation/model caches warm in PARALLEL (own thread, own network) —
        # by the first prompt the lanes + context are hot, not fetched.
        try:
            from ..providers.rotation import warm_startup as _warm
            _warm(self.cfg, getattr(self, "auth", None))
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/app.py:595")
            pass
        try:
            self._ensure_engine()
        except Exception as e:
            # A failed warm-up must not kill the app's message loop; the engine
            # is rebuilt synchronously on first real use anyway.
            sys.stderr.write(f"[tui] engine warm-up failed: {e}\n")
            return
        # Pre-warm the per-model context/output lookups (and their lazy provider
        # imports: zen.py / openai_compat ~180ms) so the FIRST turn isn't held
        # up by a one-time lookup on the request path.
        try:
            from ..providers import model_context_size, model_output_limit

            model_context_size(self.cfg.provider, self.cfg.model, auth=self.auth)
            model_output_limit(self.cfg.provider, self.cfg.model)
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/app.py:612")
            pass
        # Thread-safe schedule via post_message; returns False (no-op) when the
        # app's message pump isn't running yet/anymore instead of dropping a
        # never-awaited coroutine.
        self.call_later(self._update_header)

    def compose(self) -> ComposeResult:
        with Vertical(id="root"):
            with Vertical(id="chat-stack"):
                yield ChatView()
            yield SubagentFooter()
            yield InputBar(
                # Aliases get their own dropdown entry so typing /q, /clear or
                # /continue takes the SAME popup/confirm path as the canonical
                # name — aliases used to be raw-submitted, skipping the Run/
                # Cancel safety net (/q quit instantly with no confirmation).
                commands=self._dropdown_commands()
            )
            yield StatusBar()

    def _dropdown_commands(self) -> list[dict[str, str]]:
        cmds: list[dict[str, str]] = []
        names: set[str] = set()
        for c in self.command_registry.list():
            if c.hidden:
                continue
            custom = (getattr(c, "metadata", {}) or {}).get("custom")
            cmds.append({"name": c.name, "description": c.description, "custom": bool(custom)})
            names.add(c.name)
        for c in self.command_registry.list():
            if c.hidden:
                continue
            for alias in c.aliases:
                if alias in names:
                    continue  # canonical names always win
                names.add(alias)
                cmds.append({"name": alias, "description": c.description})
        return cmds

    def on_mount(self) -> None:
        self._thread_id = threading.get_ident()
        self._main_screen = self.screen
        # Register the Textual design-token theme from the active palette and
        # keep it live: every later set_active_theme() (picker, /theme,
        # Settings) re-applies it so CSS chrome restyles instantly.
        from .theme import set_theme_applier

        self._apply_textual_theme()
        set_theme_applier(self._apply_textual_theme)
        # Give the model eyes: screen_view tool captures THIS rendered screen.
        from ..tools.screen_view import set_capture_fn

        set_capture_fn(self._capture_for_model)
        self._update_header()
        status = self.query_one(StatusBar)
        status.set_directory(str(self.directory))
        self._main_chat = self.query_one(ChatView)
        self._chats[self.session.id] = self._main_chat
        self._footer = self.query_one(SubagentFooter)
        # First open: show the opencode logo banner until the first message
        # starts a real conversation (opencode shows its logo on the launch
        # screen, then it disappears once you begin typing/chatting).
        self._main_chat.show_logo()
        self.query_one(InputBar).focus()
        # Live `↳ Xs` elapsed ticks from app start, not first turn: a
        # background agent can run while the parent turn is idle, and the
        # old lazy start left its row frozen. Cheap no-op when idle.
        if self._elapsed_timer is None:
            try:
                self._elapsed_timer = self.set_interval(1.0, self._refresh_task_elapsed)
            except Exception:
                self._elapsed_timer = None
        # permanence transparency: log exactly which config file is live
        # and warn when layers shadow it — a "vanished" agent is usually a
        # different file/env winning the merge, not lost data.
        try:
            from ..config import config_shadow_warnings
            from ..globals import Path as _GP

            self._config_path = str(_GP.config / "opencode.json")
            for w in config_shadow_warnings(self.cfg):
                self.notify(w, timeout=8, markup=False)
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/app.py:695")
            pass
        if self._main_engine is None:
            # First paint first: the engine chain imports ~0.4s of heavy modules
            # (agent.loop, tools, commands, …) that don't touch the widgets on
            # screen. Build that on a background thread so the frame is up while
            # it warms; _update_header shows cfg defaults until it's ready.
            threading.Thread(target=self._warm_engine, daemon=True).start()

    def _apply_textual_theme(self) -> None:
        """Rebuild + reapply the Textual design-token theme from the active
        palette. Registering a fresh name each time guarantees the reactive
        `theme` setter fires and every $variable-driven CSS rule re-resolves."""
        from .theme import build_textual_theme

        t = build_textual_theme()
        self.register_theme(t)
        self.theme = t.name

    # -- session routing --------------------------------------------------
    _MAX_HIDDEN_CHATS = 8


def run_tui(cfg: Config | None = None, directory: Path | None = None) -> None:
    import os
    import signal

    # The engine chain (agent.loop, tools, commands) and the provider internals
    # (zen model list, OpenAI-compat SSE layer) are ~0.6s of lazy imports that
    # aren't needed until the first Enter. Warm them on a background thread NOW
    # so they overlap app.run()'s one-time compose/layout/first-paint work and
    # the first prompt responds immediately instead of waiting for the chain.
    # (This runs from run_tui, not at module import, so it can't race the app's
    # own imports on the main thread.)
    threading.Thread(target=_prewarm_heavy_deps, daemon=True).start()

    app = OpenCodeTUI(cfg=cfg, directory=directory)

    def _close_save_all(signum: int, frame: Any) -> None:
        """Termux-close / kill save (SIGTERM, SIGHUP).

        ``on_exit_app``/``on_unmount`` only run on a graceful exit. When the
        user closes Termux the process gets a signal instead, so save every
        live conversation synchronously here (best effort), then re-raise the
        default signal so shutdown stays immediate.
        """
        try:
            app._save_all_live_sessions()
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/app.py:743")
            pass
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    try:
        signal.signal(signal.SIGTERM, _close_save_all)
    except (AttributeError, ValueError) as _oc_exc:  # pragma: no cover
        swallow(_oc_exc, "opencode_py/tui/app.py:756")
        pass
    try:
        signal.signal(signal.SIGHUP, _close_save_all)
    except (AttributeError, ValueError) as _oc_exc:  # pragma: no cover
        swallow(_oc_exc, "opencode_py/tui/app.py:760")
        pass
    try:
        app.run()
    finally:
        # Release MCP server processes (and any other engine resources) so a
        # server started for this session isn't left dangling after exit.
        engine = getattr(app, "_main_engine", None)
        if engine is not None:
            close = getattr(engine, "close", None)
            if close is not None:
                try:
                    close()
                except Exception as _oc_exc:  # pragma: no cover - best effort on exit
                    swallow(_oc_exc, "tui/app.py:767")
                    pass


if __name__ == "__main__":
    run_tui()

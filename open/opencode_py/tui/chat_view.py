"""Chat view: scrollable message list mirroring opencode's session screen.

Rendering mirrors opencode's TUI (packages/tui/src/routes/session/index.tsx):

  - User messages are a full-width block with a single left border strip in the
    agent accent color, a `backgroundPanel` fill and padding (no title), plus an
    optional ` QUEUED ` badge when the turn hasn't started yet.
  - Assistant text flows as plain markdown indented from the left with a block
    cursor (▍) while streaming, then a muted `▣ Build · model` mode line.
  - Reasoning streams as a spinner `Thinking...` and collapses to a clickable
    `+ Thought: <title>` line (opencode's hide mode); clicking toggles the body.
  - Tools render as compact inline rows (`{icon} {label}`, spinner while running)
    or, for tools that produce a result block (bash output, edit diff, todos,
    questions, apply-patch), a subtle left-bordered block on the panel background.
    Per-tool rendering mirrors opencode: Read shows `↳ Loaded <file>`, Glob/Grep
    show `(N matches)`, etc.
"""

from __future__ import annotations

from ..util.log import swallow
import time
from typing import Any

from textual.containers import VerticalScroll
from textual.reactive import reactive
from textual.widgets import Static

from .logo import opencode_logo_text
from .input_bar import SessionNavRequested

# tool -> icon. Mirrors opencode's InlineTool usage.
TOOL_ICONS = {
    "bash": "$",
    "shell": "$",
    "execute": "$",
    "background_task": "$",
    "read": "→",
    "write": "←",
    "glob": "✱",
    "grep": "✱",
    "find_symbols": "⎆",
    "screen_view": "◉",
    "device": "⚡",
    "history_search": "◷",
    "checkpoint": "⟲",
    "webfetch": "✦",
    "webfetch_many": "⟡",
    "websearch": "◈",
    "webreach": "◉",
    "attach": "◉",
    "edit": "←",
    "apply_patch": "✦",
    "todowrite": "☰",
    "task": "│",
    "question": "→",
    "skill": "→",
    "lsp": "⌁",
    "speak": "♪",
    "mcp": "⊙",
    "notify": "·",
}
# tool -> label used in inline rows / titles.
TOOL_NAMES = {
    "bash": "Shell",
    "shell": "Shell",
    "execute": "Execute",
    "background_task": "Background",
    "read": "Read",
    "write": "Write",
    "glob": "Glob",
    "grep": "Grep",
    "find_symbols": "Symbols",
    "screen_view": "Screen",
    "device": "Device",
    "history_search": "History",
    "checkpoint": "Checkpoint",
    "webfetch": "WebFetch",
    "webfetch_many": "WebFetch Batch",
    "websearch": "WebSearch",
    "webreach": "WebReach",
    "attach": "Attach Image",
    "edit": "Edit",
    "apply_patch": "Apply Patch",
    "todowrite": "TodoWrite",
    "task": "Task",
    "skill": "Skill",
    "lsp": "LSP",
    "speak": "Speak",
}

# tools that have a dedicated renderer in opencode (everything else is "generic").
# tool -> display name for the live `↳ Bash npm run test` line under a running
# task row (mirrors opencode's `titlecase(tool)` in the Subagent component).
# MessageBubble moved out: 68 methods, six responsibilities. Composed here so the
# class stays a single Textual widget — events.py and sessions.py do
# `chat.query(MessageBubble)`, which needs one identity.
#
# Re-exported so every existing `from .chat_view import X` keeps resolving:
# the mixins (needed below) plus the names this module used to define.
# These ARE consumed -- tests/test_chat_bubble_split.py imports them from here
# to prove the split left nothing unimportable, so the re-exports are a real
# contract, not dead weight.
from .chat_bubble import (  # noqa: E402,F401
    BubbleCoreMixin,
    BubbleFrameMixin,
    BubbleRefreshMixin,
    BubbleStreamMixin,
    BubbleTaskMixin,
    BubbleToolMixin,
    OpenTaskSession,
    _format_input,
    _plain,
    _render_diff,
    _safe_input,
    _tool_display,
    collapse_tool_output,
    tool_label,
    tool_title,
)
from .chat_parts import reasoning_summary  # noqa: E402  (ChatView still uses it)


class MessageBubble(
    BubbleCoreMixin,
    BubbleToolMixin,
    BubbleTaskMixin,
    BubbleFrameMixin,
    BubbleRefreshMixin,
    BubbleStreamMixin,
    Static,
):
    """One chat element (user, assistant, reasoning, mode-line, meta) or a tool run.

    The behaviour lives in the mixins above (see chat_bubble.py for what each
    owns). Nothing was rewritten: every method moved verbatim, which is why this
    class is now just reactives plus the composition order.
    """

    queued: reactive = reactive(False)  # user message waiting for the turn to start
    streaming: reactive = reactive(False)
    expanded: reactive = reactive(False)  # reasoning body collapsed/expanded
    selected: reactive = reactive(False)  # this task row is the active sub-agent
    frozen: reactive = reactive(False)  # finished bubble: skip re-render (same pixels, ~0 cost)

class ChatView(VerticalScroll):
    """Scrollable list of message bubbles + streaming cursor."""

    messages: reactive = reactive([])
    streaming: reactive = reactive(False)
    streaming_text: reactive = reactive("")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._stream_bubble: MessageBubble | None = None
        self._reasoning_bubble: MessageBubble | None = None
        self._compaction_bubble: MessageBubble | None = None
        # Queued user bubbles in submission order (mirrors opencode's FIFO
        # prompt queue). The engine clears them one-by-one as it folds each
        # prompt into the running turn.
        self._queued_bubbles: list[MessageBubble] = []
        # True while the user is reading the newest output (at the bottom).
        # When they scroll up to re-read history, we stop yanking the view back
        # down on every stream delta / tool update.
        self._follow_bottom = True
        # True when this chat belongs to a sub-agent session: the arrow keys
        # navigate the session tree instead of scrolling (so ↑ inside a
        # sub-agent always returns to the parent, even when the chat — not the
        # prompt — has focus).
        self._session_is_child = False
        # First-open welcome banner (ASCII opencode logo). Mounted only when
        # the chat has no messages/stream yet, removed once the conversation
        # starts.
        self._welcome_logo: Static | None = None
        # O(1) bubble lookups: the hot paths (find_tool per tool event,
        # find_task per sub-agent event, last_reasoning per toggle) used to
        # scan every mounted bubble on EVERY event — O(bubbles) per token-adjacent
        # update. Indexes are validated (still mounted) on hit, rebuilt on miss.
        self._tool_index: dict[str, MessageBubble] = {}
        self._task_index: dict[str, MessageBubble] = {}
        self._last_reasoning_bubble: MessageBubble | None = None
        self._history_pending: list = []
        self._history_loading: bool = False
        self._history_anchor: Any = None
        self._history_session_id: str = ""
        self._history_chunk: int = 80
        # Live-window hides (display:none, same pill as resume): oldest
        # bubbles past chat_live_window stay mounted-but-hidden so scroll
        # and layout stay O(window). Rebuilt by _trim_live_window.
        self._hidden_count: int = 0

    # -- first-open banner -------------------------------------------------
    def show_logo(self) -> None:
        """Show the opencode logo when the chat is empty (first launch)."""
        if self._welcome_logo is not None:
            return
        logo = Static(opencode_logo_text(), classes="chat-welcome-logo")
        self.mount(logo)
        self._welcome_logo = logo
        try:
            self.scroll_home(animate=False)
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/chat_view.py:1594")
            pass

    def _dismiss_logo(self) -> None:
        """Remove the first-open logo once the conversation begins."""
        logo = self._welcome_logo
        self._welcome_logo = None
        if logo is not None:
            try:
                logo.remove()
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/chat_view.py:1604")
                pass


    # -- key routing -------------------------------------------------------
    async def on_key(self, event: Any) -> None:
        if getattr(self, "_session_is_child", False) and event.key in ("up", "left", "right"):
            event.stop()
            event.prevent_default()
            self.post_message(SessionNavRequested(event.key))
        # Not intercepted: let the key bubble to the App's binding dispatch
        # (scroll bindings, etc.). There is no base `on_key` message handler
        # to chain to, so just return.

    # -- scrolling ---------------------------------------------------------
    def _auto_scroll(self, throttled: bool = False) -> None:
        """Scroll to the newest message, unless the user scrolled up to read
        earlier conversation (opencode keeps your position while the model
        keeps streaming/tool-running below). Mid-stream calls pass
        throttled=True: scroll_end forces a full layout, so scrolls stay
        smooth and fast while tokens flow (final positions still land via
        the unthrottled end_stream/end_reasoning calls).

        Bottom-follow is sticky AND cheap: when already at the bottom we
        skip the layout when nothing grew (same max_scroll_y as last time),
        so 10 calls/s cost ~0 layouts instead of 10. ponytail: 0.05s floor
        + growth-gate; upgrade path is Textual's own smooth-scroll if it
        ever lands on phone builds.
        """
        if not self._follow_bottom:
            return
        if throttled:
            try:
                now = time.monotonic()
                last = getattr(self, "_last_scroll_mono", 0.0)
                if now - last < 0.05:
                    return
                try:
                    mx = float(self.max_scroll_y)
                except Exception:
                    mx = -1.0
                if mx >= 0 and mx == float(getattr(self, "_last_scroll_max", -1.0)):
                    return  # nothing grew: skip the layout entirely
                self._last_scroll_mono = now
                try:
                    self._last_scroll_max = mx
                except Exception as _oc_exc:
                    swallow(_oc_exc, "tui/chat_view.py:1650")
                    pass
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/chat_view.py:1652")
                pass
        else:
            try:
                self._last_scroll_max = float(self.max_scroll_y)
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/chat_view.py:1657")
                pass
        self.scroll_end(animate=False)

    def watch_scroll_y(self, old_value: float, new_value: float) -> None:
        super().watch_scroll_y(old_value, new_value)
        try:
            self._follow_bottom = new_value >= self.max_scroll_y - 1
        except Exception:
            self._follow_bottom = True
        try:
            self._maybe_prefetch()
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/chat_view.py:1669")
            pass

    def set_history_pending(self, pending: list, session_id: str = "", chunk: int = 80) -> None:
        self._history_pending = list(pending or [])
        self._history_loading = False
        self._history_session_id = session_id or ""
        self._history_chunk = int(chunk or 80)
        self._update_history_pill()

    def _history_pill_text(self) -> str:
        try:
            n = len(getattr(self, "_history_pending", []) or [])
        except Exception:
            n = 0
        try:
            hidden = int(getattr(self, "_hidden_count", 0) or 0)
        except Exception:
            hidden = 0
        # Live-trimmed bubbles outrank resume-pending: hidden are already
        # mounted (display:none, unhide is a style flip) while pending still
        # needs a disk load — but the pill text is shared.
        total = hidden if hidden > 0 else n
        if total <= 0:
            return ""
        return f"↑ {total} older messages — scroll up to load"

    def _history_count(self) -> int:
        """Pill number without touching the DOM (headless/test safe)."""
        try:
            n = len(getattr(self, "_history_pending", []) or [])
        except Exception:
            n = 0
        try:
            hidden = int(getattr(self, "_hidden_count", 0) or 0)
        except Exception:
            hidden = 0
        return hidden if hidden > 0 else n

    def _update_history_pill(self) -> None:
        try:
            anchor = getattr(self, "_history_anchor", None)
            n = len(getattr(self, "_history_pending", []) or [])
            try:
                hidden = int(getattr(self, "_hidden_count", 0) or 0)
            except Exception:
                hidden = 0
            if n <= 0 and hidden <= 0:
                if anchor is not None:
                    try:
                        anchor.remove()
                    except Exception as _oc_exc:
                        swallow(_oc_exc, "tui/chat_view.py:1720")
                        pass
                    self._history_anchor = None
                return
            text = self._history_pill_text()
            if anchor is None or not getattr(anchor, "is_attached", False):
                bubble = MessageBubble("meta", text)
                try:
                    kids = list(self.children)
                    if kids:
                        self.mount(bubble, before=kids[0])
                    else:
                        self.mount(bubble)
                except Exception:
                    try:
                        self.mount(bubble)
                    except Exception as _oc_exc:
                        swallow(_oc_exc, "tui/chat_view.py:1736")
                        return
                self._history_anchor = bubble
            else:
                try:
                    anchor._message = text
                    anchor._refresh()
                except Exception as _oc_exc:
                    swallow(_oc_exc, "tui/chat_view.py:1743")
                    pass
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/chat_view.py:1745")
            pass

    def _live_window(self) -> int:
        """Max mounted message bubbles on this chat (0 = unlimited)."""
        try:
            cfg = getattr(getattr(self, "app", None), "cfg", None)
            w = int(getattr(cfg, "chat_live_window", 120) or 0)
            return max(0, w)
        except Exception:
            return 120

    def _visible_bubbles(self) -> list:
        try:
            return [c for c in self.query(MessageBubble) if getattr(c, "display", "") != "none"]
        except Exception:
            return []

    def _trim_live_window(self) -> None:
        """Hide oldest bubbles past the live window (display:none, not remove).

        Same pill/scroll-up flow as the resume path, but for live growth:
        mounting stays O(window), unhide is a style flip, and every hot path
        (find_tool/find_task/last_reasoning/tool_runs) already validates
        membership so a display:none bubble is skipped, never stale.
        """
        try:
            window = self._live_window()
            if window <= 0:
                return
            visible = self._visible_bubbles()
            over = len(visible) - window
            if over <= 0:
                return
            # Never hide the streaming / in-flight rows: newest-first scan
            # skips stream/reasoning bubbles and in-progress tool runs.
            hide: list = []
            for b in visible:
                if len(hide) >= over:
                    break
                try:
                    if getattr(b, "streaming", False):
                        continue
                    content = getattr(b, "content", None)
                    if isinstance(content, dict) and content.get("status") in ("running", "pending"):
                        continue
                except Exception as _oc_exc:
                    swallow(_oc_exc, "tui/chat_view.py:1791")
                    pass
                hide.append(b)
                if len(hide) >= over:
                    break
            for b in hide:
                try:
                    b.display = "none"
                except Exception as _oc_exc:
                    swallow(_oc_exc, "tui/chat_view.py:1799")
                    pass
            self._hidden_count = int(getattr(self, "_hidden_count", 0) or 0) + len(hide)
            self._update_history_pill()
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/chat_view.py:1803")
            pass

    def _unhide_older(self, count: int = 0) -> int:
        """Unhide hidden bubbles oldest-first (scroll-up loads them back)."""
        try:
            kids = [c for c in self.children if c.__class__.__name__ == "MessageBubble"
                    and getattr(c, "display", "") == "none"]
            if not kids:
                self._hidden_count = 0
                self._update_history_pill()
                return 0
            try:
                chunk = int(getattr(self, "_history_chunk", 80) or 80)
            except Exception:
                chunk = 80
            take = min(len(kids), count if count > 0 else chunk * 2)
            for b in kids[:take]:
                try:
                    b.display = "block"
                except Exception as _oc_exc:
                    swallow(_oc_exc, "tui/chat_view.py:1823")
                    pass
            left = len(kids) - take
            self._hidden_count = max(0, left)
            self._update_history_pill()
            return take
        except Exception:
            return 0

    def _maybe_prefetch(self) -> None:
        # Live-hidden bubbles unhide first (already mounted, zero layout
        # cost); only when none remain does disk-backed pending load.
        try:
            if int(getattr(self, "_hidden_count", 0) or 0) > 0:
                self._unhide_older()
                return
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/chat_view.py:1839")
            pass
        pending = getattr(self, "_history_pending", None) or []
        if not pending or getattr(self, "_history_loading", False):
            return
        try:
            y = float(self.scroll_y)
            mx = float(self.max_scroll_y)
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/chat_view.py:1847")
            return
        if mx <= 0:
            return
        if y > 400 and y > mx * 0.3:
            return
        self._load_older()

    def _load_older(self) -> None:
        # Live-hidden bubbles (same session) unhide before disk pending loads.
        try:
            if int(getattr(self, "_hidden_count", 0) or 0) > 0:
                if self._unhide_older() > 0:
                    return
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/chat_view.py:1861")
            pass
        pending = getattr(self, "_history_pending", None) or []
        if not pending or getattr(self, "_history_loading", False):
            return
        self._history_loading = True
        try:
            old_max = float(self.max_scroll_y)
        except Exception:
            old_max = 0.0
        try:
            old_y = float(self.scroll_y)
        except Exception:
            old_y = 0.0
        try:
            chunk = int(getattr(self, "_history_chunk", 80) or 80)
        except Exception:
            chunk = 80
        take = min(len(pending), chunk * 2)
        if take <= 0:
            self._history_loading = False
            return
        batch = pending[:take]
        del pending[:take]
        self._history_pending = pending
        batch.reverse()
        try:
            anchor = getattr(self, "_history_anchor", None)
            kids = list(self.children)
            if anchor is not None and anchor in kids:
                idx = kids.index(anchor)
                first = kids[idx + 1] if idx + 1 < len(kids) else None
            else:
                first = kids[0] if kids else None
            if first is None:
                for fn in batch:
                    try:
                        fn(None)
                    except Exception as _oc_exc:
                        swallow(_oc_exc, "tui/chat_view.py:1899")
                        continue
            else:
                for fn in batch:
                    try:
                        fn(first)
                    except Exception:
                        try:
                            fn(None)
                        except Exception as _oc_exc:
                            swallow(_oc_exc, "tui/chat_view.py:1908")
                            continue
        finally:
            self._update_history_pill()
            try:
                def _fix() -> None:
                    try:
                        new_max = float(self.max_scroll_y)
                        self.scroll_to(y=old_y + max(0.0, new_max - old_max), animate=False)
                    except Exception as _oc_exc:
                        swallow(_oc_exc, "tui/chat_view.py:1917")
                        pass
                    self._history_loading = False
                    try:
                        self._maybe_prefetch()
                    except Exception as _oc_exc:
                        swallow(_oc_exc, "tui/chat_view.py:1922")
                        pass
                try:
                    self.call_after_refresh(_fix)
                except Exception:
                    self.call_later(_fix)
            except Exception:
                self._history_loading = False

    def append_user(self, text: str, agent: str = "build", queued: bool = False, before: Any = None, scroll: bool = True) -> MessageBubble:
        bubble = MessageBubble("user", text, agent=agent)
        bubble.queued = queued
        if queued:
            self._queued_bubbles.append(bubble)
        self._mount_bubble(bubble, before=before)
        if scroll:
            self._auto_scroll()
        return bubble

    def append_directive(self, text: str, title: str = "", agent: str = "") -> MessageBubble:
        """The parent's instruction rendered at the top of a sub-agent's chat
        (official opencode shows the task directive as the first message)."""
        self._dismiss_logo()
        bubble = MessageBubble("user", text, directive=title or "Task")
        if agent.strip():
            bubble.agent = agent.strip()
        bubble.add_class("directive")
        self.mount(bubble)
        self._auto_scroll()
        return bubble

    def promote_next_queued(self) -> str | None:
        """Clear the ` QUEUED ` badge on the oldest queued message.

        Called when the engine folds that prompt into the running turn
        (opencode's Session Drain) — the message keeps its place in the chat,
        just loses its badge. Returns the promoted text.
        """
        if not self._queued_bubbles:
            return None
        bubble = self._queued_bubbles.pop(0)
        try:
            bubble.queued = False
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/chat_view.py:1965")
            pass
        return str(bubble.content)


    def _mount_bubble(self, bubble: Any, before: Any = None) -> Any:
        self._dismiss_logo()
        try:
            if before is not None and getattr(before, "is_attached", False):
                self.mount(bubble, before=before)
            else:
                self.mount(bubble)
        except Exception:
            try:
                self.mount(bubble)
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/chat_view.py:1980")
                pass
        try:
            if getattr(bubble, "role", "") == "tool":
                self._index_tool_bubble(bubble)
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/chat_view.py:1985")
            pass
        # Live chats that grow past the window trim here — same pill flow as
        # resume, so a 500-message session never holds 500 live layouts.
        try:
            if before is None:
                self._trim_live_window()
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/chat_view.py:1992")
            pass
        return bubble

    def clear(self) -> None:
        """Drop every bubble and reset the streaming cursors. Used when the
        main session is deleted and the workspace resets to a fresh one."""
        self._stream_bubble = None
        self._reasoning_bubble = None
        self._follow_bottom = True
        self._queued_bubbles = []
        self._history_pending = []
        self._history_loading = False
        self._history_anchor = None
        self._hidden_count = 0
        for child in list(self.children):
            try:
                child.remove()
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/chat_view.py:2010")
                pass
        try:
            self.scroll_home(animate=False)
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/chat_view.py:2014")
            pass
        # A removed-but-unflushed logo widget may still be in `children` (Textual
        # defers DOM removal), so drop the reference: the banner belongs to the
        # app's very first open, not to a workspace reset.
        self._welcome_logo = None

    def append_assistant(self, text: str, before: Any = None, scroll: bool = True) -> None:
        self._mount_bubble(MessageBubble("assistant", text), before=before)
        if scroll:
            self._auto_scroll()

    def append_meta(self, text: str, before: Any = None, scroll: bool = True) -> None:
        """Persistent command/system output (e.g. /models, /help, /config)."""
        self._mount_bubble(MessageBubble("meta", text), before=before)
        if scroll:
            self._auto_scroll()

    def append_compaction(self, summary: str, before: Any = None, scroll: bool = True) -> None:
        """Render a ` Compaction ` divider + markdown summary (opencode's
        compaction part).

        Ends any in-flight reasoning/stream bubble first so the divider lands
        between the finished text and whatever the model continues after the
        summarized history is replayed.
        """
        self.end_reasoning()
        self.remove_last_stream_bubble()
        self._mount_bubble(MessageBubble("compaction", summary), before=before)
        if scroll:
            self._auto_scroll()

    def begin_compaction_stream(self) -> None:
        """Start the live `▸ Compacted summary` bubble before the summary
        model streams; deltas append via stream_compaction_delta."""
        self._dismiss_logo()
        self.end_reasoning()
        self.remove_last_stream_bubble()
        bubble = MessageBubble("compaction", "", streaming=True)
        self._compaction_bubble = bubble
        self.mount(bubble)
        self._auto_scroll()

    def stream_compaction_delta(self, text: str) -> None:
        """Append one live summary token to the streaming compaction bubble."""
        bubble = self._compaction_bubble
        if bubble is None:
            self.begin_compaction_stream()
            bubble = self._compaction_bubble
        bubble.append_stream(text)
        self._auto_scroll()

    def end_compaction_stream(self, summary: str) -> None:
        """Finalize the live compaction bubble into the ` Compaction ` divider.

        The live bubble was rendering plain text while the summary streamed;
        on `compacted` we re-render the FULL final summary as markdown inside
        the divider (same look as the non-live path). An empty outcome is never
        silently dropped — a muted note keeps the failure visible."""
        bubble = self._compaction_bubble
        self._compaction_bubble = None
        if bubble is None:
            # No live bubble (compaction failed before the first delta) — fall
            # back to the static divider so the outcome is still visible.
            if summary:
                self.append_compaction(summary)
            return
        if not summary and not bubble.content:
            bubble.remove()
            self.mount(MessageBubble("meta", "(no summary returned)"))
            self._auto_scroll()
            return
        bubble._message = summary or bubble.content
        bubble.streaming = False  # re-render as the finished ` Compaction ` divider
        bubble._refresh()

    def append_tool(self, tool_run: dict[str, Any], before: Any = None, scroll: bool = True) -> None:
        bubble = MessageBubble("tool", tool_run)
        if tool_run.get("status") == "running":
            bubble._start_spinner()
        self._mount_bubble(bubble, before=before)
        if scroll:
            self._auto_scroll()

    def _thoughts_visible(self) -> bool:
        """Whether thought bubbles may render (Thinking-picker on/off).

        Reads live config off the running app so a toggle applies instantly
        without rebuilding chats. Defaults True when no app/config exists
        (headless tests, harness mounts) — existing behavior untouched.
        """
        try:
            cfg = getattr(getattr(self, "app", None), "cfg", None)
            if cfg is None:
                return True
            return bool(getattr(cfg, "show_thoughts", True))
        except Exception:
            return True

    def append_reasoning(self, text: str, seconds: float | None = None, before: Any = None, scroll: bool = True) -> None:
        """Render a finished `+ Thought` bubble (used when replaying saved
        history; live thoughts stream through begin_thinking/stream_reasoning)."""
        bubble = MessageBubble("reasoning", text)
        bubble._thought_seconds = seconds
        bubble.visible = self._thoughts_visible()
        self._mount_bubble(bubble, before=before)
        self._last_reasoning_bubble = bubble
        if scroll:
            self._auto_scroll()

    def set_thoughts_visible(self, visible: bool) -> int:
        """Show/hide every thought bubble in this chat. Returns count."""
        n = 0
        try:
            for b in self.reasoning_bubbles():
                try:
                    b.visible = visible
                    n += 1
                except Exception as _oc_exc:
                    swallow(_oc_exc, "tui/chat_view.py:2132")
                    continue
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/chat_view.py:2134")
            pass
        return n

    def begin_stream(self) -> None:
        self._dismiss_logo()
        self._stream_bubble = MessageBubble("assistant", "")
        self._stream_bubble.streaming = True
        self.mount(self._stream_bubble)
        self._auto_scroll()

    def stream_delta(self, text: str) -> None:
        if self._stream_bubble is None:
            self.begin_stream()
        self._stream_bubble.append_stream(text)
        self._auto_scroll(throttled=True)

    def stream_reasoning_delta(self, text: str) -> None:
        self._dismiss_logo()
        if self._reasoning_bubble is None:
            bubble = MessageBubble("reasoning", "")
            bubble.streaming = True
            bubble._start_spinner()
            bubble._thought_started = time.monotonic()
            bubble.visible = self._thoughts_visible()
            # chronological order: each new thought mounts below the previous
            # tool runs (not above a stale empty stream bubble)
            self.mount(bubble)
            self._reasoning_bubble = bubble
            self._last_reasoning_bubble = bubble
        self._reasoning_bubble.append_stream(text)
        self._auto_scroll(throttled=True)

    def begin_thinking(self) -> None:
        """Mount an eager `Thinking...` bubble the moment a turn starts, before
        the first token arrives, so the UI reacts instantly to Enter (mirrors
        opencode). Real reasoning deltas stream into this same bubble; if no
        reasoning ever arrives, end_reasoning drops the empty placeholder."""
        self._dismiss_logo()
        if self._reasoning_bubble is not None:
            return
        bubble = MessageBubble("reasoning", "")
        bubble.streaming = True
        bubble._start_spinner()
        bubble._thought_started = time.monotonic()
        self.mount(bubble)
        self._reasoning_bubble = bubble
        self._last_reasoning_bubble = bubble
        self._auto_scroll()

    def end_reasoning(self) -> None:
        if self._reasoning_bubble is not None:
            if not self._reasoning_bubble.content:
                # the eager placeholder produced no actual reasoning — remove
                # it instead of leaving a misleading empty `Thought` line
                bubble = self._reasoning_bubble
                self._reasoning_bubble = None
                bubble._stop_spinner()
                try:
                    bubble.remove()
                except Exception as _oc_exc:
                    swallow(_oc_exc, "tui/chat_view.py:2194")
                    pass
            else:
                self._reasoning_bubble.end_reasoning()
                self._reasoning_bubble = None
        self._auto_scroll()

    def end_stream(self, text: str = "") -> None:
        """Finalize the streaming cursor. An assistant bubble that never
        received any text is DROPPED instead of left as a blank line — e.g. a
        thought + tool turn whose final "summary" streamed an empty chunk."""
        if self._stream_bubble is not None:
            if text:
                try:
                    self._stream_bubble.flush_stream()
                except Exception as _oc_exc:
                    swallow(_oc_exc, "tui/chat_view.py:2209")
                    pass
                self._stream_bubble.update_text(text)
            else:
                # No authoritative final text: the streamed deltas are the only
                # copy of the reply. Folding them in is REQUIRED — leaving them
                # in _stream_parts silently drops the tail of every message.
                try:
                    self._stream_bubble.flush_stream()
                except Exception as _oc_exc:
                    swallow(_oc_exc, "tui/chat_view.py:end_stream flush")
            if self._stream_bubble.content:
                self._stream_bubble.streaming = False
                self._stream_bubble._refresh()
            else:
                try:
                    self._stream_bubble.remove()
                except Exception as _oc_exc:
                    swallow(_oc_exc, "tui/chat_view.py:2218")
                    pass
            self._stream_bubble = None
        self._auto_scroll()

    def remove_last_stream_bubble(self) -> None:
        """Remove the empty streaming bubble left behind when there's no reply.

        A bubble that already holds partial text is KEPT (only its streaming
        cursor is ended) so a mid-stream error doesn't discard model output.
        """
        target = self._stream_bubble
        if target is None:
            target = None
            # The only empty assistant bubbles are streaming-cursor leftovers,
            # which always sit in the trailing assistant run at the bottom of
            # the chat. Walk just that run instead of a full container query
            # (O(call) → O(trailing run) per remove).
            for child in reversed(self.children):
                try:
                    role = child.role
                except Exception as _oc_exc:
                    swallow(_oc_exc, "tui/chat_view.py:2239")
                    break
                if role != "assistant":
                    break
                if not child.content and not child.streaming:
                    target = child
                    break
        if target is not None:
            if target.content:
                # Same as end_reasoning: _stream_parts is the only copy of the
                # text, and `content` just read it instead of _message.
                target.flush_stream()
                target.streaming = False
            else:
                try:
                    target.remove()
                except Exception as _oc_exc:
                    swallow(_oc_exc, "tui/chat_view.py:2252")
                    pass
        self._stream_bubble = None
        self._auto_scroll()

    def _indexed(self, bubble: MessageBubble | None) -> MessageBubble | None:
        """Return the bubble if still mounted here, else None (stale index)."""
        if bubble is None:
            return None
        try:
            return bubble if bubble.parent is self else None
        except Exception:
            return None

    def _index_tool_bubble(self, bubble: MessageBubble) -> None:
        try:
            content = bubble.content
            cid = content.get("call_id") if isinstance(content, dict) else ""
            if cid:
                self._tool_index[str(cid)] = bubble
            if isinstance(content, dict) and content.get("tool") == "task":
                meta = content.get("metadata")
                sid = meta.get("sessionId") if isinstance(meta, dict) else ""
                if sid:
                    self._task_index[str(sid)] = bubble
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/chat_view.py:2277")
            pass

    def note_tool_metadata(self, bubble: MessageBubble) -> None:
        """Re-index after set_tool_metadata (e.g. task row gains sessionId)."""
        self._index_tool_bubble(bubble)

    def find_tool(self, tool: str, call_id: str = "") -> MessageBubble | None:
        if call_id:
            hit = self._indexed(self._tool_index.get(str(call_id)))
            if hit is not None:
                return hit
        candidates = []
        for child in self.query(MessageBubble):
            if child.role == "tool" and child.content.get("tool") == tool:
                candidates.append(child)
        if not candidates:
            return None
        if not call_id:
            # no call id to match: the most RECENT row is the one still being
            # updated by the engine; candidates[0] stamped running/completed
            # status onto the OLDEST same-name row instead
            return candidates[-1]
        for child in candidates:
            if child.content.get("call_id") == call_id:
                self._tool_index[str(call_id)] = child
                return child
        return None

    def find_task(self, session_id: str) -> MessageBubble | None:
        """The task row whose metadata.sessionId matches a child session id.
        Mirrors the reverse lookup opencode does when a sub-agent's events
        bubble up to the parent chat."""
        hit = self._indexed(self._task_index.get(str(session_id)))
        if hit is not None:
            return hit
        for child in self.query(MessageBubble):
            if child.role != "tool" or child.content.get("tool") != "task":
                continue
            meta = child.content.get("metadata")
            if isinstance(meta, dict) and meta.get("sessionId") == session_id:
                self._task_index[str(session_id)] = child
                return child
        return None

    def tool_runs(self) -> list[dict[str, Any]]:
        """Every tool run currently shown in this chat (opencode's per-session
        tool-part store; the Subagent parent row reads the child's to show the
        live current tool / toolcall count)."""
        return [
            child.content
            for child in self.query(MessageBubble)
            if child.role == "tool" and isinstance(child.content, dict)
        ]

    def update_tool_bubble(self, tool_run: dict[str, Any]) -> None:
        bubble = self.find_tool(tool_run.get("tool", ""), tool_run.get("call_id", ""))
        if bubble:
            bubble.update_tool(tool_run)
            self._auto_scroll()
            return True
        return False

    def last_reasoning(self) -> MessageBubble | None:
        hit = self._indexed(self._last_reasoning_bubble)
        if hit is not None:
            return hit
        found = None
        for child in self.query(MessageBubble):
            if child.role == "reasoning":
                found = child
        self._last_reasoning_bubble = found
        return found

    def toggle_last_reasoning(self) -> None:
        bubble = self.last_reasoning()
        if bubble is not None:
            bubble.expanded = not bubble.expanded
            bubble.focus()

    def reasoning_bubbles(self) -> list[MessageBubble]:
        """Every thought bubble in this chat, oldest first. Never raises."""
        try:
            return [c for c in self.query(MessageBubble) if c.role == "reasoning"]
        except Exception:
            return []

    def set_all_reasoning(self, expanded: bool) -> int:
        """Expand/collapse every thought bubble. Returns the count changed."""
        n = 0
        try:
            for b in self.reasoning_bubbles():
                try:
                    if b.expanded != expanded:
                        b.expanded = expanded
                        n += 1
                except Exception as _oc_exc:
                    swallow(_oc_exc, "tui/chat_view.py:2373")
                    continue
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/chat_view.py:2375")
            pass
        return n

    def last_reasoning_text(self) -> str:
        """Plain text of the newest thought (title + body), or ''."""
        try:
            bubble = self.last_reasoning()
            if bubble is None:
                return ""
            summary = reasoning_summary(bubble.content if isinstance(bubble.content, str) else bubble._message)
            title = str(summary.get("title") or "")
            body = str(summary.get("body") or summary.get("text") or bubble._message or "")
            text = (f"{title}\n{body}".strip() if title else body.strip())
            return text
        except Exception:
            return ""

    def watch_messages(self, value: list) -> None:
        self.refresh()

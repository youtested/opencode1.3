"""Message bubble: the 1,211-line MessageBubble class split by responsibility.

MessageBubble drew a user message, assistant markdown, reasoning, tool output,
diffs, todos, questions, sub-agent task rows and the streaming cursor — 68
methods in one class, so any edit risked the other 67. It is now six mixins,
composed back into one class in chat_view.py.

Splitting rather than extracting keeps three things intact: the class identity
(`chat.query(MessageBubble)` in events.py and sessions.py depends on it), the
reactive attributes, and `self` — so no method body needed to change.

Mixin boundaries are contiguous regions of the original class, and every method
moved verbatim by line span.

    BubbleCoreMixin      construction, watchers, markdown/reasoning bodies
    BubbleToolMixin      one renderer per tool plus their shared row helpers
    BubbleTaskMixin      sub-agent task rows and their helpers
    BubbleFrameMixin     execute/skill/generic rows and the shared frame builder
    BubbleRefreshMixin   tool result block, refresh, freeze and the spinner tick
    BubbleStreamMixin    mouse/keyboard handlers and the streaming cursor

Dependency direction is strictly downward (chat_view -> chat_bubble ->
chat_parts), so there is no cycle.
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any

from rich.console import Group, RenderableType
from rich.text import Text
from textual.message import Message

from .. import errors
from ..util.log import swallow
from .chat_parts import reasoning_summary
from .diff_renderer import render_diff
from .markdown_renderer import render_markdown
from .theme import active_theme

if TYPE_CHECKING:
    # The composed classes live in chat_view, which imports this module. Only
    # annotations need them here; the two runtime checks import lazily inside the
    # method, which is the only way to break that cycle.
    from .chat_view import ChatView

SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
SPINNER_INTERVAL = 0.08  # opencode uses 80ms


class OpenTaskSession(Message):
    """A completed task tool row was clicked; open that sub-session."""

    def __init__(self, sid: str) -> None:
        super().__init__()
        self.sid = sid


def _plain(content: Any, width: int | None = None) -> RenderableType:
    """Plain flowing markdown with no surrounding box (assistant text)."""
    return render_markdown(str(content), width=width)


def _render_diff(
    diff_text: str,
    filepath: str = "",
    width: int | None = None,
    opts: dict[str, Any] | None = None,
) -> RenderableType:
    """Render a unified diff the way opencode's `<diff>` edit block does.

    Includes a line-number gutter, +/- signs and syntax-highlighted content
    (all matched to the official opencode dark theme). ``opts`` mirrors the
    official diff config: ``diff_style`` (``"split"``/``"stacked"``),
    ``diff_wrap_mode`` (``"word"``/``"none"``) and ``suppress_backgrounds``.
    """
    opts = opts or {}
    style = opts.get("diff_style", "split")
    # opencode chooses split only when the terminal is wider than 120 cols and
    # the diff_style config has not forced "stacked".
    view = "auto" if style == "split" else "unified"
    return render_diff(
        diff_text,
        filename=filepath,
        view=view,
        width=width or 0,
        wrap=opts.get("diff_wrap_mode", "word"),
        suppress_backgrounds=opts.get("suppress_backgrounds", False),
    )


def collapse_tool_output(output: str, max_lines: int, max_chars: int) -> dict:
    """Mirror opencode's collapse-tool-output: cap lines and chars with '…'."""
    lines = output.split("\n")
    if len(lines) <= max_lines and len(output) <= max_chars:
        return {"output": output, "overflow": False}
    preview = "\n".join(lines[:max_lines])
    if len(preview) > max_chars:
        return {"output": preview[: max(0, max_chars - 1)] + "…", "overflow": True}
    return {"output": "\n".join(lines[:max_lines] + ["…"]), "overflow": True}


def _safe_input(run: dict[str, Any]) -> dict[str, Any]:
    inp = run.get("input") or {}
    if isinstance(inp, dict):
        return inp
    if isinstance(inp, str):
        try:
            import json as _j
            parsed = _j.loads(inp)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _format_input(inp: object, omit: tuple[str, ...] = ()) -> str:
    """Mirror opencode's `input()` helper: `[key=value, key2=value2]`."""
    if isinstance(inp, str):
        try:
            import json as _j
            parsed = _j.loads(inp)
            inp = parsed if isinstance(parsed, dict) else {}
        except Exception:
            return ""
    if not isinstance(inp, dict):
        return ""
    parts = []
    for key, value in inp.items():
        if key in omit:
            continue
        if isinstance(value, (str, int, float, bool)):
            parts.append(f"{key}={value}")
    if not parts:
        return ""
    return "[" + ", ".join(parts) + "]"


def tool_title(run: dict[str, Any]) -> str:
    """The title bit of a tool run (opencode's ``state.title``): Bash -> the
    command, Read/Edit/Write -> the file path, Glob/Grep -> the pattern, etc."""
    tool = str(run.get("tool", ""))
    inp = _safe_input(run)
    if tool in ("bash", "shell", "execute"):
        return str(inp.get("command", "")).strip()
    if tool in ("read", "write", "edit"):
        return str(inp.get("filePath", "")).strip()
    if tool == "apply_patch":
        return str(inp.get("command", "") or inp.get("filePath", "")).strip()
    if tool in ("glob", "grep"):
        pattern = str(inp.get("pattern", "")).strip()
        path = str(inp.get("path", "")).strip()
        title = pattern
        if pattern and path:
            title += f" in {path}"
        return title
    if tool == "webfetch":
        return str(inp.get("url", "")).strip()
    if tool == "webfetch_many":
        urls = inp.get("urls") or []
        if isinstance(urls, list) and urls:
            first = str(urls[0])[:80]
            return f"×{len(urls)} {first}"
        return str(inp.get("url", "")).strip()
    if tool == "websearch":
        return str(inp.get("query", "")).strip()
    if tool == "skill":
        return str(inp.get("name", "")).strip()
    if tool == "lsp":
        op = str(inp.get("operation", "")).strip()
        fp = str(inp.get("filePath", "")).strip()
        ln = str(inp.get("line", "")).strip()
        title = op
        if fp:
            title += f" {fp}"
            if ln:
                title += f":{ln}"
        return title.strip()
    return ""


def tool_label(run: dict[str, Any]) -> str:
    """Full `↳ Bash npm run test` style label for a tool run."""
    tool = str(run.get("tool", "?"))
    title = tool_title(run)
    base = _TOOL_TITLES.get(tool, tool)
    return f"{base} {title}".strip() if title else base


def _tool_display(tool: str) -> str:
    return tool if tool in _TOOL_DISPLAYS else "generic"



_TOOL_DISPLAYS = {
    "bash",
    "glob",
    "read",
    "grep",
    "find_symbols",
    "webfetch",
    "webfetch_many",
    "websearch",
    "webreach",
    "write",
    "edit",
    "task",
    "apply_patch",
    "todowrite",
    "question",
    "skill",
    "execute",
    "lsp",
}


def format_subagent_title(agent: str, description: str, background: bool = False) -> str:
    """Mirror opencode's ``formatSubagentTitle``:
    ``Build Task — fix the login bug`` (+ ``(background)`` for background agents)."""
    label = str(agent).title() if str(agent).strip() else "General"
    return f"{label} Task{' (background)' if background else ''} — {description}"


def format_subagent_toolcalls(count: int) -> str:
    """Mirror opencode's ``formatSubagentToolcalls``: `1 toolcall` / `3 toolcalls`."""
    return f"{count} toolcall{'s' if count != 1 else ''}"


def format_subagent_retry(attempt: int, message: str) -> str:
    """Mirror opencode's ``formatSubagentRetry``:
    ``Retrying (attempt 2) · <message>``."""
    return f"Retrying (attempt {attempt}) · {message}"


def format_completed_subagent_detail(toolcalls: int, duration: str) -> str:
    """Mirror opencode's ``formatCompletedSubagentDetail`:
    ``3 toolcalls · 12.5s`` (just the duration when no tools ran)."""
    if toolcalls == 0:
        return duration
    if not duration:
        return format_subagent_toolcalls(toolcalls)
    return f"{format_subagent_toolcalls(toolcalls)} · {duration}"


_TOOL_TITLES = {
    "bash": "Bash",
    "shell": "Bash",
    "execute": "Execute",
    "read": "Read",
    "write": "Write",
    "edit": "Edit",
    "apply_patch": "Patch",
    "glob": "Glob",
    "grep": "Grep",
    "webfetch": "WebFetch",
    "webfetch_many": "WebFetch Batch",
    "websearch": "WebSearch",
    "todowrite": "TodoWrite",
    "task": "Task",
    "question": "Ask",
    "skill": "Skill",
    "lsp": "LSP",
    "mcp": "MCP",
    "notify": "Notify",
}



class BubbleCoreMixin:
    """Construction, reactive watchers, and the main bodies
    (assistant markdown, reasoning, compaction)."""

    def __init__(
        self,
        role: str,
        content: Any = "",
        agent: str = "build",
        queued: bool = False,
        streaming: bool = False,
        directive: str = "",
        **kwargs: Any,
    ) -> None:
        self.role = role
        self.agent = agent
        self._message = content
        # Streaming deltas accumulate here and are joined once per flush
        # (append-only: `"".join(parts)` is O(n) per flush instead of an
        # O(n²) `content + text` rebuild of the growing string).
        self._stream_parts: list[str] = []
        # Optional header for the sub-agent "directive" block: the text of the
        # parent's instruction shown at the very top of a sub-agent's chat.
        self.directive = directive
        self._spinner = 0
        self._timer: Any = None
        self._thought_started: float | None = None
        self._thought_seconds: float | None = None
        super().__init__("", **kwargs)
        self.can_focus = role == "reasoning"
        self.set_reactive(type(self).queued, queued)
        self.set_reactive(type(self).streaming, streaming)
        self._refresh()


    def _diff_opts(self) -> dict[str, Any]:
        try:
            cfg = self.app.cfg
        except Exception:
            cfg = None
        if cfg is not None:
            try:
                return {
                    "diff_style": getattr(cfg, "diff_style", "split"),
                    "diff_wrap_mode": getattr(cfg, "diff_wrap_mode", "word"),
                    "suppress_backgrounds": getattr(cfg, "suppress_backgrounds", False),
                }
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/chat_view.py:376")
                pass
        return {}


    def watch_queued(self, value: bool) -> None:
        self.unfreeze()
        self._refresh()


    def watch_streaming(self, value: bool) -> None:
        if value:
            self.unfreeze()
        self._refresh()


    def on_unmount(self) -> None:
        """Clean up spinner timer when bubble is removed to prevent timer leaks."""
        self._stop_spinner()


    def watch_expanded(self, value: bool) -> None:
        self.unfreeze()
        self._refresh()


    def watch_selected(self, value: bool) -> None:
        self.unfreeze()
        self._refresh()


    @property
    def content(self) -> Any:
        """Raw payload (assistant text or tool-run dict).

        While streaming, received text also lives in `_stream_parts` until the
        next flush, so string payloads report the FULL text received so far.
        This matters beyond display: `ChatView.end_stream` DROPS a bubble whose
        `content` is falsy, so a mid-stream reply must never read as empty.
        """
        msg = self._message
        if isinstance(msg, str):
            return self._stream_text()
        return msg


    def _content_width(self) -> int:
        """Usable wrap width for this bubble, in columns.

        Never returns None/0. Passing `width=None` to `render_markdown` made rich
        fall back to its console default (80, or 142 in a piped env) and lay the
        whole reply out for a screen three times wider than the phone. The
        renderable then overflowed the widget, got cut mid-sentence and repainted,
        which showed the same message twice — once truncated, once whole.

        Order: our own size, then the screen, then a narrow safe default (a phone
        is never 80 columns; a too-wide guess is what caused the bug).
        """
        try:
            w = int(self.size.width or 0)
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/chat_view.py:_content_width")
            w = 0
        if w <= 0:
            try:
                w = int(self.app.size.width or 0)
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/chat_view.py:_content_width app")
                w = 0
        return w if w > 0 else 40


    def _build_content(self) -> RenderableType:
        theme = active_theme()
        if self.role == "user":
            inner: list[RenderableType] = []
            if self.directive:
                header = Text()
                header.append("▣ ", style=theme.agent_color(self.agent or "build"))
                header.append(self.directive, style=f"bold {theme.c('text')}")
                header.append(" · ", style=theme.c("text_muted"))
                header.append("Directive", style=theme.c("text_muted"))
                inner.append(header)
            inner.append(Text(str(self._message), style=theme.c("user_bubble")))
            if self.queued:
                color = theme.c("queue_badge")
                inner.append(Text(" QUEUED ", style=f"bold {theme.c('background')} on {color}"))
            return Group(*inner)
        if self.role == "assistant":
            width = self._content_width()
            if self.streaming:
                # Plain wrapped text for the ENTIRE streaming phase. Parsing the
                # whole accumulated markdown document on every delta flush was
                # O(reply-length²) and caused visible layout jumps mid-reply.
                # Streaming is append-only plain text; the final message
                # re-renders once as full markdown in end_stream.
                # _stream_text() so the un-flushed tail is visible NOW.
                body = self._stream_text()
                return Group(
                    Text(body, style=theme.c("text")),
                    Text("▍", style=theme.c("streaming_cursor")),
                )
            return _plain(self._message, width=width)
        if self.role == "reasoning":
            return self._build_reasoning()
        if self.role == "assistant_mode":
            # e.g. `▣ Build · model`
            t = Text()
            t.append("▣ ", style=theme.agent_color(self.agent))
            t.append(self.agent.title(), style=theme.c("text"))
            if self._message:
                t.append(f" · {self._message}", style=theme.c("text_muted"))
            return t
        if self.role == "meta":
            return Text(str(self._message), style=theme.c("text_muted"))
        if self.role == "compaction":
            if self.streaming:
                # Live compaction: show the anchored summary as it streams in
                # (plain text + cursor), then re-render as the ` Compaction `
                # divider + markdown in end_compaction_stream.
                theme = active_theme()
                header = Text("▸ Compacted summary", style=theme.c("warning"))
                body = self._stream_text()
                if body:
                    return Group(header, Text(body, style=theme.c("text")), Text("▍", style=theme.c("primary")))
                return Group(header, Text("▍", style=theme.c("primary")))
            return self._build_compaction()
        return self._render_tool(self._message) if isinstance(self._message, dict) else _plain(self._message, width=self._content_width())


    def _build_compaction(self) -> RenderableType:
        """Centered ` Compaction ` divider (opencode's compaction part,
        i18n `ui.messagePart.compaction`) with the anchored summary beneath —
        rendered as markdown so the headings (`## Objective`, `## Important
        Details`, `## Work State`, `## Next Move`, `## Relevant Files`) get the
        same colored styling as the official opencode summary message."""
        theme = active_theme()
        width = self._content_width()
        title = " Compaction "
        n = max(1, (width - len(title)) // 2)
        divider = Text("─" * n, style=theme.c("border_active"))
        divider.append(title, style=theme.c("border_active"))
        divider.append("─" * (width - len(title) - n), style=theme.c("border_active"))
        parts: list[RenderableType] = [divider]
        summary = str(self._message or "").strip()
        if summary:
            parts.append(render_markdown(summary, width=width))
        return Group(*parts)


    def _build_reasoning(self) -> RenderableType:
        """Collapsed by default: `+ Thought: <title>`. Streaming shows a spinner
        with `Thinking...`; clicking the header toggles the muted markdown body
        (also while the thought is still streaming)."""
        theme = active_theme()
        # _stream_text() while streaming: the newest deltas live in
        # _stream_parts until the next flush and would otherwise be invisible.
        summary = reasoning_summary(
            self._stream_text() if self.streaming else self._message
        )
        title = summary["title"]
        body = summary["body"]
        prefix = "- " if self.expanded else "+ "

        if self.streaming:
            header = Text(
                f"{prefix}{SPINNER_FRAMES[self._spinner]} Thinking" + (f": {title}" if title else ""),
                style=theme.c("tool_running"),
            )
        else:
            header = Text(prefix, style=theme.c("tool_running"))
            header.append("Thought", style=theme.c("tool_running"))
            if self._thought_seconds is not None:
                header.append(f" for {self._thought_seconds:.1f}s", style=theme.c("thinking_time"))
            if title:
                header.append(f": {title}", style=theme.c("tool_running"))
        parts: list[RenderableType] = [header]
        # No bold `**Title**` from the model: preview the raw thinking so the
        # collapsed row still says WHAT was thought, not just `+ Thought`.
        if not self.expanded and not title and body:
            preview = " ".join(body.split())
            if len(preview) > 100:
                preview = preview[:97].rstrip() + "\u2026"
            parts.append(Text(preview, style=theme.c("text_muted")))
        if self.expanded and body:
            parts.append(Text(body, style=theme.c("text_muted")))
        return Group(*parts)



class BubbleToolMixin:
    """One renderer per tool, plus the shared row helpers
    (status, error, denied, failed, inline, error line)."""

    def _render_tool(self, tool_run: dict[str, Any]) -> RenderableType:
        display = _tool_display(tool_run.get("tool", "?"))
        fn = getattr(self, f"_render_{display}", None)
        if fn is None:
            fn = self._render_generic
        return fn(tool_run)


    def _status(self, run: dict[str, Any]) -> str:
        return run.get("status", "pending")


    def _error(self, run: dict[str, Any]) -> str:
        return run.get("error") or ""


    def _denied(self, run: dict[str, Any]) -> bool:
        # permission denials may arrive as an error field or as output text
        err = run.get("error") or (run.get("output") or "")
        return any(
            m in err
            for m in (
                "QuestionRejectedError",
                "rejected permission",
                "permission denied",
                "denied by permission",
                "user dismissed",
                "specified a rule",
            )
        )


    def _failed(self, run: dict[str, Any]) -> bool:
        """Mirror opencode's InlineTool.failed: a real error that isn't a denial."""
        if self._denied(run):
            return False
        if run.get("status") == "error":
            return True
        return bool(run.get("error"))


    def _inline(
        self,
        icon: str,
        pending: str,
        label: str,
        *,
        spinner: bool = False,
        complete: bool | str | None = None,
        color: str | None = None,
    ) -> RenderableType:
        """Mirror opencode's InlineToolRow: ~ pending / spinner running / icon label.

        While running, once the tool input is known (complete truthy) we show the
        actual action (`← Edit /x.py`) instead of `~ Preparing edit...`, matching
        opencode. Colors mirror opencode: running -> primary, completed -> textMuted,
        failed -> error (red), denied -> strikethrough. ``color`` overrides the
        running/completed color (opencode passes ``theme.error`` for a retrying
        sub-agent row).
        """
        theme = active_theme()
        status = self._status(self._message)
        denied = self._denied(self._message)
        failed = self._failed(self._message)
        done = status in ("completed", "error") if complete is None else bool(complete)

        if status == "running":
            if spinner:
                return Text(f"{SPINNER_FRAMES[self._spinner]} {label}", style=color or theme.c("tool_running"))
            if complete:
                return Text(f"{icon} {label}", style=color or theme.c("tool_running"))
            return Text(f"~ {pending}", style=theme.c("text_muted"))
        if not done:
            return Text(f"~ {pending}", style=theme.c("text_muted"))

        main = color or (theme.c("tool_error") if failed else theme.c("tool_success"))
        style = f"{theme.c('tool_denied')} strike" if denied else main
        return Text(f"{icon} {label}", style=style)


    def _error_line(self, run: dict[str, Any]) -> RenderableType | None:
        """A red error line for a failed tool (None when not a real error)."""
        if not self._failed(run) or self._denied(run):
            return None
        err = (self._error(run) or "").strip()
        if not err:
            return None
        return Text(err, style=active_theme().c("tool_error"))


    def _render_bash(self, run: dict[str, Any]) -> RenderableType:
        theme = active_theme()
        status = self._status(run)
        command = str((run.get("input") or {}).get("command", ""))
        output = (run.get("output") or "").strip()
        workdir = str((run.get("input") or {}).get("workdir") or "")
        if status == "running":
            return Text(f"{SPINNER_FRAMES[self._spinner]} {command}", style=theme.c("text"))
        if output:
            lines: list[RenderableType] = []
            if workdir and workdir != ".":
                lines.append(Text(f"# Running in {workdir}", style=theme.c("text_muted")))
            lines.append(Text(f"$ {command}", style=theme.c("text")))
            collapsed = self._tool_collapse(run)
            output_text = output if self.expanded else collapsed["output"]
            lines.append(Text(output_text, style=theme.c("text")))
            if collapsed["overflow"]:
                label = "Click to collapse" if self.expanded else "Click to expand"
                lines.append(Text(label, style=theme.c("text_muted")))
            return Group(*lines)
        return self._inline("$", "Writing command...", command)


    def _render_read(self, run: dict[str, Any]) -> RenderableType:
        theme = active_theme()
        status = self._status(run)
        filepath = str((run.get("input") or {}).get("filePath", ""))
        loaded = (run.get("metadata") or {}).get("loaded") or []
        extra = _format_input((run.get("input") or {}), omit=("filePath",))
        row = self._inline(
            "→",
            "Reading file...",
            f"Read {filepath}{extra}",
            spinner=status == "running",
            complete=bool(filepath or extra),
        )
        err = self._error_line(run)
        if err is not None:
            return Group(row, err)
        if status == "completed" and loaded:
            sub = [
                Text(f"↳ Loaded {p}", style=theme.c("text_muted"))
                for p in (loaded if isinstance(loaded, list) else [loaded])
            ]
            output = (run.get("output") or "").strip()
            if output:
                collapsed = self._tool_collapse(run)
                body = output if self.expanded else collapsed["output"]
                parts: list[RenderableType] = [row, *sub, Text(body, style=active_theme().c("text"))]
                if collapsed["overflow"]:
                    parts.append(Text("Click to collapse" if self.expanded else "Click to expand", style=active_theme().c("text_muted")))
                return Group(*parts)
            return Group(row, *sub)
        return row


    def _render_glob(self, run: dict[str, Any]) -> RenderableType:
        inp = run.get("input") or {}
        pattern = str(inp.get("pattern", ""))
        path = str(inp.get("path", "")) if inp.get("path") else ""
        count = (run.get("metadata") or {}).get("count")
        label = f'Glob "{pattern}"'
        if path:
            label += f" in {path}"
        if isinstance(count, int):
            label += f" ({count} {'match' if count == 1 else 'matches'})"
        row = self._inline("✱", "Finding files...", label, complete=bool(pattern or path))
        err = self._error_line(run)
        if err is not None:
            return Group(row, err)
        return row


    def _render_grep(self, run: dict[str, Any]) -> RenderableType:
        inp = run.get("input") or {}
        pattern = str(inp.get("pattern", ""))
        path = str(inp.get("path", "")) if inp.get("path") else ""
        count = (run.get("metadata") or {}).get("matches")
        label = f'Grep "{pattern}"'
        if path:
            label += f" in {path}"
        if isinstance(count, int):
            label += f" ({count} {'match' if count == 1 else 'matches'})"
        row = self._inline("✱", "Searching content...", label, complete=bool(pattern or path))
        err = self._error_line(run)
        if err is not None:
            return Group(row, err)
        return row


    def _render_find_symbols(self, run: dict[str, Any]) -> RenderableType:
        inp = _safe_input(run)
        meta = run.get("metadata") or {}
        if not isinstance(meta, dict):
            meta = {}
        query = str(inp.get("query", ""))
        matches = meta.get("matches")
        label = f'Symbols "{query}"' if query else "Symbols"
        if isinstance(matches, int):
            label += f" ({matches} {'match' if matches == 1 else 'matches'})"
        row = self._inline("⎆", "Finding symbols...", label, spinner=self._status(run) == "running", complete=bool(query))
        err = self._error_line(run)
        if err is not None:
            return Group(row, err)
        if self._status(run) == "completed":
            output = (run.get("output") or "").strip()
            if output:
                collapsed = self._tool_collapse(run)
                body = output if self.expanded else collapsed["output"]
                parts: list[RenderableType] = [row, Text(body, style=active_theme().c("text"))]
                if collapsed["overflow"]:
                    parts.append(Text("Click to collapse" if self.expanded else "Click to expand", style=active_theme().c("text_muted")))
                return Group(*parts)
        return row


    def _render_webfetch(self, run: dict[str, Any]) -> RenderableType:
        url = str((_safe_input(run)).get("url", ""))
        row = self._inline("✦", "Fetching from the web...", f"WebFetch {url}", spinner=self._status(run) == "running", complete=bool(url))
        err = self._error_line(run)
        if err is not None:
            return Group(row, err)
        return row


    def _render_webfetch_many(self, run: dict[str, Any]) -> RenderableType:
        inp = _safe_input(run)
        meta = run.get("metadata") or {}
        if not isinstance(meta, dict):
            meta = {}
        urls = inp.get("urls") or []
        n = len(urls) if isinstance(urls, list) else 0
        done = meta.get("done", meta.get("progress_done"))
        total = meta.get("total", n or meta.get("count", 0))
        try:
            done_i = int(done) if done is not None else None
            total_i = int(total) if total else n
        except Exception:
            done_i, total_i = None, n
        if self._status(run) == "running":
            if isinstance(done_i, int) and total_i:
                label = f"WebFetch Batch {done_i}/{total_i}"
            elif total_i:
                label = f"WebFetch Batch ×{total_i}"
            else:
                label = "WebFetch Batch"
            row = self._inline("⟡", "Fetching batch...", label, spinner=True, complete=True)
        else:
            ok = meta.get("succeeded")
            fail = meta.get("failed")
            tail = f" ({ok} ok" + (f", {fail} failed" if fail else "") + ")" if isinstance(ok, int) else (f" ×{total_i}" if total_i else "")
            first = str(urls[0])[:60] if isinstance(urls, list) and urls else ""
            label = f"WebFetch Batch{tail} {first}".strip()
            row = self._inline("⟡", "Fetching batch...", label, complete=True)
        err = self._error_line(run)
        if err is not None:
            return Group(row, err)
        if self._status(run) == "completed":
            output = (run.get("output") or "").strip()
            if output:
                collapsed = self._tool_collapse(run)
                body = output if self.expanded else collapsed["output"]
                parts: list[RenderableType] = [row, Text(body, style=active_theme().c("text"))]
                if collapsed["overflow"]:
                    parts.append(Text("Click to collapse" if self.expanded else "Click to expand", style=active_theme().c("text_muted")))
                return Group(*parts)
        return row


    def _render_lsp(self, run: dict[str, Any]) -> RenderableType:
        inp = _safe_input(run)
        meta = run.get("metadata") or {}
        if not isinstance(meta, dict):
            meta = {}
        op = str(inp.get("operation", ""))
        fp = str(inp.get("filePath", ""))
        ln = str(inp.get("line", "")) if inp.get("line") else ""
        matches = meta.get("matches", meta.get("issues", ""))
        label = op
        if fp:
            label += f" {fp}"
            if ln:
                label += f":{ln}"
        if isinstance(matches, int):
            label += f" ({matches})"
        row = self._inline("⌁", "Jumping...", label.strip() or "LSP", spinner=self._status(run) == "running", complete=bool(op or fp))
        err = self._error_line(run)
        if err is not None:
            return Group(row, err)
        return row


    def _render_websearch(self, run: dict[str, Any]) -> RenderableType:
        inp = _safe_input(run)
        meta = run.get("metadata") or {}
        if not isinstance(meta, dict):
            meta = {}
        provider = str(meta.get("provider", ""))
        query = str(inp.get("query", ""))
        count = meta.get("numResults", meta.get("numresults"))
        base = f'{provider + " " if provider else ""}WebSearch "{query}"' if query else "WebSearch"
        label = base.strip()
        if isinstance(count, int):
            label += f" ({count} results)"
        row = self._inline("◈", "Searching web...", label.strip() or "WebSearch", spinner=self._status(run) == "running", complete=bool(query))
        err = self._error_line(run)
        if err is not None:
            return Group(row, err)
        if self._status(run) == "completed":
            output = (run.get("output") or "").strip()
            if output:
                collapsed = self._tool_collapse(run)
                body = output if self.expanded else collapsed["output"]
                parts: list[RenderableType] = [row, Text(body, style=active_theme().c("text"))]
                if collapsed["overflow"]:
                    parts.append(Text("Click to collapse" if self.expanded else "Click to expand", style=active_theme().c("text_muted")))
                return Group(*parts)
        return row


    def _render_write(self, run: dict[str, Any]) -> RenderableType:
        theme = active_theme()
        filepath = str((run.get("input") or {}).get("filePath", ""))
        written = (run.get("metadata") or {}).get("content")
        if self._status(run) == "completed" and written:
            content = str(written)
            collapsed = collapse_tool_output(content, 10, 10 * 80)
            output = content if self.expanded else collapsed["output"]
            rows: list[RenderableType] = [
                Text(f"# Wrote {filepath}", style=theme.c("text_muted")),
                Text(output, style=theme.c("text")),
            ]
            if collapsed["overflow"]:
                label = "Click to collapse" if self.expanded else "Click to expand"
                rows.append(Text(label, style=theme.c("text_muted")))
            return Group(*rows)
        return self._inline("←", "Preparing write...", f"Write {filepath}", complete=bool(filepath))


    def _render_edit(self, run: dict[str, Any]) -> RenderableType:
        theme = active_theme()
        filepath = str((run.get("input") or {}).get("filePath", ""))
        diff = (run.get("metadata") or {}).get("diff")
        if self._status(run) == "completed" and diff:
            return Group(
                Text(f"← Edit {filepath}", style=theme.c("text_muted")),
                _render_diff(diff, filepath, self._content_width(), self._diff_opts()),
            )
        err = self._error_line(run)
        if err is not None:
            replace_all = _format_input((run.get("input") or {}), omit=("filePath", "oldString", "newString"))
            return Group(
                self._inline("←", "Preparing edit...", f"Edit {filepath}{replace_all}", complete=bool(filepath or replace_all)),
                err,
            )
        replace_all = _format_input((run.get("input") or {}), omit=("filePath", "oldString", "newString"))
        return self._inline("←", "Preparing edit...", f"Edit {filepath}{replace_all}", complete=bool(filepath or replace_all))


    def _render_apply_patch(self, run: dict[str, Any]) -> RenderableType:
        theme = active_theme()
        files = (run.get("metadata") or {}).get("files") or []
        if self._status(run) == "completed" and files:
            lines: list[RenderableType] = []
            for f in files if isinstance(files, list) else [files]:
                if not isinstance(f, dict):
                    continue
                rel = f.get("relativePath", "")
                title = f"← Patched {rel}"
                if f.get("type") == "delete":
                    title = f"# Deleted {rel}"
                elif f.get("type") == "add":
                    title = f"# Created {rel}"
                lines.append(Text(title, style=theme.c("text_muted")))
                patch = f.get("patch")
                if patch:
                    fpath = f.get("filePath") or f.get("relativePath") or ""
                    lines.append(_render_diff(patch, fpath, self._content_width(), self._diff_opts()))
            return Group(*lines)
        return self._inline("✦", "Preparing patch...", "Patch", complete=True)


    def _render_todowrite(self, run: dict[str, Any]) -> RenderableType:
        theme = active_theme()
        todos = (run.get("metadata") or {}).get("todos") or (run.get("input") or {}).get("todos") or []
        if self._status(run) == "completed" and todos:
            lines: list[RenderableType] = [Text("# Todos", style=theme.c("text_muted"))]
            for todo in todos if isinstance(todos, list) else [todos]:
                if not isinstance(todo, dict):
                    continue
                status = todo.get("status", "pending")
                mark = "✓" if status == "completed" else ("•" if status == "in_progress" else " ")
                color = theme.c("tool_running") if status == "in_progress" else (theme.c("markdown_todo_done") if status == "completed" else theme.c("markdown_todo_open"))
                lines.append(Text(f"[{mark}] {todo.get('content', '')}", style=color))
            return Group(*lines)
        return self._inline("☰", "Updating todos...", "Updating todos...", complete=True)


    def _render_question(self, run: dict[str, Any]) -> RenderableType:
        theme = active_theme()
        questions = (run.get("input") or {}).get("questions") or []
        answers = (run.get("metadata") or {}).get("answers")
        count = len(questions) if isinstance(questions, list) else 0
        if self._status(run) == "completed" and answers is not None:
            lines: list[RenderableType] = [Text("# Questions", style=theme.c("text_muted"))]
            for i, q in enumerate(questions if isinstance(questions, list) else []):
                if not isinstance(q, dict):
                    continue
                lines.append(Text(str(q.get("question", "")), style=theme.c("text_muted")))
                ans = answers[i] if isinstance(answers, list) and i < len(answers) else None
                lines.append(
                    Text(
                        ", ".join(ans) if isinstance(ans, list) else str(ans) if ans else "(no answer)",
                        style=theme.c("text"),
                    )
                )
            return Group(*lines)
        return self._inline("→", "Asking questions...", f"Asked {count} question{'s' if count != 1 else ''}", complete=True)



class BubbleTaskMixin:
    """Sub-agent task rows: the task tool output and every helper
    that reads child-session state."""

    def _render_task(self, run: dict[str, Any]) -> RenderableType:
        """Render a `task` tool row mirroring opencode's Subagent component
        (packages/tui/src/routes/session/index.tsx):

          running   : `│ Build Task — fix login` (spinner) + live `↳ Bash …`
          completed : `✓ Build Task — fix login` then `↳ 3 toolcalls · 12.5s`
          retrying  : the whole row turns error-red with `↳ Retrying (attempt ·)`
          hint      : a persistent `↓ ctrl+down view subagents` line

        Clicking the row opens the sub-agent's own session (see on_click).
        """
        theme = active_theme()
        status = self._status(run)
        inp = run.get("input") or {}
        meta = run.get("metadata") or {}
        # Resolved agent actually launched wins over the requested input:
        # read-only forcing or validation may change what ran, and the row
        # must show that — not the request. Falls back to input, then build.
        agent_label = (
            str(meta.get("agent") or "").strip()
            or str(inp.get("subagent_type", "")).strip()
            or "build"
        )
        description = str(inp.get("description", "")).strip()
        if not description:
            description = str(meta.get("title") or "")
        background = bool(meta.get("background"))
        sid = str(meta.get("sessionId") or "")
        retry = meta.get("retry") or {}
        title = format_subagent_title(agent_label, description, background)
        icon = "✓" if status == "completed" else "│"
        # the active sub-agent (the one selected / being viewed) gets an accent
        # marker on its row, mirroring opencode's highlighted current agent.
        if self.selected:
            icon = f"▸ {icon}"

        sub: list[RenderableType] = []
        if status == "running":
            runs = self._task_child_runs(sid)
            # official `current()`: the last running/completed child tool that
            # has a real `state.title` (`↳ Bash npm run test`); otherwise just
            # the toolcall count.
            current = next(
                (tool_label(r) for r in reversed(runs) if r.get("status") in ("running", "completed") and tool_title(r)),
                None,
            )
            if meta.get("bg_done"):
                # background worker finished, reply uncollected: frozen
                # final line instead of live ticking (Fix B stamped these).
                tc = meta.get("toolcalls")
                dur = str(meta.get("duration") or "")
                detail = format_completed_subagent_detail(
                    tc if isinstance(tc, int) and tc >= 0 else 0, dur
                )
                if detail:
                    sub.append(Text(f"↳ {detail} (done — task_read to collect)", style=theme.c("text_muted")))
                elif current:
                    sub.append(Text(f"↳ {current}", style=theme.c("text_muted")))
            elif current:
                sub.append(Text(f"↳ {current}", style=theme.c("text_muted")))
            elif runs:
                sub.append(Text(f"↳ {format_subagent_toolcalls(len(runs))}", style=theme.c("text_muted")))
            # live elapsed behind the running row (`↳ 3 toolcalls · 12.5s`)
            # so the agent's runtime is visible while it works, not only
            # after it finishes. Skipped once bg_done froze the final line.
            live_elapsed = str(meta.get("elapsed") or "")
            if live_elapsed and not meta.get("bg_done"):
                sub.append(Text(f"↳ {live_elapsed}", style=theme.c("text_muted")))
        elif status == "completed":
            # Official: `if (!isRunning() && completed)` ALWAYS pushes
            # `↳ N toolcalls · duration`, with duration derived from the
            # child session itself at render time — never gated on a
            # finalize event having fired. Same here: stamped metadata
            # first, then the child session (live map, else disk), so the
            # footer shows even on relinked/resumed rows whose finalize
            # never ran in this process.
            runs = self._task_child_runs(sid)
            toolcalls = self._task_render_toolcalls(sid, meta, runs)
            duration = self._task_render_duration(sid, meta)
            detail = format_completed_subagent_detail(toolcalls or 0, duration)
            if detail:
                sub.append(Text(f"↳ {detail}", style=theme.c("text_muted")))
            elif toolcalls is not None:
                sub.append(Text(f"↳ {format_subagent_toolcalls(toolcalls)}", style=theme.c("text_muted")))
        elif status == "error" and sid:
            err = (run.get("error") or "").strip()
            if err:
                sub.append(Text(f"↳ {err}", style=theme.c("error")))

        lines: list[RenderableType] = [
            self._inline(
                icon,
                "Delegating...",
                title,
                spinner=status == "running",
                complete=bool(description),
                color=theme.c("error") if retry.get("attempt") else None,
            )
        ]
        if retry.get("attempt"):
            lines.append(
                Text(
                    f"↳ {format_subagent_retry(int(retry['attempt']), str(retry.get('message') or ''))}",
                    style=theme.c("error"),
                )
            )
        lines += sub
        if sid and self._is_last_task_row():
            lines.append(Text("↓  ctrl+down  view subagents", style=theme.c("text_muted")))
        return Group(*lines)


    def _task_parent_chat(self) -> ChatView | None:
        """The ChatView this task row is mounted in (the parent session's chat)."""
        parent = self.parent
        from .chat_view import ChatView  # deferred: chat_view imports this module

        if isinstance(parent, ChatView):
            return parent
        return None


    def _is_last_task_row(self) -> bool:
        """True when this is the newest task row in the parent chat — the row
        under which opencode draws the `view subagents` hint (once, per message)."""
        chat = self._task_parent_chat()
        if chat is None:
            return False
        from .chat_view import MessageBubble  # deferred: chat_view imports this module

        latest: MessageBubble | None = None
        try:
            nodes = list(chat.walk_children(MessageBubble))
        except Exception:
            nodes = [n for n in getattr(chat, "children", ()) if isinstance(n, MessageBubble)]
        for child in nodes:
            if child.role == "tool" and isinstance(child.content, dict) and child.content.get("tool") == "task":
                latest = child
        return latest is self


    def _task_child_session(self, session_id: str) -> Any | None:
        """The sub-agent's Session object (live map, else disk) — the source
        for render-time duration, mirroring official's message timestamps.

        Official computes duration from the child session's first-user and
        last-assistant message times at RENDER time, so the footer never
        depends on a finalize event firing. Ours stores session-level
        created/completed instead; same guarantee, cheaper lookup."""
        if not session_id:
            return None
        try:
            app = self.app
        except Exception:
            app = None
        if app is not None:
            try:
                sessions = getattr(app, "_sessions", None)
                if isinstance(sessions, dict) and sessions.get(session_id) is not None:
                    return sessions.get(session_id)
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/chat_view.py:1055")
                pass
        try:
            from ..session import load_session as _load

            return _load(session_id)
        except Exception:
            return None


    def _task_render_duration(self, session_id: str, meta: dict[str, Any]) -> str:
        """Duration string for a finished task row, official-style.

        Order: live `elapsed` ticker (running) → stamped/finalized `duration`
        → child session created/completed (works even when finalize never
        ran, e.g. relinked/resumed rows). '' only when nothing is known."""
        live = str(meta.get("elapsed") or "")
        if live:
            return live
        done = str(meta.get("duration") or "")
        if done:
            return done
        try:
            sess = self._task_child_session(session_id)
            if sess is None:
                return ""
            created = float(getattr(sess, "created", 0) or 0)
            completed = float(getattr(sess, "completed", 0) or 0)
            if completed > created:
                from .input_bar import format_duration

                return format_duration(completed - created)
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/chat_view.py:1086")
            pass
        return ""


    def _task_render_toolcalls(self, session_id: str, meta: dict[str, Any], runs: list[dict[str, Any]]) -> int | None:
        """Toolcall count for a finished task row, official-style.

        Order: stamped `toolcalls` → live child-chat runs → child session
        transcript tool messages (disk, resume-proof). None only when the
        child session can't be found at all."""
        stamped = meta.get("toolcalls")
        if isinstance(stamped, int) and stamped >= 0:
            return stamped
        if runs:
            return len([r for r in runs if r.get("status") == "completed"])
        try:
            sess = self._task_child_session(session_id)
            if sess is None:
                return None
            messages = getattr(sess, "messages", None) or []
            return sum(1 for m in messages if isinstance(m, dict) and m.get("role") == "tool")
        except Exception:
            return None


    def _task_child_chat(self, session_id: str) -> ChatView | None:
        """The sub-agent's own ChatView (opencode reads the child session's parts
        via ``sync.data.part[msg.id]``; here the child's tool bubbles are the
        equivalent live store). None when it isn't registered / was pruned."""
        if not session_id:
            return None
        try:
            app = self.app
        except Exception:
            # WidgetHost / off-pump tests, or the app is already torn down —
            # there's simply no child ChatView to read live state from.
            return None
        chats = getattr(app, "_chats", None)
        if not isinstance(chats, dict):
            return None
        return chats.get(session_id)


    def _task_child_runs(self, session_id: str) -> list[dict[str, Any]]:
        chat = self._task_child_chat(session_id)
        if chat is None:
            return []
        return [
            r
            for r in chat.tool_runs()
            if r.get("status") in ("running", "completed", "error")
        ]


    def _task_sid(self) -> str:
        """Child session id for a task row, from every known location.

        Live rows carry metadata.sessionId; resumed rows re-attached by the
        relinker do too — but rows rendered before the relink (or from old
        saves mid-retry) may only have it in the bubble's fallback fields.
        """
        try:
            c = self.content or {}
            meta = c.get("metadata") or {}
            for key in ("sessionId", "session_id"):
                sid = meta.get(key)
                if sid:
                    return str(sid)
            for key in ("sessionId", "session_id", "sid"):
                sid = c.get(key)
                if sid:
                    return str(sid)
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/chat_view.py:1369")
            pass
        return ""



class BubbleFrameMixin:
    """execute / skill / generic rows and the shared frame builder."""

    def _render_execute(self, run: dict[str, Any]) -> RenderableType:
        theme = active_theme()
        status = self._status(run)
        icon = "✓" if status == "completed" else "│"
        row = self._inline(icon, "execute", "execute", spinner=status == "running")
        calls = (run.get("metadata") or {}).get("toolCalls") or []
        sub = []
        for c in calls if isinstance(calls, list) else []:
            if not isinstance(c, dict):
                continue
            name = c.get("tool", "")
            args = _format_input(c.get("input") or {})
            failed = " (failed)" if c.get("status") == "error" else ""
            sub.append(Text(f"↳ {name}{args}{failed}", style=theme.c("text_muted")))
        return Group(row, *sub) if sub else row


    def _render_skill(self, run: dict[str, Any]) -> RenderableType:
        theme = active_theme()
        status = self._status(run)
        name = str((run.get("input") or {}).get("name", ""))
        output = (run.get("output") or "").strip()
        if status == "completed" and output and not name:
            lines: list[RenderableType] = [Text("# Skills", style=theme.c("text_muted"))]
            for line in output.split("\n"):
                if line.startswith("- "):
                    lines.append(Text(line[2:], style=theme.c("text")))
            return Group(*lines) if len(lines) > 1 else self._inline("→", "Loading skill...", "Skills", complete=True)
        row = self._inline("→", "Loading skill...", f'Skill "{name}"' if name else "Skills", complete=bool(name or status == "completed"))
        err = self._error_line(run)
        parts: list[RenderableType] = [row]
        if err is not None:
            parts.append(err)
        # Auditability: a loaded skill injects up to 30KB of instructions —
        # show the body collapsed (3 lines) with click-to-expand like generic
        # tools, instead of a bare one-liner.
        if status == "completed" and output and name:
            collapsed = self._tool_collapse({**run, "tool": "skill"})
            body = output if self.expanded else collapsed["output"]
            parts.append(Text(body, style=theme.c("text")))
            if collapsed["overflow"]:
                label = "Click to collapse" if self.expanded else "Click to expand"
                parts.append(Text(label, style=theme.c("text_muted")))
        return Group(*parts) if len(parts) > 1 else row


    def _render_generic(self, run: dict[str, Any]) -> RenderableType:
        theme = active_theme()
        tool = run.get("tool", "?")
        output = (run.get("output") or "").strip()
        if self._status(run) == "completed" and output:
            collapsed = self._tool_collapse(run)
            output_text = output if self.expanded else collapsed["output"]
            rows: list[RenderableType] = [
                Text(f"# {tool} {_format_input(run.get('input') or {})}".strip(), style=theme.c("text_muted")),
                Text(output_text, style=theme.c("text")),
            ]
            if collapsed["overflow"]:
                label = "Click to collapse" if self.expanded else "Click to expand"
                rows.append(Text(label, style=theme.c("text_muted")))
            return Group(*rows)
        label = f"{tool} {_format_input(run.get('input') or {})}".strip()
        return self._inline("⚙", "Running...", label, complete=bool(label))


    def _apply_frame(self) -> None:
        theme = active_theme()
        st = self.styles
        if self.role == "user":
            st.background = theme.c("background_panel")
            st.border_left = ("solid", theme.agent_color(self.agent))
            st.padding = (1, 1, 1, 2)
        elif self.role == "tool" and self._tool_block():
            # opencode BlockTool: left border in the panel background, panel fill.
            st.background = theme.c("background_panel")
            st.border_left = ("solid", theme.c("background"))
            st.padding = (1, 1, 1, 2)
        else:
            # indented plain text (assistant / mode-line / meta / inline tool).
            # use an invisible border so Textual never paints its default.
            st.background = "transparent"
            st.border_left = ("solid", theme.c("background"))
            st.padding = (0, 0, 0, 3)
        st.margin = (1, 0, 0, 0)



class BubbleRefreshMixin:
    """The collapsible tool-result block, the refresh entry point,
    freeze/unfreeze and the spinner tick."""

    def _tool_block(self) -> bool:
        """A completed tool run renders as a block iff it produced a result block."""
        if not isinstance(self._message, dict):
            return False
        run = self._message
        if run.get("status") != "completed":
            return False
        name = run.get("tool", "?")
        display = _tool_display(name)
        if display == "bash":
            return bool((run.get("output") or "").strip())
        if display == "edit":
            return bool((run.get("metadata") or {}).get("diff"))
        if display == "apply_patch":
            return bool((run.get("metadata") or {}).get("files"))
        if display == "todowrite":
            return bool((run.get("metadata") or {}).get("todos"))
        if display == "question":
            return (run.get("metadata") or {}).get("answers") is not None
        if display == "write":
            return bool((run.get("metadata") or {}).get("content"))
        if display == "generic":
            return bool((run.get("output") or "").strip())
        if display == "skill":
            return bool((run.get("output") or "").strip())
        if display in ("websearch", "webfetch", "webfetch_many", "find_symbols", "read"):
            return bool((run.get("output") or "").strip())
        return False


    def _tool_collapse(self, run: dict[str, Any]) -> dict:
        """Mirror opencode's per-tool collapse limits (bash 10 lines, generic 3)."""
        output = (run.get("output") or "").strip()
        if not output:
            return {"output": output, "overflow": False}
        display = _tool_display(run.get("tool", "?"))
        if display == "bash":
            return collapse_tool_output(output, 10, 10 * 80)
        if display in ("websearch", "webfetch", "webfetch_many", "find_symbols", "read"):
            return collapse_tool_output(output, 8, 8 * 80)
        return collapse_tool_output(output, 3, 3 * 80)


    def _tool_overflow(self) -> bool:
        return isinstance(self._message, dict) and self._tool_collapse(self._message)["overflow"]


    def _refresh(self) -> None:
        # Frozen bubbles are done: same pixels, skip the rebuild + layout.
        # Any state flip unfreezes first (see watchers below), so a frozen
        # bubble can never go stale.
        try:
            if bool(getattr(self, "frozen", False)) and not self.streaming:
                return
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/chat_view.py:1272")
            pass
        self._apply_frame()
        self.update(self._build_content())
        # Debug-only: catch the "same message painted twice" bug that a bad
        # repaint produces (one copy truncated, one whole). Reads what is
        # actually on screen and compares it to this bubble. Off unless
        # OPENCODE_DEBUG_DUP is set, so normal use pays one dict lookup.
        try:
            if os.environ.get("OPENCODE_DEBUG_DUP", ""):
                self.call_after_refresh(self._probe_dup)
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/chat_bubble.py:_refresh dup-probe")

    def _dup_marker(self) -> str:
        """Short PREFIX of this message, used to spot a second painted copy.

        A PREFIX, not the whole line: the duplicate is the same message cut off
        mid-sentence by the bad repaint, so only the leading words exist in both
        copies. A whole-line marker misses the bug entirely.
        """
        msg = self._message
        if not isinstance(msg, str) or not msg.strip():
            return ""
        for raw in msg.splitlines():
            line = raw.strip().lstrip("#").strip()
            if len(line) >= 24:
                return line[:32]
        return msg.strip()[:32]

    def _dup_report(self, n: int, marker: str, screen_chars: int) -> None:
        """Record the doubling so it survives the app closing.

        errors.log is the only channel that persists (the in-memory log buffer
        is level-gated and dies with the process), and the full screen dump goes
        to dup-paint.log next to it for diagnosis.
        """
        msg = (
            f"DUPLICATE PAINT: this message is on screen {n}x "
            f"(marker={marker!r}, screen={screen_chars} chars). "
            "A repaint is drawing it twice."
        )
        try:
            errors.error("tui.chat_bubble", msg, None)
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/chat_bubble.py:_dup_report errors")

    def _dup_dump(self, text: str) -> None:
        """Write the painted screen so the paint order can be inspected."""
        try:
            from ..globals import Path as GPath

            out = GPath.data / "dup-paint.log"
            with open(out, "a", encoding="utf-8") as fh:
                fh.write("=== duplicate paint dump ===\n")
                fh.write(text)
                fh.write("\n")
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/chat_bubble.py:_dup_dump")

    def _dup_check(self, text: str) -> None:
        """Given the painted screen text, report if this message appears twice."""
        try:
            marker = self._dup_marker()
            if not marker:
                return
            n = text.count(marker)
            if n > 1:
                self._dup_report(n, marker, len(text))
                self._dup_dump(text)
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/chat_bubble.py:_dup_check")

    def _probe_dup_on(self, text: str) -> None:
        """Test seam: run the check against a screen text you supply."""
        self._dup_check(text)

    def _probe_dup(self) -> None:
        """Read what is ACTUALLY on screen and check it for a second copy.

        Debug-only (OPENCODE_DEBUG_DUP). The text in `_message` is always fine,
        so the doubling can only be seen by reading the compositor's output.
        """
        try:
            strips = self.app.screen._compositor.render_strips()
            text = "\n".join(
                "".join(seg.text for seg in strip) for strip in strips
            )
            self._dup_check(text)
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/chat_bubble.py:_probe_dup")

    def freeze(self) -> None:
        """Mark this bubble finished: future _refresh calls are no-ops."""
        try:
            self.streaming = False
            self.frozen = True
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/chat_view.py:1282")
            pass


    def unfreeze(self) -> None:
        try:
            self.frozen = False
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/chat_view.py:1288")
            pass


    def watch_frozen(self, value: bool) -> None:
        # Entering frozen repaints once so the cached pixels match, then stays
        # silent. (The reactive setter already stored the value.)
        try:
            if value:
                self._apply_frame()
                self.update(self._build_content())
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/chat_view.py:1298")
            pass


    def _start_spinner(self) -> None:
        if self._timer is None:
            self._timer = self.set_interval(SPINNER_INTERVAL, self._tick)


    def _stop_spinner(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None


    def _tick(self) -> None:
        # Background-session chats stay mounted but invisible: spinning them
        # at 12.5fps steals UI-thread time from the visible screen (very
        # noticeable on phone CPUs). Skip the re-render until visible again.
        try:
            if not (self.is_attached and self.screen.is_current):
                return
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/chat_view.py:1317")
            pass
        # Adaptive pace: the frame only advances on fresh tokens
        # (note_stream_activity stamps _last_token_mono). Fast net -> full
        # 12fps spin; slow net -> ~6fps; stalled (>1.5s) -> frame holds.
        # ponytail: naive arrival-age heuristic; upgrade path is provider-side
        # token timestamps if they ever exist.
        try:
            last_tok = float(getattr(self, "_last_token_mono", 0.0) or 0.0)
            if last_tok > 0.0:
                age = time.monotonic() - last_tok
                if age > 1.5:
                    return
                if age > 0.5:
                    self._tick_slow = not getattr(self, "_tick_slow", False)
                    if self._tick_slow:
                        return
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/chat_view.py:1334")
            pass
        if self.role == "tool" and self._message.get("status") == "running":
            self._spinner = (self._spinner + 1) % len(SPINNER_FRAMES)
            self._refresh()
        elif self.role == "reasoning" and self.streaming:
            self._spinner = (self._spinner + 1) % len(SPINNER_FRAMES)
            self._refresh()


    def note_stream_activity(self, n_chars: int = 0) -> None:
        """Stamp token arrival (age paces the spinner, not the frame)."""
        try:
            self._last_token_mono = time.monotonic()
            self._token_chars = int(getattr(self, "_token_chars", 0) or 0) + max(0, int(n_chars or 0))
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/chat_view.py:1348")
            pass



class BubbleStreamMixin:
    """Mouse and keyboard handlers, tool metadata, and the
    streaming cursor (append / flush / end)."""

    def on_click(self, event: Any) -> None:
        if self.role == "reasoning":
            self.expanded = not self.expanded
        elif self.role == "tool":
            if self.content.get("tool") == "task":
                sid = self._task_sid()
                if sid:
                    self.post_message(OpenTaskSession(sid))
            elif self._tool_overflow():
                self.expanded = not self.expanded


    def on_key(self, event: Any) -> None:
        if self.role == "reasoning" and self.has_focus and event.key in ("enter", "space"):
            event.stop()
            self.expanded = not self.expanded
        elif self.role == "tool" and self.has_focus and event.key in ("enter", "space"):
            if self.content.get("tool") == "task":
                sid = self._task_sid()
                if sid:
                    event.stop()
                    self.post_message(OpenTaskSession(sid))


    def update_tool(self, tool_run: dict[str, Any]) -> None:
        self.unfreeze()
        self._message = tool_run
        if tool_run.get("status") == "running":
            self._start_spinner()
        else:
            self._stop_spinner()
            self.streaming = False
        self._refresh()
        if tool_run.get("status") not in ("running", "pending"):
            self.freeze()


    def set_tool_metadata(self, key: str, value: Any) -> None:
        """Merge one metadata entry into the tool run in place (keeps input,
        output and the other metadata), then re-render. The engine supplies the
        task row in a running -> completed sequence; this lets the TUI enrich it
        with live data (duration, toolcall count, retry state) without losing
        what the engine already wrote."""
        if not isinstance(self._message, dict):
            return
        self.unfreeze()
        meta = self._message.setdefault("metadata", {})
        meta[key] = value
        # _message and content are the SAME dict object (content property
        # returns self._message), so the click handler sees this immediately.
        # keep the parent chat's lookup index fresh (task rows gain sessionId
        # after mount; without this find_task rescans every event)
        try:
            parent = self.parent
            note = getattr(parent, "note_tool_metadata", None)
            if callable(note):
                note(self)
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/chat_view.py:1427")
            pass
        self._refresh()


    def pop_tool_metadata(self, key: str) -> None:
        if not isinstance(self._message, dict):
            return
        meta = self._message.get("metadata")
        if isinstance(meta, dict) and key in meta:
            self.unfreeze()
            del meta[key]
            self._refresh()


    def update_text(self, text: str) -> None:
        self.unfreeze()
        self._message = text
        self._refresh()
        self.freeze()


    def append_stream(self, text: str) -> None:
        """Append one streamed chunk (amortized O(1) render, not per-token)."""
        if text:
            # Pace the spinner by real arrivals: fast net -> full spin,
            # stalled net -> held frame (see _tick).
            try:
                self.note_stream_activity(len(text))
            except Exception as _oc_exc:
                swallow(_oc_exc, "tui/chat_view.py:1454")
                pass
            self._stream_parts.append(text)
            # O(1) append: flush into _message every 16 chunks instead of
            # re-joining the whole reply per token (O(n^2) on long answers).
            #
            # `_stream_parts` is the ONLY accumulator while streaming. It used
            # to ALSO write into `_message` on the sub-16 path, so every flush
            # re-appended chunks that were already there — duplicating the
            # reply (the user saw the same message painted twice). Readers must
            # use `_stream_text()` to see the tail; see flush_stream().
            if len(self._stream_parts) >= 16:
                try:
                    self._message = (self._message or "") + "".join(self._stream_parts) if isinstance(self._message, str) else "".join(self._stream_parts)
                except Exception:
                    self._message = "".join(self._stream_parts)
                del self._stream_parts[:]
            self._refresh_stream()


    def _refresh_stream(self) -> None:
        # Streaming render at most ~10/s: tokens arriving faster coalesce
        # into one layout instead of thrashing per chunk (long answers
        # stuttered more and more). Final text always lands via end_stream.
        # Tail-only: the streaming body is plain Text (see _build_content),
        # so only the NEW tail needs paint — Textual still does a full
        # widget update, but the renderable itself is O(new) plain glyphs,
        # never an O(reply) markdown re-parse. Same pixels at the end.
        try:
            import time as _time

            now = _time.monotonic()
            last = getattr(self, "_last_stream_mono", 0.0)
            if now - last < 0.1:
                return
            self._last_stream_mono = now
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/chat_view.py:1498")
            pass
        self._refresh()


    def end_stream(self) -> None:
        # flush buffered chunks so the final text is exact (bubble-level;
        # ChatView.end_stream(text) below finalizes the whole turn)
        self.flush_stream()
        self.streaming = False
        try:
            self._stop_spinner()
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/chat_view.py:1509")
            pass
        self._refresh()
        self.freeze()


    def _stream_text(self) -> str:
        """Everything received so far: the flushed part plus the live tail.

        `_stream_parts` holds deltas not yet folded into `_message`, so any
        render path must read BOTH or the newest words stay invisible until the
        next 16-chunk flush. This is the single accessor — never read
        `_message` alone while streaming.
        """
        try:
            base = self._message if isinstance(self._message, str) else ""
            return base + "".join(self._stream_parts or [])
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/chat_bubble.py:_stream_text")
            return str(self._message or "")

    def flush_stream(self) -> None:
        try:
            if getattr(self, "_stream_parts", None):
                try:
                    self._message = (self._message or "") + "".join(self._stream_parts) if isinstance(self._message, str) else "".join(self._stream_parts)
                except Exception:
                    self._message = "".join(self._stream_parts)
                del self._stream_parts[:]
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/chat_view.py:1522")
            pass


    def end_reasoning(self) -> None:
        # Fold the buffered deltas into _message before the flag flips: once
        # streaming is False the render path reads _message alone, so leaving
        # them in _stream_parts orphans the whole thought (empty title/body).
        self.flush_stream()
        self.streaming = False
        self._stop_spinner()
        if self._thought_started is not None:
            self._thought_seconds = time.monotonic() - self._thought_started
            self._thought_started = None
        # keep thoughts collapsed by default: show `+ Thought for 5.0s: <title>`
        # and let the user click / Enter / Ctrl+Shift+E to expand the full body
        self.expanded = False
        self._refresh()
        self.freeze()



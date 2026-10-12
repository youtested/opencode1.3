"""The message-bubble split must not leave moved code unimportable or unrunnable.

Why this file exists: splitting MessageBubble into chat_bubble.py / chat_parts.py
passed 939 tests and imported cleanly, then crashed the TUI on the first Enter
press — `NameError: name 're' is not defined` from reasoning_summary, reached
via MessageBubble(...) -> _refresh() -> _build_content() -> _build_reasoning().

Three gaps let that through, and each has a test below:

- importing a module never runs its function bodies, so a missing import inside
  a moved function is invisible to an import check;
- no test called the moved helpers with realistic arguments;
- no test built a bubble inside a running Textual app, which is the only place
  the theme/app-dependent branches (`meta`, `assistant_mode`) actually execute.
"""

from __future__ import annotations

import ast
import asyncio
import pathlib

import pytest

from opencode_py.tui import chat_bubble as B
from opencode_py.tui import chat_parts as P
from opencode_py.tui import chat_view as CV
from opencode_py.tui.chat_view import ChatView, MessageBubble

MIXINS = (
    B.BubbleCoreMixin,
    B.BubbleToolMixin,
    B.BubbleTaskMixin,
    B.BubbleFrameMixin,
    B.BubbleRefreshMixin,
    B.BubbleStreamMixin,
)
CHAT_DIR = pathlib.Path(__file__).resolve().parents[1] / "opencode_py" / "tui"

RUN = {"tool": "bash", "input": {"command": "pytest -q"}}


# ---------------------------------------------------- the bug that escaped


def test_reasoning_summary_runs_with_real_input():
    """The exact call that crashed the TUI: reasoning_summary with empty text."""
    assert P.reasoning_summary("") == {"title": None, "body": ""}
    out = P.reasoning_summary("**Checking the parser**\n\nit is a branch issue")
    assert out["title"] == "Checking the parser"
    assert out["body"] == "it is a branch issue"


def test_no_undefined_globals_in_the_split_modules():
    """Every global a moved function reads must exist in its own module.

    This is the check that catches the `re` class of bug: an import alone never
    touches function bodies, so a missing import only shows up at call time.
    """
    import builtins

    for name in ("chat_bubble.py", "chat_parts.py"):
        path = CHAT_DIR / name
        tree = ast.parse(path.read_text(encoding="utf-8"))
        bound = set(dir(builtins)) | {"__name__", "__file__", "__doc__", "self"}
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                bound.update(a.asname or a.name.split(".")[0] for a in node.names)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                bound.add(node.name)
            elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                bound.add(node.id)
            elif isinstance(node, ast.arg):
                bound.add(node.arg)
            elif isinstance(node, ast.ExceptHandler) and node.name:
                bound.add(node.name)
        missing = sorted({
            n.id for n in ast.walk(tree)
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id not in bound
        })
        assert not missing, f"{name} references undefined globals: {missing}"


# ------------------------------------------------ the moved module helpers


def test_moved_module_helpers_all_run():
    """Call every module-level helper that moved out of chat_view, for real."""
    assert B._plain("hello **world**") is not None
    assert B._safe_input(RUN) == {"command": "pytest -q"}
    # a string input is JSON-parsed; anything unparseable degrades to {}
    assert B._safe_input({"input": '{"path": "a.py"}'}) == {"path": "a.py"}
    assert B._safe_input({"input": "not json"}) == {}
    assert B._safe_input({}) == {}
    assert "pytest -q" in B._format_input(RUN["input"])
    assert B.tool_title(RUN) == "pytest -q"
    assert "Bash" in B.tool_label(RUN)
    assert B._tool_display("bash") == "bash"
    assert B._tool_display("not_a_tool") == "generic"
    assert B.collapse_tool_output("x" * 500, 10, 40)["overflow"] is True
    assert B.collapse_tool_output("short", 10, 40)["overflow"] is False
    assert B._render_diff("--- a\n+++ b\n-old\n+new") is not None
    assert isinstance(B._TOOL_TITLES, dict) and B._TOOL_TITLES
    assert isinstance(B._TOOL_DISPLAYS, set) and B._TOOL_DISPLAYS
    assert B.SPINNER_FRAMES and B.SPINNER_INTERVAL > 0
    assert B.format_subagent_title("build", "task")
    assert isinstance(B.format_subagent_toolcalls(2), str)
    assert isinstance(B.format_subagent_retry(1, "rate limited"), str)
    assert isinstance(B.format_completed_subagent_detail(3, "12s"), str)


# ------------------------------------------------------ the composed class


def test_every_mixin_is_composed_into_message_bubble():
    from textual.widgets import Static

    for mixin in MIXINS:
        assert issubclass(MessageBubble, mixin), mixin.__name__
    assert issubclass(MessageBubble, Static)


def test_every_moved_method_is_reachable():
    tree = ast.parse((CHAT_DIR / "chat_bubble.py").read_text(encoding="utf-8"))
    declared = {
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name.startswith("Bubble")
        for node in node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert len(declared) >= 60, f"only {len(declared)} methods in the mixins"
    for name in declared:
        assert hasattr(MessageBubble, name), f"{name} unreachable on the class"


def test_public_api_still_resolves_from_chat_view():
    """Other modules import these names FROM chat_view; they must stay there."""
    for name in ("ChatView", "MessageBubble", "TOOL_ICONS", "TOOL_NAMES",
                 "OpenTaskSession", "tool_title", "tool_label"):
        assert hasattr(CV, name), f"chat_view lost {name}"
    assert CV.TOOL_NAMES.get("find_symbols") == "Symbols"


def test_chatview_can_reach_the_shared_helper():
    assert CV.reasoning_summary("**T**\n\nb")["title"] == "T"


# --------------------------------------------- the crash, in a live app


def _host_app():
    from textual.app import App

    class _Host(App):
        def compose(self):
            yield ChatView()

    return _Host()


@pytest.mark.parametrize("role,content", [
    ("reasoning", ""),
    ("reasoning", "**Title**\n\nbody"),
    ("assistant", "hello **world**"),
    ("user", "hi"),
    ("meta", ""),
    ("assistant_mode", ""),
])
def test_every_role_renders_in_a_running_app(role, content):
    """`meta` and `assistant_mode` read the app theme, so they only execute
    inside a live Textual app — which is why the plain unit suite missed them."""

    async def go():
        app = _host_app()
        async with app.run_test(size=(80, 30)) as pilot:
            chat = app.query_one(ChatView)
            bubble = MessageBubble(role, content)
            await chat.mount(bubble)
            bubble._refresh()
            await pilot.pause()
            assert bubble.content is not None

    asyncio.run(go())


def test_begin_thinking_is_the_exact_reported_crash():
    """prompt.py:_start_turn -> chat.begin_thinking() -> MessageBubble(...)
    -> _refresh() -> _build_content() -> _build_reasoning(): the reported path."""

    async def go():
        app = _host_app()
        async with app.run_test(size=(80, 30)) as pilot:
            chat = app.query_one(ChatView)
            chat.begin_thinking()          # <- this raised NameError: re
            await pilot.pause()
            assert chat._reasoning_bubble is not None
            assert chat._reasoning_bubble.content is not None

    asyncio.run(go())


# ----------------------------------- every renderer the split moved, executed
#
# 57 of the 69 moved methods were not called by any test, so a missing name in
# a renderer would only surface when that tool is used. `_render_tool` dispatches
# by name, so one call per tool covers both the dispatch and the renderer.

TOOL_RUNS = [
    ("bash", {"command": "pytest -q"}, "collected 42 items\n42 passed"),
    ("read", {"path": "opencode_py/main.py"}, "line 1\nline 2"),
    ("glob", {"pattern": "**/*.py", "path": "."}, "a.py\nb.py"),
    ("grep", {"pattern": "def run", "path": "."}, "x.py:12:def run()"),
    ("find_symbols", {"query": "run", "kind": "function"}, "run -> main.py:10"),
    ("webfetch", {"url": "https://example.com"}, "<html>body</html>"),
    ("webfetch_many", {"urls": ["https://a.example"]}, "a.example -> 200"),
    ("lsp", {"operation": "findReferences"}, "3 references"),
    ("websearch", {"query": "termux python"}, "result 1"),
    ("write", {"path": "a.py", "content": "x"}, "wrote 1 line"),
    ("edit", {"path": "a.py"}, "--- a\n+++ b\n-old\n+new"),
    ("apply_patch", {"path": "a.py"}, "--- a\n+++ b\n-old\n+new"),
    ("todowrite", {"todos": [{"content": "do it", "status": "pending"}]}, "1 todo"),
    ("question", {"questions": [{"question": "which?", "header": "pick"}]}, "which?"),
    ("execute", {"command": "ls"}, "a\nb"),
    ("skill", {"name": "dataviz"}, "loaded skill"),
    ("not_a_real_tool", {"x": 1}, "fell back to generic"),
]


@pytest.mark.parametrize("tool,args,output", TOOL_RUNS)
def test_every_tool_renderer_runs(tool, args, output):
    """Exercise each per-tool renderer inside a live app."""

    async def go():
        app = _host_app()
        async with app.run_test(size=(80, 30)) as pilot:
            chat = app.query_one(ChatView)
            bubble = MessageBubble("tool", {"tool": tool, "input": args,
                                            "output": output, "status": "completed"})
            await chat.mount(bubble)
            rendered = bubble._render_tool(bubble._message)
            assert rendered is not None, tool
            # the same renderer with a running status takes a different branch
            bubble._message["status"] = "running"
            assert bubble._render_tool(bubble._message) is not None, tool
            await pilot.pause()

    asyncio.run(go())


def test_frame_and_task_renderers_run():
    """The remaining moved renderers, including the sub-agent task row."""

    async def go():
        app = _host_app()
        async with app.run_test(size=(80, 30)) as pilot:
            chat = app.query_one(ChatView)
            bubble = MessageBubble("tool", {"tool": "task", "input": {},
                                            "output": "done", "status": "completed"})
            await chat.mount(bubble)
            assert bubble._render_task(bubble._message) is not None
            assert bubble._apply_frame() is None
            # helpers that read app/task state must degrade, not raise
            bubble._task_parent_chat()
            bubble._is_last_task_row()
            bubble._task_child_chat("")
            bubble._task_child_runs("")
            bubble._task_sid()
            bubble._content_width()
            bubble._tool_block()
            bubble._tool_collapse({"output": "x" * 200})
            bubble.update_text("x" * 5000)
            assert isinstance(bubble._tool_overflow(), bool)
            bubble._build_compaction()
            await pilot.pause()

    asyncio.run(go())


def test_tool_metadata_and_text_updates_run():
    """Stream/tool metadata writers — the ones the TUI calls on every event."""

    async def go():
        app = _host_app()
        async with app.run_test(size=(80, 30)) as pilot:
            chat = app.query_one(ChatView)
            bubble = MessageBubble("assistant", "start")
            await chat.mount(bubble)
            bubble.set_tool_metadata("call-1", {"note": "n"})
            bubble.update_tool({"tool": "read", "input": {"path": "a.py"}})
            bubble.pop_tool_metadata("call-1")
            bubble.update_text("replaced")
            bubble.note_stream_activity()
            bubble._start_spinner()
            bubble._tick()
            bubble._stop_spinner()
            bubble.freeze()
            assert bubble.frozen is True
            bubble.unfreeze()
            assert bubble.frozen is False
            bubble.flush_stream()
            bubble.end_reasoning()
            await pilot.pause()

    asyncio.run(go())


def test_streaming_lifecycle_end_to_end():
    """append_stream -> _refresh_stream -> end_stream, the live streaming path."""

    async def go():
        app = _host_app()
        async with app.run_test(size=(80, 30)) as pilot:
            chat = app.query_one(ChatView)
            bubble = MessageBubble("assistant", "")
            await chat.mount(bubble)
            bubble.streaming = True
            bubble.append_stream("par")
            bubble.append_stream("tial ")
            bubble.append_stream("answer")
            bubble._refresh_stream()
            assert "partial answer" in str(bubble.content)
            bubble.end_stream()
            bubble.streaming = False
            await pilot.pause()
            assert "partial answer" in str(bubble.content)

    asyncio.run(go())


def test_row_helpers_and_reactive_watchers_run():
    """The small status/row helpers, the reactive watchers and the click path."""

    async def go():
        app = _host_app()
        async with app.run_test(size=(80, 30)) as pilot:
            chat = app.query_one(ChatView)
            bubble = MessageBubble("tool", {"tool": "bash", "input": {"command": "ls"},
                                            "output": "a\nb", "status": "completed"})
            await chat.mount(bubble)
            run = bubble._message
            for status in ("running", "completed", "denied", "failed", "error"):
                run["status"] = status
                assert bubble._status(run) is not None
                assert bubble._error(run) is not None
                assert bubble._denied(run) is not None
                assert bubble._failed(run) is not None
            assert bubble._inline(run, False, "label") is not None
            bubble._error_line(run)   # None for a non-error status is fine; must not raise
            assert bubble._diff_opts() is not None

            # reactive watchers
            bubble.queued = True
            bubble.streaming = True
            bubble.expanded = True
            bubble.selected = True
            bubble.frozen = True
            await pilot.pause()
            bubble.frozen = False
            await pilot.pause()

            # keyboard/mouse handlers must not raise
            # on_click/on_key expect a Textual event; None makes them bail
            bubble.on_click(None)
            bubble.on_key(None)

            # sub-agent helpers degrade rather than raise when nothing is attached
            bubble._task_child_session("missing-session")
            bubble._task_render_duration("missing", {})
            bubble._task_render_toolcalls("missing", {}, [])

            await bubble.remove()
            await pilot.pause()

    asyncio.run(go())


def test_debug_only_dup_probe_helpers_run():
    """The OPENCODE_DEBUG_DUP probe moved too; its helpers must still work."""

    async def go():
        app = _host_app()
        async with app.run_test(size=(80, 30)) as pilot:
            chat = app.query_one(ChatView)
            bubble = MessageBubble("assistant", "hello")
            await chat.mount(bubble)
            assert bubble._dup_marker() is not None
            bubble._dup_check("hello world hello world")
            bubble._probe_dup_on("hello world hello world")
            await pilot.pause()

    asyncio.run(go())


# ------------------------------------------- streamed text must not vanish

def test_end_reasoning_flushes_the_thought_body():
    """`end_reasoning()` used to flip streaming=False without folding
    `_stream_parts` into `_message`. The render path then reads `_message`
    alone, so the thought collapsed to a bare `+ Thought` with no title and
    nothing to expand — the reasoning was still in the buffer, unpainted."""

    async def go():
        app = _host_app()
        async with app.run_test(size=(80, 30)) as pilot:
            chat = app.query_one(ChatView)
            bubble = MessageBubble("reasoning", "")
            await chat.mount(bubble)
            bubble.streaming = True
            bubble.append_stream("**Checking the buffer**\n\nthe tail")
            bubble.end_reasoning()
            assert bubble._stream_parts == []
            assert bubble._message == "**Checking the buffer**\n\nthe tail"
            assert B.reasoning_summary(bubble._message)["title"] == "Checking the buffer"
            bubble.expanded = True
            assert "the tail" in bubble._build_reasoning().renderables[1].plain
            await pilot.pause()

    asyncio.run(go())


def test_remove_last_stream_bubble_flushes_the_briefing():
    """The tool-call path (events.py) calls remove_last_stream_bubble() after
    every step. It kept a bubble holding text, but only turned the cursor off,
    so the one-line briefing under the thought painted as a blank row."""

    async def go():
        app = _host_app()
        async with app.run_test(size=(80, 30)) as pilot:
            chat = app.query_one(ChatView)
            bubble = MessageBubble("assistant", "")
            await chat.mount(bubble)
            bubble.streaming = True
            bubble.append_stream("Found 3 files, reading login.py")
            chat._stream_bubble = bubble
            chat.remove_last_stream_bubble()
            assert bubble._stream_parts == []
            assert bubble._message == "Found 3 files, reading login.py"
            assert "login.py" in bubble._build_content().renderables[0].plain
            await pilot.pause()

    asyncio.run(go())

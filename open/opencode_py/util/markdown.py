"""Rich-based markdown rendering for terminal output (mirrors opencode's markdown view)."""

from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown


def render_markdown(text: str, width: int | None = None) -> str:
    """Render markdown to a plain-text/ANSI string suitable for printing."""
    console = Console(width=width, force_terminal=True, soft_wrap=False)
    md = Markdown(text, code_theme="monokai")
    with console.capture() as capture:
        console.print(md)
    return capture.get()





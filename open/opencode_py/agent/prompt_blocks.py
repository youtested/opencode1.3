"""Prompt-block rendering, shared by the agent and the skill tool.

``_block_template`` renders a fenced block out of a ``.md`` template that sits
next to this module. It lived in ``agent/system.py``, which the skill tool
imported — a two-way dependency between ``agent`` and ``tools``. Both live in
this leaf now, so neither module needs the other.

``_PROMPT_DIR`` is this file's own directory, which is still ``opencode_py/agent/``,
so the ``.md`` files resolve exactly as before.
"""

from __future__ import annotations

from pathlib import Path

_PROMPT_DIR = Path(__file__).resolve().parent


def _read_prompt_file(name: str) -> str:
    """Read a prompt .md next to this module. Raises FileNotFoundError
    with a clear message when missing — no hardcoded prompt text lives
    in code anymore."""
    try:
        text = (_PROMPT_DIR / name).read_text(encoding="utf-8").strip()
    except OSError as e:
        raise FileNotFoundError(
            f"prompt file missing: {_PROMPT_DIR / name} ({e})"
        ) from e
    if not text:
        raise ValueError(f"prompt file empty: {_PROMPT_DIR / name}")
    return text


def _block_template(name: str, placeholder: str, content: str) -> str:
    """Render a prompt block from its .md template file.

    The file holds a fenced example showing the shape; the fence body is
    the template. No hardcoded copy lives here — the file is the source.
    """
    template = _read_prompt_file(name)
    body = template
    lines = template.splitlines()
    start = next((i for i, l in enumerate(lines) if l.strip() == "```"), None)
    end = next(
        (i for i, l in enumerate(lines) if l.strip() == "```" and i != start),
        None,
    ) if start is not None else None
    if start is not None and end is not None and end > start:
        body = "\n".join(lines[start + 1 : end])
    return body.replace(placeholder, content)

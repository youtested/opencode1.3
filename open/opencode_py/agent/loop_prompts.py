"""Prompt queueing and provider-rotation dirty-marking.

Split out of :mod:`opencode_py.agent.loop` as a mixin; bodies byte-for-byte unchanged."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..providers import build_rotation

if TYPE_CHECKING:  # pragma: no cover - annotation only
    from .loop import AgentLoop


class LoopPromptsMixin:
    def find_subagent(self, session_id: str) -> "AgentLoop | None":
        """Depth-first search for a sub-agent by session id (nested included)."""
        stack = [self]
        while stack:
            loop = stack.pop()
            if loop.session_id == session_id:
                return loop
            stack.extend(loop.subagents.values())
        return None

    def queue_prompt(self, text: str) -> int:
        """Add a prompt submitted while a turn is running.

        It sits in the FIFO until the running turn reaches its next
        provider-turn boundary, then the engine folds it into the SAME turn
        (opencode's Session Drain). Returns the queue depth after adding.

        Bounded: at most MAX_PROMPT_QUEUE prompts are kept (oldest dropped
        first) and each prompt is truncated to MAX_PROMPT_CHARS, so typing
        during a long turn cannot grow memory without bound.
        """
        text = str(text or "")
        if len(text) > self.MAX_PROMPT_CHARS:
            text = text[: self.MAX_PROMPT_CHARS]
        with self._prompt_lock:
            while len(self._prompt_queue) >= self.MAX_PROMPT_QUEUE:
                self._prompt_queue.pop(0)
            self._prompt_queue.append(text)
            return len(self._prompt_queue)

    def prompt_pending(self) -> int:
        with self._prompt_lock:
            return len(self._prompt_queue)


    def pop_prompt(self) -> str | None:
        """Pop the oldest queued prompt (used by the TUI for the leftover after a
        drain ends; run_turn itself uses ``_next_prompt`` internally)."""
        return self._next_prompt()

    def _next_prompt(self) -> str | None:
        """Pop the oldest queued prompt (engine thread, run_turn)."""
        with self._prompt_lock:
            if not self._prompt_queue:
                return None
            return self._prompt_queue.pop(0)

    def clear_prompts(self) -> None:
        with self._prompt_lock:
            self._prompt_queue.clear()

    def rebuild_rotation(self) -> None:
        """Rebuild the failover lanes from the current config.

        The rotation is built once at startup, so picking a different model or
        provider at runtime would otherwise keep using the old lanes. Call this
        whenever `cfg.model` / `cfg.provider` change so the next turn uses the
        newly picked model.
        """
        self.rotation = build_rotation(self.cfg, self.auth, self._session_id)

    def mark_rotation_dirty(self) -> None:
        """Defer rebuild_rotation() to the start of the next turn (engine thread).

        build_rotation() can hit the network (catalog refresh), so callers on
        the UI thread (turn done, model pick, settings) flag it instead of
        rebuilding inline — an inline rebuild there froze the entire screen
        whenever the models cache was stale and the network slow.
        """
        self._rotation_dirty = True

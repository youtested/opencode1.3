"""Permission + question dialogs and focus handling for the Textual TUI.

Split out of :mod:`opencode_py.tui.app` as a mixin; method bodies are unchanged.
"""
from __future__ import annotations

from ..util.log import swallow
import threading
import time
from typing import TYPE_CHECKING, Any

from .app_meta import _DIALOG_TIMEOUT
from .input_bar import InputBar

if TYPE_CHECKING:
    from ..question import QuestionInfo

class DialogsMixin:

    # -- auto-refocus -------------------------------------------------------
    def _is_main_screen_active(self) -> bool:
        """True when the normal chat screen is on top (no modal / app exiting)."""
        try:
            return self._main_screen is not None and self.screen is self._main_screen
        except Exception:
            return False


    def on_descendant_focus(self, event: Any) -> None:
        """Drag the prompt cursor back ~1s after focus leaves the input box.

        Tapping a reasoning bubble (to expand it) or the chat area on a phone
        steals focus, so the blinking cursor disappears and typing stops. Any
        focus change outside the prompt arms a short timer; if the user touches
        nothing else in the meantime, the cursor comes back by itself."""
        if self._refocus_timer is not None:
            self._refocus_timer.stop()
            self._refocus_timer = None
        if not self._is_main_screen_active():
            # a picker / dialog on top, or the app is shutting down — never
            # steal focus or arm timers
            return
        try:
            if self.query_one(InputBar).input.has_focus:
                return
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/dialogs.py:45")
            return
        self._refocus_timer = self.set_timer(1.0, self._refocus_prompt)


    def _refocus_prompt(self) -> None:
        self._refocus_timer = None
        if not self._is_main_screen_active():
            return
        try:
            self.query_one(InputBar).input.focus()
        except Exception as _oc_exc:
            swallow(_oc_exc, "tui/dialogs.py:56")
            pass


    # -- permission dialog (engine thread -> UI) --------------------------
    def _permission_ask(self, description: str, always_patterns: list[str]) -> str:
        """Bridge the engine thread's permission.ask to a modal dialog.

        Runs on the engine worker thread. Pushes the dialog on the UI thread via
        call_from_thread (which blocks until the push returns), then waits for the
        user's decision. Returns "once" / "always" / "reject".
        """
        outcome: dict[str, str] = {}
        decided = threading.Event()

        def on_decision(decision: str) -> None:
            outcome["decision"] = decision
            decided.set()

        with self._dialog_lock:
            try:
                self.call_from_thread(
                    self._show_permission_dialog, description, on_decision
                )
            except Exception:
                return "reject"
            # Wait for user decision with timeout; if timeout expires, default to "reject"
            # to avoid hanging the engine thread indefinitely if UI is unresponsive.
            # The window starts AFTER the dialog is shown (lock held), so queued
            # agents each get their full 30s instead of timing out behind others.
            # A force-stop (2nd ESC) breaks the wait immediately as "reject" —
            # the worker must never sit out the full timeout after STOP.
            start = time.monotonic()
            force = getattr(self, "_force_stop", None)
            while not self._exit_requested.is_set():
                if force is not None and force.is_set():
                    break
                remaining = _DIALOG_TIMEOUT - (time.monotonic() - start)
                if remaining <= 0:
                    break
                if decided.wait(timeout=min(0.5, remaining)):
                    break
        return outcome.get("decision", "reject")


    def _show_permission_dialog(
        self, description: str, on_decision: Any
    ) -> None:
        if not self.is_attached:
            on_decision("deny")
            return
        from .permission_dialog import PermissionDialog

        self.push_screen(PermissionDialog(description, on_decision=on_decision))


    # -- question dialog (engine thread -> UI) ----------------------------
    def _question_ask(self, questions: list[QuestionInfo]) -> list[list[str]]:
        """Bridge the engine thread's question.ask to a modal dialog.

        Runs on the engine worker thread, mirroring ``_permission_ask``.
        Returns the answers (list of list[str], one per question) or raises
        QuestionRejectedError when the user dismisses / the app is quitting.
        """
        from ..question import QuestionRejectedError

        result: dict[str, Any] = {}
        answered = threading.Event()

        def on_done(answers: list[list[str]] | None) -> None:
            result["answers"] = answers
            answered.set()

        with self._dialog_lock:
            try:
                self.call_from_thread(
                    self._show_question_dialog, questions, on_done
                )
            except Exception:
                raise QuestionRejectedError("no UI to ask the user") from None
            # Wait for user answers with timeout; if timeout expires, treat as dismissed.
            # A force-stop (2nd ESC) breaks the wait immediately as dismissed.
            start = time.monotonic()
            force = getattr(self, "_force_stop", None)
            while not self._exit_requested.is_set():
                if force is not None and force.is_set():
                    break
                remaining = _DIALOG_TIMEOUT - (time.monotonic() - start)
                if remaining <= 0:
                    break
                if answered.wait(timeout=min(0.5, remaining)):
                    break
        answers = result.get("answers")
        if answers is None:
            raise QuestionRejectedError("user dismissed the question")
        return answers


    def _show_question_dialog(
        self, questions: list[QuestionInfo], on_done: Any
    ) -> None:
        if not self.is_attached:
            on_done(None)
            return
        from .question_dialog import QuestionDialog

        self.push_screen(QuestionDialog(questions, on_done=on_done))

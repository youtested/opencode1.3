"""Regression: Enter on an empty prompt must not open the model picker.

Enter and Ctrl+M are the same byte (0x0D) on a terminal and Textual reports
that byte as "enter" only, so the prompt used to hijack empty-Enter as its
Ctrl+M shortcut and pop the picker on a stray keypress.  The picker now lives
on a binding a terminal can really send.
"""

import asyncio

from textual.app import App, ComposeResult

from opencode_py.tui.app import OpenCodeTUI
from opencode_py.tui.input_bar import InputBar, PromptSubmitted

PASTE_GAP = 0.025


class _Harness(App):
    """Mounts the real InputBar under the real app key bindings."""

    BINDINGS = OpenCodeTUI.BINDINGS

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.models_opened = 0
        self.models_requested = 0
        self.submitted: list[str] = []

    def compose(self) -> ComposeResult:
        yield InputBar()

    def action_models(self) -> None:
        self.models_opened += 1

    def on_prompt_submitted(self, event: PromptSubmitted) -> None:
        self.submitted.append(event.value)

    def on_models_requested(self, event) -> None:
        self.models_requested += 1


def _models_binding_key() -> str:
    for binding in OpenCodeTUI.BINDINGS:
        if binding.action == "models":
            return binding.key
    raise AssertionError("no models binding on the app")


async def test_empty_enter_does_nothing():
    app = _Harness()
    async with app.run_test() as pilot:
        bar = app.query_one(InputBar)
        await pilot.pause()
        assert app.focused is bar.input

        await pilot.press("enter")
        await pilot.pause()

        assert app.models_opened == 0, "Enter opened the model picker"
        assert app.models_requested == 0, "Enter asked for the model picker"
        assert app.submitted == [], "empty Enter submitted a prompt"
        assert bar.input.text == "", "empty Enter wrote into the prompt"


async def test_enter_with_text_still_submits():
    app = _Harness()
    async with app.run_test() as pilot:
        bar = app.query_one(InputBar)
        await pilot.pause()

        await pilot.press(*"hello")
        await asyncio.sleep(PASTE_GAP * 3)
        await pilot.press("enter")
        await pilot.pause()

        assert app.submitted == ["hello"], app.submitted
        assert app.models_opened == 0, "a real prompt opened the model picker"
        assert bar.input.text == "", "a sent prompt must clear the box"


async def test_models_shortcut_is_reachable_from_a_terminal():
    key = _models_binding_key()
    assert key not in ("ctrl+m", "enter"), (
        f"{key!r} is the Enter byte, so a terminal can never send it alone"
    )

    app = _Harness()
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press(key)
        await pilot.pause()
        assert app.models_opened == 1, f"{key!r} did not open the model picker"

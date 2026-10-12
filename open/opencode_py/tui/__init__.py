"""Textual TUI package."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .app import OpenCodeTUI, run_tui

# Lazy (PEP 562): config.py imports tui.theme for the custom-theme registry,
# and tui.theme itself needs no textual. Eagerly importing .app here made
# every load_config() call require textual, so a plain `pip install .`
# crashed all three console scripts.


def __getattr__(name):
    if name in ("OpenCodeTUI", "run_tui"):
        from .app import OpenCodeTUI, run_tui
        return {"OpenCodeTUI": OpenCodeTUI, "run_tui": run_tui}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["OpenCodeTUI", "run_tui"]

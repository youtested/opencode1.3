"""Screen-capture (screen_view tool) helpers for the Textual TUI.

Split out of :mod:`opencode_py.tui.app` as a mixin; method bodies are unchanged.
"""
from __future__ import annotations

import threading

from .app_meta import _MAX_LINES_CAP, _MAX_WIDGET_LINES

class CaptureMixin:

    # -- screen capture (screen_view tool) ---------------------------------

    def _capture_widget_tree(self) -> dict:
        """The bones under the screen: one line per widget, depth-first.

        This is the ui_probe view: type, #id, CSS classes, exact position and
        size in cells, plus FOCUSED / hidden / zero-size markers — so a layout
        bug can be traced to the specific broken widget instead of guessed
        from pixels. Read-only and bounded; a misbehaving widget's properties
        can never crash the capture.
        """
        focused = self.focused
        lines: list[str] = []
        count = 0

        def _safe(fn, default):
            try:
                return fn()
            except Exception:  # pragma: no cover - defensive per-widget guard
                return default

        try:
            nodes = [self.screen] + list(self.screen.walk_children())
        except Exception as e:  # pragma: no cover - defensive
            return {"output": f"Widget tree walk failed: {e}", "error": True}

        for depth, w in enumerate(nodes):
            if count >= _MAX_WIDGET_LINES:
                lines.append(f"… ({len(nodes) - count} more widgets not shown)")
                break
            indent = "  " * min(depth, 12)
            name = type(w).__name__
            ident = f" #{w.id}" if getattr(w, "id", None) else ""
            classes = list(_safe(lambda: w.classes, []) or [])
            css = (" ." + ".".join(classes)) if classes else ""
            region = _safe(lambda: w.region, None)
            geo = ""
            if region is not None:
                geo = f"  ({region.x},{region.y} {region.width}×{region.height})"
            markers = ""
            if focused is not None and w is focused:
                markers += " ▸FOCUSED"
            display = _safe(lambda: str(w.styles.display), "block")
            if display == "none":
                markers += " ✗hidden"
            elif region is not None and (region.width == 0 or region.height == 0):
                markers += " ␀zero-size"
            lines.append(f"{indent}{name}{ident}{css}{geo}{markers}")
            count += 1
        return {
            "output": (
                "\n".join(lines)
                + f"\n[{count} widgets, screen {self.size.width}x{self.size.height}]"
            ),
            "metadata": {
                "count": count,
                "focused": type(focused).__name__ if focused else None,
            },
        }


    def _capture_for_model(self, action: str) -> dict:
        """Bridge a worker-thread tool call to the app thread.

        Tool calls run on engine worker threads; the compositor must be read
        on the app thread. When the caller already IS the app thread (tests,
        on_mount-time queries), render directly instead of deadlocking.
        """
        if getattr(self, "_thread_id", None) == threading.get_ident():
            return self._capture_on_app_thread(action)
        return self.call_from_thread(self._capture_on_app_thread, action)


    def _capture_on_app_thread(self, action: str) -> dict:
        if action == "widgets":
            return self._capture_widget_tree()
        if action == "info":
            focused = self.focused
            return {
                "output": (
                    f"Terminal: {self.size.width}x{self.size.height} cells\n"
                    f"App title: {self.title}\n"
                    f"Screen: {type(self.screen).__name__}\n"
                    f"Focused: {type(focused).__name__}"
                    + (f" (id={focused.id})" if focused is not None and focused.id else "")
                ),
                "metadata": {
                    "width": self.size.width,
                    "height": self.size.height,
                    "focused": type(focused).__name__ if focused else None,
                },
            }
        # action == "text": render the full visible screen as plain text rows
        try:
            strips = self.screen._compositor.render_strips()
        except Exception as e:  # pragma: no cover - defensive
            return {"output": f"Screen render failed: {e}", "error": True}
        lines = [strip.text for strip in strips]
        # trim trailing blank rows / right padding so the model sees layout,
        # not hundreds of spaces
        while lines and not lines[-1].strip():
            lines.pop()
        lines = [ln.rstrip() for ln in lines]
        truncated = False
        if len(lines) > _MAX_LINES_CAP:
            lines = lines[:_MAX_LINES_CAP]
            truncated = True
        body = "\n".join(lines) if lines else "(empty screen)"
        footer = (
            f"\n[{self.size.width}x{self.size.height} cells"
            + (", truncated" if truncated else "")
            + "]"
        )
        return {
            "output": body + footer,
            "metadata": {"width": self.size.width, "height": self.size.height},
        }

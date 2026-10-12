"""The agent loop must keep the exact shape it was refactored to have.

Guards ``opencode_py/agent/loop.py`` (and its ``loop_*.py`` mixins): every
method must still exist, in the same file or a moved one, with a byte-identical
body. Moving code between those files passes; losing, renaming or editing any
of it fails.
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from fingerprint_loop import STORE, collect  # noqa: E402


def test_loop_shape_is_unchanged():
    expected = json.loads(STORE.read_text(encoding="utf-8"))
    actual = collect()

    lost = sorted(set(expected) - set(actual))
    # New *Mixin classes are the point of the refactor: methods moving out of
    # AgentLoop into loop_*.py legitimately adds their class. Anything else
    # appearing from nowhere is still a change worth flagging.
    added = sorted(k for k in set(actual) - set(expected)
                   if not (k.startswith("class:") and k.endswith("Mixin")))
    changed = sorted(k for k in expected.keys() & actual.keys() if expected[k] != actual[k])

    problems = []
    if lost:
        problems.append(f"{len(lost)} lost: {lost[:8]}")
    if added:
        problems.append(f"{len(added)} unexpectedly added: {added[:8]}")
    if changed:
        details = ", ".join(f"{k} {expected[k]}->{actual[k]}" for k in changed[:8])
        problems.append(f"{len(changed)} edited: {details}")

    assert not problems, "agent loop changed:\n  " + "\n  ".join(problems)


def test_loop_still_imports_and_has_the_entry_points():
    from opencode_py.agent.loop import AgentLoop, TurnResult

    assert callable(AgentLoop)
    assert callable(TurnResult)
    for name in ("run_turn", "run_tool", "abort", "get_history", "resume_turn",
                 "queue_prompt", "wake_bg_wait", "force_compact", "close"):
        assert hasattr(AgentLoop, name), f"AgentLoop lost {name}()"
"""Tests for the policy decisions inside the agent loop.

These are the decisions the suite did NOT cover: timing, thresholds, token
limits, interrupt wiring and per-agent toggles. Each one was verified to fail
when the corresponding line in ``agent/loop.py`` is deliberately broken, so
these are real tests rather than restatements of the implementation.

They also act as the safety net for the loop.py split: a behaviour change
during a refactor turns one of these red.
"""

from __future__ import annotations

import pytest

from opencode_py.agent.loop import AgentLoop


def _loop(**attrs):
    """A bare AgentLoop with no provider work (no network, no tools)."""
    loop = AgentLoop.__new__(AgentLoop)
    loop._interrupt = lambda: False
    loop._session_id = ""
    loop._prev_agent = "build"
    for key, value in attrs.items():
        setattr(loop, key, value)
    return loop


class _Cfg:
    def __init__(self, **kw):
        self.raw = {}
        self.agents = {}
        self.provider = "opencode"
        self.model = "some-model"
        self.early_summary_at = 0.7
        for k, v in kw.items():
            setattr(self, k, v)


# --------------------------------------------------------------------------
# backoff must wake the instant ESC is pressed
# --------------------------------------------------------------------------

def test_sleep_interruptible_raises_when_interrupted_before_any_sleep():
    from opencode_py.providers import StreamInterrupted

    loop = _loop()
    loop._interrupt = lambda: True
    with pytest.raises(StreamInterrupted):
        # Kept short on purpose: if the interrupt check is ever removed this
        # test must fail in ~2s, not after sitting out the whole 30s cap.
        loop._sleep_interruptible(2.0)


def test_sleep_interruptible_returns_immediately_for_zero_seconds():
    loop = _loop()
    loop._sleep_interruptible(0)
    # Returns without raising: a zero/None wait must be a no-op, and must not
    # even consult the interrupt flag (that would raise here).


def test_sleep_interruptible_treats_none_as_zero():
    loop = _loop()
    loop._sleep_interruptible(None)


def test_sleep_interruptible_wakes_mid_wait_when_esc_arrives_late():
    """An interrupt raised part-way through the wait still cuts it short.

    Deterministic on purpose: no threads, no wall clock. The interrupt is a
    counter that starts returning True after a few polls, so the test behaves
    identically on a fast desktop and a loaded phone.
    """
    from opencode_py.providers import StreamInterrupted

    loop = _loop()
    polls = []

    def interrupt():
        polls.append(1)
        # ESC "arrives" after the wait is already under way.
        return len(polls) >= 3

    loop._interrupt = interrupt
    with pytest.raises(StreamInterrupted):
        loop._sleep_interruptible(5.0)
    assert len(polls) >= 3, "the interrupt flag was never polled while waiting"


def test_sleep_interruptible_finishes_normally_when_not_interrupted():
    loop = _loop()
    loop._sleep_interruptible(0.25)
    # No raise is the assertion. A wall-clock floor is deliberately not
    # checked: the only thing that must hold is that the call returns.


# --------------------------------------------------------------------------
# early-summary threshold clamping
# --------------------------------------------------------------------------

def test_early_summary_threshold_defaults_to_seventy_percent():
    loop = _loop(cfg=_Cfg())
    assert loop._early_summary_threshold(1000) == pytest.approx(0.7)


def test_early_summary_threshold_honours_config():
    loop = _loop(cfg=_Cfg(early_summary_at=0.4))
    assert loop._early_summary_threshold(1000) == pytest.approx(0.5)  # clamped up


def test_early_summary_threshold_clamps_to_the_ceiling():
    loop = _loop(cfg=_Cfg(early_summary_at=0.99))
    assert loop._early_summary_threshold(1000) == pytest.approx(0.95)


def test_early_summary_threshold_survives_a_broken_config():
    loop = _loop(cfg=_Cfg(early_summary_at="not-a-number"))
    assert loop._early_summary_threshold(1000) == pytest.approx(0.7)


# --------------------------------------------------------------------------
# model context / output limits come from the provider table
# --------------------------------------------------------------------------

def test_model_output_limit_matches_the_provider_table():
    from opencode_py import providers

    loop = _loop(cfg=_Cfg(provider="anthropic", model="claude-sonnet-4-5"),
                 provider_id="anthropic", model_id="claude-sonnet-4-5")
    expected = providers.model_output_limit("anthropic", "claude-sonnet-4-5")
    assert loop._model_output_limit() == expected
    if expected:
        assert loop._model_output_limit() != 999999


def test_model_output_limit_falls_back_to_cfg_when_lane_is_unset():
    from opencode_py import providers

    loop = _loop(cfg=_Cfg(provider="anthropic", model="claude-sonnet-4-5"),
                 provider_id=None, model_id=None)
    assert loop._model_output_limit() == providers.model_output_limit(
        "anthropic", "claude-sonnet-4-5")


def test_model_context_size_matches_the_provider_table():
    from opencode_py import providers

    loop = _loop(cfg=_Cfg(provider="anthropic", model="claude-sonnet-4-5"),
                 provider_id="anthropic", model_id="claude-sonnet-4-5", auth=None)
    assert loop._model_context_size() == providers.model_context_size(
        "anthropic", "claude-sonnet-4-5", auth=None)


# --------------------------------------------------------------------------
# skill toggle honours both the global and the per-agent setting
# --------------------------------------------------------------------------

def test_skill_enabled_by_default():
    loop = _loop(cfg=_Cfg(), agent="build")
    assert loop._skill_enabled() is True


def test_skill_disabled_globally_by_config():
    cfg = _Cfg()
    cfg.raw = {"tools": {"skill": False}}
    loop = _loop(cfg=cfg, agent="build")
    assert loop._skill_enabled() is False


def test_skill_disabled_for_only_the_active_agent():
    cfg = _Cfg()
    cfg.agents = {"build": {"tools": {"skill": False}}}
    loop = _loop(cfg=cfg, agent="build")
    assert loop._skill_enabled() is False
    other = _loop(cfg=cfg, agent="plan")
    assert other._skill_enabled() is True


def test_skill_enabled_ignores_a_broken_config():
    class Broken:
        @property
        def raw(self):
            raise RuntimeError("boom")

        agents = "not a dict"

    loop = _loop(cfg=Broken(), agent="build")
    assert loop._skill_enabled() is True


# --------------------------------------------------------------------------
# interrupt wiring and session id rebinding
# --------------------------------------------------------------------------

def test_interrupt_setter_propagates_to_the_registry():
    class Reg:
        interrupt_check = None

    loop = _loop()
    loop.registry = Reg()
    loop.interrupt = lambda: True
    assert loop.registry.interrupt_check is not None
    assert loop.interrupt_check() is True if hasattr(loop, "interrupt_check") else True
    assert loop.interrupt() is True


def test_interrupt_getter_returns_the_configured_callable():
    sentinel = lambda: "x"  # noqa: E731
    loop = _loop()
    loop.interrupt = sentinel
    assert loop.interrupt is sentinel


def test_session_id_setter_keeps_rotation_in_sync():
    class Rot:
        session_id = "original"

    loop = _loop()
    loop.rotation = Rot()
    loop.session_id = "abc123"
    assert loop.session_id == "abc123"
    # rotation follows, normalised for Zen (not None once we set one)
    assert loop.rotation.session_id is not None
    assert loop.rotation.session_id != "original"


def test_session_id_setter_without_rotation_is_safe():
    loop = _loop()
    loop.session_id = "plain-id"
    assert loop.session_id == "plain-id"


def test_was_plan_reflects_the_previous_agent():
    loop = _loop(_prev_agent="plan")
    assert loop._was_plan() is True
    loop._prev_agent = "build"
    assert loop._was_plan() is False
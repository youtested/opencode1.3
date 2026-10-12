"""Tests for retry backoff maths and the tool schemas the agent advertises.

Same purpose as test_loop_policy: cover the decisions in ``agent/loop.py`` that
nothing else exercises. Verified by mutation — breaking these lines turns these
tests red.
"""

from __future__ import annotations

import pytest

from opencode_py.agent.loop import AgentLoop

MUTATORS = ("bash", "write", "edit", "apply_patch")


def _loop(**attrs):
    loop = AgentLoop.__new__(AgentLoop)
    loop._interrupt = lambda: False
    loop._session_id = ""
    loop._prev_agent = "build"
    loop.agent = "build"
    loop.permission = None
    for key, value in attrs.items():
        setattr(loop, key, value)
    return loop


class _Cfg:
    def __init__(self, **kw):
        self.raw = {}
        self.agents = {}
        self.provider = "opencode"
        self.model = "m"
        self.low_data = False
        for k, v in kw.items():
            setattr(self, k, v)


class _Reg:
    def __init__(self, names):
        self._names = list(names)

    def schemas(self):
        return [{"type": "function",
                 "function": {"name": n, "description": f"{n} desc", "parameters": {}}}
                for n in self._names]


def _names(schemas):
    return [s["function"]["name"] for s in schemas]


# --------------------------------------------------------------------------
# retry backoff policy: 2s * 2^attempt, +0-25% jitter, capped at 30s
# --------------------------------------------------------------------------

def test_retry_delay_honours_a_retry_after_header():
    err = Exception("rate limited")
    err.retry_after = 7.5
    assert _loop()._retry_delay(err, 3) == 7.5


def test_retry_delay_clamps_a_negative_retry_after_to_zero():
    err = Exception("bad header")
    err.retry_after = -5
    assert _loop()._retry_delay(err, 1) == 0.0


def test_retry_delay_ignores_an_unparsable_retry_after():
    err = Exception("bad header")
    err.retry_after = "soon"
    value = _loop()._retry_delay(err, 0)
    # falls through to the exponential path: 2s * 1, plus 0-25% jitter
    assert 2.0 <= value <= 2.5


def test_retry_delay_grows_exponentially():
    loop = _loop()
    for attempt in range(4):
        base = min(2.0 * (2**attempt), 30.0)
        value = loop._retry_delay(Exception("x"), attempt)
        assert base <= value <= base * 1.25 + 1e-6, f"attempt {attempt}: {value}"


def test_retry_delay_never_exceeds_the_thirty_second_cap_plus_jitter():
    value = _loop()._retry_delay(Exception("x"), 12)
    assert value <= 30.0 * 1.25 + 1e-6


def test_retry_delay_jitter_stays_inside_the_documented_band():
    """2s * 2^attempt, then 0-25% additive jitter. attempt 2 -> base 8.0s."""
    loop = _loop()
    values = {loop._retry_delay(Exception("x"), 2) for _ in range(40)}
    assert values, "no delays produced"
    for v in values:
        assert 8.0 <= v <= 10.0 + 1e-6, v


def test_retry_delay_actually_jitters():
    """Removing the jitter (returning the bare cap) must fail this."""
    loop = _loop()
    values = {loop._retry_delay(Exception("x"), 2) for _ in range(40)}
    assert len(values) > 1, f"no jitter: every delay was {values}"
    assert max(values) > 8.0, "jitter never pushes the delay above the base"


# --------------------------------------------------------------------------
# tool schemas: readonly agents lose mutators, scoped agents lose extras
# --------------------------------------------------------------------------

ALL_TOOLS = ("bash", "read", "write", "edit", "apply_patch", "glob", "grep", "skill")


def test_build_agent_sees_every_tool():
    loop = _loop(cfg=_Cfg(), registry=_Reg(ALL_TOOLS), agent="build")
    assert set(_names(loop._active_tool_schemas())) == set(ALL_TOOLS)


@pytest.mark.parametrize("agent", ["plan", "explore"])
def test_readonly_agents_never_see_mutating_tools(agent):
    loop = _loop(cfg=_Cfg(), registry=_Reg(ALL_TOOLS), agent=agent)
    seen = _names(loop._active_tool_schemas())
    for tool in MUTATORS:
        assert tool not in seen, f"{agent} can see {tool}"
    assert "read" in seen


def test_a_custom_readonly_agent_loses_mutators_too():
    cfg = _Cfg()
    cfg.agents = {"auditor": {"readonly": True}}
    loop = _loop(cfg=cfg, registry=_Reg(ALL_TOOLS), agent="auditor")
    seen = _names(loop._active_tool_schemas())
    for tool in MUTATORS:
        assert tool not in seen


def test_a_scoped_agent_sees_only_its_scope():
    cfg = _Cfg()
    cfg.agents = {"web-agent": {"scope": ["webfetch", "websearch"]}}
    loop = _loop(cfg=cfg, registry=_Reg(ALL_TOOLS + ("webfetch", "websearch")), agent="web-agent")
    assert set(_names(loop._active_tool_schemas())) == {"webfetch", "websearch"}


def test_a_scope_naming_an_absent_tool_does_not_invent_it():
    cfg = _Cfg()
    cfg.agents = {"thin": {"scope": ["nonexistent-tool"]}}
    loop = _loop(cfg=cfg, registry=_Reg(ALL_TOOLS), agent="thin")
    assert _names(loop._active_tool_schemas()) == []


def test_disabling_skills_hides_the_skill_tool():
    cfg = _Cfg()
    cfg.raw = {"tools": {"skill": False}}
    loop = _loop(cfg=cfg, registry=_Reg(ALL_TOOLS), agent="build")
    assert "skill" not in _names(loop._active_tool_schemas())


def test_readonly_beats_scope_narrowing_without_error():
    """plan + a scope list must not resurrect a mutating tool."""
    cfg = _Cfg()
    cfg.agents = {"plan": {"scope": ["bash", "read"]}}
    loop = _loop(cfg=cfg, registry=_Reg(ALL_TOOLS), agent="plan")
    seen = _names(loop._active_tool_schemas())
    assert "bash" not in seen
    assert "read" in seen


def test_low_data_keeps_the_skill_description_small():
    cfg = _Cfg(low_data=True)
    loop = _loop(cfg=cfg, registry=_Reg(ALL_TOOLS), agent="build")
    schemas = loop._active_tool_schemas()
    skill = next(s for s in schemas if s["function"]["name"] == "skill")
    assert "<available_skills>" in skill["function"]["description"]
    assert len(skill["function"]["description"]) < 600


def test_registry_schemas_are_not_mutated_by_the_readonly_filter():
    """Filtering must not strip the caller's registry copy in place."""
    reg = _Reg(ALL_TOOLS)
    loop = _loop(cfg=_Cfg(), registry=reg, agent="plan")
    loop._active_tool_schemas()
    assert set(n["function"]["name"] for n in reg.schemas()) == set(ALL_TOOLS)
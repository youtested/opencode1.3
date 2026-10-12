import time

from opencode_py.plugins import PluginDispatcher


def test_unconfigured_dispatcher_is_effectively_free():
    d = PluginDispatcher(None)
    t = time.perf_counter()
    for _ in range(20000):
        d.has("tool.before")
    per_call_us = (time.perf_counter() - t) / 20000 * 1e6
    assert per_call_us < 50, per_call_us

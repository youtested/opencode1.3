from opencode_py.commands import build_registry


def test_built_in_commands_have_no_duplicate_aliases():
    registry = build_registry()
    assert registry.get("setting") is not None
    assert registry.get("settings") is None
    assert registry.get("skill") is None
    assert registry.get("skills") is not None
    assert registry.get("clear") is None
    assert registry.get("new") is not None
    assert all(not command.aliases for command in registry.list())

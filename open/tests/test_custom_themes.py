"""Custom themes: registry, validation, config round-trip."""

import pytest

from opencode_py.config import Config
from opencode_py.tui import theme as T
from opencode_py.tui.theme_editor import CURATED_KEYS, advanced_keys


@pytest.fixture(autouse=True)
def _clean_registry():
    T.register_custom_themes({})
    T.set_active_theme("opencode")
    yield
    T.register_custom_themes({})
    T.set_active_theme("opencode")


def test_normalize_hex_accepts_variants():
    assert T.normalize_hex("#fab283") == "#fab283"
    assert T.normalize_hex("FAB283") == "#fab283"
    assert T.normalize_hex("#fab") == "#ffaabb"
    with pytest.raises(ValueError):
        T.normalize_hex("xyz")
    with pytest.raises(ValueError):
        T.normalize_hex("#12345")


def test_valid_theme_name():
    assert T.valid_theme_name("my-theme-1")
    assert not T.valid_theme_name("My Theme")
    assert not T.valid_theme_name("opencode!")
    assert not T.valid_theme_name("")


def test_sanitize_palette_keeps_known_valid_only():
    pal = T.sanitize_palette({"primary": "FF0000", "nope": "#123456", "text": "zzz"})
    assert pal == {"primary": "#ff0000"}


def test_register_rejects_builtin_shadow_and_bad_names():
    T.register_custom_themes({"dark": {"primary": "#111111"}, "Bad Name!": {"primary": "#111111"}})
    assert T.custom_names() == []


def test_set_get_rename_delete_custom():
    T.set_custom_theme("mine", {"primary": "#112233", "background": "#000000"})
    assert "mine" in T.theme_names()
    assert T.is_custom("mine")
    got = T.get_theme("mine")
    assert got.c("primary") == "#112233"
    assert got.c("text") == T.OPENSE_DARK["text"]  # merged fallback
    T.set_active_theme("mine")
    assert T.active_theme().name == "mine"
    assert T.rename_custom_theme("mine", "mine2") == "mine2"
    assert T.active_theme().name == "mine2"
    T.delete_custom_theme("mine2")
    assert not T.is_custom("mine2")
    assert T.active_theme().name == "opencode"  # deleting active falls back


def test_delete_inactive_keeps_active():
    T.set_custom_theme("a", {"primary": "#111111"})
    T.set_custom_theme("b", {"primary": "#222222"})
    T.set_active_theme("a")
    T.delete_custom_theme("b")
    assert T.active_theme().name == "a"


def test_config_round_trip():
    cfg = Config()
    cfg.custom_themes = {"mine": {"primary": "#112233"}}
    d = cfg.as_dict()
    assert d["custom_themes"] == {"mine": {"primary": "#112233"}}
    cfg2 = Config.from_dict({"custom_themes": {"mine": {"primary": "#112233"}, "dark": {"primary": "#000000"}}})
    assert cfg2.custom_themes == {"mine": {"primary": "#112233"}}
    assert T.is_custom("mine")


def test_editor_key_coverage_matches_opense_dark():
    assert set(CURATED_KEYS) | set(advanced_keys()) == set(T.OPENSE_DARK)
    assert not (set(CURATED_KEYS) & set(advanced_keys()))


def test_theme_command_lists_customs():
    from opencode_py.commands import CommandContext
    from opencode_py.commands import _theme as theme_cmd

    T.set_custom_theme("mine", {"primary": "#112233"})
    cfg = Config()
    seen: list[str] = []
    ctx = CommandContext(config=cfg, auth=None, reply=seen.append)
    theme_cmd(ctx, "")
    body = "\n".join(seen)
    assert "Custom themes:" in body
    assert "/mine" in body

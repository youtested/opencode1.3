from pathlib import Path

from opencode_py.config import Config, load_config
from opencode_py import command_manager as _cm


def test_add_then_list_sees_it(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = Config()
    cfg.raw = {}
    assert "review" not in _cm.list_commands(cfg)

    _cm.save_command("review", "Review my changes for bugs.", "Review changes", str(tmp_path), False)

    fresh = load_config(Path(tmp_path))
    cfg.raw = fresh.raw
    entry = _cm.list_commands(cfg)
    assert "review" in entry
    assert entry["review"]["prompt"] == "Review my changes for bugs."


def test_add_then_disabled_list(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _cm.save_command("deploy", "Run the deploy checklist.", "Deploy", str(tmp_path), False)
    fresh = load_config(Path(tmp_path))
    cfg = Config(); cfg.raw = fresh.raw
    assert "deploy" in _cm.list_commands(cfg)
    assert "deploy" not in _cm.list_disabled(cfg)

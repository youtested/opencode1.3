import pytest
from textual.app import App

from opencode_py import provider_manager as pm
from opencode_py.config import Config
from opencode_py.tui.provider_picker import ProviderPicker


def test_provider_rows_include_custom_and_status(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    auth = type("A", (), {"get": lambda self, key: "secret" if key == "openai" else None, "set": lambda self, key, value: None})()
    cfg = Config()
    cfg.raw = {}
    name = pm.add_custom(auth, cfg, "local-llm", "http://127.0.0.1:11434/v1", "key")
    assert name == "local-llm"
    rows = pm.rows(auth, cfg)
    assert any(row["id"] == "local-llm" for row in rows)
    assert next(row for row in rows if row["id"] == "openai")["status"] == "Connected"


class Harness(App):
    def on_mount(self) -> None:
        self.push_screen(ProviderPicker(rows=[
            {"id": "openai", "name": "OpenAI", "env": "OPENAI_API_KEY", "url": "https://platform.openai.com/api-keys", "kind": "paid", "status": "Key needed", "model": "gpt-4o"},
        ], on_result=lambda a: None))


@pytest.mark.asyncio
async def test_provider_picker_has_add_row_and_keeps_selection():
    app = Harness()
    async with app.run_test() as pilot:
        await pilot.pause()
        rows = [str(o.prompt) for o in app.screen.query_one("#provider-list").options]
        assert any("Add provider" in row for row in rows)
        app.screen.query_one("#provider-list").highlighted = 1
        app.screen.selected_id = "__provider__openai"
        app.screen.reload([{"id": "openai", "name": "OpenAI", "env": "", "url": "", "kind": "paid", "status": "Connected", "model": "gpt-4o"}])
        await pilot.pause()
        assert app.screen.query_one("#provider-list").highlighted == 1

import pytest

from opencode_py.tui.mcp_picker import McpAddDialog, McpPicker


SERVERS = {
    "notes": {"command": "python", "args": ["-m", "notes"]},
    "github": {"url": "https://mcp.example.com/mcp", "transport": "streamable-http", "allowRemote": True},
    "old": {"command": "python", "args": ["-m", "old"], "disabled": True},
}


def test_mcp_picker_rows_include_local_remote_and_state():
    picker = McpPicker(servers=SERVERS)
    ids = [o.id for o in picker._options() if o.id]
    assert "__srv__notes" in ids
    assert "__srv__github" in ids
    assert "__srv__old" in ids
    assert "__add_local__" in ids
    assert "__add_remote__" in ids
    labels = {o.id: str(o.prompt) for o in picker._options() if o.id}
    assert "local" in labels["__srv__notes"]
    assert "remote" in labels["__srv__github"]
    assert "off" in labels["__srv__old"]


@pytest.mark.asyncio
async def test_mcp_picker_renders_and_reacts_to_buttons():
    from textual.app import App, ComposeResult

    seen: list = []

    class Harness(App):
        def compose(self) -> ComposeResult:
            yield McpPicker(servers=SERVERS, on_result=lambda a: seen.append(a))

    app = Harness()
    async with app.run_test() as pilot:
        picker = app.query_one(McpPicker)
        await pilot.pause()
        assert picker.query_one("#mcp-picker-list")
        assert picker.query_one("#mcp-add-local")
        assert picker.query_one("#mcp-add-remote")
        assert picker.query_one("#mcp-test")
        assert picker.query_one("#mcp-toggle")
        picker.action_test()
        picker.action_toggle()
        await pilot.pause()
        assert ("test", "notes") in seen
        assert ("toggle", "notes") in seen


@pytest.mark.asyncio
async def test_mcp_add_dialog_local_and_remote_fields():
    from textual.app import App, ComposeResult
    from textual.widgets import Input

    class LocalHarness(App):
        def compose(self) -> ComposeResult:
            yield McpAddDialog(mode="local")

    async with LocalHarness().run_test() as pilot:
        app = pilot.app
        name = app.query_one("#mcp-add-name", Input)
        endpoint = app.query_one("#mcp-add-endpoint", Input)
        assert len(app.query("#mcp-add-token")) == 0
        name.focus()
        await pilot.press(*"notes")
        endpoint.focus()
        await pilot.press(*"python -m notes")
        await pilot.press("enter")
        await pilot.pause()
        assert not isinstance(app.screen, McpAddDialog)

    class RemoteHarness(App):
        def compose(self) -> ComposeResult:
            yield McpAddDialog(mode="remote")

    async with RemoteHarness().run_test() as pilot:
        app = pilot.app
        assert len(app.query("#mcp-add-token")) == 1

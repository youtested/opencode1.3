import pytest
from textual.app import App
from textual.widgets import Button

from opencode_py.tui.mcp_picker import McpPicker


SERVERS = {
    "notes": {"command": "python", "args": ["-m", "notes"]},
    "github": {"url": "https://mcp.example.com/mcp", "transport": "streamable-http"},
}


class Harness(App):
    def on_mount(self) -> None:
        self.push_screen(McpPicker(servers=SERVERS, on_result=lambda a: None))


@pytest.mark.asyncio
@pytest.mark.parametrize("width", [60, 72, 80, 100])
async def test_every_button_fits_at_phone_widths(width):
    app = Harness()
    async with app.run_test(size=(width, 30)) as pilot:
        await pilot.pause()
        for button_id in ("mcp-add-local", "mcp-add-remote", "mcp-test", "mcp-toggle", "mcp-close"):
            button = app.screen.query_one("#" + button_id, Button)
            region = button.region
            assert region.width > 0, button_id
            assert region.x >= 0, button_id
            assert region.x + region.width <= width, f"{button_id} clipped at width {width}: {region}"


@pytest.mark.asyncio
async def test_buttons_sit_on_one_row_and_do_not_overlap():
    app = Harness()
    async with app.run_test(size=(80, 30)) as pilot:
        await pilot.pause()
        row = [
            app.screen.query_one("#" + i, Button).region
            for i in ("mcp-add-local", "mcp-add-remote", "mcp-test", "mcp-toggle", "mcp-close")
        ]
        assert all(r.y == row[0].y for r in row)
        for left, right in zip(row, row[1:]):
            assert left.x + left.width <= right.x + 1
            assert left.width > 0

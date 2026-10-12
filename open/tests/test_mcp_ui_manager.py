from opencode_py.mcp_manager import save_remote_server, set_server_enabled


def test_remote_save_and_toggle(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = save_remote_server("github", "https://mcp.example.com/mcp", str(tmp_path), False, token="TOK")
    assert path.exists()
    text = path.read_text()
    assert "https://mcp.example.com/mcp" in text
    assert "TOK" in text

    assert set_server_enabled("github", False, str(tmp_path), False) is True
    assert '"disabled": true' in path.read_text()
    assert set_server_enabled("github", True, str(tmp_path), False) is True
    assert '"disabled"' not in path.read_text()

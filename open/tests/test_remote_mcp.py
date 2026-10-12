import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from opencode_py.config import Config
from opencode_py.tools import build_registry


TOKEN = "remote-token-123456789"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        return

    def do_POST(self):
        if self.headers.get("Authorization") != "Bearer " + TOKEN:
            self.send_response(401)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length) or b"{}")
        method = request.get("method")
        if method == "initialize":
            result = {"protocolVersion": "2024-11-05", "capabilities": {}, "instructions": "Use remote data carefully."}
        elif method == "tools/list":
            result = {"tools": [{"name": "echo", "description": "Echo text", "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}]}
        elif method == "tools/call":
            result = {"content": "echo:" + str((request.get("params") or {}).get("arguments", {}).get("text", ""))}
        elif method == "prompts/list":
            result = {"prompts": [{"name": "summary"}]}
        elif method == "resources/list":
            result = {"resources": [{"uri": "file:///tmp/doc.txt", "name": "doc"}]}
        elif method == "prompts/get":
            result = {"messages": [{"role": "user", "content": "summary prompt"}]}
        elif method == "resources/read":
            result = {"contents": [{"uri": (request.get("params") or {}).get("uri", ""), "text": "remote text"}]}
        else:
            result = {}
        body = json.dumps({"jsonrpc": "2.0", "id": request.get("id"), "result": result}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_remote_mcp_tools_prompts_resources_and_auth():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{httpd.server_address[1]}/mcp"
        cfg = Config()
        cfg.raw = {
            "mcpServers": {
                "remote": {
                    "url": url,
                    "transport": "http",
                    "allowRemote": True,
                    "auth": {"token": TOKEN},
                }
            }
        }
        registry = build_registry(cfg)
        try:
            names = registry.names()
            assert "mcp__remote__echo" in names
            assert "mcp__remote__prompts" in names
            assert "mcp__remote__resources" in names
            echoed = registry.get("mcp__remote__echo").run({"text": "hello"})
            assert echoed["output"] == "echo:hello"
            prompts = registry.get("mcp__remote__prompts").run({})
            assert "summary" in prompts["output"]
            resources = registry.get("mcp__remote__resources").run({})
            assert "doc.txt" in resources["output"]
        finally:
            registry.close()
    finally:
        httpd.shutdown()
        thread.join(timeout=3)
        httpd.server_close()

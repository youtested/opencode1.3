"""MCP server manager: list/add/remove/test with 32-bit Termux in mind.

Config shape (project ./opencode.json or ~/.config/opencode_py/opencode.json):
  {"mcpServers": {"name": {"command": "python", "args": ["-m", "my_server"]}}}

Only stdio servers (command + args). Both `python -m ...` (preferred on a
32-bit Termux phone: tiny, no Node needed) and `npx -y ...` (needs Node, often
missing/heavy on armv7) are accepted; setup warns when the binary is missing.
"""

from __future__ import annotations

from .util.log import swallow
import json
import shlex
import shutil
import sys
from pathlib import Path
from typing import Any


def list_servers(cfg: Any) -> dict[str, dict]:
    raw = (getattr(cfg, "raw", None) or {})
    servers = raw.get("mcpServers", {}) or {}
    if not isinstance(servers, dict):
        return {}
    return {k: (dict(v) if isinstance(v, dict) else v) for k, v in servers.items()}


def parse_add_args(args: str) -> tuple[str, str, list[str], bool]:
    """Parse `/mcp add <name> [--global] -- <command...>` (shlex).

    Returns (name, command, cmd_args, is_global). Raises ValueError on misuse.
    """
    tokens = shlex.split(args)
    if tokens and tokens[0].lower() == "add":
        tokens = tokens[1:]
    is_global = False
    if "--global" in tokens:
        is_global = True
        tokens = [t for t in tokens if t != "--global"]
    if "--" in tokens:
        i = tokens.index("--")
        head, cmd = tokens[:i], tokens[i + 1:]
    else:
        head, cmd = tokens[:1], tokens[1:]
    if not head:
        raise ValueError("Usage: /mcp add <name> -- <command...>")
    if not cmd:
        raise ValueError(
            f"Usage: /mcp add {head[0]} -- <command...>\n"
            "  e.g. /mcp add files -- python -m my_server\n"
            "  e.g. /mcp add files -- npx -y @modelcontextprotocol/server-filesystem /tmp"
        )
    return head[0], cmd[0], cmd[1:], is_global


def check_command(command: str, args: list[str]) -> list[str]:
    """Phone-friendly warnings for a server command. Empty = looks fine."""
    warnings: list[str] = []
    if not command:
        return ["empty command"]
    resolved = shutil.which(command)
    if "/" in command:
        p = Path(command)
        if not p.exists():
            warnings.append(f"'{command}' not found on this device.")
        elif not p.is_file():
            warnings.append(f"'{command}' is not a file.")
    elif resolved is None:
        if command in ("npx", "node", "npm", "bun", "deno"):
            warnings.append(
                f"'{command}' not found. On Termux: pkg install nodejs "
                "(heavy, ~100MB+, may be missing on 32-bit) — "
                "prefer a Python server: python -m my_server."
            )
        elif command == "python":
            warnings.append(f"'{command}' not found. Try '{sys.executable}'.")
        else:
            warnings.append(f"'{command}' not found in PATH.")
    if command in ("python", "python3", sys.executable) or (
        resolved and Path(resolved).name.startswith("python")
    ):
        if args[:1] == ["-m"] and len(args) >= 2:
            mod = args[1]
            try:
                import importlib.util as _ilu

                if _ilu.find_spec(mod) is None:
                    warnings.append(f"python module '{mod}' not importable here.")
            except Exception as _oc_exc:
                swallow(_oc_exc, "mcp_manager.py:90")
                pass
        elif args[:1] == ["-m"]:
            warnings.append("'python -m' needs a module name after it.")
    return warnings


def test_server(
    name: str,
    command: str | None = None,
    args: list[str] | None = None,
    timeout: float = 20.0,
    *,
    url: str | None = None,
    transport: str = "streamable-http",
    headers: dict | None = None,
    auth: dict | None = None,
) -> str:
    """Start a local or remote server, run tools/list, and close it."""
    server = None
    try:
        if url:
            from .remote_mcp import RemoteMCPServer
            server = RemoteMCPServer(
                name=name, url=url, transport=transport, headers=headers or {}, auth=auth or {}, timeout=timeout
            )
        else:
            from .tools.mcp import MCPServer
            server = MCPServer(name=name, command=str(command or ""), args=list(args or []), timeout=timeout)
        tools = server.list_tools()
    except Exception as exc:
        return f"{name}: FAILED — {exc}"
    finally:
        if server is not None:
            try:
                server.close()
            except Exception as _oc_exc:
                swallow(_oc_exc, "mcp_manager.py:126")
                pass
    names = [t.get("name", "?") for t in tools if isinstance(t, dict)][:5]
    extra = f" (+{len(tools) - 5} more)" if len(tools) > 5 else ""
    shown = ", ".join(names) if names else "no tools"
    return f"{name}: OK — {len(tools)} tool(s): {shown}{extra}"


def config_path(worktree: str = "", is_global: bool = False) -> Path:
    from .globals import Path as GPath

    if is_global:
        return GPath.config / "opencode.json"
    base = Path(worktree) if worktree else Path.cwd()
    return base / "opencode.json"


def save_server(name: str, command: str, args: list[str], worktree: str = "", is_global: bool = False) -> Path:
    """Write one server into opencode.json (preserving other keys)."""
    from .config import _load_json, _strip_jsonc

    path = config_path(worktree, is_global)
    data: dict[str, Any] = {}
    if path.exists():
        try:
            data = _load_json(path)
        except Exception:
            try:
                data = json.loads(_strip_jsonc(path.read_text(encoding="utf-8")))
            except Exception:
                data = {}
    servers = data.get("mcpServers")
    if not isinstance(servers, dict):
        servers = {}
        data["mcpServers"] = servers
    servers[name] = {"command": command, "args": list(args)}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def save_remote_server(
    name: str,
    url: str,
    worktree: str = "",
    is_global: bool = False,
    *,
    transport: str = "streamable-http",
    token: str = "",
    allow_remote: bool = True,
) -> Path:
    from .config import _load_json, _strip_jsonc

    path = config_path(worktree, is_global)
    data: dict[str, Any] = {}
    if path.exists():
        try:
            data = _load_json(path)
        except Exception:
            try:
                data = json.loads(_strip_jsonc(path.read_text(encoding="utf-8")))
            except Exception:
                data = {}
    servers = data.get("mcpServers")
    if not isinstance(servers, dict):
        servers = {}
        data["mcpServers"] = servers
    spec: dict[str, Any] = {"url": url, "transport": transport, "allowRemote": True}
    if token:
        spec["auth"] = {"tokenEnv": token} if token.isupper() else {"token": token}
    servers[name] = spec
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def set_server_enabled(name: str, enabled: bool, worktree: str = "", is_global: bool = False) -> bool:
    from .config import _load_json, _strip_jsonc

    for path in ([config_path(worktree, is_global)] if not is_global else [config_path(worktree, True)]):
        if not path.exists():
            continue
        try:
            data = _load_json(path)
        except Exception:
            try:
                data = json.loads(_strip_jsonc(path.read_text(encoding="utf-8")))
            except Exception as _oc_exc:
                swallow(_oc_exc, "mcp_manager.py:213")
                continue
        servers = data.get("mcpServers")
        if not isinstance(servers, dict) or name not in servers:
            continue
        spec = servers[name]
        if not isinstance(spec, dict):
            spec = {}
            servers[name] = spec
        if enabled:
            spec.pop("disabled", None)
        else:
            spec["disabled"] = True
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return True
    return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def remove_server(name: str, worktree: str = "", is_global: bool = False) -> Path | None:
    """Delete one server from opencode.json. Returns path, or None if absent."""
    from .config import _load_json, _strip_jsonc

    path = config_path(worktree, is_global)
    if not path.exists():
        return None
    try:
        data = _load_json(path)
    except Exception:
        try:
            data = json.loads(_strip_jsonc(path.read_text(encoding="utf-8")))
        except Exception:
            return None
    servers = data.get("mcpServers")
    if not isinstance(servers, dict) or name not in servers:
        return None
    del servers[name]
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def remove_server_everywhere(name: str, worktree: str = "") -> list[Path]:
    """Delete a server from BOTH project and global files.

    The merged config can't tell which scope provided the entry, so a
    single-scope delete resurrects on restart. Returns removed paths.
    """
    removed: list[Path] = []
    for is_global in (False, True):
        try:
            p = remove_server(name, worktree, is_global)
        except Exception:
            p = None
        if p is not None:
            removed.append(p)
    try:
        from .tools import _mcp_cache_evict_closed
    except Exception as _oc_exc:
        swallow(_oc_exc, "mcp_manager.py:272")
        pass
    else:
        try:
            _mcp_cache_evict_closed()
        except Exception as _oc_exc:
            swallow(_oc_exc, "mcp_manager.py:277")
            pass
    return removed


def refresh_engine_mcp(engine: Any, cfg: Any) -> str:
    """Reload MCP tools into a live engine registry (next turn uses them).

    Drops stale mcp__*/mcp:* tools, releases old server refs, re-runs the
    shared _load_mcp_servers() against the updated cfg.
    """
    if engine is None or getattr(engine, "registry", None) is None:
        return "saved (applies on next session start)."
    registry = engine.registry
    lock = getattr(registry, "_lock", None)
    if lock is not None:
        lock.acquire()
    try:
        for tool_name in list(registry.names()):
            if tool_name.startswith("mcp__") or tool_name.startswith("mcp:"):
                try:
                    registry._tools.pop(tool_name, None)
                except Exception as _oc_exc:
                    swallow(_oc_exc, "mcp_manager.py:299")
                    pass
        try:
            registry.close()
        except Exception as _oc_exc:
            swallow(_oc_exc, "mcp_manager.py:303")
            pass
        from .tools import _load_mcp_servers, _mcp_cache_evict_closed

        _load_mcp_servers(registry, cfg)
        try:
            _mcp_cache_evict_closed()
        except Exception as _oc_exc:
            swallow(_oc_exc, "mcp_manager.py:310")
            pass
    except Exception as e:
        return f"saved (live reload failed: {e} — applies on next session start)."
    finally:
        try:
            if lock is not None:
                lock.release()
        except Exception as _oc_exc:
            swallow(_oc_exc, "mcp_manager.py:318")
            pass
    names = [n for n in registry.names() if n.startswith("mcp__")]
    return f"live now ({len(names)} MCP tool(s) loaded)."

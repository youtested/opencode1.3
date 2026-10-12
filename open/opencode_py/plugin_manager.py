"""Plugin manager: list / add / remove / enable for plugins and plugin hooks.

Config shape (project ./opencode.json):

    {
      "plugins": ["my_tools.plugin", "./tools/demo.py"],
      "pluginHooks": [
        {"name": "fmt", "target": "./hooks/format.py", "events": ["file.changed"]}
      ]
    }

`plugins` are tool providers (a module exposing TOOLS).
`pluginHooks` are event hooks (python / js / ts / shell command).
Disabled entries keep their settings but are skipped at load time.
"""
from __future__ import annotations

from .util.log import swallow
import json
from pathlib import Path
from typing import Any

PLUGIN_KEY = "plugins"
HOOK_KEY = "pluginHooks"
HOOK_EVENTS = (
    "tool.before",
    "tool.after",
    "file.changed",
    "file.created",
    "file.deleted",
    "session.start",
    "session.end",
)


def config_path(worktree: str = "", is_global: bool = False) -> Path:
    from .globals import Path as GPath

    if is_global:
        return GPath.config / "opencode.json"
    base = Path(worktree) if worktree else Path.cwd()
    return base / "opencode.json"


def _read(path: Path) -> dict[str, Any]:
    from .config import _load_json, _strip_jsonc

    if not path.exists():
        return {}
    try:
        data = _load_json(path)
        return data if isinstance(data, dict) else {}
    except Exception:
        try:
            value = json.loads(_strip_jsonc(path.read_text(encoding="utf-8")))
            return value if isinstance(value, dict) else {}
        except Exception:
            return {}


def _write(path: Path, data: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def list_plugins(cfg: Any) -> list[str]:
    raw = (getattr(cfg, "raw", None) or {}) if cfg is not None else {}
    values = raw.get(PLUGIN_KEY) or []
    return [str(v) for v in values] if isinstance(values, list) else []


def list_hooks(cfg: Any) -> list[dict[str, Any]]:
    raw = (getattr(cfg, "raw", None) or {}) if cfg is not None else {}
    values = raw.get(HOOK_KEY) or []
    if isinstance(values, dict):
        out = []
        for name, value in values.items():
            if isinstance(value, dict):
                out.append({"name": str(name), **value})
        return out
    return [dict(v) for v in values if isinstance(v, dict)] if isinstance(values, list) else []


def list_disabled(cfg: Any) -> set[str]:
    disabled: set[str] = set()
    raw = (getattr(cfg, "raw", None) or {}) if cfg is not None else {}
    for value in raw.get("pluginDisabled") or []:
        disabled.add(str(value))
    for hook in list_hooks(cfg):
        if hook.get("disabled") is True and hook.get("name"):
            disabled.add(f"hook:{hook['name']}")
    return disabled


def save_plugin(name: str, target: str, worktree: str = "", is_global: bool = False) -> Path:
    """Add/register a tool plugin (a python module path or file)."""
    path = config_path(worktree, is_global)
    data = _read(path)
    plugins = data.get(PLUGIN_KEY)
    if not isinstance(plugins, list):
        plugins = []
    target = str(target).strip()
    if target not in plugins:
        plugins.append(target)
    data[PLUGIN_KEY] = plugins
    data.setdefault("pluginNames", {})[str(name)] = target
    disabled = data.get("pluginDisabled")
    if isinstance(disabled, list) and name in disabled:
        disabled.remove(name)
    return _write(path, data)


def save_hook(
    name: str,
    target: str,
    events: list[str],
    worktree: str = "",
    is_global: bool = False,
    *,
    timeout: float = 3.0,
) -> Path:
    path = config_path(worktree, is_global)
    data = _read(path)
    hooks = data.get(HOOK_KEY)
    if isinstance(hooks, dict):
        hooks = [{"name": k, **v} for k, v in hooks.items() if isinstance(v, dict)]
    if not isinstance(hooks, list):
        hooks = []
    clean_events = [e for e in events if e in HOOK_EVENTS] or ["tool.after"]
    entry = {
        "name": str(name).strip(),
        "target": str(target).strip(),
        "events": clean_events,
        "timeout": max(0.2, min(float(timeout or 3.0), 10.0)),
    }
    hooks = [h for h in hooks if h.get("name") != entry["name"]]
    hooks.append(entry)
    data[HOOK_KEY] = hooks
    return _write(path, data)


def remove(name: str, worktree: str = "", is_global: bool = False) -> bool:
    path = config_path(worktree, is_global)
    data = _read(path)
    changed = False
    plugins = data.get(PLUGIN_KEY)
    names = data.get("pluginNames")
    if isinstance(names, dict) and names.get(name) in (plugins or []):
        target = str(names.get(name))
        data[PLUGIN_KEY] = [p for p in plugins if str(p) != target]
        names.pop(name, None)
        if not names:
            data.pop("pluginNames", None)
        changed = True
    elif isinstance(plugins, list):
        keep_plugins = [p for p in plugins if str(p) != name]
        if len(keep_plugins) != len(plugins):
            data[PLUGIN_KEY] = keep_plugins
            changed = True
    hooks = data.get(HOOK_KEY)
    if isinstance(hooks, list):
        keep = [h for h in hooks if h.get("name") != name]
        if len(keep) != len(hooks):
            data[HOOK_KEY] = keep
            changed = True
    if not changed and not path.exists():
        return False
    if changed:
        _write(path, data)
    return changed


def set_enabled(name: str, enabled: bool, worktree: str = "", is_global: bool = False) -> bool:
    path = config_path(worktree, is_global)
    data = _read(path)
    if not path.exists():
        return False
    found = False
    plugins = data.get(PLUGIN_KEY)
    if isinstance(plugins, list) and name in plugins:
        found = True
        disabled = data.setdefault("pluginDisabled", [])
        if enabled and name in disabled:
            disabled.remove(name)
        elif not enabled and name not in disabled:
            disabled.append(name)
    hooks = data.get(HOOK_KEY)
    if isinstance(hooks, list):
        for hook in hooks:
            if isinstance(hook, dict) and hook.get("name") == name:
                found = True
                if enabled:
                    hook.pop("disabled", None)
                else:
                    hook["disabled"] = True
    if found:
        _write(path, data)
    return found


def test_plugin(name: str, cfg: Any, worktree: str = "") -> str:
    """Load a plugin and report its tools (or the error)."""
    from .tools import build_registry

    registry = build_registry(cfg)
    try:
        names = [n for n in registry.names() if "plugin" in n or n.startswith(name.split(".")[-1])]
        mine = [n for n in registry.names() if not n.startswith("mcp__") and n in names]
        if mine:
            return f"{name}: OK — {len(mine)} tool(s): {', '.join(mine[:5])}"
        return f"{name}: loaded (no TOOLS exported)"
    except Exception as exc:
        return f"{name}: FAILED — {exc}"
    finally:
        try:
            registry.close()
        except Exception as _oc_exc:
            swallow(_oc_exc, "plugin_manager.py:217")
            pass



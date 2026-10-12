"""Custom slash command manager: list / add / edit / remove / enable.

Config shape (project ./opencode.json):

    {
      "commands": {
        "review": {
          "description": "Review my current changes",
          "prompt": "Review my current changes for bugs and security problems.",
          "agent": "build"
        },
        "deploy": {
          "description": "Deploy checklist",
          "prompt": "Run the deploy checklist for this project."
        }
      },
      "commandDisabled": ["old"]
    }

A user runs /review and the stored prompt is sent as if typed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

COMMAND_KEY = "commands"
DISABLED_KEY = "commandDisabled"


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


def list_commands(cfg: Any) -> dict[str, dict[str, Any]]:
    raw = (getattr(cfg, "raw", None) or {}) if cfg is not None else {}
    values = raw.get(COMMAND_KEY) or {}
    out: dict[str, dict[str, Any]] = {}
    if isinstance(values, dict):
        for name, value in values.items():
            if isinstance(value, dict):
                entry = dict(value)
                entry.setdefault("name", str(name))
                entry.setdefault("description", "")
                entry.setdefault("prompt", "")
                out[str(name).lstrip("/")] = entry
            elif isinstance(value, str) and value.strip():
                out[str(name).lstrip("/")] = {
                    "name": str(name).lstrip("/"),
                    "description": "",
                    "prompt": value.strip(),
                }
    return out


def list_disabled(cfg: Any) -> set[str]:
    raw = (getattr(cfg, "raw", None) or {}) if cfg is not None else {}
    values = raw.get(DISABLED_KEY) or []
    return {str(v).lstrip("/") for v in values} if isinstance(values, list) else set()


def normalize_name(name: str) -> str:
    return str(name or "").strip().lstrip("/").replace(" ", "-").lower()


def save_command(
    name: str,
    prompt: str,
    description: str = "",
    worktree: str = "",
    is_global: bool = False,
    *,
    agent: str = "",
) -> str:
    key = normalize_name(name)
    if not key:
        raise ValueError("Command name can't be empty")
    text = str(prompt or "").strip()
    if not text:
        raise ValueError("Command prompt can't be empty")
    path = config_path(worktree, is_global)
    data = _read(path)
    commands = data.get(COMMAND_KEY)
    if not isinstance(commands, dict):
        commands = {}
        data[COMMAND_KEY] = commands
    entry: dict[str, Any] = {"description": str(description or "").strip(), "prompt": text}
    if agent:
        entry["agent"] = str(agent)
    commands[key] = entry
    disabled = data.get(DISABLED_KEY)
    if isinstance(disabled, list) and key in disabled:
        disabled.remove(key)
    _write(path, data)
    return key


def remove_command(name: str, worktree: str = "", is_global: bool = False) -> bool:
    key = normalize_name(name)
    path = config_path(worktree, is_global)
    data = _read(path)
    commands = data.get(COMMAND_KEY)
    if not isinstance(commands, dict) or key not in commands:
        return False
    commands.pop(key, None)
    data[COMMAND_KEY] = commands
    disabled = data.get(DISABLED_KEY)
    if isinstance(disabled, list) and key in disabled:
        disabled.remove(key)
    _write(path, data)
    return True


def set_enabled(name: str, enabled: bool, worktree: str = "", is_global: bool = False) -> bool:
    key = normalize_name(name)
    path = config_path(worktree, is_global)
    data = _read(path)
    commands = data.get(COMMAND_KEY)
    if not isinstance(commands, dict) or key not in commands:
        return False
    disabled = data.get(DISABLED_KEY)
    if not isinstance(disabled, list):
        disabled = []
        data[DISABLED_KEY] = disabled
    if enabled and key in disabled:
        disabled.remove(key)
    elif not enabled and key not in disabled:
        disabled.append(key)
    _write(path, data)
    return True


def describe(name: str, entry: dict[str, Any]) -> str:
    desc = str(entry.get("description") or "").strip()
    if not desc:
        desc = " ".join(str(entry.get("prompt") or "").split()[:12])
    return desc or "(no description)"

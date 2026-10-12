"""Lazy, light plugin hook dispatcher.

Design goals (phone-first):
- Zero measurable cost when no plugin hooks are configured.
- Nothing is imported or started until an event it listens for actually fires.
- A single dict lookup (`has`) guards every hook call site.
- Every hook is isolated: errors never break the agent.
- Hard per-hook timeout + a per-emit total budget keep a bad hook cheap.
- Python hooks import lazily; JS/TS hooks start `node` only when needed.

Hook module contract (Python)::

    def hook(event, payload): ...            # any plugin event
    # or
    HOOKS = {"file.changed": fn, ...}       # selective

Hook contract (JS/TS) - read one JSON event on stdin, write JSON on stdout::

    let raw = ""; process.stdin.on("data", d => raw += d);
    process.stdin.on("end", () => {
      const p = JSON.parse(raw || "{}");
      process.stdout.write(JSON.stringify({ ok: true, note: p.filePath }));
    });

Config in opencode.json::

    "pluginHooks": [
      {"name": "fmt", "target": "./hooks/format.py", "events": ["file.changed"]}
    ]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MAX_HOOK_SECONDS = 10.0
DEFAULT_HOOK_SECONDS = 3.0
EMIT_BUDGET_SECONDS = 8.0

KNOWN_EVENTS = (
    "tool.before",
    "tool.after",
    "file.changed",
    "file.created",
    "file.deleted",
    "session.start",
    "session.end",
)

_NODE_CANDIDATES = ("node", "nodejs")
_NPX_CANDIDATES = ("npx",)


def _is_npm_target(target: str) -> bool:
    t = str(target or "").strip()
    if t.startswith("npm:") or t.startswith("npx:"):
        return True
    if t.startswith("@") and "/" in t and " " not in t:
        return True
    if " " not in t and t.count("@") == 1 and not t.startswith(".") and "/" not in t and t.endswith((".js", ".mjs")):
        return False
    if " " not in t and "/" not in t and t.endswith((".npm",)):
        return True
    return False


def _npx_bin() -> str | None:
    import shutil

    for name in _NPX_CANDIDATES:
        found = shutil.which(name)
        if found:
            return found
    return None


def _node_bin() -> str | None:
    import shutil

    for name in _NODE_CANDIDATES:
        found = shutil.which(name)
        if found:
            return found
    return None


@dataclass
class HookSpec:
    name: str
    target: str
    events: tuple[str, ...]
    timeout: float = DEFAULT_HOOK_SECONDS
    cwd: str = ""
    env: dict[str, str] = field(default_factory=dict)
    _kind: str = "python"
    _loaded: Any = field(default=None, repr=False)
    _load_failed: bool = field(default=False, repr=False)
    _lock: Any = field(default_factory=threading.Lock, repr=False)


class PluginDispatcher:
    """Registry of hook specs; lazily loads and runs only matching hooks."""

    def __init__(self, raw: Any = None) -> None:
        self._by_event: dict[str, list[HookSpec]] = {}
        self._all: list[HookSpec] = []
        self._errors: list[str] = []
        self._node: str | None = None
        self._node_checked = False
        self._npx: str | None = None
        self._npx_checked = False
        self._closed = False
        self._lock = threading.RLock()
        for spec in self._parse(raw or []):
            self._all.append(spec)
            for event in spec.events:
                self._by_event.setdefault(event, []).append(spec)

    @staticmethod
    def _parse(raw: Any) -> list[HookSpec]:
        items: list[Any] = []
        if isinstance(raw, dict):
            for name, value in raw.items():
                if isinstance(value, dict):
                    items.append({"name": name, **value})
        elif isinstance(raw, list):
            items = [x for x in raw if isinstance(x, dict)]
        specs: list[HookSpec] = []
        for item in items:
            if item.get("disabled") is True:
                continue
            target = str(item.get("target") or item.get("run") or item.get("command") or "").strip()
            if not target:
                continue
            events = item.get("events") or [e for e in KNOWN_EVENTS]
            if isinstance(events, str):
                events = [events]
            events = tuple(str(e) for e in events if str(e) in KNOWN_EVENTS)
            if not events:
                continue
            try:
                timeout = float(item.get("timeout", DEFAULT_HOOK_SECONDS))
            except (TypeError, ValueError):
                timeout = DEFAULT_HOOK_SECONDS
            timeout = max(0.2, min(timeout, MAX_HOOK_SECONDS))
            specs.append(
                HookSpec(
                    name=str(item.get("name") or target),
                    target=target,
                    events=events,
                    timeout=timeout,
                    cwd=str(item.get("cwd") or ""),
                    env={str(k): str(v) for k, v in (item.get("env") or {}).items()},
                )
            )
        return specs

    def has(self, event: str) -> bool:
        if self._closed:
            return False
        return bool(self._by_event.get(event))

    @property
    def count(self) -> int:
        return len(self._all)

    def names(self) -> list[str]:
        return [s.name for s in self._all]

    def close(self) -> None:
        self._closed = True

    def emit(self, event: str, payload: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        specs = self._by_event.get(event)
        if not specs or self._closed:
            return []
        data = {"event": event, **(payload or {})}
        results: list[dict[str, Any]] = []
        deadline_budget = EMIT_BUDGET_SECONDS
        for spec in specs:
            if deadline_budget <= 0:
                results.append({"error": True, "output": f"plugin {spec.name}: hook budget exhausted"})
                break
            started = _now()
            try:
                out = self._run_one(spec, data)
            except Exception as exc:
                out = {"error": True, "output": f"plugin {spec.name}: {exc}"}
                self._errors.append(f"{spec.name}: {exc}")
            if isinstance(out, dict):
                results.append(out)
            elif out is not None:
                results.append({"output": str(out)})
            deadline_budget -= max(0.0, _now() - started)
        return results

    def _run_one(self, spec: HookSpec, data: dict[str, Any]) -> Any:
        handler = self._load(spec)
        if handler is None:
            return None
        kind = getattr(spec, "_kind", "python")
        if kind == "python":
            if isinstance(handler, _MultiEventHook):
                return handler(data)
            try:
                return handler(data["event"], data)
            except TypeError:
                return handler(data)
        if kind == "node":
            return self.run_node(spec, data)
        if kind == "npm":
            return self.run_npm(spec, data)
        if kind == "command":
            return self._run_command(spec, spec.target, data)
        return None

    def _load(self, spec: HookSpec) -> Any:
        if spec._load_failed:
            return None
        if spec._loaded is not None:
            return spec._loaded
        with spec._lock:
            if spec._loaded is not None:
                return spec._loaded
            target = spec.target
            path = Path(target).expanduser()
            suffix = path.suffix.lower()
            if suffix in (".py",):
                spec._kind = "python"
                try:
                    mod = self._import_file(spec, path)
                except Exception:
                    # the negative cache only covered the None return, so a hook
                    # that RAISED was re-imported and re-reported every event
                    spec._load_failed = True
                    raise
                if mod is None:
                    spec._load_failed = True
                    return None
                hooks = getattr(mod, "HOOKS", None)
                if isinstance(hooks, dict) and hooks:
                    spec._loaded = _MultiEventHook(hooks, spec.name)
                else:
                    fn = getattr(mod, "hook", None) or getattr(mod, "on_event", None)
                    if fn is None or not callable(fn):
                        spec._load_failed = True
                        return None
                    spec._loaded = fn
            elif suffix in (".js", ".mjs", ".cjs", ".ts", ".mts") or _is_npm_target(target):
                spec._kind = "npm" if _is_npm_target(target) else "node"
                if spec._kind == "node":
                    node = self._ensure_node()
                    if node is None:
                        spec._load_failed = True
                        return None
                    spec._loaded = {"__node__": True, "node": node, "path": str(path)}
                else:
                    npx = _npx_bin()
                    if npx is None:
                        spec._load_failed = True
                        return None
                    spec._loaded = {"__npm__": True, "npx": npx, "package": target}
            else:
                spec._kind = "command"
                spec._loaded = {"__command__": True, "command": target}
        return spec._loaded

    def _ensure_node(self) -> str | None:
        if not self._node_checked:
            self._node = _node_bin()
            self._node_checked = True
        return self._node

    @staticmethod
    def _import_file(spec: HookSpec, path: Path) -> Any:
        import importlib.util

        if not path.exists():
            raise FileNotFoundError(str(path))
        cwd = str(path.parent)
        added = cwd not in sys.path
        if added:
            sys.path.insert(0, cwd)
        name = f"_opencode_hook_{abs(hash(str(path)))}"
        module_spec = importlib.util.spec_from_file_location(name, path)
        if module_spec is None or module_spec.loader is None:
            raise ImportError(f"cannot load {path}")
        module = importlib.util.module_from_spec(module_spec)
        try:
            sys.modules[name] = module
            module_spec.loader.exec_module(module)
        except Exception:
            # never leave a half-initialized module cached for the process
            sys.modules.pop(name, None)
            raise
        finally:
            # the hook dir used to shadow imports process-wide, forever
            if added and cwd in sys.path:
                sys.path.remove(cwd)
        return module

    def _run_command(self, spec: HookSpec, command: str, data: dict[str, Any]) -> Any:
        env = os.environ.copy()
        env.update(spec.env)
        env["OPENCODE_HOOK_EVENT"] = data.get("event", "")
        env["OPENCODE_HOOK_JSON"] = json.dumps(data, ensure_ascii=False)
        try:
            proc = subprocess.run(
                command,
                shell=True,
                input=json.dumps(data, ensure_ascii=False),
                capture_output=True,
                text=True,
                timeout=spec.timeout,
                cwd=spec.cwd or None,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return {"error": True, "output": f"plugin {spec.name}: timed out after {spec.timeout}s"}
        if proc.returncode != 0:
            return {"error": True, "output": f"plugin {spec.name}: exit {proc.returncode} {proc.stderr.strip()[:200]}"}
        out = (proc.stdout or "").strip()
        if not out:
            return None
        try:
            return json.loads(out)
        except json.JSONDecodeError:
            return {"output": out}

    def run_npm(self, spec: HookSpec, data: dict[str, Any]) -> Any:
        npx = self._ensure_npx()
        package = str(spec._loaded.get("package") or spec.target)
        if package.startswith("npm:") or package.startswith("npx:"):
            package = package.split(":", 1)[1]
        return self._run_command(spec, _shell_quote(npx) + " -y " + _shell_quote(package), data)

    def _ensure_npx(self) -> str:
        if not self._npx_checked:
            self._npx = _npx_bin()
            self._npx_checked = True
        return self._npx or "npx"

    def run_node(self, spec: HookSpec, data: dict[str, Any]) -> Any:
        node = self._ensure_node()
        path = str(spec._loaded.get("path") or spec.target)
        script = path if path.endswith((".js", ".mjs", ".cjs")) else _compile_ts(node, path, spec)
        if not script:
            return {"error": True, "output": f"plugin {spec.name}: cannot run {path}"}
        return self._run_command(spec, _shell_quote(node) + " " + _shell_quote(script), data)



class _MultiEventHook:
    def __init__(self, hooks: dict[str, Any], name: str) -> None:
        self._hooks = {str(k): v for k, v in hooks.items() if callable(v)}
        self._name = name

    def __call__(self, data: dict[str, Any]) -> Any:
        fn = self._hooks.get(str(data.get("event")))
        if fn is None:
            return None
        try:
            return fn(data.get("event"), data)
        except TypeError:
            return fn(data)


def _shell_quote(value: str) -> str:
    return "'" + str(value).replace("'", "'\\''") + "'"


def _compile_ts(node: str, path: str, spec: HookSpec) -> str | None:
    out = path + ".opencode.mjs"
    try:
        proc = subprocess.run(
            [node, "--experimental-strip-types", path],
            capture_output=True,
            text=True,
            timeout=spec.timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode == 0 and proc.stdout:
        try:
            Path(out).write_text(proc.stdout, encoding="utf-8")
            return out
        except OSError:
            return None
    return None


def _now() -> float:
    import time

    return time.monotonic()


_GLOBAL: PluginDispatcher | None = None


def get_dispatcher(cfg: Any = None) -> PluginDispatcher:
    global _GLOBAL
    if _GLOBAL is not None:
        return _GLOBAL
    raw = None
    try:
        raw = (getattr(cfg, "raw", None) or {}).get("pluginHooks") if cfg is not None else None
    except Exception:
        raw = None
    _GLOBAL = PluginDispatcher(raw)
    return _GLOBAL


def reset_dispatcher() -> None:
    global _GLOBAL
    if _GLOBAL is not None:
        _GLOBAL.close()
    _GLOBAL = None

"""Slash command registry + handlers.

Built-ins mirror opencode's TUI command set. Each handler receives a context
with the engine, session, config, auth, and a reply callback.
"""

from __future__ import annotations

from .util.log import swallow
import json
from dataclasses import dataclass, field
from typing import Any, Callable

from .config import Config, save_config

# where to get a key per provider (shown by /models)
KEY_HINTS: dict[str, str] = {
    "groq": "https://console.groq.com/keys",
    "cerebras": "https://cloud.cerebras.ai/",
    "google": "https://aistudio.google.com/apikey",
    "openrouter": "https://openrouter.ai/keys",
    "nvidia": "https://build.nvidia.com/",
    "mistral": "https://console.mistral.ai/",
    "github": "https://github.com/settings/tokens",
    "sambanova": "https://cloud.sambanova.ai/",
    "togetherai": "https://api.together.ai/",
    "anthropic": "https://console.anthropic.com/",
    "openai": "https://platform.openai.com/api-keys",
    "ollama": "local",
}


@dataclass
class CommandContext:
    config: Config
    auth: Any
    session: Any = None
    engine: Any = None
    worktree: str = ""
    reply: Callable[[str], None] = field(default=lambda s: print(s))
    get_session: Callable[[], Any] | None = None
    set_agent: Callable[[str], None] | None = None
    set_model: Callable[[str], None] | None = None
    exit_app: Callable[[], None] | None = None
    resume: Callable[[str], None] | None = None
    connect: Callable[[str], None] | None = None
    registry: Any = None
    # True while the centered command popup renders its PREVIEW: handlers must
    # show what they WOULD do without doing it (/export used to write the file
    # here, then again on Run — a double write from merely browsing).
    preview_only: bool = False


@dataclass
class Command:
    name: str
    aliases: list[str]
    description: str
    handler: Callable[[CommandContext, str], None]
    hidden: bool = False
    preview: bool = True  # safe to run for the centered popup preview
    metadata: dict[str, Any] = field(default_factory=dict)


class CommandRegistry:
    def __init__(self) -> None:
        self._commands: dict[str, Command] = {}

    def register(self, command: Command) -> None:
        self._commands[command.name] = command
        canonical = {c.name for c in self._commands.values()}
        for alias in command.aliases:
            # An alias must never steal another command's canonical name:
            # registration order used to silently decide whether "sessions"
            # meant /resume or /sessions.
            owner = self._commands.get(alias)
            if alias in canonical and owner is not command:
                continue
            self._commands[alias] = command

    def get(self, name: str) -> Command | None:
        return self._commands.get(name)

    def list(self) -> list[Command]:
        seen: set[str] = set()
        out = []
        for c in self._commands.values():
            if c.name not in seen:
                seen.add(c.name)
                out.append(c)
        return out

    def names(self) -> list[str]:
        return [c.name for c in self.list()]


def _print_help(ctx: CommandContext, args: str) -> None:
    ctx.reply(
        "Commands:\n"
        + "\n".join(f"  /{c.name:<12} {c.description}" for c in ctx.registry.list() if not c.hidden)
    )


def _settings(ctx: CommandContext, args: str) -> None:
    ctx.reply("Opening Settings.")


def _new(ctx: CommandContext, args: str) -> None:
    if ctx.get_session:
        s = ctx.get_session()
        if s:
            ctx.reply("Starting a new session (clear current view).")
    else:
        ctx.reply("New session.")


def _clear(ctx: CommandContext, args: str) -> None:
    if ctx.engine:
        ctx.engine.set_history([])
    ctx.reply("Conversation cleared.")


def _models(ctx: CommandContext, args: str) -> None:
    from .providers import FREE_DEFAULT_MODELS, FREE_PROVIDERS, fetch_zen_models

    models = fetch_zen_models()
    free = [m for m in models if m.get("free")]
    paid = [m for m in models if not m.get("free")]
    lines = ["Free models (OpenCode Zen) - no key needed:"]
    for m in free:
        ctx_size = f"{m['context']:,}" if m.get("context") else "?"
        lines.append(f"  opencode/{m['id']:<26} {m.get('name') or m['id']}  ctx={ctx_size}")
    lines.append("")
    lines.append("Paid models (need OPENCODE_API_KEY - https://opencode.ai/auth):")
    for m in paid[:20]:
        lines.append(f"  opencode/{m['id']:<26} {m.get('name') or m['id']}")
    if len(paid) > 20:
        lines.append(f"  ... and {len(paid) - 20} more")
    lines.append("")
    lines.append("Free-tier providers (bring your own key):")
    for pid in FREE_PROVIDERS:
        model = FREE_DEFAULT_MODELS.get(pid, "?")
        url = KEY_HINTS.get(pid, "")
        lines.append(f"  {pid:<12} {model:<26} {url}")
    lines.append("")
    lines.append("Local: ollama  (localhost:11434, no key)")
    ctx.reply("\n".join(lines))


def _connect(ctx: CommandContext, args: str) -> None:
    arg = args.strip()
    if ctx.connect:
        # TUI: hand off to the connect screen (optionally preselecting a provider)
        ctx.connect(arg.split()[0] if arg else "")
        return
    if not arg:
        ctx.reply(
            "Usage: /connect <provider>\n\nProviders: opencode (zen), groq, cerebras, "
            "google, openrouter, nvidia, mistral, github, sambanova, togetherai, anthropic, openai, ollama\n\n"
            "In the TUI, /connect opens the key-entry screen."
        )
        return
    provider = arg.split()[0]
    ctx.reply(f"Connecting {provider}... Use the TUI to paste your API key, or set the env var.")


def _permissions(ctx: CommandContext, args: str) -> None:
    perm = ctx.config.permission or {}
    ctx.reply(f"Current permission config:\n{json.dumps(perm, indent=2)}")


def _config(ctx: CommandContext, args: str) -> None:
    action = args.strip()
    if action == "print" or action.startswith("print "):
        show = "--show-secrets" in action
        ctx.reply(json.dumps(ctx.config.as_dict(show_secrets=show), indent=2))
        return
    if action == "validate":
        problems = _validate_config(ctx.config)
        if problems:
            ctx.reply("Config issues:\n" + "\n".join(f"  - {p}" for p in problems))
        else:
            ctx.reply("Config is valid.")
        return
    ctx.reply("Usage: /config print|validate")


def _validate_config(cfg: Config) -> list[str]:
    """Best-effort config validation: known providers/themes, rotation shape, permission values."""
    problems: list[str] = []
    from .providers import FREE_PROVIDERS

    known = set(FREE_PROVIDERS) | {"opencode", "zen", "anthropic", "openai", "ollama"} | set(cfg.providers or {})
    if cfg.provider and cfg.provider not in known:
        problems.append(f"unknown provider '{cfg.provider}' (known: {', '.join(sorted(known))})")
    if not cfg.model:
        problems.append("no model configured (set 'model' or use /model)")
    for i, lane in enumerate(cfg.rotation or []):
        if not isinstance(lane, dict):
            problems.append(f"rotation[{i}] must be an object {{provider, model}}")
            continue
        if lane.get("provider") not in known:
            problems.append(f"rotation[{i}]: unknown provider '{lane.get('provider')}'")
        if not lane.get("model"):
            problems.append(f"rotation[{i}]: missing model")
    valid_actions = ("allow", "deny", "ask")
    for tool, action in (cfg.permission or {}).items():
        if isinstance(action, dict):
            for pattern, sub in action.items():
                if sub not in valid_actions:
                    problems.append(f"permission['{tool}']['{pattern}'] = '{sub}' (want one of {', '.join(valid_actions)})")
        elif action not in valid_actions:
            problems.append(f"permission['{tool}'] = '{action}' (want one of {', '.join(valid_actions)})")
    tts_engine = str(getattr(cfg, "tts_engine", "auto") or "auto").lower()
    if tts_engine not in ("auto", "offline", "elevenlabs"):
        problems.append(f"tts.engine = '{tts_engine}' (want one of auto, offline, elevenlabs)")
    try:
        rate = float(getattr(cfg, "tts_rate", 1.0))
        if not 0.25 <= rate <= 4.0:
            problems.append(f"tts.rate = {rate} (want 0.25-4.0)")
    except (TypeError, ValueError):
        problems.append("tts.rate must be a number (0.25-4.0)")
    try:
        pitch = float(getattr(cfg, "tts_pitch", 1.0))
        if not 0.5 <= pitch <= 2.0:
            problems.append(f"tts.pitch = {pitch} (want 0.5-2.0)")
    except (TypeError, ValueError):
        problems.append("tts.pitch must be a number (0.5-2.0)")
    lr = str(getattr(cfg, "low_ram", "auto") or "auto").lower()
    if lr not in ("auto", "on", "off"):
        problems.append(f"low_ram = '{lr}' (want one of auto, on, off)")
    try:
        w = int(getattr(cfg, "chat_live_window", 120))
        if w < 0:
            problems.append(f"chat_live_window = {w} (want 0 or more, 0 = unlimited)")
    except (TypeError, ValueError):
        problems.append("chat_live_window must be a number (0 = unlimited)")
    return problems


def _theme(ctx: CommandContext, args: str) -> None:
    theme = args.strip().lower()
    try:
        from .tui.theme import DARK_THEMES, LIGHT_THEMES, custom_names, set_active_theme, theme_names
    except ImportError as _oc_exc:
        swallow(_oc_exc, "commands.py:/theme needs textual")
        ctx.reply("Themes need textual: pip install 'opencode-py[tui]'")
        return

    if not theme:
        current = ctx.config.theme
        valid = theme_names()
        lines = [f"Current theme: {current}", ""]
        customs = custom_names()
        if customs:
            lines.append("Custom themes:")
            lines += [f"  /{name}" if name != current else f"  /{name}  ← current" for name in customs]
            lines.append("")
        lines.append("Dark themes:")
        lines += [f"  /{name}" if name != current else f"  /{name}  ← current" for name in DARK_THEMES if name in valid]
        lines.append("")
        lines.append("Light themes:")
        lines += [f"  /{name}" if name != current else f"  /{name}  ← current" for name in LIGHT_THEMES]
        lines += ["", "Switch: /theme <name>  (applies immediately)", "Create/edit: open the theme popup in the TUI"]
        ctx.reply("\n".join(lines))
        return
    if theme not in theme_names():
        ctx.reply(
            f"Unknown theme '{theme}'. Available (dark first):\n  "
            + ", ".join(theme_names())
        )
        return
    applied = set_active_theme(theme)
    ctx.config.theme = applied
    try:
        save_config(ctx.config)
    except Exception as e:
        ctx.reply(f"Theme set to {applied}, but NOT saved ({e}).")
        return
    ctx.reply(f"Theme set to {applied}.")


def _help(ctx: CommandContext, args: str) -> None:
    _print_help(ctx, args)


def _exit(ctx: CommandContext, args: str) -> None:
    if ctx.exit_app:
        ctx.exit_app()
    else:
        ctx.reply("Exiting.")


def _resume(ctx: CommandContext, args: str) -> None:
    session_id = args.strip()
    if not session_id:
        ctx.reply("Usage: /resume <session-id>  (see /sessions)")
        return
    if ctx.resume:
        ctx.resume(session_id)
    else:
        ctx.reply("Resuming needs the interactive TUI; use `opencode-py` and Ctrl+R.")


def _sessions(ctx: CommandContext, args: str) -> None:
    from .session import list_sessions

    sessions = list_sessions()
    if not sessions:
        ctx.reply("No saved sessions.")
        return
    lines = ["Sessions:"]
    shown = 0
    for s in sessions:
        if shown >= 20:
            break
        if getattr(s, "parent_id", None):
            continue  # launched agents live under their parent (see picker)
        title = s.title or "(untitled)"
        lines.append(f"  {s.id[:12]}  {title}  ({s.model or '?'})  [{s.agent}]")
        shown += 1
        kids = [k for k in sessions if getattr(k, "parent_id", None) == s.id]
        for k in kids[:5]:
            kt = k.title or "(untitled)"
            lines.append(f"    ↳ {k.id[:12]}  {kt}  ({k.model or '?'})  [{k.agent}]")
    ctx.reply("\n".join(lines))


def _export(ctx: CommandContext, args: str) -> None:
    """Write a session transcript (tool calls included) to a Markdown file.

    With ctx.preview_only (the popup preview pass) nothing is written — the
    preview just says what WOULD be exported. The real write happens exactly
    once, when the user presses Run."""
    from pathlib import Path

    from .session import load_session, session_to_markdown

    session_id = args.strip() or (ctx.engine.session_id if ctx.engine else "")
    if not session_id:
        ctx.reply("Usage: /export <session-id>  (omit to export the current session)")
        return
    sess = load_session(session_id)
    if sess is None:
        ctx.reply(f"Session {session_id} not found.")
        return
    base = Path(ctx.worktree) if getattr(ctx, "worktree", "") else Path.cwd()
    path = base / f"opencode-session-{sess.id[:12]}.md"
    if getattr(ctx, "preview_only", False):
        ctx.reply(
            f"Will export {len(sess.messages)} messages → {base / path.name}"
        )
        return
    try:
        path.write_text(session_to_markdown(sess), encoding="utf-8")
    except OSError as e:
        ctx.reply(f"Export failed: {e}")
        return
    ctx.reply(f"Exported {len(sess.messages)} messages → {path.name}")


def _agent(ctx: CommandContext, args: str) -> None:
    name = args.strip().lower()
    if not name:
        # bare /agent in headless: list agents; the TUI intercepts first
        # and opens the picker instead (see app._run_command).
        try:
            from .permission import agents_config_error as _cfg_err
            from .permission import list_agents as _list

            err = _cfg_err(ctx.config)
            if err is not None:
                ctx.reply(f"Agent config error: {err}. Fix opencode.json `agents`.")
                return
            names = ", ".join(n for n, _d, _c in _list(ctx.config))
        except Exception as e:
            ctx.reply(f"Could not list agents (config error: {e}).")
            return
        ctx.reply(f"Agents: {names}. Usage: /agent <name>")
        return
    if ctx.set_agent:
        try:
            from .permission import agents_config_error as _cfg_err
            from .permission import list_agents as _list

            err = _cfg_err(ctx.config)
            if err is not None:
                ctx.reply(f"Agent config error: {err}. Fix opencode.json `agents`.")
                return
            known = {n for n, _d, _c in _list(ctx.config)}
        except Exception as e:
            ctx.reply(f"Could not switch agent (config error: {e}).")
            return
        if name not in known:
            ctx.reply(f"Unknown agent '{name}'. Agents: {', '.join(sorted(known))}")
            return
        ctx.set_agent(name)
        ctx.reply(f"Switched to {name} agent.")
    else:
        ctx.reply("Agent switching needs the interactive TUI; use --agent here.")


def _model(ctx: CommandContext, args: str) -> None:
    model = args.strip()
    if not model:
        ctx.reply("Usage: /model <model-id>  (e.g. x-preview-f-free, big-pickle)")
        return
    if ctx.set_model:
        ctx.set_model(model)
        ctx.reply(f"Model set to opencode/{model}.")
    else:
        # no UI callback wired (headless one-shot): a silent no-op that still
        # claims success made users think they had switched models
        ctx.reply(
            f"Model switching needs the interactive TUI (run `opencode-py`);"
            f" this session stays on {ctx.config.model}."
            " Or launch with --model."
        )


def _thinking(ctx: CommandContext, args: str) -> None:
    """Show/hide the model's thought bubbles (`+ Thought` rows) AND set
    reasoning effort (official variant parity).

    - `/thinking` — toggle the newest thought open/closed (same as Ctrl+Shift+E)
    - `/thinking show` — expand every thought in this session
    - `/thinking hide` — collapse every thought in this session
    - `/thinking last` — print the newest thought's text (headless-friendly)
    - `/thinking <level>` — set reasoning effort for models that support it
      (e.g. `/thinking high` for muse-spark; levels come from the live
      catalog so future models work too). Persists to config; applies on the
      next turn. `/thinking off` clears it back to the gateway default.

    TUI runs delegate to the live chat (no state of its own is touched);
    headless runs read the newest reasoning from the engine/session history.
    Preview-safe: never mutates anything when preview_only.
    """
    arg = (args or "").strip().lower()

    def _history_texts() -> list[str]:
        texts: list[str] = []
        try:
            history = []
            if ctx.engine is not None and hasattr(ctx.engine, "get_history"):
                history = ctx.engine.get_history() or []
            if not history and ctx.session is not None:
                history = getattr(ctx.session, "messages", None) or []
            for m in history:
                if isinstance(m, dict) and m.get("reasoning_content"):
                    texts.append(str(m["reasoning_content"]))
        except Exception as _oc_exc:
            swallow(_oc_exc, "commands.py:445")
            pass
        return texts

    app = None
    try:
        from textual.app import App as _App

        import inspect

        frame = inspect.currentframe()
        while frame is not None:
            cand = frame.f_locals.get("self")
            if isinstance(cand, _App):
                app = cand
                break
            frame = frame.f_back
    except Exception:
        app = None

    chat = None
    if app is not None:
        try:
            for meth in ("_chat_for", "_active_session"):
                _ = getattr(app, meth, None)
            sid = getattr(app, "_current_session_id", "")
            chat = app._chat_for(sid) if sid else None
        except Exception:
            chat = None

    if arg in ("", "toggle"):
        if chat is not None and not ctx.preview_only:
            try:
                chat.toggle_last_reasoning()
                ctx.reply("Toggled the newest thought.")
            except Exception:
                ctx.reply("No thought to toggle yet.")
        else:
            texts = _history_texts()
            ctx.reply(texts[-1].strip()[:2000] if texts else "No thought yet in this session.")
        return
    if arg in ("show", "on", "expand", "all"):
        if chat is not None and not ctx.preview_only:
            try:
                n = chat.set_all_reasoning(True)
                ctx.reply(f"Expanded {n} thought(s).")
            except Exception:
                ctx.reply("No thoughts to expand yet.")
        else:
            texts = _history_texts()
            if not texts:
                ctx.reply("No thoughts yet in this session.")
            else:
                ctx.reply("\n\n---\n\n".join(t.strip()[:2000] for t in texts[-5:]))
        return
    if arg in ("hide", "collapse"):
        if chat is not None and not ctx.preview_only:
            try:
                n = chat.set_all_reasoning(False)
                ctx.reply(f"Collapsed {n} thought(s).")
            except Exception:
                ctx.reply("No thoughts to collapse yet.")
        else:
            ctx.reply(f"{len(_history_texts())} thought(s) in this session (already collapsed in text mode).")
        return
    if arg in ("off", "default", "auto", "none", "clear"):
        if ctx.preview_only:
            ctx.reply("Would clear reasoning effort back to the gateway default.")
            return
        try:
            ctx.config.reasoning_effort = ""
            from .config import save_config
            save_config(ctx.config)
        except Exception as _oc_exc:
            swallow(_oc_exc, "commands.py:518")
            pass
        try:
            if app is not None and hasattr(app, "_apply_runtime_settings"):
                app._apply_runtime_settings()
        except Exception as _oc_exc:
            swallow(_oc_exc, "commands.py:523")
            pass
        ctx.reply("Reasoning effort cleared — gateway default from the next turn.")
        return
    if arg in ("last", "show-last", "print"):
        texts = _history_texts()
        if chat is not None:
            try:
                t = chat.last_reasoning_text()
                ctx.reply(t[:2000] if t else "No thought yet in this session.")
                return
            except Exception as _oc_exc:
                swallow(_oc_exc, "commands.py:534")
                pass
        ctx.reply(texts[-1].strip()[:2000] if texts else "No thought yet in this session.")
        return
    # effort level? valid only when the CURRENT model advertises it in the
    # live catalog (future models + paid models included automatically).
    try:
        from .providers.rotation import model_effort_levels
        model_id = ""
        provider_id = ""
        try:
            if ctx.engine is not None:
                model_id = getattr(ctx.engine, "model_id", "") or ""
                provider_id = getattr(ctx.engine, "provider_id", "") or ""
            if not model_id:
                model_id = getattr(ctx.config, "model", "") or ""
                provider_id = getattr(ctx.config, "provider", "") or "opencode"
        except Exception as _oc_exc:
            swallow(_oc_exc, "commands.py:551")
            pass
        levels = model_effort_levels(model_id, provider_id or "opencode")
        if levels and arg in [str(v).lower() for v in levels]:
            if ctx.preview_only:
                ctx.reply(f"Would set reasoning effort to {arg} for {model_id or 'current model'}.")
                return
            try:
                ctx.config.reasoning_effort = arg
                from .config import save_config
                save_config(ctx.config)
            except Exception as _oc_exc:
                swallow(_oc_exc, "commands.py:562")
                pass
            try:
                if app is not None and hasattr(app, "_apply_runtime_settings"):
                    app._apply_runtime_settings()
            except Exception as _oc_exc:
                swallow(_oc_exc, "commands.py:567")
                pass
            ctx.reply(f"Reasoning effort → {arg} (applies from the next turn).")
            return
        if levels:
            ctx.reply(f"Usage: /thinking [show|hide|last|{'|'.join(levels)}|off]  (this model supports: {', '.join(levels)})")
        else:
            ctx.reply("Usage: /thinking [show|hide|last]  (this model has no effort levels — it thinks at a fixed level)")
        return
    except Exception as _oc_exc:
        swallow(_oc_exc, "commands.py:576")
        pass
    ctx.reply("Usage: /thinking [show|hide|last]  (bare /thinking toggles the newest thought)")


def _undo(ctx: CommandContext, args: str) -> None:
    if not ctx.engine:
        ctx.reply("No active session to undo.")
        return
    ctx.reply(ctx.engine.undo_last())


def _compact(ctx: CommandContext, args: str) -> None:
    if ctx.engine:
        history = ctx.engine.get_history()
        if len(history) <= 2:
            ctx.reply("History too short to compact.")
            return
        # Upstream opencode's /compact runs the same summary-based compaction
        # as auto-compaction (anchored summary + recent tail), it does not just
        # drop the older turns. Delegate to the engine so the model summarizes
        # and the resulting ` Compaction ` panel shows in the session.
        try:
            summary = ctx.engine.force_compact()
        except Exception as e:
            ctx.reply(f"Compaction failed: {e}")
            return
        if summary:
            ctx.reply("History compacted.")
        else:
            ctx.reply("History compacted (kept last turn).")
    else:
        ctx.reply("No active session.")


def _exit_fn(ctx: CommandContext, args: str) -> None:
    _exit(ctx, args)


def _plugins(ctx: CommandContext, args: str) -> None:
    """List/add/remove/enable/disable/test plugins and hooks from chat."""
    from . import plugin_manager as _pm

    worktree = getattr(ctx, "worktree", "") or ""
    raw = (args or "list").strip()
    tokens = raw.split()
    sub = tokens[0].lower() if tokens else "list"
    if sub in ("list", "ls"):
        plugins = _pm.list_plugins(ctx.config)
        hooks = _pm.list_hooks(ctx.config)
        disabled = _pm.list_disabled(ctx.config)
        if not plugins and not hooks:
            ctx.reply("No plugins configured. Add one with: /plugins add <name> <target>")
            return
        rows = []
        for name in plugins:
            rows.append(f"  {'off' if name in disabled else 'on ':>3}  {name}  (tool plugin)")
        for hook in hooks:
            name = str(hook.get("name") or "?")
            state = "off" if (hook.get("disabled") is True or f"hook:{name}" in disabled) else "on"
            rows.append(f"  {state:>3}  {name}  (hook: {hook.get('target')} · {','.join(hook.get('events') or [])})")
        ctx.reply("Plugins:\n" + "\n".join(rows))
        return
    if sub in ("add", "hook") and len(tokens) >= 3:
        name, target = tokens[1], tokens[2]
        if sub == "add" and "." not in target and "/" not in target and not target.endswith(".py"):
            ctx.reply("Usage: /plugins add <name> <module-or-file>   e.g. /plugins add fmt ./hooks/format.py")
            return
        if sub == "hook":
            events = [e for e in (tokens[3] if len(tokens) > 3 else "").split(",") if e]
            _pm.save_hook(name, target, events, worktree, False)
            ctx.reply(f"Added hook '{name}' → {target} ({', '.join(events) or 'tool.after'})")
            return
        _pm.save_plugin(name, target, worktree, False)
        ctx.reply(f"Added tool plugin '{name}' → {target}")
        return
    if sub in ("remove", "rm", "delete") and len(tokens) >= 2:
        name = tokens[1]
        ok = _pm.remove(name, worktree, False)
        ctx.reply(f"Removed '{name}'." if ok else f"No plugin '{name}' found.")
        return
    if sub in ("on", "enable", "off", "disable") and len(tokens) >= 2:
        name = tokens[1]
        ok = _pm.set_enabled(name, sub in ("on", "enable"), worktree, False)
        ctx.reply(f"'{name}' {'enabled' if sub in ('on', 'enable') else 'disabled'}." if ok else f"No plugin '{name}' found.")
        return
    if sub in ("test", "check") and len(tokens) >= 2:
        name = tokens[1]
        ctx.reply(_pm.test_plugin(name, ctx.config, worktree))
        return
    ctx.reply("Usage: /plugins [list|add <name> <target>|hook <name> <file> <events>|on <name>|off <name>|remove <name>|test <name>]")
    return


def _mcp(ctx: CommandContext, args: str) -> None:
    import shlex

    from . import mcp_manager as _mm

    try:
        tokens = shlex.split(args)
    except ValueError:
        tokens = args.split()
    sub = tokens[0].lower() if tokens else ""
    if sub in ("", "picker", "ui"):
        if getattr(ctx, "preview_only", False):
            ctx.reply("Would open the MCP manager (list/add/remove/test).")
            return
        ctx.reply(
            "MCP servers (each tool appears as mcp__<server>__<tool>):\n"
            + _mcp_list_text(ctx)
            + "\nAdd: /mcp add <name> -- <command...>\n"
            "  e.g. /mcp add files -- python -m my_server\n"
            "  e.g. /mcp add files -- npx -y @modelcontextprotocol/server-filesystem /tmp\n"
            "Remove: /mcp remove <name>   Test: /mcp test [name]"
        )
        return
    if sub in ("list", "ls"):
        ctx.reply("MCP servers:\n" + _mcp_list_text(ctx))
        return
    if sub in ("add", "create"):
        try:
            name, command, cmd_args, is_global = _mm.parse_add_args(args)
        except ValueError as e:
            ctx.reply(str(e))
            return
        if getattr(ctx, "preview_only", False):
            ctx.reply(f"Would add MCP server '{name}': {command} {' '.join(cmd_args)}".rstrip())
            return
        lines = [f"Adding '{name}': {command} {' '.join(cmd_args)}".rstrip()]
        for w in _mm.check_command(command, cmd_args):
            lines.append(f"  ! {w}")
        lines.append("  testing (up to ~15s)...")
        lines.append("  " + _mm.test_server(name, command, cmd_args, timeout=15.0))
        try:
            path = _mm.save_server(name, command, cmd_args, ctx.worktree, is_global)
        except OSError as e:
            lines.append(f"  save FAILED: {e}")
            ctx.reply("\n".join(lines))
            return
        raw = ctx.config.raw if isinstance(getattr(ctx.config, "raw", None), dict) else {}
        servers = raw.get("mcpServers")
        if not isinstance(servers, dict):
            servers = {}
            raw["mcpServers"] = servers
        servers[name] = {"command": command, "args": list(cmd_args)}
        lines.append(f"  saved → {path}")
        lines.append("  " + _mm.refresh_engine_mcp(ctx.engine, ctx.config))
        ctx.reply("\n".join(lines))
        return
    if sub in ("remove", "rm", "del", "delete"):
        rest = [t for t in tokens[1:] if t != "--global"]
        is_global = "--global" in tokens
        if not rest:
            ctx.reply("Usage: /mcp remove <name> [--global]")
            return
        name = rest[0]
        if getattr(ctx, "preview_only", False):
            ctx.reply(f"Would remove MCP server '{name}'.")
            return
        if is_global:
            path = _mm.remove_server(name, ctx.worktree, True)
            paths = [path] if path else []
        else:
            # No flag: remove from BOTH scopes — the merged config can't tell
            # which file provided the entry, and single-scope deletes
            # resurrect on restart.
            paths = _mm.remove_server_everywhere(name, ctx.worktree)
            path = paths[0] if paths else None
        raw = getattr(ctx.config, "raw", None)
        if isinstance(raw, dict) and isinstance(raw.get("mcpServers"), dict):
            raw["mcpServers"].pop(name, None)
        if path is None:
            ctx.reply(f"No MCP server '{name}' found.")
            return
        extra = f" (+{len(paths) - 1} more scope)" if len(paths) > 1 else ""
        ctx.reply(f"Removed '{name}' → {path}{extra}\n" + _mm.refresh_engine_mcp(ctx.engine, ctx.config))
        return
    if sub in ("test", "check", "ping"):
        servers = _mm.list_servers(ctx.config)
        rest = tokens[1:]
        if rest:
            name = rest[0]
            spec = servers.get(name)
            if not isinstance(spec, dict) or not (spec.get("command") or spec.get("url")):
                ctx.reply(f"No MCP server '{name}' found.")
                return
            if getattr(ctx, "preview_only", False):
                ctx.reply(f"Would test MCP server '{name}'.")
                return
            if spec.get("url"):
                ctx.reply(_mm.test_server(
                    name, None, None, url=spec.get("url"),
                    transport=str(spec.get("transport") or "streamable-http"),
                    headers=spec.get("headers") if isinstance(spec.get("headers"), dict) else None,
                    auth=spec.get("auth") if isinstance(spec.get("auth"), dict) else None,
                ))
            else:
                ctx.reply(_mm.test_server(name, spec.get("command"), spec.get("args") or []))
            return
        if not servers:
            ctx.reply("No MCP servers configured. Open /mcp to add one.")
            return
        if getattr(ctx, "preview_only", False):
            ctx.reply(f"Would test {len(servers)} MCP server(s).")
            return
        lines = []
        for n, s in servers.items():
            if not isinstance(s, dict) or not (s.get("command") or s.get("url")):
                lines.append(f"{n}: BAD ENTRY — remove and re-add it (/mcp remove {n})")
                continue
            if s.get("url"):
                lines.append(_mm.test_server(
                    n, None, None, url=s.get("url"),
                    transport=str(s.get("transport") or "streamable-http"),
                    headers=s.get("headers") if isinstance(s.get("headers"), dict) else None,
                    auth=s.get("auth") if isinstance(s.get("auth"), dict) else None,
                ))
            else:
                lines.append(_mm.test_server(n, s.get("command"), (s.get("args") or [])))
        ctx.reply("\n".join(lines) or "No testable servers.")
        return
    if sub in ("on", "enable", "off", "disable") and tokens[1:]:
        name = tokens[1]
        enable = sub in ("on", "enable")
        if _mm.set_server_enabled(name, enable, getattr(ctx, "worktree", "") or "", False):
            ctx.reply(f"MCP '{name}' {'enabled' if enable else 'disabled'}.")
            return
        ctx.reply(f"No MCP server '{name}' found.")
        return
    ctx.reply(
        f"Unknown /mcp action '{sub}'. Usage: /mcp [list|add <name> -- <command...>|remove <name>|test [name]]"
    )


def _mcp_list_text(ctx: CommandContext) -> str:
    from . import mcp_manager as _mm

    servers = _mm.list_servers(ctx.config)
    if not servers:
        return "  (none configured)"
    rows = []
    for name, spec in servers.items():
        if isinstance(spec, dict) and spec.get("command"):
            cmdline = str(spec.get("command")) + "".join(f" {a}" for a in (spec.get("args") or []))
            rows.append(f"  {name}: {cmdline}")
        else:
            rows.append(f"  {name}: (bad entry — remove and re-add it)")
    rows.append("  tools appear as mcp__<server>__<tool> from the next turn.")
    return "\n".join(rows)


def _server(ctx: CommandContext, args: str) -> None:
    from .server_control import server_status, start_server, stop_server

    action = (args or "status").strip().lower()
    if action in ("on", "start"):
        if getattr(ctx, "preview_only", False):
            ctx.reply("Would start the local OpenCode server.")
            return
        config = getattr(ctx.config, "server", {}) or {}
        host = str(config.get("host") or "127.0.0.1")
        try:
            port = int(config.get("port") or 4096)
        except (TypeError, ValueError):
            port = 4096
        cors = config.get("cors") or []
        if isinstance(cors, (list, tuple)):
            cors = ",".join(str(item) for item in cors)
        result = start_server(ctx.worktree, host=host, port=port, cors=str(cors) or None)
        ctx.reply(f"Server started: {result['url']}\nServer token: {result['token']}")
        return
    if action in ("off", "stop"):
        if getattr(ctx, "preview_only", False):
            ctx.reply("Would stop the local OpenCode server.")
            return
        result = stop_server()
        ctx.reply("Server stopped" if result.get("stopped") else "Server is not running")
        return
    if action in ("status", "state"):
        result = server_status()
        if result.get("running"):
            ctx.reply(f"Server running: {result.get('url', '')}")
        else:
            ctx.reply("Server stopped")
        return
    ctx.reply("Usage: /server [start|stop|status]")


CUSTOM_COMMANDS: dict[str, dict[str, Any]] = {}


def register_custom_commands(reg: CommandRegistry) -> None:
    """Register user-defined slash commands from ~/.config or project config."""
    from .config import load_config
    from .command_manager import list_commands, list_disabled

    try:
        cfg = load_config()
        entries = list_commands(cfg)
        disabled = list_disabled(cfg)
    except Exception as _oc_exc:
        swallow(_oc_exc, "commands.py:877")
        return
    for name, entry in entries.items():
        if name in disabled or reg.get(name) is not None:
            continue
        CUSTOM_COMMANDS[name] = entry

        def _make(cmd_name: str, cmd_entry: dict[str, Any]):
            def handler(ctx: CommandContext, args: str) -> None:
                ctx._custom_name = cmd_name
                ctx._custom_entry = cmd_entry
                _custom_command(ctx, args)
            return handler

        reg.register(Command(
            name,
            [],
            str(entry.get("description") or f"Custom command /{name}"),
            _make(name, entry),
            metadata={"custom": True, "entry": entry, "command_name": name},
        ))


def _custom_command(ctx: CommandContext, args: str) -> None:
    """Send a user-defined command's stored prompt as a normal agent turn."""
    name = getattr(ctx, "_custom_name", "")
    entry = (getattr(ctx, "_custom_entry", {}) or {})
    prompt = str(entry.get("prompt") or "").strip()
    if not prompt:
        ctx.reply(f"/{name} has no prompt.")
        return
    extra = (args or "").strip()
    text = f"{prompt}\n\n{extra}" if extra else prompt
    worker = getattr(ctx, "on_run_prompt", None)
    if callable(worker):
        worker(text)
        return
    ctx.reply(text)


def _webdoctor(ctx: CommandContext, args: str) -> None:
    """Ping every webreach site channel and report OK / N/A / FAIL + a fix.

    Same shape as `--check`, but for the web channels instead of the provider
    lanes: which sites the agent can actually read right now, and how to fix the
    ones it cannot. ``/webdoctor github`` checks a single site.
    """
    from .tools.webreach import doctor_text

    site = (args or "").strip().split()[0] if (args or "").strip() else ""
    if ctx.preview_only:
        sites = ", ".join(name for name, _u, _a in _probe_sites())
        ctx.reply(f"Ping each webreach site channel ({sites}) and report OK/fail + fix.")
        return
    ctx.reply(doctor_text(only=site))


def _probe_sites() -> list[tuple[str, str, str]]:
    from .tools.webreach import PROBES

    return list(PROBES)


def build_registry() -> CommandRegistry:
    reg = CommandRegistry()
    for cmd in [
        Command("agent", [], "Switch agent (build|plan|explore)", _agent),
        Command("commands", [], "Manage custom slash commands", _commands, preview=False),
        Command("compact", [], "Compact conversation history", _compact, preview=False),
        Command("config", [], "Print/validate config", _config),
        Command("connect", [], "Add a provider/API key", _connect, preview=False),
        Command("exit", [], "Exit", _exit_fn, preview=False),
        Command("export", [], "Export session transcript to Markdown", _export),
        Command("help", [], "Show help", _help),
        Command("init", [], "Guided AGENTS.md setup", _init, preview=False),
        Command("mcp", [], "Manage MCP servers (list/add/remove/test)", _mcp),
        Command("model", [], "Switch model", _model),
        Command("models", [], "Open the model picker", _models),
        Command("new", [], "Start a new session", _new, preview=False),
        Command("permissions", [], "Show permission config", _permissions),
        Command("plugins", [], "Manage plugins and event hooks", _plugins, preview=False),
        Command("provider", [], "Manage provider keys and models", _providers, preview=False),
        # NB: no "sessions" alias here — /sessions is its own command below;
        # aliasing it to /resume was a silent-collision trap.
        Command("resume", [], "Resume a session", _resume),
        Command("review", [], "Review changes", _review),
        Command("sessions", [], "Open the session picker (Ctrl+R)", _sessions),
        Command("setting", [], "Open Settings", _settings, preview=False),
        Command("server", [], "Start, stop, or check the local server", _server, preview=False),
        Command("skills", [], "Skill manager (view/add/rename/edit/delete)", _skills),
        Command("theme", [], "Switch theme", _theme),
        Command("webdoctor", [], "Check which webreach site channels work", _webdoctor),
        Command("thinking", [], "Show/hide model thoughts", _thinking),
        Command("undo", [], "Revert last tool action", _undo, preview=False),
        Command("cleanup", [], "Squeeze old chats + prune past the cap", _cleanup),
        Command("pin", [], "Pin a session so cleanup never deletes it", _pin),
    ]:
        reg.register(cmd)
    register_custom_commands(reg)
    return reg


def _commands(ctx: CommandContext, args: str) -> None:
    """List/add/edit/remove/enable/disable custom slash commands from chat."""
    from . import command_manager as _cm

    worktree = getattr(ctx, "worktree", "") or ""

    def _refresh() -> None:
        # command_manager writes the file itself and never touches ctx.config,
        # so without this the very next `/commands list` in the same session
        # reported the state from BEFORE the change.
        try:
            from pathlib import Path as _P
            from .config import load_config
            fresh = load_config(_P(worktree or "."))
            ctx.config.raw = fresh.raw
            ctx.config.commands = fresh.commands
        except Exception as _oc_exc:
            swallow(_oc_exc, "commands.py:/commands refresh")

    parts = (args or "list").strip().split()
    sub = parts[0].lower() if parts else "list"
    if sub in ("list", "ls"):
        entries = _cm.list_commands(ctx.config)
        disabled = _cm.list_disabled(ctx.config)
        if not entries:
            ctx.reply("No custom commands. Add one with: /commands add")
            return
        rows = [f"  {'off' if name in disabled else 'on ':>3}  /{name}  {_cm.describe(name, entry)}" for name, entry in entries.items()]
        ctx.reply("Custom commands:\n" + "\n".join(rows))
        return
    if sub == "add" and len(parts) >= 2:
        name = parts[1]
        ctx.reply(f"Open the /commands popup to add /{name} with a description and prompt.")
        return
    if sub in ("on", "enable", "off", "disable") and len(parts) >= 2:
        name = parts[1]
        ok = _cm.set_enabled(name, sub in ("on", "enable"), worktree, False)
        if ok:
            _refresh()
        ctx.reply(f"/{name} {'enabled' if sub in ('on', 'enable') else 'disabled'}." if ok else f"No command /{name} found.")
        return
    if sub in ("remove", "rm", "delete") and len(parts) >= 2:
        name = parts[1]
        ok = _cm.remove_command(name, worktree, False)
        if ok:
            _refresh()
        ctx.reply(f"Removed /{name}." if ok else f"No command /{name} found.")
        return
    ctx.reply("Usage: /commands [list|add <name>|on <name>|off <name>|remove <name>] — or open the popup with no arguments")


def _providers(ctx: CommandContext, args: str) -> None:
    from . import provider_manager as _pm
    from .auth import Auth

    parts = (args or "list").strip().split()
    sub = parts[0].lower() if parts else "list"
    auth = Auth()
    if sub in ("list", "ls"):
        rows = _pm.rows(auth)
        ctx.reply("Providers:\n" + "\n".join(f"  {r['id']}: {r['status']}" for r in rows))
        return
    if sub in ("key", "login") and len(parts) >= 2:
        ctx.reply(f"Open /provider for {parts[1]} settings.")
        return
    ctx.reply("Usage: /provider [list] — open the popup to add keys, check status, and refresh models.")


def _cleanup(ctx: CommandContext, args: str) -> None:
    from .session import cleanup_report, session_disk_stats, squeeze_sessions, vacuum_sessions

    parts = (args or "").strip().split()
    cap = 0
    try:
        cap = int(getattr(ctx.config, "session_file_cap", 0) or 0)
    except (TypeError, ValueError):
        cap = 0
    if parts[:1] == ["dry"] or ctx.preview_only:
        before = cleanup_report()
        try:
            from . import snapshots as _snaps

            snap = _snaps.report()
            extra = (f" Snapshots: {int(snap.get('count', 0))} folder(s), "
                     f"{float(snap.get('bytes', 0) or 0) / (1024 * 1024):.1f}MB, "
                     f"{len(snap.get('delete') or [])} past the keep policy.")
        except Exception as _oc_exc:
            swallow(_oc_exc, "commands.py:_cleanup dry")
            extra = ""
        ctx.reply(f"Dry run — {before}.{extra} Run /cleanup to squeeze + prune.")
        return
    live = set()
    try:
        if ctx.get_session is not None:
            s = ctx.get_session()
            if s is not None and getattr(s, "id", ""):
                live.add(str(s.id))
    except Exception as _oc_exc:
        swallow(_oc_exc, "commands.py:1023")
        pass
    sq = squeeze_sessions(live_ids=live)
    vac = vacuum_sessions(cap) if cap > 0 else {"pruned": 0, "bytes_freed": 0, "kept": 0}
    # Snapshot folders are pruned by the same command, and the live session is
    # passed in so an open-but-idle chat can never lose its safety net just
    # because it was started long enough ago.
    try:
        from . import snapshots as _snaps

        swept = _snaps.sweep_trash()
        snap = _snaps.report(keep_ids=live)
        gone = _snaps.prune(keep_ids=live)
    except Exception as _oc_exc:
        swallow(_oc_exc, "commands.py:_cleanup")
        swept, snap, gone = 0, {"bytes": 0, "count": 0}, []
    after = session_disk_stats()
    mb = float(after.get("bytes", 0) or 0) / (1024 * 1024)
    gzmb = float(after.get("gz_bytes", 0) or 0) / (1024 * 1024)
    snapmb = float(snap.get("bytes", 0) or 0) / (1024 * 1024)
    extra = ""
    if swept:
        extra += f", finished {int(swept)} interrupted snapshot deletion(s)"
    extra += (
        f", deleted {len(gone)} old snapshot folder(s) — "
        f"{int(snap.get('count', 0))} kept, {snapmb:.1f}MB on disk"
    )
    ctx.reply(
        f"Cleaned {int(sq.get('squeezed', 0))} old chats "
        f"({float(sq.get('bytes_saved', 0)) / (1024 * 1024):.1f}MB squeezed), "
        f"deleted {int(vac.get('pruned', 0))} past the cap{extra} — "
        f"{int(after.get('files', 0))} chats, {mb:.1f}MB + {gzmb:.1f}MB squeezed on disk."
    )


def _pin(ctx: CommandContext, args: str) -> None:
    from .session import set_pinned

    parts = (args or "").strip().split()
    sid = parts[0] if parts else ""
    if not sid and ctx.get_session is not None:
        try:
            s = ctx.get_session()
            sid = str(getattr(s, "id", "") or "")
        except Exception:
            sid = ""
    if not sid:
        ctx.reply("Usage: /pin [session-id]  (no id = pin this chat; /unpin to release)")
        return
    if parts[:1] == ["off"] or parts[:1] == ["unpin"]:
        ok = set_pinned(sid, False)
        ctx.reply(f"Unpinned {sid[:12]}." if ok else "Unpin failed.")
        return
    ok = set_pinned(sid, True)
    ctx.reply(f"Pinned {sid[:12]} — cleanup will never delete it." if ok else "Pin failed.")


def _skills(ctx: CommandContext, args: str) -> None:
    """List, validate, or reload SKILL.md skills."""
    from .tools import skill as _sk

    sub = (args or "").strip().lower()
    if sub in ("reload", "refresh", "clear"):
        _sk.clear_cache()
        skills = _sk.list_skills(fresh=True)
        ctx.reply(f"Reloaded {len(skills)} skill(s).")
        return
    if sub in ("validate", "check", "test"):
        from pathlib import Path as _P

        roots: list[_P] = []
        try:
            from .globals import resolve_worktree

            wt = resolve_worktree(_P(ctx.worktree or "."))
            roots.append(wt)
        except Exception as _oc_exc:
            swallow(_oc_exc, "commands.py:1079")
            pass
        seen: list[str] = []
        for base in roots:
            for subdir in _sk.SKILL_DIRS:
                root = base / subdir
                try:
                    kids = sorted(p for p in root.iterdir() if p.is_dir())
                except OSError as _oc_exc:
                    swallow(_oc_exc, "opencode_py/commands.py:1099")
                    continue
                for kid in kids:
                    f = kid / "SKILL.md"
                    if not f.exists():
                        seen.append(f"{kid.name}: missing SKILL.md")
                        continue
                    try:
                        text = f.read_text(encoding="utf-8", errors="replace")
                        meta, body = _sk._parse_frontmatter(text)
                        ok, reason = _sk.validate_skill(
                            meta.get("name", ""), meta.get("description", ""),
                            kid.name, body,
                        )
                        seen.append(f"{kid.name}: {'OK' if ok else 'INVALID — ' + reason}")
                    except OSError as e:
                        seen.append(f"{kid.name}: unreadable ({e})")
        skills = _sk.visible_skills(getattr(ctx.engine, "permission", None) if getattr(ctx, "engine", None) else None)
        lines = [f"Loaded {len(skills)} skill(s): " + (", ".join(s.name for s in skills) or "(none)")]
        lines += seen
        ctx.reply("\n".join(lines))
        return
    skills = _sk.visible_skills(getattr(ctx.engine, "permission", None) if getattr(ctx, "engine", None) else None)
    if not skills:
        ctx.reply("No skills installed. Create .opencode/skills/<name>/SKILL.md.")
        return
    ctx.reply("Skills:\n" + "\n".join(f"  {s.name}: {s.description}" for s in skills))


def _init(ctx: CommandContext, args: str) -> None:
    from pathlib import Path

    path = Path(ctx.worktree) / "AGENTS.md"
    if path.exists():
        ctx.reply("AGENTS.md already exists.")
        return
    path.write_text("# Project Instructions\n\nAdd guidance for the agent here.\n", encoding="utf-8")
    ctx.reply(f"Created {path}")


def _review(ctx: CommandContext, args: str) -> None:
    ctx.reply("Run `git diff` manually to review changes.")


def handle_command(registry: CommandRegistry, ctx: CommandContext, line: str) -> bool:
    """Handle a /command line. Returns True if it was a command."""
    if not line.startswith("/"):
        return False
    parts = line.split(maxsplit=1)
    name = parts[0][1:]
    args = parts[1] if len(parts) > 1 else ""
    cmd = registry.get(name)
    if cmd is None:
        ctx.reply(f"Unknown command: /{name}. Try /help")
        return True
    cmd.handler(ctx, args)
    return True


# attach registry to context convenience

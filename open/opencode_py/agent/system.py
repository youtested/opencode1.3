"""System prompt assembly.

Mirrors opencode: base prompt (default.txt) + environment block + AGENTS.md
instructions + user system override. Plan/build differences are injected as
<system-reminder> text parts appended to the user message (in loop.py).
"""

from __future__ import annotations

from ..util.log import swallow
import datetime
import os
from pathlib import Path

from ..config import Config

# _read_prompt_file / _block_template moved to prompt_blocks (they were the
# agent->tools->agent cycle). Re-exported here so existing imports keep working.
from .prompt_blocks import _block_template, _read_prompt_file

_PROMPT_DIR = Path(__file__).resolve().parent
_base_cache: str | None = None

# Full system-prompt cache: build_system_prompt re-read AGENTS.md files +
# agent .md files + memory + skills from disk on EVERY turn (~6-7ms steady,
# 2.1s spikes on agent-md). Keyed by (agent, dir, worktree, model, override,
# low_data) + per-file (mtime_ns, size) stamps, so any edit to an
# instruction file invalidates exactly. Bounded + never raises, mirroring
# the skill _CACHE idiom. clear_prompt_cache() also drops this (tests).
_SYSTEM_CACHE: dict[tuple, str] = {}
_SYSTEM_CACHE_MAX = 16




def _base_text() -> str:
    """Base prompt from base.md (cached)."""
    global _base_cache
    if _base_cache is None:
        _base_cache = _read_prompt_file("base.md")
    return _base_cache


def clear_prompt_cache() -> None:
    """Forget the cached base.md (tests / live reload)."""
    global _base_cache
    _base_cache = None
    try:
        _SYSTEM_CACHE.clear()
    except Exception as _oc_exc:
        swallow(_oc_exc, "agent/system.py:49")
        pass

# NOTE: agent behavior rules used to live here as hardcoded reminder
# constants (PLAN/EXPLORE/BUILD_SWITCH/KEEP_GOING) injected into every
# user message. They now live ONLY in the agents' .md files
# (build/workflow.md, plan/rules.md, explore/rules.md via the Agent.md
# manager) — keeping both sent every rule twice.


def find_instruction_files(directory: Path, worktree: Path, cfg: Config) -> list[Path]:
    """Find AGENTS.md/CLAUDE.md files: global then project (walk up to worktree)."""
    files: list[Path] = []

    global_home = Path(os.path.expanduser("~/.config/opencode/AGENTS.md"))
    if global_home.exists():
        files.append(global_home)

    d = directory.resolve()
    seen_dirs: set[Path] = set()
    while True:
        if d in seen_dirs:
            # symlink cycle (defense-in-depth): never walk a path twice
            break
        seen_dirs.add(d)
        # NOTE: workflow rules live in the build agent's workflow.md
        # (Agent.md manager) — the single source of truth, no duplicate file.
        for name in ("AGENTS.md", "CLAUDE.md"):
            p = d / name
            if p.exists():
                files.append(p)
                break
        if d == worktree:
            break
        if d.parent == d:
            break
        d = d.parent

    for pattern in cfg.instructions:
        try:
            matches = worktree.glob(pattern)
            for m in matches:
                if m.is_file():
                    files.append(m)
        except ValueError as _oc_exc:
            swallow(_oc_exc, "opencode_py/agent/system.py:95")
            pass

    # dedupe, preserve order
    seen: set[Path] = set()
    out = []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


def build_environment(directory: Path, worktree: Path, provider_id: str, model_id: str) -> str:
    """One-line env facts. Format lives in environment.md (template docs);
    the builder fills it — identical output with or without the file."""
    is_git = (worktree / ".git").exists()
    platform = os.uname().sysname if hasattr(os, "uname") else "?"
    return (
        f"Model: {provider_id}/{model_id} | dir: {directory} | root: {worktree} | "
        f"git: {'yes' if is_git else 'no'} | {platform} | "
        f"{datetime.date.today().isoformat()}"
    )


_MEMORY_LABEL = "memory"
_RECENT_LABEL = "recent session"


def _raw_prompt_parts(
    *,
    directory: Path,
    worktree: Path,
    provider_id: str,
    model_id: str,
    cfg: Config,
    agent: str = "build",
) -> list[tuple[str, str]]:
    """The disk-derived blocks, carrying an EMPTY `memory` slot. This is the
    cached half: every block here changes only when a file on disk changes,
    whereas memory is picked per turn against the user's message and so must
    not be baked in."""
    blocks: list[tuple[str, str]] = []
    try:
        base = _base_text()
        if getattr(cfg, "low_data", False):
            base = _slim_prompt(base)
    except Exception as e:
        base = f"[base.md unreadable: {e}]"
    blocks.append(("base.md", base))
    try:
        blocks.append(("environment", build_environment(directory, worktree, provider_id, model_id)))
    except Exception as e:
        blocks.append(("environment", f"[environment unreadable: {e}]"))
    try:
        found = find_instruction_files(directory, worktree, cfg)
    except Exception:
        found = []
    for path in found:
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as e:
            content = f"[unreadable: {e}]"
        blocks.append((f"Instructions from: {path}", content))
    # per-agent .md files (Agent.md manager): stored under
    # agents.<name>.md, injected only for the active agent.
    try:
        from ..permission import agent_combined_md as _agent_md

        md_text = _agent_md(cfg, agent)
        if md_text.strip():
            blocks.append((f"Agent instructions ({agent}.md)", md_text))
    except Exception as _oc_exc:
        swallow(_oc_exc, "agent/system.py:165")
        pass
    # Placeholders only — filled per turn by labeled_prompt_parts().
    blocks.append((_MEMORY_LABEL, ""))
    blocks.append((_RECENT_LABEL, ""))
    try:
        skills = _load_skills(cfg, agent)
    except Exception:
        skills = ""
    if skills:
        blocks.append(("skills", skills))
    try:
        override = cfg.system_prompt or ""
    except Exception:
        override = ""
    if override:
        blocks.append(("system_prompt (config override)", override))
    return [
        (label, text) for label, text in blocks
        if text or label in (_MEMORY_LABEL, _RECENT_LABEL)
    ]


def _cached_raw_parts(
    *,
    directory: Path,
    worktree: Path,
    provider_id: str,
    model_id: str,
    cfg: Config,
    agent: str,
) -> list[tuple[str, str]] | None:
    """Cached disk-derived blocks, or None when the cache cannot be used."""
    try:
        key = _system_cache_key(
            directory=directory, worktree=worktree, provider_id=provider_id,
            model_id=model_id, cfg=cfg, agent=agent,
        )
    except Exception:
        key = None
    if key is None:
        return None
    try:
        hit = _SYSTEM_CACHE.get(key)
    except Exception:
        hit = None
    if isinstance(hit, list):
        return hit
    parts = _raw_prompt_parts(
        directory=directory, worktree=worktree, provider_id=provider_id,
        model_id=model_id, cfg=cfg, agent=agent,
    )
    try:
        if len(_SYSTEM_CACHE) >= _SYSTEM_CACHE_MAX:
            _SYSTEM_CACHE.pop(next(iter(_SYSTEM_CACHE)), None)
        _SYSTEM_CACHE[key] = parts
    except Exception as _oc_exc:
        swallow(_oc_exc, "agent/system.py:221")
        pass
    return parts


def labeled_prompt_parts(
    *,
    directory: Path,
    worktree: Path,
    provider_id: str,
    model_id: str,
    cfg: Config,
    agent: str = "build",
    query: str = "",
    recent: str = "",
) -> list[tuple[str, str]]:
    """Every system-prompt block as (label, text) — the SINGLE builder the
    sender AND the preview both use, so the preview misses not one letter.
    Labels name the source file exactly as injected below.

    `query` is the current user message: memory notes are scored against it so
    a turn carries the notes it actually needs instead of the newest N.
    `recent` is the previous session's distilled lines, shown on the first
    turn of a session (see memory_capture.peek_pending).
    """
    blocks = _cached_raw_parts(
        directory=directory, worktree=worktree, provider_id=provider_id,
        model_id=model_id, cfg=cfg, agent=agent,
    )
    if blocks is None:
        blocks = _raw_prompt_parts(
            directory=directory, worktree=worktree, provider_id=provider_id,
            model_id=model_id, cfg=cfg, agent=agent,
        )
    try:
        memory = _load_memory(str(worktree), query=query)
    except Exception:
        memory = ""
    if memory:
        try:
            memory = _memory_block(memory)
        except Exception as _oc_exc:
            swallow(_oc_exc, "agent/system.py:262")
            pass
    recent_text = ""
    if recent and recent.strip():
        try:
            recent_text = _block_template("recent.md", "{content}", recent)
        except Exception:
            recent_text = f"From your last session:\n{recent}"
    out: list[tuple[str, str]] = []
    for label, text in blocks:
        if label == _MEMORY_LABEL:
            if memory:
                out.append((label, memory))
        elif label == _RECENT_LABEL:
            if recent_text:
                out.append((label, recent_text))
        else:
            out.append((label, text))
    return [(label, text) for label, text in out if text]


def _stat_stamp(path: Path) -> tuple[int, int]:
    """(mtime_ns, size) for a prompt source file, (0, -1) when unreadable."""
    try:
        st = path.stat()
        return (int(st.st_mtime_ns), int(st.st_size))
    except OSError:
        return (0, -1)
    except Exception:
        return (0, -1)


def _system_cache_key(
    *,
    directory: Path,
    worktree: Path,
    provider_id: str,
    model_id: str,
    cfg: Config,
    agent: str = "build",
) -> tuple | None:
    """Cache key for the full system prompt, or None to skip caching."""
    try:
        found = find_instruction_files(directory, worktree, cfg)
        stamps = tuple((str(p), _stat_stamp(p)) for p in found)
        try:
            from ..permission import agent_combined_md_stamp as _md_stamp

            md_stamp = _md_stamp(cfg, agent)
        except Exception:
            md_stamp = ()
        try:
            from ..globals import Path as GPath

            mem_stamp = _stat_stamp(GPath.data / "memory.json")
        except Exception:
            mem_stamp = (0, -1)
        try:
            override = str(cfg.system_prompt or "")
        except Exception:
            override = ""
        try:
            low = bool(getattr(cfg, "low_data", False))
        except Exception:
            low = False
        try:
            instr = tuple(str(x) for x in (getattr(cfg, "instructions", None) or []))
        except Exception:
            instr = ()
        return (
            str(agent), str(directory), str(worktree),
            str(provider_id), str(model_id),
            override, low, instr, stamps, md_stamp, mem_stamp,
        )
    except Exception:
        return None


def build_system_prompt(
    *,
    directory: Path,
    worktree: Path,
    provider_id: str,
    model_id: str,
    cfg: Config,
    agent: str = "build",
    query: str = "",
    recent: str = "",
) -> str:
    blocks = labeled_prompt_parts(
        directory=directory, worktree=worktree, provider_id=provider_id,
        model_id=model_id, cfg=cfg, agent=agent, query=query, recent=recent,
    )
    return "\n\n".join(t for _label, t in blocks)




def _memory_block(notes: str) -> str:
    """Memory notes wrapped per memory.md (no hardcoded prefix here)."""
    return _block_template("memory.md", "{content}", notes)


def _load_memory(worktree: str, query: str = "", limit: int = 0) -> str:
    """Return the `remember` notes worth injecting this turn as a bulleted
    block, or '' when nothing is saved. Notes are scored against `query` so an
    old-but-relevant note still surfaces instead of being crowded out by
    recent chatter. Safely tolerant of a corrupt/absent memory file."""
    try:
        from ..tools.remember import INJECT_LIMIT, select_notes

        notes = select_notes(
            worktree, query=query, limit=limit or INJECT_LIMIT,
        )
    except Exception:
        return ""
    lines = [
        f"- {str(e.get('text', '')).replace(chr(10), ' ')}"
        for e in notes
        if e.get("text")
    ]
    return "\n".join(lines)


def _load_skills(cfg: Config, agent: str = "build") -> str:
    try:
        from ..permission import PermissionEngine, merge_permissions
        from ..tools.skill import skills_block

        raw = getattr(cfg, "raw", None) or {}
        tools_cfg = raw.get("tools") if isinstance(raw, dict) else None
        if isinstance(tools_cfg, dict) and tools_cfg.get("skill") is False:
            return ""
        agents_cfg = getattr(cfg, "agents", None) or {}
        if isinstance(agents_cfg, dict):
            spec = agents_cfg.get(agent) or {}
            if isinstance(spec, dict):
                tools = spec.get("tools")
                if isinstance(tools, dict) and tools.get("skill") is False:
                    return ""
        perm = getattr(cfg, "permission", None) or {}
        mode = str(getattr(cfg, "permission_mode", "") or "auto").lower()
        if mode not in ("auto", "ask", "deny", "fully_auto"):
            mode = "auto"
        engine = PermissionEngine.from_config(merge_permissions(perm, agent), mode=mode)
        return skills_block(engine)
    except Exception:
        return ""


def _slim_prompt(prompt: str) -> str:
    """Low-data variant: the base is already short; only strip any
    <example> blocks if present. Everything left is directives."""
    import re as _re

    slim = _re.sub(r"<example>.*?</example>", "", prompt, flags=_re.DOTALL)
    slim = _re.sub(r"\n{3,}", "\n\n", slim).strip()
    return slim or prompt


def agent_reminder(
    agent: str,
    was_plan: bool,
    cfg: Config | None = None,
    *,
    previous: str | None = None,
) -> str | None:
    """Tell the model the ACTIVE AGENT CHANGED, or None when nothing switched.

    Why this cannot live only in the agents' .md files: those files ARE re-read
    and re-injected every turn (build/workflow.md, plan/rules.md,
    explore/rules.md), and the system prompt really does change on a switch --
    in both directions, verified. What they cannot express is that the rules the
    model was obeying one message ago have just been replaced. Its own recent
    turns still read "I am in PLAN mode, I will not make changes until you
    approve", and with nothing marking that as void it goes on honouring them.
    That showed up as the model insisting it was still in plan mode after
    switching back to build, and refusing to act until argued with.

    It worked in one direction only because plan/rules.md is written as an
    absolute prohibition ("CRITICAL", "overrides ALL other instructions", "ZERO
    exceptions"), so it outshouts everything on the way IN, while
    build/workflow.md is procedural advice ("keep going") that never states a
    mode -- so on the way OUT nothing actively lifted the old constraint.

    `previous` is the agent that ran last turn. `was_plan` is kept for old
    callers and used as the fallback when `previous` is absent. The caller
    updates `_prev_agent` at END of turn, so this fires exactly once per switch
    and never on a normal turn.
    """
    prev = str(previous or "").strip().lower()
    if not prev:
        prev = "plan" if was_plan else ""
    now = str(agent or "").strip().lower()
    if not prev or not now or prev == now:
        return None
    jobs = {
        "build": (
            "build mode, with full tools. The read-only constraint is GONE. If "
            "you were planning, analysing or waiting for approval a moment ago, "
            "stop proposing and carry the work out now: make the changes, "
            "verify them, and report what you did. Do not ask again for "
            "permission you have already been given."
        ),
        "plan": (
            "read-only plan mode. STOP editing and STOP running anything that "
            "changes the system. Read, search and think, then present a plan "
            "and wait for approval."
        ),
        "explore": (
            "read-only exploration mode. Search and read the code and report "
            "what you find. Do NOT edit files or change anything, and do not "
            "present an implementation plan unless asked."
        ),
    }
    job = jobs.get(now)
    if job is None:
        return (
            f"[AGENT SWITCHED: you were '{prev}', you are now '{now}'. The "
            f"rules for '{now}' replace everything you followed as '{prev}'. "
            f"Follow '{now}' from this message on.]"
        )
    return (
        f"[AGENT SWITCHED: you were '{prev}', you are now '{now}'. You are in "
        f"{job} Any earlier statement about your mode is void. Follow '{now}' "
        f"from this message on.]"
    )

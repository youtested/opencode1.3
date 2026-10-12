from __future__ import annotations

from ..util.log import swallow
import json
from pathlib import Path
from typing import Any

from ..plugins import get_dispatcher as _plugins
from ..tools import apply_patch as apply_patch_mod
from ..tools import glob as glob_mod
from .. import snapshots as _snapshots

from .loop_config import MAX_UNDO, _missing_directories


def _session_title(loop: Any) -> str:
    sess = getattr(loop, "session", None)
    return str(getattr(sess, "title", "") or "")


def _ensure_session_snapshot(loop: Any) -> None:
    """Take this session's whole-project capture ONCE, before its first edit.

    Must run BEFORE the mutating tool: a capture taken afterwards would record
    the file's already-edited content, and restoring it would be a no-op — worse
    than having no snapshot, because it looks like protection.
    """
    sid = str(getattr(loop, "session_id", "") or "").strip()
    if not sid:
        return
    try:
        _snapshots.ensure_full(sid, Path(loop.worktree), _session_title(loop))
    except Exception as e:
        swallow(e, "opencode_py/agent/loop_tools.py:_ensure_session_snapshot")


def _record_session_change(loop: Any, path: Path | None, before: bytes | None,
                           dirs: list | None = None) -> None:
    """Persist one file's pre-edit bytes into the session's change log."""
    sid = str(getattr(loop, "session_id", "") or "").strip()
    if not sid or path is None:
        return
    try:
        _snapshots.record_change(sid, Path(loop.worktree), str(path), before,
                                 [str(d) for d in (dirs or [])])
    except Exception as e:
        swallow(e, "opencode_py/agent/loop_tools.py:_record_session_change")

class LoopToolsMixin:
    def _emit_tool(self, kind: str, tool: str, **kwargs: Any) -> None:
        self._emit(kind, tool=tool, **kwargs)

    def check_permission(self, tool: str, input_value: str, display: str, call_id: str = "", arguments: dict | None = None, permission: str | None = None, action: str | None = None) -> bool:
        permission_name = permission or tool
        if tool in ("write", "edit", "apply_patch") and permission is None:
            permission_name = "edit"
        if action is None:
            action = self.permission.evaluate(permission_name, input_value)
        if action == "allow":
            return True
        kwargs: dict[str, Any] = {"reason": "denied by permission", "call_id": call_id}
        if action == "deny":
            if arguments is not None:
                kwargs["input"] = arguments
            self._emit_tool("tool_denied", tool, **kwargs)
            return False
        # ask
        # "Always" scope: the specific permission+input being approved, NOT a
        # universal "*" — one approval granted to `bash npm install` must not
        # silently auto-approve every future ask for any tool/command.
        always_patterns: list[str] = [f"{permission_name} {input_value}".strip()]
        # Cross-thread safe: when a step runs several tools in parallel this
        # lock guarantees only one permission/question modal is visible at a
        # time (the TUI shows exactly one dialog at once).
        with self._ask_lock:
            allowed = self.permission.ask(display, always_patterns)
        if not allowed:
            kwargs = {"reason": "rejected by user", "call_id": call_id}
            if arguments is not None:
                kwargs["input"] = arguments
            self._emit_tool("tool_denied", tool, **kwargs)
        return allowed

    def run_tool(self, name: str, arguments: dict, call_id: str = "") -> dict[str, Any]:
        if name == "skill" and not self._skill_enabled():
            return {"output": "Skill tool is disabled for this agent.", "error": True}
        tool = self.registry.get(name)
        if tool is None:
            # Wire names are sanitized (mcp:n -> mcp_n); an old session or a
            # custom plugin may reply with the sanitized form while the
            # registry holds the raw name (or vice versa). Resolve by
            # sanitized comparison before giving up.
            try:
                from ..tools.registry import _sanitize_wire_name as _san

                want = _san(name)
                for cand in self.registry.list():
                    if _san(cand.name) == want:
                        tool = cand
                        break
            except Exception:
                tool = None
            if tool is None:
                return {"output": f"Unknown tool: {name}", "error": True}
        # Relative paths are resolved against the session directory (git
        # worktree root), NOT Path.cwd(): launching opencode from a
        # subdirectory would otherwise make read/write/edit/bash act on the
        # wrong files while the undo snapshots use self.directory. Normalize
        # here so every component agrees. `resolved_path` keeps the canonical
        # target used both for permission matching and worktree confinement.
        resolved_path = None
        for key in ("filePath", "path", "workdir"):
            if key in arguments and isinstance(arguments[key], str) and arguments[key]:
                p = Path(arguments[key])
                if not p.is_absolute():
                    p = self.directory / p
                p = p.resolve()
                arguments[key] = str(p)
                resolved_path = str(p)
                break

        # Permission match value: path-based tools match their rules (e.g.
        # "*.env" -> ask) against the RESOLVED PATH, not the JSON blob of all
        # arguments (which ends in `}` and never matches a path pattern). task
        # matches on the sub-agent type, mirroring opencode. bash matches on the
        # COMMAND STRING — rules like {"bash": {"npm install *": "deny"}} must
        # actually fire, and the JSON dump of arguments would never match any
        # command pattern (it's "{...}", not "npm install ...").
        input_value = json.dumps(arguments, sort_keys=True)
        if name == "task":
            # opencode's task input id is the sub-agent type, so a plan
            # agent's `task: {grant: deny}` rule actually matches.
            input_value = str(arguments.get("subagent_type", "build"))
        elif name == "bash":
            input_value = str(arguments.get("command", ""))
        elif name in ("browser", "phone"):
            # Permission patterns like {"browser": {"snapshot *": ...}} must
            # match the ACTION NAME, not the JSON blob '{"action": ...}'.
            # The trailing ' *' matcher also covers "browser" bare.
            action_name = str(arguments.get("action", "") or "").strip().lower()
            input_value = f"{action_name} {json.dumps(arguments, sort_keys=True)}".strip()
        elif name in ("skill",):
            # Permission patterns like {"skill": {"internal-*": "deny"}} must
            # match the SKILL NAME, not the JSON blob '{"name": ...}' (which
            # never matches a name pattern, so denies silently never fired).
            input_value = str(arguments.get("name", "") or "")
        elif resolved_path is not None and name in ("read", "write", "edit", "apply_patch", "grep", "glob"):
            input_value = resolved_path

        permission = name
        if name in ("write", "edit", "apply_patch"):
            permission = "edit"
        # Worktree confinement: a path-based tool targeting a file OUTSIDE the
        # canonical worktree root goes through the `external_directory`
        # permission (default: ask). The rule existed in the config defaults
        # but was never evaluated, so every absolute path was silently allowed.
        # NOTE: external_directory is an ADDITIONAL gate, NOT a replacement for
        # the tool's own permission — otherwise a plan agent's edit/write "deny"
        # could be bypassed by pointing the tool at an absolute path outside the
        # worktree. Combine both with the strictest action (deny > ask > allow)
        # and gate exactly once, so there is a single permission dialog.
        is_external = (
            resolved_path is not None
            and (
                name in ("read", "write", "edit", "apply_patch", "grep", "glob", "bash", "background_task")
                or name.startswith("mcp__")
            )
            and not Path(resolved_path).is_relative_to(self.worktree)
        )
        if is_external:
            combined = [
                self.permission.evaluate(permission, input_value),
                self.permission.evaluate("external_directory", resolved_path),
            ]
            if getattr(self.permission, "mode", "auto") == "fully_auto":
                action: str | None = "deny" if "deny" in combined else "allow"
            else:
                action = "deny" if "deny" in combined else ("ask" if "ask" in combined else "allow")
        else:
            action = None

        display = f"{name} {input_value[:120]}"
        if is_external:
            # Jail visibility: the tool result names the outside path so the
            # user always sees when the model reached outside the project.
            # Enforcement stays in the permission gate above (ask by default,
            # deny with an explicit rule) — see default_permissions.
            try:
                self._emit(
                    "notice",
                    text=f"{name} outside worktree: {resolved_path}",
                )
            except Exception as _oc_exc:
                swallow(_oc_exc, "agent/loop_tools.py:156")
                pass

        if not self.check_permission(name, input_value, display, call_id=call_id, arguments=arguments, permission=permission, action=action):
            return {
                "output": f"Permission denied for {name}. Tell the user what to do differently.",
                "error": True,
                "denied": True,
            }

        mutates = name in ("edit", "write", "apply_patch")
        file_path = arguments.get("filePath")
        plugins = _plugins()
        if plugins is not None and plugins.has("tool.before"):
            try:
                plugins.emit("tool.before", {"tool": name, "input": arguments, "sessionId": self.session_id})
            except Exception as _oc_exc:
                swallow(_oc_exc, "agent/loop_tools.py:172")
                pass
        snapshot: bytes | None = None
        snapshot_path: Path | None = None
        created_dirs: list[Path] = []
        if mutates and file_path:
            p = Path(file_path)
            if not p.is_absolute():
                p = self.directory / p
            snapshot_path = p
            snapshot = p.read_bytes() if p.exists() else None
            # record which parent directories would be newly created by the
            # write's mkdir(parents=True) so undo can clean them back up.
            if snapshot is None:
                created_dirs = _missing_directories(p.parent)

        if mutates:
            # Whole-project capture for this session, before anything is
            # written. Once per session, so the cost is paid on the first
            # edit only.
            _ensure_session_snapshot(self)

        # The glob tool resolves a missing `path` against this thread's
        # session worktree (each worker thread gets its own), so parallel
        # globs never fall back to the process CWD.
        glob_mod.set_worktree(self.worktree)
        apply_patch_mod.set_worktree(self.worktree)
        # Same thread-local idea, for the snapshot store: the checkpoint tool
        # needs to know which session it is acting for, and the only reliable
        # answer is the loop that is running right now on this thread.
        _snapshots.set_session(
            str(getattr(self, "session_id", "") or ""),
            str(self.worktree),
            _session_title(self),
        )

        self._emit_tool("tool_start", name, input=arguments, status="running", call_id=call_id)
        try:
            _pctx = getattr(self.registry, "_progress_ctx", None)
            if _pctx is not None:
                def _emit_progress(done: int, total: int, _n=name, _c=call_id) -> None:
                    try:
                        self._emit_tool("tool_progress", _n, call_id=_c, done=int(done), total=int(total))
                    except Exception as _oc_exc:
                        swallow(_oc_exc, "agent/loop_tools.py:200")
                        pass
                _pctx.call_id = call_id
                _pctx.emitter = _emit_progress
        except Exception as _oc_exc:
            swallow(_oc_exc, "agent/loop_tools.py:204")
            pass
        try:
            if name == "task":
                # let spawn_task tag its subagent_start event with this call id
                self._spawn_ctx.call_id = call_id
                action = str(arguments.get("action") or "start").strip().lower()
                if action in ("status", "read", "stop", "list"):
                    if action == "status":
                        result = self.task_status(str(arguments.get("session_id") or ""))
                    elif action == "read":
                        result = self.task_read(str(arguments.get("session_id") or ""))
                    elif action == "stop":
                        result = self.task_stop(str(arguments.get("session_id") or ""))
                    else:
                        result = self.task_status("")
                else:
                    result = tool.run(arguments)
            else:
                result = tool.run(arguments)
        except Exception as e:
            result = {"output": f"{name} failed: {e}", "error": True}
        if mutates and snapshot_path is not None and not result.get("error"): 
            # Only a call that SUCCEEDED gets an undo entry. A failed edit used
            # to push one anyway, so `/undo` after a failure silently reverted
            # the previous, unrelated, successful edit — and the 20-slot stack
            # burned a slot per failure.
            self._undo_stack.append(
                {
                    "path": str(snapshot_path),
                    "original": snapshot,
                    "dirs": [str(d) for d in created_dirs],
                }
            )
            if len(self._undo_stack) > MAX_UNDO:
                self._undo_stack.pop(0)
            # Durable copy on disk: the in-memory stack above dies with the
            # process, this one does not. `apply_patch` needs no equivalent —
            # its own journal already keeps before-images, and this session's
            # full capture covers it anyway.
            _record_session_change(self, snapshot_path, snapshot, created_dirs)
        try:
            _pctx2 = getattr(self.registry, "_progress_ctx", None)
            if _pctx2 is not None:
                _pctx2.call_id = ""
                _pctx2.emitter = None
        except Exception as _oc_exc:
            swallow(_oc_exc, "agent/loop_tools.py:241")
            pass
        if plugins is not None and plugins.has("tool.after"):
            try:
                plugins.emit(
                    "tool.after",
                    {
                        "tool": name,
                        "input": arguments,
                        "output": result.get("output", ""),
                        "error": bool(result.get("error")),
                        "sessionId": self.session_id,
                    },
                )
            except Exception as _oc_exc:
                swallow(_oc_exc, "agent/loop_tools.py:255")
                pass
        if plugins is not None and mutates and file_path:
            try:
                p = Path(file_path)
                if not p.is_absolute():
                    p = self.directory / p
                event = "file.created" if not p.exists() else "file.changed"
                plugins.emit(event, {"filePath": str(p), "tool": name, "sessionId": self.session_id})
            except Exception as _oc_exc:
                swallow(_oc_exc, "agent/loop_tools.py:264")
                pass
        self._emit_tool(
            "tool_complete",
            name,
            input=arguments,
            status="error" if result.get("error") else "completed",
            output=result.get("output", ""),
            metadata=result.get("metadata", {}),
            call_id=call_id,
        )
        return result

    def _tool_pool_locked(self):
        """Persistent worker pool shared by all steps of this agent."""
        from concurrent.futures import ThreadPoolExecutor

        with self._tool_pool_lock:
            pool = self._tool_pool
            if pool is None:
                pool = ThreadPoolExecutor(
                    max_workers=self._PARALLEL_TOOL_WORKERS,
                    thread_name_prefix="opencode-tools",
                )
                self._tool_pool = pool
            return pool

    def _shutdown_tool_pool(self) -> None:
        with self._tool_pool_lock:
            pool, self._tool_pool = self._tool_pool, None
        if pool is not None:
            try:
                pool.shutdown(wait=False, cancel_futures=True)
            except Exception as _oc_exc:
                swallow(_oc_exc, "agent/loop_tools.py:297")
                pass

    def _run_tools_parallel(self, prepared):
        """Run every tool call of a step concurrently and return their results
        in the original call order.

        `prepared` is a list of ``(call, name, arguments)`` triples. Reads and
        other independent calls run concurrently on worker threads — `task` calls
        therefore spawn and run their sub-agents at the same time (official
        opencode's parallel agent fan-out). MUTATING tools (edit/write/apply_patch)
        are kept strictly sequential, in call order, so two writes to the same
        file can't interleave their read-modify-write and their undo snapshots
        can't race. Permission prompts stay serialized via ``_ask_lock`` so the
        TUI never shows stacked dialogs.
        """
        from concurrent.futures import TimeoutError as FuturesTimeout
        from concurrent.futures import wait

        def execute(index: int, call: dict, name: str, arguments: dict):
            if self.interrupt():
                return index, call, name, {"output": "(interrupted)", "error": True, "stopped": True}
            try:
                tool_result = self.run_tool(name, arguments, call_id=call.get("id", ""))
            except Exception as e:  # run_tool already guards, but be safe
                tool_result = {"output": f"{name} failed: {e}", "error": True}
            return index, call, name, tool_result

        ordered: list[tuple[dict, str, dict] | None] = [None] * len(prepared)
        pool = self._tool_pool_locked()
        if pool is None:
            # pool exhausted at shutdown: run inline, sequentially
            for i, (call, name, arguments) in enumerate(prepared):
                _, _, _, tool_result = execute(i, call, name, arguments)
                ordered[i] = (call, name, tool_result)
            return [o for o in ordered if o is not None]
        try:
            futures = {}
            for i, (call, name, arguments) in enumerate(prepared):
                if name not in self._MUTATING_TOOLS:
                    futures[pool.submit(execute, i, call, name, arguments)] = i
            # Mutating calls execute on this thread one at a time, in original
            # order (strictly serialized; never concurrent with one another).
            for i, (call, name, arguments) in enumerate(prepared):
                if name in self._MUTATING_TOOLS:
                    _, _, _, tool_result = execute(i, call, name, arguments)
                    ordered[i] = (call, name, tool_result)
            pending = set(futures)
            while pending:
                if self.interrupt():
                    for f in pending:
                        f.cancel()
                    # Mark remaining as interrupted (like original behavior)
                    for f, idx in futures.items():
                        if ordered[idx] is None:
                            # Don't call result() on cancelled futures - construct directly
                            call = prepared[idx][0]  # original call dict
                            name = prepared[idx][1]  # original tool name
                            ordered[idx] = (call, name, {"output": "(interrupted)", "error": True, "stopped": True})
                    break
                done, pending = wait(pending, timeout=0.5)
                for fut in done:
                    index, call, name, tool_result = fut.result()
                    ordered[index] = (call, name, tool_result)
            for fut, idx in futures.items():
                if ordered[idx] is None:
                    # Interruptible join: a bare fut.result() here blocked
                    # forever on a hung tool, making ESC look ignored.
                    while True:
                        if self.interrupt():
                            call = prepared[idx][0]
                            name = prepared[idx][1]
                            ordered[idx] = (call, name, {"output": "(interrupted)", "error": True, "stopped": True})
                            break
                        try:
                            index, call, name, tool_result = fut.result(timeout=0.5)
                        except FuturesTimeout as _oc_exc:
                            swallow(_oc_exc, "opencode_py/agent/loop_tools.py:382")
                            continue
                        ordered[index] = (call, name, tool_result)
                        break
        except RuntimeError:
            # pool shut down mid-step (engine closed): finish inline
            for i, (call, name, arguments) in enumerate(prepared):
                if ordered[i] is None:
                    _, _, _, tool_result = execute(i, call, name, arguments)
                    ordered[i] = (call, name, tool_result)
        return [o for o in ordered if o is not None]

    def _skill_enabled(self) -> bool:
        """Honor tools.skill=false and agents.<agent>.tools.skill=false."""
        try:
            raw = getattr(self.cfg, "raw", None) or {}
            tools_cfg = raw.get("tools") if isinstance(raw, dict) else None
            if isinstance(tools_cfg, dict) and tools_cfg.get("skill") is False:
                return False
            agents_cfg = getattr(self.cfg, "agents", None) or {}
            if isinstance(agents_cfg, dict):
                spec = agents_cfg.get(self.agent) or {}
                if isinstance(spec, dict):
                    tools = spec.get("tools")
                    if isinstance(tools, dict) and tools.get("skill") is False:
                        return False
        except Exception as _oc_exc:
            swallow(_oc_exc, "agent/loop_tools.py:399")
            pass
        return True

    def _active_tool_schemas(self) -> list[dict]:
        schemas = self.registry.schemas()
        from ..permission import agent_readonly as _is_ro

        try:
            readonly = bool(_is_ro(self.cfg, self.agent))
        except Exception:
            readonly = self.agent in ("plan", "explore")
        if readonly:
            schemas = [
                s
                for s in schemas
                if s["function"]["name"] not in ("bash", "write", "edit", "apply_patch")
            ]
        # specialist scope: an agent whose spec names an explicit `scope`
        # list sees exactly those tools (intersected with what exists).
        # This keeps slim agents slim — web-agent never sees edit schemas.
        # Agents without a scope list see everything (minus readonly
        # stripping above). The permission editor manages allow/ask/deny
        # within whatever is visible.
        try:
            from ..permission import agent_spec as _spec

            spec = _spec(self.cfg, self.agent) or {}
            scope = spec.get("scope")
            if isinstance(scope, list) and scope:
                want = {str(t) for t in scope}
                schemas = [s for s in schemas if s["function"]["name"] in want]
        except Exception as _oc_exc:
            swallow(_oc_exc, "agent/loop_tools.py:431")
            pass
        if not self._skill_enabled():
            schemas = [s for s in schemas if s["function"]["name"] != "skill"]
        for s in schemas:
            if s["function"]["name"] == "skill":
                try:
                    from ..tools.skill import visible_skills

                    skills = visible_skills(self.permission)
                    if skills:
                        if getattr(self.cfg, "low_data", False):
                            # Save data: the system prompt already carries the
                            # full <available_skills> block — don't duplicate
                            # it (~N KB) inside the tool description too.
                            s["function"]["description"] = (
                                "Load a skill: reusable instructions from SKILL.md files. "
                                "Names are listed in the system prompt <available_skills>; "
                                "call skill({name}) to load the full content when the task matches."
                            )
                        else:
                            from ..tools.skill import SKILL_LIST_LIMIT

                            entries = "".join(
                                f"<skill><name>{sk.name}</name><description>{sk.description}</description></skill>"
                                for sk in skills[:SKILL_LIST_LIMIT]
                            )
                            s["function"]["description"] = (
                                "Load a skill: reusable instructions from SKILL.md files. "
                                f"Available skills: <available_skills>{entries}</available_skills> "
                                "Call skill({name}) to load the full content when the task matches, then follow it. "
                                'Call with no name to list them. Example: skill({"name": "git-release"}).'
                            )
                except Exception as _oc_exc:
                    swallow(_oc_exc, "agent/loop_tools.py:464")
                    pass
                break
        return schemas

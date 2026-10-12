"""Main agentic loop.

Flow per turn:
  1. build messages (system + trimmed history + latest user turn + agent reminder)
  2. stream model -> emit text/reasoning/tool_call events to on_event
  3. if tool calls: permission check -> run tool -> append tool result; loop again
  4. cap iterations (safety) and honor interrupt

build agent: full tools. plan agent: edit/write/bash denied by permissions.
"""

from __future__ import annotations


from ..util.log import swallow
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from ..config import Config
from ..globals import resolve_worktree
from ..permission import PermissionEngine, merge_permissions
from ..providers import build_rotation
from ..providers.base import usage_total_dict
from ..session import new_session
from ..tools.registry import Registry
from . import compaction as compact_mod
from . import messages as msg_mod
from . import parse as parse_mod
from . import system as system_mod
from . import trim as trim_mod







# nudge/step limits: read here at call time so tests can tune them via
# monkeypatch.setattr(opencode_py.agent.loop, ...)
MAX_STEPS = 1000
AUTO_CONTINUE_MSG = "keep going dont stop keep going never stop"
AUTO_CONTINUE_MAX_NUDGES = 25
AUTO_CONTINUE_WAIT_S = 2.0

# Asked only when the model spoke but ran NO tool this step — the one case where
# "did it finish?" cannot be read off the transcript. One word back decides:
# DONE stops the drain, CONTINUE keeps nudging. An unclear answer counts as
# CONTINUE (keep working), so silence never ends a job that is really unfinished.
AUTO_CONTINUE_CHECK_MSG = (
    "Status check. Reply with exactly one word and nothing else:\n"
    "DONE = the task is completely finished, no work remains.\n"
    "CONTINUE = there is still work left to do."
)

# Used when the step budget runs out mid-job: one last tools-free turn so the
# user always gets words instead of a silent stop.
AUTO_CONTINUE_SUMMARY_MSG = (
    "You hit the step limit before finishing. Do NOT call any tools. "
    "Briefly state what you completed, what is still outstanding, and what "
    "the user should do next."
)

# Re-exported for compatibility: these live in loop_config / the loop_*
# mixins now, but `from opencode_py.agent.loop import ...` must keep working.
from .loop_config import (
    MAX_UNDO,
    TurnResult,
    _BgAbandoned,
    _memory_capture_enabled,
    _missing_directories,
)

from .loop_tasks import LoopTasksMixin
from .loop_turn import LoopTurnMixin
from .loop_prompts import LoopPromptsMixin
from .loop_tools import LoopToolsMixin
from .loop_history import LoopHistoryMixin


def _text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in content
        )
    return str(content)


# Re-exported for compatibility: these moved to loop_config / the
# loop_* mixins, but `from opencode_py.agent.loop import ...` must keep
# working (tests and tools import them from here).
__all__ = [
    "AgentLoop",
    "TurnResult",
    "MAX_STEPS",
    "MAX_UNDO",
    "AUTO_CONTINUE_MSG",
    "AUTO_CONTINUE_MAX_NUDGES",
    "AUTO_CONTINUE_WAIT_S",
    "_missing_directories",
    "_memory_capture_enabled",
    "_BgAbandoned",
]


class AgentLoop(LoopHistoryMixin, LoopToolsMixin, LoopPromptsMixin, LoopTurnMixin, LoopTasksMixin):
    def __init__(
        self,
        *,
        cfg: Config,
        registry: Registry,
        directory: Path,
        provider=None,
        auth=None,
        permission_engine: PermissionEngine | None = None,
        question_service: Any = None,
        on_event: Callable[[dict], None] | None = None,
        agent: str = "build",
        provider_id: str = "",
        model_id: str = "",
        interrupt: Callable[[], bool] | None = None,
        session_id: str | None = None,
        provider_factory: Callable[[], Any] | None = None,
        ask_lock: threading.Lock | None = None,
        depth: int = 0,
    ):
        self.cfg = cfg
        self.registry = registry
        # nesting level of THIS engine: main agent = 0, its sub-agents = 1, ...
        # bounded by cfg.subagent_depth in spawn_task
        self._depth = max(0, int(depth))
        self.directory = directory
        self.worktree = resolve_worktree(directory)
        self.auth = auth
        self.agent = agent
        self._prev_agent: str = agent
        self._turn_agent: str = agent
        # Fall back to the configured provider/model so the active lane is the
        # source of truth (a hardcoded "opencode" default would resolve the
        # wrong context window / output limit for any other provider).
        self.provider_id = provider_id or cfg.provider
        self.model_id = model_id or cfg.model
        # `interrupt` is a property: assigning it also re-points the registry's
        # `interrupt_check` hook (which bash/webfetch read at call time). The
        # TUI wires the live callback AFTER construction (`_wire_engine`), so a
        # plain attribute would leave the tools frozen on the init default and
        # ESC/Ctrl+C could never abort a running command.
        self.interrupt = interrupt or (lambda: False)
        self._session_id = session_id or uuid.uuid4().hex
        # factory used to build rotations for spawned sub-agents (override in
        # tests); default matches the parent's own rotation construction.
        self.provider_factory = provider_factory or (lambda: build_rotation(cfg, auth, self._session_id))

        self.rotation = provider or build_rotation(cfg, auth, self._session_id)
        # When True the selected model is pinned: rate limits / hard failures
        # surface instead of failing over to another lane (the TUI lock dot).
        self.rotation_locked = bool(getattr(cfg, "rotation_lock", False))

        # mode comes from cfg so the Settings "allow all permissions" toggle
        # is the single source of truth ("ask" -> popups; "auto"/"fully_auto" ->
        # never ask; explicit deny rules apply in ALL modes)
        raw_mode = str(getattr(cfg, "permission_mode", "auto") or "auto").lower()
        perm_mode = raw_mode if raw_mode in ("ask", "deny", "fully_auto") else "auto"
        self.permission = permission_engine or PermissionEngine.from_config(
            merge_permissions(cfg.permission, agent, cfg),
            mode=perm_mode,
        )
        self.on_event = on_event
        self._history: list[dict] = []
        self._call_seq = 0
        self._pending_calls: list[dict] = []
        # FIFO of prompts submitted while a turn was running. Mirroring
        # opencode, these are consumed INSIDE the running turn (a single
        # "Session Drain") at the next provider-turn boundary — the turn keeps
        # working straight into the next prompt instead of ending and starting
        # a fresh turn.
        self._prompt_queue: list[str] = []
        # Lazy rotation rebuild: build_rotation() can hit the network (model
        # catalogs). UI-side callers mark dirty instead of rebuilding on the
        # UI thread; run_turn rebuilds here on the engine thread.
        self._rotation_dirty = False
        self._prompt_lock = threading.Lock()
        # Serializes permission/question prompts for parallel tool calls WITHIN
        # this agent only: the TUI shows exactly one dialog at a time. Each
        # sub-agent gets its OWN lock (see spawn_task) — siblings keep
        # streaming while one waits on its modal, and the app's dialog queue
        # displays their prompts one by one instead of stacking screens.
        self._ask_lock = ask_lock or threading.Lock()
        # Persistent worker pool reused across steps (creating a pool per step
        # churned threads on every tool-loop iteration). Shut down in close().
        self._tool_pool: Any = None
        self._tool_pool_lock = threading.Lock()
        # Bounds simultaneously running sub-agents (see _MAX_CONCURRENT_SUBAGENTS).
        self._subagent_sem = threading.Semaphore(self._MAX_CONCURRENT_SUBAGENTS)
        # Per-thread id of the tool call currently spawning a sub-agent. With
        # parallel execution each worker thread sets its own call id, so a
        # subagent_start event can be linked back to the exact tool row that
        # launched it (official opencode keys sub-agents by their task call).
        self._spawn_ctx = threading.local()
        self._undo_stack: list[dict] = []
        # Background sub-agents: at most ONE live detached child (RAM +
        # context budget on phones). Keyed by session id while running;
        # finished results stay readable until collected or replaced.
        self._bg_agents: dict[str, dict[str, Any]] = {}
        self._bg_lock = threading.Lock()
        self._usage_total: dict[str, int] = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }
        self._compaction_summary: str = ""
        # Early-summary worker (fix 2): a daemon summary pass started at ~70%
        # of the window, so the freeze-turn never happens. Keyed by history
        # length so a newer turn invalidates a stale worker's result.
        self._early_summary_text: str = ""
        self._early_summary_at: int = -1
        self._early_summary_head: dict = {}
        self._early_summary_lock = threading.Lock()
        self.subagents: dict[str, "AgentLoop"] = {}
        # the task tool looks this up lazily so sub-agents can nest
        self.registry.task_spawner = self.spawn_task
        # Registry.reads `interrupt_check` at call time so ESC/Ctrl+C aborts a
        # running command/fetch. The `interrupt` property keeps it in sync
        # whenever the TUI wires the live callback after construction.
        self.registry.interrupt_check = self.interrupt
        # the question tool asks the user through this service (TUI attaches an
        # ask_callback that surfaces a modal and blocks until answered). Built
        # after permissions so the per-agent "question" rule is already loaded.
        from ..question import QuestionService

        self.question_service = question_service or QuestionService()
        if perm_mode == "fully_auto":
            # fully_auto: the question tool never opens a dialog. Auto-answer
            # with the model's own recommended (first) option so the turn
            # continues toward the goal with zero popups. A `notice` event
            # tells the TUI (which shows a small toast) so the user knows
            # why no question appeared.
            def _fully_auto_ask(questions: list) -> list[list[str]]:
                try:
                    self._emit("notice", text=(
                        "fully_auto: question auto-answered — "
                        "switch permission mode to see questions"
                    ))
                except Exception as _oc_exc:
                    swallow(_oc_exc, "agent/loop.py:228")
                    pass
                out: list[list[str]] = []
                for q in questions:
                    opts = getattr(q, "options", None) or []
                    first = getattr(opts[0], "label", "") if opts else ""
                    out.append([first] if first else [])
                return out
            self.registry.question_asker = _fully_auto_ask
        else:
            self.registry.question_asker = self.question_service.ask
        # Live permission for skill listing/enforcement (skill tool reads
        # registry._skill_permission; without this it always saw None = all).
        try:
            self.registry._skill_permission = self.permission
        except Exception as _oc_exc:
            swallow(_oc_exc, "agent/loop.py:243")
            pass

    @property
    def interrupt(self) -> Callable[[], bool]:
        return self._interrupt

    @interrupt.setter
    def interrupt(self, value: Callable[[], bool]) -> None:
        self._interrupt = value
        registry = getattr(self, "registry", None)
        if registry is not None:
            try:
                registry.interrupt_check = value
            except Exception as _oc_exc:
                swallow(_oc_exc, "agent/loop.py:257")
                pass

    @property
    def session_id(self) -> str:
        return self._session_id

    @session_id.setter
    def session_id(self, value: str) -> None:
        """Bind a session id, keeping the rotation's Zen identity in sync.

        The TUI assigns the persisted session id after construction; the
        rotation's `x-opencode-session` must follow so every provider built
        for the current conversation reports the same id.
        """
        self._session_id = value
        rotation = getattr(self, "rotation", None)
        if rotation is not None and getattr(rotation, "session_id", None) is not None:
            # Wire value must stay Zen-acceptable (free-tier gate rejects
            # raw uuid hex with 403); Rotation normalizes on construction
            # but direct rebinding bypasses it, so normalize here too.
            try:
                from ..providers.zen import zen_session_id as _zen_sid
            except Exception:
                _zen_sid = lambda s: s  # noqa: E731
            rotation.session_id = _zen_sid(value)


    # -- event plumbing --------------------------------------------------
    def _emit(self, kind: str, **kwargs: Any) -> None:
        event = {"kind": kind, "session_id": self._session_id, **kwargs}
        if self.on_event:
            self.on_event(event)

    def _job_is_done(self, messages: list[dict], system_prompt: str) -> bool:
        """Ask the model whether the job is genuinely finished.

        Used ONLY when it spoke without calling a tool — the ambiguous case. When
        tools ran alongside text the loop already trusts that as the final answer
        and never asks.

        Runs with tools=[] so the check can never start new work. The probe is
        never committed to history: the transcript must not grow a stray assistant
        turn or a "DONE" the user never typed. A probe that errors, goes silent,
        or answers nonsense counts as NOT done.
        """
        mark = len(self._history)
        text = ""
        try:
            probe_messages = list(messages) + [
                {"role": "user", "content": AUTO_CONTINUE_CHECK_MSG}
            ]
            probe = TurnResult()
            self._stream(probe_messages, [], probe, system_prompt)
            text = (probe.text or "").strip()
        except Exception as _oc_exc:
            swallow(_oc_exc, "agent/loop.py:_job_is_done")
        finally:
            # _stream grows history live; drop anything the probe appended.
            try:
                del self._history[mark:]
            except Exception as _oc_exc:
                swallow(_oc_exc, "agent/loop.py:_job_is_done trim")
        if not text:
            return False
        # CONTINUE is tested FIRST so "not done, please continue" never reads as
        # DONE just because it contains the word.
        low = text.lower()
        if "continue" in low:
            return False
        return "done" in low

    def _forced_summary(self, messages: list[dict], system_prompt: str,
                        result: TurnResult) -> None:
        """The step budget ran out with no answer: force one tools-free turn that
        can only summarise. Without this a stuck-but-busy model ends the turn in
        blank silence, indistinguishable from a crash."""
        try:
            if self.interrupt():
                return
            probe_messages = list(messages) + [
                {"role": "user", "content": AUTO_CONTINUE_SUMMARY_MSG}
            ]
            probe = TurnResult()
            self._stream(probe_messages, [], probe, system_prompt)
            text = (probe.text or "").strip()
        except Exception as _oc_exc:
            swallow(_oc_exc, "agent/loop.py:_forced_summary")
            return
        if not text:
            return
        try:
            self._history.append({"role": "assistant", "content": text})
            result.text = text
        except Exception as _oc_exc:
            swallow(_oc_exc, "agent/loop.py:_forced_summary commit")

    # -- prompt queue (opencode's queue-and-promote, in-turn drain) -------
    MAX_PROMPT_QUEUE = 10
    MAX_PROMPT_CHARS = 32 * 1024










    # -- permission ------------------------------------------------------

    # -- tools ------------------------------------------------------------

    # -- memory capture ---------------------------------------------------
    def _start_memory_capture(self) -> None:
        """Distil the previous session into durable notes, in the background.

        Runs on launch rather than on session close: a phone has no business
        holding a background process, Android kills those anyway, and closing
        must stay instant. The thread does one small model call while the user
        is already typing, so nothing is ever waited on.

        Nothing happens unless a real-work session is actually queued, so an
        idle launch costs one small file read.
        """
        if not _memory_capture_enabled(self.cfg):
            return
        try:
            from . import memory_capture as _mc

            if not _mc.peek_pending():
                return
            import threading as _threading

            def _run() -> None:
                try:
                    from ..providers import build_provider as _bp

                    _mc.run_pending(
                        _bp(self.cfg, self.provider_id, self.model_id),
                        str(self.worktree),
                    )
                except Exception as _oc_exc:
                    swallow(_oc_exc, "agent/loop.py:337")
                    pass

            _threading.Thread(target=_run, name="memory-capture", daemon=True).start()
        except Exception as _oc_exc:
            swallow(_oc_exc, "agent/loop.py:341")
            pass

    def _queue_session_capture(self) -> None:
        """Queue this session for the next launch's capture. Instant: it distils
        locally and writes a few KB, with no model call."""
        if not _memory_capture_enabled(self.cfg):
            return
        try:
            from . import memory_capture as _mc

            _mc.mark_pending(self.session_id, list(self._history))
        except Exception as _oc_exc:
            swallow(_oc_exc, "agent/loop.py:353")
            pass

    # -- sub-agents -------------------------------------------------------




    def _prepare_task(
        self, arguments: dict[str, Any]
    ) -> tuple[dict | None, dict | None]:
        """Validate + build a child AgentLoop WITHOUT running it.

        Returns (bundle, error): exactly one is not None. Shared by the
        inline path (run now via _finish_task) and the background path
        (run on a worker thread, poll later).
        """
        from ..tools import build_registry

        prompt = str(arguments.get("prompt", "")).strip()
        if not prompt:
            return None, {"output": "task: no prompt provided.", "error": True}
        description = str(arguments.get("description", "")).strip() or "sub-agent"
        requested_type = str(arguments.get("subagent_type", "")).strip() or "build"
        # Launch what was asked: an unknown name is a model mistake, not a
        # build request. Fail loudly with the valid names so the model
        # retries correctly — silently running build with full tools when
        # a scoped specialist was intended is the dangerous wrong choice.
        try:
            from ..permission import list_agents as _la

            known = {n for n, _d, _c in _la(self.cfg)} | {"general"}
        except Exception:
            known = {"build", "plan", "explore", "general"}
        if requested_type not in known:
            return None, {
                "output": (
                    f"Unknown subagent_type '{requested_type}'. "
                    f"Valid: {', '.join(sorted(known))}."
                ),
                "error": True,
            }
        subagent_type = requested_type

        # Enforce cfg.subagent_depth: the number of sub-agent LEVELS allowed
        # below the main agent (default 1). Without this the setting was dead:
        # a model could recurse task -> task -> task without bound, each level
        # minting a new session + rotation + registry on a low-RAM phone.
        child_depth = self._depth + 1
        allowed = max(0, int(getattr(self.cfg, "subagent_depth", 1)))
        if child_depth > allowed:
            return None, {
                "output": (
                    f"sub-agent depth limit reached (cfg.subagent_depth={allowed}):"
                    " this agent is already at the deepest allowed level."
                    " Do the work directly instead of delegating."
                ),
                "error": True,
            }

        # A read-only agent (plan/explore, or a custom with readonly) must
        # not spawn a mutating sub-agent: the child would inherit
        # read-write tools. Force the child to stay read-only too.
        from ..permission import agent_readonly as _is_ro2

        try:
            _ro = bool(_is_ro2(self.cfg, self.agent))
        except Exception:
            _ro = self.agent in ("plan", "explore")
        if _ro and subagent_type not in ("plan", "explore"):
            try:
                _child_ro = bool(_is_ro2(self.cfg, subagent_type))
            except Exception:
                _child_ro = False
            if not _child_ro:
                subagent_type = "plan" if self.agent == "plan" else "explore"

        # Bound simultaneously running children (phone RAM): overflow tells
        # the model to do the work directly instead of OOMing.
        if not self._subagent_sem.acquire(blocking=False):
            return None, {
                "output": (
                    f"too many sub-agents running (max {self._MAX_CONCURRENT_SUBAGENTS}):"
                    " wait for one to finish, or do the work directly instead of delegating."
                ),
                "error": True,
            }

        sub_session = new_session(
            directory=str(self.directory),
            provider=self.provider_id or self.cfg.provider,
            model=self.model_id or self.cfg.model,
            agent=subagent_type,
            title=description,
            parent_id=self.session_id,
        )
        # No save here: the sub-agent session has no conversation yet and must
        # not mint an empty file. It is persisted only once it has content (the
        # completion saves below), or by the app's exit/close save-all.

        sub = AgentLoop(
            cfg=self.cfg,
            registry=build_registry(self.cfg),
            directory=self.directory,
            # The child gets its OWN Zen session identity: sharing the
            # parent's id pinned every parallel child to the same upstream
            # lane (correlated 429s), and one child's rotation poisoned the
            # siblings' lane via the global epoch. The factory seam stays
            # intact for tests — we just rebind the built rotation to the
            # child session. Same for the grandchild factory, so nesting
            # derives from the child, not the grandparent.
            provider=self._child_rotation(self.provider_factory, sub_session.id),            auth=self.auth,
            permission_engine=self.permission,
            question_service=self.question_service,
            on_event=self._subagent_bridge(sub_session.id),
            ask_lock=self._ask_lock,
            agent=subagent_type,
            provider_id=self.provider_id,
            model_id=self.model_id,
            interrupt=self.interrupt,
            session_id=sub_session.id,
            provider_factory=lambda: build_rotation(self.cfg, self.auth, sub_session.id),
            depth=child_depth,
        )
        self.subagents[sub_session.id] = sub
        bundle = {
            "sub": sub,
            "session": sub_session,
            "prompt": prompt,
            "description": description,
            "agent": subagent_type,
            "started": time.time(),
        }
        self._emit(
            "subagent_start",
            session_id=sub_session.id,
            agent=subagent_type,
            title=description,
            prompt=prompt,
            call_id=getattr(self._spawn_ctx, "call_id", None) or "",
        )
        return bundle, None


    # -- background sub-agents (fire-and-keep-working) -------------------
    # At most ONE detached child: each holds history + registry + rotation
    # in RAM and burns its own context window. The model starts it, keeps
    # working, and collects the reply later with task_read (or task_status
    # to peek, task_stop to kill). Unread results are folded into history
    # at turn end so nothing is silently lost.
    _MAX_BG_AGENTS = 1













    _PARALLEL_TOOL_WORKERS = 6
    _TOOL_TIMEOUT = 120.0
    _MUTATING_TOOLS = ("edit", "write", "apply_patch")
    # Cap on simultaneously RUNNING sub-agents per parent: each live child
    # holds a full history + registry + rotation + (in the TUI) a chat view,
    # so unbounded model fan-out OOMs low-RAM phones. Overflow degrades to a
    # guidance error (do the work directly) instead of a crash.
    _MAX_CONCURRENT_SUBAGENTS = 4




    # -- main turn --------------------------------------------------------





    def _run_turn_body(self, user_text: str) -> TurnResult:
        result = TurnResult()

        # Honor a deferred rotation rebuild (model pick / settings change /
        # end-of-turn lane switch): the network-touching rebuild stays OFF
        # the UI thread by running here, before anything streams.
        if getattr(self, "_rotation_dirty", False):
            self._rotation_dirty = False
            try:
                self.rebuild_rotation()
            except Exception as _oc_exc:
                swallow(_oc_exc, "agent/loop.py:545")
                pass

        # Stable session across turns (official parity): Rotation.new_turn()
        # is now a no-op that keeps the same x-opencode-session so Zen holds
        # its sticky healthy lane. Dead lanes still rotate via
        # _rotate_lane()/rotate_session() on real failures.
        try:
            rotation = getattr(self, "rotation", None)
            new_turn = getattr(rotation, "new_turn", None)
            if callable(new_turn):
                new_turn()
        except Exception as _oc_exc:
            swallow(_oc_exc, "agent/loop.py:557")
            pass

        reset = getattr(self.permission, "reset_doom_tracking", None)
        if reset:
            reset()

        # agent reminder (plan/build-switch): fires only when the active agent
        # differs from the one that ran the previous turn, so the model is told
        # its rules were replaced instead of silently keeping its old stance.
        reminder = system_mod.agent_reminder(
            self.agent, self._was_plan(), self.cfg, previous=self._prev_agent
        )
        system_prompt = system_mod.build_system_prompt(
            directory=self.directory,
            worktree=self.worktree,
            provider_id=self.provider_id,
            model_id=self.model_id,
            cfg=self.cfg,
            agent=self.agent,
            query=user_text,
            recent=self._recent_context(),
        )

        history = self.request_history()
        messages = msg_mod.build_messages(history=history[:-1], user_text=user_text, reminder=reminder)
        # Images queued by the `attach` tool ride THIS user message. Drained
        # after build_messages so it attaches to the new turn, not a replayed
        # one, and drained even when nothing was queued (a no-op).
        try:
            from .images import take_pending

            _imgs = take_pending()
        except Exception as _oc_exc:
            swallow(_oc_exc, "agent/loop.py:pending images")
            _imgs = []
        if _imgs and messages:
            try:
                messages[-1]["images"] = _imgs
            except Exception as _oc_exc:
                swallow(_oc_exc, "agent/loop.py:attach images")
        messages = self._prepend_system(messages, system_prompt)

        # Stable tool-schema order (fix 3): deterministic names order so the
        # request prefix (system + schemas) is byte-stable across turns for
        # providers with prefix caching. No behavior change otherwise.
        tools = self._active_tool_schemas()
        try:
            tools = sorted(tools, key=lambda s: str((s.get("function") or {}).get("name", "")))
        except Exception as _oc_exc:
            swallow(_oc_exc, "agent/loop.py:587")
            pass

        # The selected model's real window drives both compaction and the
        # trimming safety net, so this works for ANY model — not just the
        # bundled free ones. An unknown window (0) falls back to the configured
        # hard budget and lets the post-overflow recovery path compact.
        ctx = self._model_context_size()
        output_limit = self._model_output_limit()
        usable = compact_mod.usable_context(ctx, output_limit) if ctx > 0 else 0

        # Fix 1: shrink OLD tool outputs for the request copy (history on disk
        # stays full). The last keep_turns go verbatim; older tool bodies
        # become one-line receipts. Pairing/ids untouched — safe to replay.
        try:
            keep_turns = int(getattr(self.cfg, "trim_keep_turns", 2) or 2)
        except (TypeError, ValueError):
            keep_turns = 2
        try:
            trim_chars = int(getattr(self.cfg, "trim_max_chars", 500) or 500)
        except (TypeError, ValueError):
            trim_chars = 500
        if keep_turns > 0:
            try:
                messages = trim_mod.shrink_tool_history(messages, keep_turns, trim_chars)
            except Exception as _oc_exc:
                swallow(_oc_exc, "agent/loop.py:612")
                pass

        # Fix 2: adopt a ready early summary (built quietly behind at ~70%).
        # It replaces the SAME head the overflow path would summarize, so the
        # turn continues with zero extra model calls instead of freezing.
        try:
            adopted = self._adopt_early_summary(system_prompt)
            if adopted is not None:
                messages = adopted
        except Exception as _oc_exc:
            swallow(_oc_exc, "agent/loop.py:622")
            pass

        # Proactive compaction (mirrors upstream `compactIfNeeded`): estimate the
        # request about to be sent — system + messages + tools — and compact it
        # before it ever reaches the provider's length limit. This runs on the
        # UNTRIMMED history so compaction (which preserves the conversation via
        # the anchored summary) gets first pick; the trim below is only a
        # last-resort safety net that may drop old turns. The estimate alone can
        # undercount tool-heavy conversations, so ALSO honor the actual usage
        # reported by the provider for the last completion (persists across
        # turns on the loop) — mirrors opencode's `lastFinished.tokens` check.
        if self.cfg.compaction_enabled and usable > 0:
            overflow = compact_mod.is_overflow(ctx, compact_mod.estimate_request(messages, tools), output_limit)
            if not overflow and self._usage_total:
                actual = self._usage_total.get("total_tokens") or usage_total_dict(self._usage_total)
                overflow = actual >= usable
            if overflow:
                compacted = self._compact_context(system_prompt)
                if compacted is not None:
                    messages = compacted

        # Kick the early summary for the NEXT turn while this one streams
        # (fix 2): at >=70% of usable it pre-builds the summary behind, so a
        # later overflow adopts it instead of freezing the turn.
        try:
            self._maybe_kick_early_summary(system_prompt, messages, tools, ctx, output_limit, usable)
        except Exception as _oc_exc:
            swallow(_oc_exc, "agent/loop.py:649")
            pass

        # Last-resort cap: never send past the model's usable window (or its raw
        # context when the reserve can't be sized, or the configured budget when
        # the window is unknown). After compaction this is normally a no-op.
        # Runs on the SHRUNK request so the estimate reflects what's sent.
        trim_budget = usable if usable > 0 else (ctx if ctx > 0 else self.cfg.context_budget)
        messages = msg_mod.trim_history(messages, trim_budget)

        turn_start_calls = int(getattr(result, "tool_calls_made", 0) or 0)
        for step in range(MAX_STEPS):
            if self.interrupt():
                self._emit("interrupted")
                break

            # A queued prompt invalidates any pre-built summary (it was made
            # for the old history) — the worker's length key won't match, and
            # we clear here too so nothing stale can be adopted.
            # opencode's queue-and-promote: if a prompt was queued while a
            # provider turn ran, fold it into this SAME drain at the next
            # provider-turn boundary — the generation "pauses for a beat", the
            # chat enters context as a continuation, then the model reasons
            # about it (keep working / stop, whatever it asks). Mirror the
            # official runLoop, which only exits when the last assistant's
            # parent IS the last user message, so a promoted prompt keeps the
            # drain going. Only after the first turn (step>0) — the initial
            # prompt is already in `messages`, and folding before the first
            # stream would merge two user messages.
            if step > 0:
                queued = self._next_prompt()
                if queued is not None:
                    self._history.append({"role": "user", "content": queued})
                    self._emit("prompt_promoted", text=queued)
                    try:
                        with self._early_summary_lock:
                            self._early_summary_text = ""
                            self._early_summary_at = -1
                            self._early_summary_head = {}
                    except Exception as _oc_exc:
                        swallow(_oc_exc, "agent/loop.py:688")
                        pass
                    # shrink the promoted rebuild too (same receipt rules)
                    try:
                        _kt = int(getattr(self.cfg, "trim_keep_turns", 2) or 2)
                        _mc = int(getattr(self.cfg, "trim_max_chars", 500) or 500)
                        messages = trim_mod.shrink_tool_history(list(self._history), _kt, _mc)
                    except Exception:
                        messages = list(self._history)

            self._emit("step", step=step)
            # stream
            self._stream(messages, tools, result, system_prompt)

            if result.error:
                break

            # collect tool calls
            if self._pending_calls:
                calls = self._pending_calls
                self._pending_calls = []
                # Drop degenerate calls (missing name); a model that only emits
                # empty tool calls would otherwise spin a silent 50-step loop.
                calls = [c for c in calls if c.get("name")]
                if not calls:
                    result.error = "model produced an invalid tool call (missing name)"
                    self._emit("error", error=result.error)
                    break
            elif getattr(self.cfg, "auto_continue", False) and not self.interrupt() and step < MAX_STEPS - 1 and self.prompt_pending() == 0 and sum(1 for m in self._history if isinstance(m, dict) and m.get("content") == AUTO_CONTINUE_MSG) < AUTO_CONTINUE_MAX_NUDGES:
                # auto_continue ON: text-only stop is not the end.
                # Rule: real work + real answer -> stop. Anything idle -> nudge.
                # - tools ran this turn AND this step has non-empty text:
                #   the final summary landed (your whoami fix target) -> stop.
                # - empty text (model went silent, incl. right after tools):
                #   wait a beat, send the nudge, run another provider step in
                #   THIS same drain so the summary always lands.
                # - no tools at all + chatter: idle -> nudge (never-stop mode).
                # Real queued message wins; OFF stops as before.
                worked = int(getattr(result, "tool_calls_made", 0) or 0) > turn_start_calls
                final_text = (getattr(result, "text", "") or "").strip()
                if worked and final_text:
                    if self.prompt_pending() == 0:
                        break
                    continue
                # Repeat guard (new-session safety): model echoing the identical
                # text as its PREVIOUS turn is chatting, not working — stop
                # instead of burning all 25 nudges on repeats. Compares the two
                # most recent assistant texts (the current reply is already in
                # history, so one match alone is not a repeat). Free (no provider
                # call), so it runs BEFORE the done-check probe below.
                if final_text:
                    try:
                        found: list[str] = []
                        for m in reversed(self._history):
                            if isinstance(m, dict) and m.get("role") == "assistant":
                                c = m.get("content")
                                if isinstance(c, str) and c.strip():
                                    found.append(c.strip())
                                    if len(found) >= 2:
                                        break
                        if len(found) >= 2 and found[0] == found[1] == final_text:
                            if self.prompt_pending() == 0:
                                break
                            continue
                    except Exception as _oc_exc:
                        swallow(_oc_exc, "agent/loop.py:751")
                        pass
                # Model spoke but ran NO tool and is not repeating itself: the one
                # case where the transcript cannot tell us whether the job is done.
                # Ask it outright instead of guessing from wording — DONE ends the
                # drain, CONTINUE (or silence, or nonsense) keeps nudging. A model
                # that worked and answered already returned above, so a genuinely
                # finished job never reaches this probe.
                if final_text and self._job_is_done(messages, system_prompt):
                    if self.prompt_pending() == 0:
                        break
                    continue
                try:
                    self._emit("auto_continue", text=AUTO_CONTINUE_MSG)
                except Exception as _oc_exc:
                    swallow(_oc_exc, "agent/loop.py:755")
                    pass
                deadline = time.monotonic() + AUTO_CONTINUE_WAIT_S
                while time.monotonic() < deadline:
                    if self.interrupt() or self.prompt_pending() > 0:
                        break
                    time.sleep(0.1)
                if self.interrupt():
                    break
                if self.prompt_pending() > 0:
                    continue
                self._history.append({"role": "user", "content": AUTO_CONTINUE_MSG})
                messages = list(self._history)
                continue
            else:
                # The model replied with text but made no tool call — a
                # complete answer for the current prompt. If a prompt was
                # queued meanwhile, don't exit: the loop-top fold on the next
                # iteration promotes it into this same drain (the stream "pauses
                # a beat", the chat enters context, then the model reasons).
                if self.prompt_pending() == 0:
                    break
                continue

            # Some models emit tool calls without an id. Assign a stable
            # fallback (must match assistant_message_from_calls) so the
            # assistant declaration and the following tool-result messages use
            # the same id; otherwise strict OpenAI-compatible backends reject
            # with "insufficient tool messages following tool_calls". The
            # counter keeps ids unique across every step and turn so the UI can
            # reliably match tool rows by call_id (per-step indices collide and
            # cause duplicate/incorrect tool rows).
            for call in calls:
                if not call.get("id"):
                    self._call_seq += 1
                    call["id"] = f"call_{self._call_seq}"

            # append assistant message with calls to history — keep the model's
            # own text and reasoning in the same message so the next tool-loop
            # request is a faithful replay (reasoning models lose thread when
            # their previous message is stored empty).
            self._history.append(
                parse_mod.assistant_message_from_calls(
                    calls,
                    reasoning=result.reasoning,
                    content=result.text,
                )
            )
            messages = list(self._history)

            # Execute the step's tool calls. A step with several calls runs them
            # concurrently (official opencode launches sub-agents in parallel
            # when the model emits multiple `task` calls in one reply). Each
            # call runs on its own worker; results come back on this thread and
            # history is written in the ORIGINAL call order so the
            # assistant_message -> tool_result pairing stays valid for replay.
            # No early-return may happen between committing the assistant
            # message and writing its tool results: strict backends reject a
            # request whose assistant tool_calls are not all answered by tool
            # messages ("insufficient tool messages following tool_calls"), and
            # the poisoned history would then be persisted and re-sent forever.
            # An interrupt here instead yields placeholder "(interrupted)"
            # results so the pair always closes.
            prepared = []
            for call in calls:
                name = call.get("name", "")
                try:
                    arguments = parse_mod.parse_arguments(call.get("arguments", "{}"))
                except Exception:
                    arguments = {"arguments": call.get("arguments", "{}")}
                prepared.append((call, name, arguments))

            for call, name, arguments in prepared:
                self._emit_tool("tool_call", name, arguments=arguments, call_id=call.get("id", ""))

            if len(prepared) > 1:
                outcomes = self._run_tools_parallel(prepared)
            else:
                outcomes = []
                for call, name, arguments in prepared:
                    if self.interrupt():
                        outcomes.append(
                            (call, name, {"output": "(interrupted)", "error": True, "stopped": True})
                        )
                        break
                    tool_result = self.run_tool(name, arguments, call_id=call.get("id", ""))
                    outcomes.append((call, name, tool_result))

            for call, name, tool_result in outcomes:
                result.tool_calls_made += 1
                msg = parse_mod.tool_result_message(
                    call.get("id", ""),
                    name,
                    tool_result.get("output", ""),
                    error=bool(tool_result.get("error")),
                )
                # Persist the child's session link INSIDE the saved transcript:
                # without this a resumed parent's task rows have no sessionId
                # (it only lived in the live tool-result metadata), so
                # clicking them after a restart went nowhere.
                try:
                    meta = (tool_result.get("metadata") or {})
                    sid = meta.get("sessionId") if isinstance(meta, dict) else None
                    if name == "task" and sid:
                        msg["session_id"] = sid
                        msg["sessionId"] = sid
                except Exception as _oc_exc:
                    swallow(_oc_exc, "agent/loop.py:861")
                    pass
                self._history.append(msg)
            messages = list(self._history)

            if self.interrupt():
                self._emit("interrupted")
                return result
            # A fresh user message arrived while parked in task_read (the
            # TUI broke the wait via wake_bg_wait): end the drain after
            # this step instead of starting new provider steps, so the
            # message starts its own turn immediately — no QUEUED wait.
            # The bg agent keeps running; its reply folds in later.
            try:
                if getattr(self, "_bg_end_turn", False):
                    self._bg_end_turn = False
                    break
            except Exception as _oc_exc:
                swallow(_oc_exc, "agent/loop.py:878")
                pass

            # Auto-compaction between steps (mirrors upstream opencode): after a
            # step completes, compact as soon as the ACTUAL provider-reported
            # usage fills the usable window — not only when the request is
            # estimated to overflow at the start of a turn. Tool loops grow the
            # context faster than the turn-start estimate predicts, so without
            # this a long session runs to 100% and stalls. Uses the SELECTED
            # model's window, so it works for any provider/model lane.
            if self.cfg.compaction_enabled:
                compacted = self._maybe_compact_by_usage(system_prompt)
                if compacted is not None:
                    messages = compacted

        # Step budget exhausted. Falling out of the for-loop instead of a break
        # means the model never produced an answer — it ran tools every step or
        # stayed silent. Force one tools-free summary so the turn ends with words
        # instead of a blank stop that looks like a crash.
        if getattr(self.cfg, "auto_continue", False) and not (getattr(result, "text", "") or "").strip():
            self._forced_summary(messages, system_prompt, result)

        # NOTE: errors are NOT appended to history. A permanent "[system]"
        # user message would re-send every past failure to the provider on
        # every future turn — polluting the context window and confusing the
        # model with stale noise. Errors reach the user through
        # TurnResult.error / the "error" event instead.
        # Track agent state for build-switch detection
        self._prev_agent = self.agent
        self._fold_finished_bg_agents()
        # Single-store pass (end of turn): big OLD tool bodies move to spill
        # files; live RAM keeps receipts, disk saves stay full (get_history
        # rehydrates before save — see below). Only when history is long
        # enough to matter; never touches the last keep_turns.
        try:
            if len(self._history) > 20:
                _kt = int(getattr(self.cfg, "trim_keep_turns", 2) or 2)
                _sid = str(getattr(self, "_session_id", "") or getattr(self, "session_id", "") or "")
                slim, _n = trim_mod.spill_old_tools(
                    self._history,
                    session_id=_sid,
                    keep_turns=_kt,
                )
                if _n > 0:
                    self._history = slim
        except Exception as _oc_exc:
            swallow(_oc_exc, "agent/loop.py:916")
            pass
        return result

    # -- streaming --------------------------------------------------------









    # continuation nudges sent to the SAME model when it stops mid-operation.
    # A plain re-send would replay every tool call from scratch; nudging lets
    # the model continue where it left off. Never rotates to another lane.
    _RETRY_NUDGES = (
        "keep going",
        "don't stop, keep going",
        "continue working, don't stop",
        "keep going, finish the task",
    )








    # -- compaction --------------------------------------------------------


    # -- early background summary (fix 2) --------------------------------
    # At >=70% of the usable window a daemon pass summarizes the head BEHIND
    # the running turn. A later overflow then adopts the ready text with ZERO
    # extra model calls instead of freezing the turn on a sync summary.
    # Same head/tail split + prompt + application as _compact_context, so the
    # result is interchangeable. Disabled with compaction, single-flight per
    # history length, invalidated by interrupts/new prompts.





    # -- session glue -----------------------------------------------------





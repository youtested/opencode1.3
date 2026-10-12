from __future__ import annotations

from ..util.log import swallow
from pathlib import Path

from ..errors import warn as _warn_error
from ..providers.base import usage_total_dict
from ..tools.write import _atomic_write_bytes

from . import compaction as compact_mod
from . import system as system_mod
from . import trim as trim_mod
from .loop_config import _BgAbandoned

class LoopHistoryMixin:
    def undo_last(self) -> str:
        """Revert the most recent edit/write tool call (file-level snapshot)."""
        if not self._undo_stack:
            return "Nothing to undo."
        entry = self._undo_stack.pop()
        path = Path(entry["path"])
        try:
            if entry["original"] is None:
                if path.exists():
                    path.unlink()
                dirs = entry.get("dirs") or []
                for d in dirs:
                    # Already deepest-first from _missing_directories: removing a
                    # child can empty its parent, never the reverse. This used to
                    # be reversed(), so the parent was always tried first, found
                    # non-empty, and left behind on every undo of a new file.
                    d = Path(d)
                    # only remove directories we created and that are now empty
                    try:
                        if d.is_dir() and not any(d.iterdir()):
                            d.rmdir()
                    except OSError as _oc_exc:
                        swallow(_oc_exc, "opencode_py/agent/loop_history.py:32")
                        pass
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                # Atomic (tmp + rename + fsync), not a bare write_bytes: a plain
                # write truncates the target first, so a crash or a full disk
                # mid-restore left the user with a half-written file — the one
                # outcome an undo command must never produce.
                _atomic_write_bytes(path, entry["original"])
        except OSError as e:
            return f"Undo failed for {path}: {e}"
        self._emit("undo", path=str(path))
        return f"Reverted {path}."

    def _on_rotated(self, provider_id: str, model_id: str, reason: str = "") -> None:
        """A failover lane succeeded; announce it (with the switch reason)."""
        self._emit("rotated", provider=provider_id, model=model_id, reason=reason)

    def _model_context_size(self) -> int:
        """Context-window size of the active lane (0 when unknown)."""
        from ..providers import model_context_size

        pid = self.provider_id or self.cfg.provider
        mid = self.model_id or self.cfg.model
        return model_context_size(pid, mid, auth=self.auth)

    def _model_output_limit(self) -> int:
        """Max output tokens of the active lane (0 when unknown)."""
        from ..providers import model_output_limit

        pid = self.provider_id or self.cfg.provider
        mid = self.model_id or self.cfg.model
        return model_output_limit(pid, mid)

    def _estimate_request_tokens(self, system_prompt: str) -> int:
        """Request-size footprint of the current history.

        Same math the engine sends: system prompt + history + tool schemas
        (mirrors upstream `estimate({system, messages, tools})`). Shared by
        every place that repaints the footer's `12,345 (6%)` without a fresh
        provider usage event (compaction, interrupts, resume estimate).
        """
        try:
            msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + list(self._history)
            return compact_mod.estimate_request(msgs, self._active_tool_schemas())
        except Exception:
            return 0

    def estimate_history_tokens(self) -> int:
        """Request-size footprint of the current history, system prompt built
        in. Used by the TUI to paint a true count when reopening a session."""
        try:
            system_prompt = system_mod.build_system_prompt(
                directory=self.directory,
                worktree=self.worktree,
                provider_id=self.provider_id,
                model_id=self.model_id,
                cfg=self.cfg,
                agent=self.agent,
            )
        except Exception:
            system_prompt = ""
        return self._estimate_request_tokens(system_prompt)

    def _emit_request_usage(self, messages: list, tools: list, output_text: str = "") -> None:
        """Paint the footer after a turn that produced no provider usage event
        (interrupt / failure): the request WAS sent, so count the request
        estimate plus any partial output. Callers skip this when a real
        usage already landed for the turn."""
        try:
            sent = compact_mod.estimate_request(messages, tools)
            partial = compact_mod.estimate_tokens(output_text) if output_text else 0
            total = sent + partial
            self._usage_total = {"input_tokens": sent, "output_tokens": partial, "total_tokens": total}
            usage_evt = dict(self._usage_total)
            ctx = self._model_context_size()
            if ctx:
                usage_evt["context_size"] = ctx
            self._emit("usage", usage=usage_evt)
        except Exception as exc:
            # Falling back to an estimate is fine; failing to report the
            # usage is not, because the UI then shows no token/cost figure
            # at all with nothing to explain the gap.
            _warn_error("agent.loop", "could not emit the usage estimate", exc)

    def _was_plan(self) -> bool:
        return self._prev_agent == "plan"

    def _prepend_system(self, messages: list[dict], system_prompt: str) -> list[dict]:
        return [{"role": "system", "content": system_prompt}] + messages

    def force_compact(self) -> str:
        """Manually compact the conversation now (the `/compact` command).

        Mirrors upstream opencode: `/compact` runs the same AI compaction as the
        automatic overflow path. Returns the anchored summary ("" when nothing
        was compacted). Raises on provider failure.
        """
        system_prompt = system_mod.build_system_prompt(
            directory=self.directory,
            worktree=self.worktree,
            provider_id=self.provider_id,
            model_id=self.model_id,
            cfg=self.cfg,
            agent=self.agent,
        )
        messages = self._compact_context(system_prompt)
        if messages is None:
            return ""
        return self._compaction_summary or ""

    def _maybe_compact_by_usage(self, system_prompt: str) -> list[dict] | None:
        """Compact after a step when actual usage has filled the usable window.

        Mirrors upstream opencode's `compaction.isOverflow({tokens, model})`,
        which runs after every completed step: once the provider-reported token
        count (input + output of the last completion) reaches the usable window
        (context minus the reserve), the conversation is summarized so the next
        request never hits the provider's hard length limit. The window comes
        from the SELECTED model, so this works for any provider/model lane.
        """
        ctx = self._model_context_size()
        if ctx <= 0:
            return None
        output_limit = self._model_output_limit()
        usable = compact_mod.usable_context(ctx, output_limit)
        if usable <= 0:
            return None
        count = self._usage_total.get("total_tokens") or usage_total_dict(self._usage_total)
        if count < usable:
            return None
        return self._compact_context(system_prompt)

    def _early_summary_threshold(self, usable: int) -> float:
        try:
            frac = float(getattr(self.cfg, "early_summary_at", 0.7) or 0.7)
        except (TypeError, ValueError):
            frac = 0.7
        return min(0.95, max(0.5, frac))

    def _maybe_kick_early_summary(self, system_prompt, messages, tools, ctx, output_limit, usable) -> None:
        if not self.cfg.compaction_enabled:
            return
        if usable <= 0 or ctx <= 0:
            return
        try:
            est = compact_mod.estimate_request(messages, tools)
        except Exception as _oc_exc:
            swallow(_oc_exc, "agent/loop_history.py:174")
            return
        if est < int(usable * self._early_summary_threshold(usable)):
            return
        if compact_mod.is_overflow(ctx, est, output_limit):
            return  # the sync path handles it this turn; no point pre-building
        with self._early_summary_lock:
            if self._early_summary_at == len(self._history):
                return  # already building/built for this history
            self._early_summary_at = len(self._history)
            self._early_summary_text = ""
            self._early_summary_head = {}
        hist = [m for m in self._history if not m.get("compaction")]
        if len(hist) < 4:
            with self._early_summary_lock:
                self._early_summary_at = -1
            return
        try:
            head, tail = compact_mod.select_tail(
                hist, tail_turns=self.cfg.compaction_tail_turns,
                context=ctx, output_limit=output_limit,
            )
        except Exception:
            with self._early_summary_lock:
                self._early_summary_at = -1
            return
        if not head:
            with self._early_summary_lock:
                self._early_summary_at = -1
            return
        try:
            prompt = compact_mod.summarize_conversation_prompt(head, tail)
        except Exception:
            with self._early_summary_lock:
                self._early_summary_at = -1
            return
        at = len(self._history)
        try:
            def _mid2(m):
                if not isinstance(m, dict):
                    return ""
                return str(m.get("id") or "") + "|" + str(m.get("role") or "") + "|" + str(m.get("content", ""))[:120]
            snap = {"len": len(head), "first": _mid2(head[0]), "last": _mid2(head[-1])}
        except Exception:
            snap = {}
        with self._early_summary_lock:
            self._early_summary_head = snap
        try:
            import threading as _th

            worker = _th.Thread(
                target=self._early_summary_run,
                args=(prompt, at),
                name="opencode_py-early-summary", daemon=True,
            )
            worker.start()
        except Exception:
            with self._early_summary_lock:
                self._early_summary_at = -1
                self._early_summary_head = {}

    def _early_summary_run(self, prompt: str, at: int) -> None:
        """Background summary on a FRESH provider — never self.rotation.

        rotation.stream mutates shared state (_active_provider, lane epochs,
        sticky sessions): sharing it with the live turn would corrupt the
        interrupt path and lane affinity. A fresh one-shot provider costs one
        extra lane pick and touches nothing shared.
        """
        texts: list[str] = []
        try:
            from ..providers import build_provider as _bp
            provider = _bp(self.cfg, self.provider_id or self.cfg.provider,
                           self.model_id or self.cfg.model, self.auth,
                           getattr(self.rotation, "session_id", None))
        except Exception:
            with self._early_summary_lock:
                if self._early_summary_at == at:
                    self._early_summary_at = -1
                    self._early_summary_head = {}
            return

        def _bg_event(evt) -> None:
            try:
                if evt.kind == "text_delta":
                    texts.append(evt.text)
            except Exception as _oc_exc:
                swallow(_oc_exc, "agent/loop_history.py:260")
                pass
            # user hit Esc mid-turn: abandon the worker fast instead of
            # burning a summary nobody will adopt (length key won't match,
            # but fail fast rather than wasting the lane + battery).
            try:
                if self.interrupt():
                    raise _BgAbandoned()
            except _BgAbandoned:
                raise
            except Exception as _oc_exc:
                swallow(_oc_exc, "agent/loop_history.py:270")
                pass

        try:
            provider.stream_chat([{"role": "user", "content": prompt}], [], _bg_event)
        except _BgAbandoned:
            with self._early_summary_lock:
                if self._early_summary_at == at:
                    self._early_summary_at = -1
                    self._early_summary_head = {}
            return
        except Exception:
            with self._early_summary_lock:
                if self._early_summary_at == at:
                    self._early_summary_at = -1
                    self._early_summary_head = {}
            return
        summary = "".join(texts).strip()
        with self._early_summary_lock:
            if self._early_summary_at == at and summary:
                self._early_summary_text = summary
            elif self._early_summary_at == at:
                self._early_summary_at = -1

    def _adopt_early_summary(self, system_prompt: str) -> list[dict] | None:
        """Use the ready background summary if its head still matches.

        The kick snapshots the head it summarized (not just a length): the
        turn keeps appending after the kick, so a length key could never
        match. Adoption verifies the snapshotted head is still the prefix
        of this history — same split, same first/last head ids.
        """
        with self._early_summary_lock:
            text = self._early_summary_text
            snap = dict(getattr(self, "_early_summary_head", None) or {})
        if not text or not snap:
            return None
        history = [m for m in self._history if not m.get("compaction")]
        if len(history) < 4:
            return None
        try:
            ctx = self._model_context_size()
            head, tail = compact_mod.select_tail(
                history, tail_turns=self.cfg.compaction_tail_turns,
                context=ctx, output_limit=self._model_output_limit(),
            )
        except Exception:
            return None
        if not head:
            return None
        # head-identity check: same length, same boundary message ids
        try:
            def _mid(m):
                if not isinstance(m, dict):
                    return ""
                return str(m.get("id") or "") + "|" + str(m.get("role") or "") + "|" + str(m.get("content", ""))[:120]
            if len(head) != int(snap.get("len", -1)):
                return None
            if _mid(head[0]) != snap.get("first", None) or _mid(head[-1]) != snap.get("last", None):
                return None
        except Exception:
            return None
        self._compaction_summary = text
        summary_text = f"[Summary of earlier conversation]\n{text}"
        if tail and tail[0].get("role") == "user":
            tail0 = dict(tail[0])
            content = tail0.get("content")
            if isinstance(content, str):
                tail0["content"] = f"{summary_text}\n\n{content}"
            elif isinstance(content, list):
                tail0["content"] = [{"type": "text", "text": summary_text}] + list(content)
            tail0["compaction"] = True
            new_history = [tail0] + tail[1:]
        else:
            new_history = [{"role": "user", "content": summary_text, "compaction": True}] + tail
        self._history = new_history
        with self._early_summary_lock:
            self._early_summary_text = ""
            self._early_summary_at = -1
            self._early_summary_head = {}
        try:
            from ..tools.context_ledger import reset_for_compaction
            reset_for_compaction()
        except Exception as _oc_exc:
            swallow(_oc_exc, "agent/loop_history.py:353")
            pass
        self._emit("compacted", summary=text)
        try:
            total = self._estimate_request_tokens(system_prompt)
            self._usage_total = {"input_tokens": total, "output_tokens": 0, "total_tokens": total}
            # The adopted-summary path never emitted usage, so the footer
            # kept showing the pre-compaction high number until the next
            # reply — repaint it now like the other compaction paths.
            usage_evt = dict(self._usage_total)
            if ctx:
                usage_evt["context_size"] = ctx
            self._emit("usage", usage=usage_evt)
        except Exception as exc:
            _warn_error(
                "agent.loop",
                "could not repaint usage after adopting a compaction summary",
                exc,
            )
        return self._prepend_system(list(self._history), system_prompt)
        return self._prepend_system(list(self._history), system_prompt)

    def _compact_context(self, system_prompt: str) -> list[dict] | None:
        """Summarize the conversation into an anchored summary and continue.

        Mirrors upstream opencode's compaction: split the history into a head
        (to summarize) and a recent tail (kept verbatim), ask a model to write
        an anchored summary, then rebuild the request from summary + tail so the
        turn continues instead of erroring out. Returns the rebuilt request
        messages, or None if compaction can't run (no history / model failure).

        Free-first: before spending a model call, try the free rule-cut
        (trim.free_compact). When it frees enough to fit, the turn continues
        with ZERO model calls — same tail, recall notes in place of cut
        bodies, self-recall via history_search action=recall. The model
        summary below runs only when the free cut was NOT enough.
        """
        if not self.cfg.compaction_enabled:
            return None
        history = [m for m in self._history if not m.get("compaction")]
        if len(history) < 4:
            return None
        # Size the preserved tail from the active lane's context window (the
        # model that is actually answering), not a hardcoded 200k assumption.
        ctx = self._model_context_size()
        head, tail = compact_mod.select_tail(
            history,
            tail_turns=self.cfg.compaction_tail_turns,
            context=ctx,
            output_limit=self._model_output_limit(),
        )
        if not head:
            return None
        # Free-first: rule-cut the head; if the result fits, skip the model.
        try:
            _sid = str(getattr(self, "_session_id", "") or getattr(self, "session_id", "") or "")
            _keep = int(getattr(self.cfg, "trim_keep_turns", 2) or 2)
            _usable = compact_mod.usable_context(ctx, self._model_output_limit()) if ctx > 0 else 0
            _cut_hist, _ncut = trim_mod.free_compact(list(history), session_id=_sid, keep_turns=_keep)
            if _ncut > 0:
                _req = compact_mod.estimate_request(
                    self._prepend_system(_cut_hist, system_prompt), self._active_tool_schemas())
                if _usable <= 0 or _req < _usable:
                    self._history = _cut_hist
                    from ..tools.context_ledger import reset_for_compaction

                    reset_for_compaction()
                    self._emit("compacted", summary=f"Free compact: cut {_ncut} old tool outputs (no model call). Re-fetch any with history_search action=recall.")
                    total = self._estimate_request_tokens(system_prompt)
                    self._usage_total = {"input_tokens": total, "output_tokens": 0, "total_tokens": total}
                    usage_evt = dict(self._usage_total)
                    if ctx:
                        usage_evt["context_size"] = ctx
                    self._emit("usage", usage=usage_evt)
                    messages = self._prepend_system(list(self._history), system_prompt)
                    return messages
        except Exception as _oc_exc:
            swallow(_oc_exc, "agent/loop_history.py:429")
            pass
        prompt = compact_mod.summarize_conversation_prompt(head, tail)
        summary_texts: list[str] = []

        # Emit BEFORE the summary stream so the TUI can show the official
        # opencode "Compacting conversation…" indicator while the model works.
        self._emit("compaction_start", reason="auto")

        def on_summary(evt) -> None:
            if evt.kind == "text_delta":
                summary_texts.append(evt.text)
                # Stream the summary live so the TUI shows it being written
                # instead of a static "Compacting…" spinner that dumps the whole
                # block at the end.
                self._emit("summary_delta", text=evt.text)

        try:
            self.rotation.stream(
                [{"role": "user", "content": prompt}],
                [],
                on_summary,
                on_notice=self._on_rotated,
                locked=self.rotation_locked,
            )
        except Exception as e:
            # `compaction_start` was already emitted; a failure must still emit a
            # terminal event or the TUI's "Compacting…" spinner never clears
            # (the InputBar only resets on `compacted`). Report it there so both
            # the state clears and the user sees what went wrong.
            self._emit("compacted", summary=f"Compaction failed: {e}")
            return None
        summary = "".join(summary_texts).strip()
        if not summary:
            self._emit("compacted", summary="Compaction produced no summary")
            return None
        self._compaction_summary = summary
        # Upstream opencode emits the compaction checkpoint as a USER message
        # (`<conversation-checkpoint>` in to-llm-message.ts), NOT an assistant
        # message. Fabricating an assistant message here would be replayed to the
        # API in thinking mode without a `reasoning_content`, which strict
        # gateways reject with "reasoning_content in thinking mode must be passed
        # back". The tail already starts with a user turn, so fold the summary
        # into that first message to keep role alternation intact (a bare
        # `user,user` sequence is rejected by Anthropic).
        summary_text = f"[Summary of earlier conversation]\n{summary}"
        if tail and tail[0].get("role") == "user":
            tail0 = dict(tail[0])
            content = tail0.get("content")
            if isinstance(content, str):
                tail0["content"] = f"{summary_text}\n\n{content}"
            elif isinstance(content, list):
                # Multimodal user turn: keep the original parts verbatim but
                # PREPEND the summary as a text part — the old code replaced
                # the list unchanged, quietly losing the entire anchored
                # summary (the conversation history evaporated on the next
                # compaction/overflow).
                tail0["content"] = [{"type": "text", "text": summary_text}] + list(content)
            tail0["compaction"] = True
            summary_msg = tail0
            new_history = [summary_msg] + tail[1:]
        else:
            summary_msg = {
                "role": "user",
                "content": summary_text,
                "compaction": True,
            }
            new_history = [summary_msg] + tail
        self._history = new_history
        # The pre-compaction bodies (file reads, tool outputs) are gone from
        # the model's head now — the read tool's dedup ledger must forget what
        # was delivered, or re-reads would return "already sent" stubs for
        # content the model can no longer see.
        from ..tools.context_ledger import reset_for_compaction

        reset_for_compaction()
        self._emit("compacted", summary=summary)
        # Recompute the context estimate so the TUI's `12,345 (6%)` reflects
        # the compacted conversation, not the pre-summary size that triggered
        # the overflow (mirrors opencode recomputing tokens after compaction).
        total = self._estimate_request_tokens(system_prompt)
        self._usage_total = {"input_tokens": total, "output_tokens": 0, "total_tokens": total}
        usage_evt = dict(self._usage_total)
        if ctx:
            usage_evt["context_size"] = ctx
        self._emit("usage", usage=usage_evt)
        messages = self._prepend_system(list(self._history), system_prompt)
        return messages

    def set_history(self, history: list[dict]) -> None:
        self._history = list(history)
        if not history:
            self._usage_total = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
            }

    def get_history(self) -> list[dict]:
        # Disk saves + exports must stay FULL: rehydrate spilled bodies so
        # the persisted transcript never loses a byte (live RAM keeps the
        # slim receipts; see the end-of-turn spill above).
        try:
            return trim_mod.rehydrate_spills(list(self._history))
        except Exception:
            return list(self._history)

    def request_history(self) -> list[dict]:
        """Slim request history: rehydrate ONLY bodies the request trim keeps.

        The request path shrink_tool_history re-cuts every OLD spilled body
        back to a receipt — rehydrating all 24 first (53ms) just to re-shrink
        24 (22ms) is pure waste. Trim on the slim receipts, then restore only
        the survivors (measured: 0 of 24 survive). Same bytes sent, ~0 rehydrate.
        Falls back to full rehydrate on any error (never lose content).
        """
        try:
            keep_turns = int(getattr(self.cfg, "trim_keep_turns", 2) or 2)
        except (TypeError, ValueError):
            keep_turns = 2
        try:
            trim_chars = int(getattr(self.cfg, "trim_max_chars", 500) or 500)
        except (TypeError, ValueError):
            trim_chars = 500
        try:
            slim = trim_mod.shrink_tool_history(list(self._history), keep_turns, trim_chars)
            need = any(isinstance(m, dict) and isinstance(m.get("spill_meta"), dict)
                       and not str(m.get("content") or "").startswith(trim_mod.RECEIPT_PREFIX)
                       for m in slim)
            if not need:
                return slim
            return trim_mod.rehydrate_spills(slim)
        except Exception:
            try:
                return trim_mod.rehydrate_spills(list(self._history))
            except Exception:
                return list(self._history)



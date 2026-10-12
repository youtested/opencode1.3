"""Running, streaming and resuming a turn, plus abort and retry timing.

Split out of :mod:`opencode_py.agent.loop` as a mixin; method bodies are
byte-for-byte unchanged. `_run_turn_body` stays in loop.py because it reads the
step/nudge limits from that module's namespace, which tests tune by name."""

from __future__ import annotations

from ..util.log import swallow
import time

from ..plugins import get_dispatcher as _plugins
from ..providers import (
    ContextOverflowError,
    ProviderError,
    RateLimitError,
    StreamInterrupted,
)
from ..providers.base import usage_total

from . import messages as msg_mod
from .loop_config import TurnResult, _memory_capture_enabled


class LoopTurnMixin:
    def close(self) -> None:
        """Release OS resources held by this engine (MCP server processes).

        Called when an engine (or a finished sub-agent) is done; the parent
        keeps its own refcount on shared servers, so a child's close only
        drops the reference the sub-agent acquired.
        """
        try:
            self.task_stop("")
        except Exception as _oc_exc:
            swallow(_oc_exc, "agent/loop_turn.py:34")
            pass
        self._shutdown_tool_pool()
        registry = getattr(self, "registry", None)
        close = getattr(registry, "close", None)
        if close is not None:
            try:
                close()
            except Exception as _oc_exc:
                swallow(_oc_exc, "agent/loop_turn.py:42")
                pass

    def abort(self) -> None:
        """Force-close the active provider stream(s) for this engine and every
        sub-agent it spawned.

        Used right after the shared interrupt flag is flipped so a blocked
        stream read wakes up immediately — even during an idle "thinking" gap
        where no chunk arrives to trigger the per-chunk interrupt check."""
        rotation = getattr(self, "rotation", None)
        if getattr(rotation, "abort", None) is not None:
            try:
                rotation.abort()
            except Exception as _oc_exc:
                swallow(_oc_exc, "agent/loop_turn.py:56")
                pass
        # Force-close any webfetch responses this engine has in flight, so a
        # blocked HTTP read surfaces the interrupt right away instead of letting
        # the fetch run to its timeout after the user pressed ESC.
        registry = getattr(self, "registry", None)
        abort_fetches = getattr(registry, "abort_fetches", None)
        if callable(abort_fetches):
            try:
                abort_fetches()
            except Exception as _oc_exc:
                swallow(_oc_exc, "agent/loop_turn.py:66")
                pass
        abort_servers = getattr(registry, "abort_servers", None)
        if callable(abort_servers):
            try:
                abort_servers()
            except Exception as _oc_exc:
                swallow(_oc_exc, "agent/loop_turn.py:72")
                pass
        # detached workers check the parent flag via their own override, so
        # abort them explicitly too (their streams would otherwise keep
        # burning tokens until the next chunk check).
        try:
            with self._bg_lock:
                for entry in list(self._bg_agents.values()):
                    try:
                        entry.get("stop_flag", {})["stopped"] = True
                    except Exception as _oc_exc:
                        swallow(_oc_exc, "agent/loop_turn.py:82")
                        pass
                    try:
                        rot = getattr((entry.get("bundle") or {}).get("sub"), "rotation", None)
                        if getattr(rot, "abort", None) is not None:
                            rot.abort()
                    except Exception as _oc_exc:
                        swallow(_oc_exc, "agent/loop_turn.py:88")
                        pass
        except Exception as _oc_exc:
            swallow(_oc_exc, "agent/loop_turn.py:90")
            pass
        subagents = getattr(self, "subagents", {}) or {}
        for sub in tuple(subagents.values()):
            try:
                sub.abort()
            except Exception as _oc_exc:
                swallow(_oc_exc, "agent/loop_turn.py:96")
                pass

    def run_turn(self, user_text: str) -> TurnResult:
        plugins = _plugins()
        if plugins is not None and plugins.has("session.start"):
            try:
                plugins.emit("session.start", {"sessionId": self.session_id, "prompt": user_text[:200]})
            except Exception as _oc_exc:
                swallow(_oc_exc, "agent/loop_turn.py:104")
                pass
        # Snapshot the last session's lines BEFORE the capture thread can drain
        # the queue, so the first turn of this session is not blind.
        if not getattr(self, "_recent_taken", False):
            self._recent_taken = True
            self._recent_text = self._peek_recent()
        # the previous session is distilled into durable notes right here, in the
        # background, so the user never waits for it
        if not getattr(self, "_capture_started", False):
            self._capture_started = True
            self._start_memory_capture()
        self._history.append({"role": "user", "content": user_text})
        self._turn_active = True
        try:
            result = self._run_turn_body(user_text)
        finally:
            if plugins is not None and plugins.has("session.end"):
                try:
                    plugins.emit("session.end", {"sessionId": self.session_id})
                except Exception as _oc_exc:
                    swallow(_oc_exc, "agent/loop_turn.py:124")
                    pass
            try:
                self._turn_active = False
            except Exception as _oc_exc:
                swallow(_oc_exc, "agent/loop_turn.py:128")
                pass
        # Credit any note the reply actually reflected, then queue this session
        # for the next launch's capture. Both are instant and local.
        self._after_turn_memory(result)
        return result

    def _after_turn_memory(self, result: TurnResult) -> None:
        """Post-turn memory upkeep: notice which notes were used, then queue
        this session for capture. Never raises — a turn that already succeeded
        must not be spoiled by bookkeeping."""
        if not _memory_capture_enabled(self.cfg):
            return
        try:
            text = getattr(result, "text", "") or ""
            if text.strip():
                from ..tools import remember as _remember

                _remember.mark_used(text)
        except Exception as _oc_exc:
            swallow(_oc_exc, "agent/loop_turn.py:147")
            pass
        try:
            self._queue_session_capture()
        except Exception as _oc_exc:
            swallow(_oc_exc, "agent/loop_turn.py:151")
            pass

    def resume_turn(self) -> TurnResult:
        """Re-run the last user prompt after a failed turn (e.g. the network
        died mid-stream and is back now).

        Drops everything after the last user message (partial assistant text,
        tool results of the dead turn — the model regenerates them cleanly)
        and re-runs WITHOUT appending a duplicate prompt. Returns an error
        result when there is no user message to resume.
        """
        idx = -1
        for i, m in enumerate(self._history):
            if isinstance(m, dict) and m.get("role") == "user":
                content = m.get("content")
                if isinstance(content, str) and content.strip():
                    idx = i
        if idx < 0:
            result = TurnResult()
            result.error = "nothing to resume"
            return result
        user_text = self._history[idx]["content"]
        del self._history[idx + 1:]
        return self._run_turn_body(user_text)

    def _peek_recent(self) -> str:
        """Snapshot the previous session's distilled lines for this turn.

        Must be called BEFORE the capture thread starts: that thread consumes
        the queue immediately, so peeking afterwards would usually find it
        already drained and the first turn would lose the context.
        """
        try:
            from . import memory_capture as _mc

            items = _mc.peek_pending()
        except Exception:
            return ""
        if not items:
            return ""
        return "\n\n".join(str(i.get("text") or "") for i in items)

    def _recent_context(self) -> str:
        """The snapshot taken at the start of the turn. Shown once: by the second
        turn the model pass has decided what is worth storing, and those notes
        are in the memory block instead."""
        return getattr(self, "_recent_text", "") or ""

    def _stream(self, messages, tools, result: TurnResult, system_prompt: str) -> None:
        self._pending_calls = []
        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        # Mid-turn effort switch: an effort change saved while streaming (the
        # thinking popup stays usable mid-turn) rebuilds the lanes HERE, so
        # the very next provider step — retry, failover, or tool-loop step —
        # uses the new effort instead of waiting for the next user turn.
        try:
            if getattr(self, "_rotation_dirty", False):
                self._rotation_dirty = False
                try:
                    self.rebuild_rotation()
                except Exception as e:
                    # Never stream silently on stale lanes: tell the user the
                    # model switch failed (the turn continues on the previous
                    # lanes, now visible).
                    try:
                        self._emit("notice", text=f"model switch failed ({e}); using previous model")
                    except Exception as _oc_exc:
                        swallow(_oc_exc, "agent/loop_turn.py:219")
                        pass
        except Exception as _oc_exc:
            swallow(_oc_exc, "agent/loop_turn.py:221")
            pass
        # The assistant reply being streamed is kept LIVE in self._history as it
        # grows, so the app's periodic autosave (get_history) persists the exact
        # conversation up to the last token even if the app is killed suddenly
        # mid-stream — the session resumes where the user left off.
        live_assistant: dict | None = None

        def _ensure_live_assistant() -> dict:
            nonlocal live_assistant
            if live_assistant is None:
                live_assistant = {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "",
                }
                self._history.append(live_assistant)
            return live_assistant

        content_buf: list[str] = []
        reasoning_buf: list[str] = []

        def _live_append(live: dict, field: str, buf: list, text: str) -> None:
            # Amortized O(1) streaming: buffer chunks in a closure-local list,
            # flush into the live message every 64 chunks. The old
            # `live["content"] += text` copied the whole reply per token
            # (O(n^2) on long answers: each token slower than the last).
            # Buffers stay OUT of the live dict so autosave/sessions on disk
            # never persist internal keys; the settle path flushes exactly.
            buf.append(text)
            if len(buf) >= 64:
                live[field] = (live.get(field) or "") + "".join(buf)
                del buf[:]

        def _drop_live_assistant() -> None:
            nonlocal live_assistant
            if live_assistant is None:
                return
            idx = next((i for i, m in enumerate(self._history) if m is live_assistant), None)
            if idx is not None:
                self._history.pop(idx)
            live_assistant = None

        # Tool-loop requests are rebuilt from raw history (which never holds the
        # system prompt); re-prepend it so every request is well-formed.
        if not messages or messages[0].get("role") != "system":
            messages = self._prepend_system(messages, system_prompt)
        # An interrupted/force-killed turn can leave the request with an
        # assistant message declaring tool_calls but no following tool results;
        # strict backends reject that payload. Repair the local request copy so
        # the provider always sees a well-formed conversation.
        messages = msg_mod.repair_tool_pairs(messages)

        def on_event(evt) -> None:
            kind = evt.kind
            if kind == "text_delta":
                text_parts.append(evt.text)
                _live_append(_ensure_live_assistant(), "content", content_buf, evt.text)
                self._emit("text_delta", text=evt.text)
            elif kind == "reasoning_delta":
                reasoning_parts.append(evt.text)
                _live_append(_ensure_live_assistant(), "reasoning_content", reasoning_buf, evt.text)
                self._emit("reasoning_delta", text=evt.text)
            elif kind == "tool_call":
                for tc in evt.tool_calls or []:
                    self._pending_calls.append(
                        {
                            "id": tc.id,
                            "name": tc.name,
                            "arguments": tc.arguments,
                        }
                    )
            elif kind == "usage":
                u = evt.usage
                # Context-window usage = the last completion's full footprint
                # (official: total || input+output+cache.read+cache.write).
                # Summing across streams would double-count history on
                # multi-step tool loops.
                total = usage_total(u)
                self._usage_total["input_tokens"] = u.input_tokens
                self._usage_total["output_tokens"] = u.output_tokens
                self._usage_total["total_tokens"] = total
                result.usage = dict(self._usage_total)
                usage_evt = dict(self._usage_total)
                ctx = self._model_context_size()
                if ctx:
                    result.usage["context_size"] = ctx
                    usage_evt["context_size"] = ctx
                self._emit("usage", usage=usage_evt)
            elif kind == "error":
                result.error = evt.error
                self._emit("error", error=evt.error)
            elif kind == "done":
                result.finish_reason = getattr(evt, "finish_reason", "") or result.finish_reason

        try:
            # Automatic retry for transient failures (streaming timeouts, 5xx,
            # overload, short rate limits) — mirrors upstream opencode's retry
            # policy, which waits out transient errors with exponential backoff
            # instead of surfacing them. Mirrors the official behavior:
            #   BASE_DELAY_MS=500, factor 2, jitter ±20%, cap 10s,
            #   honoring Retry-After when the provider sends it.
            # Events stay buffered inside rotation until a lane completes, so a
            # failed attempt emits nothing visible and any retry is clean.
            # Smart resume: if the model was mid-operation (it already ran
            # tools this turn), a plain re-send would replay every tool call
            # from scratch. Instead nudge the SAME model to continue; if
            # nothing happened yet, re-send the prompt as-is.
            for attempt in range(self.cfg.auto_retry_count + 1):
                try:
                    provider_id, model_id = self.rotation.stream(
                        self._retry_messages(messages, result, attempt),
                        tools,
                        on_event,
                        on_notice=self._on_rotated,
                        is_interrupted=self.interrupt,
                        locked=self.rotation_locked,
                    )
                    break
                except ContextOverflowError:
                    # has its own compaction recovery path below — never retry
                    raise
                except RateLimitError as e:
                    # Burst 429s (Zen free tier sends NO Retry-After header —
                    # live-verified: retry-after=None on every 429) are
                    # transient: the official client waits 2s/4s/8s... and
                    # retries the SAME model up to 5 times instead of dying
                    # mid-chat. Mirror that: backoff (honors Retry-After when
                    # present, else exponential) and only surface after the
                    # full retry budget is spent.
                    last_error = e
                    if not self.cfg.auto_retry or attempt >= self.cfg.auto_retry_count:
                        raise
                    if self.interrupt():
                        raise
                    remaining = self.cfg.auto_retry_count - attempt
                    self._emit(
                        "retry",
                        attempt=attempt + 1,
                        total=self.cfg.auto_retry_count,
                        message=f"↻ rate limited — retrying ({remaining} left)…",
                    )
                    self._sleep_interruptible(self._retry_delay(e, attempt))
                except ProviderError as e:
                    last_error = e
                    if not self.cfg.auto_retry or not e.retryable or attempt >= self.cfg.auto_retry_count:
                        raise
                    if self.interrupt():
                        raise
                    # Like official's session.status retry event: show the
                    # provider's own cause plus the attempt, not one static
                    # string for every failure kind.
                    remaining = self.cfg.auto_retry_count - attempt
                    cause = (getattr(e, "message", "") or str(e)).split("\n")[0][:100]
                    self._emit(
                        "retry",
                        attempt=attempt + 1,
                        total=self.cfg.auto_retry_count,
                        message=f"↻ {cause} — retrying ({remaining} left)…" if cause else f"↻ retrying ({remaining} left)…",
                    )
                    self._sleep_interruptible(self._retry_delay(e, attempt))
            else:
                # loop exhausted without success (the `break` never ran)
                raise last_error
            result.provider_id = provider_id
            result.model_id = model_id
        except ContextOverflowError as e:
            # The history overflowed the model's window even after budget
            # trimming (estimates are cheap). Recover by summarizing the
            # conversation into an anchored summary and keeping the recent tail
            # verbatim (upstream opencode's compaction), then retry once. This
            # fixes the turn instead of surfacing a hard error mid-conversation.
            if self.cfg.compaction_enabled:
                compacted = self._compact_context(system_prompt)
            else:
                compacted = None
            if compacted is not None:
                messages = compacted
                try:
                    provider_id, model_id = self.rotation.stream(
                        messages, tools, on_event, on_notice=self._on_rotated, is_interrupted=self.interrupt,
                        locked=self.rotation_locked,
                    )
                    result.provider_id = provider_id
                    result.model_id = model_id
                    result.error = ""
                except ContextOverflowError as e2:
                    result.error = f"context overflow (even after compaction): {e2}"
                    self._emit("error", error=result.error, retryable=True)
                    if not result.usage:
                        self._emit_request_usage(messages, tools)
                except ProviderError as e2:
                    result.error = str(e2)
                    self._emit("error", error=result.error, retryable=bool(e2.retryable))
                    if not result.usage:
                        self._emit_request_usage(messages, tools)
                except StreamInterrupted:
                    self._emit_request_usage(messages, tools, output_text="".join(text_parts) + "".join(reasoning_parts))
                    raise
                except Exception as e2:
                    result.error = str(e2)
                    self._emit("error", error=result.error)
                    if not result.usage:
                        self._emit_request_usage(messages, tools)
                if result.error:
                    pass  # surfaced above
            else:
                result.error = f"context overflow: {e}"
                self._emit("error", error=result.error, retryable=True)
                if not result.usage:
                    self._emit_request_usage(messages, tools)
        except RateLimitError as e:
            result.error = f"rate limit: {e}"
            self._emit("error", error=result.error, retryable=True)
            if not result.usage:
                self._emit_request_usage(messages, tools)
        except ProviderError as e:
            result.error = str(e)
            result.network_failed = bool(getattr(e, "network", False))
            self._emit("error", error=result.error, retryable=bool(e.retryable))
            if not result.usage:
                self._emit_request_usage(messages, tools)
        except StreamInterrupted:
            # User aborted mid-stream (Esc / Ctrl+C): NOT an
            # error — end the turn as interrupted. Partial text already
            # streamed stays on screen, like upstream opencode's abort.
            # The request WAS fully sent, so count it; otherwise the footer
            # freezes on the older lower number as if nothing was spent.
            self._emit_request_usage(messages, tools, output_text="".join(text_parts) + "".join(reasoning_parts))
            self._emit("interrupted")
        except KeyboardInterrupt:
            raise
        except Exception as e:
            result.error = str(e)
            self._emit("error", error=result.error)

        result.text = "".join(text_parts)
        result.reasoning = "".join(reasoning_parts)
        if live_assistant is not None:
            # flush buffered live chunks (2.2 O(1) streaming keeps up to 63
            # unflushed) so history/autosave see the exact final text
            for _field, _buf in (("content", content_buf), ("reasoning_content", reasoning_buf)):
                try:
                    if _buf:
                        live_assistant[_field] = (live_assistant.get(_field) or "") + "".join(_buf)
                        del _buf[:]
                except Exception as _oc_exc:
                    swallow(_oc_exc, "agent/loop_turn.py:467")
                    pass
        if (
            not self._pending_calls
            and not result.text
            and result.finish_reason == "length"
            and not result.error
        ):
            # Thinking burned the whole output budget before any answer text.
            # Without this the turn ends silently right after the visible
            # thinking phase — the reported "stops in the thinking part".
            _drop_live_assistant()
            result.error = (
                "the model hit its output-token limit before answering"
                " (finish_reason=length) — usually right after a long thinking"
                " phase. Retry, split the task smaller, or pick a lane with a"
                " larger output limit."
            )
            self._emit("error", error=result.error)
        if self._pending_calls:
            # run_turn builds ONE assistant message carrying text + reasoning +
            # tool_calls from result.text/result.reasoning — drop the live
            # text-only message first so one assistant turn isn't split (strict
            # backends and reasoning models lose thread on the next step).
            _drop_live_assistant()
        elif result.text:
            # keep the live message (it already holds every streamed delta);
            # sync it to the final buffers in case no delta path weirdly fired
            if live_assistant is None:
                _ensure_live_assistant()
            live_assistant["content"] = result.text
            live_assistant["reasoning_content"] = result.reasoning
        elif result.reasoning:
            # reasoning-only reply: replay it faithfully as what the model
            # actually returned — empty content plus a separate reasoning
            # signal. Storing the thinking also as `content` produces a message
            # that ever after disagrees with the model's real output, which
            # strict thinking-mode backends reject on the next request
            # ("reasoning_content must be passed back").
            if live_assistant is None:
                _ensure_live_assistant()
            live_assistant["content"] = ""
            live_assistant["reasoning_content"] = result.reasoning
        elif result.error:
            # nothing streamed — don't leave an empty assistant message behind
            _drop_live_assistant()
        else:
            # keep role alternation valid and avoid an empty-content message
            self._history.append(
                {"role": "assistant", "content": "(no response)", "reasoning_content": ""}
            )

    def _retry_messages(self, messages: list, result: TurnResult, attempt: int) -> list:
        """Rebuild the request for a retry on the same model.

        - If the turn already made progress (tools executed), the model was
          mid-operation: don't replay from scratch — append a short continuation
          nudge so it picks up where it was cut off. Each retry uses the next
          nudge in the list.
        - After every nudge is used, fall back to re-sending the full context
          unchanged: a fresh continuation beats piling on more "keep going"
          noise (which reasoning models read as a mid-thought interrupt and
          answer with nothing).
        - If nothing has happened yet, re-send the original prompt unchanged.
        """
        nudges = self._RETRY_NUDGES
        if result.tool_calls_made > 0 and attempt > 0:
            if attempt <= len(nudges):
                nudge = nudges[(attempt - 1) % len(nudges)]
                return [*messages, {"role": "user", "content": nudge}]
            return list(messages)
        return list(messages)

    def _sleep_interruptible(self, seconds: float) -> None:
        # Backoff that wakes the moment ESC is pressed: poll the interrupt
        # flag in 0.1s slices instead of one blocking time.sleep (which left
        # the app looking frozen for up to 30s). Raises StreamInterrupted so
        # the turn ends as interrupted, not as a retry.
        remaining = max(0.0, float(seconds or 0.0))
        while remaining > 0:
            if self.interrupt():
                from ..providers import StreamInterrupted as _SI

                raise _SI()
            step = 0.1 if remaining > 0.1 else remaining
            time.sleep(step)
            remaining -= step

    def _retry_delay(self, error: Exception, attempt: int) -> float:
        """Backoff before a retry, mirroring upstream opencode's retry policy
        (session/retry.ts): 2s * 2^attempt plus up to +25% additive jitter.

        Honors ``Retry-After`` (seconds) when the provider sent it — header
        waits bypass the idle cap, like official. Without headers the wait is
        capped at 30s.
        """
        import random

        retry_after = getattr(error, "retry_after", None)
        if retry_after is not None:
            try:
                return max(float(retry_after), 0.0)
            except (TypeError, ValueError) as _oc_exc:
                swallow(_oc_exc, "opencode_py/agent/loop_turn.py:587")
                pass
        base = 2.0 * (2**attempt)
        cap = min(base, 30.0)
        return round(cap * (1.0 + random.random() * 0.25), 3)

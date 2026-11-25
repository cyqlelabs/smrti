"""The engine every decision point calls: policy, provider, cache, audit.

``decide`` is the one door. It answers ``None`` whenever there is no
decision to act on — the task is off, no provider is configured, the
provider failed or timed out, the reply was invalid — and every caller
treats ``None`` as "do what you did before decisions existed". A caller
never sees an exception from here and never has to know which of those
happened; the audit log does.

Shadow mode goes through the same door: the provider is asked and the
record filed, and the caller is told the decision was not applied
(``DecisionOutcome.applied`` is false), so it computes what it *would* have
done, records that as the outcome, and does the deterministic thing.

The cache is keyed on everything that could change the answer — provider,
model, task, the whole state, the whole question set — so a state that
names the atoms it was built from and their ``updated_at`` invalidates
itself when one of them changes, and a forgotten atom never enters a state
in the first place.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Mapping

from . import audit
from .policies import MODE_ACTIVE, MODE_OFF, DecisionPolicy
from .provider import (
    DecisionBusy,
    DecisionInvalid,
    DecisionNotReady,
    DecisionProvider,
    DecisionTimeout,
    DecisionUnavailable,
    DecisionUnsupported,
    Decisions,
    Question,
    State,
)

logger = logging.getLogger("smrti.decisions")


@dataclass
class DecisionOutcome:
    """A provider's answers plus the mode they were asked under."""

    decisions: Decisions
    mode: str
    cached: bool = False

    @property
    def applied(self) -> bool:
        """Whether the caller may act on the answers."""
        return self.mode == MODE_ACTIVE


class DecisionEngine:
    def __init__(
        self,
        policy: DecisionPolicy,
        provider: DecisionProvider | None = None,
    ) -> None:
        self.policy = policy
        self.provider = provider
        self._cache: OrderedDict[str, Decisions] = OrderedDict()
        self._cache_lock = threading.Lock()
        # When the provider may be asked again after failing. A provider that
        # just could not answer is unlikely to answer the next caller either,
        # and the deadline is paid by the request the caller is waiting on, so
        # asking every time turns one unavailable model into a fixed delay in
        # front of every recall. See ``_offline``/``_cool``. Two scopes: a
        # provider that is down (a failed load, an unreachable server, a model
        # that ran out of time with the worker to itself) is not asked for
        # anything; one that answered a task's question badly is not asked
        # that task's questions. One malformed rerank reply used to silence
        # tone, supersession, entity and routing for a minute.
        self._retry_at = 0.0
        self._task_retry_at: dict[str, float] = {}
        self._retry_lock = threading.Lock()
        self._unanswered: set[str] = set()
        self._unanswered_lock = threading.Lock()

    # ── policy passthrough ───────────────────────────────────────────────

    def mode(self, task: str) -> str:
        if self.provider is None:
            return MODE_OFF
        return self.policy.mode(task)

    def enabled(self, task: str) -> bool:
        return self.mode(task) != MODE_OFF and self._answers(task)

    def _answers(self, task: str) -> bool:
        """Whether the provider has a head for the task. Most answer
        everything; a student names its list, and a task off it is decided
        the way it was before decisions existed — said once, since the
        alternative is a refused request per mention."""
        answers = getattr(self.provider, "answers", None)
        if answers is None:
            return True
        try:
            if answers(task):
                return True
        except Exception:  # a provider's own probe failing is not a verdict
            return True
        with self._unanswered_lock:
            if task not in self._unanswered:
                self._unanswered.add(task)
                logger.info("the %s model answers no %s question; deciding it the old way", self.provider_name, task)
        return False

    @property
    def offline(self) -> bool:
        """Whether the provider is inside its cooldown after failing as a
        whole, which stops every task."""
        return self._offline()

    def offline_for(self, task: str) -> bool:
        """Whether *task* is not being asked: the provider is offline, or
        this task's own cooldown is open after a reply it could not use."""
        return self._offline(task)

    def active(self, task: str) -> bool:
        return self.mode(task) == MODE_ACTIVE

    @property
    def provider_name(self) -> str:
        return getattr(self.provider, "name", "none") if self.provider else "none"

    @property
    def model(self) -> str:
        return getattr(self.provider, "model", "") if self.provider else ""

    # ── cache ────────────────────────────────────────────────────────────

    def _key(self, task: str, state: State, questions: Mapping[str, Question]) -> str:
        payload = json.dumps(
            {
                "provider": self.provider_name,
                "model": self.model,
                "task": task,
                "state": state,
                "questions": {k: q.payload() for k, q in questions.items()},
            },
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _cached(self, key: str) -> Decisions | None:
        if self.policy.cache_size <= 0:
            return None
        with self._cache_lock:
            hit = self._cache.get(key)
            if hit is not None:
                self._cache.move_to_end(key)
            return hit

    def _store(self, key: str, decisions: Decisions) -> None:
        if self.policy.cache_size <= 0:
            return
        with self._cache_lock:
            self._cache[key] = decisions
            self._cache.move_to_end(key)
            while len(self._cache) > self.policy.cache_size:
                self._cache.popitem(last=False)

    # ── deciding ─────────────────────────────────────────────────────────

    def _record(
        self, task: str, mode: str, tenant_id: str, space: str, *, outcome: str, applied: bool,
        decisions: Decisions | None = None, cached: bool = False, error: str | None = None,
        summary: dict[str, Any] | None = None,
    ) -> None:
        audit.record(
            task=task,
            mode=mode,
            tenant_id=tenant_id,
            space=space,
            provider=self.provider_name,
            model=(decisions.model if decisions and decisions.model else self.model),
            outcome=outcome,
            applied=applied,
            latency_ms=decisions.latency_ms if decisions else 0.0,
            input_tokens=decisions.input_tokens if decisions else 0,
            output_tokens=decisions.output_tokens if decisions else 0,
            cached=cached,
            error=error,
            summary=summary,
            answers=decisions.compact() if decisions else None,
        )

    def _mirror(self, task: str, tenant_id: str, state: State, questions: Mapping[str, Question],
                decisions: Decisions | None, error: str | None, started: float) -> None:
        """Copy the call into the shared LLM call log, so the visualizer's
        debug tab shows decisions beside extraction calls."""
        try:
            from smrti.call_log import append as _log

            _log({
                "kind": "decision",
                "subkind": task,
                "tenant_id": tenant_id,
                "upstream": getattr(self.provider, "base_url", self.provider_name),
                "model": decisions.model if decisions and decisions.model else self.model,
                "source": self.provider_name,
                "request": {"state": state, "questions": {k: q.payload() for k, q in questions.items()}},
                "status": 200 if decisions else 0,
                "response_raw": "",
                "response_parsed": decisions.compact() if decisions else None,
                "error": error,
                "duration_ms": round((time.monotonic() - started) * 1000, 1),
            })
        except Exception:  # the log is a convenience, never a failure
            logger.debug("could not mirror a decision into the call log", exc_info=True)

    def _offline(self, task: str | None = None) -> bool:
        """Whether the provider, or *task* on it, is inside its cooldown."""
        if self.policy.cooldown <= 0:
            return False
        now = time.monotonic()
        with self._retry_lock:
            if now < self._retry_at:
                return True
            return task is not None and now < self._task_retry_at.get(task, 0.0)

    def _cool(self, task: str | None) -> None:
        """Stop asking for a while: the whole provider when *task* is None,
        otherwise that task's questions only."""
        until = time.monotonic() + max(0.0, self.policy.cooldown)
        with self._retry_lock:
            if task is None:
                self._retry_at = until
            else:
                self._task_retry_at[task] = until

    def _answered(self, task: str) -> None:
        """The provider answered *task*: whatever cooldown it was under is over."""
        with self._retry_lock:
            self._retry_at = 0.0
            self._task_retry_at.pop(task, None)

    def decide(
        self,
        task: str,
        state: State,
        questions: Mapping[str, Question],
        *,
        tenant_id: str,
        space: str,
        timeout: float | None = None,
        partial: bool = False,
    ) -> DecisionOutcome | None:
        """Ask the provider the task's questions about *state*.

        ``None`` when the task is off or nothing could be decided. The
        caller files the outcome it drew through :meth:`conclude`.

        *partial* says *timeout* is what is left of a budget the caller is
        spending across several calls, after the provider has already
        answered inside it: the rerank walking its shortlist. A provider
        that runs out of that slice is a budget spent, filed as such, and
        not a provider down — which is what a timeout otherwise says, and
        it opened the cooldown that silenced every other task for a minute
        at the end of one walk in three.
        """
        mode = self.mode(task)
        if mode == MODE_OFF or self.provider is None:
            return None
        key = self._key(task, state, questions)
        hit = self._cached(key)
        if hit is not None:
            return DecisionOutcome(decisions=hit, mode=mode, cached=True)
        if self._offline(task):
            # No record and no log line: the failure that opened the cooldown
            # filed both, and one entry per skipped call would bury it.
            return None
        started = time.monotonic()
        try:
            decisions = self.provider.ask(
                state, questions, timeout=self.policy.timeout if timeout is None else timeout
            )
        except Exception as exc:
            self._failed(exc, task, mode, tenant_id, space, state, questions, started, partial)
            return None
        self._answered(task)
        self._store(key, decisions)
        self._mirror(task, tenant_id, state, questions, decisions, None, started)
        return DecisionOutcome(decisions=decisions, mode=mode)

    def _failed(
        self, exc: Exception, task: str, mode: str, tenant_id: str, space: str,
        state: State, questions: Mapping[str, Question], started: float, partial: bool,
    ) -> None:
        """File a question the provider did not answer, and decide how long
        to stop asking — and whom: nobody, this task, or the whole provider."""
        error = str(exc) if isinstance(exc, DecisionUnavailable) else repr(exc)
        if isinstance(exc, DecisionUnsupported):
            # The provider is fine; this question is not one it answers.
            outcome = "unsupported"
        elif isinstance(exc, DecisionNotReady):
            # Ready in a moment: this caller falls back, the next one asks.
            outcome = "loading"
            logger.debug("decision %s skipped while the model loads", task)
        elif isinstance(exc, DecisionBusy) or (partial and isinstance(exc, DecisionTimeout)):
            # The caller's own budget ran out — on a queue behind other
            # callers, or on the last slice of a walk — and the provider is
            # answering everyone else.
            outcome = "budget"
            logger.debug("decision %s ran out of its budget: %s", task, exc)
        elif isinstance(exc, DecisionInvalid) or not isinstance(exc, DecisionUnavailable):
            # An answer this task cannot use, or a provider bug met on this
            # task's question: the other tasks' questions may be fine.
            outcome = "unavailable"
            self._cool(task)
            if isinstance(exc, DecisionUnavailable):
                logger.warning("decision %s unavailable: %s", task, exc)
            else:
                logger.warning("decision %s failed: %r", task, exc, exc_info=True)
        else:
            outcome = "unavailable"
            self._cool(None)
            logger.warning("decision %s unavailable: %s", task, exc)
        self._record(task, mode, tenant_id, space, outcome=outcome, applied=False, error=error)
        self._mirror(task, tenant_id, state, questions, None, error, started)

    async def decide_async(
        self,
        task: str,
        state: State,
        questions: Mapping[str, Question],
        *,
        tenant_id: str,
        space: str,
        timeout: float | None = None,
        partial: bool = False,
    ) -> DecisionOutcome | None:
        mode = self.mode(task)
        if mode == MODE_OFF or self.provider is None:
            return None
        key = self._key(task, state, questions)
        hit = self._cached(key)
        if hit is not None:
            return DecisionOutcome(decisions=hit, mode=mode, cached=True)
        if self._offline(task):
            # No record and no log line: the failure that opened the cooldown
            # filed both, and one entry per skipped call would bury it.
            return None
        started = time.monotonic()
        try:
            decisions = await self.provider.ask_async(
                state, questions, timeout=self.policy.timeout if timeout is None else timeout
            )
        except Exception as exc:
            self._failed(exc, task, mode, tenant_id, space, state, questions, started, partial)
            return None
        self._answered(task)
        self._store(key, decisions)
        self._mirror(task, tenant_id, state, questions, decisions, None, started)
        return DecisionOutcome(decisions=decisions, mode=mode)

    def conclude(
        self,
        task: str,
        outcome: DecisionOutcome,
        *,
        tenant_id: str,
        space: str,
        result: str,
        applied: bool,
        summary: dict[str, Any] | None = None,
    ) -> None:
        """File what the caller concluded from the answers and whether it
        acted on them."""
        self._record(
            task, outcome.mode, tenant_id, space,
            outcome=result, applied=applied and outcome.applied,
            decisions=outcome.decisions, cached=outcome.cached, summary=summary,
        )

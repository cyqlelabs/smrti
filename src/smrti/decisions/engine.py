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
from .provider import DecisionProvider, DecisionUnavailable, DecisionUnsupported, Decisions, Question, State

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
        # front of every recall. See ``_offline``/``_note``.
        self._retry_at = 0.0
        self._retry_lock = threading.Lock()

    # ── policy passthrough ───────────────────────────────────────────────

    def mode(self, task: str) -> str:
        if self.provider is None:
            return MODE_OFF
        return self.policy.mode(task)

    def enabled(self, task: str) -> bool:
        return self.mode(task) != MODE_OFF

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

    def invalidate(self) -> None:
        """Drop every cached decision — after a forget or a bulk change a
        caller cannot name in a state key."""
        with self._cache_lock:
            self._cache.clear()

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

    def _offline(self) -> bool:
        """Whether the provider is inside its cooldown after a failure."""
        if self.policy.cooldown <= 0:
            return False
        with self._retry_lock:
            return bool(self._retry_at) and time.monotonic() < self._retry_at

    def _note(self, ok: bool) -> None:
        """Record that the provider answered, or did not."""
        with self._retry_lock:
            self._retry_at = 0.0 if ok else time.monotonic() + max(0.0, self.policy.cooldown)

    def decide(
        self,
        task: str,
        state: State,
        questions: Mapping[str, Question],
        *,
        tenant_id: str,
        space: str,
        timeout: float | None = None,
    ) -> DecisionOutcome | None:
        """Ask the provider the task's questions about *state*.

        ``None`` when the task is off or nothing could be decided. The
        caller files the outcome it drew through :meth:`conclude`.
        """
        mode = self.mode(task)
        if mode == MODE_OFF or self.provider is None:
            return None
        key = self._key(task, state, questions)
        hit = self._cached(key)
        if hit is not None:
            return DecisionOutcome(decisions=hit, mode=mode, cached=True)
        if self._offline():
            # No record and no log line: the failure that opened the cooldown
            # filed both, and one entry per skipped call would bury it.
            return None
        started = time.monotonic()
        try:
            decisions = self.provider.ask(
                state, questions, timeout=self.policy.timeout if timeout is None else timeout
            )
        except DecisionUnsupported as exc:
            # The provider is fine; this question is not one it answers.
            self._record(task, mode, tenant_id, space, outcome="unsupported", applied=False, error=str(exc))
            self._mirror(task, tenant_id, state, questions, None, str(exc), started)
            return None
        except DecisionUnavailable as exc:
            self._note(False)
            self._record(task, mode, tenant_id, space, outcome="unavailable", applied=False, error=str(exc))
            self._mirror(task, tenant_id, state, questions, None, str(exc), started)
            logger.warning("decision %s unavailable: %s", task, exc)
            return None
        except Exception as exc:  # a provider bug is not the engine's failure to bear
            self._note(False)
            self._record(task, mode, tenant_id, space, outcome="unavailable", applied=False, error=repr(exc))
            self._mirror(task, tenant_id, state, questions, None, repr(exc), started)
            logger.warning("decision %s failed: %r", task, exc, exc_info=True)
            return None
        self._note(True)
        self._store(key, decisions)
        self._mirror(task, tenant_id, state, questions, decisions, None, started)
        return DecisionOutcome(decisions=decisions, mode=mode)

    async def decide_async(
        self,
        task: str,
        state: State,
        questions: Mapping[str, Question],
        *,
        tenant_id: str,
        space: str,
        timeout: float | None = None,
    ) -> DecisionOutcome | None:
        mode = self.mode(task)
        if mode == MODE_OFF or self.provider is None:
            return None
        key = self._key(task, state, questions)
        hit = self._cached(key)
        if hit is not None:
            return DecisionOutcome(decisions=hit, mode=mode, cached=True)
        if self._offline():
            # No record and no log line: the failure that opened the cooldown
            # filed both, and one entry per skipped call would bury it.
            return None
        started = time.monotonic()
        try:
            decisions = await self.provider.ask_async(
                state, questions, timeout=self.policy.timeout if timeout is None else timeout
            )
        except DecisionUnsupported as exc:
            self._record(task, mode, tenant_id, space, outcome="unsupported", applied=False, error=str(exc))
            self._mirror(task, tenant_id, state, questions, None, str(exc), started)
            return None
        except DecisionUnavailable as exc:
            self._note(False)
            self._record(task, mode, tenant_id, space, outcome="unavailable", applied=False, error=str(exc))
            self._mirror(task, tenant_id, state, questions, None, str(exc), started)
            logger.warning("decision %s unavailable: %s", task, exc)
            return None
        except Exception as exc:
            self._note(False)
            self._record(task, mode, tenant_id, space, outcome="unavailable", applied=False, error=repr(exc))
            self._mirror(task, tenant_id, state, questions, None, repr(exc), started)
            logger.warning("decision %s failed: %r", task, exc, exc_info=True)
            return None
        self._note(True)
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

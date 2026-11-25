"""What every in-process decision model shares: a lazy load on a daemon
thread, one inference worker, deadlines, and usage accounting.

**The model is never loaded on a caller's thread.** Loading means mapping a
graph and, on a machine that has not got the weights, fetching them first —
seconds to minutes of work behind a decision that exists to save
milliseconds. A caller that arrives before the model is ready is told so
immediately (``DecisionUnavailable``) and falls back to the deterministic
path, which is what every decision site already does with that answer. The
load runs once, on a daemon thread, and the first caller after it finishes
gets the model.

Paying it on the caller instead is what a recall costs when it goes wrong:
the load sits behind the decision deadline, every recall waits the whole
deadline because nothing remembers the last one already did, and a
30-second wall appears in front of a retrieval that takes 48ms. A failed
load then backs off rather than being retried per request, because each
attempt may re-try a 250 MB download before it can fail.

A subclass supplies ``_load_agent`` — anything with
``predict(state, questions_payload) -> {"answers", "usage"}`` — and says how
many questions ride one native call.
"""
from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from concurrent.futures import Future, TimeoutError as FutureTimeout
from typing import Any, Callable, Mapping

from .provider import (
    DecisionBusy,
    DecisionInvalid,
    DecisionNotReady,
    DecisionTimeout,
    DecisionUnavailable,
    Decisions,
    Question,
    State,
    parse_response,
    questions_payload,
)

logger = logging.getLogger("smrti.decisions.local")

# How long a failed load is left alone before another is attempted. A failing
# attempt may re-try a 250 MB download before it can fail, so retrying per
# request costs far more than the decisions are worth.
LOAD_RETRY_SECONDS = 900.0


def _tokens(usage: Mapping[str, Any], name: str) -> int:
    try:
        return int(usage.get(name, 0) or 0)
    except (TypeError, ValueError):
        return 0


class _Worker:
    """One daemon thread that runs the calls submitted to it in turn.

    A ``ThreadPoolExecutor`` would serialize them as well, but its workers
    are not daemons: the interpreter joins them at exit, so one inference
    that hangs held the process open for fifteen seconds after the decision
    that started it had already given up. Nothing a decision is doing is
    worth keeping a process alive for.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._closed = False

    def submit(self, fn: Callable[..., Any], *args: Any) -> Future:
        future: Future = Future()
        with self._lock:
            if self._closed:
                raise DecisionUnavailable("the decision provider has been closed")
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
                self._thread.start()
            self._queue.put((future, fn, args))
        return future

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            future, fn, args = item
            # False when the caller cancelled it while it waited: skip it.
            if not future.set_running_or_notify_cancel():
                continue
            try:
                result = fn(*args)
            except BaseException as exc:
                future.set_exception(exc)
            else:
                future.set_result(result)

    def close(self) -> None:
        """Cancel what is still queued and let the thread end; a call already
        running cannot be interrupted and finishes on its own."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is not None:
                item[0].cancel()
        self._queue.put(None)


class LocalProvider:
    """A lazy, process-local model with bounded, serialized calls."""

    name = "local"
    # How many questions ride in one native call. A model that reads the
    # state once per question wants 1 (see :mod:`laya`); one that answers a
    # whole head in one pass wants them all.
    questions_per_call = 1

    def __init__(self, model: str = "") -> None:
        self.model = model
        self._agent: Any | None = None
        self._worker = _Worker(f"smrti-{self.name}")
        # When the one worker is running a prediction, this is when it
        # started; 0.0 while it is idle. Read by ``_refuse_if_overrun`` to
        # tell a busy model from one that has stopped answering.
        self._running_since = 0.0
        self._accounting_lock = threading.Lock()
        # Calls submitted to the worker and not yet settled. A call that
        # found another ahead of it shares the worker, and a deadline it
        # spends waiting is its own budget, not the model failing.
        self._inflight = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.requests = 0
        # Load state, guarded by ``_load_lock``. The loader runs on its own
        # daemon thread rather than on the inference worker: a checkpoint
        # download must not sit in front of the inference queue.
        self._load_lock = threading.Lock()
        self._loader: threading.Thread | None = None
        self._load_error: str | None = None
        self._retry_at = 0.0

    def _load_agent(self) -> Any:
        """Import the runtime and map the graph; return the agent. Runs on
        the loader thread. Sets ``self.model`` to what was loaded."""
        raise NotImplementedError

    def _load(self) -> Any:
        """The agent: the one already held, else a fresh load."""
        if self._agent is not None:
            return self._agent
        return self._load_agent()

    def _run_load(self) -> None:
        started = time.monotonic()
        try:
            self._agent = self._load()
        except Exception as exc:
            with self._load_lock:
                self._load_error = str(exc)
                self._retry_at = time.monotonic() + LOAD_RETRY_SECONDS
            logger.warning(
                "the decision model %r could not be loaded (%s); decisions fall back to the "
                "deterministic path and the load is retried in %.0f minutes",
                self.model, exc, LOAD_RETRY_SECONDS / 60,
            )
        else:
            with self._load_lock:
                self._load_error = None
            logger.info(
                "the decision model %r is ready after %.1fs; decisions are live",
                self.model, time.monotonic() - started,
            )
        finally:
            with self._load_lock:
                self._loader = None

    def _begin_load(self) -> None:
        """Start the load in the background, at most one at a time."""
        with self._load_lock:
            if self._agent is not None or self._loader is not None:
                return
            if self._retry_at and time.monotonic() < self._retry_at:
                return
            self._loader = threading.Thread(
                target=self._run_load, name=f"smrti-{self.name}-load", daemon=True
            )
            thread = self._loader
        thread.start()

    def _require_ready(self) -> Any:
        """The loaded agent, or ``DecisionUnavailable`` without waiting for one.

        A decision is worth a few milliseconds of a recall and nothing like
        the minutes a first load can take, so the caller is never made to
        wait for it: it is told the model is not ready and takes the path it
        had before decisions existed. A load still running is
        ``DecisionNotReady``, which the engine does not hold against the
        provider; a load that failed is ``DecisionUnavailable``, which it does.
        """
        agent = self._agent
        if agent is not None:
            return agent
        self._begin_load()
        with self._load_lock:
            error, loading = self._load_error, self._loader is not None
        if error is not None and not loading:
            raise DecisionUnavailable(f"the decision model is not available: {error}")
        raise DecisionNotReady("the decision model is still loading")

    @property
    def ready(self) -> bool:
        """Whether a decision asked right now would reach the model."""
        return self._agent is not None

    def preload(self, timeout: float | None = None) -> bool:
        """Load now rather than on the first decision.

        Blocking is the point here — this is the explicit "get ready" call, so
        a server can make the first recall a fast one — but it is bounded, and
        it reports whether the model came up instead of raising.
        """
        if self._agent is not None:
            return True
        self._begin_load()
        with self._load_lock:
            thread = self._loader
        if thread is not None:
            thread.join(timeout)
        return self._agent is not None

    def _predict(
        self,
        state: State,
        questions: Mapping[str, Question],
        stop_at: float | None = None,
    ) -> Any:
        if not questions:
            raise ValueError("a decision needs at least one question")
        agent = self._require_ready()
        items = list(questions.items())
        answers: dict[str, Any] = {}
        input_tokens = 0
        output_tokens = 0
        per_call = self.questions_per_call
        self._running_since = time.monotonic()
        try:
            for start in range(0, len(items), per_call):
                if start and stop_at is not None and time.monotonic() >= stop_at:
                    # Nobody is waiting for this any more. Every question has
                    # to be answered for the reply to parse, so there is
                    # nothing here worth salvaging — and the whole point of
                    # stopping is to not spend the next chunk's memory and
                    # cores on an answer that will never be read.
                    raise DecisionTimeout(
                        f"the decision model answered {len(answers)} of {len(items)} "
                        "questions before the deadline"
                    )
                chunk = dict(items[start : start + per_call])
                try:
                    piece = agent.predict(state, questions_payload(chunk))
                except DecisionUnavailable:
                    raise
                except Exception as exc:
                    raise DecisionInvalid(f"decision inference failed: {exc}") from exc
                if not isinstance(piece, Mapping):
                    raise DecisionInvalid("reply is not an object")
                part = piece.get("answers")
                if isinstance(part, Mapping):
                    answers.update(part)
                usage = piece.get("usage")
                if isinstance(usage, Mapping):
                    input_tokens += _tokens(usage, "input_tokens")
                    output_tokens += _tokens(usage, "output_tokens")
        finally:
            self._running_since = 0.0
        return {
            "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
            # The runtime reports the generic architecture name. The model
            # actually loaded is the useful identity for audit and cache.
            "model": self.model,
        }

    def _refuse_if_overrun(self, deadline: float) -> None:
        """Fail now rather than queue behind a call already later than this.

        Inference is serialized on one worker and a running call cannot be
        cancelled — ``Future.cancel`` does nothing once the work has started.
        So a caller that queues behind one which has already run longer than
        this caller's whole deadline spends that deadline in a wait that
        cannot succeed, and so does every caller behind it. The deterministic
        path is right there and costs nothing. The refusal is ``DecisionBusy``:
        another call holds the worker, which is this caller's budget and not
        an outage — a model that has stopped answering is found by the call
        that had the worker to itself and still ran out of time.
        """
        started = self._running_since
        if started and time.monotonic() - started >= deadline:
            raise DecisionBusy(
                "the decision model is busy with a call that outran its deadline"
            )

    def _submit(self, state: State, questions: Mapping[str, Question], stop_at: float) -> tuple[Future, bool]:
        """Queue a prediction; return it and whether another call was ahead."""
        with self._accounting_lock:
            contended = self._inflight > 0
            self._inflight += 1
        try:
            future = self._worker.submit(self._predict, state, questions, stop_at)
        except BaseException:
            self._settled(None)
            raise
        future.add_done_callback(self._settled)
        return future, contended

    def _settled(self, _future: Future | None) -> None:
        with self._accounting_lock:
            self._inflight -= 1

    @staticmethod
    def _late(deadline: float, contended: bool) -> DecisionTimeout:
        """What a deadline that ran out means. A call that shared the worker
        spent some of it waiting for another caller's inference, which is its
        own budget; a call that had the worker to itself and still ran out is
        the model failing to answer in time."""
        if contended:
            return DecisionBusy(
                f"no answer within {deadline:.1f}s: the decision model was busy with other callers"
            )
        return DecisionTimeout(f"no answer from the decision model within {deadline:.1f}s")

    def _read(
        self,
        result: Any,
        questions: Mapping[str, Question],
        started: float,
    ) -> Decisions:
        decisions = parse_response(
            result,
            questions,
            latency_ms=(time.monotonic() - started) * 1000,
        )
        with self._accounting_lock:
            self.requests += 1
            self.input_tokens += decisions.input_tokens
            self.output_tokens += decisions.output_tokens
        return decisions

    @staticmethod
    def _deadline(timeout: float | None) -> float:
        if timeout is None:
            return 30.0
        try:
            deadline = float(timeout)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"decision timeout must be a number, got {timeout!r}") from exc
        if deadline <= 0:
            raise DecisionTimeout("decision deadline has already expired")
        return deadline

    def ask(
        self,
        state: State,
        questions: Mapping[str, Question],
        *,
        timeout: float | None = None,
    ) -> Decisions:
        deadline = self._deadline(timeout)
        self._require_ready()
        self._refuse_if_overrun(deadline)
        started = time.monotonic()
        future, contended = self._submit(state, questions, started + deadline)
        try:
            result = future.result(timeout=deadline)
        except (FutureTimeout, DecisionTimeout) as exc:
            # Cancelling only helps while the work is still queued; one that
            # has started stops at its next question, on ``stop_at``.
            future.cancel()
            raise self._late(deadline, contended) from exc
        return self._read(result, questions, started)

    async def ask_async(
        self,
        state: State,
        questions: Mapping[str, Question],
        *,
        timeout: float | None = None,
    ) -> Decisions:
        deadline = self._deadline(timeout)
        self._require_ready()
        self._refuse_if_overrun(deadline)
        started = time.monotonic()
        future, contended = self._submit(state, questions, started + deadline)
        wrapped = asyncio.wrap_future(future)
        try:
            result = await asyncio.wait_for(asyncio.shield(wrapped), timeout=deadline)
        except (asyncio.TimeoutError, DecisionTimeout) as exc:
            # Cancelling only helps while the work is still queued; one that
            # has started stops at its next question, on ``stop_at``.
            future.cancel()
            raise self._late(deadline, contended) from exc
        return self._read(result, questions, started)

    def close(self) -> None:
        self._worker.close()

    async def aclose(self) -> None:
        self.close()

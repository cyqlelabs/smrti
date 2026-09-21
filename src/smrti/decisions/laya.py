"""Local adapter for the multilingual Laya typed-decision model.

The model is Laya; the runtime is EdgeJev, which serves the same checkpoint
as an int8 ONNX graph through onnxruntime. Nothing about the decisions
changes — same three primitives, same calibration, same thresholds — but the
weights are 343 MB instead of 1290 MB, resident size is about 560 MB instead
of 2.9 GB, and torch is gone from the dependency tree entirely. On the boxes
Smrti is meant to share with an agent, that is the difference between a
decision engine and a machine that swaps.

Laya exposes the same three primitives Smrti uses (``noul``, ``choice`` and
``score``), so the adapter only has to translate the question dataclasses to
Laya's dictionaries and validate its answer through the shared provider
boundary. Inference is serialized on one worker because one model instance
is shared by every sync and async caller.

**The checkpoint is never loaded on a caller's thread.** Loading it means
mapping a 343 MB graph and, on a machine that has not got the weights,
fetching them first — seconds to minutes of work behind a decision that
exists to save milliseconds. A caller that arrives
before the model is ready is told so immediately (``DecisionUnavailable``)
and falls back to the deterministic path, which is what every decision site
already does with that answer. The load runs once, on a daemon thread, and
the first caller after it finishes gets the model.

Paying it on the caller instead is what a recall costs when it goes wrong:
the load sits behind the decision deadline, every recall waits the whole
deadline because nothing remembers the last one already did, and a
30-second wall appears in front of a retrieval that takes 48ms. A failed
load then backs off rather than being retried per request, because each
attempt may re-try a 250 MB download before it can fail.

A local model directory can be supplied through ``SMRTI_DECISIONS_MODEL`` to
avoid any first-run download; see :mod:`smrti.decisions.model` for where the
weights are looked for otherwise.
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Any, Mapping

from . import model as model_source
from .provider import (
    DecisionUnavailable,
    Decisions,
    Question,
    State,
    parse_response,
    questions_payload,
)

logger = logging.getLogger("smrti.decisions.laya")


def _threads() -> int | None:
    """How many cores inference may hold.

    ``SMRTI_DECISIONS_THREADS`` decides; unset leaves the runtime's own
    default. The cap exists because the machines this matters on have two
    slow cores, and a decision that takes the whole box for 200 ms is worse
    than one that takes half of it for 400.
    """
    raw = os.environ.get("SMRTI_DECISIONS_THREADS", "").strip()
    if not raw:
        return None
    try:
        threads = int(raw)
    except ValueError:
        logger.warning("SMRTI_DECISIONS_THREADS=%r is not a number; ignoring it", raw)
        return None
    return threads if threads > 0 else None

# Empty means "resolve it": an explicit directory, the one Factor already
# has, or Smrti's own — see :func:`smrti.decisions.model.resolve`.
DEFAULT_MODEL = ""

# How long a failed load is left alone before another is attempted. A failing
# attempt may re-try a 250 MB download before it can fail, so retrying per
# request costs far more than the decisions are worth.
LOAD_RETRY_SECONDS = 900.0

# How many questions ride in one native call.
#
# Batching is not an optimization here, and it does not leave the answer
# alone either. Measured on the real checkpoint against twenty candidates'
# eighty evidence questions: one at a time took 185.6s and peaked at 1015 MB,
# while all eighty in a single call took 197.9s and peaked at 21798 MB. The
# one big call is slower *and* twenty-one times the memory, because the
# runtime reserves the whole batch's attention buffers before it starts and
# its session holds them at that peak afterwards. That is how a 100 MB engine
# became a 25 GB one within a minute of a single turn.
#
# It is also a different answer, reproducibly: run twice, each shape repeats
# itself exactly (drift 0.0000), while one question's probability moves by up
# to 0.33 depending only on which other questions were sent with it. These
# are independent propositions about independent candidates, so an answer
# that moves with its neighbours is the contaminated one. One question per
# call is the only shape that answers each on its own evidence.
MAX_QUESTIONS_PER_CALL = 1


def _tokens(usage: Mapping[str, Any], name: str) -> int:
    try:
        return int(usage.get(name, 0) or 0)
    except (TypeError, ValueError):
        return 0


class LayaProvider:
    """A lazy, process-local Laya provider with bounded, serialized calls."""

    name = "laya"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        device: str | None = None,
        agent: Any | None = None,
    ) -> None:
        self.model = model
        self.device = device or None
        self._agent = agent
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="smrti-laya")
        # When the one worker is running a prediction, this is when it
        # started; 0.0 while it is idle. Read by ``_refuse_if_overrun`` to
        # tell a busy model from one that has stopped answering.
        self._running_since = 0.0
        self._accounting_lock = threading.Lock()
        self.input_tokens = 0
        self.output_tokens = 0
        self.requests = 0
        # Load state, guarded by ``_load_lock``. The loader runs on a daemon
        # thread rather than on ``_executor``: a checkpoint download must not
        # sit in front of the inference queue, and a non-daemon worker stuck
        # in a download keeps the interpreter from exiting at all — the
        # ThreadPoolExecutor atexit hook joins it, which is how a stopped
        # engine leaves a process that will not die.
        self._load_lock = threading.Lock()
        self._loader: threading.Thread | None = None
        self._load_error: str | None = None
        self._retry_at = 0.0

    def _load(self) -> Any:
        """Import the runtime and map the graph. Runs on the loader thread."""
        if self._agent is not None:
            return self._agent
        try:
            import edgejev
        except ImportError as exc:
            raise DecisionUnavailable(
                "the decision runtime is missing from the Smrti installation; "
                "reinstall Smrti's dependencies"
            ) from exc
        try:
            directory = self.model or str(model_source.resolve())
        except Exception as exc:
            raise DecisionUnavailable(f"no decision model to load: {exc}") from exc
        try:
            agent = edgejev.Agent(directory, threads=_threads(), provider=self.device or None)
        except Exception as exc:
            raise DecisionUnavailable(f"could not load the decision model at {directory!r}: {exc}") from exc
        self.model = directory
        self._agent = agent
        return agent

    def _run_load(self) -> None:
        started = time.monotonic()
        try:
            self._load()
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
        """Start the checkpoint load in the background, at most one at a time."""
        with self._load_lock:
            if self._agent is not None or self._loader is not None:
                return
            if self._retry_at and time.monotonic() < self._retry_at:
                return
            self._loader = threading.Thread(
                target=self._run_load, name="smrti-laya-load", daemon=True
            )
            thread = self._loader
        thread.start()

    def _require_ready(self) -> Any:
        """The loaded agent, or ``DecisionUnavailable`` without waiting for one.

        A decision is worth a few milliseconds of a recall and nothing like
        the minutes a first load can take, so the caller is never made to
        wait for it: it is told the model is not ready and takes the path it
        had before decisions existed.
        """
        agent = self._agent
        if agent is not None:
            return agent
        self._begin_load()
        with self._load_lock:
            error, loading = self._load_error, self._loader is not None
        if error is not None:
            raise DecisionUnavailable(f"the decision model is not available: {error}")
        raise DecisionUnavailable(
            "the decision model is still loading"
            if loading
            else "the decision model is not loaded"
        )

    @property
    def ready(self) -> bool:
        """Whether a decision asked right now would reach the model."""
        return self._agent is not None

    def preload(self, timeout: float | None = None) -> bool:
        """Load the checkpoint now rather than on the first decision.

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
        self._running_since = time.monotonic()
        try:
            for start in range(0, len(items), MAX_QUESTIONS_PER_CALL):
                if start and stop_at is not None and time.monotonic() >= stop_at:
                    # Nobody is waiting for this any more. Every question has
                    # to be answered for the reply to parse, so there is
                    # nothing here worth salvaging — and the whole point of
                    # stopping is to not spend the next chunk's memory and
                    # cores on an answer that will never be read.
                    raise DecisionUnavailable(
                        f"the decision model answered {len(answers)} of {len(items)} "
                        "questions before the deadline"
                    )
                chunk = dict(items[start : start + MAX_QUESTIONS_PER_CALL])
                try:
                    piece = agent.predict(state, questions_payload(chunk))
                except DecisionUnavailable:
                    raise
                except Exception as exc:
                    raise DecisionUnavailable(f"decision inference failed: {exc}") from exc
                if not isinstance(piece, Mapping):
                    raise DecisionUnavailable("reply is not an object")
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
        So a caller that queues behind one which has already outrun its own
        deadline spends its whole deadline in a wait that cannot succeed, and
        so does every caller behind it. The deterministic path is right there
        and costs nothing.
        """
        started = self._running_since
        if started and time.monotonic() - started >= deadline:
            raise DecisionUnavailable(
                "the decision model is busy with a call that outran its deadline"
            )

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
            raise DecisionUnavailable("Laya decision deadline has already expired")
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
        future = self._executor.submit(self._predict, state, questions, started + deadline)
        try:
            result = future.result(timeout=deadline)
        except FutureTimeout as exc:
            # Cancelling only helps while the work is still queued; one that
            # has started stops at its next question, on ``stop_at``.
            future.cancel()
            raise DecisionUnavailable(f"no answer from the decision model within {deadline:.1f}s") from exc
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
        future = self._executor.submit(self._predict, state, questions, started + deadline)
        wrapped = asyncio.wrap_future(future)
        try:
            result = await asyncio.wait_for(asyncio.shield(wrapped), timeout=deadline)
        except asyncio.TimeoutError as exc:
            # Cancelling only helps while the work is still queued; one that
            # has started stops at its next question, on ``stop_at``.
            future.cancel()
            raise DecisionUnavailable(f"no answer from the decision model within {deadline:.1f}s") from exc
        return self._read(result, questions, started)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    async def aclose(self) -> None:
        self.close()

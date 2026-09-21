"""Local adapter for the multilingual Laya typed-decision model.

Laya exposes the same three primitives Smrti uses (``noul``, ``choice`` and
``score``), so the adapter only has to translate the question dataclasses to
Laya's dictionaries and validate its answer through the shared provider
boundary. Inference is serialized on one worker because one model instance
is shared by every sync and async caller.

**The checkpoint is never loaded on a caller's thread.** Loading it means
importing torch and, on a machine that has not cached the weights,
downloading a 322M-parameter checkpoint from Hugging Face — minutes of work
behind a decision that exists to save milliseconds. A caller that arrives
before the model is ready is told so immediately (``DecisionUnavailable``)
and falls back to the deterministic path, which is what every decision site
already does with that answer. The load runs once, on a daemon thread, and
the first caller after it finishes gets the model.

Paying it on the caller instead is what a recall costs when it goes wrong:
the load sits behind the decision deadline, every recall waits the whole
deadline because nothing remembers the last one already did, and a
30-second wall appears in front of a retrieval that takes 48ms. A failed
load then backs off rather than being retried per request, because each
attempt imports torch before it can fail.

A local checkpoint path can be supplied through ``SMRTI_DECISIONS_MODEL``
to avoid any first-run download.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Any, Mapping

from .provider import (
    DecisionUnavailable,
    Decisions,
    Question,
    State,
    parse_response,
    questions_payload,
)

logger = logging.getLogger("smrti.decisions.laya")

DEFAULT_MODEL = "convaiinnovations/laya-multilingual"

# How long a failed load is left alone before another is attempted. Each
# attempt imports torch and may re-try a download before it can fail, so
# retrying per request costs far more than the decisions are worth.
LOAD_RETRY_SECONDS = 900.0


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
        if not model:
            raise ValueError("a Laya model id or local path is required")
        self.model = model
        self.device = device or None
        self._agent = agent
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="smrti-laya")
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
        """Import Laya and load the checkpoint. Runs on the loader thread."""
        if self._agent is not None:
            return self._agent
        try:
            import laya
        except ImportError as exc:
            raise DecisionUnavailable(
                "Laya is missing from the Smrti core installation; reinstall Smrti's dependencies"
            ) from exc
        try:
            agent = laya.load(self.model, device=self.device)
        except Exception as exc:
            raise DecisionUnavailable(f"could not load Laya model {self.model!r}: {exc}") from exc
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
                "Laya model %r could not be loaded (%s); decisions fall back to the "
                "deterministic path and the load is retried in %.0f minutes",
                self.model, exc, LOAD_RETRY_SECONDS / 60,
            )
        else:
            with self._load_lock:
                self._load_error = None
            logger.info(
                "Laya model %r ready after %.1fs; decisions are live",
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
            raise DecisionUnavailable(f"Laya is not available: {error}")
        raise DecisionUnavailable(
            f"the Laya model {self.model!r} is still loading"
            if loading
            else f"the Laya model {self.model!r} is not loaded"
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

    def _predict(self, state: State, questions: Mapping[str, Question]) -> Any:
        if not questions:
            raise ValueError("a decision needs at least one question")
        try:
            result = self._require_ready().predict(state, questions_payload(questions))
        except DecisionUnavailable:
            raise
        except Exception as exc:
            raise DecisionUnavailable(f"Laya inference failed: {exc}") from exc
        if isinstance(result, Mapping):
            result = dict(result)
            # Laya currently reports the generic architecture name. The
            # configured checkpoint is the useful identity for audit/cache.
            result["model"] = self.model
        return result

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
        started = time.monotonic()
        future = self._executor.submit(self._predict, state, questions)
        try:
            result = future.result(timeout=deadline)
        except FutureTimeout as exc:
            future.cancel()
            raise DecisionUnavailable(f"no answer from Laya within {deadline:.1f}s") from exc
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
        started = time.monotonic()
        future = self._executor.submit(self._predict, state, questions)
        wrapped = asyncio.wrap_future(future)
        try:
            result = await asyncio.wait_for(asyncio.shield(wrapped), timeout=deadline)
        except asyncio.TimeoutError as exc:
            future.cancel()
            raise DecisionUnavailable(f"no answer from Laya within {deadline:.1f}s") from exc
        return self._read(result, questions, started)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    async def aclose(self) -> None:
        self.close()

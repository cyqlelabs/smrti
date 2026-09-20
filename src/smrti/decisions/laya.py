"""Local adapter for the multilingual Laya typed-decision model.

Laya exposes the same three primitives Smrti uses (``noul``, ``choice`` and
``score``), so the adapter only has to translate the question dataclasses to
Laya's dictionaries and validate its answer through the shared provider
boundary. Inference is serialized on one worker because one model instance
is shared by every sync and async caller.

The model is loaded on the first decision. Hugging Face downloads its
weights once and serves every subsequent request from the local cache; a
local checkpoint path can be supplied through ``SMRTI_DECISIONS_MODEL`` to
avoid any first-run download.
"""
from __future__ import annotations

import asyncio
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

DEFAULT_MODEL = "convaiinnovations/laya-multilingual"


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

    def _load(self) -> Any:
        if self._agent is not None:
            return self._agent
        try:
            import laya
        except ImportError as exc:
            raise DecisionUnavailable(
                "Laya is missing from the Smrti core installation; reinstall Smrti's dependencies"
            ) from exc
        try:
            self._agent = laya.load(self.model, device=self.device)
        except Exception as exc:
            raise DecisionUnavailable(f"could not load Laya model {self.model!r}: {exc}") from exc
        return self._agent

    def preload(self) -> None:
        """Load the checkpoint now instead of making the first decision pay for it."""
        self._executor.submit(self._load).result()

    def _predict(self, state: State, questions: Mapping[str, Question]) -> Any:
        if not questions:
            raise ValueError("a decision needs at least one question")
        try:
            result = self._load().predict(state, questions_payload(questions))
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

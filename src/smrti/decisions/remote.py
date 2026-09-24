"""A decision provider that asks a server already running the model.

Factor runs the same Laya checkpoint Smrti decides with, as an EdgeJev
server on loopback, and a machine running both was loading the weights
twice: 560 MB resident each, on boxes where the second copy is the
difference between an engine that answers and one that swaps. The weights
were already shared on disk (see :mod:`smrti.decisions.model`); this shares
the process. ``SMRTI_DECISIONS_URL`` names the server and Smrti loads
nothing of its own.

The route is ``POST /v1/systemone`` — the contract EdgeJev serves and the
one Factor's own client speaks — so any server that answers it will do. One
question rides per call, for the reason :mod:`smrti.decisions.laya` gives:
the model's answers move with their neighbours when several are asked at
once. A server that is down, still loading or refusing the request is
:class:`DecisionUnavailable`, which every caller reads as "decide the way
you did before decisions existed"; the engine's cooldown then keeps a dead
server from costing a connect per recall.
"""
from __future__ import annotations

import time
from typing import Any, Mapping

import httpx

from .laya import MAX_QUESTIONS_PER_CALL
from .local import _tokens
from .provider import DecisionUnavailable, DecisionUnsupported, Decisions, Question, State, parse_response, questions_payload

from .student.serve import UNSUPPORTED

# The one route a decision travels.
REQUEST_PATH = "/v1/systemone"

# The model name the request carries. The server answers with what it
# actually loaded, which is what the audit record keeps.
MODEL_NAME = "laya"


class RemoteProvider:
    """Decisions over HTTP, from a server that holds the model."""

    name = "edgejev"

    def __init__(self, base_url: str, *, transport: httpx.BaseTransport | None = None,
                 async_transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        # Filled in from the first reply; until then the request's own name.
        self.model = MODEL_NAME
        self._client = httpx.Client(base_url=self.base_url, transport=transport)
        self._async = httpx.AsyncClient(base_url=self.base_url, transport=async_transport)
        self._questions_per_call: int | None = None

    @staticmethod
    def _deadline(timeout: float | None) -> float:
        if timeout is None:
            return 30.0
        try:
            deadline = float(timeout)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"decision timeout must be a number, got {timeout!r}") from exc
        if deadline <= 0:
            raise DecisionUnavailable("decision deadline has already expired")
        return deadline

    @staticmethod
    def _body(state: State, chunk: Mapping[str, Question]) -> dict[str, Any]:
        return {"model": MODEL_NAME, "state": state, "questions": questions_payload(chunk)}

    def _per_call(self) -> int:
        """How many questions ride one call: one for Laya (see ``laya``),
        every one for a student, which answers a whole head in one pass and
        would otherwise pay that pass per question. Asked of ``/health``
        once; a server that does not say is taken for Laya."""
        if self._questions_per_call is None:
            per = MAX_QUESTIONS_PER_CALL
            try:
                health = self._client.get("/health", timeout=5.0).json()
                if isinstance(health, Mapping) and health.get("backend") == "student":
                    per = 256
            except (httpx.HTTPError, ValueError):
                pass
            self._questions_per_call = per
        return self._questions_per_call

    def _chunks(self, questions: Mapping[str, Question]) -> list[dict[str, Question]]:
        if not questions:
            raise ValueError("a decision needs at least one question")
        items = list(questions.items())
        per = self._per_call()
        return [dict(items[i : i + per]) for i in range(0, len(items), per)]

    @staticmethod
    def _reply(response: httpx.Response) -> Mapping[str, Any]:
        if response.status_code >= 400:
            if response.status_code == 400 and UNSUPPORTED in response.text:
                raise DecisionUnsupported(f"the decision server does not answer this question: {response.text[:200]}")
            raise DecisionUnavailable(
                f"the decision server answered HTTP {response.status_code}: {response.text[:200]}"
            )
        try:
            piece = response.json()
        except ValueError as exc:
            raise DecisionUnavailable("the decision server's reply is not JSON") from exc
        if not isinstance(piece, Mapping):
            raise DecisionUnavailable("the decision server's reply is not an object")
        return piece

    def _merge(self, pieces: list[Mapping[str, Any]]) -> dict[str, Any]:
        answers: dict[str, Any] = {}
        input_tokens = output_tokens = 0
        for piece in pieces:
            part = piece.get("answers")
            if isinstance(part, Mapping):
                answers.update(part)
            usage = piece.get("usage")
            if isinstance(usage, Mapping):
                input_tokens += _tokens(usage, "input_tokens")
                output_tokens += _tokens(usage, "output_tokens")
            model = piece.get("model")
            if isinstance(model, str) and model:
                self.model = model
        return {
            "answers": answers,
            "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
            "model": self.model,
        }

    @staticmethod
    def _remaining(stop_at: float) -> float:
        left = stop_at - time.monotonic()
        if left <= 0:
            raise DecisionUnavailable("the decision server did not answer every question before the deadline")
        return left

    def ask(self, state: State, questions: Mapping[str, Question], *, timeout: float | None = None) -> Decisions:
        deadline = self._deadline(timeout)
        started = time.monotonic()
        stop_at = started + deadline
        pieces = []
        for chunk in self._chunks(questions):
            try:
                response = self._client.post(
                    REQUEST_PATH, json=self._body(state, chunk), timeout=self._remaining(stop_at)
                )
            except httpx.HTTPError as exc:
                raise self._unreachable(exc) from exc
            pieces.append(self._reply(response))
        return parse_response(self._merge(pieces), questions, latency_ms=(time.monotonic() - started) * 1000)

    def _unreachable(self, exc: httpx.HTTPError) -> DecisionUnavailable:
        # httpx raises some transport errors with an empty message (a reset
        # mid-response is one), and "unreachable: " says nothing.
        return DecisionUnavailable(
            f"the decision server at {self.base_url} is unreachable: {str(exc) or type(exc).__name__}"
        )

    async def ask_async(
        self, state: State, questions: Mapping[str, Question], *, timeout: float | None = None
    ) -> Decisions:
        deadline = self._deadline(timeout)
        started = time.monotonic()
        stop_at = started + deadline
        pieces = []
        for chunk in self._chunks(questions):
            try:
                response = await self._async.post(
                    REQUEST_PATH, json=self._body(state, chunk), timeout=self._remaining(stop_at)
                )
            except httpx.HTTPError as exc:
                raise self._unreachable(exc) from exc
            pieces.append(self._reply(response))
        return parse_response(self._merge(pieces), questions, latency_ms=(time.monotonic() - started) * 1000)

    def close(self) -> None:
        self._client.close()

    async def aclose(self) -> None:
        self._client.close()
        await self._async.aclose()

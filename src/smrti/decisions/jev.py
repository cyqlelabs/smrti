"""Adapter for TypeSafe's Jev, the System One decision model.

One request is ``POST {base_url}/v1/systemone`` with a bearer key and a body
of ``{"state", "model", "questions": {key: question}}``; the reply carries
``{"answers": {key: answer}, "model", "usage": {"input_tokens",
"output_tokens"}}``. Every question in a request is evaluated against the
same state and the API answers them in parallel, so a request with forty
questions costs about the time of one — it does not cost the tokens of one,
since question text is billed as input.

The adapter owns the transport concerns the engine should not see: a shared
client per process (and per event loop for the async path), a deadline on
every call, one bounded retry on the statuses that mean "try again" and on
a dropped connection, and usage accounting. Anything else — the key rejected,
the request refused as invalid, a reply that is not an answer — is a
:class:`DecisionUnavailable` for the caller to fall back on.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any, Mapping

import httpx

from .provider import DecisionUnavailable, Decisions, Question, State, parse_response, questions_payload

logger = logging.getLogger("smrti.decisions.jev")

DEFAULT_BASE_URL = "https://api.typesafe.ai"
SYSTEMONE_PATH = "/v1/systemone"
DEFAULT_MODEL = "jev-latest"

# TypeSafe's published input price, US dollars per million input tokens, at
# the time of writing. Output is billed at zero. A reference figure for the
# audit log's cost column, not a measurement and not a guarantee.
USD_PER_MILLION_INPUT_TOKENS = 0.042

# Statuses worth one more attempt: rate limited, saturated, or a gateway
# that dropped the request. Everything else is answered by the first reply.
_RETRY_STATUSES = frozenset({429, 502, 503, 504, 529})


class JevProvider:
    """A provider over the Jev HTTP API. Thread-safe; one instance per process."""

    name = "jev"

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        timeout: float = 5.0,
        retries: int = 1,
        transport: httpx.BaseTransport | None = None,
        async_transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("a Jev API key is required")
        self._api_key = api_key
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or DEFAULT_MODEL
        self.timeout = timeout
        self.retries = max(0, retries)
        self._transport = transport
        self._async_transport = async_transport
        self._client: httpx.Client | None = None
        self._client_lock = threading.Lock()
        # AsyncClient is bound to the loop it was created on; one per loop.
        self._async_clients: dict[int, tuple[asyncio.AbstractEventLoop, httpx.AsyncClient]] = {}
        self.input_tokens = 0
        self.output_tokens = 0
        self.requests = 0

    # ── transport ────────────────────────────────────────────────────────

    @property
    def url(self) -> str:
        return f"{self.base_url}{SYSTEMONE_PATH}"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

    def _body(self, state: State, questions: Mapping[str, Question]) -> dict[str, Any]:
        if not questions:
            raise ValueError("a decision needs at least one question")
        return {"state": state, "model": self.model, "questions": questions_payload(questions)}

    def _sync_client(self) -> httpx.Client:
        with self._client_lock:
            if self._client is None:
                self._client = httpx.Client(transport=self._transport, timeout=httpx.Timeout(self.timeout))
            return self._client

    def _async_client(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        for loop_id, (cached, _) in list(self._async_clients.items()):
            if cached.is_closed():
                self._async_clients.pop(loop_id, None)
        entry = self._async_clients.get(id(loop))
        if entry is None:
            entry = (loop, httpx.AsyncClient(transport=self._async_transport, timeout=httpx.Timeout(self.timeout)))
            self._async_clients[id(loop)] = entry
        return entry[1]

    def close(self) -> None:
        with self._client_lock:
            if self._client is not None:
                self._client.close()
                self._client = None

    async def aclose(self) -> None:
        entry = self._async_clients.pop(id(asyncio.get_running_loop()), None)
        if entry is not None:
            await entry[1].aclose()

    # ── replies ──────────────────────────────────────────────────────────

    @staticmethod
    def _should_retry(response: httpx.Response) -> bool:
        return response.status_code in _RETRY_STATUSES

    @staticmethod
    def _refuse(response: httpx.Response) -> DecisionUnavailable:
        detail = response.text.strip()[:300]
        status = response.status_code
        if status == 401:
            return DecisionUnavailable(f"the Jev API key was rejected: {detail}")
        if status == 422:
            return DecisionUnavailable(f"the Jev request was refused as invalid: {detail}")
        return DecisionUnavailable(f"Jev returned HTTP {status}: {detail}")

    def _read(self, response: httpx.Response, questions: Mapping[str, Question], started: float) -> Decisions:
        if response.status_code >= 400:
            raise self._refuse(response)
        try:
            data = response.json()
        except ValueError as exc:
            raise DecisionUnavailable(f"Jev reply is not JSON: {exc}") from exc
        decisions = parse_response(data, questions, latency_ms=(time.monotonic() - started) * 1000)
        self.requests += 1
        self.input_tokens += decisions.input_tokens
        self.output_tokens += decisions.output_tokens
        return decisions

    @staticmethod
    def _retry_delay(response: httpx.Response | None, attempt: int) -> float:
        if response is not None:
            header = response.headers.get("Retry-After")
            if header:
                try:
                    return min(2.0, max(0.0, float(header)))
                except ValueError:
                    pass
        return 0.2 * (attempt + 1)

    # ── asking ───────────────────────────────────────────────────────────

    def ask(self, state: State, questions: Mapping[str, Question], *, timeout: float | None = None) -> Decisions:
        body = self._body(state, questions)
        deadline = self.timeout if timeout is None else timeout
        client = self._sync_client()
        started = time.monotonic()
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            remaining = deadline - (time.monotonic() - started)
            if remaining <= 0:
                break
            try:
                response = client.post(self.url, json=body, headers=self._headers(), timeout=remaining)
            except httpx.HTTPError as exc:
                last_error = DecisionUnavailable(f"Jev request failed: {exc}")
                time.sleep(min(self._retry_delay(None, attempt), max(0.0, deadline - (time.monotonic() - started))))
                continue
            if self._should_retry(response) and attempt < self.retries:
                last_error = self._refuse(response)
                time.sleep(min(self._retry_delay(response, attempt), max(0.0, deadline - (time.monotonic() - started))))
                continue
            return self._read(response, questions, started)
        raise last_error or DecisionUnavailable(f"no answer from Jev within {deadline:.1f}s")

    async def ask_async(
        self, state: State, questions: Mapping[str, Question], *, timeout: float | None = None
    ) -> Decisions:
        body = self._body(state, questions)
        deadline = self.timeout if timeout is None else timeout
        client = self._async_client()
        started = time.monotonic()
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            remaining = deadline - (time.monotonic() - started)
            if remaining <= 0:
                break
            try:
                response = await client.post(self.url, json=body, headers=self._headers(), timeout=remaining)
            except httpx.HTTPError as exc:
                last_error = DecisionUnavailable(f"Jev request failed: {exc}")
                await asyncio.sleep(min(self._retry_delay(None, attempt), max(0.0, deadline - (time.monotonic() - started))))
                continue
            if self._should_retry(response) and attempt < self.retries:
                last_error = self._refuse(response)
                await asyncio.sleep(min(self._retry_delay(response, attempt), max(0.0, deadline - (time.monotonic() - started))))
                continue
            return self._read(response, questions, started)
        raise last_error or DecisionUnavailable(f"no answer from Jev within {deadline:.1f}s")

    # ── accounting ───────────────────────────────────────────────────────

    @property
    def estimated_cost_usd(self) -> float:
        return self.input_tokens / 1_000_000 * USD_PER_MILLION_INPUT_TOKENS


def estimated_cost_usd(input_tokens: int) -> float:
    return input_tokens / 1_000_000 * USD_PER_MILLION_INPUT_TOKENS

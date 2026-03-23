"""Bounded async dialogue queue — enriches agent dialogue via LLM without blocking ticks."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine

from smrti_town.llm import LLMClient

log = logging.getLogger(__name__)


@dataclass
class DialogueRequest:
    speaker: str
    target: str | None
    location: str
    time_of_day: str
    season: str
    personality: str
    urgent_need: str | None
    memories: list[dict]
    fallback: str
    tick_number: int


class DialogueQueue:
    """Bounded async queue that drains dialogue requests in batches and
    broadcasts enriched dialogue lines back to connected clients.

    Prevents unbounded LLM task accumulation by:
    - Capping the queue to *queue_size* entries (oldest are dropped).
    - Tracking in-flight speakers to avoid duplicate requests.
    - Discarding stale requests (older than *stale_ticks* behind current).
    """

    def __init__(
        self,
        llm_client: LLMClient,
        broadcast_fn: Callable[[dict], Coroutine[Any, Any, None]],
        queue_size: int = 20,
        batch_size: int = 5,
        stale_ticks: int = 3,
    ) -> None:
        self._queue: asyncio.Queue[DialogueRequest] = asyncio.Queue(maxsize=queue_size)
        self._llm = llm_client
        self._broadcast = broadcast_fn
        self._batch_size = batch_size
        self._stale_ticks = stale_ticks
        self._in_flight: set[str] = set()
        self._worker_task: asyncio.Task[None] | None = None
        self._current_tick: int = 0
        self._running = False

    # ── lifecycle ───────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background worker coroutine."""
        if self._worker_task is not None and not self._worker_task.done():
            return
        self._running = True
        self._worker_task = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        """Stop the worker and drain remaining items."""
        self._running = False
        if self._worker_task is not None:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None
        self._in_flight.clear()

    # ── submit ──────────────────────────────────────────────────────────

    def submit(self, request: DialogueRequest) -> bool:
        """Submit a dialogue request.

        Returns ``False`` if the queue is full or the speaker already has
        an in-flight request.
        """
        if request.speaker in self._in_flight:
            return False
        try:
            self._queue.put_nowait(request)
            self._in_flight.add(request.speaker)
            self._current_tick = max(self._current_tick, request.tick_number)
            return True
        except asyncio.QueueFull:
            log.debug("Dialogue queue full, dropping request for %s", request.speaker)
            return False

    # ── worker ──────────────────────────────────────────────────────────

    async def _worker(self) -> None:
        """Background loop: drain up to batch_size, call LLM, broadcast results."""
        while self._running:
            batch: list[DialogueRequest] = []

            # Block on first item
            try:
                first = await asyncio.wait_for(self._queue.get(), timeout=2.0)
                batch.append(first)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                return

            # Drain remaining up to batch_size
            while len(batch) < self._batch_size:
                try:
                    item = self._queue.get_nowait()
                    batch.append(item)
                except asyncio.QueueEmpty:
                    break

            # Filter stale requests
            fresh: list[DialogueRequest] = []
            for req in batch:
                if self._current_tick - req.tick_number > self._stale_ticks:
                    self._in_flight.discard(req.speaker)
                    log.debug("Discarding stale dialogue request for %s (tick %d)", req.speaker, req.tick_number)
                    continue
                fresh.append(req)

            if not fresh:
                continue

            # Build LLM batch request
            llm_requests = [
                {
                    "speaker": req.speaker,
                    "target": req.target,
                    "location": req.location,
                    "time_of_day": req.time_of_day,
                    "season": req.season,
                    "personality": req.personality,
                    "urgent_need": req.urgent_need,
                    "memories": req.memories,
                    "fallback": req.fallback,
                }
                for req in fresh
            ]

            try:
                lines = await self._llm.generate_dialogue_batch(llm_requests)
            except Exception:
                log.exception("Dialogue batch LLM call failed")
                lines = [req.fallback for req in fresh]

            # Broadcast results
            for req, line in zip(fresh, lines):
                self._in_flight.discard(req.speaker)
                patch = {
                    "type": "dialogue_patch",
                    "speaker": req.speaker,
                    "target": req.target,
                    "location": req.location,
                    "line": line,
                    "tick": req.tick_number,
                }
                try:
                    await self._broadcast(patch)
                except Exception:
                    log.debug("Failed to broadcast dialogue patch for %s", req.speaker, exc_info=True)

"""Bounded single-worker dialogue queue with batch LLM calls and per-agent deduplication.

Architecture
------------
Producer (engine tick loop)  →  asyncio.Queue(maxsize)  →  single Worker coroutine
                                                               │
                                         ┌─────────────────────┘
                                         │  drain up to batch_size requests
                                         │  send ONE LLM call (batch prompt)
                                         │  broadcast dialogue_patch per result
                                         └─────────────────────────────────────

Backpressure guarantees
- Queue full → new requests are dropped silently (template fallback stays).
- Agent already in-flight → request dropped (per-agent deduplication).
- Request tick > _STALE_TICKS behind current tick → patch not broadcast.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
from typing import TYPE_CHECKING, Any, Callable, Coroutine

if TYPE_CHECKING:
    from smrti_town.llm import LLMClient

logger = logging.getLogger("smrti_town.dialogue_queue")

# Patches for ticks this far in the past are not broadcast;
# the frontend event-log entry is already gone.
_STALE_TICKS = 10


@dataclasses.dataclass
class DialogueRequest:
    speaker: str
    target: str
    location: str
    time_of_day: str
    season: str
    personality: str
    urgent_drive: str | None
    memories: list[dict]   # each dict: {content, salience, valence}
    fallback: str
    tick_number: int


class DialogueQueue:
    """Single-worker, bounded queue for LLM dialogue enrichment.

    Parameters
    ----------
    llm_client:
        LLMClient used to call the model.
    maxsize:
        Maximum number of pending requests before new ones are dropped.
    batch_size:
        Maximum requests drained into a single batched LLM call.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        maxsize: int = 8,
        batch_size: int = 3,
    ) -> None:
        self._llm = llm_client
        self._batch_size = batch_size
        self._queue: asyncio.Queue[DialogueRequest] = asyncio.Queue(maxsize=maxsize)
        self._inflight: set[str] = set()
        self._worker_task: asyncio.Task | None = None
        self._broadcast: Callable[[dict], Coroutine[Any, Any, None]] | None = None
        self._current_tick: int = 0

    # ── Lifecycle ────────────────────────────────────────────────────

    def set_broadcast(self, fn: Callable[[dict], Coroutine[Any, Any, None]] | None) -> None:
        self._broadcast = fn

    def update_tick(self, tick_number: int) -> None:
        self._current_tick = tick_number

    def start(self) -> None:
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker())
            self._worker_task.add_done_callback(
                lambda t: logger.debug("Dialogue worker stopped: %s", t.exception() if not t.cancelled() else "cancelled")
            )

    async def stop(self) -> None:
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
        self._worker_task = None
        self._inflight.clear()

    # ── Producer API ─────────────────────────────────────────────────

    def enqueue(self, req: DialogueRequest) -> bool:
        """Try to add a request.

        Returns True if accepted, False if dropped (agent in-flight or queue full).
        Never blocks — the caller's tick must not be delayed.
        """
        if req.speaker in self._inflight:
            return False
        try:
            self._queue.put_nowait(req)
            self._inflight.add(req.speaker)
            return True
        except asyncio.QueueFull:
            return False

    # ── Worker ───────────────────────────────────────────────────────

    async def _worker(self) -> None:
        while True:
            try:
                first = await self._queue.get()
            except asyncio.CancelledError:
                return

            # Drain additional requests that arrived while we were waiting.
            batch = [first]
            for _ in range(self._batch_size - 1):
                try:
                    batch.append(self._queue.get_nowait())
                except asyncio.QueueEmpty:
                    break

            try:
                await self._process_batch(batch)
            except Exception as exc:
                logger.debug("Dialogue batch error: %s", exc)
            finally:
                for req in batch:
                    self._inflight.discard(req.speaker)
                    self._queue.task_done()

    async def _process_batch(self, batch: list[DialogueRequest]) -> None:
        if len(batch) == 1:
            req = batch[0]
            text = await self._llm.generate_dialogue(
                speaker=req.speaker,
                target=req.target,
                location=req.location,
                time_of_day=req.time_of_day,
                season=req.season,
                personality=req.personality,
                urgent_drive=req.urgent_drive,
                memories=req.memories,
                fallback=req.fallback,
            )
            await self._maybe_broadcast(req, text)
        else:
            texts = await self._llm.generate_dialogue_batch(batch)
            for req, text in zip(batch, texts):
                await self._maybe_broadcast(req, text)

    async def _maybe_broadcast(self, req: DialogueRequest, text: str) -> None:
        if text == req.fallback:
            return
        age = self._current_tick - req.tick_number
        if age > _STALE_TICKS:
            logger.debug(
                "Dropping stale dialogue_patch (age=%d ticks, speaker=%s)",
                age, req.speaker,
            )
            return
        if self._broadcast:
            try:
                await self._broadcast({
                    "type": "dialogue_patch",
                    "tick": req.tick_number,
                    "speaker": req.speaker,
                    "target": req.target,
                    "content": text,
                })
            except Exception as exc:
                logger.debug("Broadcast dialogue_patch failed: %s", exc)

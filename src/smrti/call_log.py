"""Shared in-memory ring buffer for LLM call interception across all serve modes."""
from __future__ import annotations

import asyncio
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any

_CALL_LOG: deque[dict[str, Any]] = deque(maxlen=200)
_subscribers: list[asyncio.Queue[dict[str, Any]]] = []


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _notify(msg: dict[str, Any]) -> None:
    for q in _subscribers:
        try:
            q.put_nowait(msg)
        except asyncio.QueueFull:
            pass


def append(entry: dict[str, Any]) -> None:
    entry.setdefault("id", uuid.uuid4().hex[:8])
    entry.setdefault("ts", _now())
    _CALL_LOG.append(entry)
    _notify(entry)


def update(entry: dict[str, Any]) -> None:
    """The deque holds a reference to the same dict so caller mutations are
    already reflected in get_all(). We still notify SSE subscribers."""
    _notify({"__update__": True, **entry})


def get_all() -> list[dict[str, Any]]:
    return list(reversed(_CALL_LOG))


def clear() -> None:
    _CALL_LOG.clear()
    _notify({"__clear__": True})


def subscribe() -> asyncio.Queue[dict[str, Any]]:
    q: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=200)
    _subscribers.append(q)
    return q


def unsubscribe(q: asyncio.Queue[dict[str, Any]]) -> None:
    try:
        _subscribers.remove(q)
    except ValueError:
        pass

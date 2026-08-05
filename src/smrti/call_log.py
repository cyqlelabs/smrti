"""Shared in-memory ring buffer for LLM call interception across all serve modes."""
from __future__ import annotations

import asyncio
import json
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any

_CALL_LOG: deque[dict[str, Any]] = deque(maxlen=200)
_subscribers: list[asyncio.Queue[dict[str, Any]]] = []

_MAX_ENTRY_BYTES = 50_000
_TRUNCATION_MARKER = "…[truncated]"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _entry_bytes(entry: dict[str, Any]) -> int:
    try:
        return len(json.dumps(entry, default=str).encode())
    except (TypeError, ValueError):
        return 0


def _truncate_value(value: Any, max_len: int) -> Any:
    if isinstance(value, str) and len(value) > max_len:
        return value[:max_len] + _TRUNCATION_MARKER
    if isinstance(value, list):
        return [_truncate_value(v, max_len) for v in value]
    if isinstance(value, dict):
        return {k: _truncate_value(v, max_len) for k, v in value.items()}
    return value


def _cap_entry(entry: dict[str, Any]) -> None:
    """Bound the serialized entry size (~50KB), mutating the dict in place.

    Structure (message lists, dict keys) is preserved — individual strings are
    shortened and marked with a truncation marker.
    """
    if _entry_bytes(entry) <= _MAX_ENTRY_BYTES:
        return
    for max_len in (4000, 1000, 200):
        for key in list(entry):
            entry[key] = _truncate_value(entry[key], max_len)
        if _entry_bytes(entry) <= _MAX_ENTRY_BYTES:
            return


def _notify(msg: dict[str, Any]) -> None:
    for q in _subscribers:
        try:
            q.put_nowait(msg)
        except asyncio.QueueFull:
            pass


def append(entry: dict[str, Any]) -> None:
    entry.setdefault("id", uuid.uuid4().hex[:8])
    entry.setdefault("ts", _now())
    _cap_entry(entry)
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

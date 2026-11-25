"""Shared in-memory ring buffer for LLM call interception across all serve modes."""
from __future__ import annotations

import asyncio
import json
import re
import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any

_CALL_LOG: deque[dict[str, Any]] = deque(maxlen=200)
# A subscriber reads entries until it reads None, which close_streams() sends.
_subscribers: list[asyncio.Queue[dict[str, Any] | None]] = []

_MAX_ENTRY_BYTES = 50_000
_TRUNCATION_MARKER = "…[truncated]"

# The user:password part of a URL. Upstream and extraction endpoints may carry
# basic-auth credentials, and both the URL and the errors that quote it are
# served back on /llm-calls and in the proxy's error bodies.
_URL_CREDENTIALS = re.compile(r"(?<=://)[^/@\s]+@")
_REDACTED_FIELDS = ("upstream", "error")


def redact(text: str) -> str:
    """Mask the credentials of every URL in *text*."""
    return _URL_CREDENTIALS.sub("***@", text)


def _redact_fields(entry: dict[str, Any]) -> None:
    for key in _REDACTED_FIELDS:
        if isinstance(entry.get(key), str):
            entry[key] = redact(entry[key])


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _entry_bytes(entry: dict[str, Any]) -> int:
    try:
        return len(json.dumps(entry, default=str).encode())
    except (TypeError, ValueError):
        return 0


def _truncate_value(value: Any, max_len: int, max_items: int) -> Any:
    if isinstance(value, str) and len(value) > max_len:
        return value[:max_len] + _TRUNCATION_MARKER
    if isinstance(value, list):
        # A long conversation is mostly old turns, so the newest are kept.
        dropped = len(value) - max_items
        items = [_truncate_value(v, max_len, max_items) for v in value[max(dropped, 0):]]
        return [f"…[{dropped} earlier items truncated]", *items] if dropped > 0 else items
    if isinstance(value, dict):
        return {k: _truncate_value(v, max_len, max_items) for k, v in value.items()}
    return value


def _cap_entry(entry: dict[str, Any]) -> None:
    """Bound the serialized entry size (~50KB), mutating the dict in place.

    Strings are shortened and long lists cut to their newest items, each cut
    marked. Shortening strings alone left a conversation of a thousand short
    turns at any size, and every pass over it ran on the event loop. An
    entry still over the bound after the tightest pass gives up its largest
    lists and dicts whole — emptied, not retyped, so the visualizer still
    reads them — while the short fields that identify it are kept.
    """
    if _entry_bytes(entry) <= _MAX_ENTRY_BYTES:
        return
    for max_len, max_items in ((4000, 100), (1000, 20), (200, 5)):
        for key in list(entry):
            entry[key] = _truncate_value(entry[key], max_len, max_items)
        if _entry_bytes(entry) <= _MAX_ENTRY_BYTES:
            return
    bulky = [k for k, v in entry.items() if isinstance(v, (list, dict))]
    for key in sorted(bulky, key=lambda k: _entry_bytes({k: entry[k]}), reverse=True):
        entry[key] = [_TRUNCATION_MARKER] if isinstance(entry[key], list) else {}
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
    _redact_fields(entry)
    _cap_entry(entry)
    _CALL_LOG.append(entry)
    _notify(entry)


def update(entry: dict[str, Any]) -> None:
    """The deque holds a reference to the same dict so caller mutations are
    already reflected in get_all(). We still notify SSE subscribers."""
    _redact_fields(entry)
    _notify({"__update__": True, **entry})


def get_all() -> list[dict[str, Any]]:
    return list(reversed(_CALL_LOG))


def clear() -> None:
    _CALL_LOG.clear()
    _notify({"__clear__": True})


def subscribe() -> asyncio.Queue[dict[str, Any] | None]:
    q: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=200)
    _subscribers.append(q)
    return q


def close_streams() -> None:
    """End every live subscription: each reader stops at the None it is sent.

    Called as the server shuts down, since a subscriber left open is a
    connection uvicorn waits on. A full queue gives up its oldest entry to
    make room, as a reader that far behind has lost entries already.
    """
    for q in list(_subscribers):
        while True:
            try:
                q.put_nowait(None)
                break
            except asyncio.QueueFull:
                q.get_nowait()


def unsubscribe(q: asyncio.Queue[dict[str, Any]]) -> None:
    try:
        _subscribers.remove(q)
    except ValueError:
        pass

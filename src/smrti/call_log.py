"""Shared in-memory ring buffer for LLM call interception across all serve modes."""
from __future__ import annotations

import uuid
from collections import deque
from datetime import datetime, timezone
from typing import Any

_CALL_LOG: deque[dict[str, Any]] = deque(maxlen=200)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append(entry: dict[str, Any]) -> None:
    entry.setdefault("id", uuid.uuid4().hex[:8])
    entry.setdefault("ts", _now())
    _CALL_LOG.append(entry)


def get_all() -> list[dict[str, Any]]:
    return list(reversed(_CALL_LOG))


def clear() -> None:
    _CALL_LOG.clear()

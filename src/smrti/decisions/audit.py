"""Decision provenance: what was asked, what came back, what the code did.

A ring buffer of records, one per decision the engine made or failed to
make, served by ``GET /decisions`` on the REST and proxy servers and
counted into ``/metrics``. Shadow mode is only useful if its decisions can
be read back and compared against what the deterministic path did, which
is what this log is for; in active mode it is the record of why the graph
changed.

Records carry the model version and the usage the provider reported, so an
evaluation can say which model produced which decisions, and the state the
provider saw is kept in compact form — a memory's text, not its embedding
— under the same access rules as the memories themselves (the endpoints sit
behind ``SMRTI_API_KEY`` when it is set).
"""
from __future__ import annotations

import threading
import uuid
from collections import Counter, deque
from datetime import datetime, timezone
from typing import Any

_MAX_RECORDS = 500

_records: deque[dict[str, Any]] = deque(maxlen=_MAX_RECORDS)
_counts: Counter = Counter()
_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def record(
    *,
    task: str,
    mode: str,
    tenant_id: str,
    space: str,
    provider: str,
    model: str,
    outcome: str,
    applied: bool,
    latency_ms: float = 0.0,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cached: bool = False,
    error: str | None = None,
    summary: dict[str, Any] | None = None,
    answers: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """File one decision. ``outcome`` is what the code concluded (``skip``,
    ``explicit_update``, ``reranked``, ``unavailable`` …) and ``applied`` is
    whether that conclusion changed anything — false in shadow mode and on
    every failure."""
    entry = {
        "id": uuid.uuid4().hex[:8],
        "ts": _now(),
        "task": task,
        "mode": mode,
        "tenant_id": tenant_id,
        "space": space,
        "provider": provider,
        "model": model,
        "outcome": outcome,
        "applied": applied,
        "latency_ms": round(latency_ms, 1),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached": cached,
        "error": error,
        "summary": summary or {},
        "answers": answers or {},
    }
    with _lock:
        _records.append(entry)
        _counts[(task, mode, outcome)] += 1
    return entry


def get_all() -> list[dict[str, Any]]:
    """Every record, newest first."""
    with _lock:
        return list(reversed(_records))


def counters() -> dict[tuple[str, str, str], int]:
    """Decisions by (task, mode, outcome) since the process started — the
    counts survive ``clear()``, as a counter should."""
    with _lock:
        return dict(_counts)


def clear() -> None:
    with _lock:
        _records.clear()


def reset() -> None:
    """Records and counters both — test teardown."""
    with _lock:
        _records.clear()
        _counts.clear()

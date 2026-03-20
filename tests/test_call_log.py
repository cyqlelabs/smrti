"""Tests for the in-memory call log ring buffer."""
from __future__ import annotations

import asyncio

import pytest

import smrti.call_log as call_log


@pytest.fixture(autouse=True)
def reset_call_log():
    """Clear call log and subscribers before each test."""
    call_log._CALL_LOG.clear()
    call_log._subscribers.clear()
    yield
    call_log._CALL_LOG.clear()
    call_log._subscribers.clear()


def test_append_adds_entry():
    call_log.append({"kind": "test", "data": 1})
    assert len(call_log._CALL_LOG) == 1


def test_get_all_returns_reversed():
    call_log.append({"kind": "a"})
    call_log.append({"kind": "b"})
    result = call_log.get_all()
    assert result[0]["kind"] == "b"
    assert result[1]["kind"] == "a"


def test_get_all_empty():
    assert call_log.get_all() == []


def test_clear_empties_log():
    call_log.append({"kind": "x"})
    call_log.clear()
    assert call_log.get_all() == []


def test_subscribe_returns_queue():
    q = call_log.subscribe()
    assert q is not None
    assert len(call_log._subscribers) == 1


def test_unsubscribe_removes_queue():
    q = call_log.subscribe()
    call_log.unsubscribe(q)
    assert q not in call_log._subscribers


def test_unsubscribe_missing_queue_does_not_raise():
    q: asyncio.Queue = asyncio.Queue()
    call_log.unsubscribe(q)  # should not raise


def test_append_notifies_subscriber():
    q = call_log.subscribe()
    call_log.append({"kind": "notify_test"})
    assert not q.empty()
    msg = q.get_nowait()
    assert msg["kind"] == "notify_test"


def test_clear_notifies_subscriber():
    q = call_log.subscribe()
    call_log.clear()
    msg = q.get_nowait()
    assert msg.get("__clear__") is True


def test_update_notifies_subscriber():
    q = call_log.subscribe()
    call_log.update({"kind": "update", "id": "abc"})
    msg = q.get_nowait()
    assert msg.get("__update__") is True
    assert msg["id"] == "abc"


def test_notify_with_full_queue_does_not_raise():
    """QueueFull must be silently swallowed."""
    q: asyncio.Queue = asyncio.Queue(maxsize=1)
    call_log._subscribers.append(q)
    call_log.append({"kind": "first"})
    # Queue is full; second append must not raise
    call_log.append({"kind": "second"})


def test_append_adds_id_and_ts():
    call_log.append({"kind": "meta"})
    entry = call_log.get_all()[0]
    assert "id" in entry
    assert "ts" in entry

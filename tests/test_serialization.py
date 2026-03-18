"""Tests for per-session extraction serialization."""
from __future__ import annotations

import asyncio

import pytest

from smrti.extraction.extract import _session_locks, extract_and_link_serialized


@pytest.fixture(autouse=True)
def _clear_locks():
    """Reset module-level locks between tests."""
    _session_locks.clear()
    yield
    _session_locks.clear()


class FakeMem:
    def __init__(self, tenant_id: str = "t1", write_space: str = "s1"):
        self.tenant_id = tenant_id
        self.write_space = write_space


def _run(coro):
    return asyncio.run(coro)


def test_creates_lock_per_session_key():
    """Each (tenant_id, write_space) pair gets its own lock."""
    import smrti.extraction.extract as mod
    original = mod.extract_and_link_hybrid

    call_log: list[str] = []

    async def fake_hybrid(episode_id, content, mem, auth, model, upstream, source, mode):
        call_log.append(f"{mem.tenant_id}:{mem.write_space}")

    mod.extract_and_link_hybrid = fake_hybrid
    try:
        async def run():
            mem_a = FakeMem("t1", "s1")
            mem_b = FakeMem("t1", "s2")
            await extract_and_link_serialized("e1", "text", mem_a, "", "", "")
            await extract_and_link_serialized("e2", "text", mem_b, "", "", "")

        _run(run())
        assert "t1:s1" in _session_locks
        assert "t1:s2" in _session_locks
        assert _session_locks["t1:s1"] is not _session_locks["t1:s2"]
        assert call_log == ["t1:s1", "t1:s2"]
    finally:
        mod.extract_and_link_hybrid = original


def test_same_session_serialized():
    """Two extractions on the same session key run sequentially, not concurrently."""
    import smrti.extraction.extract as mod
    original = mod.extract_and_link_hybrid

    order: list[str] = []

    async def slow_hybrid(episode_id, content, mem, auth, model, upstream, source, mode):
        order.append(f"start:{episode_id}")
        await asyncio.sleep(0.05)
        order.append(f"end:{episode_id}")

    mod.extract_and_link_hybrid = slow_hybrid
    try:
        async def run():
            mem = FakeMem("t1", "s1")
            await asyncio.gather(
                extract_and_link_serialized("e1", "text", mem, "", "", ""),
                extract_and_link_serialized("e2", "text", mem, "", "", ""),
            )

        _run(run())
        # Because they're serialized, e1 must complete before e2 starts
        assert order == ["start:e1", "end:e1", "start:e2", "end:e2"]
    finally:
        mod.extract_and_link_hybrid = original


def test_different_sessions_concurrent():
    """Extractions on different session keys can run concurrently."""
    import smrti.extraction.extract as mod
    original = mod.extract_and_link_hybrid

    order: list[str] = []

    async def slow_hybrid(episode_id, content, mem, auth, model, upstream, source, mode):
        order.append(f"start:{episode_id}")
        await asyncio.sleep(0.05)
        order.append(f"end:{episode_id}")

    mod.extract_and_link_hybrid = slow_hybrid
    try:
        async def run():
            mem_a = FakeMem("t1", "s1")
            mem_b = FakeMem("t1", "s2")
            await asyncio.gather(
                extract_and_link_serialized("e1", "text", mem_a, "", "", ""),
                extract_and_link_serialized("e2", "text", mem_b, "", "", ""),
            )

        _run(run())
        # Both should start before either ends (concurrent)
        assert order[0].startswith("start:")
        assert order[1].startswith("start:")
    finally:
        mod.extract_and_link_hybrid = original

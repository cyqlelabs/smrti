"""Tests for the background reflect loop and SMRTI_REFLECT_INTERVAL."""
import asyncio
from unittest.mock import MagicMock, patch

import pytest

import smrti.servers.reflect_loop as rl_module
from smrti.servers.reflect_loop import run_reflect_loop


# ── helpers ──────────────────────────────────────────────────────────────────

def _mock_mem(tenant="t1", space="default"):
    mem = MagicMock()
    mem.tenant_id = tenant
    mem.write_space = space
    mem.reflect.return_value = None
    return mem


def arun(coro):
    return asyncio.run(coro)


# ── REFLECT_INTERVAL=0 disables the loop ─────────────────────────────────────

def test_disabled_when_interval_zero():
    get_instances = MagicMock(return_value=[_mock_mem()])
    with patch.object(rl_module, "REFLECT_INTERVAL", 0):
        arun(run_reflect_loop(get_instances))
    get_instances.assert_not_called()


def test_disabled_when_interval_negative():
    get_instances = MagicMock(return_value=[_mock_mem()])
    with patch.object(rl_module, "REFLECT_INTERVAL", -5):
        arun(run_reflect_loop(get_instances))
    get_instances.assert_not_called()


# ── REFLECT_INTERVAL env-var is read at module import ────────────────────────

def test_interval_defaults_to_60(monkeypatch):
    monkeypatch.delenv("SMRTI_REFLECT_INTERVAL", raising=False)
    import importlib
    reloaded = importlib.reload(rl_module)
    assert reloaded.REFLECT_INTERVAL == 60


def test_interval_overridden_by_env(monkeypatch):
    monkeypatch.setenv("SMRTI_REFLECT_INTERVAL", "30")
    import importlib
    reloaded = importlib.reload(rl_module)
    assert reloaded.REFLECT_INTERVAL == 30
    monkeypatch.setenv("SMRTI_REFLECT_INTERVAL", "60")
    importlib.reload(rl_module)


def test_interval_zero_disables_via_env(monkeypatch):
    monkeypatch.setenv("SMRTI_REFLECT_INTERVAL", "0")
    import importlib
    reloaded = importlib.reload(rl_module)
    assert reloaded.REFLECT_INTERVAL == 0
    monkeypatch.setenv("SMRTI_REFLECT_INTERVAL", "60")
    importlib.reload(rl_module)


# ── Loop calls reflect() on each instance after sleeping ─────────────────────

def test_loop_calls_reflect_after_sleep():
    mem = _mock_mem()
    iterations = [0]

    async def fake_sleep(seconds):
        iterations[0] += 1
        if iterations[0] >= 2:
            raise asyncio.CancelledError

    async def _run():
        with patch("asyncio.sleep", side_effect=fake_sleep):
            await run_reflect_loop(lambda: [mem])

    with patch.object(rl_module, "REFLECT_INTERVAL", 1):
        try:
            arun(_run())
        except asyncio.CancelledError:
            pass

    assert mem.reflect.call_count >= 1


def test_loop_calls_reflect_on_all_instances():
    mem1 = _mock_mem(tenant="t1")
    mem2 = _mock_mem(tenant="t2")
    iterations = [0]

    async def fake_sleep(_):
        iterations[0] += 1
        if iterations[0] >= 2:
            raise asyncio.CancelledError

    async def _run():
        with patch("asyncio.sleep", side_effect=fake_sleep):
            await run_reflect_loop(lambda: [mem1, mem2])

    with patch.object(rl_module, "REFLECT_INTERVAL", 1):
        try:
            arun(_run())
        except asyncio.CancelledError:
            pass

    assert mem1.reflect.called
    assert mem2.reflect.called


# ── Errors in reflect() do not crash the loop ────────────────────────────────

def test_loop_survives_reflect_exception():
    bad_mem = _mock_mem()
    bad_mem.reflect.side_effect = RuntimeError("db locked")
    ok_mem = _mock_mem(tenant="t2")
    iterations = [0]

    async def fake_sleep(_):
        iterations[0] += 1
        if iterations[0] >= 2:
            raise asyncio.CancelledError

    async def _run():
        with patch("asyncio.sleep", side_effect=fake_sleep):
            await run_reflect_loop(lambda: [bad_mem, ok_mem])

    with patch.object(rl_module, "REFLECT_INTERVAL", 1):
        try:
            arun(_run())
        except asyncio.CancelledError:
            pass

    assert ok_mem.reflect.called


# ── get_instances is called fresh each iteration ─────────────────────────────

def test_get_instances_called_each_iteration():
    mem = _mock_mem()
    call_count = [0]

    def get_instances():
        call_count[0] += 1
        return [mem]

    iterations = [0]

    async def fake_sleep(_):
        iterations[0] += 1
        if iterations[0] >= 3:
            raise asyncio.CancelledError

    async def _run():
        with patch("asyncio.sleep", side_effect=fake_sleep):
            await run_reflect_loop(get_instances)

    with patch.object(rl_module, "REFLECT_INTERVAL", 1):
        try:
            arun(_run())
        except asyncio.CancelledError:
            pass

    assert call_count[0] >= 2


# ── Empty instance list is handled gracefully ────────────────────────────────

def test_empty_instance_list():
    iterations = [0]

    async def fake_sleep(_):
        iterations[0] += 1
        if iterations[0] >= 2:
            raise asyncio.CancelledError

    async def _run():
        with patch("asyncio.sleep", side_effect=fake_sleep):
            await run_reflect_loop(lambda: [])

    with patch.object(rl_module, "REFLECT_INTERVAL", 1):
        try:
            arun(_run())
        except asyncio.CancelledError:
            pass

    assert iterations[0] >= 1


# ── Sleep duration matches REFLECT_INTERVAL ──────────────────────────────────

def test_sleep_duration_matches_interval():
    sleep_durations = []

    async def fake_sleep(seconds):
        sleep_durations.append(seconds)
        raise asyncio.CancelledError

    async def _run():
        with patch("asyncio.sleep", side_effect=fake_sleep):
            await run_reflect_loop(lambda: [])

    with patch.object(rl_module, "REFLECT_INTERVAL", 42):
        try:
            arun(_run())
        except asyncio.CancelledError:
            pass

    assert sleep_durations[0] == 42

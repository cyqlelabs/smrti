"""Pytest configuration: ensure each test gets a fresh event loop and database registry."""
import asyncio

import pytest


@pytest.fixture(autouse=True)
def reset_event_loop():
    """Create a new event loop before each test and set it as current."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield
    try:
        loop.close()
    except Exception:
        pass


@pytest.fixture(autouse=True)
def disable_live_decisions(monkeypatch):
    """Keep unit tests offline; individual policy tests exercise the active default."""
    monkeypatch.setenv("SMRTI_DECISIONS", "off")
    for task in ("ROUTING", "RERANK", "SUPERSESSION", "ENTITY"):
        monkeypatch.delenv(f"SMRTI_DECISIONS_{task}", raising=False)

    from smrti.decisions import reset_decisions

    reset_decisions(None)
    yield
    reset_decisions(None)


@pytest.fixture(autouse=True)
def reset_db_registry():
    """Close only databases opened during this test, leaving module-scoped ones intact.

    Prevents fd leaks from function-scoped tmp_db fixtures without killing
    module-scoped DB connections that are shared across tests in a module.
    """
    from smrti.core.db import _registry, _registry_lock
    with _registry_lock:
        before = set(_registry.keys())
    yield
    with _registry_lock:
        new_paths = set(_registry.keys()) - before
        entries = [(p, _registry.pop(p)) for p in new_paths]
    for _, db in entries:
        db.close()

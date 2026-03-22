"""Pytest configuration: ensure each test gets a fresh event loop and database registry."""
import asyncio
import pytest

from smrti.core.db import clear_registry


@pytest.fixture(autouse=True)
def reset_event_loop():
    """Create a new event loop before each test and set it as current.

    Prevents asyncio.run() in one test from leaving a closed loop that breaks
    subsequent tests using asyncio.get_event_loop().
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield
    try:
        loop.close()
    except Exception:
        pass


@pytest.fixture(autouse=True)
def reset_db_registry():
    """Close all Database connections and clear the registry after each test.

    The registry is process-scoped by design (for production use), but between
    tests every temp database must be fully released so file descriptors don't
    accumulate. Without this, tests that unlink their db file while connections
    remain open exhaust the fd limit and cause sqlite3.OperationalError on later
    tests.
    """
    yield
    clear_registry()

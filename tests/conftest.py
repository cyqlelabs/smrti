"""Pytest configuration: ensure each test gets a fresh event loop."""
import asyncio
import pytest


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

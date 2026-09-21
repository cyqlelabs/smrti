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


@pytest.fixture(autouse=True, scope="session")
def never_load_a_real_checkpoint():
    """No test may start a real Laya checkpoint load.

    ``LayaProvider`` loads on a background daemon thread by design, so a
    test that builds one without an agent starts a real import of torch and
    a real download. The thread outlives the test, and a daemon thread still
    inside native code when the interpreter exits takes the process with it:
    ``terminate called without an active exception``, SIGABRT, exit 134,
    after every test in the file has passed. It only shows up where laya is
    installed, which is CI and not a laptop, and only when the process exits
    soon after — which is why splitting the suite into three surfaced it and
    one long run had hidden it.

    The stub answers the import instantly and refuses, so the thread ends
    where it starts. A test that needs a particular load behaviour installs
    its own stub over this one.
    """
    import sys
    import types

    stub = types.ModuleType("laya")

    def _refuse(model, device=None):
        raise RuntimeError(
            f"the test suite never loads a real checkpoint (asked for {model!r})"
        )

    stub.load = _refuse
    previous = sys.modules.get("laya")
    sys.modules["laya"] = stub
    try:
        yield
    finally:
        if previous is None:
            sys.modules.pop("laya", None)
        else:
            sys.modules["laya"] = previous


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

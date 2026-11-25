"""Pytest configuration: ensure each test gets a fresh event loop and database registry."""
import asyncio

import pytest


@pytest.fixture(autouse=True, scope="session")
def never_open_the_users_memory(tmp_path_factory):
    """No test may open a database under ~/.smrti.

    The servers' default database is ``~/.smrti/memory.db``, the file a live
    server on the same machine is serving, and every database opened runs
    this checkout's migrations. Proxy tests that stub ``_recall`` but not
    ``get_mem`` opened that file for every run of the suite — writing a
    personality row for their tenant and, once the code under test changed
    the schema, migrating the user's memory to it. Every default now points
    at a scratch directory, and opening anything under ~/.smrti fails the
    test that tried.
    """
    import os

    from smrti.core import db as db_module
    from smrti.servers import config as cfg

    home = tmp_path_factory.mktemp("smrti-home")
    patch = pytest.MonkeyPatch()
    patch.setenv("SMRTI_DB", str(home / "memory.db"))
    patch.setenv("SMRTI_HOME", str(home))
    patch.setenv("SMRTI_RUN_DIR", str(home / "run"))
    patch.setenv("SMRTI_TOWN_DB", str(home / "town.db"))
    patch.setattr(cfg, "DB", str(home / "memory.db"))

    protected = os.path.realpath(os.path.expanduser("~/.smrti"))
    original_init = db_module.Database.__init__

    def guarded_init(self, db_path, *args, **kwargs):
        resolved = os.path.realpath(os.path.expanduser(db_path))
        if resolved == protected or resolved.startswith(protected + os.sep):
            raise RuntimeError(f"a test tried to open the user's database {db_path!r}")
        original_init(self, db_path, *args, **kwargs)

    patch.setattr(db_module.Database, "__init__", guarded_init)
    yield
    patch.undo()


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
    # A server module caches one instance per tenant and space; one whose
    # database was just closed must not serve the next test.
    import sys

    closed = {id(db) for _, db in entries}
    for name, attr in (
        ("smrti.servers.proxy", "_instances"),
        ("smrti.servers.rest", "_space_mems"),
        ("smrti.servers.rest", "_tenant_mems"),
    ):
        cache = getattr(sys.modules.get(name), attr, None)
        if cache:
            for key in [k for k, m in cache.items() if id(getattr(m, "db", None)) in closed]:
                cache.pop(key, None)

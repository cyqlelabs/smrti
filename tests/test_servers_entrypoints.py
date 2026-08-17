"""Coverage tests for server entry points and streaming edge cases.

Covers the pieces the request-level suites never reach: `run_*_server`,
the proxy lifespan and its CORS wiring, the SSE call-log stream, and the
non-`data:` branches of the streaming proxy.
"""
from __future__ import annotations

import asyncio
import importlib
import json
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from smrti import Smrti
from smrti.servers import proxy as proxy_mod
from smrti.servers import rest as rest_mod
from smrti.servers import viz_routes


def run(coro):
    return asyncio.run(coro)


async def _noop_reflect(*_args, **_kwargs):
    return


@pytest.fixture
def db_path():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture
def mem(db_path):
    engine = Smrti(db_path=db_path, tenant_id="default", write_space="default")
    yield engine
    engine.close()


# ── run_*_server ──────────────────────────────────────────────────────────────

def test_run_rest_server_hands_the_app_to_uvicorn():
    with patch("uvicorn.run") as uvicorn_run:
        rest_mod.run_rest_server(host="127.0.0.1", port=9001)
    uvicorn_run.assert_called_once_with(rest_mod.app, host="127.0.0.1", port=9001)


def test_run_proxy_server_hands_the_app_to_uvicorn():
    with patch("uvicorn.run") as uvicorn_run:
        proxy_mod.run_proxy_server(host="127.0.0.1", port=9002)
    uvicorn_run.assert_called_once_with(proxy_mod.app, host="127.0.0.1", port=9002)


def test_rest_get_mem_creates_the_instance_once(mem):
    with patch.object(rest_mod, "_mem", None):
        with patch("smrti.servers.rest.create_smrti", return_value=mem) as create:
            assert rest_mod.get_mem() is mem
            assert rest_mod.get_mem() is mem
        create.assert_called_once()


# ── proxy lifespan / CORS ─────────────────────────────────────────────────────

def test_proxy_lifespan_bootstraps_and_starts_the_reflect_loop():
    started = []

    async def _reflect(get_instances):
        started.append(get_instances())
        await asyncio.sleep(3600)

    with patch("smrti.servers.proxy._bootstrap") as bootstrap:
        with patch("smrti.servers.proxy.run_reflect_loop", new=_reflect):
            with TestClient(proxy_mod.app):
                pass
    bootstrap.assert_called_once_with()
    assert started  # the loop ran until the lifespan cancelled it


def test_proxy_adds_cors_middleware_when_origins_are_configured(monkeypatch):
    monkeypatch.setenv("SMRTI_CORS_ORIGINS", "https://app.example, https://admin.example")
    from smrti.servers import config as cfg_mod
    try:
        importlib.reload(cfg_mod)
        assert cfg_mod.CORS_ORIGINS == ["https://app.example", "https://admin.example"]
        reloaded = importlib.reload(proxy_mod)
        classes = [m.cls.__name__ for m in reloaded.app.user_middleware]
        assert "CORSMiddleware" in classes
    finally:
        # Restore the default (CORS-free) modules for the rest of the suite.
        monkeypatch.delenv("SMRTI_CORS_ORIGINS", raising=False)
        importlib.reload(cfg_mod)
        importlib.reload(proxy_mod)


# ── proxy background extraction ───────────────────────────────────────────────

def test_extract_and_link_targets_the_requested_tenant(mem):
    seen = {}

    async def _fake_extract(episode_id, content, memory, auth, model, upstream, source, mode=""):
        seen.update(
            episode_id=episode_id, content=content, memory=memory,
            auth=auth, source=source, mode=mode,
        )

    with patch("smrti.servers.proxy.get_mem", return_value=mem) as get_mem:
        with patch("smrti.extraction.extract.extract_and_link_serialized", new=_fake_extract):
            run(proxy_mod._extract_and_link(
                "ep-1", "Nico ships smrti", "acme", "notes", "Bearer k", "gpt-4o", "user",
            ))
    get_mem.assert_called_once_with("acme", "notes")
    assert seen["episode_id"] == "ep-1"
    assert seen["memory"] is mem
    assert seen["auth"] == "Bearer k"
    assert seen["source"] == "user"


def test_store_exchange_survives_a_failing_extraction(mem):
    async def _boom(*_args, **_kwargs):
        raise RuntimeError("extractor down")

    messages = [{"role": "user", "content": "Remember the deploy window"}]
    with patch("smrti.servers.proxy.get_mem", return_value=mem):
        with patch("smrti.servers.proxy._extract_and_link", new=_boom):
            with patch("smrti.servers.config.EXTRACT", True):
                run(proxy_mod._store_exchange(
                    messages, "Noted.", "default", "default", "", "gpt-4o",
                ))
    # The episode is stored even though extraction blew up.
    rows = mem.db.fetchall("SELECT content FROM atoms WHERE type = 'episode'")
    assert any("deploy window" in (r["content"] or "") for r in rows)


# ── proxy streaming edge cases ────────────────────────────────────────────────

def _stream_client(lines, status=200):
    async def _aiter_lines():
        for line in lines:
            yield line

    response = MagicMock()
    response.status_code = status
    response.aiter_lines = _aiter_lines

    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=response)
    ctx.__aexit__ = AsyncMock(return_value=None)
    client = MagicMock()
    client.stream = MagicMock(return_value=ctx)
    return client


def _collect_stream(lines):
    async def _run():
        chunks = []
        with patch("smrti.servers.proxy.get_http", return_value=_stream_client(lines)):
            with patch("smrti.servers.proxy._store_exchange", new_callable=AsyncMock):
                gen = proxy_mod._stream_proxy(
                    {"model": "gpt-4o"}, [{"role": "user", "content": "Hi"}],
                    "default", "default", {}, {"kind": "chat"}, 0.0,
                )
                async for chunk in gen:
                    chunks.append(chunk)
        return b"".join(chunks)

    return run(_run())


def test_stream_forwards_sse_comments_and_non_data_lines():
    body = _collect_stream([
        ": ping",
        "",
        "event: status",
        'data: {"choices":[{"delta":{"content":"hi"}}]}',
        "data: [DONE]",
    ])
    assert b": ping\n\n" in body
    assert b"event: status\n\n" in body
    assert b"\n" in body  # the blank line is forwarded as a bare newline
    assert body.endswith(b"data: [DONE]\n\n")


def test_stream_forwards_a_malformed_json_payload_verbatim():
    body = _collect_stream(["data: {not json", "data: [DONE]"])
    assert b"data: {not json\n\n" in body


def test_stream_reraises_cancellation():
    async def _run():
        async def _aiter_lines():
            raise asyncio.CancelledError()
            yield  # pragma: no cover

        response = MagicMock()
        response.status_code = 200
        response.aiter_lines = _aiter_lines
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=response)
        ctx.__aexit__ = AsyncMock(return_value=None)
        client = MagicMock()
        client.stream = MagicMock(return_value=ctx)

        with patch("smrti.servers.proxy.get_http", return_value=client):
            gen = proxy_mod._stream_proxy(
                {"model": "gpt-4o"}, [], "default", "default", {}, {"kind": "chat"}, 0.0,
            )
            with pytest.raises(asyncio.CancelledError):
                async for _ in gen:
                    pass

    run(_run())


# ── viz_routes: _db_mem cache ─────────────────────────────────────────────────

def test_db_mem_rejects_an_unregistered_path(tmp_path):
    from fastapi import HTTPException

    rogue = str(tmp_path / "not-registered.db")
    with pytest.raises(HTTPException) as exc:
        viz_routes._db_mem(rogue)
    assert exc.value.status_code == 403
    assert rogue in exc.value.detail
    assert not os.path.exists(rogue)


def test_db_mem_accepts_an_allowlisted_path_outside_the_registry(db_path):
    """SMRTI_VIZ_DBS lets the viz browse a DB this server never opened itself."""
    from smrti.core.db import _registry, _resolve_path, close_database

    Smrti(db_path=db_path, tenant_id="default", write_space="default").close()
    close_database(db_path)  # drop it from the registry — allowlist is the only way in

    resolved = _resolve_path(db_path)
    viz_routes._db_cache.pop(resolved, None)
    try:
        with patch("smrti.servers.config.VIZ_DBS", [db_path]):
            opened = viz_routes._db_mem(db_path)
        assert opened.db is _registry[resolved]  # bound to the allowlisted file
    finally:
        viz_routes._db_cache.pop(resolved, None)
        close_database(db_path)


def test_db_mem_rejects_an_allowlisted_path_that_does_not_exist(tmp_path):
    """A typo in SMRTI_VIZ_DBS must 404, never materialize an empty database."""
    from fastapi import HTTPException

    missing = str(tmp_path / "typo.db")
    with patch("smrti.servers.config.VIZ_DBS", [missing]):
        with pytest.raises(HTTPException) as exc:
            viz_routes._db_mem(missing)
    assert exc.value.status_code == 404
    assert not os.path.exists(missing)


def test_db_mem_drops_a_wrapper_whose_database_was_reopened(mem, db_path):
    from smrti.core.db import _resolve_path

    resolved = _resolve_path(db_path)
    first = viz_routes._db_mem(db_path)
    assert viz_routes._db_cache[resolved] is first
    # Simulate a close/re-register cycle: the cached wrapper now points at a
    # database object the registry no longer owns.
    stale = viz_routes._db_cache[resolved]
    stale.db = MagicMock()
    second = viz_routes._db_mem(db_path)
    assert second is not stale
    viz_routes._db_cache.pop(resolved, None)


def test_db_mem_evicts_the_oldest_entry_when_full(mem, db_path):
    from smrti.core.db import _resolve_path

    resolved = _resolve_path(db_path)
    viz_routes._db_cache.clear()
    for i in range(viz_routes._DB_CACHE_MAX):
        viz_routes._db_cache[f"/filler/{i}.db"] = MagicMock()
    viz_routes._db_mem(db_path)
    assert "/filler/0.db" not in viz_routes._db_cache
    assert resolved in viz_routes._db_cache
    viz_routes._db_cache.clear()


# ── viz_routes: endpoints ─────────────────────────────────────────────────────

@pytest.fixture
def client(mem):
    with patch.object(rest_mod, "get_mem", return_value=mem):
        with patch("smrti.servers.rest.run_reflect_loop", new=_noop_reflect):
            with patch("smrti.servers.config.EXTRACT", False):
                with TestClient(rest_mod.app) as c:
                    yield c


def test_visualizer_returns_404_when_the_asset_is_missing(client):
    with patch("smrti.servers.viz_routes.os.path.exists", return_value=False):
        response = client.get("/viz")
    assert response.status_code == 404
    assert response.json()["detail"] == "Visualizer not found"


def test_metrics_skips_personality_params_that_are_unset(client, mem):
    status = {
        "total_atoms": 2,
        "by_type": {"episode": 2},
        "personality": {
            "tenant_id": "default",
            "space": "default",
            "epoch_count": 3,
            "sti_decay_rate": 0.02,
            "valence_weight": None,
        },
    }
    with patch.object(mem, "status", return_value=status):
        response = client.get("/metrics")
    body = response.text
    assert response.status_code == 200
    assert "smrti_personality_sti_decay_rate" in body
    assert "smrti_personality_valence_weight" not in body
    assert "smrti_epoch_count" in body


# ── viz_routes: /llm-calls/stream ─────────────────────────────────────────────

async def _timeout_immediately(awaitable, timeout=None):
    """Stand-in for asyncio.wait_for that always times out, cleanly."""
    awaitable.close()
    raise asyncio.TimeoutError


def _stream_endpoint():
    """The /llm-calls/stream handler, taken from a freshly built viz router."""
    router = viz_routes.create_viz_router(lambda tenant, space: None)
    return next(r.endpoint for r in router.routes if r.path == "/llm-calls/stream")


def test_llm_call_stream_emits_appended_entries():
    from smrti import call_log

    async def _run():
        response = await _stream_endpoint()()
        assert response.media_type == "text/event-stream"
        agen = response.body_iterator
        # Start the generator so it subscribes before the entry is appended.
        pending = asyncio.create_task(agen.__anext__())
        while not call_log._subscribers:
            await asyncio.sleep(0)
        call_log.append({"kind": "extraction", "model": "local"})
        chunk = await asyncio.wait_for(pending, timeout=2)
        assert json.loads(chunk.split("data: ", 1)[1])["model"] == "local"
        await agen.aclose()

    run(_run())
    call_log.clear()


def test_llm_call_stream_sends_keepalives_while_idle():
    async def _run():
        response = await _stream_endpoint()()
        agen = response.body_iterator
        with patch("smrti.servers.viz_routes.asyncio.wait_for", new=_timeout_immediately):
            chunk = await agen.__anext__()
        assert chunk == ": keepalive\n\n"
        await agen.aclose()

    run(_run())


def test_llm_call_stream_unsubscribes_when_the_client_disconnects():
    from smrti import call_log

    async def _run():
        response = await _stream_endpoint()()
        agen = response.body_iterator
        with patch("smrti.servers.viz_routes.asyncio.wait_for", new=_timeout_immediately):
            await agen.__anext__()
        assert call_log._subscribers  # the stream registered a queue
        with pytest.raises(StopAsyncIteration):
            await agen.athrow(asyncio.CancelledError())
        assert call_log._subscribers == []

    run(_run())

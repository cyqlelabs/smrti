"""Regressions from the servers audit: tenant-bound keys, the proxy, shutdown and limits."""
import pytest
from unittest.mock import patch

from fastapi.testclient import TestClient

from smrti import Smrti, call_log

ADMIN = {"X-Api-Key": "admin-key"}
ACME = {"X-Api-Key": "acme-key"}
MULTI = {"X-Api-Key": "multi-key"}


async def _noop_reflect(*_args, **_kwargs):
    return


def _reset(rest_mod):
    rest_mod._mem = None
    rest_mod._space_mems.clear()
    rest_mod._tenant_mems.clear()


@pytest.fixture()
def keyed(tmp_path, monkeypatch):
    """A REST server on its own database with an all-tenants key and two bound keys."""
    from smrti.servers import config as cfg
    from smrti.servers import rest as rest_mod

    db = str(tmp_path / "mem.db")
    for name, value in {
        "DB": db, "TENANT_ID": "default", "SPACE": "default", "READ_SPACES": None,
        "EXTRACT": False, "IGNORE_PATTERNS": [], "API_KEY": "admin-key",
        "API_KEYS": {"acme-key": ("acme",), "multi-key": ("beta", "gamma")},
    }.items():
        monkeypatch.setattr(cfg, name, value)
    _reset(rest_mod)
    Smrti(db_path=db, tenant_id="default").remember("default tenant: the launch codes are 1234")
    Smrti(db_path=db, tenant_id="acme").remember("acme tenant: payroll runs on Fridays")
    call_log._CALL_LOG.clear()
    with patch("smrti.servers.rest.run_reflect_loop", new=_noop_reflect):
        with TestClient(rest_mod.app, raise_server_exceptions=True) as client:
            client.db = db
            yield client
    _reset(rest_mod)
    call_log._CALL_LOG.clear()


def _contents(response):
    return [n["content"] for n in response.json()["nodes"]]


def test_a_bound_key_cannot_read_another_tenants_graph_or_atoms(keyed):
    """One key used to read any tenant: /graph and /atoms took the tenant from the query."""
    other = Smrti(db_path=keyed.db, tenant_id="default").db.fetchone(
        "SELECT id FROM atoms WHERE tenant_id = 'default'"
    )["id"]

    assert keyed.get("/graph", params={"tenant_id": "default"}, headers=ACME).status_code == 403
    assert keyed.get(f"/atoms/{other}", params={"tenant_id": "default"}, headers=ACME).status_code == 403
    assert _contents(keyed.get("/graph", headers=ACME)) == ["acme tenant: payroll runs on Fridays"]
    assert _contents(keyed.get("/graph", params={"tenant_id": "acme"}, headers=ADMIN)) == [
        "acme tenant: payroll runs on Fridays"
    ]


def test_a_bound_key_reads_and_writes_its_own_tenant_when_it_names_none(keyed):
    """The default tenant of a bound key is its first, on every route that names none."""
    written = keyed.post("/remember", json={"content": "acme: invoices go out monthly"}, headers=ACME)
    tenant = Smrti(db_path=keyed.db, tenant_id="default").db.fetchone(
        "SELECT tenant_id FROM atoms WHERE id = ?", (written.json()["atom_id"],)
    )["tenant_id"]
    assert tenant == "acme"

    assert keyed.get("/status", headers=ACME).json()["total_atoms"] == 2
    assert keyed.get("/status", headers=ADMIN).json()["total_atoms"] == 1
    assert keyed.get("/tenants", headers=ACME).json() == ["acme"]
    assert keyed.get("/tenants", headers=ADMIN).json() == ["acme", "default"]
    assert 'tenant="acme"' in keyed.get("/metrics", headers=ACME).text


def test_shared_logs_are_filtered_by_tenant_and_only_the_admin_key_clears_them(keyed):
    """/llm-calls and /decisions held every tenant's conversations and queries for any key."""
    call_log.append({"kind": "proxy", "tenant_id": "acme", "original_messages": ["acme says hi"]})
    call_log.append({"kind": "proxy", "tenant_id": "default", "original_messages": ["secret"]})

    assert [e["tenant_id"] for e in keyed.get("/llm-calls", headers=ACME).json()] == ["acme"]
    assert len(keyed.get("/llm-calls", headers=ADMIN).json()) == 2
    assert keyed.delete("/llm-calls", headers=ACME).status_code == 403
    assert keyed.delete("/decisions", headers=MULTI).status_code == 403
    assert keyed.delete("/llm-calls", headers=ADMIN).status_code == 200


def test_decision_records_are_read_per_tenant(keyed, monkeypatch):
    """/decisions served every tenant's queries; it now serves the caller's tenants,
    and with no key configured the server's own tenant."""
    from smrti.decisions import audit
    from smrti.servers import config as cfg

    audit.clear()
    for tenant in ("default", "acme", "beta"):
        audit.record(task="tone", mode="active", tenant_id=tenant, space="default",
                     provider="static", model="m", outcome="kept", applied=False)

    def tenants(headers=None):
        return sorted(r["tenant_id"] for r in keyed.get("/decisions", headers=headers or {}).json())

    assert tenants(ACME) == ["acme"]
    assert tenants(MULTI) == ["beta"]
    assert tenants(ADMIN) == ["acme", "beta", "default"]
    monkeypatch.setattr(cfg, "API_KEY", "")
    monkeypatch.setattr(cfg, "API_KEYS", {})
    assert tenants() == ["default"]
    audit.clear()


def test_a_bound_key_may_not_open_another_database(keyed):
    """An alternate ?db= has tenants of its own, which no key was bound to."""
    response = keyed.get("/tenants", params={"db": keyed.db}, headers=ACME)
    assert response.status_code == 403
    assert keyed.get("/tenants", params={"db": keyed.db}, headers=ADMIN).status_code == 200


def test_the_proxy_refuses_a_tenant_header_outside_the_key_and_caps_its_length(tmp_path, monkeypatch):
    """X-Smrti-Tenant-Id selected any tenant, and accepted names of any length."""
    from smrti.servers import config as cfg
    from smrti.servers import proxy

    monkeypatch.setattr(cfg, "API_KEYS", {"acme-key": ("acme",)})
    with patch("smrti.servers.proxy._bootstrap", return_value=("default", "default", str(tmp_path / "p.db"))):
        with TestClient(proxy.app, raise_server_exceptions=True) as client:
            body = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
            refused = client.post(
                "/v1/chat/completions", json=body,
                headers={**ACME, "X-Smrti-Tenant-Id": "victim"},
            )
            assert refused.status_code == 403
            assert refused.json()["error"]["code"] == "tenant_not_allowed"

            monkeypatch.setattr(cfg, "API_KEYS", {})
            too_long = client.post(
                "/v1/chat/completions", json=body, headers={"X-Smrti-Tenant-Id": "T" * 129},
            )
            assert too_long.status_code == 400


def test_a_malformed_api_keys_entry_stops_the_server_without_echoing_the_key():
    """A bound key that grants nothing must not start silently, nor leak into the log."""
    from smrti.servers.config import _parse_api_keys

    assert _parse_api_keys(" k1:acme ; k2:beta|gamma,") == {"k1": ("acme",), "k2": ("beta", "gamma")}
    for raw in ("sekrit-without-tenants", "sekrit:", "k1:a;k1:b"):
        with pytest.raises(ValueError) as exc:
            _parse_api_keys(raw)
        assert "sekrit" not in str(exc.value)


# ── T2: the Smrti key never leaves the server ────────────────────────────────

def _proxy_upstream(seen, respond):
    """An httpx client whose transport records each upstream request and
    answers it with ``respond[0]`` — a test swaps the responder to change the reply."""
    import httpx

    def handler(request):
        seen.append(request)
        return respond[0](request)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _ok(request):
    import httpx

    return httpx.Response(200, json={"choices": [{"index": 0, "message": {"content": "ok"}}]})


@pytest.fixture()
def proxied(tmp_path, monkeypatch):
    """The proxy against a recording fake upstream, on its own database."""
    from smrti.servers import config as cfg
    from smrti.servers import proxy

    seen, respond = [], [_ok]
    monkeypatch.setattr(cfg, "EXTRACT", False)
    monkeypatch.setattr(proxy, "_http", _proxy_upstream(seen, respond))
    monkeypatch.setattr(proxy, "_UPSTREAM", "http://upstream.test")
    monkeypatch.setattr(proxy, "_default_tenant_id", None)
    monkeypatch.setattr(proxy, "_instances", type(proxy._instances)())
    monkeypatch.setattr(cfg, "DB", str(tmp_path / "proxy.db"))
    with patch("smrti.servers.proxy.run_reflect_loop", new=_noop_reflect):
        with TestClient(proxy.app, raise_server_exceptions=True) as client:
            client.seen = seen
            client.respond = respond
            client.proxy = proxy
            yield client


def _chat(client, content="hello", headers=None, **extra):
    body = {"model": "m", "messages": [{"role": "user", "content": content}], **extra}
    return client.post("/v1/chat/completions", json=body, headers=headers or {})


def test_the_proxy_forwards_an_upstream_key_but_never_the_smrti_key(proxied, monkeypatch):
    """Authorization: Bearer <SMRTI_API_KEY> was forwarded verbatim to the upstream."""
    from smrti.servers import config as cfg

    monkeypatch.setattr(cfg, "API_KEY", "smrti-key")
    assert _chat(proxied, headers={"Authorization": "Bearer smrti-key"}).status_code == 200
    assert "authorization" not in proxied.seen[-1].headers

    _chat(proxied, headers={"X-Api-Key": "smrti-key", "Authorization": "Bearer sk-upstream"})
    assert proxied.seen[-1].headers["authorization"] == "Bearer sk-upstream"


def test_rest_extraction_never_receives_the_smrti_key(keyed, monkeypatch):
    """REST /remember handed the client's Authorization — the Smrti key — to extraction."""
    from smrti.servers import config as cfg

    received = []

    async def _capture(episode_id, content, mem, auth, *args, **kwargs):
        received.append(auth)

    monkeypatch.setattr(cfg, "EXTRACT", True)
    with patch("smrti.extraction.extract.extract_and_link_serialized", _capture):
        keyed.post("/remember", json={"content": "Alice joined Acme"},
                   headers={"Authorization": "Bearer admin-key"})
        keyed.post("/remember", json={"content": "Bob joined Acme"},
                   headers={"X-Api-Key": "admin-key", "Authorization": "Bearer sk-extract"})
    assert received == ["", "Bearer sk-extract"]


def test_upstream_credentials_are_redacted_from_errors_and_the_call_log(proxied, monkeypatch):
    """A failed upstream echoed http://user:secret@host in the error body and on /llm-calls."""
    import httpx

    def _refuse(request):
        raise httpx.ConnectError("connection failed to http://user:secret@upstream.test")

    monkeypatch.setattr(proxied.proxy, "_http", httpx.AsyncClient(transport=httpx.MockTransport(_refuse)))
    monkeypatch.setattr(proxied.proxy, "_UPSTREAM", "http://user:secret@upstream.test")
    response = _chat(proxied)
    logged = proxied.get("/llm-calls").json()[0]
    assert response.status_code == 502
    assert "secret" not in response.text
    assert "secret" not in logged["upstream"] and "secret" not in logged["error"]


# ── T3 / T4: personality writes are validated, reads write nothing ───────────

def test_a_refused_personality_change_is_a_422_and_recall_keeps_working(keyed):
    """PUT /personality answered 200 with the refusal in the body, or 500 on an unknown preset."""
    bad = keyed.put("/personality", json={"action": "set", "params": {"w_sti": "abc"}}, headers=ADMIN)
    assert bad.status_code == 422
    assert "w_sti" in bad.json()["detail"]
    unknown = keyed.put("/personality", json={"action": "preset", "preset": "nonsense"}, headers=ADMIN)
    assert unknown.status_code == 422
    assert keyed.post("/recall", json={"query": "launch codes"}, headers=ADMIN).status_code == 200


def test_a_refused_mcp_personality_change_is_a_tool_error(tmp_path):
    """The MCP tool returned the refusal as an ordinary result; it is now raised, which the SDK marks isError."""
    import contextlib

    from smrti.servers import mcp as mcp_mod

    handlers, outcome = {}, {}

    class _Server:
        def __init__(self, name):
            pass

        def list_tools(self):
            return lambda fn: fn

        def call_tool(self):
            return lambda fn: handlers.setdefault("call_tool", fn)

        def create_initialization_options(self):
            return {}

        async def run(self, *_args):
            try:
                await handlers["call_tool"]("smrti_personality", {"action": "set", "params": {"w_sti": "abc"}})
            except ValueError as exc:
                outcome["error"] = str(exc)

    @contextlib.asynccontextmanager
    async def _stdio():
        yield (None, None)

    mem = Smrti(db_path=str(tmp_path / "mcp.db"))
    with patch("smrti.servers.mcp.Server", _Server), \
            patch("smrti.servers.mcp.create_smrti", return_value=mem), \
            patch("smrti.servers.mcp.stdio_server", _stdio), \
            patch("smrti.servers.mcp.run_reflect_loop", new=_noop_reflect):
        mcp_mod.run_mcp_server()
    assert "w_sti" in outcome["error"]


def test_visualizer_and_personality_reads_write_no_personality_rows(keyed, monkeypatch):
    """GET /status?db=<own db> rewrote the live preset, and GET /personality?space=x minted rows."""
    from smrti.servers import config as cfg
    from smrti.servers import rest as rest_mod

    monkeypatch.setattr(cfg, "PERSONALITY", "deterministic")
    monkeypatch.setenv("SMRTI_PERSONALITY", "deterministic")
    _reset(rest_mod)
    db = rest_mod.get_mem().db

    def preset():
        return db.fetchone(
            "SELECT preset_name FROM personality WHERE tenant_id = 'default' AND space = 'default'"
        )["preset_name"]

    def rows():
        return db.fetchone("SELECT COUNT(*) AS n FROM personality")["n"]

    assert preset() == "deterministic"
    before = rows()
    keyed.get("/status", params={"db": keyed.db}, headers=ADMIN)
    keyed.get("/metrics", params={"db": keyed.db}, headers=ADMIN)
    for i in range(3):
        assert keyed.get("/personality", params={"space": f"ghost{i}"}, headers=ADMIN).json() == {}
    assert preset() == "deterministic"
    assert rows() == before


# ── T21: the proxy speaks chat completions correctly ─────────────────────────

def _episodes(client):
    """Every episode the proxy stored, once its background stores have run."""
    import asyncio

    proxy = client.proxy

    async def _settle():
        while proxy._background_tasks:
            await asyncio.gather(*list(proxy._background_tasks), return_exceptions=True)

    client.portal.call(_settle)  # the stores run on the TestClient's own loop
    mem = proxy.get_mem(*proxy._bootstrap()[:2])
    # Sorted: the two turns of an exchange are stored concurrently.
    return sorted(r["content"] for r in mem.db.fetchall(
        "SELECT content FROM atoms WHERE type = 'episode'"
    ))


def _sse(*chunks, done=True):
    import json

    import httpx

    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + ("data: [DONE]\n\n" if done else "")
    return lambda request: httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})


def test_a_refused_stream_passes_its_status_and_retry_after_through(proxied):
    """An upstream 429 on a stream reached the client as a 200 with a bare JSON line."""
    import httpx

    proxied.respond[0] = lambda request: httpx.Response(
        429, json={"error": {"message": "rate limited"}}, headers={"retry-after": "30"}
    )
    response = _chat(proxied, "stream please", stream=True)
    assert response.status_code == 429
    assert response.headers["retry-after"] == "30"
    assert response.json()["error"]["message"] == "rate limited"
    assert _episodes(proxied) == []


def test_multipart_turns_are_recalled_stored_and_kept_as_parts(proxied):
    """Content parts were ignored for recall and storage, stringified in the system
    prompt, and a list-valued reply raised and lost both turns."""
    import json

    import httpx

    proxied.respond[0] = lambda request: httpx.Response(200, json={"choices": [{"index": 0, "message": {
        "content": [{"type": "text", "text": "A reply in parts"}]}}]})
    messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are terse."}]},
        {"role": "user", "content": "an earlier question"},
        {"role": "user", "content": [
            {"type": "text", "text": "What is in this picture?"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        ]},
    ]
    proxied.proxy.get_mem(*proxied.proxy._bootstrap()[:2]).remember("pictures of the harbour at dawn")
    proxied.post("/v1/chat/completions", json={"model": "m", "messages": messages})

    sent = json.loads(proxied.seen[-1].read())["messages"][0]["content"]
    assert sent[0] == {"type": "text", "text": "You are terse."}
    assert "harbour" in sent[1]["text"]
    assert _episodes(proxied) == ["A reply in parts", "What is in this picture?", "pictures of the harbour at dawn"]


def test_only_choice_zero_of_an_n2_stream_is_stored(proxied):
    """Interleaved choices of an n=2 stream were spliced into one stored reply."""
    proxied.respond[0] = _sse(
        {"choices": [{"index": 0, "delta": {"content": "Alpha"}}]},
        {"choices": [{"index": 1, "delta": {"content": "Beta"}}]},
        {"choices": [{"index": 0, "delta": {"content": " one"}}]},
        {"choices": [{"index": 1, "delta": {"content": " two"}}]},
    )
    _chat(proxied, "give me two", stream=True, n=2)
    assert _episodes(proxied) == ["Alpha one", "give me two"]


def test_only_a_complete_answer_is_stored(proxied):
    """A failed request stored its user turn while a stream cut short stored nothing."""
    import httpx

    proxied.respond[0] = lambda request: httpx.Response(500, json={"error": {"message": "boom"}})
    assert _chat(proxied, "this request failed").status_code == 500
    proxied.respond[0] = _sse({"choices": [{"index": 0, "delta": {"content": "cut off"}}]}, done=False)
    _chat(proxied, "this stream was cut off", stream=True)
    assert _episodes(proxied) == []

    proxied.respond[0] = _sse(
        {"choices": [{"index": 0, "delta": {"content": "finished"}, "finish_reason": "stop"}]}, done=False
    )
    _chat(proxied, "this stream ended without DONE", stream=True)
    assert _episodes(proxied) == ["finished", "this stream ended without DONE"]


def test_models_are_passed_through_and_a_v1_upstream_is_not_doubled(proxied, monkeypatch):
    """Only chat completions were routed, and an upstream ending in /v1 got /v1/v1/…."""
    import httpx

    monkeypatch.setattr(proxied.proxy, "_UPSTREAM", "http://upstream.test/v1/")
    proxied.respond[0] = lambda request: httpx.Response(200, json={"data": [{"id": "m"}]})
    models = proxied.get("/v1/models")
    assert models.status_code == 200 and models.json()["data"] == [{"id": "m"}]
    assert proxied.seen[-1].method == "GET"
    assert proxied.seen[-1].url.path == "/v1/models"
    proxied.respond[0] = _ok
    _chat(proxied)
    assert proxied.seen[-1].url.path == "/v1/chat/completions"


def test_malformed_chat_bodies_are_a_400(proxied):
    """A body that is not a JSON object with a list of messages was a 500."""
    url = "/v1/chat/completions"
    assert proxied.post(url, content=b"{not json", headers={"content-type": "application/json"}).status_code == 400
    assert proxied.post(url, json=[1, 2]).status_code == 400
    assert proxied.post(url, json={"model": "m", "messages": "hi"}).status_code == 400
    assert proxied.post(url, json={"model": "m", "messages": ["hi"]}).status_code == 400


def test_background_stores_wait_for_room_instead_of_growing_without_bound(monkeypatch):
    """The set of in-flight stores and extractions grew with every request."""
    import asyncio

    from smrti.servers import proxy

    async def _run():
        monkeypatch.setattr(proxy, "_BACKGROUND_MAX", 2)
        gates = [asyncio.Event() for _ in range(2)]
        for gate in gates:
            proxy._spawn(gate.wait())
        waiting = asyncio.create_task(proxy._make_room())
        await asyncio.sleep(0.01)
        assert not waiting.done()
        gates[0].set()
        await asyncio.wait_for(waiting, timeout=1)
        gates[1].set()
        await asyncio.gather(*list(proxy._background_tasks))

    asyncio.run(_run())


# ── T14 (proxy half): a stated rule is injected the way it was stated ────────

def test_a_negated_warning_is_not_injected_under_a_second_negation():
    """'YOU MUST NOT: Never deploy…' told the model to deploy; the imperative is now the preamble's."""
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    from smrti.core.models import Atom, AtomType, RecallResult, TruthValue, Valence
    from smrti.servers import proxy

    rule = Atom(
        type=AtomType.EPISODE, label="rule", content="Never deploy on Friday without a backup",
        truth=TruthValue(probability=0.9, confidence=0.8),
        valence=Valence(valence=-0.9, intensity=0.9), metadata={"valence_stated": True},
    )
    body = {"messages": [{"role": "user", "content": "Can I deploy today?"}]}
    with patch.object(proxy, "_recall", AsyncMock(return_value=[RecallResult(atom=rule, salience=1, similarity=1)])), \
            patch.object(proxy, "get_mem", return_value=MagicMock()):
        _, injection, _ = asyncio.run(proxy._inject_context(body, "t", "s", ["s"]))

    assert "YOU MUST NOT:" not in injection
    assert "- Constraint: Never deploy on Friday without a backup (confidence: high)" in injection
    assert injection.count("YOU MUST NOT") == 1
    assert "Outdated" not in injection  # only the rule for a severity present is stated
    assert proxy._scrub_injected_memory(injection + "\nThe real answer.") == "The real answer."


# ── T23: the event loop stays free, and shutdown finishes ────────────────────

def test_a_slow_graph_read_does_not_stall_other_requests_and_its_limit_is_capped(keyed, monkeypatch):
    """/graph ran SQLite on the event loop, and limit=0 read the whole space."""
    import asyncio
    import time

    import httpx

    from smrti.servers import rest as rest_mod
    from smrti.servers import viz_routes

    monkeypatch.setattr(viz_routes, "_GRAPH_LIMIT_MAX", 1)
    Smrti(db_path=keyed.db, tenant_id="default").remember("a second default note")
    assert len(keyed.get("/graph", params={"limit": 0}, headers=ADMIN).json()["nodes"]) == 1

    real_graph = viz_routes._Reader.graph

    def _slow_graph(self, *args):
        time.sleep(0.5)
        return real_graph(self, *args)

    monkeypatch.setattr(viz_routes._Reader, "graph", _slow_graph)

    async def _run():
        transport = httpx.ASGITransport(app=rest_mod.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t", headers=ADMIN) as c:
            graph = asyncio.create_task(c.get("/graph"))
            await asyncio.sleep(0.05)
            ping = await c.get("/llm-calls")
            # On the loop, the graph read would have finished before the ping
            # could even start; off it, the ping returns while it still runs.
            graph_still_running = not graph.done()
            assert (await graph).status_code == 200
            return ping.status_code, graph_still_running

    assert asyncio.run(_run()) == (200, True)


def test_a_call_log_entry_is_bounded_by_bytes_however_many_turns_it_holds():
    """The cap shortened strings only: a thousand short turns stayed at 133 KB."""
    entry = {"kind": "proxy", "tenant_id": "t", "original_messages": [
        {"role": "user", "content": f"turn {i} " + "x" * 100} for i in range(1000)
    ]}
    call_log.append(entry)
    stored = call_log.get_all()[0]
    assert call_log._entry_bytes(stored) <= call_log._MAX_ENTRY_BYTES
    assert stored["original_messages"][-1]["content"].startswith("turn 999")
    assert stored["tenant_id"] == "t"
    call_log._CALL_LOG.clear()


def test_the_live_log_stream_ends_when_the_server_shuts_down():
    """An open /llm-calls/stream never ended, so uvicorn waited on it forever."""
    import asyncio

    from starlette.requests import Request

    from smrti.servers import viz_routes

    router = viz_routes.create_viz_router(lambda tenant, space: None)
    endpoint = next(r.endpoint for r in router.routes if r.path == "/llm-calls/stream")

    async def _run():
        response = await endpoint(Request({"type": "http", "method": "GET", "headers": [], "query_string": b""}))
        reading = asyncio.create_task(response.body_iterator.__anext__())
        while not call_log._subscribers:
            await asyncio.sleep(0)
        call_log.close_streams()
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(reading, timeout=1)
        assert call_log._subscribers == []

    asyncio.run(_run())


def test_rest_shutdown_drains_extractions_but_not_forever(monkeypatch):
    """REST dropped in-flight extractions at shutdown; the wait is now bounded."""
    import asyncio

    from smrti.servers import common
    from smrti.servers import rest as rest_mod

    monkeypatch.setattr(common, "DRAIN_SECONDS", 0.2)
    done = []

    async def _extraction():
        await asyncio.sleep(0.05)
        done.append(True)

    async def _run():
        with patch("smrti.servers.rest.run_reflect_loop", new=_noop_reflect):
            async with rest_mod.lifespan(rest_mod.app):
                for coro in (_extraction(), asyncio.sleep(3600)):
                    task = asyncio.create_task(coro)
                    rest_mod._background_tasks.add(task)
                    task.add_done_callback(rest_mod._background_tasks.discard)
                started = asyncio.get_running_loop().time()
            elapsed = asyncio.get_running_loop().time() - started
        for task in list(rest_mod._background_tasks):
            task.cancel()
        return elapsed

    assert asyncio.run(_run()) < 1
    assert done == [True]


def test_an_oversized_request_body_is_refused_with_413(keyed, monkeypatch):
    """REST and the proxy read a request body of any size into memory."""
    from smrti.servers import config as cfg

    monkeypatch.setattr(cfg, "MAX_BODY_BYTES", 1000)
    big = {"content": "x" * 2000}
    assert keyed.post("/remember", json=big, headers=ADMIN).status_code == 413

    def _chunks():
        yield b'{"content": "'
        yield b"x" * 2000
        yield b'"}'

    chunked = keyed.post("/remember", content=_chunks(), headers={**ADMIN, "content-type": "application/json"})
    assert chunked.status_code == 413
    assert keyed.post("/remember", json={"content": "small"}, headers=ADMIN).status_code == 200


def test_the_visualizer_takes_the_key_once_and_then_rides_a_cookie(keyed):
    """With a key set, browser navigation and EventSource sent none, so /viz was unusable."""
    assert keyed.get("/viz", params={"key": "wrong"}, follow_redirects=False).status_code == 401
    login = keyed.get("/viz", params={"key": "admin-key"}, follow_redirects=False)
    assert login.status_code == 303
    assert login.headers["location"] == "viz"
    cookie = login.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie
    assert keyed.get("/viz").status_code == 200
    assert keyed.get("/llm-calls").status_code == 200
    keyed.cookies.clear()
    assert keyed.get("/llm-calls").status_code == 401


def test_a_non_ascii_api_key_is_a_401_not_a_500(keyed):
    """compare_digest raised TypeError on a non-ASCII header value."""
    response = keyed.get("/status", headers=[(b"x-api-key", "k\xe9y".encode("latin-1"))])
    assert response.status_code == 401


def test_an_evicted_instance_with_pending_activity_gets_its_epoch(tmp_path, monkeypatch):
    """Eviction dropped the activity count, so past 64 active spaces none ever aged."""
    import asyncio

    from smrti.servers import proxy, reflect_loop

    monkeypatch.setattr(reflect_loop, "REFLECT_INTERVAL", 60)
    monkeypatch.setattr(proxy, "_INSTANCES_MAX", 2)
    monkeypatch.setattr(proxy, "_instances", type(proxy._instances)())
    db = str(tmp_path / "evict.db")
    monkeypatch.setattr(proxy, "_bootstrap", lambda: ("default", "default", db))

    def epochs(space):
        return Smrti(db_path=db, write_space=space).db.fetchone(
            "SELECT epoch_count FROM personality WHERE tenant_id = 'default' AND space = ?", (space,)
        )["epoch_count"]

    async def _run():
        proxy.get_mem("default", "busy").remember("a memory in a busy space")
        proxy.get_mem("default", "idle")
        proxy.get_mem("default", "newcomer")  # evicts "busy", which has an epoch owed
        await asyncio.gather(*list(proxy._background_tasks))

    asyncio.run(_run())
    assert epochs("busy") == 1


# ── CLI: an exposed bind is always warned about ──────────────────────────────

def test_servers_without_auth_warn_on_any_exposed_bind(monkeypatch, tmp_path, capsys):
    """`serve town` and `serve decisions` bound to 0.0.0.0 in silence, though they check no key."""
    from typer.testing import CliRunner

    from smrti import cli

    monkeypatch.setattr(cli, "_RUN_DIR", tmp_path / "run")
    monkeypatch.setenv("SMRTI_API_KEY", "set-but-irrelevant")
    monkeypatch.setenv("SMRTI_TOWN_DB", str(tmp_path / "town.db"))
    runner = CliRunner()
    with patch("smrti_town.server.serve"), \
            patch("smrti.decisions.student.serve.run_student_server"):
        town = runner.invoke(cli.app, ["serve", "town", "--host", "0.0.0.0", "--no-browser", "--port", "18530"])
        decisions = runner.invoke(
            cli.app, ["serve", "decisions", "--host", "0.0.0.0", "--port", "18531", "--model", str(tmp_path)]
        )
    for result, server in ((town, "town simulation"), (decisions, "decision server")):
        assert result.exit_code == 0, result.output
        assert f"binding the {server} to 0.0.0.0" in result.output
        assert "checks no API key" in result.output

    cli._warn_if_unauthenticated("127.0.0.1", "decision server")
    assert capsys.readouterr().err == ""

    monkeypatch.delenv("SMRTI_API_KEY")
    monkeypatch.setenv("SMRTI_API_KEYS", "k:acme")
    cli._warn_if_exposed("0.0.0.0")  # a bound key turns authentication on too
    assert capsys.readouterr().err == ""

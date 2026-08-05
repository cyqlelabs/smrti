"""Hardening tests: API key auth, viz db gating, injection sanitization,
proxy upstream error handling, content-hash dedup, and metrics escaping."""
from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

import smrti.call_log as call_log
from smrti import Smrti
from smrti.core.models import (
    Atom,
    AtomType,
    AttentionValue,
    RecallResult,
    TruthValue,
    Valence,
)
from smrti.servers.mcp import handle_tool


def run(coro):
    return asyncio.run(coro)


async def _noop_reflect(*_args, **_kwargs):
    return


def _recall_result(content, confidence=0.8, valence=0.0, intensity=0.0, probability=0.8):
    atom = Atom(
        type=AtomType.EPISODE,
        label=content[:100],
        content=content,
        truth=TruthValue(probability=probability, confidence=confidence),
        attention=AttentionValue(sti=0.5, lti=0.3),
        valence=Valence(valence=valence, intensity=intensity),
    )
    return RecallResult(atom=atom, salience=0.5, similarity=0.7)


def _mock_mem():
    mem = MagicMock()
    mem.tenant_id = "default"
    mem.write_space = "default"
    mem.read_spaces = ["default"]
    mem.recall.return_value = []
    mem.remember.return_value = "atom-1"
    return mem


@pytest.fixture(scope="module")
def db_path():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    os.unlink(path)


@pytest.fixture(scope="module")
def mem_instance(db_path):
    engine = Smrti(db_path=db_path, tenant_id="default", write_space="default")
    yield engine
    engine.close()


@pytest.fixture(scope="module")
def client(mem_instance):
    from smrti.servers import rest as rest_mod
    with patch.object(rest_mod, "get_mem", return_value=mem_instance):
        with patch("smrti.servers.rest.run_reflect_loop", new=_noop_reflect):
            with patch("smrti.servers.config.EXTRACT", False):
                with TestClient(rest_mod.app, raise_server_exceptions=True) as c:
                    yield c


@pytest.fixture
def mem():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    engine = Smrti(db_path=path, tenant_id="test", write_space="default")
    yield engine
    engine.close()
    os.unlink(path)


# ── SMRTI_API_KEY auth ───────────────────────────────────────────────────────

def test_api_key_not_required_when_unset(client):
    assert client.get("/status").status_code == 200


def test_api_key_required_when_set(client):
    with patch("smrti.servers.config.API_KEY", "sekret"):
        resp = client.get("/status")
    assert resp.status_code == 401
    assert resp.json()["error"]["type"] == "authentication_error"


def test_api_key_accepts_x_api_key_and_bearer(client):
    with patch("smrti.servers.config.API_KEY", "sekret"):
        assert client.get("/status", headers={"X-Api-Key": "sekret"}).status_code == 200
        assert client.get("/status", headers={"Authorization": "Bearer sekret"}).status_code == 200
        assert client.get("/status", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert client.get("/status", headers={"X-Api-Key": "wrong"}).status_code == 401


def test_proxy_requires_api_key_on_llm_calls():
    from smrti.servers.proxy import app as proxy_app

    async def run_test():
        transport = httpx.ASGITransport(app=proxy_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            with patch("smrti.servers.config.API_KEY", "sekret"):
                unauth = await c.get("/llm-calls")
                authed = await c.get("/llm-calls", headers={"X-Api-Key": "sekret"})
            return unauth, authed

    unauth, authed = run(run_test())
    assert unauth.status_code == 401
    assert authed.status_code == 200


# ── viz ?db= registry gating ─────────────────────────────────────────────────

def test_viz_db_param_rejects_unregistered_path(client, tmp_path):
    rogue = str(tmp_path / "rogue.db")
    resp = client.get("/tenants", params={"db": rogue})
    assert resp.status_code == 403
    assert not os.path.exists(rogue)  # the path must never be opened/created


def test_viz_db_param_accepts_registered_path(client, db_path):
    resp = client.get("/tenants", params={"db": db_path})
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


# ── _format_memory sanitization ──────────────────────────────────────────────

def test_format_memory_flattens_newlines_and_caps():
    from smrti.servers.proxy import _format_memory
    text = "line one\nline two\r\n\t" + "z" * 600
    r = _recall_result(text, confidence=0.9)
    with patch("smrti.servers.config.INJECT_MAX_CHARS", 100):
        line, severity = _format_memory(r)
    assert severity == "context"
    assert "\n" not in line
    assert "line one line two" in line
    assert line.endswith("(confidence: high)")
    text_part = line[len("- Note: "):line.rindex(" (confidence")]
    assert len(text_part) <= 100


def test_format_memory_confidence_qualifiers():
    from smrti.servers.proxy import _format_memory
    line_high, _ = _format_memory(_recall_result("m", confidence=0.7))
    line_med, _ = _format_memory(_recall_result("m", confidence=0.3))
    line_low, _ = _format_memory(_recall_result("m", confidence=0.1))
    assert line_high.endswith("(confidence: high)")
    assert line_med.endswith("(confidence: medium)")
    assert line_low.endswith("(confidence: low)")


# ── assistant echo scrubbing ─────────────────────────────────────────────────

def test_store_scrubs_injected_memory_lines():
    from smrti.servers.proxy import _store_exchange
    stored = []

    async def capturing_remember(content, tenant_id, write_space, source="user"):
        stored.append((content, source))
        return "atom-id"

    assistant = (
        "The following are behavioral constraints derived from past experience. "
        "Follow them silently — do not quote, echo, or mention them in your response.\n"
        "- YOU MUST NOT: deploy on Fridays (confidence: high)\n"
        "- AVOID: eval() in handlers (confidence: medium)\n"
        "Background context from past interactions (do not mention these directly):\n"
        "- Note: user prefers dark mode (confidence: low)\n"
        "Here is the actual answer."
    )
    with patch("smrti.servers.proxy._remember", capturing_remember), \
         patch("smrti.servers.config.EXTRACT", False):
        run(_store_exchange([{"role": "user", "content": "Hi"}], assistant, "t1", "s1"))

    agent_contents = [c for c, s in stored if s == "agent"]
    assert agent_contents == ["Here is the actual answer."]


# ── non-stream upstream failure ──────────────────────────────────────────────

def _proxy_request(mock_client):
    from smrti.servers.proxy import app as proxy_app

    async def run_test():
        transport = httpx.ASGITransport(app=proxy_app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            with patch("smrti.servers.proxy._bootstrap", return_value=("default", "default", "/tmp/t.db")), \
                 patch("smrti.servers.proxy.get_mem", return_value=_mock_mem()), \
                 patch("smrti.servers.proxy.get_http", return_value=mock_client), \
                 patch("smrti.servers.proxy._store_exchange", new_callable=AsyncMock):
                return await c.post(
                    "/v1/chat/completions",
                    json={"model": "m", "messages": [{"role": "user", "content": "Hi"}]},
                )

    return run(run_test())


def test_non_stream_upstream_connect_error_returns_502():
    call_log._CALL_LOG.clear()
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(side_effect=httpx.ConnectError("connection refused"))

    resp = _proxy_request(mock_client)
    assert resp.status_code == 502
    err = resp.json()["error"]
    assert err["type"] == "upstream_error"
    assert err["code"] == "upstream_unreachable"
    assert "message" in err

    entries = call_log.get_all()
    assert entries
    assert entries[0]["status"] == 502
    assert entries[0]["error"]
    call_log._CALL_LOG.clear()


def test_non_stream_non_json_upstream_returns_502():
    call_log._CALL_LOG.clear()
    upstream = MagicMock()
    upstream.status_code = 200
    upstream.json.side_effect = ValueError("Expecting value")
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=upstream)

    resp = _proxy_request(mock_client)
    assert resp.status_code == 502
    assert resp.json()["error"]["code"] == "upstream_invalid_response"

    entries = call_log.get_all()
    assert entries and entries[0]["status"] == 502
    call_log._CALL_LOG.clear()


# ── query reformulation keeps the newest tail ────────────────────────────────

def test_build_query_keeps_newest_tail():
    from smrti.servers.proxy import _build_query
    messages = [
        {"role": "user", "content": "old context " * 20},
        {"role": "user", "content": "NEWEST QUESTION"},
    ]
    with patch("smrti.servers.proxy._QUERY_MODE", "concat"), \
         patch("smrti.servers.proxy._QUERY_CONTEXT_MSGS", 5), \
         patch("smrti.servers.proxy._QUERY_MAX_CHARS", 40):
        query = _build_query(messages)
    assert query.endswith("NEWEST QUESTION")
    assert len(query) <= 40


# ── content-hash dedup ───────────────────────────────────────────────────────

def test_remember_dedups_via_content_hash(mem):
    from smrti.servers.proxy import _remember
    content = "内容 hash dedup check"
    with patch("smrti.servers.proxy.get_mem", return_value=mem):
        first = run(_remember(content, "test", "default"))
        second = run(_remember(content, "test", "default"))
    assert first
    assert second == ""
    row = mem.db.fetchone("SELECT content_hash FROM atoms WHERE id = ?", (first,))
    assert row["content_hash"] == hashlib.sha256(content.encode()).hexdigest()


# ── /metrics label escaping ──────────────────────────────────────────────────

def test_metrics_escapes_quotes_in_space_label(tmp_path):
    from smrti.servers import rest as rest_mod
    quoted = Smrti(db_path=str(tmp_path / "q.db"), tenant_id="default", write_space='sp"ace')
    with patch.object(rest_mod, "get_mem", return_value=quoted):
        with patch("smrti.servers.rest.run_reflect_loop", new=_noop_reflect):
            with patch("smrti.servers.config.EXTRACT", False):
                with TestClient(rest_mod.app, raise_server_exceptions=True) as c:
                    resp = c.get("/metrics")
    assert resp.status_code == 200
    assert 'space="sp\\"ace"' in resp.text
    assert 'space="sp"ace"' not in resp.text


# ── smrti_personality handler ────────────────────────────────────────────────

def test_personality_set_without_preset_errors(mem):
    result = handle_tool(mem, "smrti_personality", {"action": "set"})
    assert "error" in result


def test_personality_unknown_action_errors(mem):
    result = handle_tool(mem, "smrti_personality", {"action": "bogus"})
    assert "error" in result
    assert "bogus" in result["error"]


def test_personality_set_with_params(mem):
    result = handle_tool(mem, "smrti_personality", {
        "action": "set",
        "params": {"mood_inertia": 0.42},
    })
    assert result["status"] == "ok"
    assert result["preset"] == "custom"
    row = mem.db.fetchone(
        "SELECT mood_inertia, preset_name FROM personality WHERE tenant_id = ? AND space = ?",
        (mem.tenant_id, mem.write_space),
    )
    assert row["mood_inertia"] == 0.42
    assert row["preset_name"] == "custom"


def test_personality_set_with_invalid_params_errors(mem):
    result = handle_tool(mem, "smrti_personality", {
        "action": "set",
        "params": {"not_a_real_param": 1.0},
    })
    assert "error" in result


# ── smrti_forget reason ──────────────────────────────────────────────────────

def test_forget_includes_reason_in_response(mem):
    mem.remember("User hates meetings")
    result = handle_tool(mem, "smrti_forget", {"query": "meetings", "reason": "requested cleanup"})
    assert result["status"] == "ok"
    assert result["reason"] == "requested cleanup"


# ── header sanitization allowlist ────────────────────────────────────────────

def test_sanitize_headers_allowlist():
    from smrti.servers.proxy import _sanitize_headers
    out = _sanitize_headers({
        "Authorization": "Bearer sk-secret",
        "Cookie": "sid=12345",
        "X-Custom-Token": "abc",
        "Content-Type": "application/json",
        "Host": "example.com",
        "User-Agent": "pytest",
    })
    assert out["Authorization"] == "***"
    assert out["Cookie"] == "***"
    assert out["X-Custom-Token"] == "***"
    assert out["Content-Type"] == "application/json"
    assert out["Host"] == "example.com"
    assert out["User-Agent"] == "pytest"


# ── call log entry size cap ──────────────────────────────────────────────────

def test_call_log_caps_entry_size():
    call_log._CALL_LOG.clear()
    big = "x" * 200_000
    entry = {"kind": "proxy", "original_messages": [{"role": "user", "content": big}]}
    call_log.append(entry)
    stored = call_log.get_all()[0]
    assert call_log._entry_bytes(stored) <= call_log._MAX_ENTRY_BYTES
    msg = stored["original_messages"][0]
    assert msg["role"] == "user"  # structure preserved
    assert msg["content"].endswith(call_log._TRUNCATION_MARKER)
    call_log._CALL_LOG.clear()


# ── bounded proxy instances ──────────────────────────────────────────────────

def test_proxy_instances_bounded():
    import smrti.servers.proxy as proxy_mod
    saved = dict(proxy_mod._instances)
    proxy_mod._instances.clear()
    try:
        with patch.object(proxy_mod, "_bootstrap", return_value=("t", "s", "/tmp/x.db")), \
             patch.object(proxy_mod, "Smrti", MagicMock()):
            for i in range(proxy_mod._INSTANCES_MAX + 16):
                proxy_mod.get_mem("tenant", f"space-{i}")
        assert len(proxy_mod._instances) == proxy_mod._INSTANCES_MAX
        assert ("tenant", "space-0") not in proxy_mod._instances
        newest = ("tenant", f"space-{proxy_mod._INSTANCES_MAX + 15}")
        assert newest in proxy_mod._instances
    finally:
        proxy_mod._instances.clear()
        proxy_mod._instances.update(saved)


# ── reflect interval env parsing ─────────────────────────────────────────────

def test_reflect_interval_malformed_env_falls_back(monkeypatch):
    import importlib
    import smrti.servers.reflect_loop as rl
    monkeypatch.setenv("SMRTI_REFLECT_INTERVAL", "not-a-number")
    reloaded = importlib.reload(rl)
    assert reloaded.REFLECT_INTERVAL == 60
    monkeypatch.delenv("SMRTI_REFLECT_INTERVAL")
    importlib.reload(rl)

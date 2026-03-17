"""Tests for the OpenAI-compatible proxy server."""
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from starlette.requests import Request as StarletteRequest

from engram.servers.proxy import _agent_id_for, _inject_context, _store_exchange, app


def run(coro):
    return asyncio.run(coro)


def _mem_recall_result(content, confidence=0.8):
    r = MagicMock()
    r.atom.content = content
    r.atom.truth.confidence = confidence
    return r


def _mock_mem():
    mem = MagicMock()
    mem.agent_id = "default"
    mem.recall.return_value = []
    mem.remember.return_value = "atom-1"
    return mem


def _starlette_request(headers: dict = None):
    raw = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    scope = {
        "type": "http",
        "method": "POST",
        "headers": raw,
        "path": "/v1/chat/completions",
        "query_string": b"",
    }
    return StarletteRequest(scope)


# ── _inject_context ──────────────────────────────────────────────────────────

def test_inject_no_messages_returns_body_unchanged():
    body = {"model": "gpt-4o", "messages": []}
    with patch("engram.servers.proxy._recall", AsyncMock(return_value=[])):
        assert run(_inject_context(body, "test")) == body


def test_inject_no_user_message_returns_body_unchanged():
    body = {"messages": [{"role": "system", "content": "Be helpful."}]}
    with patch("engram.servers.proxy._recall", AsyncMock(return_value=[])):
        assert run(_inject_context(body, "test")) == body


def test_inject_no_memories_returns_body_unchanged():
    body = {"messages": [{"role": "user", "content": "Hello"}]}
    with patch("engram.servers.proxy._recall", AsyncMock(return_value=[])):
        assert run(_inject_context(body, "test")) == body


def test_inject_creates_system_message_when_none_exists():
    body = {"messages": [{"role": "user", "content": "Tell me about Python"}]}
    memories = [_mem_recall_result("User prefers Python", 0.9)]
    with patch("engram.servers.proxy._recall", AsyncMock(return_value=memories)):
        result = run(_inject_context(body, "test"))

    msgs = result["messages"]
    assert msgs[0]["role"] == "system"
    assert "User prefers Python" in msgs[0]["content"]
    assert "confidence=0.90" in msgs[0]["content"]
    assert msgs[1]["role"] == "user"


def test_inject_appends_to_existing_system_message():
    original = "You are a helpful assistant."
    body = {
        "messages": [
            {"role": "system", "content": original},
            {"role": "user", "content": "What is ML?"},
        ]
    }
    memories = [_mem_recall_result("User works on ML", 0.85)]
    with patch("engram.servers.proxy._recall", AsyncMock(return_value=memories)):
        result = run(_inject_context(body, "test"))

    content = result["messages"][0]["content"]
    assert original in content
    assert "User works on ML" in content


def test_inject_all_memories_appear_in_block():
    body = {"messages": [{"role": "user", "content": "Hello"}]}
    memories = [
        _mem_recall_result("User likes Python", 0.9),
        _mem_recall_result("User dislikes Java", 0.7),
    ]
    with patch("engram.servers.proxy._recall", AsyncMock(return_value=memories)):
        result = run(_inject_context(body, "test"))

    content = result["messages"][0]["content"]
    assert "User likes Python" in content
    assert "User dislikes Java" in content


def test_inject_uses_last_user_message_as_recall_query():
    body = {
        "messages": [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Second question"},
        ]
    }
    queries = []

    async def capturing_recall(query, agent_id):
        queries.append(query)
        return []

    with patch("engram.servers.proxy._recall", capturing_recall):
        run(_inject_context(body, "test"))

    assert queries == ["Second question"]


def test_inject_skips_multimodal_content():
    """User message with list content (vision API) should be passed through unchanged."""
    body = {
        "messages": [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "..."}}]}
        ]
    }

    async def should_not_be_called(query, agent_id):
        pytest.fail("_recall must not be called for non-string content")

    with patch("engram.servers.proxy._recall", should_not_be_called):
        result = run(_inject_context(body, "test"))

    assert result == body


# ── _store_exchange ───────────────────────────────────────────────────────────

def test_store_saves_user_messages_and_assistant_reply():
    messages = [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "What is Python?"},
    ]
    stored = []

    async def capturing_remember(content, agent_id):
        stored.append(content)

    with patch("engram.servers.proxy._remember", capturing_remember):
        run(_store_exchange(messages, "Python is a language.", "test"))

    assert "What is Python?" in stored
    assert "Python is a language." in stored
    assert "Be helpful." not in stored  # system messages are not stored


def test_store_skips_empty_assistant_text():
    messages = [{"role": "user", "content": "Hello"}]
    stored = []

    async def capturing_remember(content, agent_id):
        stored.append(content)

    with patch("engram.servers.proxy._remember", capturing_remember):
        run(_store_exchange(messages, "", "test"))

    assert stored == ["Hello"]


def test_store_nothing_when_empty():
    stored = []

    async def capturing_remember(content, agent_id):
        stored.append(content)

    with patch("engram.servers.proxy._remember", capturing_remember):
        run(_store_exchange([], "", "test"))

    assert stored == []


def test_store_passes_agent_id_to_remember():
    messages = [{"role": "user", "content": "Hi"}]
    agent_ids = []

    async def capturing_remember(content, agent_id):
        agent_ids.append(agent_id)

    with patch("engram.servers.proxy._remember", capturing_remember):
        run(_store_exchange(messages, "Hello back", "agent-42"))

    assert all(a == "agent-42" for a in agent_ids)


# ── _agent_id_for ─────────────────────────────────────────────────────────────

def test_agent_id_from_header():
    request = _starlette_request({"x-engram-agent-id": "agent-xyz"})
    with patch("engram.servers.proxy._bootstrap", return_value=("default", "/tmp/test.db")):
        assert _agent_id_for(request) == "agent-xyz"


def test_agent_id_defaults_to_bootstrap():
    request = _starlette_request({})
    with patch("engram.servers.proxy._bootstrap", return_value=("my-default", "/tmp/test.db")):
        assert _agent_id_for(request) == "my-default"


# ── HTTP endpoint: non-streaming ──────────────────────────────────────────────

def _make_upstream_response(content="Hi there", status=200):
    resp = MagicMock()
    resp.json.return_value = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    }
    resp.status_code = status
    resp.headers = {}
    return resp


def test_non_stream_returns_upstream_response():
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=_make_upstream_response("Hi there"))

    async def run_test():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            with patch("engram.servers.proxy._bootstrap", return_value=("default", "/tmp/t.db")):
                with patch("engram.servers.proxy.get_mem", return_value=_mock_mem()):
                    with patch("engram.servers.proxy.get_http", return_value=mock_client):
                        with patch("engram.servers.proxy._store_exchange", new_callable=AsyncMock):
                            return await client.post(
                                "/v1/chat/completions",
                                json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
                            )

    resp = run(run_test())
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "Hi there"


def test_non_stream_forwards_auth_header():
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=_make_upstream_response())

    async def run_test():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            with patch("engram.servers.proxy._bootstrap", return_value=("default", "/tmp/t.db")):
                with patch("engram.servers.proxy.get_mem", return_value=_mock_mem()):
                    with patch("engram.servers.proxy.get_http", return_value=mock_client):
                        with patch("engram.servers.proxy._store_exchange", new_callable=AsyncMock):
                            await client.post(
                                "/v1/chat/completions",
                                json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hi"}]},
                                headers={"Authorization": "Bearer sk-mykey"},
                            )
        return mock_client.post.call_args

    call_args = run(run_test())
    fwd = call_args.kwargs.get("headers") or call_args[1].get("headers")
    assert fwd["Authorization"] == "Bearer sk-mykey"


def test_non_stream_injects_memories_into_upstream_request():
    """Recalled memories must be present in the request forwarded upstream."""
    mock_client = AsyncMock()
    mock_client.post = AsyncMock(return_value=_make_upstream_response())

    mock_mem = _mock_mem()
    mock_mem.recall.return_value = [_mem_recall_result("User likes dark mode", 0.9)]

    async def run_test():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            with patch("engram.servers.proxy._bootstrap", return_value=("default", "/tmp/t.db")):
                with patch("engram.servers.proxy.get_mem", return_value=mock_mem):
                    with patch("engram.servers.proxy.get_http", return_value=mock_client):
                        with patch("engram.servers.proxy._store_exchange", new_callable=AsyncMock):
                            await client.post(
                                "/v1/chat/completions",
                                json={"model": "gpt-4o", "messages": [{"role": "user", "content": "Hello"}]},
                            )
        return mock_client.post.call_args

    call_args = run(run_test())
    forwarded_body = call_args.kwargs.get("json") or call_args[1].get("json")
    system_msg = next(m for m in forwarded_body["messages"] if m["role"] == "system")
    assert "User likes dark mode" in system_msg["content"]


# ── HTTP endpoint: streaming ──────────────────────────────────────────────────

def _mock_stream_client(sse_lines: list[str]):
    async def aiter_lines():
        for line in sse_lines:
            yield line

    mock_response = MagicMock()
    mock_response.aiter_lines = aiter_lines

    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=mock_response)
    ctx.__aexit__ = AsyncMock(return_value=None)

    mock_client = MagicMock()
    mock_client.stream = MagicMock(return_value=ctx)
    return mock_client


def test_stream_passes_through_sse_chunks():
    sse = [
        'data: {"id":"1","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}',
        "",
        'data: {"id":"1","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}',
        "",
        "data: [DONE]",
        "",
    ]

    async def run_test():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            with patch("engram.servers.proxy._bootstrap", return_value=("default", "/tmp/t.db")):
                with patch("engram.servers.proxy.get_mem", return_value=_mock_mem()):
                    with patch("engram.servers.proxy.get_http", return_value=_mock_stream_client(sse)):
                        with patch("engram.servers.proxy._store_exchange", new_callable=AsyncMock):
                            return await client.post(
                                "/v1/chat/completions",
                                json={
                                    "model": "gpt-4o",
                                    "messages": [{"role": "user", "content": "Hi"}],
                                    "stream": True,
                                },
                            )

    resp = run(run_test())
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers["content-type"]
    body = resp.text
    assert "Hello" in body
    assert "world" in body
    assert "[DONE]" in body


def test_stream_upstream_error_yields_sse_error_frame():
    """An exception during streaming must produce a parseable SSE error frame."""

    async def failing_aiter_lines():
        raise httpx.ConnectError("upstream down")
        yield  # makes this an async generator

    mock_response = MagicMock()
    mock_response.aiter_lines = failing_aiter_lines

    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=mock_response)
    ctx.__aexit__ = AsyncMock(return_value=None)

    mock_client = MagicMock()
    mock_client.stream = MagicMock(return_value=ctx)

    async def run_test():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            with patch("engram.servers.proxy._bootstrap", return_value=("default", "/tmp/t.db")):
                with patch("engram.servers.proxy.get_mem", return_value=_mock_mem()):
                    with patch("engram.servers.proxy.get_http", return_value=mock_client):
                        with patch("engram.servers.proxy._store_exchange", new_callable=AsyncMock):
                            return await client.post(
                                "/v1/chat/completions",
                                json={
                                    "model": "gpt-4o",
                                    "messages": [{"role": "user", "content": "Hi"}],
                                    "stream": True,
                                },
                            )

    resp = run(run_test())
    body = resp.text
    # Error frame must be a valid SSE event containing parseable JSON
    data_lines = [l[6:] for l in body.splitlines() if l.startswith("data: ") and l != "data: [DONE]"]
    assert data_lines, "expected at least one data: frame"
    err_payload = json.loads(data_lines[0])
    assert err_payload["error"]["type"] == "proxy_error"
    assert "[DONE]" in body

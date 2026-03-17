"""OpenAI-compatible proxy server that transparently injects Engram memory."""
from __future__ import annotations

import asyncio
import json
import os
from typing import AsyncIterator, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from engram import Engram
from engram.servers.mcp import create_engram

app = FastAPI(title="Engram Proxy", version="0.1.0")

# Per-agent Engram instances — avoids agent_id mutation races on a shared singleton
_agents: dict[str, Engram] = {}
_default_agent_id: Optional[str] = None
_default_db_path: Optional[str] = None
_http: Optional[httpx.AsyncClient] = None

_UPSTREAM = os.environ.get("ENGRAM_UPSTREAM_URL", "https://api.openai.com")
_RECALL_TOP_K = int(os.environ.get("ENGRAM_RECALL_TOP_K", "5"))
_RECALL_MIN_CONF = float(os.environ.get("ENGRAM_RECALL_MIN_CONFIDENCE", "0.3"))


def _bootstrap() -> tuple[str, str]:
    """Return (default_agent_id, db_path), bootstrapping from env vars once."""
    global _default_agent_id, _default_db_path
    if _default_agent_id is None:
        default = create_engram()
        _default_agent_id = default.agent_id
        _default_db_path = os.environ.get("ENGRAM_DB", "~/.engram/memory.db")
        _agents[_default_agent_id] = default
    return _default_agent_id, _default_db_path  # type: ignore[return-value]


def get_mem(agent_id: Optional[str] = None) -> Engram:
    default_id, db_path = _bootstrap()
    key = agent_id or default_id
    if key not in _agents:
        _agents[key] = Engram(
            db_path=db_path,
            personality=os.environ.get("ENGRAM_PERSONALITY", "balanced"),
            agent_id=key,
        )
    return _agents[key]


def get_http() -> httpx.AsyncClient:
    global _http
    if _http is None:
        _http = httpx.AsyncClient(timeout=httpx.Timeout(120.0))
    return _http


def _agent_id_for(request: Request) -> str:
    default_id, _ = _bootstrap()
    return request.headers.get("x-engram-agent-id") or default_id


async def _recall(query: str, agent_id: str) -> list:
    mem = get_mem(agent_id)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, lambda: mem.recall(query, top_k=_RECALL_TOP_K, min_confidence=_RECALL_MIN_CONF)
    )


async def _remember(content: str, agent_id: str) -> None:
    mem = get_mem(agent_id)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None, lambda: mem.remember(content, type="episode", probability=0.75)
    )


async def _inject_context(body: dict, agent_id: str) -> dict:
    """Recall memories relevant to the last user message and inject them into the system prompt."""
    messages: list[dict] = body.get("messages", [])
    if not messages:
        return body

    last_user = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "user"),
        None,
    )
    if not last_user or not isinstance(last_user, str):
        return body

    memories = await _recall(last_user, agent_id)
    if not memories:
        return body

    memory_block = "\n".join(
        f"- {r.atom.content} (confidence={r.atom.truth.confidence:.2f})"
        for r in memories
    )
    injection = f"Relevant context from memory:\n{memory_block}"

    messages = list(messages)
    system_idx = next((i for i, m in enumerate(messages) if m.get("role") == "system"), None)

    if system_idx is not None:
        existing = messages[system_idx]["content"]
        messages[system_idx] = {**messages[system_idx], "content": f"{existing}\n\n{injection}"}
    else:
        messages.insert(0, {"role": "system", "content": injection})

    return {**body, "messages": messages}


async def _store_exchange(messages: list[dict], assistant_text: str, agent_id: str) -> None:
    """Persist user messages and the assistant reply as episodic memories."""
    tasks = []
    for m in messages:
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            tasks.append(_remember(m["content"], agent_id))
    if assistant_text:
        tasks.append(_remember(assistant_text, agent_id))
    if tasks:
        await asyncio.gather(*tasks)


def _upstream_headers(request: Request) -> dict:
    headers = {"Content-Type": "application/json"}
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if auth:
        headers["Authorization"] = auth
    return headers


_DROP_RESPONSE_HEADERS = {"content-encoding", "transfer-encoding", "content-length"}


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: Request) -> StreamingResponse | JSONResponse:
    body: dict = await request.json()
    agent_id = _agent_id_for(request)

    body = await _inject_context(body, agent_id)
    original_messages: list[dict] = body.get("messages", [])

    if body.get("stream", False):
        return StreamingResponse(
            _stream_proxy(body, original_messages, agent_id, _upstream_headers(request)),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
        )

    return await _non_stream_proxy(body, original_messages, agent_id, _upstream_headers(request))


async def _non_stream_proxy(
    body: dict, original_messages: list[dict], agent_id: str, headers: dict
) -> JSONResponse:
    response = await get_http().post(
        f"{_UPSTREAM}/v1/chat/completions",
        headers=headers,
        json=body,
    )
    data = response.json()

    assistant_text = (
        data.get("choices", [{}])[0].get("message", {}).get("content", "")
    )
    asyncio.create_task(_store_exchange(original_messages, assistant_text, agent_id))

    passthrough_headers = {
        k: v
        for k, v in response.headers.items()
        if k.lower() not in _DROP_RESPONSE_HEADERS
    }
    return JSONResponse(content=data, status_code=response.status_code, headers=passthrough_headers)


async def _stream_proxy(
    body: dict, original_messages: list[dict], agent_id: str, headers: dict
) -> AsyncIterator[bytes]:
    accumulated: list[str] = []
    try:
        async with get_http().stream(
            "POST", f"{_UPSTREAM}/v1/chat/completions", headers=headers, json=body
        ) as upstream:
            async for line in upstream.aiter_lines():
                if not line:
                    yield b"\n"
                    continue

                if line.startswith(":"):
                    yield f"{line}\n\n".encode()
                    continue

                if not line.startswith("data: "):
                    yield f"{line}\n\n".encode()
                    continue

                payload = line[6:]

                if payload.strip() == "[DONE]":
                    asyncio.create_task(
                        _store_exchange(original_messages, "".join(accumulated), agent_id)
                    )
                    yield b"data: [DONE]\n\n"
                    return

                try:
                    chunk = json.loads(payload)
                except json.JSONDecodeError:
                    yield f"data: {payload}\n\n".encode()
                    continue

                delta = chunk.get("choices", [{}])[0].get("delta", {})
                piece = delta.get("content", "")
                if piece:
                    accumulated.append(piece)

                yield f"data: {json.dumps(chunk)}\n\n".encode()
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        err = {"error": {"message": str(exc), "type": "proxy_error"}}
        yield f"data: {json.dumps(err)}\n\ndata: [DONE]\n\n".encode()


def run_proxy_server(host: str = "0.0.0.0", port: int = 8421) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)

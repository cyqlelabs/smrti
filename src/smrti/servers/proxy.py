"""OpenAI-compatible proxy server that transparently injects Smrti memory."""
from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from smrti import Smrti
from smrti.core.models import RecallResult
from smrti.extraction.sentiment import estimate_valence
from smrti.retrieval.classify import classify_memory
from smrti.servers.mcp import create_smrti
from smrti.servers.reflect_loop import run_reflect_loop


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(run_reflect_loop(lambda: list(_instances.values())))
    yield
    task.cancel()


app = FastAPI(title="Smrti Proxy", version="0.1.0", lifespan=lifespan)

# Per-(tenant_id, write_space) Smrti instances
_instances: dict[tuple[str, str], Smrti] = {}
_default_tenant_id: Optional[str] = None
_default_write_space: Optional[str] = None
_default_db_path: Optional[str] = None
_http: Optional[httpx.AsyncClient] = None

_UPSTREAM = os.environ.get("SMRTI_UPSTREAM_URL", "https://api.openai.com")
_RECALL_TOP_K = int(os.environ.get("SMRTI_RECALL_TOP_K", "5"))
_RECALL_MIN_CONF = float(os.environ.get("SMRTI_RECALL_MIN_CONFIDENCE", "0.3"))
_QUERY_MODE = os.environ.get("SMRTI_QUERY_MODE", "concat")
_QUERY_CONTEXT_MSGS = int(os.environ.get("SMRTI_QUERY_CONTEXT_MSGS", "5"))
_QUERY_MAX_CHARS = int(os.environ.get("SMRTI_QUERY_MAX_CHARS", "500"))


def _bootstrap() -> tuple[str, str, str]:
    """Return (tenant_id, write_space, db_path), initialising from env vars once."""
    global _default_tenant_id, _default_write_space, _default_db_path
    if _default_tenant_id is None:
        default = create_smrti()
        _default_tenant_id = default.tenant_id
        _default_write_space = default.write_space
        _default_db_path = os.environ.get("SMRTI_DB", "~/.smrti/memory.db")
        _instances[(_default_tenant_id, _default_write_space)] = default
    return _default_tenant_id, _default_write_space, _default_db_path  # type: ignore[return-value]


def get_mem(tenant_id: str, write_space: str) -> Smrti:
    key = (tenant_id, write_space)
    if key not in _instances:
        _, _, db_path = _bootstrap()
        _instances[key] = Smrti(
            db_path=db_path,
            personality=os.environ.get("SMRTI_PERSONALITY", "balanced"),
            tenant_id=tenant_id,
            write_space=write_space,
        )
    return _instances[key]


def get_http() -> httpx.AsyncClient:
    global _http
    if _http is None:
        _http = httpx.AsyncClient(timeout=httpx.Timeout(120.0))
    return _http


def _parse_request_identity(request: Request) -> tuple[str, str, list[str]]:
    """Extract (tenant_id, write_space, read_spaces) from request headers."""
    default_tenant, default_space, _ = _bootstrap()

    tenant_id = request.headers.get("x-smrti-tenant-id") or default_tenant
    write_space = request.headers.get("x-smrti-write-space") or default_space

    raw_read = request.headers.get("x-smrti-read-spaces", "")
    read_spaces = [s.strip() for s in raw_read.split(",") if s.strip()]
    if not read_spaces:
        read_spaces = [write_space]

    return tenant_id, write_space, read_spaces


async def _recall(query: str, tenant_id: str, write_space: str, read_spaces: list[str]) -> list:
    mem = get_mem(tenant_id, write_space)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: mem.recall(query, top_k=_RECALL_TOP_K, min_confidence=_RECALL_MIN_CONF, read_spaces=read_spaces),
    )


async def _remember(content: str, tenant_id: str, write_space: str) -> None:
    mem = get_mem(tenant_id, write_space)
    valence = estimate_valence(content, mem.embed)
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None,
        lambda: mem.remember(content, type="episode", probability=0.75, valence=valence),
    )


def _format_memory(r: RecallResult) -> str:
    """Format a single recall result with severity-aware XML tags."""
    severity = classify_memory(r)
    content = r.atom.content or r.atom.label
    conf = r.atom.truth.confidence
    prob = r.atom.truth.probability
    if severity == "critical_warning":
        return f"<critical_warning>PAST MISTAKE: {content}. Confidence: {conf:.2f}. DO NOT repeat.</critical_warning>"
    if severity == "known_antipattern":
        return f"<known_antipattern>DISPROVEN: {content}. Probability: {prob:.2f}. Avoid this approach.</known_antipattern>"
    return f"<context>{content} (confidence={conf:.2f})</context>"


def _build_query(messages: list[dict]) -> str | None:
    """Build a recall query from conversation messages based on configured mode."""
    if _QUERY_MODE == "concat":
        recent = [
            m["content"]
            for m in messages[-_QUERY_CONTEXT_MSGS:]
            if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)
        ]
        if not recent:
            return None
        joined = " ".join(recent)
        return joined[:_QUERY_MAX_CHARS]
    # "last" mode: original behavior
    last_user = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "user"),
        None,
    )
    if not last_user or not isinstance(last_user, str):
        return None
    return last_user


async def _inject_context(body: dict, tenant_id: str, write_space: str, read_spaces: list[str]) -> dict:
    """Recall memories relevant to the conversation and inject into the system prompt."""
    messages: list[dict] = body.get("messages", [])
    if not messages:
        return body

    query = _build_query(messages)
    if not query:
        return body

    memories = await _recall(query, tenant_id, write_space, read_spaces)
    if not memories:
        return body

    memory_lines = [_format_memory(r) for r in memories]
    has_warnings = any("<critical_warning>" in l or "<known_antipattern>" in l for l in memory_lines)
    preamble = "You MUST pay attention to any <critical_warning> or <known_antipattern> items below.\n" if has_warnings else ""
    injection = f"{preamble}Relevant context from memory:\n" + "\n".join(memory_lines)

    messages = list(messages)
    system_idx = next((i for i, m in enumerate(messages) if m.get("role") == "system"), None)

    if system_idx is not None:
        existing = messages[system_idx]["content"]
        messages[system_idx] = {**messages[system_idx], "content": f"{existing}\n\n{injection}"}
    else:
        messages.insert(0, {"role": "system", "content": injection})

    return {**body, "messages": messages}


async def _store_exchange(
    messages: list[dict], assistant_text: str, tenant_id: str, write_space: str
) -> None:
    """Persist user messages and the assistant reply as episodic memories in write_space."""
    tasks = []
    for m in messages:
        if m.get("role") == "user" and isinstance(m.get("content"), str):
            tasks.append(_remember(m["content"], tenant_id, write_space))
    if assistant_text:
        tasks.append(_remember(assistant_text, tenant_id, write_space))
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
    tenant_id, write_space, read_spaces = _parse_request_identity(request)

    body = await _inject_context(body, tenant_id, write_space, read_spaces)
    original_messages: list[dict] = body.get("messages", [])

    if body.get("stream", False):
        return StreamingResponse(
            _stream_proxy(body, original_messages, tenant_id, write_space, _upstream_headers(request)),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
        )

    return await _non_stream_proxy(body, original_messages, tenant_id, write_space, _upstream_headers(request))


async def _non_stream_proxy(
    body: dict,
    original_messages: list[dict],
    tenant_id: str,
    write_space: str,
    headers: dict,
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
    asyncio.create_task(_store_exchange(original_messages, assistant_text, tenant_id, write_space))

    passthrough_headers = {
        k: v
        for k, v in response.headers.items()
        if k.lower() not in _DROP_RESPONSE_HEADERS
    }
    return JSONResponse(content=data, status_code=response.status_code, headers=passthrough_headers)


async def _stream_proxy(
    body: dict,
    original_messages: list[dict],
    tenant_id: str,
    write_space: str,
    headers: dict,
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
                        _store_exchange(original_messages, "".join(accumulated), tenant_id, write_space)
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

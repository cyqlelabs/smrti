"""OpenAI-compatible proxy server that transparently injects Smrti memory."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from smrti import Smrti
from smrti.core.models import AtomType, RecallResult
from smrti.retrieval.classify import classify_memory
from smrti.servers import config as cfg
from smrti.servers.mcp import create_smrti
from smrti.servers.reflect_loop import run_reflect_loop
from smrti.servers.viz_routes import api_key_middleware, create_viz_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    _bootstrap()  # write personality to DB before any other process can claim it
    task = asyncio.create_task(run_reflect_loop(lambda: list(_instances.values())))
    yield
    task.cancel()


app = FastAPI(title="Smrti Proxy", version="0.1.0", lifespan=lifespan)
app.middleware("http")(api_key_middleware)
if cfg.CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.CORS_ORIGINS,
        allow_methods=["GET", "DELETE"],
        allow_headers=["*"],
    )

# Per-(tenant_id, write_space) Smrti instances — bounded LRU, oldest evicted on
# overflow (wrapper only; the registry owns the underlying Database connections)
_instances: OrderedDict[tuple[str, str], Smrti] = OrderedDict()
_INSTANCES_MAX = 64
_default_tenant_id: Optional[str] = None
_default_write_space: Optional[str] = None
_default_db_path: Optional[str] = None
_http: Optional[httpx.AsyncClient] = None

# Strong references to fire-and-forget tasks so the GC cannot cancel them mid-flight
_background_tasks: set[asyncio.Task] = set()


def _spawn(coro) -> asyncio.Task:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task

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
        _default_db_path = cfg.DB
        _instances[(_default_tenant_id, _default_write_space)] = default
    return _default_tenant_id, _default_write_space, _default_db_path  # type: ignore[return-value]


def get_mem(tenant_id: str, write_space: str) -> Smrti:
    key = (tenant_id, write_space)
    if key not in _instances:
        _, _, db_path = _bootstrap()
        if key not in _instances:  # _bootstrap may have registered the default key
            while len(_instances) >= _INSTANCES_MAX:
                _instances.popitem(last=False)
            _instances[key] = Smrti(
                db_path=db_path,
                personality=cfg.PERSONALITY,
                tenant_id=tenant_id,
                write_space=write_space,
                ignore_patterns=cfg.IGNORE_PATTERNS or None,
                temporal=cfg.TEMPORAL,
            )
    _instances.move_to_end(key)
    return _instances[key]


app.include_router(create_viz_router(get_mem))


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


async def _remember(content: str, tenant_id: str, write_space: str, source: str = "user") -> str:
    mem = get_mem(tenant_id, write_space)
    if mem.is_ignored(content):
        return ""
    # Dedup: skip if an identical episode already exists for this tenant/space
    content_hash = hashlib.sha256(content.encode()).hexdigest()
    existing = mem.db.fetchone(
        "SELECT id FROM atoms WHERE tenant_id = ? AND space = ? AND type = 'episode' AND content_hash = ?",
        (tenant_id, write_space, content_hash),
    )
    if existing:
        return ""
    # Valence is left to the engine to read from the text. Passing an estimate
    # in would record it as a tone the caller stated, and a proxied
    # conversation has no caller saying anything about tone — which is how an
    # exasperated turn used to come back as a constraint the agent must obey.
    meta = {"source": source} if source != "user" else {}
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None,
        lambda: mem.remember(content, type="episode", probability=0.75, metadata=meta),
    )


async def _extract_and_link(
    episode_id: str,
    content: str,
    tenant_id: str,
    write_space: str,
    auth: str,
    model: str,
    source: str = "user",
) -> None:
    from smrti.extraction.extract import extract_and_link_serialized
    mem = get_mem(tenant_id, write_space)
    # Falling back to _UPSTREAM keeps extraction pointed wherever this proxy
    # forwards, including its own default — cfg.EXTRACT_URL only sees the env var.
    await extract_and_link_serialized(
        episode_id, content, mem, auth, cfg.EXTRACT_MODEL or model,
        cfg.EXTRACT_URL or _UPSTREAM, source, mode=cfg.EXTRACT_MODE,
    )


def _enrich_content(r: RecallResult, mem) -> str:
    """For concept atoms with no content, synthesize a description from outgoing graph edges.

    Produces e.g. "Nico [person] — works_for: GetProductized, is: senior programmer"
    so the LLM receives relational context instead of a bare entity label.
    """
    atom = r.atom
    if atom.content:
        return atom.content
    if atom.type.value not in ("concept", "belief", "goal"):
        return atom.label

    rows = mem.db.fetchall(
        "SELECT relation, target_id FROM atoms "
        "WHERE source_id = ? AND type = 'relation' AND tenant_id = ? AND space = ?",
        (atom.id, atom.tenant_id, atom.space),
    )
    target_ids = [row["target_id"] for row in rows if row["target_id"]]

    entity_qualifier = f" [{atom.entity_type.value}]" if atom.entity_type else ""

    if not target_ids:
        return f"{atom.label}{entity_qualifier}"

    ph = ",".join("?" * len(target_ids))
    target_rows = mem.db.fetchall(
        f"SELECT id, label FROM atoms WHERE id IN ({ph})", tuple(target_ids)
    )
    target_map = {t["id"]: t["label"] for t in target_rows}
    parts = [
        f"{row['relation']}: {target_map[row['target_id']]}"
        for row in rows
        if row["target_id"] in target_map
    ]
    suffix = f" — {', '.join(parts)}" if parts else ""
    return f"{atom.label}{entity_qualifier}{suffix}"


def _temporal_suffix(r: RecallResult) -> str:
    """Dates the extraction model pinned down for this memory's deixis.

    "The session is tomorrow" is a lie by the time it is recalled, and the
    resolution cannot be written back into the text without invalidating the
    embedding taken when it was stored. It is appended here instead, where the
    model reading the memory is the one who needs it.
    """
    items = r.atom.metadata.get("temporal")
    if not isinstance(items, list):
        return ""
    # The span is the model's own words echoed back, and this lands in a system
    # prompt: collapse whitespace runs so it cannot break out of its bullet
    # line, exactly as the memory text above is. The substitution stays out of
    # the f-string — a backslash inside one is a syntax error before 3.12, and
    # this package supports 3.10.
    pairs = []
    for item in items:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("text"), str)
            or not isinstance(item.get("resolved"), str)
        ):
            continue
        span = re.sub(r"\s+", " ", item["text"]).strip()
        pairs.append(f"{span} = {item['resolved']}")
    return f" [dates: {'; '.join(pairs)}]" if pairs else ""


def _format_memory(r: RecallResult, content: str | None = None) -> tuple[str, str]:
    """Format a recall result as a plain imperative instruction plus its severity."""
    severity = classify_memory(r)
    text = content if content is not None else (r.atom.content or r.atom.label)
    # Stored memory text is injected into the system prompt: collapse whitespace
    # runs (incl. newlines) so it cannot break out of its bullet line, and cap it.
    text = re.sub(r"\s+", " ", text).strip()[: cfg.INJECT_MAX_CHARS]
    # Appended after the cap: a truncated memory still needs its dates, and
    # the suffix is a handful of characters either way.
    text += _temporal_suffix(r)
    conf = r.atom.truth.confidence
    qualifier = "high" if conf >= 0.7 else "medium" if conf >= 0.3 else "low"
    if severity == "critical_warning":
        line = f"- YOU MUST NOT: {text} (confidence: {qualifier})"
    elif severity == "known_antipattern":
        line = f"- AVOID: {text} (confidence: {qualifier})"
    else:
        line = f"- Note: {text} (confidence: {qualifier})"
    return line, severity


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
        return joined[-_QUERY_MAX_CHARS:]
    # "last" mode: original behavior
    last_user = next(
        (m["content"] for m in reversed(messages) if m.get("role") == "user"),
        None,
    )
    if not last_user or not isinstance(last_user, str):
        return None
    return last_user


async def _inject_context(
    body: dict, tenant_id: str, write_space: str, read_spaces: list[str]
) -> tuple[dict, str, list[dict]]:
    """Recall memories relevant to the conversation and inject into the system prompt.

    Returns (modified_body, injected_context_text, recalled_memory_dicts).
    """
    messages: list[dict] = body.get("messages", [])
    if not messages:
        return body, "", []

    query = _build_query(messages)
    if not query:
        return body, "", []

    memories = await _recall(query, tenant_id, write_space, read_spaces)
    if not memories:
        return body, "", []

    # Filter out agent-sourced episodes — they are stored for extraction purposes
    # but should not be injected back as context (they are the LLM's own output).
    memories = [
        r for r in memories
        if not (r.atom.type == AtomType.EPISODE and r.atom.metadata.get("source") == "agent")
    ]
    if not memories:
        return body, "", []

    mem = get_mem(tenant_id, write_space)
    enriched_contents = [_enrich_content(r, mem) for r in memories]
    formatted = [_format_memory(r, c) for r, c in zip(memories, enriched_contents)]

    memory_dicts = [
        {
            "label": r.atom.label,
            "content": c,
            "severity": sev,
            "confidence": round(r.atom.truth.confidence, 3),
            "probability": round(r.atom.truth.probability, 3),
            "valence": round(r.atom.valence.valence, 3),
            "salience": round(r.salience, 3),
        }
        for r, c, (_, sev) in zip(memories, enriched_contents, formatted)
    ]

    warning_lines = [line for line, sev in formatted if sev in ("critical_warning", "known_antipattern")]
    context_lines = [line for line, sev in formatted if sev == "context"]

    parts: list[str] = []
    if warning_lines:
        parts.append(
            "The following are behavioral constraints derived from past experience. "
            "Follow them silently — do not quote, echo, or mention them in your response.\n"
            + "\n".join(warning_lines)
        )
    if context_lines:
        parts.append(
            "Background context from past interactions (do not mention these directly):\n"
            + "\n".join(context_lines)
        )
    injection = "\n\n".join(parts)

    messages = list(messages)
    system_idx = next((i for i, m in enumerate(messages) if m.get("role") == "system"), None)

    if system_idx is not None:
        existing = messages[system_idx]["content"]
        messages[system_idx] = {**messages[system_idx], "content": f"{existing}\n\n{injection}"}
    else:
        messages.insert(0, {"role": "system", "content": injection})

    return {**body, "messages": messages}, injection, memory_dicts


# Markers actually emitted by _format_memory / _inject_context — assistant echoes
# of these lines must not be stored back as agent episodes.
_MEMORY_LINE_MARKERS = ("- YOU MUST NOT:", "- AVOID:", "- Note:")
_MEMORY_PREAMBLE_MARKERS = (
    "The following are behavioral constraints derived from past experience.",
    "Background context from past interactions",
)


def _scrub_injected_memory(text: str) -> str:
    """Remove injected-memory bullet lines and section preambles from assistant text."""
    kept = [
        line
        for line in text.splitlines()
        if not line.lstrip().startswith(_MEMORY_LINE_MARKERS)
        and not line.lstrip().startswith(_MEMORY_PREAMBLE_MARKERS)
    ]
    return "\n".join(kept)


async def _store_exchange(
    messages: list[dict],
    assistant_text: str,
    tenant_id: str,
    write_space: str,
    auth: str = "",
    model: str = "",
) -> None:
    """Persist the most recent user message and the assistant reply as episodic memories.

    Only the last user message is stored to avoid duplicating conversation history on every
    request (the full history is passed in messages on every turn).
    """
    last_user = next(
        (m["content"] for m in reversed(messages)
         if m.get("role") == "user" and isinstance(m.get("content"), str)),
        None,
    )
    clean_assistant = ""
    if assistant_text:
        clean = _scrub_injected_memory(assistant_text).strip()
        if clean:
            clean_assistant = clean

    to_store: list[tuple[str, str]] = []
    if last_user:
        to_store.append((last_user, "user"))
    if clean_assistant:
        to_store.append((clean_assistant, "agent"))

    if not to_store:
        return

    episode_ids = await asyncio.gather(*[_remember(c, tenant_id, write_space, source=s) for c, s in to_store])

    if cfg.EXTRACT:
        for eid, (content, source) in zip(episode_ids, to_store):
            if eid:
                try:
                    await _extract_and_link(eid, content, tenant_id, write_space, auth, model, source)
                except Exception:
                    pass


def _upstream_headers(request: Request) -> dict:
    headers = {"Content-Type": "application/json"}
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if auth:
        headers["Authorization"] = auth
    return headers


_DROP_RESPONSE_HEADERS = {"content-encoding", "transfer-encoding", "content-length"}

# Allowlist: only structurally-interesting headers are logged verbatim; every
# other header value (cookies, API keys, custom auth schemes, …) is masked.
_LOG_HEADER_ALLOWLIST = {"content-type", "accept", "user-agent", "content-length", "host"}


def _sanitize_headers(headers: dict) -> dict:
    return {
        k: (v if k.lower() in _LOG_HEADER_ALLOWLIST else "***")
        for k, v in headers.items()
    }


@app.post("/v1/chat/completions", response_model=None)
async def chat_completions(request: Request) -> StreamingResponse | JSONResponse:
    from smrti.call_log import append as _log

    raw_body: dict = await request.json()
    tenant_id, write_space, read_spaces = _parse_request_identity(request)

    pre_inject_messages: list[dict] = raw_body.get("messages", [])
    t0 = time.monotonic()

    body, injected_context, recalled_memories = await _inject_context(
        raw_body, tenant_id, write_space, read_spaces
    )
    post_inject_messages: list[dict] = body.get("messages", [])
    upstream_hdrs = _upstream_headers(request)

    log_entry: dict = {
        "id": uuid.uuid4().hex[:8],
        "ts": datetime.now(timezone.utc).isoformat(),
        "kind": "proxy",
        "tenant_id": tenant_id,
        "write_space": write_space,
        "model": body.get("model", ""),
        "stream": bool(body.get("stream", False)),
        "upstream": _UPSTREAM,
        "request_headers": _sanitize_headers(dict(request.headers)),
        "upstream_headers": _sanitize_headers(upstream_hdrs),
        "original_messages": pre_inject_messages,
        "injected_messages": post_inject_messages,
        "injected_context": injected_context,
        "memories": recalled_memories,
        "status": 0,
        "response_snippet": "",
        "duration_ms": 0.0,
    }
    _log(log_entry)

    if body.get("stream", False):
        return StreamingResponse(
            _stream_proxy(body, post_inject_messages, tenant_id, write_space,
                          upstream_hdrs, log_entry, t0),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
        )

    return await _non_stream_proxy(body, post_inject_messages, tenant_id, write_space,
                                   upstream_hdrs, log_entry, t0)


async def _non_stream_proxy(
    body: dict,
    original_messages: list[dict],
    tenant_id: str,
    write_space: str,
    headers: dict,
    log_entry: dict,
    t0: float,
) -> JSONResponse:
    from smrti.call_log import update as _update_log

    def _upstream_error(message: str, code: str) -> JSONResponse:
        log_entry["status"] = 502
        log_entry["error"] = message
        log_entry["duration_ms"] = round((time.monotonic() - t0) * 1000, 1)
        _update_log(log_entry)
        return JSONResponse(
            status_code=502,
            content={"error": {"message": message, "type": "upstream_error", "code": code}},
        )

    try:
        response = await get_http().post(
            f"{_UPSTREAM}/v1/chat/completions",
            headers=headers,
            json=body,
        )
    except httpx.HTTPError as exc:
        return _upstream_error(f"upstream request failed: {exc}", "upstream_unreachable")
    try:
        data = response.json()
    except ValueError as exc:
        return _upstream_error(f"upstream returned non-JSON body: {exc}", "upstream_invalid_response")

    assistant_text = (
        data.get("choices", [{}])[0].get("message", {}).get("content", "")
    )
    log_entry["status"] = response.status_code
    log_entry["response_snippet"] = (assistant_text or "")[:500]
    log_entry["duration_ms"] = round((time.monotonic() - t0) * 1000, 1)
    _update_log(log_entry)

    auth = headers.get("Authorization", "")
    model = body.get("model", "")
    _spawn(_store_exchange(original_messages, assistant_text, tenant_id, write_space, auth, model))

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
    log_entry: dict,
    t0: float,
) -> AsyncIterator[bytes]:
    accumulated: list[str] = []
    auth = headers.get("Authorization", "")
    model = body.get("model", "")
    try:
        async with get_http().stream(
            "POST", f"{_UPSTREAM}/v1/chat/completions", headers=headers, json=body
        ) as upstream:
            log_entry["status"] = upstream.status_code
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
                    full_text = "".join(accumulated)
                    log_entry["response_snippet"] = full_text[:500]
                    log_entry["duration_ms"] = round((time.monotonic() - t0) * 1000, 1)
                    from smrti.call_log import update as _update_log
                    _update_log(log_entry)
                    _spawn(
                        _store_exchange(original_messages, full_text, tenant_id, write_space, auth, model)
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
        log_entry["status"] = 500
        log_entry["error"] = str(exc)
        log_entry["duration_ms"] = round((time.monotonic() - t0) * 1000, 1)
        from smrti.call_log import update as _update_log
        _update_log(log_entry)
        err = {"error": {"message": str(exc), "type": "proxy_error"}}
        yield f"data: {json.dumps(err)}\n\ndata: [DONE]\n\n".encode()


def run_proxy_server(host: str = "0.0.0.0", port: int = 8421) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)

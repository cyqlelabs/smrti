"""FastAPI server: static frontend, WebSocket for real-time ticks, REST endpoints."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from smrti_town.llm import LLMClient, LLMSettings

logger = logging.getLogger("smrti_town.server")

app = FastAPI(title="smrti-town", version="0.1.0")

# ── Global state ──────────────────────────────────────────────────────

_engine = None
_engine_task: asyncio.Task | None = None
_connected_clients: set[WebSocket] = set()
_lock = asyncio.Lock()

# LLM state (initialised once; settings updated via /settings endpoint)
_llm_settings = LLMSettings()
_llm_client = LLMClient(_llm_settings)


def _get_db_path() -> str:
    return os.environ.get("SMRTI_TOWN_DB", "~/.smrti/town.db")


def _get_tenant() -> str:
    return os.environ.get("SMRTI_TOWN_TENANT", "millbrook")


async def _broadcast(data: dict) -> None:
    """Broadcast a message to all connected WebSocket clients."""
    if not _connected_clients:
        return
    payload = json.dumps(data)
    disconnected: list[WebSocket] = []
    for ws in _connected_clients:
        try:
            await ws.send_text(payload)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        _connected_clients.discard(ws)


async def _ensure_engine():
    """Return the running engine, creating it (via LLM world gen) if needed."""
    global _engine
    if _engine is not None:
        return _engine
    async with _lock:
        # Re-check under lock (another coroutine may have created it while we waited)
        if _engine is None:
            from smrti_town.worldgen import create_engine_from_llm
            _engine = await create_engine_from_llm(
                llm_client=_llm_client,
                db_path=_get_db_path(),
                tenant_id=_get_tenant(),
            )
            _engine.set_broadcast(_broadcast)
    return _engine


# ── WebSocket ─────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    _connected_clients.add(ws)
    logger.info("WebSocket client connected. Total: %d", len(_connected_clients))

    try:
        # If no engine exists yet, tell the client we're generating before blocking
        if _engine is None:
            await ws.send_text(json.dumps({
                "type": "generating",
                "message": "Generating world…",
                "hint": (
                    f"Using {_llm_settings.model} at {_llm_settings.base_url}"
                    if _llm_settings.enabled else "Using default Millbrook scenario"
                ),
            }))

        engine = await _ensure_engine()
        init_data = engine.get_state()
        init_data["agents"] = [a.to_dict() for a in engine.agents]
        await ws.send_text(json.dumps({"type": "state", "data": init_data}))

        await _start_engine()

        while True:
            try:
                msg = await asyncio.wait_for(ws.receive_text(), timeout=30.0)
                cmd = json.loads(msg)
                await _handle_ws_command(cmd, ws)
            except asyncio.TimeoutError:
                try:
                    await ws.send_text(json.dumps({"type": "ping"}))
                except Exception:
                    break
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.warning("WebSocket error: %s", exc)
    finally:
        _connected_clients.discard(ws)
        logger.info("WebSocket client disconnected. Total: %d", len(_connected_clients))


async def _handle_ws_command(cmd: dict, ws: WebSocket) -> None:
    action = cmd.get("action")
    engine = await _ensure_engine()

    if action == "start":
        await _start_engine()
        await ws.send_text(json.dumps({"type": "ack", "action": "start"}))
    elif action == "pause":
        engine.pause()
        await ws.send_text(json.dumps({"type": "ack", "action": "pause"}))
    elif action == "resume":
        engine.resume()
        await ws.send_text(json.dumps({"type": "ack", "action": "resume"}))
    elif action == "skip":
        engine.skip_week()
        await ws.send_text(json.dumps({"type": "ack", "action": "skip"}))
    elif action == "state":
        await ws.send_text(json.dumps({"type": "state", "data": engine.get_state()}))


# ── REST: simulation control ──────────────────────────────────────────

@app.post("/start")
async def start_simulation():
    await _start_engine()
    return {"status": "started"}


@app.post("/pause")
async def pause_simulation():
    engine = await _ensure_engine()
    engine.pause()
    return {"status": "paused"}


@app.post("/resume")
async def resume_simulation():
    engine = await _ensure_engine()
    engine.resume()
    return {"status": "resumed"}


@app.post("/skip")
async def skip_week():
    engine = await _ensure_engine()
    engine.skip_week()
    return {"status": "skip_requested"}


@app.get("/state")
async def get_state():
    engine = await _ensure_engine()
    return engine.get_state()


@app.get("/agents")
async def get_agents():
    engine = await _ensure_engine()
    return [a.to_dict() for a in engine.agents]


@app.get("/agents/{name}")
async def get_agent(name: str):
    engine = await _ensure_engine()
    data = engine.get_agent(name)
    if not data:
        return JSONResponse(status_code=404, content={"error": "Agent not found"})
    return data


@app.get("/agents/{name}/memories")
async def get_agent_memories(
    name: str,
    query: str = Query(default="", description="Memory search query"),
    top_k: int = Query(default=10, ge=1, le=50),
):
    engine = await _ensure_engine()
    memories = engine.get_agent_memories(name, query=query, top_k=top_k)
    if memories is None:
        return JSONResponse(status_code=404, content={"error": "Agent not found"})
    return memories


# ── REST: LLM settings ────────────────────────────────────────────────

@app.get("/settings")
async def get_settings():
    return _llm_settings.to_dict()


@app.post("/settings")
async def update_settings(body: dict):
    global _llm_settings
    try:
        new_settings = LLMSettings.from_dict(body)
        _llm_settings = new_settings
        _llm_client.update_settings(new_settings)
        # Propagate to running engine if present
        if _engine is not None:
            _engine.llm_client = _llm_client
        return {"status": "ok", "settings": _llm_settings.to_dict()}
    except Exception as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})


# ── REST: world regeneration ──────────────────────────────────────────

@app.post("/regenerate")
async def regenerate_world():
    """Stop the current simulation and start world generation in the background.

    Returns 202 immediately so the client is not blocked by a potentially
    slow local model.  Clients receive:
      1. {"type": "reset"}        — clear UI now
      2. {"type": "generating"}   — show loading state
      3. {"type": "state", ...}   — new world ready
    """
    asyncio.create_task(_do_regenerate())
    return {"status": "regenerating"}


async def _do_regenerate() -> None:
    """Background task: teardown → world gen → restart."""
    global _engine, _engine_task

    await _broadcast({"type": "reset"})

    if _engine_task is not None and not _engine_task.done():
        _engine_task.cancel()
        try:
            await _engine_task
        except (asyncio.CancelledError, Exception):
            pass

    async with _lock:
        if _engine is not None:
            _engine.stop()
        _engine = None
        _engine_task = None

    await _broadcast({
        "type": "generating",
        "message": "Generating new world…",
        "hint": (
            f"Using {_llm_settings.model} at {_llm_settings.base_url}"
            if _llm_settings.enabled else "Using default Millbrook scenario"
        ),
    })

    try:
        engine = await _ensure_engine()
        await _start_engine()
        init_data = engine.get_state()
        init_data["agents"] = [a.to_dict() for a in engine.agents]
        await _broadcast({"type": "state", "data": init_data})
    except Exception as exc:
        logger.error("Regeneration failed: %s", exc)
        await _broadcast({"type": "error", "message": f"World generation failed: {exc}"})


# ── Engine lifecycle ──────────────────────────────────────────────────

async def _start_engine() -> None:
    global _engine_task
    engine = await _ensure_engine()
    if engine.paused:
        engine.resume()
        return
    if engine.running or (_engine_task and not _engine_task.done()):
        return
    _engine_task = asyncio.create_task(engine.run())


# ── Static files (mounted last so routes take priority) ───────────────

_static_dir = os.environ.get(
    "SMRTI_TOWN_STATIC",
    str(Path(__file__).parent / "static"),
)
if os.path.isdir(_static_dir):
    app.mount("/", StaticFiles(directory=_static_dir, html=True), name="static")


def serve(host: str = "0.0.0.0", port: int = 8430) -> None:
    """Run the server via uvicorn."""
    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    serve()

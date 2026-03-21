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

from smrti_town.scenarios.millbrook import create_millbrook

logger = logging.getLogger("smrti_town.server")

app = FastAPI(title="smrti-town", version="0.1.0")

# ── Global state ─────────────────────────────────────────────────────

_engine = None
_engine_task: asyncio.Task | None = None
_connected_clients: set[WebSocket] = set()
_lock = asyncio.Lock()


def _get_db_path() -> str:
    return os.environ.get("SMRTI_TOWN_DB", "~/.smrti/town.db")


def _get_tenant() -> str:
    return os.environ.get("SMRTI_TOWN_TENANT", "millbrook")


async def _broadcast(data: dict) -> None:
    """Broadcast a tick result to all connected WebSocket clients."""
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


def _ensure_engine():
    global _engine
    if _engine is None:
        _engine = create_millbrook(
            db_path=_get_db_path(),
            tenant_id=_get_tenant(),
        )
        _engine.set_broadcast(_broadcast)
    return _engine


# ── WebSocket ────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    _connected_clients.add(ws)
    logger.info("WebSocket client connected. Total: %d", len(_connected_clients))

    try:
        # Send current state on connect
        engine = _ensure_engine()
        init_data = engine.get_state()
        init_data["agents"] = [a.to_dict() for a in engine.agents]
        await ws.send_text(json.dumps({
            "type": "state",
            "data": init_data,
        }))

        # Auto-start the engine on first connection
        await _start_engine()

        # Keep connection alive and listen for commands
        while True:
            try:
                msg = await asyncio.wait_for(ws.receive_text(), timeout=30.0)
                cmd = json.loads(msg)
                await _handle_ws_command(cmd, ws)
            except asyncio.TimeoutError:
                # Send ping/keepalive
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
    """Handle commands received over WebSocket."""
    action = cmd.get("action")
    engine = _ensure_engine()

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


# ── REST endpoints ───────────────────────────────────────────────────

@app.post("/start")
async def start_simulation():
    await _start_engine()
    return {"status": "started"}


@app.post("/pause")
async def pause_simulation():
    engine = _ensure_engine()
    engine.pause()
    return {"status": "paused"}


@app.post("/resume")
async def resume_simulation():
    engine = _ensure_engine()
    engine.resume()
    return {"status": "resumed"}


@app.post("/skip")
async def skip_week():
    engine = _ensure_engine()
    engine.skip_week()
    return {"status": "skip_requested"}


@app.get("/state")
async def get_state():
    engine = _ensure_engine()
    return engine.get_state()


@app.get("/agents")
async def get_agents():
    engine = _ensure_engine()
    return [a.to_dict() for a in engine.agents]


@app.get("/agents/{name}")
async def get_agent(name: str):
    engine = _ensure_engine()
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
    engine = _ensure_engine()
    memories = engine.get_agent_memories(name, query=query, top_k=top_k)
    if memories is None:
        return JSONResponse(status_code=404, content={"error": "Agent not found"})
    return memories


# ── Engine lifecycle ─────────────────────────────────────────────────

async def _start_engine() -> None:
    global _engine_task
    async with _lock:
        engine = _ensure_engine()
        if engine.running and not engine.paused:
            return
        if engine.paused:
            engine.resume()
            return
        _engine_task = asyncio.create_task(engine.run())


# ── Static files (mounted last so routes take priority) ──────────────

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

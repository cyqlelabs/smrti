"""FastAPI REST server for smrti."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request
from pydantic import BaseModel, field_validator

from smrti import Smrti
from smrti.servers import config as cfg
from smrti.servers.mcp import create_smrti, handle_tool
from smrti.servers.reflect_loop import run_reflect_loop
from smrti.servers.viz_routes import api_key_middleware, create_viz_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(run_reflect_loop(lambda: [get_mem()]))
    yield
    task.cancel()


app = FastAPI(
    title="Smrti Memory API",
    description="AtomSpace-inspired memory engine for AI agents",
    version="0.1.0",
    lifespan=lifespan,
)
app.middleware("http")(api_key_middleware)

_mem: Optional[Smrti] = None

# Strong references to fire-and-forget tasks so the GC cannot cancel them mid-flight
_background_tasks: set[asyncio.Task] = set()


def get_mem() -> Smrti:
    global _mem
    if _mem is None:
        _mem = create_smrti()
    return _mem


async def _run_sync(fn, *args):
    """Run blocking work (ONNX inference, SQLite) off the event loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, fn, *args)


app.include_router(create_viz_router(lambda t, s: get_mem()))


class RememberRequest(BaseModel):
    content: str
    type: str = "episode"
    probability: float = 0.8
    valence: Optional[float] = None


class RecallRequest(BaseModel):
    query: str
    top_k: int = 10
    min_confidence: float = 0.1

    @field_validator("query")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("query must not be empty or whitespace-only")
        return v


class BelieveRequest(BaseModel):
    statement: str
    probability: float
    evidence: Optional[str] = None


class ForgetRequest(BaseModel):
    query: str
    reason: Optional[str] = None

    @field_validator("query")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("query must not be empty or whitespace-only")
        return v


class PersonalityRequest(BaseModel):
    action: str  # "get", "set", "preset"
    preset: Optional[str] = None
    params: Optional[dict] = None


@app.post("/remember")
async def remember(req: RememberRequest, request: Request):
    result = await _run_sync(handle_tool, get_mem(), "smrti_remember", req.model_dump())
    if cfg.EXTRACT:
        episode_id = result.get("atom_id", "")
        if episode_id:
            from smrti.extraction.extract import extract_and_link_serialized
            auth = request.headers.get("Authorization", "")
            task = asyncio.create_task(
                extract_and_link_serialized(episode_id, req.content, get_mem(), auth, cfg.EXTRACT_MODEL, cfg.EXTRACT_URL, mode=cfg.EXTRACT_MODE)
            )
            _background_tasks.add(task)
            task.add_done_callback(_background_tasks.discard)
    return result


@app.post("/recall")
async def recall(req: RecallRequest):
    return await _run_sync(handle_tool, get_mem(), "smrti_recall", req.model_dump())


@app.post("/reflect")
async def reflect():
    return await _run_sync(handle_tool, get_mem(), "smrti_reflect", {})


@app.post("/believe")
async def believe(req: BelieveRequest):
    return await _run_sync(handle_tool, get_mem(), "smrti_believe", req.model_dump())


@app.post("/forget")
async def forget(req: ForgetRequest):
    return await _run_sync(handle_tool, get_mem(), "smrti_forget", req.model_dump())


@app.get("/personality")
async def get_personality():
    return await _run_sync(handle_tool, get_mem(), "smrti_personality", {"action": "get"})


@app.put("/personality")
async def set_personality(req: PersonalityRequest):
    return await _run_sync(handle_tool, get_mem(), "smrti_personality", req.model_dump())


@app.delete("/spaces/current")
async def clear_current_space():
    count = await _run_sync(get_mem().clear_space)
    return {"status": "ok", "deleted": count}


def run_rest_server(host: str = "0.0.0.0", port: int = 8420) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)

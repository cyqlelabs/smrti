"""FastAPI REST server for smrti."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from smrti import Smrti
from smrti.servers.mcp import create_smrti, handle_tool
from smrti.servers.reflect_loop import run_reflect_loop


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

_mem: Optional[Smrti] = None


def get_mem() -> Smrti:
    global _mem
    if _mem is None:
        _mem = create_smrti()
    return _mem


class RememberRequest(BaseModel):
    content: str
    type: str = "episode"
    probability: float = 0.8
    valence: float = 0.0


class RecallRequest(BaseModel):
    query: str
    top_k: int = 10
    min_confidence: float = 0.1


class BelieveRequest(BaseModel):
    statement: str
    probability: float
    evidence: Optional[str] = None


class ForgetRequest(BaseModel):
    query: str
    reason: Optional[str] = None


class PersonalityRequest(BaseModel):
    action: str  # "get", "set", "preset"
    preset: Optional[str] = None
    params: Optional[dict] = None


@app.post("/remember")
async def remember(req: RememberRequest):
    return handle_tool(get_mem(), "smrti_remember", req.model_dump())


@app.post("/recall")
async def recall(req: RecallRequest):
    return handle_tool(get_mem(), "smrti_recall", req.model_dump())


@app.post("/reflect")
async def reflect():
    return handle_tool(get_mem(), "smrti_reflect", {})


@app.post("/believe")
async def believe(req: BelieveRequest):
    return handle_tool(get_mem(), "smrti_believe", req.model_dump())


@app.post("/forget")
async def forget(req: ForgetRequest):
    return handle_tool(get_mem(), "smrti_forget", req.model_dump())


@app.get("/personality")
async def get_personality():
    return handle_tool(get_mem(), "smrti_personality", {"action": "get"})


@app.put("/personality")
async def set_personality(req: PersonalityRequest):
    return handle_tool(get_mem(), "smrti_personality", req.model_dump())


@app.get("/status")
async def status():
    return handle_tool(get_mem(), "smrti_status", {})


@app.get("/atoms/{atom_id}")
async def get_atom(atom_id: str):
    mem = get_mem()
    atom = mem.atomspace.get_atom(atom_id, mem.tenant_id, mem.write_space)
    if not atom:
        raise HTTPException(status_code=404, detail="Atom not found")
    return atom.model_dump()


@app.delete("/spaces/current")
async def clear_current_space():
    count = get_mem().clear_space()
    return {"status": "ok", "deleted": count}


def run_rest_server(host: str = "0.0.0.0", port: int = 8420) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)

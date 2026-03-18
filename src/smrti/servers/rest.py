"""FastAPI REST server for smrti."""
from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from smrti import Smrti
from smrti.servers import config as cfg
from smrti.servers.mcp import create_smrti, handle_tool
from smrti.servers.reflect_loop import run_reflect_loop

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


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
async def remember(req: RememberRequest, request: Request):
    result = handle_tool(get_mem(), "smrti_remember", req.model_dump())
    if cfg.EXTRACT:
        episode_id = result.get("atom_id", "")
        if episode_id:
            from smrti.extraction.extract import extract_and_link
            auth = request.headers.get("Authorization", "")
            asyncio.create_task(
                extract_and_link(episode_id, req.content, get_mem(), auth, cfg.EXTRACT_MODEL, cfg.EXTRACT_URL)
            )
    return result


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


@app.get("/viz", include_in_schema=False)
async def visualizer():
    path = os.path.join(_STATIC_DIR, "visualizer.html")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Visualizer not found")
    return FileResponse(path, media_type="text/html")


@app.get("/tenants")
async def list_tenants():
    rows = get_mem().db.fetchall("SELECT DISTINCT tenant_id FROM atoms ORDER BY tenant_id")
    return [r["tenant_id"] for r in rows]


@app.get("/spaces")
async def list_spaces(tenant_id: str = Query("default")):
    rows = get_mem().db.fetchall(
        "SELECT DISTINCT space FROM atoms WHERE tenant_id = ? ORDER BY space",
        (tenant_id,),
    )
    return [r["space"] for r in rows]


@app.get("/graph")
async def get_graph(
    tenant_id: str = Query("default"),
    space: str = Query("default"),
    limit: int = Query(200),
    min_confidence: float = Query(0.0),
    types: str = Query("concept,belief,episode,goal"),
):
    type_list = [t.strip() for t in types.split(",") if t.strip()]
    if not type_list:
        type_list = ["concept", "belief", "episode", "goal"]
    ph = ",".join("?" * len(type_list))
    mem = get_mem()

    rows = mem.db.fetchall(
        f"""SELECT * FROM atoms
            WHERE tenant_id=? AND space=? AND type IN ({ph})
              AND confidence >= ?
            ORDER BY (sti + lti) DESC
            LIMIT ?""",
        (tenant_id, space, *type_list, min_confidence, limit),
    )
    nodes = [dict(r) for r in rows]
    node_ids = {n["id"] for n in nodes}
    node_labels = {n["id"]: n["label"] for n in nodes}

    if node_ids:
        edge_rows = mem.db.fetchall(
            """SELECT * FROM atoms
               WHERE tenant_id=? AND space=? AND type='relation'
                 AND source_id IS NOT NULL AND target_id IS NOT NULL
               LIMIT 5000""",
            (tenant_id, space),
        )
        edges = []
        for r in edge_rows:
            if r["source_id"] in node_ids and r["target_id"] in node_ids:
                e = dict(r)
                e["source_label"] = node_labels.get(r["source_id"])
                e["target_label"] = node_labels.get(r["target_id"])
                edges.append(e)
    else:
        edges = []

    return {"nodes": nodes, "edges": edges}


def run_rest_server(host: str = "0.0.0.0", port: int = 8420) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)

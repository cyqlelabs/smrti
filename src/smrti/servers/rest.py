"""FastAPI REST server for smrti."""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Annotated, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field, field_validator

from smrti import Smrti
from smrti.servers import config as cfg
from smrti.servers.mcp import create_smrti, handle_tool
from smrti.servers.reflect_loop import run_reflect_loop
from smrti.servers.viz_routes import api_key_middleware, create_viz_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(run_reflect_loop(_all_mems))
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

# Per-space Smrti instances minted for requests naming a space other than the
# configured one — bounded LRU, oldest evicted on overflow (wrapper only; the
# registry owns the underlying Database connections).
_space_mems: OrderedDict[str, Smrti] = OrderedDict()
_SPACE_MEMS_MAX = 64

# Strong references to fire-and-forget tasks so the GC cannot cancel them mid-flight
_background_tasks: set[asyncio.Task] = set()


def get_mem(space: Optional[str] = None) -> Smrti:
    """Return the Smrti instance for *space* — the env-configured one when unset.

    An unset or empty space keeps the exact pre-space behavior (including the
    SMRTI_READ_SPACES overlay); any other name gets its own instance writing to
    that space and reading only it unless the request says otherwise.

    Called from the event loop thread only — every route resolves its instance
    before handing the blocking work to ``_run_sync``, so ``_space_mems`` needs
    no lock. Keep it that way: minting also writes this space's personality
    row, and moving that into the executor would race the cache.
    """
    global _mem
    if _mem is None:
        _mem = create_smrti()
    if not space or space == _mem.write_space:
        return _mem
    if space not in _space_mems:
        while len(_space_mems) >= _SPACE_MEMS_MAX:
            _space_mems.popitem(last=False)
        _space_mems[space] = Smrti(
            db_path=cfg.DB,
            personality=cfg.PERSONALITY,
            tenant_id=cfg.TENANT_ID,
            write_space=space,
            ignore_patterns=cfg.IGNORE_PATTERNS or None,
            temporal=cfg.TEMPORAL,
        )
    _space_mems.move_to_end(space)
    return _space_mems[space]


def _all_mems() -> list[Smrti]:
    """Every live instance, for the reflect loop — each space gets its own epochs."""
    mems: list[Smrti] = [get_mem()]
    mems.extend(_space_mems.values())
    return mems


async def _run_sync(fn, *args):
    """Run blocking work (ONNX inference, SQLite) off the event loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, fn, *args)


def _other_space_mem(space: Optional[str], other_space: str) -> Smrti:
    """Resolve the instance for a two-space operation, refusing a self-compare.

    Both operations are defined between *different* spaces. Against itself
    overlap is the trivial 1.0 — bought with a 500-atom scan and an embedding
    pass per atom — and a merge is worse than useless: it would materialize a
    ``x_x_x`` bridge of every atom paired with itself, permanently, in a space
    nothing else writes. Refusing costs the caller one 400 and no graph.
    """
    mem = get_mem(space)
    if other_space == mem.write_space:
        raise HTTPException(
            status_code=400,
            detail=f"other_space must differ from the write space ({mem.write_space})",
        )
    return mem


app.include_router(create_viz_router(lambda t, s: get_mem()))


# A space name is a partition key on every atom, a cache key for the per-space
# instance, and a personality row of its own — long enough to be expressive,
# capped so a malformed value cannot become any of those.
SPACE_MAX_LEN = 128
SpaceName = Annotated[str, Field(max_length=SPACE_MAX_LEN)]


class RememberRequest(BaseModel):
    content: str
    type: str = "episode"
    probability: float = 0.8
    valence: Optional[float] = None
    source: str = "user"
    space: Optional[SpaceName] = None

    @field_validator("source")
    @classmethod
    def _known_source(cls, v: str) -> str:
        if v not in ("user", "agent"):
            raise ValueError("source must be 'user' or 'agent'")
        return v


class RecallRequest(BaseModel):
    query: str
    top_k: int = 10
    min_confidence: float = 0.1
    space: Optional[SpaceName] = None
    read_spaces: Optional[list[SpaceName]] = Field(default=None, max_length=32)

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
    space: Optional[SpaceName] = None


class ForgetRequest(BaseModel):
    query: str
    reason: Optional[str] = None
    space: Optional[SpaceName] = None

    @field_validator("query")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("query must not be empty or whitespace-only")
        return v


class ReflectRequest(BaseModel):
    space: Optional[SpaceName] = None


class PersonalityRequest(BaseModel):
    action: str  # "get", "set", "preset"
    preset: Optional[str] = None
    params: Optional[dict] = None
    space: Optional[SpaceName] = None


class SpaceQueryRequest(BaseModel):
    op: str  # "overlap", "intersection", "diff"
    other_space: SpaceName
    threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    space: Optional[SpaceName] = None

    @field_validator("op")
    @classmethod
    def _known_op(cls, v: str) -> str:
        if v not in ("overlap", "intersection", "diff"):
            raise ValueError("op must be 'overlap', 'intersection', or 'diff'")
        return v


class SpaceMergeRequest(BaseModel):
    other_space: SpaceName
    threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    min_jaccard: float = Field(default=0.1, ge=0.0, le=1.0)
    space: Optional[SpaceName] = None


@app.post("/remember")
async def remember(req: RememberRequest, request: Request):
    mem = get_mem(req.space)
    result = await _run_sync(handle_tool, mem, "smrti_remember", req.model_dump())
    if cfg.EXTRACT:
        episode_id = result.get("atom_id", "")
        if episode_id:
            from smrti.extraction.extract import extract_and_link_serialized
            auth = request.headers.get("Authorization", "")
            task = asyncio.create_task(
                extract_and_link_serialized(
                    episode_id, req.content, mem, auth, cfg.EXTRACT_MODEL, cfg.EXTRACT_URL,
                    req.source, mode=cfg.EXTRACT_MODE,
                )
            )
            _background_tasks.add(task)
            task.add_done_callback(_background_tasks.discard)
    return result


@app.post("/recall")
async def recall(req: RecallRequest):
    return await _run_sync(handle_tool, get_mem(req.space), "smrti_recall", req.model_dump())


@app.post("/reflect")
async def reflect(req: Optional[ReflectRequest] = None):
    return await _run_sync(handle_tool, get_mem(req.space if req else None), "smrti_reflect", {})


@app.post("/believe")
async def believe(req: BelieveRequest):
    return await _run_sync(handle_tool, get_mem(req.space), "smrti_believe", req.model_dump())


@app.post("/forget")
async def forget(req: ForgetRequest):
    return await _run_sync(handle_tool, get_mem(req.space), "smrti_forget", req.model_dump())


@app.get("/personality")
async def get_personality(space: Optional[SpaceName] = Query(None)):
    return await _run_sync(handle_tool, get_mem(space), "smrti_personality", {"action": "get"})


@app.put("/personality")
async def set_personality(req: PersonalityRequest):
    return await _run_sync(handle_tool, get_mem(req.space), "smrti_personality", req.model_dump())


@app.post("/space_query")
async def space_query(req: SpaceQueryRequest):
    """Compare two spaces: op=overlap (Jaccard), op=intersection, op=diff."""
    mem = _other_space_mem(req.space, req.other_space)
    return await _run_sync(handle_tool, mem, "smrti_space_query", req.model_dump())


@app.post("/space_merge")
async def space_merge(req: SpaceMergeRequest):
    """Materialize a bridge space from the overlap between two spaces."""
    mem = _other_space_mem(req.space, req.other_space)
    return await _run_sync(handle_tool, mem, "smrti_space_merge", req.model_dump())


@app.delete("/spaces/current")
async def clear_current_space(space: Optional[SpaceName] = Query(None)):
    count = await _run_sync(get_mem(space).clear_space)
    return {"status": "ok", "deleted": count}


def run_rest_server(host: str = "0.0.0.0", port: int = 8420) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)

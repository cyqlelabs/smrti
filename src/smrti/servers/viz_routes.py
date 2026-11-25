"""Shared visualization endpoints — mounted by both REST and proxy servers."""
from __future__ import annotations

import asyncio
import json as _json
import os
from typing import Callable, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from smrti import Smrti, __version__
from smrti.core.db import Database, _registry, _registry_lock, _resolve_path, get_database
from smrti.servers import config as cfg
from smrti.servers.common import tenant_for, visible

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# Type alias: callable that returns (or lazily creates) a Smrti instance.
GetMemFn = Callable[[str, str], Smrti]

# Cache of read handles keyed by resolved absolute DB path.
_db_cache: dict[str, "_Reader"] = {}
_DB_CACHE_MAX = 20

# The most atoms one /graph request returns.
_GRAPH_LIMIT_MAX = 5000


async def _off_loop(fn, *args):
    """Run a blocking read in the executor.

    These handlers ran their SQLite queries on the event loop, and one
    /graph of a large space stalled every other request the server held,
    the proxy's chat completions included, for as long as it took.
    """
    return await asyncio.get_running_loop().run_in_executor(None, fn, *args)


class _Reader:
    """The reads the visualizer routes make over one database, and nothing else.

    Reads used to go through a Smrti instance, and constructing one writes its
    space's personality row — with SMRTI_PERSONALITY set it overwrote the
    preset the live server runs, so browsing a graph changed how it decays.
    Status and the space list are taken per tenant and space here instead, so
    a key bound to another tenant sees its own figures and nothing is written.
    """

    def __init__(self, db: Database) -> None:
        self.db = db

    def list_spaces(self, tenant_id: str) -> list[str]:
        rows = self.db.fetchall(
            "SELECT DISTINCT space FROM atoms WHERE tenant_id = ? ORDER BY space",
            (tenant_id,),
        )
        return [r["space"] for r in rows]

    def status(self, tenant_id: str, space: str) -> dict:
        """The figures ``Smrti.status()`` reports, for any tenant and space."""
        total = self.db.fetchone(
            "SELECT COUNT(*) as n FROM atoms WHERE tenant_id=? AND space=?",
            (tenant_id, space),
        )
        by_type = self.db.fetchall(
            "SELECT type, COUNT(*) as n FROM atoms WHERE tenant_id=? AND space=? GROUP BY type",
            (tenant_id, space),
        )
        personality = self.db.fetchone(
            "SELECT * FROM personality WHERE tenant_id=? AND space=?",
            (tenant_id, space),
        )
        return {
            "total_atoms": total["n"] if total else 0,
            "by_type": {row["type"]: row["n"] for row in by_type},
            "personality": dict(personality) if personality else {},
        }

    def graph(
        self, tenant_id: str, space: str, type_list: list[str], limit: int, min_confidence: float
    ) -> dict:
        """The *limit* most salient atoms of a space and the edges among them."""
        ph = ",".join("?" * len(type_list))
        node_sql = f"""SELECT * FROM atoms
                        WHERE tenant_id=? AND space=? AND type IN ({ph})
                          AND confidence >= ?
                        ORDER BY (sti + lti) DESC
                        LIMIT ?"""
        node_params = (tenant_id, space, *type_list, min_confidence, limit)
        rows = self.db.fetchall(node_sql, node_params)
        nodes = [dict(r) for r in rows]
        node_ids = {n["id"] for n in nodes}
        node_labels = {n["id"]: n["label"] for n in nodes}

        edges = []
        if node_ids:
            # Anchor the edge query to the node window instead of reading a
            # capped slice of every relation in the space: that cap silently
            # dropped edges once a space held more relations than it allowed.
            edge_rows = self.db.fetchall(
                f"""WITH sel AS ({node_sql})
                    SELECT * FROM atoms
                    WHERE tenant_id=? AND space=? AND type='relation'
                      AND source_id IN (SELECT id FROM sel)""",
                (*node_params, tenant_id, space),
            )
            for r in edge_rows:
                if r["target_id"] in node_ids:
                    e = dict(r)
                    e["source_label"] = node_labels.get(r["source_id"])
                    e["target_label"] = node_labels.get(r["target_id"])
                    edges.append(e)

        return {"nodes": nodes, "edges": edges}


def _decision_tenants(request: Request) -> Optional[tuple[str, ...]]:
    """The tenants whose decision records a request may read, None for every one.

    A key's own tenants, every tenant for the all-tenants key, and with no key
    configured the server's configured tenant — the ``tenant_ids`` that
    ``audit.records()`` filters by.
    """
    if not (cfg.API_KEY or cfg.API_KEYS):
        return (cfg.TENANT_ID,)
    return getattr(request.state, "tenants", None)


def _db_mem(db_path: str) -> _Reader:
    """Return a cached read handle for *db_path* — registry or allowlist only.

    Accepts a path already present in the shared Database registry (the DB this
    server was started with) or one the operator named in SMRTI_VIZ_DBS. Any
    other path is rejected so the query param cannot open — or create —
    arbitrary filesystem paths as SQLite DBs. An allowlisted path that does not
    exist yet is refused too, so a typo surfaces as a 404 instead of silently
    materializing an empty database. Cache eviction only drops the wrapper —
    the registry owns connections.
    """
    resolved = _resolve_path(db_path)
    with _registry_lock:
        registered = _registry.get(resolved)
    if registered is None:
        if resolved not in {_resolve_path(p) for p in cfg.VIZ_DBS}:
            raise HTTPException(
                status_code=403,
                detail=(
                    f"Cannot open {db_path}: this server only serves the database it "
                    "was started with. Add the path to SMRTI_VIZ_DBS to browse it too."
                ),
            )
        if not os.path.isfile(resolved):
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No database file at {db_path} — allowed by SMRTI_VIZ_DBS, "
                    "but it does not exist."
                ),
            )
    cached = _db_cache.get(resolved)
    if cached is not None and cached.db is not registered:
        _db_cache.pop(resolved)  # path was closed and re-registered — drop stale wrapper
    if resolved not in _db_cache:
        if len(_db_cache) >= _DB_CACHE_MAX:
            _db_cache.pop(next(iter(_db_cache)))
        _db_cache[resolved] = _Reader(get_database(resolved))
    return _db_cache[resolved]


def create_viz_router(get_mem: GetMemFn) -> APIRouter:
    """Return an APIRouter with all visualization-support endpoints.

    ``get_mem(tenant_id, space)`` must return the Smrti instance for that pair.
    """
    router = APIRouter()

    def _configured_mem() -> Smrti:
        """Return the instance for this server's configured tenant/space.

        Read-only endpoints route their queries through this one handle and
        pass the requested tenant/space as SQL parameters instead of asking
        ``get_mem`` for an instance per requested pair: on the proxy every
        distinct pair mints a Smrti and writes a personality row, so browsing
        the graph would create rows for tenants that do not exist. It also
        keeps ``/status`` and ``/metrics`` reporting the tenant the server was
        actually started for rather than a hardcoded "default".
        """
        return get_mem(cfg.TENANT_ID, cfg.SPACE)

    def _reader(db: Optional[str]) -> _Reader:
        return _db_mem(db) if db else _Reader(_configured_mem().db)

    @router.get("/viz", include_in_schema=False)
    async def visualizer():
        path = os.path.join(_STATIC_DIR, "visualizer.html")
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Visualizer not found")
        return FileResponse(path, media_type="text/html")

    @router.get("/tenants")
    async def list_tenants(request: Request, db: str | None = Query(None)):
        rows = await _off_loop(
            _reader(db).db.fetchall, "SELECT DISTINCT tenant_id FROM atoms ORDER BY tenant_id"
        )
        return [r["tenant_id"] for r in rows if visible(request, r["tenant_id"])]

    @router.get("/spaces")
    async def list_spaces(
        request: Request, tenant_id: str | None = Query(None), db: str | None = Query(None)
    ):
        return await _off_loop(_reader(db).list_spaces, tenant_for(request, tenant_id, cfg.TENANT_ID))

    @router.get("/graph")
    async def get_graph(
        request: Request,
        tenant_id: str | None = Query(None),
        space: str | None = Query(None),
        limit: int = Query(200, ge=0),
        min_confidence: float = Query(0.0),
        types: str = Query("concept,belief,episode,goal"),
        db: str | None = Query(None),
    ):
        # Unset tenant and space mean the server's own, as on every other
        # route: the literal "default" this used to fall back to showed an
        # empty graph to a server configured for any other tenant.
        tenant_id = tenant_for(request, tenant_id, cfg.TENANT_ID)
        space = space or cfg.SPACE
        type_list = [t.strip() for t in types.split(",") if t.strip()] or [
            "concept", "belief", "episode", "goal"
        ]
        # 0 is the visualizer's "All": as many as the cap allows, since an
        # unbounded read of a large space held a worker for seconds and
        # shipped the whole graph to a browser that cannot draw it.
        limit = min(limit or _GRAPH_LIMIT_MAX, _GRAPH_LIMIT_MAX)
        reader = _reader(db)
        # Encoded in the executor as well: returned as a dict, thousands of
        # rows were serialized on the loop after the query had left it.
        return await _off_loop(
            lambda: JSONResponse(reader.graph(tenant_id, space, type_list, limit, min_confidence))
        )

    @router.get("/status")
    async def status(request: Request, db: str | None = Query(None)):
        # `spaces` doubles as the capability signal: clients treat its presence
        # as proof the server understands per-request space routing. `space`
        # names the space this server writes to when a request does not say —
        # a client whose own config disagrees would otherwise route writes away
        # from the graph it has been building.
        home = _configured_mem()
        tenant = tenant_for(request, None, home.tenant_id)
        mem = _reader(db)
        result = await _off_loop(mem.status, tenant, home.write_space)
        result["spaces"] = await _off_loop(mem.list_spaces, tenant)
        result["space"] = home.write_space
        result["version"] = __version__
        return result

    @router.get("/atoms/{atom_id}")
    async def get_atom(
        request: Request,
        atom_id: str,
        tenant_id: str | None = Query(None),
        space: str | None = Query(None),
    ):
        tenant = tenant_for(request, tenant_id, cfg.TENANT_ID)
        spc = space or cfg.SPACE
        mem = _configured_mem()
        atom = await _off_loop(mem.atomspace.get_atom, atom_id, tenant, spc)
        if not atom:
            raise HTTPException(status_code=404, detail="Atom not found")
        return atom.model_dump()

    @router.get("/llm-calls")
    async def get_llm_calls(request: Request):
        from smrti.call_log import get_all
        return [e for e in get_all() if visible(request, e.get("tenant_id"))]

    @router.get("/llm-calls/stream")
    async def stream_llm_calls(request: Request):
        from smrti.call_log import subscribe, unsubscribe

        async def event_generator():
            q = subscribe()
            try:
                while True:
                    try:
                        entry = await asyncio.wait_for(q.get(), timeout=15.0)
                        if entry is None:  # the server is shutting down
                            return
                        # A clear carries no tenant and only the all-tenants
                        # key can send one; every other message is a record.
                        if entry.get("__clear__") or visible(request, entry.get("tenant_id")):
                            yield f"data: {_json.dumps(entry)}\n\n"
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
            except asyncio.CancelledError:
                pass
            finally:
                unsubscribe(q)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @router.delete("/llm-calls")
    async def clear_llm_calls():
        from smrti.call_log import clear
        clear()
        return {"status": "ok"}

    # ── Decision audit ─────────────────────────────────────────────────────
    @router.get("/decisions")
    async def get_decisions_log(request: Request):
        """Every semantic decision the engine made or failed to make, newest
        first, with the mode it ran under and whether it was applied — the
        record shadow mode exists to produce."""
        from smrti.decisions import audit
        tenants = _decision_tenants(request)
        return [r for r in audit.get_all() if tenants is None or r.get("tenant_id") in tenants]

    @router.delete("/decisions")
    async def clear_decisions_log():
        from smrti.decisions import audit
        audit.clear()
        return {"status": "ok"}

    # ── Prometheus / OpenMetrics exposition ─────────────────────────────────
    @router.get("/metrics")
    async def prometheus_metrics(request: Request, db: str | None = Query(None)):
        """Export smrti status as Prometheus-format metrics.

        Zero external deps — plain text/plain response matching the Prometheus
        exposition format. Safe to scrape from Grafana / any prom-compatible
        system. Use ?db=<path> to target a non-default on-disk database.
        """
        def _esc(value: str) -> str:
            """Escape a Prometheus label value: backslash, quote, newline."""
            return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

        home = _configured_mem()
        tenant_id = tenant_for(request, None, home.tenant_id)
        s = await _off_loop(_reader(db).status, tenant_id, home.write_space)
        labels = f'tenant="{_esc(tenant_id)}",space="{_esc(home.write_space)}"'

        lines: list[str] = []

        # total atoms
        lines.append("# HELP smrti_atoms_total Total number of atoms in the space.")
        lines.append("# TYPE smrti_atoms_total gauge")
        lines.append(f'smrti_atoms_total{{{labels}}} {int(s.get("total_atoms", 0))}')

        # per-type atom counts
        lines.append("# HELP smrti_atoms_by_type Count of atoms by type.")
        lines.append("# TYPE smrti_atoms_by_type gauge")
        for atom_type, count in (s.get("by_type") or {}).items():
            safe_type = _esc(str(atom_type))
            lines.append(
                f'smrti_atoms_by_type{{{labels},type="{safe_type}"}} {int(count)}'
            )

        # epoch count
        epoch = s.get("personality", {}).get("epoch_count")
        if epoch is not None:
            lines.append("# HELP smrti_epoch_count Number of reflect epochs completed.")
            lines.append("# TYPE smrti_epoch_count counter")
            lines.append(f'smrti_epoch_count{{{labels}}} {int(epoch)}')

        # personality tuning params (useful to see drift vs baseline)
        for key in (
            "confidence_decay_rate",
            "sti_decay_rate",
            "sti_boost_on_access",
            "lti_promotion_threshold",
            "min_confidence_to_surface",
            "valence_weight",
        ):
            val = s.get("personality", {}).get(key)
            if val is None:
                continue
            metric = f"smrti_personality_{key}"
            lines.append(f"# HELP {metric} Personality tuning parameter: {key}.")
            lines.append(f"# TYPE {metric} gauge")
            lines.append(f'{metric}{{{labels}}} {float(val)}')

        # decisions made since the process started, by task, mode and outcome
        from smrti.decisions import audit as _audit
        from smrti.decisions import get_decisions as _get_decisions
        from smrti.decisions.policies import TASKS as _TASKS

        engine = _get_decisions()
        lines.append("# HELP smrti_decisions_mode The mode each decision task runs under (0 off, 1 shadow, 2 active).")
        lines.append("# TYPE smrti_decisions_mode gauge")
        for task in _TASKS:
            level = {"off": 0, "shadow": 1, "active": 2}.get(engine.mode(task), 0)
            lines.append(f'smrti_decisions_mode{{task="{task}"}} {level}')
        # The counters add up every tenant's decisions, so only the
        # all-tenants key reads them.
        counts = _audit.counters() if visible(request, None) else {}
        if counts:
            lines.append("# HELP smrti_decisions_total Decisions by task, mode and outcome.")
            lines.append("# TYPE smrti_decisions_total counter")
            for (task, mode, outcome), n in sorted(counts.items()):
                lines.append(
                    f'smrti_decisions_total{{task="{_esc(task)}",mode="{_esc(mode)}",outcome="{_esc(outcome)}"}} {int(n)}'
                )

        body = "\n".join(lines) + "\n"
        from fastapi.responses import Response
        return Response(content=body, media_type="text/plain; version=0.0.4")

    return router

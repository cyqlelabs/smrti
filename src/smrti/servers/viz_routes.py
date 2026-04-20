"""Shared visualization endpoints — mounted by both REST and proxy servers."""
from __future__ import annotations

import asyncio
import json as _json
import os
from pathlib import Path
from typing import Callable

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse

from smrti import Smrti

_STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# Type alias: callable that returns (or lazily creates) a Smrti instance.
GetMemFn = Callable[[str, str], Smrti]

# Cache of Smrti instances keyed by resolved absolute DB path.
_db_cache: dict[str, Smrti] = {}
_DB_CACHE_MAX = 20


def _db_mem(db_path: str) -> Smrti:
    """Return a cached Smrti instance connected to *db_path* (any tenant/space)."""
    expanded = str(Path(db_path).expanduser().resolve())
    if expanded not in _db_cache:
        if len(_db_cache) >= _DB_CACHE_MAX:
            _db_cache.pop(next(iter(_db_cache)))
        _db_cache[expanded] = Smrti(
            db_path=expanded,
            personality="balanced",
            tenant_id="default",
            write_space="default",
        )
    return _db_cache[expanded]


def create_viz_router(get_mem: GetMemFn) -> APIRouter:
    """Return an APIRouter with all visualization-support endpoints.

    ``get_mem(tenant_id, space)`` must return the Smrti instance for that pair.
    """
    router = APIRouter()

    @router.get("/viz", include_in_schema=False)
    async def visualizer():
        path = os.path.join(_STATIC_DIR, "visualizer.html")
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Visualizer not found")
        return FileResponse(path, media_type="text/html")

    @router.get("/tenants")
    async def list_tenants(db: str | None = Query(None)):
        mem = _db_mem(db) if db else get_mem("default", "default")
        rows = mem.db.fetchall("SELECT DISTINCT tenant_id FROM atoms ORDER BY tenant_id")
        return [r["tenant_id"] for r in rows]

    @router.get("/spaces")
    async def list_spaces(tenant_id: str = Query("default"), db: str | None = Query(None)):
        mem = _db_mem(db) if db else get_mem(tenant_id, "default")
        rows = mem.db.fetchall(
            "SELECT DISTINCT space FROM atoms WHERE tenant_id = ? ORDER BY space",
            (tenant_id,),
        )
        return [r["space"] for r in rows]

    @router.get("/graph")
    async def get_graph(
        tenant_id: str = Query("default"),
        space: str = Query("default"),
        limit: int = Query(200),
        min_confidence: float = Query(0.0),
        types: str = Query("concept,belief,episode,goal"),
        db: str | None = Query(None),
    ):
        type_list = [t.strip() for t in types.split(",") if t.strip()] or [
            "concept", "belief", "episode", "goal"
        ]
        ph = ",".join("?" * len(type_list))
        mem = _db_mem(db) if db else get_mem(tenant_id, space)
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

        edges = []
        if node_ids:
            edge_rows = mem.db.fetchall(
                """SELECT * FROM atoms
                   WHERE tenant_id=? AND space=? AND type='relation'
                     AND source_id IS NOT NULL AND target_id IS NOT NULL
                   LIMIT 5000""",
                (tenant_id, space),
            )
            for r in edge_rows:
                if r["source_id"] in node_ids and r["target_id"] in node_ids:
                    e = dict(r)
                    e["source_label"] = node_labels.get(r["source_id"])
                    e["target_label"] = node_labels.get(r["target_id"])
                    edges.append(e)

        return {"nodes": nodes, "edges": edges}

    @router.get("/status")
    async def status(db: str | None = Query(None)):
        mem = _db_mem(db) if db else get_mem("default", "default")
        return mem.status()

    @router.get("/atoms/{atom_id}")
    async def get_atom(atom_id: str):
        mem = get_mem("default", "default")
        atom = mem.atomspace.get_atom(atom_id, mem.tenant_id, mem.write_space)
        if not atom:
            raise HTTPException(status_code=404, detail="Atom not found")
        return atom.model_dump()

    @router.get("/llm-calls")
    async def get_llm_calls():
        from smrti.call_log import get_all
        return get_all()

    @router.get("/llm-calls/stream")
    async def stream_llm_calls():
        from smrti.call_log import subscribe, unsubscribe

        async def event_generator():
            q = subscribe()
            try:
                while True:
                    try:
                        entry = await asyncio.wait_for(q.get(), timeout=15.0)
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

    # ── Prometheus / OpenMetrics exposition ─────────────────────────────────
    @router.get("/metrics")
    async def prometheus_metrics(db: str | None = Query(None)):
        """Export smrti status as Prometheus-format metrics.

        Zero external deps — plain text/plain response matching the Prometheus
        exposition format. Safe to scrape from Grafana / any prom-compatible
        system. Use ?db=<path> to target a non-default on-disk database.
        """
        mem = _db_mem(db) if db else get_mem("default", "default")
        s = mem.status()
        tenant = str(s.get("personality", {}).get("tenant_id") or "default")
        space = str(s.get("personality", {}).get("space") or "default")
        labels = f'tenant="{tenant}",space="{space}"'

        lines: list[str] = []

        # total atoms
        lines.append("# HELP smrti_atoms_total Total number of atoms in the space.")
        lines.append("# TYPE smrti_atoms_total gauge")
        lines.append(f'smrti_atoms_total{{{labels}}} {int(s.get("total_atoms", 0))}')

        # per-type atom counts
        lines.append("# HELP smrti_atoms_by_type Count of atoms by type.")
        lines.append("# TYPE smrti_atoms_by_type gauge")
        for atom_type, count in (s.get("by_type") or {}).items():
            safe_type = str(atom_type).replace('"', '')
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

        body = "\n".join(lines) + "\n"
        from fastapi.responses import Response
        return Response(content=body, media_type="text/plain; version=0.0.4")

    return router

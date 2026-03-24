"""FastAPI server for smrti-town — WebSocket tick stream, REST endpoints, static frontend."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import pathlib
import re
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState

from smrti_town.config import (
    BUILDING_CATALOG,
    COUNCIL_MEETING_INTERVAL_HOURS,
    COUNCIL_ROLES,
    PHASE_GAMEPLAY,
    PHASE_GAME_OVER,
    PHASE_OPENING_CHOOSE_MAYOR,
    PHASE_OPENING_COUNCIL,
    PHASE_OPENING_PLACE_HALL,
    STARTING_TREASURY,
)
from smrti_town.llm import LLMClient, LLMSettings
from smrti_town.spatial import Place
from smrti_town.worldgen import generate_opening

log = logging.getLogger(__name__)

# ── Static files path ───────────────────────────────────────────────────────
_STATIC_DIR = os.environ.get(
    "SMRTI_TOWN_STATIC",
    str(pathlib.Path(__file__).parent / "static"),
)

# ── Global state ────────────────────────────────────────────────────────────

_connected_clients: set[WebSocket] = set()
_lock = asyncio.Lock()

# Game state — populated during opening sequence, consumed by gameplay loop.
_game: dict[str, Any] = {
    "phase": PHASE_OPENING_PLACE_HALL,
    "candidates": None,
    "mayor": None,
    "council_specs": None,
    "citizen_specs": None,
    "topology": None,
    "gridmap": None,
    "economy": None,
    "world_smrti": None,
    "culture_smrti": None,
    "engine": None,
    "tick_count": 0,
    "calendar": None,
    "council": None,
    "petition_manager": None,
    "population_manager": None,
    "event_manager": None,
    "director": None,
    "chronos": None,
    "dialogue_queue": None,
    "citizens": [],
    "last_meeting_tick": 0,
    "pending_meeting": None,
}

_llm_settings = LLMSettings()
_llm_client = LLMClient(_llm_settings)
_engine_task: asyncio.Task[None] | None = None


# ── Broadcast ───────────────────────────────────────────────────────────────

async def broadcast(message: dict) -> None:
    """Send a JSON message to all connected WebSocket clients."""
    if not _connected_clients:
        return
    payload = json.dumps(message)
    dead: list[WebSocket] = []
    for ws in list(_connected_clients):
        try:
            if ws.client_state == WebSocketState.CONNECTED:
                await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _connected_clients.discard(ws)


# ── Tick loop ───────────────────────────────────────────────────────────────

async def _tick_loop() -> None:
    """Main simulation tick loop.  Runs while phase == gameplay."""
    from smrti_town.calendar import SimCalendar
    from smrti_town.director import Chronos, Director

    # Initialize director and chronos if not yet created
    if _game["director"] is None:
        _game["director"] = Director()
    if _game["chronos"] is None:
        _game["chronos"] = Chronos()
    if _game["calendar"] is None:
        _game["calendar"] = SimCalendar()

    director: Director = _game["director"]
    chronos: Chronos = _game["chronos"]
    calendar: SimCalendar = _game["calendar"]

    while _game["phase"] == PHASE_GAMEPLAY:
        citizens = _game.get("citizens", [])
        topology = _game.get("topology")
        economy = _game.get("economy")
        gridmap = _game.get("gridmap")

        # Compute tick delta
        delta = director.compute_delta(citizens, calendar)
        calendar.advance(delta)
        _game["tick_count"] += 1
        tick = _game["tick_count"]

        alive_citizens = [c for c in citizens if getattr(c, "alive", True)]

        # Phase 1: Tick each citizen's needs
        crime_rate = 0.0
        for c in alive_citizens:
            if hasattr(c, "tick_state"):
                c.tick_state(delta, crime_rate=crime_rate)

        # Phase 2: Perceive + Decide
        actions = []
        for c in alive_citizens:
            ctx = None
            if hasattr(c, "perceive") and topology:
                nearby = [
                    a for a in alive_citizens
                    if a.name != c.name and getattr(a, "location", None) == getattr(c, "location", None)
                ]
                place = topology.places.get(getattr(c, "location", None) or "")
                try:
                    ctx = c.perceive(topology, calendar, nearby, place=place)
                except Exception:
                    log.debug("perceive() failed for %s", c.name, exc_info=True)
            if hasattr(c, "decide") and ctx is not None:
                try:
                    action = c.decide(ctx, topology=topology)
                    actions.append({"citizen": c.name, "action": action})
                except Exception:
                    log.debug("decide() failed for %s", c.name, exc_info=True)

        # Phase 3: Economy tick
        if economy and topology:
            buildings = list(topology.places.values())
            economy.collect_taxes(alive_citizens, buildings, delta)
            economy.pay_maintenance(delta)
            economy.process_commerce(buildings, alive_citizens, delta)

            council_members = [
                c for c in alive_citizens
                if getattr(c, "council_role", None) is not None
            ]
            if council_members:
                economy.pay_salaries(council_members, delta)

        # Phase 4: Milestone check
        milestone_events = chronos.check(alive_citizens, calendar)

        # Phase 5: Check for game-over conditions
        alive_count = len(alive_citizens)
        if alive_count == 0:
            _game["phase"] = PHASE_GAME_OVER
            await broadcast({
                "type": "game_over",
                "reason": "All citizens have perished.",
            })
            break

        if economy and economy.check_bankruptcy():
            # Not instant game over, but warn
            await broadcast({
                "type": "event",
                "event_type": "bankruptcy_warning",
                "description": "The treasury is empty! The town is in financial crisis.",
            })

        # Phase 6: Dialogue queue submissions
        dq = _game.get("dialogue_queue")
        if dq:
            for c in alive_citizens:
                loc = getattr(c, "location", None) or "Town Square"
                needs = getattr(c, "needs", None)
                urgent = None
                if needs and hasattr(needs, "highest_unmet_need"):
                    urgent = needs.highest_unmet_need(getattr(c, "life_stage", "adult"))

                # Get memories for dialogue context
                memories: list[dict] = []
                smrti_inst = getattr(c, "smrti", None)
                if smrti_inst:
                    try:
                        results = smrti_inst.recall(f"at {loc}", top_k=3)
                        memories = [{"content": getattr(r, "content", "")} for r in results]
                    except Exception:
                        pass

                nearby_names = [
                    a.name for a in alive_citizens
                    if a.name != c.name and getattr(a, "location", None) == getattr(c, "location", None)
                ]
                target = nearby_names[0] if nearby_names else None

                from smrti_town.dialogue_queue import DialogueRequest
                dq.submit(DialogueRequest(
                    speaker=c.name,
                    target=target,
                    location=loc,
                    time_of_day=calendar.time_of_day,
                    season=calendar.season,
                    personality=getattr(c, "personality_preset", "balanced"),
                    urgent_need=urgent,
                    memories=memories,
                    fallback=f"{c.name} goes about their business.",
                    tick_number=tick,
                ))

        # Broadcast milestone events
        for evt in milestone_events:
            await broadcast({"type": "event", **evt})

        # Broadcast tick result
        tick_result = {
            "type": "tick",
            "tick": tick,
            "delta_hours": round(delta, 2),
            "director_mode": director.mode,
            "calendar": calendar.to_dict(),
            "citizens": [c.to_dict() for c in citizens if hasattr(c, "to_dict")],
            "economy": economy.to_dict() if economy else None,
            "topology": topology.to_dict() if topology and hasattr(topology, "to_dict") else None,
            "gridmap": gridmap.to_dict() if gridmap and hasattr(gridmap, "to_dict") else None,
            "actions": actions,
            "milestone_events": milestone_events,
        }
        await broadcast(tick_result)

        # Wait for next tick
        interval_s = _llm_settings.tick_interval_ms / 1000.0
        await asyncio.sleep(interval_s)


def _start_engine() -> None:
    """Start the tick loop as a background task."""
    global _engine_task
    if _engine_task is not None and not _engine_task.done():
        _engine_task.cancel()
    _engine_task = asyncio.create_task(_tick_loop())


def _stop_engine() -> None:
    """Stop the tick loop."""
    global _engine_task
    if _engine_task is not None:
        _engine_task.cancel()
        _engine_task = None


# ── Lifespan ────────────────────────────────────────────────────────────────

@asynccontextmanager
async def _lifespan(app: FastAPI):
    log.info("smrti-town server starting")
    yield
    log.info("smrti-town server shutting down")
    _stop_engine()
    dq = _game.get("dialogue_queue")
    if dq:
        await dq.stop()
    await _llm_client.close()


# ── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(title="smrti-town", lifespan=_lifespan)


# ── WebSocket ───────────────────────────────────────────────────────────────

@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    _connected_clients.add(websocket)
    log.info("WebSocket client connected (%d total)", len(_connected_clients))

    # Send current state on connect
    try:
        await websocket.send_json({
            "type": "state",
            "phase": _game["phase"],
            "tick": _game["tick_count"],
            "candidates": _game.get("candidates"),
            "mayor": _game.get("mayor"),
            "council_specs": _game.get("council_specs"),
            "calendar": _game["calendar"].to_dict() if _game.get("calendar") else None,
            "topology": _game["topology"].to_dict() if _game.get("topology") else None,
            "gridmap": _game["gridmap"].to_dict() if _game.get("gridmap") else None,
            "economy": _game["economy"].to_dict() if _game.get("economy") else None,
            "citizens": [c.to_dict() for c in _game.get("citizens", []) if hasattr(c, "to_dict")],
        })
    except Exception:
        pass

    try:
        while True:
            data = await websocket.receive_text()
            # Handle client messages (ping, etc.)
            try:
                msg = json.loads(data)
                if msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
            except (json.JSONDecodeError, TypeError):
                pass
    except WebSocketDisconnect:
        pass
    finally:
        _connected_clients.discard(websocket)
        log.info("WebSocket client disconnected (%d remaining)", len(_connected_clients))


# ── Opening sequence endpoints ──────────────────────────────────────────────

@app.post("/opening/place-hall")
async def opening_place_hall(body: dict) -> JSONResponse:
    """Step 1: Player places the Town Hall on the grid.

    Body: {"grid_x": int, "grid_y": int}
    """
    async with _lock:
        if _game["phase"] != PHASE_OPENING_PLACE_HALL:
            return JSONResponse(
                {"error": f"Wrong phase: {_game['phase']}"},
                status_code=400,
            )

        grid_x = body.get("grid_x", 75)
        grid_y = body.get("grid_y", 50)

        db_path = os.environ.get("SMRTI_TOWN_DB", os.path.expanduser("~/.smrti/town.db"))
        tenant_id = os.environ.get("SMRTI_TOWN_TENANT", "millbrook")

        # Generate candidates
        candidates = await _llm_client.generate_mayor_candidates(
            theme=_llm_settings.world_theme,
        )
        _game["candidates"] = candidates
        _game["_place_hall_grid"] = (grid_x, grid_y)
        _game["_db_path"] = db_path
        _game["_tenant_id"] = tenant_id
        _game["phase"] = PHASE_OPENING_CHOOSE_MAYOR

        await broadcast({
            "type": "phase",
            "phase": PHASE_OPENING_CHOOSE_MAYOR,
            "candidates": candidates,
        })

        return JSONResponse({
            "phase": PHASE_OPENING_CHOOSE_MAYOR,
            "candidates": candidates,
        })


@app.post("/opening/choose-mayor")
async def opening_choose_mayor(body: dict) -> JSONResponse:
    """Step 2: Player chooses a mayor from the candidates.

    Body: {"candidate_index": int}
    """
    async with _lock:
        if _game["phase"] != PHASE_OPENING_CHOOSE_MAYOR:
            return JSONResponse(
                {"error": f"Wrong phase: {_game['phase']}"},
                status_code=400,
            )

        candidate_index = body.get("candidate_index", 0)
        grid_x, grid_y = _game.get("_place_hall_grid", (75, 50))
        db_path = _game.get("_db_path", os.path.expanduser("~/.smrti/town.db"))
        tenant_id = _game.get("_tenant_id", "millbrook")

        result = await generate_opening(
            llm_client=_llm_client,
            db_path=db_path,
            tenant_id=tenant_id,
            grid_x=grid_x,
            grid_y=grid_y,
            candidate_index=candidate_index,
            candidates=_game.get("candidates"),
        )

        _game.update({
            "mayor": result["mayor"],
            "council_specs": result["council_specs"],
            "citizen_specs": result["citizen_specs"],
            "topology": result["topology"],
            "gridmap": result["gridmap"],
            "economy": result["economy"],
            "world_smrti": result["world_smrti"],
            "culture_smrti": result["culture_smrti"],
            "phase": PHASE_OPENING_COUNCIL,
        })

        await broadcast({
            "type": "phase",
            "phase": PHASE_OPENING_COUNCIL,
            "mayor": result["mayor"],
            "council_specs": result["council_specs"],
            "citizen_specs": result["citizen_specs"],
            "topology": result["topology"].to_dict(),
            "gridmap": result["gridmap"].to_dict(),
            "economy": result["economy"].to_dict(),
        })

        return JSONResponse({
            "phase": PHASE_OPENING_COUNCIL,
            "mayor": result["mayor"],
            "council_specs": result["council_specs"],
            "citizen_specs": result["citizen_specs"],
            "topology": result["topology"].to_dict(),
            "gridmap": result["gridmap"].to_dict(),
        })


@app.post("/opening/begin")
async def opening_begin() -> JSONResponse:
    """Step 3: Start the simulation.

    Transitions from opening_council to gameplay phase.
    Creates Citizen objects from citizen_specs and starts the tick loop.
    """
    async with _lock:
        if _game["phase"] != PHASE_OPENING_COUNCIL:
            return JSONResponse(
                {"error": f"Wrong phase: {_game['phase']}"},
                status_code=400,
            )

        citizen_specs = _game.get("citizen_specs", [])
        db_path = _game.get("_db_path", os.path.expanduser("~/.smrti/town.db"))
        tenant_id = _game.get("_tenant_id", "millbrook")
        topology = _game["topology"]

        # Create lightweight citizen stand-ins if the full Citizen class isn't available yet.
        # When agent.py is integrated, replace this with real Citizen instantiation.
        citizens = []
        try:
            from smrti_town.agent import Citizen
            for spec in citizen_specs:
                citizen = Citizen(
                    name=spec["name"],
                    age_years=spec.get("age", 35),
                    personality=spec.get("personality", "balanced"),
                    db_path=db_path,
                    tenant_id=tenant_id,
                )
                if spec.get("council_role"):
                    citizen.council_role = spec["council_role"]
                if spec.get("traits"):
                    citizen.traits = dict(spec["traits"])
                if spec.get("skills") and hasattr(citizen, "skills"):
                    for cat, lvl in spec["skills"].items():
                        if hasattr(citizen.skills, "skills"):
                            citizen.skills.skills[cat] = max(0.0, min(1.0, float(lvl)))
                # Place at Town Hall
                citizen.location = "Town Hall"
                if topology:
                    topology.assign_home(citizen.name, "Town Hall")
                citizens.append(citizen)
        except ImportError:
            log.warning("agent.py not available, creating stub citizens")
            for spec in citizen_specs:
                citizens.append(_StubCitizen(spec, db_path, tenant_id))

        _game["citizens"] = citizens

        # Initialize dialogue queue
        from smrti_town.dialogue_queue import DialogueQueue
        dq = DialogueQueue(
            llm_client=_llm_client,
            broadcast_fn=broadcast,
            queue_size=_llm_settings.dialogue_queue_size,
            batch_size=_llm_settings.dialogue_batch_size,
        )
        dq.start()
        _game["dialogue_queue"] = dq

        # Start simulation
        from smrti_town.calendar import SimCalendar
        _game["calendar"] = SimCalendar()
        _game["phase"] = PHASE_GAMEPLAY
        _start_engine()

        await broadcast({
            "type": "phase",
            "phase": PHASE_GAMEPLAY,
            "citizens": [c.to_dict() for c in citizens if hasattr(c, "to_dict")],
        })

        return JSONResponse({
            "phase": PHASE_GAMEPLAY,
            "citizens": [c.to_dict() for c in citizens if hasattr(c, "to_dict")],
        })


# ── Council endpoints ───────────────────────────────────────────────────────

@app.post("/council/approve")
async def council_approve(body: dict) -> JSONResponse:
    """Approve the pending council meeting proposal."""
    async with _lock:
        meeting = _game.get("pending_meeting")
        if not meeting:
            return JSONResponse({"error": "No pending meeting"}, status_code=400)

        proposal = meeting.get("proposal", {})
        bkey = proposal.get("building_key")
        economy = _game.get("economy")
        gridmap = _game.get("gridmap")
        topology = _game.get("topology")

        if not bkey or bkey not in BUILDING_CATALOG:
            _game["pending_meeting"] = None
            return JSONResponse({"error": f"Unknown building: {bkey}"}, status_code=400)

        if economy and not economy.can_afford_building(bkey):
            _game["pending_meeting"] = None
            return JSONResponse({"error": "Cannot afford building"}, status_code=400)

        _game["pending_meeting"] = None

        result = {
            "approved": True,
            "building_key": bkey,
            "awaiting_placement": True,
        }

        await broadcast({
            "type": "council_result",
            **result,
        })

        return JSONResponse(result)


@app.post("/council/reject")
async def council_reject() -> JSONResponse:
    """Reject the pending council meeting proposal."""
    async with _lock:
        _game["pending_meeting"] = None
        await broadcast({"type": "council_result", "approved": False})
        return JSONResponse({"approved": False})


@app.post("/council/counter")
async def council_counter(body: dict) -> JSONResponse:
    """Counter-propose a different building.

    Body: {"building_key": str}
    """
    async with _lock:
        meeting = _game.get("pending_meeting")
        if not meeting:
            return JSONResponse({"error": "No pending meeting"}, status_code=400)

        bkey = body.get("building_key")
        if not bkey or bkey not in BUILDING_CATALOG:
            return JSONResponse({"error": f"Unknown building: {bkey}"}, status_code=400)

        bdef = BUILDING_CATALOG[bkey]
        meeting["proposal"] = {
            "action_type": "build",
            "building_key": bkey,
            "description": bdef.description,
            "cost": bdef.cost,
        }
        _game["pending_meeting"] = meeting

        await broadcast({
            "type": "council_counter",
            "proposal": meeting["proposal"],
        })

        return JSONResponse({"proposal": meeting["proposal"]})


# ── Gameplay endpoints ──────────────────────────────────────────────────────

@app.get("/state")
async def get_state() -> JSONResponse:
    """Full game state snapshot."""
    return JSONResponse({
        "phase": _game["phase"],
        "tick": _game["tick_count"],
        "candidates": _game.get("candidates"),
        "mayor": _game.get("mayor"),
        "council_specs": _game.get("council_specs"),
        "calendar": _game["calendar"].to_dict() if _game.get("calendar") else None,
        "topology": _game["topology"].to_dict() if _game.get("topology") else None,
        "gridmap": _game["gridmap"].to_dict() if _game.get("gridmap") else None,
        "economy": _game["economy"].to_dict() if _game.get("economy") else None,
        "citizens": [c.to_dict() for c in _game.get("citizens", []) if hasattr(c, "to_dict")],
        "pending_meeting": _game.get("pending_meeting"),
        "director": _game["director"].to_dict() if _game.get("director") else None,
    })


@app.get("/agents")
async def get_agents() -> JSONResponse:
    """List all citizens."""
    citizens = _game.get("citizens", [])
    return JSONResponse([c.to_dict() for c in citizens if hasattr(c, "to_dict")])


@app.get("/agents/{name}/memories")
async def get_agent_memories(name: str) -> JSONResponse:
    """Recall recent memories for a citizen."""
    citizens = _game.get("citizens", [])
    citizen = next((c for c in citizens if getattr(c, "name", "") == name), None)
    if citizen is None:
        return JSONResponse({"error": "Citizen not found"}, status_code=404)

    smrti = getattr(citizen, "smrti", None)
    if smrti is None:
        return JSONResponse([])

    try:
        results = smrti.recall(f"about {name}", top_k=20)
        memories = []
        for r in results:
            memories.append({
                "content": getattr(r, "content", ""),
                "label": getattr(r, "label", ""),
                "type": str(getattr(r, "type", "")),
                "probability": getattr(getattr(r, "truth", None), "probability", 0),
                "confidence": getattr(getattr(r, "truth", None), "confidence", 0),
                "valence": getattr(getattr(r, "valence", None), "valence", 0),
            })
        return JSONResponse(memories)
    except Exception:
        log.debug("Failed to recall memories for %s", name, exc_info=True)
        return JSONResponse([])


@app.get("/economy")
async def get_economy() -> JSONResponse:
    """Current economy state."""
    economy = _game.get("economy")
    if economy is None:
        return JSONResponse({"error": "Economy not initialized"}, status_code=400)
    return JSONResponse(economy.to_dict())


@app.get("/petitions")
async def get_petitions() -> JSONResponse:
    """List active petitions."""
    pm = _game.get("petition_manager")
    if pm is None:
        return JSONResponse([])
    return JSONResponse(pm.to_dict() if hasattr(pm, "to_dict") else [])


@app.post("/petitions/{idx}/approve")
async def approve_petition(idx: int) -> JSONResponse:
    """Approve a petition by index."""
    pm = _game.get("petition_manager")
    if pm is None:
        return JSONResponse({"error": "No petition manager"}, status_code=400)
    petitions = getattr(pm, "petitions", [])
    if idx < 0 or idx >= len(petitions):
        return JSONResponse({"error": "Invalid petition index"}, status_code=400)
    petition = petitions[idx]
    petition.status = "approved"
    await broadcast({"type": "petition_update", "index": idx, "status": "approved"})
    return JSONResponse({"index": idx, "status": "approved"})


@app.post("/petitions/{idx}/dismiss")
async def dismiss_petition(idx: int) -> JSONResponse:
    """Dismiss a petition by index."""
    pm = _game.get("petition_manager")
    if pm is None:
        return JSONResponse({"error": "No petition manager"}, status_code=400)
    petitions = getattr(pm, "petitions", [])
    if idx < 0 or idx >= len(petitions):
        return JSONResponse({"error": "Invalid petition index"}, status_code=400)
    petition = petitions[idx]
    petition.status = "dismissed"
    await broadcast({"type": "petition_update", "index": idx, "status": "dismissed"})
    return JSONResponse({"index": idx, "status": "dismissed"})


@app.post("/place-building")
async def place_building(body: dict) -> JSONResponse:
    """Place a building on the grid.

    Body: {"building_key": str, "grid_x": int, "grid_y": int, "name": str (optional)}
    """
    async with _lock:
        bkey = body.get("building_key")
        gx = body.get("grid_x")
        gy = body.get("grid_y")
        place_name = body.get("name")

        if not bkey or gx is None or gy is None:
            return JSONResponse({"error": "Missing building_key, grid_x, or grid_y"}, status_code=400)

        bdef = BUILDING_CATALOG.get(bkey)
        if not bdef:
            return JSONResponse({"error": f"Unknown building: {bkey}"}, status_code=400)

        gridmap = _game.get("gridmap")
        topology = _game.get("topology")
        economy = _game.get("economy")

        if not gridmap or not topology or not economy:
            return JSONResponse({"error": "Game not initialized"}, status_code=400)

        if not gridmap.can_place(bkey, gx, gy):
            return JSONResponse({"error": "Cannot place building at this location"}, status_code=400)

        if not economy.can_afford_building(bkey):
            return JSONResponse({"error": "Cannot afford building"}, status_code=400)

        # Deduct cost
        economy.deduct_building_cost(bkey)

        # Place on grid
        pname = place_name or f"{bkey.replace('_', ' ').title()} ({gx},{gy})"
        import random
        variant = random.randint(0, max(0, bdef.variants - 1))
        pb = gridmap.place(bkey, gx, gy, place_name=pname, sprite_variant=variant)

        # Add to topology
        place = Place(
            name=pname,
            place_type=bdef.category,
            building_key=bkey,
            grid_x=gx,
            grid_y=gy,
        )
        topology.add_place(place)

        # Connect to nearest existing place
        best_dist = float("inf")
        best_neighbor = None
        for name, p in topology.places.items():
            if name == pname:
                continue
            dx = p.grid_x - gx
            dy = p.grid_y - gy
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_neighbor = name
        if best_neighbor:
            topology.connect(pname, best_neighbor)

        # Register building in economy
        economy.register_building(pname, bdef)

        # Seed world space
        world_smrti = _game.get("world_smrti")
        if world_smrti:
            world_smrti.remember(
                f"{pname} ({bkey}) was built at ({gx}, {gy})",
                type="episode",
                probability=1.0,
                valence=0.2,
                metadata={"building_key": bkey, "place_name": pname},
            )

        result = {
            "placed": True,
            "building": pb.to_dict(),
            "place": place.to_dict(),
        }

        await broadcast({
            "type": "building_placed",
            **result,
        })

        return JSONResponse(result)


@app.get("/settings")
async def get_settings() -> JSONResponse:
    """Return current LLM and simulation settings."""
    return JSONResponse(_llm_settings.to_dict())


@app.post("/settings")
async def update_settings(body: dict) -> JSONResponse:
    """Update LLM and simulation settings.

    Body: partial LLMSettings dict.
    """
    global _llm_settings, _llm_client
    async with _lock:
        for key, value in body.items():
            if hasattr(_llm_settings, key):
                setattr(_llm_settings, key, type(getattr(_llm_settings, key))(value))

        # Recreate client if base_url or model changed
        if "base_url" in body or "model" in body:
            await _llm_client.close()
            _llm_client = LLMClient(_llm_settings)

        return JSONResponse(_llm_settings.to_dict())


@app.post("/regenerate")
async def regenerate() -> JSONResponse:
    """Stop the current game and reset to the opening phase."""
    global _llm_client
    async with _lock:
        _stop_engine()
        dq = _game.get("dialogue_queue")
        if dq:
            await dq.stop()

        # Reset game state
        for key in list(_game.keys()):
            if key == "phase":
                _game[key] = PHASE_OPENING_PLACE_HALL
            elif key == "tick_count":
                _game[key] = 0
            elif key == "last_meeting_tick":
                _game[key] = 0
            elif key == "citizens":
                _game[key] = []
            else:
                _game[key] = None

        await broadcast({"type": "reset"})

        return JSONResponse({"phase": PHASE_OPENING_PLACE_HALL})


@app.get("/culture")
async def get_culture() -> JSONResponse:
    """Return Space_Culture atoms."""
    culture = _game.get("culture_smrti")
    if culture is None:
        return JSONResponse([])

    try:
        results = culture.recall("shared beliefs values culture", top_k=50)
        atoms = []
        for r in results:
            atoms.append({
                "content": getattr(r, "content", ""),
                "label": getattr(r, "label", ""),
                "type": str(getattr(r, "type", "")),
                "probability": getattr(getattr(r, "truth", None), "probability", 0),
                "confidence": getattr(getattr(r, "truth", None), "confidence", 0),
                "valence": getattr(getattr(r, "valence", None), "valence", 0),
            })
        return JSONResponse(atoms)
    except Exception:
        log.debug("Failed to query culture space", exc_info=True)
        return JSONResponse([])


@app.post("/skip")
async def skip_time() -> JSONResponse:
    """Request a 1-week time skip."""
    director = _game.get("director")
    if director is None:
        return JSONResponse({"error": "Director not initialized"}, status_code=400)
    director.force_skip()
    return JSONResponse({"skip": True})


@app.post("/pause")
async def pause() -> JSONResponse:
    """Pause the simulation."""
    _stop_engine()
    await broadcast({"type": "paused"})
    return JSONResponse({"paused": True})


@app.post("/resume")
async def resume() -> JSONResponse:
    """Resume the simulation."""
    if _game["phase"] != PHASE_GAMEPLAY:
        return JSONResponse({"error": "Game not in gameplay phase"}, status_code=400)
    _start_engine()
    await broadcast({"type": "resumed"})
    return JSONResponse({"paused": False})


@app.post("/start")
async def start() -> JSONResponse:
    """Alias for /opening/begin — starts the simulation from council phase."""
    return await opening_begin()


# ── Stub citizen for when agent.py is not yet available ─────────────────────

class _StubCitizen:
    """Minimal citizen stand-in used when ``smrti_town.agent`` is not importable."""

    def __init__(self, spec: dict, db_path: str, tenant_id: str) -> None:
        self.name: str = spec.get("name", "Unknown")
        self.age_years: int = spec.get("age", 35)
        self.personality_preset: str = spec.get("personality", "balanced")
        self.council_role: str | None = spec.get("council_role")
        self.traits: dict = spec.get("traits", {})
        self.location: str | None = "Town Hall"
        self.alive: bool = True
        self.life_stage: str = "adult"
        self.home: str | None = "Town Hall"
        self.workplace: str | None = None
        self.wallet: int = 100
        self._spec = spec

        # Create a smrti instance for the citizen
        try:
            from smrti import Smrti
            self.smrti = Smrti(
                db_path=db_path,
                personality=self.personality_preset,
                tenant_id=tenant_id,
                write_space=f"Agent_Space_{self.name}",
                read_spaces=[
                    f"Agent_Space_{self.name}",
                    "World_Space",
                    "Space_Culture",
                ],
            )
            self.smrti.remember(
                spec.get("bio", f"{self.name} is a citizen."),
                type="episode",
                probability=0.9,
                valence=0.1,
            )
        except Exception:
            self.smrti = None

    def tick_state(self, delta_hours: float, crime_rate: float = 0.0) -> None:
        pass

    def perceive(self, topology: Any, calendar: Any, nearby: list, place: Any = None) -> None:
        pass

    def decide(self, context: dict) -> str:
        return "WAIT"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "age_years": self.age_years,
            "personality_preset": self.personality_preset,
            "council_role": self.council_role,
            "traits": self.traits,
            "location": self.location,
            "alive": self.alive,
            "life_stage": self.life_stage,
            "home": self.home,
            "workplace": self.workplace,
            "wallet": self.wallet,
        }


# ── Static file serving ────────────────────────────────────────────────────

_static_path = pathlib.Path(_STATIC_DIR)
if _static_path.is_dir():
    # Mount static sub-directories first
    for subdir in ("css", "js"):
        sub = _static_path / subdir
        if sub.is_dir():
            app.mount(f"/{subdir}", StaticFiles(directory=str(sub)), name=subdir)

    # Serve index.html at root with cache-busting query params on assets
    _ASSET_RE = re.compile(r'(src|href)="([^"]+\.(js|css))"')

    @app.get("/")
    async def index():
        index_file = _static_path / "index.html"
        if not index_file.exists():
            return JSONResponse({"error": "index.html not found"}, status_code=404)
        html = index_file.read_text()
        v = int(time.time())
        html = _ASSET_RE.sub(rf'\1="\2?v={v}"', html)
        from fastapi.responses import HTMLResponse
        return HTMLResponse(html)

    # Serve other static files at root level (sprites, etc.)
    @app.get("/{filename}")
    async def static_file(filename: str):
        file_path = _static_path / filename
        if file_path.exists() and file_path.is_file():
            return FileResponse(str(file_path))
        return JSONResponse({"error": "Not found"}, status_code=404)


# ── Entry point ─────────────────────────────────────────────────────────────

def serve(host: str = "127.0.0.1", port: int = 8430) -> None:
    """Run the server via uvicorn."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)

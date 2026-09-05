"""Save and restore the town, so a restart resumes the game where it was.

The snapshot lives in the town's own SQLite file beside the memory graphs,
one row per tenant. Memories are not part of it: they are already there.
"""

from __future__ import annotations

import json
from typing import Any

_TABLE = (
    "CREATE TABLE IF NOT EXISTS town_state "
    "(tenant_id TEXT PRIMARY KEY, state TEXT NOT NULL, saved_at TEXT NOT NULL)"
)


def snapshot(game: dict) -> dict:
    """The tick loop's state as JSON, each manager by its own serializer."""
    def opt(key: str):
        obj = game.get(key)
        return obj.to_dict() if obj is not None else None

    return {
        "phase": game["phase"],
        "tick_count": game["tick_count"],
        "calendar": game["calendar"].total_hours,
        "director": opt("director"),
        "chronos": opt("chronos"),
        "topology": opt("topology"),
        "gridmap": opt("gridmap"),
        "economy": opt("economy"),
        "council": opt("council"),
        "petitions": opt("petition_manager"),
        "events": opt("event_manager"),
        "citizens": [c.to_dict() for c in game.get("citizens", []) if hasattr(c, "to_dict")],
        "mayor": game.get("mayor"),
        "council_specs": game.get("council_specs"),
        "citizen_specs": game.get("citizen_specs"),
        "candidates": game.get("candidates"),
        "pending_meeting": game.get("pending_meeting"),
        "last_meeting_tick": game.get("last_meeting_tick", 0),
        "last_immigration_check": game.get("last_immigration_check", 0),
        "last_petition_check": game.get("last_petition_check", 0.0),
        "dialogue_last_tick": game.get("dialogue_last_tick", {}),
    }


def save(db: Any, tenant_id: str, data: dict) -> None:
    db.execute(_TABLE)
    db.execute(
        "INSERT OR REPLACE INTO town_state (tenant_id, state, saved_at) VALUES (?, ?, datetime('now'))",
        (tenant_id, json.dumps(data)),
    )


def load(db: Any, tenant_id: str) -> dict | None:
    db.execute(_TABLE)
    row = db.fetchone("SELECT state FROM town_state WHERE tenant_id = ?", (tenant_id,))
    return json.loads(row["state"]) if row else None


def clear(db: Any, tenant_id: str) -> None:
    db.execute(_TABLE)
    db.execute("DELETE FROM town_state WHERE tenant_id = ?", (tenant_id,))


def restore(game: dict, data: dict, db_path: str, tenant_id: str) -> None:
    """Rebuild the tick loop's state from a snapshot into *game*. Citizens
    reopen their memory graphs by name; homes and workplaces are reassigned
    from what each citizen recorded."""
    from smrti_town.agent import Citizen
    from smrti_town.calendar import SimCalendar
    from smrti_town.council import Council
    from smrti_town.director import Chronos, Director
    from smrti_town.economy import EconomyManager
    from smrti_town.events import EventManager
    from smrti_town.gridmap import GridMap
    from smrti_town.petition import PetitionManager
    from smrti_town.population import PopulationManager
    from smrti_town.spatial import TownTopology

    game["phase"] = data["phase"]
    game["tick_count"] = data["tick_count"]
    game["calendar"] = SimCalendar(total_hours=data["calendar"])
    game["director"] = Director.from_dict(data["director"]) if data.get("director") else Director()
    game["chronos"] = Chronos.from_dict(data["chronos"]) if data.get("chronos") else Chronos()
    topology = TownTopology.from_dict(data["topology"])
    game["topology"] = topology
    game["gridmap"] = GridMap.from_dict(data["gridmap"])
    game["economy"] = EconomyManager.from_dict(data["economy"])
    game["council"] = Council.from_dict(data["council"]) if data.get("council") else None
    game["petition_manager"] = PetitionManager.from_dict(data["petitions"] or {})
    game["event_manager"] = EventManager.from_dict(data["events"] or {})
    game["population_manager"] = PopulationManager()
    citizens = [Citizen.from_dict(d, db_path, tenant_id) for d in data["citizens"]]
    for c in citizens:
        if c.home in topology.places:
            topology.assign_home(c.name, c.home)
        if c.workplace in topology.places:
            topology.assign_workplace(c.name, c.workplace)
    game["citizens"] = citizens
    for key in ("mayor", "council_specs", "citizen_specs", "candidates", "pending_meeting",
                "last_meeting_tick", "last_immigration_check", "last_petition_check", "dialogue_last_tick"):
        game[key] = data.get(key)
    game["last_meeting_tick"] = game["last_meeting_tick"] or 0
    game["last_immigration_check"] = game["last_immigration_check"] or 0
    game["last_petition_check"] = game["last_petition_check"] or 0.0
    game["dialogue_last_tick"] = game["dialogue_last_tick"] or {}

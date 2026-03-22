"""Build a SimEngine from LLM-generated world data, with Millbrook fallback."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from smrti import Smrti

from smrti_town.agent import Agent
from smrti_town.calendar import SimCalendar
from smrti_town.engine import SimEngine
from smrti_town.spatial import Place, TownTopology

if TYPE_CHECKING:
    from smrti_town.llm import LLMClient

logger = logging.getLogger("smrti_town.worldgen")

_VALID_PERSONALITIES = frozenset(
    {"balanced", "analytical", "curious", "empathetic", "maverick", "deterministic"}
)


async def create_engine_from_llm(
    llm_client: LLMClient,
    db_path: str,
    tenant_id: str,
) -> SimEngine:
    """Generate a world via LLM and return a ready SimEngine.

    Falls back to the hardcoded Millbrook scenario if LLM is disabled,
    returns None/invalid JSON, or building the engine fails.
    """
    from smrti_town.scenarios.millbrook import create_millbrook  # local import avoids cycle

    world = await llm_client.generate_world(theme=llm_client.settings.world_theme)
    if not world:
        logger.info("LLM world gen unavailable — using Millbrook fallback")
        return create_millbrook(db_path=db_path, tenant_id=tenant_id, llm_client=llm_client)

    try:
        return _build_engine(world, db_path, tenant_id, llm_client)
    except Exception as exc:
        logger.warning("Engine build failed (%s) — using Millbrook fallback", exc)
        return create_millbrook(db_path=db_path, tenant_id=tenant_id, llm_client=llm_client)


# ── Engine builder ────────────────────────────────────────────────────

def _build_engine(
    world: dict,
    db_path: str,
    tenant_id: str,
    llm_client: LLMClient,
) -> SimEngine:
    topology = _build_topology(world.get("places", []))
    agents = _build_agents(world.get("agents", []), topology, db_path, tenant_id)
    _seed_world_space(world, topology, db_path, tenant_id)
    _seed_culture_space(world.get("cultural_facts", []), db_path, tenant_id)

    return SimEngine(
        agents=agents,
        topology=topology,
        calendar=SimCalendar(total_hours=0.0),
        db_path=db_path,
        tenant_id=tenant_id,
        llm_client=llm_client,
    )


# ── Topology ──────────────────────────────────────────────────────────

_VALID_PLACE_TYPES = frozenset({"street", "public", "outdoor", "home", "other"})


def _valid_place_type(value: object) -> str:
    v = str(value or "").strip().lower()
    return v if v in _VALID_PLACE_TYPES else "other"


def _build_topology(places_data: list[dict]) -> TownTopology:
    topo = TownTopology()

    for p in places_data:
        name = _norm_name(p.get("name", ""))
        if not name:
            continue
        place = Place(
            name=name,
            personality=_valid_personality(p.get("personality")),
            is_outdoor=bool(p.get("is_outdoor", False)),
            has_space=bool(p.get("has_space", True)),
            place_type=_valid_place_type(p.get("type")),
            label=(p.get("label") or name.replace("_", " ")).strip(),
            icon=str(p.get("icon") or "").strip(),
        )
        topo.add_place(place)

    # Wire connections declared in places
    all_names = set(topo.places)
    for p in places_data:
        name = _norm_name(p.get("name", ""))
        if name not in all_names:
            continue
        for conn in p.get("connects_to", []):
            conn = _norm_name(conn)
            if conn in all_names and conn != name:
                topo.connect(name, conn)

    # Guarantee connectivity: isolated nodes connect to the first place
    if topo.places:
        root = next(iter(topo.places))
        for name in list(topo.places):
            if name != root and not topo.neighbors(name):
                topo.connect(name, root)

    _assign_layout(topo, places_data)

    return topo


_TYPE_DEFAULTS: dict[str, tuple[str, str]] = {
    "public":  ("#4A6FA5", "🏛"),
    "outdoor": ("#5C9E5C", "🌳"),
    "home":    ("#D4A03C", "🏠"),
    "street":  ("#B8A88A", ""),
    "other":   ("#888888", ""),
}


def _assign_layout(topo: TownTopology, places_data: list[dict]) -> None:
    """Assign x/y/w/h/color to places for an LLM-generated world.

    Canvas viewport: 920 × 600.
    Layout zones:
      - Streets:  horizontal band at y=285, span width
      - Public:   top row  at y=130
      - Homes:    bottom-left, 2-column grid starting at y=400
      - Outdoors: bottom-right at y=390
    """
    streets  = [p for p in topo.places.values() if p.place_type == "street"]
    publics  = [p for p in topo.places.values() if p.place_type == "public"]
    homes    = [p for p in topo.places.values() if p.place_type == "home"]
    outdoors = [p for p in topo.places.values() if p.place_type == "outdoor"]
    others   = [p for p in topo.places.values() if p.place_type == "other"]

    # Streets — horizontal band
    canvas_w = 920
    if streets:
        street_w = max(80, canvas_w // len(streets) - 10)
        for i, p in enumerate(streets):
            p.x = 20 + i * (street_w + 8)
            p.y = 278
            p.w = street_w
            p.h = 28

    # Public buildings — top row
    if publics:
        pub_w = min(160, max(100, (canvas_w - 40) // len(publics) - 10))
        for i, p in enumerate(publics):
            p.x = 20 + i * (pub_w + 10)
            p.y = 130
            p.w = pub_w
            p.h = 110

    # Homes — 2-column grid, bottom-left
    if homes:
        col_w, col_h = 130, 100
        for i, p in enumerate(homes):
            col = i % 2
            row = i // 2
            p.x = 20 + col * (col_w + 10)
            p.y = 390 + row * (col_h + 10)
            p.w = col_w
            p.h = col_h

    # Outdoors — bottom-right
    if outdoors:
        out_w = min(200, max(120, (canvas_w - 40) // len(outdoors) - 10))
        for i, p in enumerate(outdoors):
            p.x = canvas_w - (i + 1) * (out_w + 10)
            p.y = 390
            p.w = out_w
            p.h = 130

    # Others — hide (virtual/root nodes)
    for p in others:
        p.display = False

    # Apply colors and default icons
    for p in topo.places.values():
        default_color, default_icon = _TYPE_DEFAULTS.get(p.place_type, ("#888888", ""))
        if not p.color or p.color == "#888888":
            p.color = default_color
        if not p.icon:
            p.icon = default_icon


# ── Agents ────────────────────────────────────────────────────────────

def _build_agents(
    agents_data: list[dict],
    topology: TownTopology,
    db_path: str,
    tenant_id: str,
) -> list[Agent]:
    available_places = list(topology.places)
    fallback_place = available_places[0] if available_places else "Town_Square"
    known_agent_names = {_norm_name(a.get("name", "")) for a in agents_data}

    agents: list[Agent] = []

    for a_data in agents_data:
        name = _norm_name(a_data.get("name", ""))
        if not name:
            continue

        personality = _valid_personality(a_data.get("personality"))
        age = float(max(1, min(200, a_data.get("age", 30))))
        start_loc = _norm_name(a_data.get("starting_location", ""))
        if start_loc not in topology.places:
            start_loc = fallback_place

        agent = Agent(
            name=name,
            personality=personality,
            location=start_loc,
            age_years=age,
            db_path=db_path,
            tenant_id=tenant_id,
        )

        # Backstory as an episode memory
        backstory = (a_data.get("backstory") or "").strip()
        if backstory:
            _safe_remember(agent, backstory, "episode", valence=0.2)

        # Initial beliefs
        for belief in a_data.get("initial_beliefs", []):
            content = (belief.get("content") or "").strip()
            if content:
                _safe_remember(
                    agent, content, "belief",
                    probability=_clamp01(belief.get("probability", 0.8)),
                    valence=_clamp_valence(belief.get("valence", 0.0)),
                )

        agents.append(agent)
        if start_loc in topology.places:
            topology.places[start_loc].add_occupant(name)

    # Second pass: seed relationships (all agents now exist)
    agents_by_name = {a.name: a for a in agents}
    for a_data in agents_data:
        name = _norm_name(a_data.get("name", ""))
        agent = agents_by_name.get(name)
        if not agent:
            continue
        for rel in a_data.get("relationships", []):
            target = _norm_name(rel.get("target", ""))
            if target not in known_agent_names or target == name:
                continue
            rel_type = rel.get("type", "friend")
            valence = _clamp_valence(rel.get("valence", 0.4))
            _safe_remember(
                agent,
                f"{name} and {target} are {rel_type}s.",
                "belief",
                probability=0.9,
                valence=valence,
                metadata={"relation": rel_type, "target": target},
            )
            # Pre-seed interaction counts so relationship gates fire sooner
            seed = {"friend": 8, "close_friend": 12, "romantic": 15}.get(rel_type, 5)
            for _ in range(seed):
                agent.increment_interaction(target)

    return agents


# ── Smrti space seeding ───────────────────────────────────────────────

def _seed_world_space(
    world: dict,
    topology: TownTopology,
    db_path: str,
    tenant_id: str,
) -> None:
    town_name = (world.get("town_name") or "Town").strip()
    ws = Smrti(
        db_path=db_path,
        personality="deterministic",
        tenant_id=tenant_id,
        write_space="World_Space",
    )

    # Town description
    desc = (world.get("description") or "").strip()
    if desc:
        ws.remember(content=desc, type="concept", probability=1.0, valence=0.0)

    # Place atoms
    place_desc_map = {
        _norm_name(p.get("name", "")): (p.get("description") or "").strip()
        for p in world.get("places", [])
    }
    for place_name in topology.places:
        ws.remember(
            content=f"{place_name} is a location in {town_name}.",
            type="concept",
            probability=1.0,
            valence=0.0,
            metadata={"entity_type": "location", "place": place_name},
        )
        pdesc = place_desc_map.get(place_name, "")
        if pdesc:
            ws.remember(
                content=pdesc,
                type="belief",
                probability=0.9,
                valence=0.1,
                metadata={"place": place_name},
            )

    ws.close()


def _seed_culture_space(cultural_facts: list[dict], db_path: str, tenant_id: str) -> None:
    if not cultural_facts:
        return
    cs = Smrti(
        db_path=db_path,
        personality="balanced",
        tenant_id=tenant_id,
        write_space="Space_Culture",
    )
    for fact in cultural_facts:
        content = (fact.get("content") or "").strip()
        if content:
            try:
                cs.remember(
                    content=content,
                    type="belief",
                    probability=_clamp01(fact.get("probability", 0.8)),
                    valence=_clamp_valence(fact.get("valence", 0.1)),
                )
            except Exception:
                pass
    cs.close()


# ── Helpers ───────────────────────────────────────────────────────────

def _norm_name(s: object) -> str:
    return str(s or "").strip().replace(" ", "_")


def _valid_personality(value: object) -> str:
    v = str(value or "").strip().lower()
    return v if v in _VALID_PERSONALITIES else "balanced"


def _clamp01(v: object) -> float:
    try:
        return max(0.0, min(1.0, float(v)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _clamp_valence(v: object) -> float:
    try:
        return max(-1.0, min(1.0, float(v)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0


def _safe_remember(
    agent: Agent,
    content: str,
    type_: str,
    *,
    probability: float = 0.8,
    valence: float = 0.0,
    metadata: dict | None = None,
) -> None:
    try:
        agent.smrti.remember(
            content=content,
            type=type_,
            probability=probability,
            valence=valence,
            **({"metadata": metadata} if metadata else {}),
        )
    except Exception:
        pass

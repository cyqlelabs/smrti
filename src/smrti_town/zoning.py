"""Incremental LLM-driven building staffing and city hall bootstrap."""

from __future__ import annotations

import hashlib
import json
import logging
import random
from typing import TYPE_CHECKING

from smrti import Smrti

from smrti_town.agent import Agent

if TYPE_CHECKING:
    from smrti_town.llm import LLMClient

logger = logging.getLogger("smrti_town.zoning")

# ── Constants ────────────────────────────────────────────────────────────

BUILDING_STAFF_COUNT: dict[str, int] = {
    "city_hall": 3,
    "house": 2,
    "farm": 1,
    "market": 1,
    "school": 1,
    "workshop": 1,
    "clinic": 1,
    "tavern": 1,
    "church": 1,
    "library": 1,
}

_MAX_PEOPLE_PER_CALL = 5

_VALID_PERSONALITIES = frozenset(
    {"balanced", "analytical", "curious", "empathetic", "maverick"}
)

_VALID_BODIES = frozenset({"round", "bean", "tall", "wide", "tiny"})
_VALID_HAIR = frozenset(
    {"bun", "short", "long", "curly", "bald", "mohawk", "ponytail", "afro"}
)
_VALID_HATS = frozenset(
    {"chef", "hardhat", "crown", "propeller", "tophat", "none"}
)
_VALID_FACES = frozenset(
    {"glasses", "mustache", "freckles", "monocle", "none"}
)
_VALID_OUTFITS = frozenset(
    {"apron", "suit", "overalls", "dress", "labcoat", "casual"}
)

_VALID_REL_TYPES = frozenset(
    {"spouse", "sibling", "friend", "rival", "parent", "child"}
)

# Deterministic fallback pools — small but varied.
_FALLBACK_NAMES = [
    "Anya", "Bram", "Celeste", "Dorian", "Elara", "Farid",
    "Greta", "Hugo", "Iris", "Jonas", "Kira", "Leandro",
    "Maren", "Niko", "Olwen", "Pavel", "Quinn", "Rosa",
    "Sven", "Talia",
]

_BUILDING_PROFESSIONS: dict[str, list[str]] = {
    "city_hall": ["Mayor", "Clerk", "Secretary"],
    "house": ["Resident", "Homemaker"],
    "farm": ["Farmer"],
    "market": ["Merchant"],
    "school": ["Teacher"],
    "workshop": ["Artisan"],
    "clinic": ["Doctor"],
    "tavern": ["Barkeep"],
    "church": ["Priest"],
    "library": ["Librarian"],
}

_STAFF_SCHEMA_EXAMPLE = {
    "agents": [
        {
            "name": "Rosa",
            "age": 38,
            "profession": "Mayor",
            "personality": "empathetic",
            "backstory": "Rose from clerk to mayor through sheer persistence. Knows every family in town.",
            "visual_dna": {
                "body": "tall",
                "color": "#C49A6C",
                "hair": "bun",
                "hat": "none",
                "face": "glasses",
                "outfit": "suit",
            },
            "relationships": [
                {"target": "Bram", "type": "spouse"},
            ],
            "initial_beliefs": [
                "I believe every voice in this town deserves to be heard.",
                "The budget crisis is my biggest challenge right now.",
            ],
        },
        {
            "name": "Bram",
            "age": 40,
            "profession": "Clerk",
            "personality": "analytical",
            "backstory": "A meticulous record-keeper who married the mayor before she held office.",
            "visual_dna": {
                "body": "bean",
                "color": "#8B6F47",
                "hair": "short",
                "hat": "none",
                "face": "mustache",
                "outfit": "suit",
            },
            "relationships": [
                {"target": "Rosa", "type": "spouse"},
            ],
            "initial_beliefs": [
                "A well-organized ledger is the backbone of good governance.",
                "Rosa relies on me more than she admits.",
            ],
        },
    ],
}

_STAFF_GEN_SYSTEM = """\
You are a character designer for a life simulation game. Generate believable \
staff and optional family members for a newly constructed building in a small town.

OUTPUT RULES — strictly enforced:
- name: single word, capitalised, no spaces (e.g. Rosa, Bram, Yuki)
- age: integer 18-70
- profession: a short job title relevant to the building
- personality: exactly one of: balanced, analytical, curious, empathetic, maverick
- backstory: 1-2 concise sentences — grounded, specific, no clichés
- visual_dna.body: exactly one of: round, bean, tall, wide, tiny
- visual_dna.color: hex color string for skin tone (e.g. "#C49A6C")
- visual_dna.hair: exactly one of: bun, short, long, curly, bald, mohawk, ponytail, afro
- visual_dna.hat: exactly one of: chef, hardhat, crown, propeller, tophat, none
- visual_dna.face: exactly one of: glasses, mustache, freckles, monocle, none
- visual_dna.outfit: exactly one of: apron, suit, overalls, dress, labcoat, casual
- relationships[].target: must be a name from the agents list or from existing_people
- relationships[].type: exactly one of: spouse, sibling, friend, rival, parent, child
- initial_beliefs: exactly 2-3 short belief sentences seeded into their memory
- Maximum 5 agents total
- Respond with ONLY valid JSON. No markdown, no code fences, no commentary.\
"""


# ── Public API ───────────────────────────────────────────────────────────


async def bootstrap_city_hall(
    grid_pos: tuple[int, int],
    llm_client: LLMClient,
    db_path: str,
    tenant_id: str,
) -> tuple[list[Agent], list[dict]]:
    """Generate initial staff for City Hall. Returns (agents, visual_dna_list)."""
    town_context = (
        f"City Hall has just been founded at grid position {grid_pos}. "
        "This is the very first building in a brand-new town. "
        "Generate a Mayor and 2 staff members (e.g. clerk, secretary). "
        "They should feel like the founding core of a small community."
    )

    prompt = _build_staff_prompt(
        building_type="city_hall",
        building_name="City Hall",
        town_context=town_context,
        existing_people=[],
    )

    try:
        raw = await _call_llm(llm_client, prompt)
        parsed = _extract_json(raw)
        if parsed and "agents" in parsed:
            agents, visuals = _parse_staff_response(
                parsed, db_path, tenant_id, location="City_Hall",
            )
            if agents:
                return agents, visuals
    except Exception as exc:
        logger.warning("City Hall LLM staffing failed: %s", exc)

    return _fallback_staff("city_hall", "City Hall", "City_Hall", db_path, tenant_id, existing_names=set())


async def staff_building(
    building_type: str,
    building_name: str,
    location: str,
    llm_client: LLMClient,
    existing_agents: list[Agent],
    db_path: str,
    tenant_id: str,
) -> tuple[list[Agent], list[dict]]:
    """Generate staff for a newly placed building. Returns (agents, visual_dna_list)."""
    existing_people = [
        {"name": a.name, "age": round(a.age_years), "personality": a.personality_preset}
        for a in existing_agents
        if a.alive
    ]

    count = BUILDING_STAFF_COUNT.get(building_type, 1)
    town_context = (
        f"A new {building_name} ({building_type}) has been built at {location}. "
        f"Generate {count} staff member(s) for this building. "
        f"You may also generate up to {_MAX_PEOPLE_PER_CALL - count} family members "
        f"if it makes narrative sense (e.g. a spouse or teenage child). "
        f"Staff should have professions appropriate for a {building_type}. "
        f"You may reference existing townspeople in relationships."
    )

    prompt = _build_staff_prompt(
        building_type=building_type,
        building_name=building_name,
        town_context=town_context,
        existing_people=existing_people,
    )

    try:
        raw = await _call_llm(llm_client, prompt)
        parsed = _extract_json(raw)
        if parsed and "agents" in parsed:
            agents, visuals = _parse_staff_response(
                parsed, db_path, tenant_id, location=location,
                existing_names={a.name for a in existing_agents},
            )
            if agents:
                return agents, visuals
    except Exception as exc:
        logger.warning(
            "LLM staffing failed for %s (%s): %s", building_name, building_type, exc,
        )

    return _fallback_staff(
        building_type, building_name, location, db_path, tenant_id,
        existing_names={a.name for a in existing_agents},
    )


# ── Prompt construction ──────────────────────────────────────────────────


def _build_staff_prompt(
    building_type: str,
    building_name: str,
    town_context: str,
    existing_people: list[dict],
) -> str:
    """Build the LLM prompt for staff generation."""
    people_block = ""
    if existing_people:
        people_block = (
            "\n\nExisting townspeople (you may reference these in relationships):\n"
            + json.dumps(existing_people, ensure_ascii=False)
        )

    return (
        f"Building type: {building_type}\n"
        f"Building name: {building_name}\n\n"
        f"{town_context}"
        f"{people_block}\n\n"
        f"Follow this JSON structure exactly:\n"
        f"{json.dumps(_STAFF_SCHEMA_EXAMPLE, indent=2, ensure_ascii=False)}\n\n"
        f"Now generate NEW and ORIGINAL characters for this {building_name}. "
        f"Maximum {_MAX_PEOPLE_PER_CALL} people total."
    )


# ── LLM call ────────────────────────────────────────────────────────────


async def _call_llm(llm_client: LLMClient, user_prompt: str) -> str:
    """Send a staff generation request to the LLM via the client's _chat method."""
    return await llm_client._chat(
        messages=[
            {"role": "system", "content": _STAFF_GEN_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=2000,
        temperature=0.85,
        timeout=llm_client.settings.worldgen_timeout or None,
    )


# ── Response parsing ────────────────────────────────────────────────────


def _extract_json(raw: str) -> dict | None:
    """Extract a JSON object from raw LLM output, stripping markdown fences."""
    text = raw.strip()
    if "```" in text:
        for part in text.split("```")[1:]:
            part = part.lstrip("json").strip()
            if part.startswith("{"):
                text = part
                break
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        # Try to find the first { ... } block
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except (json.JSONDecodeError, ValueError):
                pass
    return None


def _parse_staff_response(
    raw_json: dict,
    db_path: str,
    tenant_id: str,
    location: str,
    existing_names: set[str] | None = None,
) -> tuple[list[Agent], list[dict]]:
    """Parse and validate LLM response, create Agent objects."""
    existing_names = existing_names or set()
    agents_data = raw_json.get("agents", [])
    if not isinstance(agents_data, list):
        return [], []

    # Enforce depth limit
    agents_data = agents_data[:_MAX_PEOPLE_PER_CALL]

    agents: list[Agent] = []
    visuals: list[dict] = []
    new_names: set[str] = set()

    for entry in agents_data:
        if not isinstance(entry, dict):
            continue

        name = _norm_name(entry.get("name", ""))
        if not name or name in existing_names or name in new_names:
            continue

        personality = _valid_personality(entry.get("personality"))
        age = _clamp_int(entry.get("age", 30), 18, 70)
        visual_dna = _validate_visual_dna(entry.get("visual_dna", {}))

        agent = Agent(
            name=name,
            personality=personality,
            location=location,
            age_years=float(age),
            db_path=db_path,
            tenant_id=tenant_id,
        )

        # Backstory
        backstory = str(entry.get("backstory") or "").strip()
        if backstory:
            _safe_remember(agent, backstory, "episode", valence=0.2)

        # Profession as a belief
        profession = str(entry.get("profession") or "").strip()
        if profession:
            _safe_remember(
                agent,
                f"I work as a {profession} at {location.replace('_', ' ')}.",
                "belief",
                probability=1.0,
                valence=0.1,
            )

        # Initial beliefs
        beliefs = entry.get("initial_beliefs", [])
        if isinstance(beliefs, list):
            for belief in beliefs[:3]:
                content = str(belief or "").strip()
                if content:
                    _safe_remember(agent, content, "belief", probability=0.8, valence=0.0)

        agents.append(agent)
        visuals.append(visual_dna)
        new_names.add(name)

    # Second pass: seed relationships (all new agents exist now)
    all_known = existing_names | new_names
    agents_by_name = {a.name: a for a in agents}

    for entry in agents_data[:_MAX_PEOPLE_PER_CALL]:
        if not isinstance(entry, dict):
            continue
        name = _norm_name(entry.get("name", ""))
        agent = agents_by_name.get(name)
        if not agent:
            continue
        for rel in entry.get("relationships", []):
            if not isinstance(rel, dict):
                continue
            target = _norm_name(rel.get("target", ""))
            rel_type = str(rel.get("type", "friend")).strip().lower()
            if target not in all_known or target == name:
                continue
            if rel_type not in _VALID_REL_TYPES:
                rel_type = "friend"
            valence = 0.5 if rel_type != "rival" else -0.3
            _REL_LABEL = {
                "child": "parent and child",
                "parent": "parent and child",
                "spouse": "married",
            }
            rel_label = _REL_LABEL.get(rel_type, f"{rel_type}s")
            _safe_remember(
                agent,
                f"{name} and {target} are {rel_label}.",
                "belief",
                probability=0.9,
                valence=valence,
                metadata={"relation": rel_type, "target": target},
            )
            # Pre-seed interaction counts
            seed = {"spouse": 15, "sibling": 10, "friend": 8, "parent": 12, "child": 12}.get(
                rel_type, 5,
            )
            for _ in range(seed):
                agent.increment_interaction(target)

    return agents, visuals


# ── Fallback generation ─────────────────────────────────────────────────


def _fallback_staff(
    building_type: str,
    building_name: str,
    location: str,
    db_path: str,
    tenant_id: str,
    existing_names: set[str] | None = None,
) -> tuple[list[Agent], list[dict]]:
    """Generate deterministic fallback staff when LLM fails."""
    count = BUILDING_STAFF_COUNT.get(building_type, 1)
    professions = _BUILDING_PROFESSIONS.get(building_type, ["Worker"])

    # Deterministic seed from building name so regeneration is stable
    seed = int(hashlib.md5(building_name.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)

    # Pick names that don't collide with existing agents
    taken = existing_names or set()
    available = [n for n in _FALLBACK_NAMES if n not in taken]
    rng.shuffle(available)

    agents: list[Agent] = []
    visuals: list[dict] = []

    for i in range(min(count, len(available))):
        name = available[i]
        profession = professions[i % len(professions)]
        personality = rng.choice(list(_VALID_PERSONALITIES))
        age = rng.randint(22, 60)

        visual_dna = {
            "body": rng.choice(list(_VALID_BODIES)),
            "color": "#{:06x}".format(rng.randint(0x8B6040, 0xE8C8A0)),
            "hair": rng.choice(list(_VALID_HAIR)),
            "hat": "none",
            "face": rng.choice(list(_VALID_FACES)),
            "outfit": rng.choice(list(_VALID_OUTFITS)),
        }

        agent = Agent(
            name=name,
            personality=personality,
            location=location,
            age_years=float(age),
            db_path=db_path,
            tenant_id=tenant_id,
        )

        _safe_remember(
            agent,
            f"I work as a {profession} at {building_name}.",
            "belief",
            probability=1.0,
            valence=0.1,
        )
        _safe_remember(
            agent,
            f"I started working at {building_name} on the day it was built.",
            "episode",
            valence=0.2,
        )

        agents.append(agent)
        visuals.append(visual_dna)

    return agents, visuals


# ── Validation helpers ───────────────────────────────────────────────────


def _validate_visual_dna(raw: object) -> dict:
    """Validate and sanitize visual_dna, filling defaults for invalid values."""
    if not isinstance(raw, dict):
        raw = {}
    return {
        "body": _pick_valid(raw.get("body"), _VALID_BODIES, "bean"),
        "color": _valid_hex_color(raw.get("color"), "#C49A6C"),
        "hair": _pick_valid(raw.get("hair"), _VALID_HAIR, "short"),
        "hat": _pick_valid(raw.get("hat"), _VALID_HATS, "none"),
        "face": _pick_valid(raw.get("face"), _VALID_FACES, "none"),
        "outfit": _pick_valid(raw.get("outfit"), _VALID_OUTFITS, "casual"),
    }


def _pick_valid(value: object, valid: frozenset[str], default: str) -> str:
    v = str(value or "").strip().lower()
    return v if v in valid else default


def _valid_hex_color(value: object, default: str) -> str:
    v = str(value or "").strip()
    if len(v) == 7 and v.startswith("#"):
        try:
            int(v[1:], 16)
            return v
        except ValueError:
            pass
    return default


def _norm_name(s: object) -> str:
    return str(s or "").strip().replace(" ", "_")


def _valid_personality(value: object) -> str:
    v = str(value or "").strip().lower()
    return v if v in _VALID_PERSONALITIES else "balanced"


def _clamp_int(v: object, lo: int, hi: int) -> int:
    try:
        return max(lo, min(hi, int(v)))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return (lo + hi) // 2


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

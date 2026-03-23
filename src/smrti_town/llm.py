"""LLM client for smrti-town — world generation, dialogue, council meetings."""

from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass, field
from typing import Any

import httpx

from smrti_town.config import (
    BUILDING_CATALOG,
    COUNCIL_ROLES,
    HOUSING_IMMIGRANT_PROFILES,
    PRESET_TRAITS,
    SKILL_CATEGORIES,
)

log = logging.getLogger(__name__)


# ── Settings ────────────────────────────────────────────────────────────────

@dataclass
class LLMSettings:
    base_url: str = "http://0.0.0.0:8421/v1"
    model: str = "Qwen3.5-9B-Q8_0.gguf"
    temperature: float = 0.8
    top_p: float = 0.95
    max_tokens: int = 1024
    worldgen_max_tokens: int = 4096
    dialogue_timeout: float = 30.0
    worldgen_timeout: float = 120.0
    enabled: bool = True
    world_theme: str = ""
    tick_interval_ms: int = 2000
    dialogue_queue_size: int = 20
    dialogue_batch_size: int = 5

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> LLMSettings:
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


# ── Fallback data ───────────────────────────────────────────────────────────

_FALLBACK_CANDIDATES = [
    {
        "name": "Eleanor Blackwood",
        "bio": "A pragmatic administrator who built her career mediating disputes between farmers and merchants. Believes in steady, measured growth.",
        "personality": "balanced",
        "governing_style": "moderate",
        "traits": {"shyness": 0.2, "proactivity": 0.6, "leadership": 0.8, "laziness": 0.1,
                   "adventurous": 0.3, "nurturing": 0.5, "stubbornness": 0.4, "creativity": 0.5},
    },
    {
        "name": "Silas Thornton",
        "bio": "A former military captain turned civic planner. Favors order, strong infrastructure, and clear laws above all else.",
        "personality": "deterministic",
        "governing_style": "authoritarian",
        "traits": {"shyness": 0.1, "proactivity": 0.7, "leadership": 0.9, "laziness": 0.1,
                   "adventurous": 0.2, "nurturing": 0.2, "stubbornness": 0.8, "creativity": 0.3},
    },
    {
        "name": "Mirabel Osei",
        "bio": "A visionary artist and community organizer. Champions culture, education, and the creative spirit of the people.",
        "personality": "curious",
        "governing_style": "progressive",
        "traits": {"shyness": 0.2, "proactivity": 0.8, "leadership": 0.6, "laziness": 0.3,
                   "adventurous": 0.7, "nurturing": 0.6, "stubbornness": 0.3, "creativity": 0.9},
    },
]

_FALLBACK_FAMILIES = {
    "cottage": [
        {"name": "Thomas Reed", "age": 28, "personality": "balanced",
         "skills": {"craftsmanship": 0.3, "agriculture": 0.2}, "bio": "A quiet carpenter seeking a fresh start."},
        {"name": "Clara Reed", "age": 26, "personality": "empathetic",
         "skills": {"commerce": 0.2, "literacy": 0.3}, "bio": "A cheerful teacher who loves the countryside."},
    ],
    "house": [
        {"name": "Henrik Vasquez", "age": 35, "personality": "analytical",
         "skills": {"commerce": 0.4, "literacy": 0.3}, "bio": "A merchant looking for new trade opportunities."},
        {"name": "Ingrid Vasquez", "age": 33, "personality": "empathetic",
         "skills": {"medicine": 0.3, "teaching": 0.2}, "bio": "A nurse who wants a safe place for her family."},
        {"name": "Luca Vasquez", "age": 8, "personality": "curious",
         "skills": {}, "bio": "A bright child full of questions."},
    ],
    "apartment": [
        {"name": "Yuki Tanaka", "age": 24, "personality": "maverick",
         "skills": {"arts": 0.3, "commerce": 0.2}, "bio": "An aspiring artist drawn by the town's charm."},
    ],
    "manor": [
        {"name": "Lord Ashworth", "age": 52, "personality": "analytical",
         "skills": {"commerce": 0.6, "leadership": 0.4}, "bio": "A wealthy patron with an eye for investment."},
        {"name": "Lady Ashworth", "age": 49, "personality": "empathetic",
         "skills": {"arts": 0.5, "literacy": 0.4}, "bio": "A cultured philanthropist."},
    ],
    "inn": [
        {"name": "Finn Decker", "age": 30, "personality": "maverick",
         "skills": {"craftsmanship": 0.3, "commerce": 0.2}, "bio": "A wandering journeyman testing the waters."},
    ],
}

_DIALOGUE_TEMPLATES = [
    "{speaker} nods at {target}.",
    "\"Good {time_of_day},\" says {speaker}.",
    "{speaker} shares a thought about the town.",
    "{speaker} sighs, gazing at the {season} sky.",
    "\"I wonder what tomorrow brings,\" {speaker} murmurs.",
]


# ── Helpers ─────────────────────────────────────────────────────────────────

def _available_buildings(pop: int, built_keys: set[str]) -> list[dict]:
    """Return catalog entries unlocked for the given population and existing buildings."""
    out = []
    for key, bdef in BUILDING_CATALOG.items():
        if bdef.unlock_population > pop:
            continue
        if bdef.unlock_buildings and not all(b in built_keys for b in bdef.unlock_buildings):
            continue
        out.append({
            "key": key,
            "category": bdef.category,
            "cost": bdef.cost,
            "description": bdef.description,
        })
    return out


def _parse_json(text: str) -> Any:
    """Best-effort JSON extraction from LLM output."""
    # Strip markdown fences
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]  # remove opening fence
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object/array in the text
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        if start == -1:
            continue
        depth = 0
        for i in range(start, len(text)):
            if text[i] == start_char:
                depth += 1
            elif text[i] == end_char:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
    raise ValueError(f"Could not parse JSON from LLM response: {text[:200]}")


def _clamp_traits(traits: dict) -> dict:
    return {k: max(0.0, min(1.0, float(v))) for k, v in traits.items()
            if k in PRESET_TRAITS.get("balanced", {})}


def _validate_personality(p: str) -> str:
    return p if p in PRESET_TRAITS else "balanced"


def _validate_skills(skills: dict) -> dict:
    return {k: max(0.0, min(1.0, float(v))) for k, v in skills.items()
            if k in SKILL_CATEGORIES}


# ── LLM Client ──────────────────────────────────────────────────────────────

class LLMClient:
    """Async LLM client for town simulation enrichment."""

    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings
        self._client: httpx.AsyncClient | None = None

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.settings.base_url,
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=self.settings.worldgen_timeout,
                    write=10.0,
                    pool=10.0,
                ),
            )
        return self._client

    async def _chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout: float | None = None,
    ) -> str:
        """Send a chat completion request and return the assistant content."""
        client = await self._ensure_client()
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.settings.temperature,
            "top_p": self.settings.top_p,
            "max_tokens": max_tokens or self.settings.max_tokens,
        }
        resp = await client.post(
            "/chat/completions",
            json=payload,
            timeout=timeout or self.settings.worldgen_timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices", [])
        if not choices:
            raise ValueError("LLM returned no choices")
        msg = choices[0].get("message", {})
        content = msg.get("content", "")
        # Fallback for thinking models
        if not content:
            content = msg.get("reasoning_content", "")
        return content.strip()

    # ── Mayor candidates ────────────────────────────────────────────────

    async def generate_mayor_candidates(self, theme: str = "") -> list[dict]:
        """Generate 3 mayor candidates with distinct governing styles."""
        if not self.settings.enabled:
            return list(_FALLBACK_CANDIDATES)

        roles_desc = ", ".join(f"{r['title']} ({r['domain']})" for r in COUNCIL_ROLES.values())
        personality_options = ", ".join(PRESET_TRAITS.keys())
        trait_names = ", ".join(PRESET_TRAITS["balanced"].keys())

        theme_line = f"\nTown theme/setting: {theme}" if theme else ""

        prompt = f"""Generate 3 mayor candidates for a new town. Each should have a distinct governing philosophy.{theme_line}

The town council has these roles: {roles_desc}
Available personality presets: {personality_options}
Trait axes (each 0.0-1.0): {trait_names}

Return a JSON array of 3 objects, each with:
- "name": full name (string)
- "bio": 2-3 sentence backstory (string)
- "personality": one of the preset names (string)
- "governing_style": one of "moderate", "authoritarian", "progressive", "libertarian", "traditionalist" (string)
- "traits": object mapping trait names to float values 0.0-1.0

Example:
[{{"name":"Jane Doe","bio":"A former teacher turned administrator.","personality":"balanced","governing_style":"moderate","traits":{{"shyness":0.2,"proactivity":0.6,"leadership":0.8,"laziness":0.1,"adventurous":0.3,"nurturing":0.5,"stubbornness":0.4,"creativity":0.5}}}}]

Return ONLY the JSON array, no other text."""

        try:
            raw = await self._chat(
                [{"role": "user", "content": prompt}],
                max_tokens=self.settings.worldgen_max_tokens,
                timeout=self.settings.worldgen_timeout,
            )
            candidates = _parse_json(raw)
            if not isinstance(candidates, list) or len(candidates) < 3:
                log.warning("LLM returned %d candidates, falling back", len(candidates) if isinstance(candidates, list) else 0)
                return list(_FALLBACK_CANDIDATES)

            result = []
            for c in candidates[:3]:
                result.append({
                    "name": str(c.get("name", f"Candidate {len(result)+1}")),
                    "bio": str(c.get("bio", "")),
                    "personality": _validate_personality(c.get("personality", "balanced")),
                    "governing_style": str(c.get("governing_style", "moderate")),
                    "traits": _clamp_traits(c.get("traits", PRESET_TRAITS["balanced"])),
                })
            return result

        except Exception:
            log.exception("Failed to generate mayor candidates via LLM")
            return list(_FALLBACK_CANDIDATES)

    # ── Council meeting ─────────────────────────────────────────────────

    async def generate_council_meeting(
        self,
        town_state: dict,
        building_catalog: list[dict] | None = None,
    ) -> dict:
        """Generate a council debate and building proposal given the current town state."""
        if not self.settings.enabled:
            return self._fallback_meeting(town_state, building_catalog)

        pop = town_state.get("population", 0)
        built = set(town_state.get("built_keys", []))
        available = building_catalog or _available_buildings(pop, built)

        council_members = town_state.get("council", [])
        council_desc = "\n".join(
            f"- {m.get('name', '?')} ({m.get('role', '?')}): personality={m.get('personality', '?')}"
            for m in council_members
        )
        treasury = town_state.get("treasury", 0)
        needs_summary = town_state.get("needs_summary", "No critical needs.")

        buildings_desc = "\n".join(
            f"- {b['key']}: {b['description']} (cost: {b['cost']})"
            for b in available[:15]
        )

        prompt = f"""You are generating a town council meeting for a city-builder simulation.

Town state:
- Population: {pop}
- Treasury: {treasury} gold
- Urgent needs: {needs_summary}

Council members:
{council_desc}

Available buildings to propose:
{buildings_desc}

Generate a council debate where each member argues from their domain perspective, then a final proposal.

Return JSON with:
- "debate": array of {{"role": "mayor"|"sheriff"|etc, "name": "member name", "argument": "1-2 sentence argument"}}
- "proposal": {{"action_type": "build", "building_key": "key from available list", "description": "why this building", "cost": integer cost from catalog}}

Return ONLY the JSON object."""

        try:
            raw = await self._chat(
                [{"role": "user", "content": prompt}],
                max_tokens=self.settings.max_tokens,
                timeout=self.settings.dialogue_timeout,
            )
            meeting = _parse_json(raw)
            if not isinstance(meeting, dict):
                return self._fallback_meeting(town_state, building_catalog)

            # Validate debate entries
            debate = meeting.get("debate", [])
            if not isinstance(debate, list):
                debate = []
            validated_debate = []
            for entry in debate:
                if isinstance(entry, dict) and "argument" in entry:
                    validated_debate.append({
                        "role": str(entry.get("role", "mayor")),
                        "name": str(entry.get("name", "")),
                        "argument": str(entry["argument"]),
                    })

            # Validate proposal
            proposal = meeting.get("proposal", {})
            if not isinstance(proposal, dict) or "building_key" not in proposal:
                return self._fallback_meeting(town_state, building_catalog)

            bkey = proposal["building_key"]
            bdef = BUILDING_CATALOG.get(bkey)
            if not bdef:
                return self._fallback_meeting(town_state, building_catalog)

            return {
                "debate": validated_debate,
                "proposal": {
                    "action_type": str(proposal.get("action_type", "build")),
                    "building_key": bkey,
                    "description": str(proposal.get("description", bdef.description)),
                    "cost": bdef.cost,
                },
            }

        except Exception:
            log.exception("Failed to generate council meeting via LLM")
            return self._fallback_meeting(town_state, building_catalog)

    def _fallback_meeting(self, town_state: dict, building_catalog: list[dict] | None = None) -> dict:
        """Rule-based fallback: pick the cheapest affordable unlocked building."""
        pop = town_state.get("population", 0)
        built = set(town_state.get("built_keys", []))
        treasury = town_state.get("treasury", 0)
        available = building_catalog or _available_buildings(pop, built)

        # Priority: housing if low, then civic, then commercial
        affordable = [b for b in available if b["cost"] <= treasury]
        if not affordable:
            affordable = available[:1] if available else [{"key": "cottage", "cost": 2000, "description": "Basic housing"}]

        # Prefer housing if population is growing
        housing = [b for b in affordable if BUILDING_CATALOG.get(b["key"], None) and
                   BUILDING_CATALOG[b["key"]].provides_housing]
        pick = housing[0] if housing else affordable[0]
        bdef = BUILDING_CATALOG.get(pick["key"])

        council_members = town_state.get("council", [])
        debate = []
        for m in council_members:
            role = m.get("role", "mayor")
            name = m.get("name", "Unknown")
            domain = COUNCIL_ROLES.get(role, {}).get("domain", "governance")
            debate.append({
                "role": role,
                "name": name,
                "argument": f"From a {domain} perspective, {pick['key'].replace('_', ' ')} would serve the town well.",
            })

        return {
            "debate": debate,
            "proposal": {
                "action_type": "build",
                "building_key": pick["key"],
                "description": bdef.description if bdef else pick.get("description", ""),
                "cost": bdef.cost if bdef else pick.get("cost", 0),
            },
        }

    # ── Immigrants ──────────────────────────────────────────────────────

    async def generate_immigrants(
        self,
        housing_type: str,
        town_context: str = "",
    ) -> list[dict]:
        """Generate a family/group for a given housing type."""
        if not self.settings.enabled:
            return self._fallback_immigrants(housing_type)

        profile = HOUSING_IMMIGRANT_PROFILES.get(housing_type, HOUSING_IMMIGRANT_PROFILES["cottage"])
        personality_options = ", ".join(PRESET_TRAITS.keys())
        skill_names = ", ".join(SKILL_CATEGORIES.keys())

        context_line = f"\nTown context: {town_context}" if town_context else ""

        prompt = f"""Generate immigrants for a city-builder simulation.{context_line}

Housing type: {housing_type}
Expected profile: {profile['description']}
Adults expected: {profile['adults']}, Children expected: {profile.get('children', 0)}

Available personality presets: {personality_options}
Available skills (each 0.0-1.0): {skill_names}

Return a JSON array of people, each with:
- "name": full name
- "age": integer (adults 18-65, children 5-17)
- "personality": one of the preset names
- "skills": object mapping skill names to float levels
- "bio": 1 sentence backstory

Return ONLY the JSON array."""

        try:
            raw = await self._chat(
                [{"role": "user", "content": prompt}],
                max_tokens=self.settings.max_tokens,
                timeout=self.settings.dialogue_timeout,
            )
            people = _parse_json(raw)
            if not isinstance(people, list) or not people:
                return self._fallback_immigrants(housing_type)

            result = []
            for p in people:
                age = int(p.get("age", 25))
                age = max(0, min(100, age))
                result.append({
                    "name": str(p.get("name", f"Immigrant {len(result)+1}")),
                    "age": age,
                    "personality": _validate_personality(p.get("personality", "balanced")),
                    "skills": _validate_skills(p.get("skills", {})),
                    "bio": str(p.get("bio", "")),
                })
            return result

        except Exception:
            log.exception("Failed to generate immigrants via LLM")
            return self._fallback_immigrants(housing_type)

    def _fallback_immigrants(self, housing_type: str) -> list[dict]:
        template = _FALLBACK_FAMILIES.get(housing_type, _FALLBACK_FAMILIES["cottage"])
        return [dict(p) for p in template]

    # ── Dialogue ────────────────────────────────────────────────────────

    async def generate_dialogue(
        self,
        speaker: str,
        target: str | None,
        location: str,
        time_of_day: str,
        season: str,
        personality: str,
        urgent_need: str | None,
        memories: list[dict],
        fallback: str,
    ) -> str:
        """Generate a single in-character dialogue line."""
        if not self.settings.enabled:
            return self._format_fallback(speaker, target, time_of_day, season, fallback)

        memory_lines = ""
        if memories:
            memory_items = [f"- {m.get('content', m.get('label', ''))}" for m in memories[:5]]
            memory_lines = "\nRecent memories:\n" + "\n".join(memory_items)

        target_line = f" speaking to {target}" if target else ""
        need_line = f"\nUrgent need: {urgent_need}" if urgent_need else ""

        prompt = f"""Generate ONE short dialogue line (max 20 words) for a character in a town simulation.

Character: {speaker}{target_line}
Location: {location}
Time: {time_of_day}, Season: {season}
Personality: {personality}{need_line}{memory_lines}

Return ONLY the dialogue line in quotes, nothing else. Keep it natural and in-character."""

        try:
            raw = await self._chat(
                [{"role": "user", "content": prompt}],
                max_tokens=128,
                temperature=self.settings.temperature,
                timeout=self.settings.dialogue_timeout,
            )
            # Strip surrounding quotes if present
            line = raw.strip().strip('"').strip("'")
            if not line:
                return self._format_fallback(speaker, target, time_of_day, season, fallback)
            return line

        except Exception:
            log.debug("Dialogue LLM call failed for %s", speaker, exc_info=True)
            return self._format_fallback(speaker, target, time_of_day, season, fallback)

    async def generate_dialogue_batch(self, requests: list[dict]) -> list[str]:
        """Generate dialogue for multiple requests. Falls back per-request on failure."""
        results = []
        for req in requests:
            line = await self.generate_dialogue(
                speaker=req["speaker"],
                target=req.get("target"),
                location=req["location"],
                time_of_day=req["time_of_day"],
                season=req["season"],
                personality=req["personality"],
                urgent_need=req.get("urgent_need"),
                memories=req.get("memories", []),
                fallback=req.get("fallback", "..."),
            )
            results.append(line)
        return results

    def _format_fallback(
        self,
        speaker: str,
        target: str | None,
        time_of_day: str,
        season: str,
        fallback: str,
    ) -> str:
        if fallback and fallback != "...":
            return fallback
        template = random.choice(_DIALOGUE_TEMPLATES)
        return template.format(
            speaker=speaker,
            target=target or "someone",
            time_of_day=time_of_day,
            season=season,
        )

    # ── lifecycle ───────────────────────────────────────────────────────

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

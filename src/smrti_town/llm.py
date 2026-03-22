"""Async OpenAI-compatible LLM client for world generation and dialogue."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging

import httpx

logger = logging.getLogger("smrti_town.llm")

# Sentinel used to distinguish "caller passed None" from "use default"
_SENTINEL = object()

DEFAULT_BASE_URL = "http://0.0.0.0:8421/v1"
DEFAULT_MODEL = "Qwen3.5-9B-Q8_0.gguf"

# Schema example injected into the world-gen few-shot prompt.
_WORLD_SCHEMA_EXAMPLE = {
    "town_name": "Millbrook",
    "description": "A quiet riverside town where everyone knows everyone.",
    "places": [
        {
            "name": "Town_Square",
            "label": "Town Square",
            "description": "The beating heart of civic life, ringed by old oaks.",
            "type": "outdoor",
            "icon": "🌳",
            "personality": "empathetic",
            "is_outdoor": True,
            "has_space": True,
            "connects_to": ["Main_Street", "Bakery"],
        },
        {
            "name": "Main_Street",
            "label": "Main Street",
            "description": "The central artery connecting all public buildings.",
            "type": "street",
            "icon": "",
            "personality": "balanced",
            "is_outdoor": True,
            "has_space": False,
            "connects_to": ["Town_Square", "Bakery", "Library"],
        },
        {
            "name": "Bakery",
            "label": "Elena's Bakery",
            "description": "Smells of fresh bread from 5am. A gossip hub.",
            "type": "public",
            "icon": "🥖",
            "personality": "empathetic",
            "is_outdoor": False,
            "has_space": True,
            "connects_to": ["Main_Street", "Town_Square"],
        },
        {
            "name": "Elena_Home",
            "label": "Elena's Home",
            "description": "A tidy cottage on the edge of town.",
            "type": "home",
            "icon": "🏠",
            "personality": "balanced",
            "is_outdoor": False,
            "has_space": True,
            "connects_to": ["Main_Street"],
        },
    ],
    "agents": [
        {
            "name": "Elena",
            "age": 42,
            "personality": "empathetic",
            "starting_location": "Bakery",
            "backstory": "Runs the bakery her mother left her. Knows every family secret in town.",
            "initial_beliefs": [
                {
                    "content": "I bake every morning — it is my meditation.",
                    "probability": 1.0,
                    "valence": 0.5,
                },
                {
                    "content": "Marco and I have been close friends for twenty years.",
                    "probability": 1.0,
                    "valence": 0.6,
                },
            ],
            "relationships": [
                {"target": "Marco", "type": "close_friend", "valence": 0.6}
            ],
        },
        {
            "name": "Marco",
            "age": 45,
            "personality": "analytical",
            "starting_location": "Town_Square",
            "backstory": "Retired engineer who now restores old clocks. Quiet but razor-sharp.",
            "initial_beliefs": [
                {
                    "content": "I find peace in the precision of mechanical things.",
                    "probability": 0.9,
                    "valence": 0.4,
                }
            ],
            "relationships": [
                {"target": "Elena", "type": "close_friend", "valence": 0.6}
            ],
        },
    ],
    "cultural_facts": [
        {
            "content": "The Town Square hosts a farmers market every Saturday morning.",
            "probability": 0.95,
            "valence": 0.3,
        },
        {
            "content": "Elena's Bakery is famous for its sourdough and its gossip.",
            "probability": 0.9,
            "valence": 0.4,
        },
    ],
}

_WORLD_GEN_SYSTEM = """\
You are a narrative world designer for a life simulation game. Generate believable \
small-town scenarios with grounded, specific characters and real social dynamics.

OUTPUT RULES — strictly enforced:
- Place names: CamelCase_with_underscores, no spaces (e.g. Town_Square, Old_Mill)
- Agent names: single word, capitalised, no spaces (e.g. Elena, Marco, Yuki)
- personality: exactly one of: balanced, analytical, curious, empathetic, maverick, deterministic
- starting_location: must match a name in your places list
- relationship.target: must match an agent name in your agents list
- has_space: true for homes and socially significant buildings; false for streets and paths
- type: exactly one of: home, public, outdoor, street
- icon: single emoji for the place (e.g. ☕ for cafe, 📚 for library, 🌳 for park, 🏠 for home); use empty string for streets
- All place names in connects_to must exist in your places list
- Respond with ONLY valid JSON. No markdown, no code fences, no commentary.\
"""


@dataclasses.dataclass
class LLMSettings:
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    temperature: float = 0.8        # dialogue generation
    top_p: float = 0.9
    max_tokens: int = 80            # dialogue generation
    worldgen_max_tokens: int = 3000
    # Separate timeouts: dialogue is fire-and-forget so it can be generous;
    # worldgen blocks server startup so 0 = wait indefinitely (llama.cpp may need minutes).
    dialogue_timeout: float = 60.0      # seconds; 0 = no timeout
    worldgen_timeout: float = 300.0     # seconds; 0 = no timeout
    enabled: bool = True
    world_theme: str = ""           # e.g. "coastal fishing village, 1950s"
    tick_interval_ms: int = 2000     # wall-clock ms to sleep between ticks
    dialogue_queue_size: int = 8     # max pending requests before dropping
    dialogue_batch_size: int = 3     # max requests merged into one LLM call

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "LLMSettings":
        valid = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in valid})


class LLMClient:
    """Reusable async httpx client wrapping an OpenAI-compatible endpoint."""

    def __init__(self, settings: LLMSettings | None = None) -> None:
        self.settings = settings or LLMSettings()
        self._client: httpx.AsyncClient | None = None

    def update_settings(self, settings: LLMSettings) -> None:
        old_url = self.settings.base_url
        self.settings = settings
        if settings.base_url != old_url and self._client and not self._client.is_closed:
            asyncio.ensure_future(self._client.aclose())
            self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            # Read timeout must cover the slowest expected operation (world gen).
            # Use 0 (None) for truly unbounded if worldgen_timeout is 0.
            wt = self.settings.worldgen_timeout
            dt = self.settings.dialogue_timeout
            max_timeout = max(wt or 0, dt or 0)
            read_timeout = (max_timeout + 10.0) if max_timeout > 0 else None
            self._client = httpx.AsyncClient(
                base_url=self.settings.base_url,
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=read_timeout,
                    write=30.0,
                    pool=5.0,
                ),
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    # ── Low-level chat call ───────────────────────────────────────────

    async def _chat(
        self,
        messages: list[dict],
        max_tokens: int,
        temperature: float | None = None,
        timeout: float | None = _SENTINEL,  # type: ignore[assignment]
    ) -> str:
        """POST to /chat/completions, return content string.

        timeout: seconds for asyncio.wait_for (None = no timeout, _SENTINEL = use caller default).
        """
        client = self._get_client()
        payload = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.settings.temperature,
            "top_p": self.settings.top_p,
            "max_tokens": max_tokens,
            "stream": False,
        }
        coro = client.post("/chat/completions", json=payload)
        # Apply asyncio-level timeout only when explicitly non-zero
        effective = timeout if timeout is not _SENTINEL else None  # type: ignore[comparison-overlap]
        if effective and effective > 0:
            resp = await asyncio.wait_for(coro, timeout=effective)
        else:
            resp = await coro
        resp.raise_for_status()
        data = resp.json()
        content = data["choices"][0]["message"].get("content", "") or ""
        # Some thinking models emit content in reasoning_content when content is empty
        if not content:
            content = data["choices"][0]["message"].get("reasoning_content", "") or ""
        return content.strip()

    # ── Dialogue generation ───────────────────────────────────────────

    async def generate_dialogue(
        self,
        speaker: str,
        target: str,
        location: str,
        time_of_day: str,
        season: str,
        personality: str,
        urgent_drive: str | None,
        memories: list[dict],
        fallback: str,
    ) -> str:
        """Generate one line of in-character dialogue. Returns fallback on error."""
        if not self.settings.enabled:
            return fallback

        mem_lines = ""
        relevant = [m for m in memories[:3] if m.get("content")]
        if relevant:
            mem_lines = "\nYour recent memories:\n" + "\n".join(
                f"- {m['content']}" for m in relevant
            )

        drive_note = (
            f" You feel a strong {urgent_drive} urge right now." if urgent_drive else ""
        )

        prompt = (
            f"You are {speaker}, a {personality} person living in a small town. "
            f"You are speaking to {target} at {location.replace('_', ' ')}. "
            f"It is {time_of_day}, {season}.{drive_note}"
            f"{mem_lines}\n\n"
            f"Write exactly one natural sentence that {speaker} says to {target}. "
            f"Be specific, personal, and true to your character — avoid clichés and generic greetings. "
            f"Reply with ONLY the sentence, no quotes, no attribution."
        )

        try:
            text = await self._chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.settings.max_tokens,
                timeout=self.settings.dialogue_timeout or None,
            )
            # Strip surrounding quotes some models add despite instructions
            if len(text) >= 2 and text[0] in ('"', "'") and text[-1] == text[0]:
                text = text[1:-1]
            return text or fallback
        except Exception as exc:
            logger.debug("Dialogue generation failed (%s → %s): %s", speaker, target, exc)
            return fallback

    # ── Batched dialogue generation ───────────────────────────────────

    async def generate_dialogue_batch(self, requests: list) -> list[str]:
        """Generate one dialogue line per request in a single LLM call.

        Returns a list of strings parallel to ``requests``.  Falls back to each
        request's ``fallback`` string on any error or length mismatch so the
        caller never has to special-case failures.
        """
        fallbacks = [r.fallback for r in requests]
        if not self.settings.enabled or not requests:
            return fallbacks

        chars = []
        for i, r in enumerate(requests):
            mem_lines = [m["content"] for m in r.memories[:3] if m.get("content")]
            entry: dict = {
                "index": i,
                "speaker": r.speaker,
                "target": r.target,
                "location": r.location.replace("_", " "),
                "time_of_day": r.time_of_day,
                "season": r.season,
                "personality": r.personality,
            }
            if r.urgent_drive:
                entry["urgent_drive"] = r.urgent_drive
            if mem_lines:
                entry["memories"] = mem_lines
            chars.append(entry)

        import json as _json
        prompt = (
            "You are a narrator for a life simulation. "
            "Generate exactly one in-character dialogue line for each character below.\n"
            "Return ONLY a JSON array of strings, one per character, in index order.\n"
            "No markdown, no code fences, no commentary — only the JSON array.\n\n"
            "Rules for each line:\n"
            "- One natural sentence only — no continuation, no monologue.\n"
            "- Specific, personal, in-character — no clichés, no generic greetings.\n"
            "- No speaker attribution, no surrounding quotes.\n\n"
            f"Characters:\n{_json.dumps(chars, ensure_ascii=False)}\n\n"
            f'Respond with ONLY the JSON array, e.g.: {_json.dumps(["line for index 0"] * len(requests))}'
        )

        n = len(requests)
        max_tokens = self.settings.max_tokens * n + 100
        timeout = (self.settings.dialogue_timeout * n) if self.settings.dialogue_timeout else None

        try:
            raw = await self._chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                timeout=timeout,
            )
            # Strip markdown fences if model ignores instruction
            if "```" in raw:
                for part in raw.split("```")[1:]:
                    part = part.lstrip("json").strip()
                    if part.startswith("["):
                        raw = part
                        break
            parsed = _json.loads(raw)
            if not isinstance(parsed, list):
                return fallbacks
            # Merge parsed entries with fallbacks for any missing indices
            result = list(fallbacks)
            for idx, text in enumerate(parsed):
                if idx < n and isinstance(text, str) and text.strip():
                    t = text.strip()
                    if len(t) >= 2 and t[0] in ('"', "'") and t[-1] == t[0]:
                        t = t[1:-1]
                    result[idx] = t or fallbacks[idx]
            return result
        except Exception as exc:
            logger.debug("Batch dialogue generation failed: %s", exc)
            return fallbacks

    # ── World generation ──────────────────────────────────────────────

    async def generate_world(self, theme: str = "") -> dict | None:
        """Generate a full town scenario as a validated dict. Returns None on failure."""
        if not self.settings.enabled:
            return None

        theme_clause = f" Setting/theme: {theme}." if theme else ""

        user_msg = (
            f"Generate a complete small-town life simulation scenario.{theme_clause}\n\n"
            f"Requirements:\n"
            f"- 5-9 places: at least 1 home per couple/group, 2-3 public buildings, "
            f"1 outdoor space, 1-2 streets connecting them\n"
            f"- 4-6 agents: distinct personalities, ages 18-70, varied occupations\n"
            f"- At least one established romantic pair and one close friendship\n"
            f"- 3-6 cultural_facts that capture the town's character\n"
            f"- Characters must feel grounded: concrete jobs, specific interests, real tensions\n\n"
            f"Follow this JSON structure exactly:\n"
            f"{json.dumps(_WORLD_SCHEMA_EXAMPLE, indent=2)}\n\n"
            f"Now generate a completely NEW and ORIGINAL scenario — do not copy the example."
        )

        try:
            raw = await self._chat(
                messages=[
                    {"role": "system", "content": _WORLD_GEN_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=self.settings.worldgen_max_tokens,
                temperature=0.85,
                timeout=self.settings.worldgen_timeout or None,
            )
            # Strip markdown fences if model ignores instruction
            if "```" in raw:
                parts = raw.split("```")
                # Take first non-empty fenced block
                for part in parts[1:]:
                    part = part.lstrip("json").strip()
                    if part.startswith("{"):
                        raw = part
                        break
            world = json.loads(raw)
            logger.info(
                "World generated: %s (%d places, %d agents)",
                world.get("town_name", "?"),
                len(world.get("places", [])),
                len(world.get("agents", [])),
            )
            return world
        except Exception as exc:
            logger.warning("World generation failed: %s", exc)
            return None

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Start the town server (port 8430)
smrti serve town
# Or directly:
python -m uvicorn smrti_town.server:app --port 8430

# Env vars
SMRTI_TOWN_DB=~/.smrti/town.db      # SQLite DB path (default)
SMRTI_TOWN_TENANT=millbrook          # tenant_id (default)
SMRTI_TOWN_STATIC=<path>             # override static frontend dir
```

The package is not in `pyproject.toml`; import it by running from the repo root with `pip install -e .` and keeping `src/` on `PYTHONPATH`.

## Architecture

### Startup flow

`server.py` creates a `FastAPI` app. On the first WebSocket connection, `_ensure_engine()` calls `worldgen.create_engine_from_llm()`, which calls `llm.LLMClient.generate_world()`. If the LLM is disabled or returns invalid JSON, `scenarios/millbrook.py::create_millbrook()` provides the hardcoded fallback. The result is a `SimEngine` that owns the tick loop. Additional REST endpoints beyond the basics: `GET /culture` returns Space_Culture atoms (town beliefs); `POST /events/inject` injects one of 9 player-triggered event types into the running simulation.

### The 8-phase tick loop (`engine.py`)

Each call to `SimEngine.tick()` advances simulated time by a variable delta chosen by `Director`:

| Phase | What happens |
|-------|-------------|
| 0 | Director computes `delta_hours`; Chronos fires milestones/birthdays |
| 1 | Drive accumulation; death checks |
| 2 | Perception — each agent queries their Smrti instance for relevant memories |
| 3 | Decision — pure rule-based, **no LLM** |
| 3.5 | LLM dialogue enrichment — queued via `DialogueQueue` (bounded async, no unbounded task accumulation) |
| 4 | Engine resolution — apply action effects (move, eat, sleep, work, …) |
| 5 | Narrative `remember()` — write episode atoms to agent Smrti spaces |
| 6 | Conversation propagation — write dialogue to place space + listener space |
| Sporadic | Random events (weather, illness, gossip, …) |
| 7 | Epoch (every 24 sim-hours) — reflect + bridge discovery every 10th epoch |

The tick loop runs as an `asyncio.Task`; the loop sleeps 10ms between ticks to yield to WebSocket/HTTP handlers.

### Adaptive tick pacing (`director.py`)

`Director.compute_tick_delta()` returns:
- `TICK_SCENE` (0.25h) — ≥2 agents sharing a place
- `TICK_MONTAGE` (8h) — all agents sleeping or non-conversational
- `TICK_SKIP` (168h) — manual skip-week request
- `TICK_ROUTINE` (2h) — default

### Agent decision system (`agent.py`)

`Agent.decide()` is **pure rule-based** — no LLM calls. Priority: sleep → schedule obligation → highest urgent drive → personality-biased idle. Social targets and locations are weighted by recalled memory valence — an agent who remembers a place or person negatively will avoid them. Template dialogue is generated synchronously; LLM enrichment patches it later via `dialogue_patch` WebSocket messages.

`Agent.perceive()` queries Smrti (`top_k=5`) with a location/time/nearby-agents sentence. The resulting memories feed directly into `decide()`.

Each agent carries a `traits` dict (loaded from `PRESET_TRAITS` or custom). `effective_action_bias()` applies all 5 trait axes — laziness, leadership, creativity, stubbornness, nurturing — as additive drive modifiers. `persist_interactions()` / `restore_interactions()` checkpoint pairwise interaction counts directly to the DB so they survive server restarts.

### Smrti space hierarchy

| Space | Written by | Read by |
|-------|-----------|---------|
| `Agent_Space_{name}` | agent episodes/beliefs | that agent + location-adjusted read set |
| `Place_Space_{name}` | conversations in that place | all agents present there |
| `World_Space` | seeded once at startup (topology facts) | all agents |
| `Space_Culture` | `promote_bridges_to_culture()` | all agents |
| `{a}_x_{b}` | bridge discovery | ephemeral; feeds culture promotion |

Each agent's `read_spaces` is dynamically updated by `_update_agent_read_spaces()` every tick to include the current place's space.

### LLM integration (`llm.py`, `dialogue_queue.py`)

`LLMSettings` is a serialisable dataclass; `GET/POST /settings` updates it at runtime without restarting. Two distinct operations:

- **World generation** — structured JSON prompt with few-shot schema example; `worldgen_max_tokens=3000`, temperature 0.85. Blocks server startup — `worldgen_timeout=300s`.
- **Dialogue** — dispatched through `DialogueQueue` per TALK action; `max_tokens=80`, `dialogue_timeout=60s`. Returns fallback string on any error. The queue bounds in-flight requests so the event loop is never saturated by a burst of simultaneous TALK actions.

Default endpoint: `http://0.0.0.0:8421/v1` (the Smrti proxy). Model: `Qwen3.5-9B-Q8_0.gguf`. Override via `POST /settings`.

### Lifecycle system (`lifecycle.py`)

- **Death**: elder probability scales with years past 65 × delta, doubled if energy < 20. Starvation kills after 48 sim-hours at energy=0.
- **Reproduction**: 2% chance per TALK interaction when gate is met (mutual interaction count ≥20, or ≥10 with romance drive ≥40, both adults, both energy ≥70). Capped at `MAX_POPULATION=20`.
- **Personality inheritance**: child gets Gaussian blend of both parents' 15 `PersonalityProfile` params; variance is multiplied by stress level (derived from parents' average valence).
- **Relationship regression**: pairs regress one relationship tier when their negative-episode count exceeds a threshold. A per-pair cooldown prevents regression from firing every epoch.

### Frontend (`static/`)

Phaser 3 canvas + vanilla JS modules loaded via `<script>` tags (no bundler). All JS is namespaced under the `TOWN` global object. Rendering is **full isometric 3D** — buildings with depth faces, road tiles, walk animation, camera pan/zoom, and mood-tinted agent sprites (green/red tint driven by `mood_valence`). A relationship overlay layer draws colored lines between agents by relationship state.

Key files:
- `app.js` — entry point; creates Phaser game, wires UI, connects WebSocket
- `ws.js` — WebSocket connection with automatic reconnect and demo-mode fallback
- `tick/processor.js` — routes incoming messages (`tick`, `state`, `generating`, `dialogue_patch`, `reset`, `error`) to update `TOWN.state` and call renderers
- `scenes/TownScene.js` — Phaser scene; owns agent sprites, place labels, relationship overlay layer
- `rendering/agents.js` — isometric agent sprites with walk frames, mood tint, and per-tick location cache (O(1) offset lookup)
- `rendering/buildings.js` — isometric building and road tile renderer
- `rendering/` — also: day/night cycle, particles, speech bubbles
- `ui/` — topbar (clock/director badge), sidebar (agent inspector + Town Beliefs panel), eventlog, controls (includes event injection dropdown), settings modal
- `state.js` — shared mutable state (`TOWN.state`)
- `demo/demo.js` — synthetic tick generator used when no server is available

## Key design decisions

- **No LLM in the tick path.** `decide()` is synchronous and deterministic. LLM dialogue is always async and never blocks a tick. Template fallback dialogue is always shown immediately.
- **Year = 28 days.** One year is `HOURS_PER_YEAR = 672` sim-hours (4 seasons × 7 days × 24h). All age calculations use this constant from `config.py`.
- **`BRIDGE_THRESHOLD = 0.3`** (higher than the Smrti default of 0.1) to prevent semantic explosion when many agent spaces cross-pollinate.
- **`dialogue_patch` message type.** After a `tick` message arrives and renders template dialogue, the server may later send `dialogue_patch` with LLM-enriched text for the same tick/speaker pair. The frontend applies it retroactively to the event log.
- **`POST /regenerate` returns 202 immediately.** World generation runs in a background task; the client receives `reset` → `generating` → `state` in sequence via WebSocket.
- **Relationship state is inferred, not stored.** `_infer_relationship_state()` derives the current relationship tier entirely from mutual interaction counts. The progression stranger → acquaintance → friend → close_friend → romantic → married happens automatically.

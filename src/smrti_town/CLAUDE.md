# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Start the town server (port 8430)
smrti serve_town
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

`server.py` creates a `FastAPI` app. On the first WebSocket connection, `_ensure_engine()` calls `worldgen.create_engine_from_llm()`, which calls `llm.LLMClient.generate_world()`. If the LLM is disabled or returns invalid JSON, `scenarios/millbrook.py::create_millbrook()` provides the hardcoded fallback. The result is a `SimEngine` that owns the tick loop.

### The 8-phase tick loop (`engine.py`)

Each call to `SimEngine.tick()` advances simulated time by a variable delta chosen by `Director`:

| Phase | What happens |
|-------|-------------|
| 0 | Director computes `delta_hours`; Chronos fires milestones/birthdays |
| 1 | Drive accumulation; death checks |
| 2 | Perception — each agent queries their Smrti instance for relevant memories |
| 3 | Decision — pure rule-based, **no LLM** |
| 3.5 | LLM dialogue enrichment (fire-and-forget background tasks) |
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

`Agent.decide()` is **pure rule-based** — no LLM calls. Priority: sleep → schedule obligation → highest urgent drive → personality-biased idle. Template dialogue is generated synchronously; LLM enrichment patches it later via `dialogue_patch` WebSocket messages.

`Agent.perceive()` queries Smrti (`top_k=5`) with a location/time/nearby-agents sentence. The resulting memories feed directly into `decide()`.

### Smrti space hierarchy

| Space | Written by | Read by |
|-------|-----------|---------|
| `Agent_Space_{name}` | agent episodes/beliefs | that agent + location-adjusted read set |
| `Place_Space_{name}` | conversations in that place | all agents present there |
| `World_Space` | seeded once at startup (topology facts) | all agents |
| `Space_Culture` | `promote_bridges_to_culture()` | all agents |
| `{a}_x_{b}` | bridge discovery | ephemeral; feeds culture promotion |

Each agent's `read_spaces` is dynamically updated by `_update_agent_read_spaces()` every tick to include the current place's space.

### LLM integration (`llm.py`)

`LLMSettings` is a serialisable dataclass; `GET/POST /settings` updates it at runtime without restarting. Two distinct operations:

- **World generation** — structured JSON prompt with few-shot schema example; `worldgen_max_tokens=3000`, temperature 0.85. Blocks server startup — `worldgen_timeout=300s`.
- **Dialogue** — fire-and-forget per TALK action; `max_tokens=80`, `dialogue_timeout=60s`. Returns fallback string on any error.

Default endpoint: `http://0.0.0.0:8421/v1` (the Smrti proxy). Model: `Qwen3.5-9B-Q8_0.gguf`. Override via `POST /settings`.

### Lifecycle system (`lifecycle.py`)

- **Death**: elder probability scales with years past 65 × delta, doubled if energy < 20. Starvation kills after 48 sim-hours at energy=0.
- **Reproduction**: 2% chance per TALK interaction when gate is met (mutual interaction count ≥20, or ≥10 with romance drive ≥40, both adults, both energy ≥70). Capped at `MAX_POPULATION=20`.
- **Personality inheritance**: child gets Gaussian blend of both parents' 15 `PersonalityProfile` params; variance is multiplied by stress level (derived from parents' average valence).

### Frontend (`static/`)

Phaser 3 canvas + vanilla JS modules loaded via `<script>` tags (no bundler). All JS is namespaced under the `TOWN` global object.

Key files:
- `app.js` — entry point; creates Phaser game, wires UI, connects WebSocket
- `ws.js` — WebSocket connection with automatic reconnect and demo-mode fallback
- `tick/processor.js` — routes incoming messages (`tick`, `state`, `generating`, `dialogue_patch`, `reset`, `error`) to update `TOWN.state` and call renderers
- `scenes/TownScene.js` — Phaser scene; owns agent sprites and place labels
- `rendering/` — agents, buildings, day/night cycle, particles, speech bubbles
- `ui/` — topbar (clock/director badge), sidebar (agent inspector), eventlog, controls, settings modal
- `state.js` — shared mutable state (`TOWN.state`)
- `demo/demo.js` — synthetic tick generator used when no server is available

## Key design decisions

- **No LLM in the tick path.** `decide()` is synchronous and deterministic. LLM dialogue is always async and never blocks a tick. Template fallback dialogue is always shown immediately.
- **Year = 28 days.** One year is `HOURS_PER_YEAR = 672` sim-hours (4 seasons × 7 days × 24h). All age calculations use this constant from `config.py`.
- **`BRIDGE_THRESHOLD = 0.3`** (higher than the Smrti default of 0.1) to prevent semantic explosion when many agent spaces cross-pollinate.
- **`dialogue_patch` message type.** After a `tick` message arrives and renders template dialogue, the server may later send `dialogue_patch` with LLM-enriched text for the same tick/speaker pair. The frontend applies it retroactively to the event log.
- **`POST /regenerate` returns 202 immediately.** World generation runs in a background task; the client receives `reset` → `generating` → `state` in sequence via WebSocket.
- **Relationship state is inferred, not stored.** `_infer_relationship_state()` derives the current relationship tier entirely from mutual interaction counts. The progression stranger → acquaintance → friend → close_friend → romantic → married happens automatically.

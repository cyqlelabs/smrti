# smrti-town

A city-builder where every citizen has a real memory graph.

Each citizen carries a [Smrti](../../README.md) memory engine — beliefs stored as graph nodes with Bayesian truth values, emotional valence, and salience weights. Citizens decide what to do based on what they remember. A mayor and council debate what to build, citizens petition for what they need, and newcomers arrive when the town gives them a reason to.

Point it at a local LLM and it generates the people. Watch it run in an isometric Phaser canvas. Query any citizen's memories over REST.

---

## Quick start

```bash
# Install
pip install smrti

# Start the simulation server (port 8430)
smrti serve town

# Or directly
python -m uvicorn smrti_town.server:app --port 8430
```

Open `http://localhost:8430`. The game begins with the founding sequence:

1. **Place the Town Hall** — click the empty field. This is the only manual building placement required.
2. **Choose the Mayor** — pick from LLM-generated candidates, each with a bio, personality, and governing style. The mayor's personality biases every future council decision.
3. **Meet the Council** — the mayor appoints four advisors (sheriff, superintendent, doctor, treasurer), each arguing from their own domain. Then the simulation starts.

If no LLM is reachable, a hardcoded five-member founding council steps in and everything still runs on template fallbacks.

---

## How it works

### The tick loop

| Phase | What happens |
|-------|-------------|
| 1 | Director picks the time delta (scene / routine / montage / skip) |
| 2 | Each citizen perceives their surroundings and decides — rule-based, no LLM, weighted by recalled memories |
| 3 | Needs drift according to what the citizen is actually doing (9-level Maslow hierarchy) |
| 4 | Actions resolve: movement, eating, working, socializing — each with economic effects |
| 5 | Economy ticks: wages, expenses, taxes, treasury |
| 6 | Milestone and game-over checks |
| 7 | Dialogue enrichment — queued to the LLM via `DialogueQueue`; the tick never waits for it |
| 8 | Council meeting check (daily in sim time) |
| 9 | Immigration check — pull factors and open housing decide who arrives |

### Adaptive pacing

The director adjusts simulated time per tick:

| Mode | Delta | When |
|------|-------|------|
| Scene | 15 min | Two or more citizens share a place |
| Routine | 2 h | Default |
| Montage | 8 h | Everyone sleeping or solo |
| Skip | 168 h | Manual skip-week request |

### Memory shapes behaviour

Citizens read from their private memory space plus `World_Space` and `Space_Culture`. When deciding where to go or whom to talk to, they recall the place or person by name — options with positively-valenced memories win. A hungry citizen who remembers a good meal at the tavern heads back there; one who remembers an argument avoids it. Identical starting states diverge because memory diverges.

### Needs drive everything

Every citizen has a 9-level needs hierarchy (inspired by Maslow): hunger, shelter, health, safety, social, education, purpose, culture, actualization. Lower needs dominate — a hungry citizen won't pursue culture. Unmet needs feed the petition system.

### The council governs

Every sim-day the council convenes. Given the town state (population, treasury, buildings, unmet needs), each advisor argues from their domain and the mayor proposes one action — biased by their personality. The player can **approve** (money is deducted, construction happens), **reject** (the council reconvenes), or **counter-propose** from the building catalog. The LLM writes the debate; a template fallback keeps meetings running offline.

### Citizens petition

Petitions emerge from unmet needs, accumulate signatures when neighbors share the complaint, and appear ranked for the player to approve or dismiss.

### The economy is real

The town treasury pays for construction, salaries, and maintenance; taxes and commerce flow back in. Every citizen has a personal wallet with income and expenses. All building costs come from a canonical catalog that the LLM is also constrained to, so it can never propose something the engine can't build.

### People arrive for a reason

Immigration is driven by pull factors — open housing, jobs, services, reputation. When conditions attract someone, the LLM generates the newcomers (a family for a house, a young couple for a cottage); a fallback generator covers offline mode. Citizens also carry an 8-category skill set that gates professional roles.

---

## LLM integration

The simulation calls an OpenAI-compatible endpoint at four points, never in the hot tick path, always with fallbacks:

- **Mayor candidates** — founding-sequence character generation.
- **Council meetings** — advisor debate and proposal, constrained to the building catalog.
- **Immigrants** — newcomer names, ages, families, skills, personalities.
- **Dialogue** — in-character citizen conversation, queued via `DialogueQueue` and patched into the frontend retroactively over a `dialogue_patch` WebSocket message.

Default endpoint: `http://0.0.0.0:8421/v1` (the `smrti serve proxy` address).
Default model: `Qwen3.5-9B-Q8_0.gguf`.

Change both at runtime without restarting:

```bash
curl -X POST http://localhost:8430/settings \
  -H "Content-Type: application/json" \
  -d '{"base_url": "http://localhost:11434/v1", "model": "llama3.1"}'
```

---

## Configuration

| Env var | Default | Description |
|---------|---------|-------------|
| `SMRTI_TOWN_DB` | `~/.smrti/town.db` | SQLite database path |
| `SMRTI_TOWN_TENANT` | `millbrook` | Tenant ID for all Smrti spaces |
| `SMRTI_TOWN_STATIC` | `src/smrti_town/static` | Override the frontend directory |

---

## REST API

```
POST /opening/place-hall          Step 1: place the Town Hall ({"grid_x", "grid_y"})
POST /opening/choose-mayor        Step 2: pick a mayor ({"candidate_index"})
POST /opening/begin               Step 3: start the simulation

POST /council/approve             Approve the current council proposal
POST /council/reject              Reject it — the council reconvenes
POST /council/counter             Counter-propose a building

GET  /petitions                   Open petitions, ranked
POST /petitions/{idx}/approve     Approve a petition
POST /petitions/{idx}/dismiss     Dismiss a petition

POST /place-building              Place a building ({"building_key", "grid_x", "grid_y"})

GET  /state                       Simulation state snapshot
GET  /agents                      All citizens
GET  /agents/{name}/memories      A citizen's salient memories
GET  /economy                     Treasury and economic stats
GET  /culture                     Space_Culture atoms (town-wide shared beliefs)

POST /start | /pause | /resume    Control the tick loop
POST /skip                        Fast-forward one week of simulated time
POST /regenerate                  Reset to the founding sequence

GET  /settings                    Current LLM settings
POST /settings                    Update LLM settings
```

WebSocket at `/ws` — streams `tick`, `state`, `phase`, `council_meeting`, `council_result`, `council_counter`, `petition_update`, `building_placed`, `immigration`, `event`, `game_over`, `dialogue_patch`, `paused`, `resumed`, `reset`, and `pong` messages.

---

## Memory space hierarchy

```
World_Space          topology and council facts, seeded at founding
Agent_Space_{name}   each citizen's private beliefs and episodes
Space_Culture        town-wide shared values, read by every citizen
```

Every citizen reads from their own space plus `World_Space` and `Space_Culture`, and writes only to their own.

`spatial.py` is wired — it builds the topology every citizen navigates. Only `Place.space_name` is unused: each place is meant to own a `Place_Space_{name}` memory space, and nothing reads one yet. Lifecycle (aging, reproduction, relationship tiers with personality inheritance) and bridge-space culture promotion are implemented in `lifecycle.py` and `culture.py` but not yet wired into the tick loop — [DESIGN.md](DESIGN.md) describes the full game these build toward.

---

## Relationship to Smrti

`smrti-town` ships inside the `smrti` wheel — it lives in `src/smrti_town/` alongside `src/smrti/` and imports the `Smrti` class directly. Every citizen owns a `Smrti` instance.

See the [Smrti README](../../README.md) for the full memory engine documentation.

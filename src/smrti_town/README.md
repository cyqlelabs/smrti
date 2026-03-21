# smrti-town

A town simulation where every resident has a real memory graph.

Each agent carries a [Smrti](../smrti/) memory engine — beliefs stored as graph nodes with Bayesian truth values, emotional valence, and salience weights. Agents decide what to do based on what they remember. When they talk, beliefs cross over. Over time, shared memories consolidate into town culture.

Point it at a local LLM and it generates a new world from scratch. Watch it run in a Phaser 3 canvas. Query any resident's memories over REST.

---

## Quick start

```bash
# Install from repo root
pip install -e .

# Start the simulation server (port 8430)
smrti serve_town

# Or directly
PYTHONPATH=src python -m uvicorn smrti_town.server:app --port 8430
```

Open `http://localhost:8430` to see the canvas. The simulation starts automatically on the first browser connection.

If no LLM is reachable, the server falls back to **Millbrook** — a pre-built town with six residents, a café, a library, a market, and a park.

---

## How it works

### Every tick is 8 phases

| Phase | What happens |
|-------|-------------|
| 0 | Director picks the time delta (scene / routine / montage / skip) |
| 1 | Drive accumulation (hunger, energy, social, curiosity, duty, romance); death checks |
| 2 | Each agent queries their Smrti instance and collects the 5 most salient memories |
| 3 | Rule-based decision — no LLM calls, purely deterministic |
| 3.5 | LLM dialogue enrichment fires in the background; the tick never waits for it |
| 4 | Actions resolve: agents move, eat, sleep, work, talk |
| 5 | Each action becomes an episode written to the agent's memory space |
| 6 | Conversations propagate: place space gets the narration, listener gets their own copy |
| Sporadic | Random events (rain, festival, illness, found item, gossip, power outage, …) |
| 7 | Periodic epoch — memory consolidation, bridge discovery every 10th epoch |

### Memory shapes behaviour

At phase 2, each agent sends a sentence like `"I am at Cafe_Rosetta. It is morning, spring. Marco is here."` to Smrti. The top-5 results — ranked by salience, not recency — feed directly into the decision at phase 3.

A hungry agent who remembers `"Cafe Rosetta has excellent pastries"` will walk toward the café. An agent who never formed that belief will head somewhere else. The simulation diverges from identical starting states because memory diverges.

### Beliefs cross-pollinate through conversation

When two agents talk:
1. The dialogue is written to the **place space** (`Place_Space_Cafe_Rosetta`).
2. A copy lands in the **listener's private space** (`Agent_Space_Marco`).
3. Every agent at that location reads from the place space on the next tick.

Information spreads the way gossip does — through chains of presence and conversation, each agent keeping their own version of what was said.

### Culture emerges from shared memory

Every 10th epoch, Smrti runs bridge discovery across all agent spaces. Beliefs that show up in multiple agents' memories with high confidence get promoted to `Space_Culture`. Every agent reads from `Space_Culture` at all times, so a belief that started in one person's head can become part of the town's shared knowledge without anyone explicitly publishing it.

### Adaptive pacing

The director adjusts simulated time per tick:

| Mode | Delta | When |
|------|-------|------|
| Scene | 15 min | Two or more agents share a place |
| Routine | 2 h | Default |
| Montage | 8 h | All agents sleeping or non-conversational |
| Skip | 168 h | Manual skip-week request |

### Full lifecycle

Agents are born, age through life stages (infant → child → teen → adult → elder), form relationships (stranger → acquaintance → friend → close friend → romantic → married), reproduce, and die. Children inherit a Gaussian blend of both parents' Smrti personality parameters, with variance scaled by the parents' average emotional stress.

Relationship state is inferred from mutual interaction counts — it is never stored as a field. Two agents who keep meeting will naturally cross each threshold.

---

## LLM integration

The simulation uses an OpenAI-compatible endpoint for two things:

**World generation** — on startup (or `/regenerate`), a structured JSON prompt asks the LLM to invent a town: name, description, places with adjacency, agents with backstories and initial beliefs, cultural facts. The server validates and clamps every value. If generation fails or the LLM is unreachable, Millbrook loads instead.

**Dialogue** — every TALK action gets a background task that calls the LLM with the speaker's personality, current drive, location, season, and top memories. The task patches the already-rendered dialogue retroactively via a `dialogue_patch` WebSocket message. Template dialogue is always shown first; the patch updates it when the model responds.

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
GET  /state                       Simulation state snapshot
GET  /agents                      All agents
GET  /agents/{name}               Single agent
GET  /agents/{name}/memories      Agent's salient memories (query, top_k params)

POST /start                       Start the tick loop
POST /pause                       Pause
POST /resume                      Resume
POST /skip                        Skip one week of simulated time
POST /regenerate                  Generate a new world (returns 202, streams via WS)

GET  /settings                    Current LLM settings
POST /settings                    Update LLM settings
```

WebSocket at `/ws` — receives `tick`, `state`, `generating`, `dialogue_patch`, `reset`, and `error` messages.

---

## Memory space hierarchy

```
World_Space          topology facts, seeded once at startup
Agent_Space_{name}   each agent's private beliefs and episodes
Place_Space_{name}   conversations that happened in that place
Space_Culture        beliefs promoted from overlapping agent memories
{a}_x_{b}            ephemeral bridge spaces (cross-agent intersections)
```

Each agent's `read_spaces` list is updated every tick to include the space for their current location.

---

## Relationship to Smrti

`smrti-town` is not packaged separately — it lives in `src/smrti_town/` alongside `src/smrti/` and imports the `Smrti` class directly. Every agent and every socially significant place owns a `Smrti` instance.

See [../smrti/](../smrti/) for the full memory engine documentation.

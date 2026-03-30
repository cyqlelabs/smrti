# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (editable)
pip install -e .

# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_atomspace.py -v

# Run a single test
pytest tests/test_retrieval.py::test_remember_and_recall -s

# CLI
smrti init --db ~/.smrti/memory.db --personality balanced --tenant-id default --space default
smrti status
smrti serve mcp        # MCP stdio server
smrti serve rest       # FastAPI on :8420
smrti serve viz        # FastAPI on :8420 + opens visualizer in browser
smrti serve proxy      # OpenAI-compatible proxy on :8421
```

## Architecture

Smrti is an AtomSpace-inspired memory engine for AI agents. It stores beliefs as graph nodes with Bayesian truth values, emotional valence, and attention weights in a single SQLite file with vector indexing (sqlite-vec).

**Entry point:** `src/smrti/__init__.py` — `Smrti` class is the public facade (`remember`, `recall`, `believe`, `reflect`, `forget`, `status`).

**Core layers (`core/`):**
- `models.py` — Pydantic data structures: `Atom`, `TruthValue`, `AttentionValue`, `Valence`, `Evidence`, `RecallResult`
- `atomspace.py` — Graph operations: add/update atoms, link atoms, boost STI
- `db.py` — SQLite schema (WAL mode). Tables: `atoms`, `vec_atoms` (virtual vec0 for KNN), `evidence` (append-only observation log), `personality`, `aliases`. Key methods: `execute_many` (batched executemany), `execute_batch` (multi-statement single transaction with rollback on failure), `close_database` / `clear_registry` (evict one or all entries from the shared registry — intended for test teardown)
- `embed.py` — Thread-safe FastEmbed singleton (paraphrase-multilingual-MiniLM-L12-v2, 384 dims, ONNX CPU, 50+ languages)

**Retrieval (`retrieval/`):** `fan_out.py` does KNN → 1-hop graph expansion → salience scoring. `salience.py` formula: `w_sim×sim + w_sti×sti + w_conf×conf + w_lti×lti + w_val×|valence|×intensity`. When valence < -0.5, weight dynamically shifts from `w_sti` to `w_valence` so old-but-critical errors outrank recent trivia. `classify.py` classifies recall results into severity levels (`critical_warning`, `known_antipattern`, `context`) based on valence/intensity/probability thresholds.

**Evolution (`evolution/`):** `epoch.py` runs consolidation cycles: process evidence log → decay STI and confidence → propagate STI and valence to 1-hop neighbors → heal orphaned episodes → promote high-STI nodes to LTI → resolve contradictions → discover cross-domain connections (every 10th epoch) → materialize cross-space bridges (every 10th epoch) → prune low-salience atoms. `EpochResult` carries a `bridges_created` field reporting how many bridge atoms were created or updated. `truth.py` implements PLN (Probabilistic Logic Networks) for merging independent probability estimates. `attention.py` handles STI propagation to neighbors. `valence.py` handles emotional valence propagation using a `mood_inertia`-controlled blend (existing valence × inertia + incoming signal × (1 − inertia)); high-inertia presets (e.g. `balanced`=0.8) drift slowly, low-inertia presets (e.g. `empathetic`=0.4) react faster. `healing.py` detects orphaned episodes (episodes that mention concepts but no person atom) and reconnects them to the most salient person in the space, also creating low-confidence `associated` edges from person → concept so LLM-extracted relations supersede.

**Extraction (`extraction/`):** `resolve.py` cascades entity resolution in 5 tiers: exact match (label + entity_type) → cross-type label match (same atom type, e.g. `tool` and `concept` both map to `concept` atom — prevents GLiNER multi-label duplicates) → alias lookup → fuzzy (RapidFuzz) → embedding similarity → create new. `sentiment.py` estimates valence via cosine similarity against positive/negative anchor embeddings — language-agnostic, used by all server modes as a fallback when callers don't provide explicit valence. `ner.py` — zero-shot NER via GLiNER2 (default model `fastino/gliner2-multi-v1`); lazy-loaded thread-safe singleton; configurable via `SMRTI_NER_MODEL`. 16 NER labels (note: `tool` was removed — `technology` covers software tools); when the same text span matches multiple labels, the most specific type wins (priority: person > organization > project > role > technology > skill > preference > constraint > location > event > topic > media > health > concept > goal > pronoun). Verb-phrase spans misidentified as preference/constraint entities (e.g. "Avoid at all costs") are filtered via GLiNER2 `classify_text` (multilingual noun_phrase vs verb_phrase classification). `pronouns.py` — detects pronoun-typed entities returned by NER and merges them into named-entity atoms: first tries the alias table (`resolve_pronouns_via_aliases`), then falls back to GLiNER2 `classify_text` to pick the best candidate; also handles retroactive batch merging at extraction time. `extract.py` — hybrid GLiNER-first extraction pipeline; called by all server modes when `SMRTI_EXTRACT=1`. All server modes use `extract_and_link_serialized()` which acquires a per-`(tenant_id, write_space)` asyncio lock so extractions within the same session run sequentially — this ensures episode N's entities are committed before episode N+1's `_build_entity_context()` query, preventing graph fragmentation from race conditions. Cross-session concurrency is preserved. Three modes via `SMRTI_EXTRACT_MODE`: `hybrid` (default) runs GLiNER NER locally and only calls the LLM for claim extraction when ≥2 entities are found; `llm` restores the original LLM-only behaviour; `local` skips the LLM entirely. In hybrid mode the claims-only LLM call may also emit new entities of type `role`, `technology`, `skill`, `topic`, `media`, `health`, `concept`, `preference`, or `constraint`; preference/constraint reclassify concept atoms to belief in-place, while the rest are created as new concept atoms with a `mentions` edge to the episode. `_link_claims` auto-creates missing claim object atoms as concepts via the entity resolver rather than silently dropping claims with unknown targets. Before each LLM extraction call, `_build_entity_context()` queries the top salient concept, belief, and goal atoms (person, organization, project, role, tool, technology, skill, location, event, topic, media, health, goal, preference, constraint) from the memory graph and injects them as a `[Known entities]` block so the LLM can resolve pronouns ("I" → person, "we" → organization) even when the name isn't in the current message — using persisted memory, not raw conversation history. The extraction prompt (16 canonical entity types) includes a formal JSON schema and nine few-shot examples (tool migration, error/failure with negative valence, goal with coreference, role with coreference, goal/preference/constraint, beliefs, hobbies/pets/fears, health/skill/media/topic, role/focus-area). `max_tokens=4096` gives thinking models (Qwen3, DeepSeek-R1) enough budget to finish their chain-of-thought and still emit the JSON response; `reasoning_content` is used as a fallback when `content` is empty. Thinking mode is controlled by `SMRTI_EXTRACT_THINKING` (`auto` / `disabled` / `enabled`; default `disabled` — extraction works better with thinking off): `disabled` injects `chat_template_kwargs={"enable_thinking":false}` (llama.cpp / vLLM Qwen3 style). Extraction request timeout is controlled by `SMRTI_EXTRACT_TIMEOUT` (seconds, default 60; lower for fast-fail on slow local models). Entity lookup is case-insensitive with `setdefault` collision safety so subject/object casing mismatches never silently drop relation edges.

**Spaces (`spaces/`):** `set_ops.py` implements five set-theory operations on memory spaces — `space_overlap`, `space_intersection`, `space_difference`, `space_union`, `space_symmetric_difference` — all using a **contextual similarity** score that blends three signals: embedding cosine similarity (w=0.6), entity-type compatibility (w=0.2), and graph-neighborhood similarity (w=0.2). The neighborhood signal disambiguates homonyms: two atoms both labelled "Java" will diverge because one's neighbors are "Indonesia/Bali" and the other's are "JVM/Spring". All three signals are language-agnostic. `emergence.py` materialises a **bridge space** from a `SpaceOverlap` result when Jaccard ≥ `min_jaccard` (default 0.1): each matched pair becomes a bridge atom with PLN-merged truth values, averaged STI, max LTI, and intensity-weighted blended valence; `bridge` relation edges connect the new atom back to both source atoms. Bridge spaces are named `{sorted_a}_x_{sorted_b}` so the operation is commutative. Existing bridge atoms are updated in-place (identified by `bridge_source_a`/`bridge_source_b` metadata) to avoid duplication. Bridge discovery runs automatically every 10th epoch via `epoch.py::_discover_bridges`. The five set operations and `materialize_bridge`/`list_spaces` are also exposed on the `Smrti` facade and as five new MCP tools (`smrti_space_overlap`, `smrti_space_intersection`, `smrti_space_diff`, `smrti_space_merge`, `smrti_list_spaces`).

**Personality (`personality/`):** `PersonalityProfile` dataclass with 16 hyperparameters. Six presets (`balanced`, `analytical`, `curious`, `empathetic`, `maverick`, `deterministic`) stored as JSON in `presets/` and loaded into the `personality` DB table per tenant/space pair. `deterministic` is optimized for agentic workflows: fast learning (lr=0.4) + slow decay (0.005), high LTI promotion threshold (0.85), laser-focus attention (boost=0.8, propagation=0.05), and similarity-gated confidence ranking.

**Servers (`servers/`):** `config.py` centralises all shared env-var defaults (`SMRTI_DB`, `SMRTI_PERSONALITY`, `SMRTI_TENANT_ID`, `SMRTI_SPACE`, `SMRTI_READ_SPACES`, `SMRTI_EXTRACT`, `SMRTI_EXTRACT_MODE`, `SMRTI_EXTRACT_URL`, `SMRTI_EXTRACT_MODEL`, `SMRTI_EXTRACT_THINKING`, `SMRTI_IGNORE_PATTERNS`) read by all server modes. `SMRTI_EXTRACT_TIMEOUT` is read directly by `extraction/extract.py` (not via `config.py`). `SMRTI_NER_MODEL` is read directly by `extraction/ner.py` (not via `config.py`). `mcp.py` wraps Smrti as MCP stdio tools; `handle_tool()` recall response includes `severity` and `intensity` fields. All three modes (MCP, REST, proxy) auto-estimate valence via `extraction/sentiment.py` when callers don't supply an explicit value, activating the error-avoidance memory path for negative content. All three modes call `extraction/extract.py` (via `extract_and_link_serialized`) after every `remember` operation to extract entities/claims and build concept nodes + relation edges (enabled by default via `SMRTI_EXTRACT`; proxy uses request auth, proxy and REST forward the request `Authorization` header; MCP passes no auth — works as-is with local LLMs). The proxy's `_store_exchange` awaits user extraction before assistant extraction sequentially so the user's entities are visible when extracting the assistant's response. `rest.py` is a FastAPI REST server. `viz_routes.py` is the shared visualizer router (graph explorer, atom CRUD, LLM call log endpoints) mounted by both REST and proxy. `proxy.py` is an OpenAI-compatible proxy with content-based episode deduplication (skips storing identical episodes for the same tenant/space) and severity-aware memory injection: recalled memories are split into two sections — behavioral constraints (`YOU MUST NOT`, `AVOID` for `critical_warning`/`known_antipattern`) and background context (`Note:` for `context`), each with its own preamble — and contextual query reformulation (configurable via `SMRTI_QUERY_MODE`, `SMRTI_QUERY_CONTEXT_MSGS`, `SMRTI_QUERY_MAX_CHARS`). All severity levels include a `confidence` qualifier. `call_log.py` is a shared in-process ring buffer (200 entries) that captures every LLM call across all serve modes — full request/response, timing, status, error, and tenant_id; both REST and proxy expose `GET /llm-calls` and `DELETE /llm-calls` endpoints, and the visualizer shows them in an "LLM Calls" debug tab. `reflect_loop.py` runs periodic background consolidation across all server modes (interval controlled by `SMRTI_REFLECT_INTERVAL`, default 60s, 0 to disable). `tools.py` defines the 8 advertised MCP tool schemas (remember, recall, reflect, forget, status, personality, space_query, space_merge). `believe`, `space_overlap`, `space_intersection`, `space_diff`, and `list_spaces` are retained as legacy handlers in `handle_tool()` for REST backward-compatibility and direct callers, but are not advertised to MCP clients. `remember` subsumes `believe` (use `type=belief` + `evidence`); `status` includes the spaces list; `space_query` covers overlap/intersection/diff via an `op` parameter.

## smrti-town

Town life simulation built on the Smrti memory engine. Lives in `src/smrti_town/` alongside `src/smrti/` but is **not** included in the `pyproject.toml` wheel — import it by running from the repo root with `pip install -e .` and ensuring `src/` is on `PYTHONPATH`.

```bash
# Start the town simulation server (port 8430)
smrti serve town

# Or directly
python -m uvicorn smrti_town.server:app --port 8430

# Env vars
SMRTI_TOWN_DB=~/.smrti/town.db   # default
SMRTI_TOWN_TENANT=millbrook      # default
SMRTI_TOWN_STATIC=<path>         # override static frontend dir
```

**LLM integration (OpenAI-compatible endpoint):**

The simulation uses an LLM for two things: generating the world at startup and generating in-character dialogue every tick.

- `llm.py` — `LLMSettings` (dataclass, serialisable) + `LLMClient` (async httpx wrapper). `generate_world()` sends a structured JSON prompt with a full few-shot schema example; falls back to Millbrook on failure. `generate_dialogue()` is called concurrently for all TALK actions per tick; falls back to template strings.
- `worldgen.py` — `create_engine_from_llm()` calls `generate_world()`, validates the JSON, builds `TownTopology` + `Agent` list + Smrti spaces, then returns a `SimEngine`. Validates and clamps all LLM-supplied values (personalities, place names, ages, etc.).
- Default endpoint: `http://0.0.0.0:8421/v1` — the `smrti serve proxy` address. Model: `Qwen3.5-9B-Q8_0.gguf`. All settings configurable via `GET/POST /settings`.
- `POST /regenerate` — stops the current engine, generates a new world via LLM, broadcasts `{"type":"reset"}` to WebSocket clients to clear their state, then starts the new engine.

**Architecture:**

- `server.py` tick loop — runs the async simulation cycle (perceive → decide → resolve needs → economy → milestone check → game-over check → dialogue queue → council → immigration). Each tick returns a `TickResult` broadcast to WebSocket clients.
- `agent.py` — `Agent`: drives (Python state) + `Smrti` instance (memory). Rule-based `decide()` — **no LLM calls**; social targets and locations are weighted by recalled memory valence. `traits` dict (from `PRESET_TRAITS` or custom); `effective_action_bias()` applies all 5 trait axes (laziness, leadership, creativity, stubbornness, nurturing) to drive weights. `persist_interactions()` / `restore_interactions()` save and reload pairwise interaction counts to/from the DB directly. Each agent writes to `Agent_Space_{name}` and reads from their space + `World_Space` + `Space_Culture` + current place space.
- `dialogue_queue.py` — `DialogueQueue`: bounded async queue for LLM dialogue enrichment. Prevents unbounded task accumulation by capping concurrent in-flight requests; results are broadcast as `dialogue_patch` WebSocket messages after the tick resolves.
- `director.py` — `Director`: adaptive tick pacing — scene mode (0.25h, ≥2 agents together), routine (2h), montage (8h, all sleeping), skip (168h on demand). `Chronos` fires milestone and birthday events.
- `spatial.py` — `TownTopology` + `Place`: adjacency graph with BFS path distance. `places_by_type()` alias returns places filtered by type tag. Each socially significant place has a `Place_Space_{name}` Smrti instance.
- `lifecycle.py` — relationship gating, death, reproduction, personality inheritance with stress-boosted mutation variance. Relationship regression: pairs regress one tier when their negative-episode count exceeds a threshold; a per-pair cooldown prevents per-epoch spam.
- `culture.py` — `run_bridge_discovery` + `promote_bridges_to_culture`: every 10th epoch, bridge spaces (`{a}_x_{b}`) are scanned; high-confidence atoms flow up to `Space_Culture`.
- `scenarios/fallback.py` — `create_fallback_council()`: fallback starting scenario for LLM-offline mode (hardcoded 5-member founding council, pre-seeded World_Space and Space_Culture facts).
- `server.py` — FastAPI app: WebSocket `/ws` for real-time tick stream, REST endpoints (`/start`, `/pause`, `/resume`, `/skip`, `/state`, `/agents`, `/agents/{name}/memories`, `/culture` (GET — Space_Culture atoms), `/events/inject` (POST — 9 event types)), static frontend served from `static/`.

**Space hierarchy:** `World_Space` (topology facts, written once at startup) → `Agent_Space_{name}` (private per-agent) → `Place_Space_{name}` (per socially-active place) → `Space_Culture` (promoted shared beliefs from bridge spaces). Bridge spaces are ephemeral intersection products named `{sorted_a}_x_{sorted_b}`.

**Key design decision:** Personality hyperparameters are inherited biologically — child gets a blend of both parents' personality params with stress-boosted Gaussian mutation, bounded by `PARAM_BOUNDS` in `config.py`.

## Rules

- **No hardcoded English.** Smrti is multilingual (50+ languages). Never use English word lists, prefixes, or patterns for filtering or logic. Use language-agnostic approaches: GLiNER `classify_text`, embedding similarity, span-length ratios, or word-count heuristics.

## Key Design Decisions

- **Append-only evidence log:** Truth values are primarily updated through the evidence log and merged during epochs via PLN. Direct confidence mutations occur only for policy operations: contradiction resolution (weaken less confident belief) and forget (soften confidence)
- **Multi-tenant isolation:** Every query is partitioned by `tenant_id` and `space`
- **Lazy embedding init:** FastEmbed model loads on first use, not at import time
- **Salience over recency:** Retrieval ranks by a weighted salience score, not timestamps
- **Error-avoidance memory:** Severe negative-valence atoms (valence < -0.7, intensity > 0.7) get an LTI floor of 0.5 on creation, preventing epoch pruning. Salience weights shift dynamically for negative-valence atoms (valence < -0.5). Recall results carry a `severity` classification (`critical_warning` at valence < -0.5 and intensity > 0.5, `known_antipattern` at probability < 0.3 and confidence > 0.3, otherwise `context`) used by all serving modes.

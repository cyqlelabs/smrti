# Smrti as a game engine

An analysis of what the engine can carry in a game, what smrti-town proves and does not, whether the engine fits a mobile product, and three game directions to design on. Written against the code at commit `0cdc309` (2026-09-04).

## Summary

Smrti is a working, tested memory engine whose primitives happen to be game mechanics: beliefs that can be wrong and gain or lose confidence, attention that decides what is on a character's mind, emotional tone that spreads through associations, personalities as tunable and heritable parameter sets, private minds that overlay a shared world, and a consolidation step that changes state overnight. No commercial NPC system exposes that set, and nothing in it needs a language model to run.

Three things decide whether it becomes a game:

1. **The loop is open in smrti-town.** Citizens read memory but never write it during play. The only writes into a citizen's space are the bio at creation, so every place carries the same mood and no citizen has ever avoided a tavern because of an argument. The engine works; the demo has not yet asked it to. Closing that loop is the first job of any game built here, and the benchmark below shows the engine does the right thing the moment it is closed.
2. **The engine is Python and belongs on a server for version one.** SQLite, sqlite-vec and ONNX Runtime all ship mobile builds, so an on-device port is feasible later, but the game logic should be designed so the phone is a renderer and the engine is authoritative.
3. **Retention should come from the engine's real property: the state changes while the player is away, in a way that is deterministic but not predictable.** A night of consolidation is a better hook than a timer.

Of the three directions below, **Hearsay** (a belief and rumour strategy game) is the one that only this engine could power and the one whose sessions fit a phone best. **Millbrook** (smrti-town matured into a cosy builder) reuses the most code and is the safest. **Kin** (a companion with heritable memory) has the strongest attachment loop and the highest running cost. The recommendation is to build one shared foundation, spike Hearsay in text, and keep Millbrook as the product the foundation is measured against.

## 1. What the engine provides, read as game mechanics

Every row here is implemented and covered by tests. The right column is the mechanic the primitive gives a designer without new engine code.

| Primitive | Where | What it does | Mechanic it becomes |
| --- | --- | --- | --- |
| Truth value: probability × confidence | `core/models.py` | A belief can be likely but shaky, or unlikely but firmly held. Confidence is an evidence count (`c = n/(n+1)`). | Rumours, secrets, lies. "How sure is she" is a number. A firm belief takes many observations to move. |
| Evidence log + PLN revision | `evolution/truth.py`, `atomspace.add_evidence` | Append-only observations with `text` and `source`; the epoch revises beliefs toward the count-weighted mean. | Persuasion. Telling an NPC something is filing evidence; who said it is recorded and can be read back (`evidence(atom_id)`). |
| Supersession and contradiction | `extraction/extract.py::_link_claims`, `evolution/epoch.py` | A newer claim about the same subject demotes the older to `SUPERSEDED_PROBABILITY` (0.1). | NPCs update on new information; a debunked rumour becomes something they cite against you. |
| Severity classification | `retrieval/classify.py` | `critical_warning` needs a stated negative on a non-concept; `known_antipattern` is a belief under 0.3 probability held with confidence. | Hard lines and trauma. An NPC that was wronged with a stated tone never treats it as background again. |
| Attention: STI and LTI | `evolution/attention.py`, epoch promotion | STI decays fast and boosts on access; LTI is earned above a threshold and floors user testimony. | What is on their mind today versus what they will never forget. STI is what gets gossiped. |
| Valence: intrinsic tone and absorbed mood | `core/models.py::Valence`, `evolution/valence.py` | The tone a memory was written with never moves; the mood it reports drifts toward its neighbours each epoch, at a rate set by `mood_inertia`. | Emotional contagion. A place sours by association; an empathetic character's map of town changes faster than an analytical one's. |
| Personality: 17 parameters, 6 presets, bounds | `personality/params.py`, `smrti_town/config.py::PARAM_BOUNDS` | Weights for ranking, decay, learning rate, propagation, inertia. | Character traits with mechanical consequences. Heritable: `lifecycle.create_child` blends both parents with stress-scaled Gaussian mutation. |
| Spaces with overlay reads | `Smrti(write_space, read_spaces)` | A mind writes to one space and reads several; every mutation is scoped to the write space. | Private minds over a shared world and a shared culture. Already how smrti-town is laid out. |
| Space set operations and bridges | `spaces/set_ops.py`, `spaces/emergence.py` | Overlap, intersection, difference, union, symmetric difference by contextual similarity (embedding, entity type, neighbourhood). A bridge space holds PLN-merged atoms two spaces share. | What two people agree on. What a faction believes. Inherited memory for a child. Public opinion. |
| Hybrid retrieval with salience | `retrieval/fan_out.py`, `salience.py` | KNN and BM25 fused, one hop of graph expansion, `similarity × (w_sim + standing)`, diversity cap. Multilingual. | "What do you remember about X" in any language, ranked by what the character cares about. |
| Consolidation epochs | `evolution/epoch.py` | Evidence, decay, propagation, healing, promotion, contradiction, association discovery, pruning. Runs per unit of use, never on a clock. | Night. The state genuinely changes between sessions and idle towns do not rot. |
| Forgetting | `Smrti.forget` | Sinks below the floor, stamps `$.forgotten`, excluded everywhere, prunable. | Amnesty, therapy, propaganda. Final by design. |
| Reinforcement | `evolution/reinforcement.py` | Use is weak evidence; capped per epoch. | Repetition makes a belief firmer, but not without limit. |
| Extraction | `extraction/` | Zero-shot NER on ONNX (16 types), five-tier entity resolution, pronoun merge, sentiment, relative dates; LLM claim extraction when an endpoint exists, `local` mode when not. | Free text from a player becomes graph structure without an LLM. With one, it becomes typed claims. |
| One SQLite file, multi-tenant | `core/db.py` | A whole town of minds in one file; tenants are hard walls, spaces are rows. | A save game is a file. A player is a tenant. Export and delete are file operations. |
| Servers | `servers/` | REST, MCP, OpenAI proxy, API key auth, Prometheus metrics, background reflect loop. | A backend exists; it needs a game layer, not an infrastructure layer. |

Two properties matter more than any single row. Everything except LLM claim extraction runs on CPU with no external service, so a game's core logic is deterministic given its graph and reproducible in tests. And every judgement the engine makes reads the intrinsic tone, while the mood it reports is the drifting one, which is exactly the split a game needs: rules read the truth, characters feel the mood.

## 2. Maturity and cost

**Codebase.** The engine is 9.4k lines of Python; the town is 7.4k plus a 2.9k-line Phaser frontend. There are 66 test files and 1,026 test functions; CI runs them on Python 3.10 through 3.13. The repository is three weeks old (first commit 2026-08-17, 50 commits) with one author, so the code is young but the last five commits are a documentation-versus-behaviour audit, which is a good sign about how claims are treated.

**Benchmarks.** The project measures itself on LongMemEval-S (0.975 retrieval hit rate over 40 questions) and HaluMem (0.517 correct, 0.394 hallucination), with the honest caveats in the README. For a game these matter less than for an assistant; they establish that recall returns the memory that answers the question.

**Cost per operation.** The Hugging Face host is blocked from this session, so the embedding model could not be downloaded and the numbers below were taken with a deterministic stand-in embedder. They measure everything except ONNX inference: the graph, the two indexes, the retrieval pipeline and the epoch. The project's own figure for a full `remember` with the real model is about 18 ms, so add roughly 10 to 15 ms per embedded text for inference on a server core.

| Operation | 5 minds, 1k atoms | 5 minds, 2.5k atoms |
| --- | --- | --- |
| `remember` with a stated valence | 1.2 ms | 1.1 ms |
| `recall` top 5, with access boost | 8.6 ms | 14.8 ms |
| `recall` top 5, no boost (the town's mood read) | 5.8 ms | 8.8 ms |
| `reflect` on one mind of 200 / 500 atoms | 35 ms (p95 260 ms on the 10th epoch) | 189 ms (p95 563 ms) |
| `space_overlap` between two minds | 2.1 s (560 × 560 atoms) | 13 s (900 × 900 atoms) |
| Disk, checkpointed, vectors included | 7.9 KiB per atom | 4.3 KiB per atom |

Three consequences for design:

- **Per-tick reads are cheap; per-tick writes are cheaper.** A mind can record every experience it has.
- **Consolidation is a per-day cost, not a per-tick one.** At 500 atoms a mind, fifty minds consolidate in about ten seconds of CPU. Association discovery on every tenth epoch is the spike. Schedule it as the night phase, off the tick thread, which the town already does.
- **Bridges are batch work.** The set operations are pure-Python all-pairs cosine. They belong in a nightly pass over co-located pairs, never in a tick or a request.

One hot path in the town does not fit: `_find_building` calls `_memory_mood`, which runs a recall per candidate place per decision (`smrti_town/agent.py:600`, `:684`). Fifty citizens choosing among ten places is 500 recalls plus 500 query embeddings per tick. The fix is on the engine side and small: let `recall` accept a precomputed query embedding so place and person names are embedded once, and give the town one mood table per mind refreshed at consolidation.

## 3. What smrti-town proves, and what it does not

The town runs end to end: founding sequence, needs hierarchy, rule-based decisions, economy, council with LLM debate and template fallback, petitions, immigration, dialogue queue, adaptive pacing, milestones, game over, an isometric frontend that already has pointer handlers. It also has three modules the tick loop never calls: `lifecycle.py` (death, reproduction, relationship tiers, personality inheritance), `culture.py` (bridge discovery and promotion to `Space_Culture`) and `events.py` (weather, accidents, crises).

What it does not yet prove is the sentence on its own README: "a hungry citizen who remembers a good meal at the tavern heads back there; one who remembers an argument avoids it." The evidence:

- **No citizen writes during play.** Every `remember` call into an `Agent_Space_*` is a bio at creation (`server.py:547`, `:855`, `:864`, `:1429`). Building placement writes to `World_Space` (`server.py:1255`). Nothing in action resolution, interaction, dialogue or events writes an episode to the citizen who lived it.
- **So every place has the same mood.** A fresh citizen asked about the tavern recalls `World_Space: "Tavern is a place in town"` at valence +0.10 and their own bio at +0.10. The benchmark reproduces this: place mood is +0.10 for all eight places until something is written.
- **Recalled memories do not reach decisions.** `perceive` recalls five memories into `PerceptionContext.memories` (`agent.py:278`), but `decide` never reads them; the only consumer is the dialogue prompt (`dialogue_queue.py:145`). The generated dialogue is patched into the frontend and never stored.
- **Relationship gates approximate what the graph holds.** `check_relationship_progression` estimates LTI from interaction counts with the comment "since the actual LTI is in the Smrti memory graph" (`lifecycle.py:249`).
- **`culture.py` cannot run as written.** It calls `space_overlap(other_space=..., tenant_id=...)` and reads the result as a dict; the facade takes `(other_space, threshold)` and returns a `SpaceOverlap` model (`__init__.py:561`). `promote_bridges_to_culture` reads `r.content` and `r.truth` off a `RecallResult`, which carries them on `r.atom`. Both failures are swallowed by broad excepts.

The benchmark shows what happens the moment the loop is closed. After 200 experiences with stated valence per citizen, one citizen's place moods read Tavern −0.32, Market −0.41, Library +0.05, Town Hall −0.04; after eleven consolidation epochs they hold, and on the larger run the Market moved from −0.04 to −0.33 once consolidation had re-ranked its memories and propagated mood along the associations discovery had drawn. That is the behaviour the README describes, produced by the engine as it exists, from one missing writer.

Reading: smrti-town is a rendering of the engine's idea with the engine's feedback loop unplugged. The gap is small in code and large in product, and it is the first thing any of the directions below needs.

## 4. Fit for a mobile product

The engine is Python with ONNX Runtime, sqlite-vec and FastEmbed. None of that runs natively on iOS or Android as is. Three architectures are available.

**A. Server-authoritative engine, thin client (recommended for v1).** One SQLite file per player town on the server; the phone renders state and sends intents over the WebSocket the town already has. Everything in sections 1 and 2 applies unchanged, any LLM can be attached, and the multi-tenant, auth and metrics work is already done. Cost is CPU per *active* town: with cached name embeddings and one recall per citizen per tick, a 50-citizen town is roughly 1.2 s of CPU per tick including inference, so a tick every few seconds is a fraction of a core, and because epochs run per unit of use an idle town costs nothing. Offline play is not possible, but "nothing happens while you are away" is the engine's semantics already; a bounded catch-up on return ("three days passed") is a feature, not a workaround.

**B. On-device port.** SQLite and sqlite-vec have iOS and Android builds; ONNX Runtime has mobile SDKs; the quantised ONNX build of the multilingual MiniLM embedding model FastEmbed downloads is 220 MB, and smaller multilingual models exist. The parts of the engine a game needs (`models`, `atomspace`, `db`, `retrieval/*`, `evolution/*`, `personality`) are about 3,400 lines and would port to Kotlin and Swift, or to one Rust or C++ core. NER and LLM extraction stay off-device; a game controls its own text and can use templates. This gives full offline play, zero server cost and strong privacy. It is a second product phase, not a first.

**C. Hybrid.** Device holds a mirror of its own file for reads; server consolidates and runs any LLM. The file is the sync unit, which is convenient, but two writers on one graph is the kind of problem that eats a small team. Not for v1.

The design rule that keeps A open to B later: the game's rules must be a deterministic function of the graph plus the player's intents, with LLM output treated as flavour that is written *into* the graph (as episodes with estimated valence) rather than as authority. Both smrti-town's "never in the hot tick path, always with fallbacks" and the engine's "estimated valence can never mint a constraint" already enforce this.

## 5. Three directions

Each direction is a game the engine is the logic for, not a game with a memory feature. For each: the pitch, the loop, how the primitives carry it, what a phone session looks like, why a player comes back, what it costs to run, what it reuses, and what could sink it.

### 5.1 Hearsay: a belief and rumour strategy game

**Pitch.** You arrive in a village of twenty people who each remember, believe, doubt and gossip. Information is the only resource. You win by getting the village to believe something, or to stop believing it: elect you, exonerate a friend, find out who really burned the mill.

**Loop.** Each day you have a handful of conversations. In each you can *tell* (file evidence for a claim), *ask* (read what they believe and why), *deny* (file a counter-claim that supersedes) or *listen* (learn who told them what). Then night falls and every mind consolidates. In the morning the village has changed: rumours firmed up or faded, a contradiction resolved against someone, a grudge propagated to everyone the grudge-holder associates with, and NPCs who share a place have traded whatever was highest on their minds. A morning report shows what moved.

**What carries it.**

| Mechanic | Engine primitive |
| --- | --- |
| A claim an NPC holds, with how likely and how sure | `believe` and the truth value; the evidence log holds who said it and the words |
| Persuading someone | Evidence with a weight scaled by their trust in the speaker; the epoch revises by PLN, so a firmly held belief needs many independent sources |
| Being caught in a lie | Supersession: the truth demotes the lie to `known_antipattern`, which NPCs then cite |
| A hard line ("you insulted my mother") | `critical_warning`: a stated negative that outranks recency and never prunes |
| Gossip | Each morning, co-located pairs exchange their top-STI beliefs, filed as evidence with `source` = the gossiper's name |
| Reputation spreading | Valence propagation along relation edges; `mood_inertia` per personality decides who holds a grudge |
| Factions and public opinion | Bridge spaces between minds; promotion to a culture space is "it became common knowledge" |
| Forgetting | Decay per epoch; a rumour nobody repeats fades below the surfacing floor |
| "Why does she believe that?" | `evidence(atom_id)` and the recall trace, rendered as a card |

**Session shape.** Two to four minutes: three to five conversations, then sleep. The night is where the game happens, and the player did not have to be there. No energy meter is needed; the day structure is the pacing.

**Why players return.** The morning report is a variable-reward reveal that the player earned but could not fully predict, which is the honest version of the mechanic slot machines fake. Villages rotate weekly; a mystery scenario has a solution the graph holds. Async multiplayer follows naturally: two players in one village (one tenant), the NPCs' memories are shared state, and each player's evidence carries their own `source`.

**Cost.** The mechanics are subject–predicate–object triples, so every line can be templated in any language and the game runs with no LLM at all. Local NER in `local` extraction mode turns free-text player input into entities without a model call. An LLM is optional flavour for dialogue, which is where the town's dialogue queue already applies.

**Reuse.** The engine as is, plus `smrti_town/config.py::PARAM_BOUNDS` for mutating personalities so no two villagers are alike, `population.py` name and trait generation, `spatial.py` places, and the dialogue queue if flavour is wanted.

**Risks.** Legibility: the player must be able to read a belief state at a glance, so the belief card and the morning diff are the whole UI problem. Tuning: PLN dynamics have to be tuned into something a player can learn (how many tellings move a belief, how fast a rumour dies). The engine has two provenance sources (`user`, `agent`); per-speaker trust needs a small extension, since `evidence.source` is already a free string.

### 5.2 Millbrook: the memory town, loop closed

**Pitch.** The cosy builder smrti-town already is, on a phone, with the promise kept: citizens remember what happened to them, feel differently about places and people because of it, and act on it where you can see.

**Loop.** Approve or counter a council proposal, answer petitions, place a building, read the gossip feed, tap a citizen to see what is on their mind. Sim days pass while you play; the night consolidates every mind that lived that day. Player verbs that act on memory are the differentiator: hold a festival (positive episodes for everyone who attends), give a speech (evidence against a shared grievance), build a memorial (an LTI floor), declare an amnesty (`forget` on a grudge), keep a town chronicle (`World_Space`).

**What carries it.** Everything smrti-town already reads: mood-weighted place and person choice (`agent.py::_memory_mood`), personality presets per citizen, `World_Space` and `Space_Culture` overlays. Plus the three unwired modules, wired: relationship tiers reading LTI from the graph, personality inheritance for children, bridge discovery over co-located pairs feeding culture, events writing episodes to everyone present.

**Session shape.** Five minutes of decisions plus as long as the player wants to watch. Progress while away is bounded by design: epochs are per use, so the town does not rot; on return, a capped number of sim days catch up with a summary.

**Why players return.** Emergent stories they can inspect: a citizen who left because the tavern soured, a friendship that turned into a marriage the graph can explain, a culture belief nobody wrote. The legible mind is the hook; the visualizer in `serve viz` is a developer preview of it.

**Cost.** The council and immigration LLM calls exist with template fallbacks; dialogue is optional. Engine cost is the hot path fix in section 2 and a nightly bridge pass.

**Reuse.** Nearly all of `smrti_town`, including a frontend with pointer handlers that would run in a Capacitor shell for a first mobile build.

**Risks.** Mobile builders are a crowded genre and the memory layer must be visible in the first minute or it is a feature nobody notices. The loop-closing work in section 6 is a hard prerequisite.

### 5.3 Kin: a companion with heritable memory

**Pitch.** Raise a creature that actually remembers you. Tell it things in any language; it forms beliefs about your world, moods that spread, and lines it will not cross. It sleeps, and what you repeated becomes permanent while what you said once fades. Breed two and the child inherits a blend of both personalities and the memories both parents shared.

**Loop.** Daily care plus conversation; a nightly consolidation; occasional breeding, trading and visiting. The creature's behaviour is rule-based from needs (`drives.py`) weighted by memory mood (`agent.py` pattern), exactly the town's citizen with one owner.

**What carries it.** The personality genome (`PARAM_BOUNDS`, `create_child`) and, uniquely, the bridge space: `materialize_bridge` over two parents' spaces is a child's starting memory, PLN-merged. Sentiment estimation gives tone to whatever the player says; local NER builds the creature's map of the player's world; `critical_warning` gives it real aversions.

**Session shape.** One to three minutes, several times a day; the sleep phase is the ritual.

**Why players return.** Attachment, and a genome worth cultivating. Social loops through breeding and visiting are where the retention compounds.

**Cost.** The one direction where an LLM is near the core, because a companion that cannot talk back disappoints. Mitigate with retrieval-based replies (recall plus templates) for the free tier and metered model turns above it. Privacy is a product requirement: players confide; tenants are hard walls and one file per player makes export and deletion literal.

**Risks.** Running cost per user; designing attachment responsibly, which means no punishing the player through the creature and no selling relief from decay; and the same legibility problem as Hearsay.

## 6. Comparison and recommendation

| | Hearsay | Millbrook | Kin |
| --- | --- | --- | --- |
| Needs smrti specifically | Yes: truth values, evidence, supersession, bridges are the game | Partly: mood and personality are the differentiator | Yes: heritable personality and bridge-inherited memory |
| Reuse of existing code | Engine + fragments of the town | Nearly all of the town | Engine + drives, agent pattern, lifecycle |
| Runs without an LLM | Fully | Yes, with template council and no dialogue | Poorly |
| Session fit for a phone | Excellent (turn-based days) | Good (bounded idle) | Excellent (short rituals) |
| Multiplayer path | Async, natural | Visiting, later | Breeding and trading |
| Biggest risk | Making belief state legible and tunable | Genre crowding; loop-closing prerequisite | Cost per user; attachment ethics |
| Foundation shared with the others | Experience writer, belief card, night report, per-speaker trust | Experience writer, mood table, nightly bridge pass | Experience writer, belief card, bridge inheritance |

**Recommendation.** Build the foundation in section 7 first, because all three need it and it is a fortnight of work. Then spike **Hearsay** in text only, no art, as the direction that justifies the engine: if the morning report is fun with twelve villagers and template dialogue, there is a game nothing else can copy. Keep **Millbrook** as the product the foundation is measured against; it is the fallback with the least risk and the most code, and its first mobile build is a shell around the existing frontend. Treat **Kin** as a later product or as a mode of Millbrook (a citizen the player raises) until the per-user LLM cost is known.

A note on "addictive". The engine's honest retention hook is that the world changes overnight in ways the player caused but could not fully predict, and that a mind can be opened and read. Build on that. Timers, energy meters and paid relief from decay would work against the design (the engine is built so that idle memory does not rot) and against store policy and long-term retention.

## 7. Foundation work, in order

Concrete, file-referenced, and shared by every direction.

1. **Experience writer in the tick.** In `server.py` action resolution, after each resolved action write an episode to the acting citizen with a stated valence derived from the outcome (a meal at a place that provides food, a sale, an insult, a robbery); for `TALK` and `INTERACT`, write to both parties and link them with a relation. Events (`events.py`) write to everyone present. This is the change that turns section 3's flat +0.10 into the benchmark's spread.
2. **Feed memory into decisions.** `decide` should read `PerceptionContext.memories`; today only the dialogue prompt does.
3. **Mood table and cached embeddings.** Add `query_embedding=` to `retrieve`/`recall` so names are embedded once; materialise a per-mind entity mood table at consolidation and read it in `_find_building` and `_pick_social_target` instead of five recalls per candidate.
4. **Repair and wire `culture.py`.** Match the facade signatures, read atom fields off `RecallResult.atom`, run bridge discovery nightly over co-located pairs only (it is seconds per pair), then promote.
5. **Wire `lifecycle.py` and `events.py`.** Relationship gates read LTI and shared-episode counts from the graph rather than approximating them.
6. **Engine extensions.** Per-speaker trust (weight evidence by a trust lookup on `evidence.source`); a recall explanation (salience components per result) for the belief card; a space snapshot and restore (a mind as a file) for trading, breeding and support; a seedable RNG in association discovery for reproducible tests.
7. **Hosting shape.** One SQLite file per player town, tenant = player, the existing FastAPI and WebSocket per town, the existing reflect loop as the night. Measure CPU per active town with the real embedding model, which this analysis could not.
8. **Client.** Capacitor shell around the Phaser frontend for the first device build; a native client once the direction is chosen.

## Appendix: how the numbers were taken

`bench/game_tick.py --stub` (committed beside this document): one `World_Space` with eight places, five citizen spaces with random presets, 200 or 500 episodes per citizen of the form "*name* *verb* *place* with *person*" with a stated valence (30% negative), then timed `remember`, `recall` with and without the access boost, eleven `reflect` epochs, one `space_overlap`, and the checkpointed file size. The `--stub` flag swaps in a hashed bag-of-words and character-trigram stand-in normalised to 384 dimensions, used because the Hugging Face host that serves the real model is not reachable from this session; without the flag the script runs the real model. every figure therefore excludes ONNX inference and should be read alongside the project's own 18 ms per `remember` with the real model. Latencies are medians on one CPU core with p95 in parentheses where it differed materially.

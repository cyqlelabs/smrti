# Smrti Engine Analysis — Claims Audit & Hardening Specification

**Date:** 2026-07-21 · **Scope:** every claim in `README.md`, `CLAUDE.md`, and `docs/pipeline.md`, verified line-by-line against the code on `main` (c7b2fe9), plus a full test-suite run and a proposed specification for closing the gaps.

---

## 1. Executive summary

The core engine is real and mostly does what the documents say: the SQLite/vec0 schema, the salience formula, severity classification, the LTI floor for severe negative memories, the personality presets, the epoch phase ordering, the hybrid GLiNER→LLM extraction pipeline, the 8-tool MCP surface, and the ignore-pattern plumbing all check out with exact file:line evidence. The test suite is broad and passes.

However, the audit found **four systemic defects that quietly undermine the engine's stated purpose**, a set of **stale or false documentation claims**, and in `smrti_town` a case where **the documents describe a program that no longer exists**. None of these are surfaced by the current tests, which largely verify happy paths with mocked vectors and hand-built graph states.

The four systemic defects, in order of impact:

| # | Defect | Effect |
|---|--------|--------|
| D1 | `vec_atoms` uses **L2 distance, but the code interprets it as cosine similarity**, and has **no space column** | Salience's `w_sim` term is miscalibrated and clamps to 0 for moderately related atoms; the tier-3 embedding entity-resolution tier is effectively dead; KNN candidates can be starved out by unrelated spaces in the same tenant |
| D2 | **Valence propagation erodes protected memories** | A critical error memory (valence −0.9) next to any active atom decays past the −0.5 `critical_warning` threshold in ~3 epochs — under the default 60 s reflect loop, the error-avoidance feature self-destructs in under 5 minutes |
| D3 | **Epochs are not serialized and decay is epoch-frequency-dependent** | Concurrent `reflect()` calls double-apply evidence; memory half-life is silently a function of `SMRTI_REFLECT_INTERVAL` (a belief untouched for ~35 min halves in confidence at defaults) |
| D4 | **smrti_town's memory→behaviour loop is inert** | Every consumer reads `r.content` off `RecallResult` (correct access is `r.atom.content`) inside `except Exception: pass` — recalled memories are always empty, valence weighting always 0, and no episode is ever written during simulation |

Sections 2–4 give the claim-by-claim verdicts, the full bug inventory, and the documentation drift list. Section 5 is the forward-looking part: concrete specifications ("new logical paths") that would make the engine work as designed while honoring its purpose — salience-governed, emotionally weighted, multi-tenant graph memory.

---

## 2. Claim-by-claim verification

Verdicts: ✅ implemented as documented · 🟡 partial / caveats · ❌ broken or not implemented.

### 2.1 Core & storage (`core/`)

| Claim | Verdict | Evidence |
|---|---|---|
| WAL-mode SQLite; tables `atoms`, `vec_atoms` (vec0), `evidence`, `personality`, `aliases` | ✅ | `core/db.py:140-143`, `46-129` |
| `execute_many` batched; `execute_batch` single transaction w/ rollback; `close_database`/`clear_registry` | ✅ | `core/db.py:174-188`, `27-42` |
| Thread-safe lazy FastEmbed singleton, multilingual-MiniLM, 384 dims | ✅ | `core/embed.py:6-48` ("ONNX CPU" is fastembed's default, not pinned) |
| "**Every** query is partitioned by `tenant_id` and `space`" | 🟡 | Reads are; KNN is tenant-only (no space column in vec0, `db.py:76-81`); many mutations are id-only (`atomspace.py:26`, `248-252`, `epoch.py:78,160,189-196`). Isolation rests on UUID unguessability, not on the queries |
| Error-avoidance LTI floor (valence < −0.7 ∧ intensity > 0.7 → LTI ≥ 0.5) | ✅ | `core/atomspace.py:67-71`; tested |
| Append-only evidence; truth updated via PLN during epochs; direct mutation only for contradiction/forget | 🟡 | Evidence rows are insert-only but flagged/deleted later; the epoch path uses an ad-hoc weighted blend (`evolution/truth.py:7-26`), **not** the PLN revision rule — real PLN (`models.py:36-48`) is only used for bridge atoms; decay and `update_atom` also mutate confidence directly |
| Facade `remember/recall/believe/reflect/forget/status` | 🟡 | All present; but `believe()`'s `evidence` string is used only as a truthiness gate — the text is silently discarded (`__init__.py:177-184`; `Evidence` has no text field) |

### 2.2 Retrieval (`retrieval/`)

| Claim | Verdict | Evidence |
|---|---|---|
| Embed → KNN top-50 → 1-hop expansion → salience → top-k, filtered to read spaces | 🟡 | Pipeline exists (`fan_out.py:28-126`); but KNN candidates are tenant-wide (starvation risk, D1), person atoms are unconditionally injected as candidates (`fan_out.py:82-86`, undocumented), and severe-negative atoms bypass the confidence floor (undocumented) |
| Salience formula + dynamic w_sti→w_val shift at valence < −0.5 | ✅ | `salience.py:35-46`, exact match |
| Classification thresholds (critical_warning / known_antipattern / context) | ✅ | `classify.py:16-20`, exact match, boundary-tested |
| `min_confidence_to_surface` excludes atoms from recall | ❌ | The personality parameter is **never used in retrieval** — recall uses the caller's `min_confidence` param (default 0.1); the preset value only gates epoch pruning (`epoch.py:177`). E.g. empathetic's 0.3 floor has zero effect on recall |
| Similarity = embedding **cosine** similarity | ❌ | vec0 table declares no `distance_metric` → sqlite-vec computes **L2**; `fan_out.py:108` does `1 − distance` and clamps at 0 (D1). Ranking inside the KNN set survives; cross-signal salience calibration does not |

### 2.3 Evolution (`evolution/`)

| Claim | Verdict | Evidence |
|---|---|---|
| Epoch order (9 steps incl. every-10th connections/bridges); `EpochResult.bridges_created` | ✅ | `epoch.py:61-196`; counter at `models.py:104` |
| PLN merging of evidence | 🟡 | See 2.1 — legit PLN exists but the evidence path doesn't use it; `pln_merge` (`truth.py:29-31`) is called by nothing |
| STI propagation / decay / LTI promotion | 🟡 | Propagation real (`attention.py:17-49`); but `decay_sti`/`promote_lti` helpers are dead code (epoch inlines SQL), and promotion is a one-shot `lti = MAX(lti, sti·0.5)` snapshot, **not** "cumulative STI increments LTI" |
| Valence propagation (mood-inertia blend) | 🟡 | Formula matches docs (`valence.py:49-56`) — but the formula itself erodes strong valences toward `source × 0.1` (D2) |
| Orphaned-episode healing; low-confidence `associated` edges that LLM relations supersede | 🟡 | Implemented (`healing.py`); "supersede" is only by confidence value — `_create_relation` skips if *any* edge of that name exists, so a healed edge actually blocks a later LLM `associated` edge; all orphans bind to the single most salient person in the space |
| Contradiction resolution (weaken less-confident belief) | ❌ | Resolution code exists (`epoch.py:137-163`) but **nothing ever creates a `contradicts` edge** — detection doesn't exist, so the step is dead outside tests. If an edge were created, the ×0.8 penalty compounds **every epoch** (annihilating the loser in ~40 min at defaults) because the edge is never marked resolved |
| Pruning below floors; severe-negative atoms survive | ✅ | `epoch.py:177-196`; LTI never decays so the 0.5 floor is permanent — but cascade-deleted relations leave stale `vec_atoms` rows forever |
| Every-10th-epoch gating persisted | ✅/🟡 | `epoch_count` persisted in `personality` (`db.py:117`); but the increment-then-read is racy across concurrent epochs |

### 2.4 Extraction (`extraction/`)

| Claim | Verdict | Evidence |
|---|---|---|
| 5-tier resolution cascade in documented order | ✅ | `resolve.py:53-115` (docstring omits tier 0b) |
| Tier-3 embedding similarity | ❌ | Threshold 0.3 interpreted as cosine but is L2 → only near-identical strings match (tier effectively dead); query also has **no space filter** → cross-space merges + foreign-space aliases (`resolve.py:96-112`) |
| Anchor-embedding sentiment, no hardcoded-English logic | 🟡 | Mechanism is the sanctioned approach; anchors are English *sentences* (`sentiment.py:11-22`) working cross-lingually via the multilingual model — gray area, not a word-list violation |
| GLiNER2 NER: 16 labels, priority order, verb-phrase filter, lazy singleton, `SMRTI_NER_MODEL` | ✅ | `ner.py` throughout; caveat: verb-phrase filter's `split()` word count never fires for unspaced CJK |
| Hybrid mode: LLM only when ≥2 entities; serialized per (tenant, space); `[Known entities]` context; case-insensitive lookup; thinking/timeout env vars | ✅ | `extract.py:424-513`, `571-577`, `139-161`; gate is ≥2 *resolved atom ids*, not raw NER count |
| Pronoun resolution: aliases first, classify_text fallback picks best candidate | 🟡 | Order inverted in code: classify_text *detects* pronoun-ness, aliases *resolve*; there is no classify-picks-candidate step — unresolved pronouns are dropped (`pronouns.py:85-115`) |
| Prompt: 16 types, "formal JSON schema", "eight few-shot examples" | 🟡 | 16 types yes; "schema" is a one-line shape sketch; examples number **nine** |

### 2.5 Spaces & personality

| Claim | Verdict | Evidence |
|---|---|---|
| 5 set ops with contextual similarity 0.6/0.2/0.2 | ✅ | `set_ops.py:31-33`, `114-134` (weights only hold when an embed engine is passed — the facade always passes one) |
| Bridge materialization (PLN merge, avg STI, max LTI, blended valence, commutative naming, in-place update) | 🟡 | All implemented (`emergence.py`) — but dedup key is order-dependent → duplicate bridges when overlap runs from the other side (the smrti-town pattern); "created **or updated**" return only counts creates |
| Default threshold 0.85 workable | ❌ | Untyped, neighborless atoms (facade-created episodes) max out at 0.80 contextual similarity even for identical content — cross-space episode matching is structurally impossible via the facade |
| "16 hyperparameters", presets "stored as JSON in presets/ and loaded" | 🟡 | 15 tunables + `preset_name`; README's own table lists 15. The JSON files are **never read** — presets come from the Python dict (`personality/params.py:30-111`); `presets/analytical.json` has already drifted (`valence_propagation: 0.05` vs effective 0.1) |
| README default values match balanced; deterministic values as documented; mood_inertia 0.8/0.4 | ✅ | Verified programmatically, all 15 values match |
| `Smrti(..., personality="curious")` applies the preset | ❌ | Only on first init of a space; afterwards the constructor arg is ignored unless the `SMRTI_PERSONALITY` env var is set (`__init__.py:77-80`) |

### 2.6 Servers (`servers/`)

| Claim | Verdict | Evidence |
|---|---|---|
| `config.py` centralizes the 11 documented env vars | ✅ | `config.py:6-32` |
| MCP: exactly 8 advertised tools + legacy handlers; recall carries severity/intensity | ✅ | `tools.py`, `mcp.py:238-247`, `72-73` (one dead duplicate `smrti_believe` branch) |
| Valence auto-estimation + extraction + auth forwarding in all three modes | ✅ | `mcp.py:36-37,252-259`, `rest.py:92-99`, `proxy.py:131,339-353,440-442` |
| Proxy: dedup, two-section severity injection, query reformulation, user-before-assistant extraction | ✅ | `proxy.py:124-130`, `192-202`, `272-296`, `205-224`, `328-345` |
| "All severity levels include a **confidence qualifier**" | ❌ | No confidence appears anywhere in the injected text (`proxy.py:192-202`); it exists only in the log payload. README:178 and CLAUDE.md are both wrong |
| Streaming proxy | ❌ | `chunk.get("choices", [{}])[0]` raises IndexError on the empty-`choices` usage chunk that openai-python sends by default with `stream_options.include_usage` → stream aborts with a fake error **and the exchange is never stored**; upstream 401/429/500 during streaming returns HTTP 200 with corrupted SSE framing (`proxy.py:403-517`) |
| `SMRTI_IGNORE_PATTERNS` in all modes, dropped before embedding | ✅ | `config.py:15-16`, `__init__.py:64-69,129-130` |
| call_log ring buffer + `/llm-calls` endpoints + viz tab | ✅ | `call_log.py:10`, `viz_routes.py:126-159` (+ undocumented SSE stream endpoint) |
| reflect_loop across all modes, default 60 s, 0 disables | ✅ | `reflect_loop.py:13-30`, wired in all three lifespans |
| viz "atom CRUD" | 🟡 | Read-only: only `GET /atoms/{id}` exists (`viz_routes.py:118-124`); `/metrics` (Prometheus) exists |

### 2.7 smrti-town (`src/smrti_town/`)

The CLAUDE.md/README section describes an architecture that is **no longer in the tree**: there is no `engine.py`/`SimEngine`/`TickResult`, no `scenarios/millbrook.py` (fallback is a hard-coded 5-member council), no `generate_world()`/`create_engine_from_llm()`, no epoch/reflect call anywhere, no place Smrti spaces (`Place.smrti` is never assigned), no `/events/inject` endpoint, and no `effective_action_bias()` (traits are 8 axes, not 5). `lifecycle.py`, `culture.py`, and `events.py` exist as documented but are **dead code — never imported by the runtime** — and `culture.py` would fail on two API mismatches if it were called. What actually runs is a city-builder (opening sequence → mayor → council → economy/petitions/immigration) with the tick loop in `server.py:122-606`. Verdict for nearly every town claim: ❌, with `DialogueQueue`, `Director` pacing, spatial BFS, ports/env vars, and wheel exclusion as the ✅ exceptions.

---

## 3. Bug inventory (deduplicated, ranked)

### Critical

| ID | Bug | Location |
|----|-----|----------|
| C1 | **L2-vs-cosine mismatch** across all vec consumers: retrieval similarity, tier-3 entity resolution, connection discovery. No `distance_metric=cosine` on vec0; embeddings never explicitly normalized; `1 − L2` clamps to 0 below cos ≈ 0.5 | `db.py:76-81`, `fan_out.py:108`, `resolve.py:96-112`, `connections.py:32-37` |
| C2 | **Valence propagation erodes protected error memories** to `source×propagation` fixed point; intensity erodes identically; crosses the −0.5 classification threshold in ~3 epochs (~3 min at default reflect interval) | `valence.py:49-56`, `epoch.py:101-114` |
| C3 | **smrti_town RecallResult API mismatch** (`r.content` vs `r.atom.content`) swallowed by bare `except` at every call site → all memory-driven behaviour, `/agents/{name}/memories`, `/culture` silently return empty | `agent.py:284-291,653-671`, `server.py:362,1075-1082,1330-1338`, `culture.py:151-162` |
| C4 | **Proxy streaming aborts on empty-`choices` chunks** (standard usage chunk) → no memory stored for streamed exchanges; upstream errors relayed as 200 + garbage SSE | `proxy.py:502`, `403-517` |

### High

| ID | Bug | Location |
|----|-----|----------|
| H1 | Concurrent epochs double-apply evidence (read-unprocessed → apply → flag is not atomic; no per-space epoch lock; background loop + manual reflect + multi-process all collide); same race on the every-10th gate | `epoch.py:61-84`, `27-35` |
| H2 | Decay is per-epoch, not per-wall-clock: `SMRTI_REFLECT_INTERVAL` silently controls memory half-life (~35 min confidence half-life at defaults); fresh atoms decay immediately | `epoch.py:86-99` |
| H3 | Contradiction penalty compounds every epoch; edge never resolved; detection nonexistent | `epoch.py:137-163` |
| H4 | Bridge dedup key is order-dependent → duplicate bridge atoms/edges when spaces run overlap in both directions (every 10th epoch in multi-space tenants) | `emergence.py:129`, `epoch.py:210-232` |
| H5 | Contextual-similarity threshold 0.85 unreachable for untyped/neighborless atoms (max 0.80) | `set_ops.py:114-134` |
| H6 | Stale `vec_atoms`: content changes never re-embed (`add_atom` skips existing, `update_atom` never touches vec); pruning cascade never deletes relation-atom vec rows → KNN budget permanently consumed by ghosts | `atomspace.py:73-87,99-131`, `epoch.py:189-196` |
| H7 | Blocking ML inference on the event loop: `estimate_valence` (incl. first-call model download) and MCP's whole `handle_tool` run synchronously in async handlers; REST endpoints are `async def` calling sync embedding work — serializes all traffic | `proxy.py:131`, `mcp.py:250-251`, `rest.py:89-136` |
| H8 | Security: call log stores all client headers, masking only `authorization` — `x-api-key`, `cookie`, etc. served verbatim by unauthenticated `GET /llm-calls` with CORS `*` on default bind `0.0.0.0`; viz `?db=` opens/creates arbitrary filesystem paths | `proxy.py:359-363,391`, `viz_routes.py:25-37` |
| H9 | Tier-0b cross-type resolution merges any same-label entities whose types both map to `concept` — i.e. person "Paris" ≡ location "Paris"; fuzzy WRatio ≥ 85 merges "Python 2"/"Python 3" and poisons the alias table permanently | `resolve.py:66-73,88-93,123-127` |
| H10 | Cross-tenant/space overwrite class: `INSERT OR REPLACE` keyed on global id; id-only updates/deletes without tenant guard | `atomspace.py:26`, `epoch.py:78,160,189-196` |

### Medium (selected)

- `discover_connections` ignores space boundaries (contradicts its own docstring) and links cross-space `associated` edges — silent space leakage every 10th epoch (`connections.py:32-60`).
- Proxy dedup matches against *all* episodes ever — a repeated "sounds good" months later is dropped with no STI reinforcement (`proxy.py:124-130`).
- Proxy `_instances` unbounded, minted by unauthenticated headers (fd/memory DoS); `_db_cache` evicts without closing (`proxy.py:72-101`, `viz_routes.py:29-30`).
- Prompt injection surface: GLiNER spans and stored atom labels substituted into the extraction *system* prompt without delimiters/caps; LLM-returned predicates stored unvalidated (`extract.py:356-368,266-272`).
- Retroactive pronoun merge can delete atoms that `_link_claims` then references → dangling edges (`pronouns.py:206-207`, `extract.py:332`).
- `add_atom` is 4 separate autocommit statements → orphan vec rows under concurrent prune (`atomspace.py:24-87`).
- Extraction has zero logging; LLM parse failures (any prose before the JSON) silently no-op (`extract.py:111-118,408-415`).
- Sentiment anchor init has a double-checked-lock ordering race (`sentiment.py:38-59`).
- Non-stream proxy path has no error handling: connect errors/HTML bodies → uncaught 500, log entry stuck at status 0 (`proxy.py:424-429`).
- viz endpoints hardcode `("default","default")` — wrong tenant under the proxy (`viz_routes.py:56-170`).
- Town: elder death probability not scaled by tick delta (8× faster in scene mode); `/settings` bool coercion (`bool("false") is True`); stub-citizen signature mismatch aborts every tick; `/regenerate` leaks prior run's memories (`agent.py:260`, `server.py:1277,1422,1293-1318`).

### Packaging / CI

- **Tests run only on tag pushes** (`.github/workflows/publish.yml`, `on: push: tags`) — there is no per-PR/per-push CI. The README CI badge reflects release runs only.
- `chonkie-core` is a real PyPI dependency (verified, install succeeds), but its `Chunker` API call is guarded only against `ImportError` — an API change would raise uncaught for texts > 1500 bytes (`ner.py:24-32`).
- No dev/test extra in `pyproject.toml` (pytest et al. undeclared).

---

## 4. Documentation drift (both directions)

Claims that need editing rather than code fixes:

1. CLAUDE.md's Spaces paragraph still advertises "five new MCP tools (`smrti_space_overlap`, …)" — contradicting its own Servers paragraph and the code (consolidated `space_query`/`space_merge`). `space_union`/`space_symmetric_difference` have no server exposure at all (facade-only).
2. "16 hyperparameters" is 15 (+ `preset_name`); README's own tables list 15.
3. "Presets stored as JSON in `presets/` and loaded" — JSON is never read; source of truth is the Python dict.
4. "Confidence qualifier" in proxy injection — not implemented.
5. "Atom CRUD" in viz — read-only.
6. "Eight few-shot examples" — nine; "formal JSON schema" — a shape sketch.
7. Pronoun pipeline order (aliases-then-classify) is inverted in code.
8. "Cumulative STI increments LTI" — promotion is a snapshot max.
9. The **entire smrti-town section** of CLAUDE.md/README describes a previous incarnation; `src/smrti_town/DESIGN.md` + code are the reality. Rewrite or delete.

---

## 5. Hardening specification — new logical paths

These are the designs I'd implement, in dependency order. Each is scoped to preserve the engine's purpose: *salience-governed recall where graph topology, Bayesian truth, attention, and emotion outrank raw similarity*.

### 5.1 Vector layer v2 (fixes C1, part of H6, tier-3 death)

Schema migration (vec0 tables can't be altered — rebuild on schema-version bump):

```sql
CREATE VIRTUAL TABLE vec_atoms USING vec0(
    atom_id      TEXT,
    embedding    float[384] distance_metric=cosine,
    tenant_id    TEXT partition key,
    space        TEXT partition key,          -- NEW: prunes KNN per space
    +label       TEXT,
    +content_hash TEXT                        -- NEW: staleness detection
);
```

- `similarity = 1 − cosine_distance` becomes a true cosine similarity in [0, 2]→clamped [0,1]; recalibrate `SIM_THRESHOLD` in `resolve.py` (0.3 L2 ≈ 0.955 cos-sim today; a real synonym threshold is ~0.70 cos-sim).
- Normalize embeddings explicitly at embed time (`v / ‖v‖`) instead of trusting model output; serialize little-endian (`struct.pack('<384f', …)`).
- KNN queries pass `space IN read_spaces` (multi-partition filter), eliminating candidate starvation and tier-3 cross-space leakage in one stroke.
- `add_atom`/`update_atom` compare `content_hash`; on mismatch delete + re-insert the vec row inside one `execute_batch` with the atom write (kills stale-embedding divergence and the orphan-row race).
- Stop embedding `type='relation'` atoms entirely; their synthetic labels pollute KNN and consume LIMIT budget. Backfill: delete existing relation rows from `vec_atoms`.

### 5.2 Truth maintenance on real PLN (fixes the PLN claim, strengthens beliefs correctly)

Single count-space rule used by *both* the epoch evidence path and bridges:

```
n      = k · c / (1 − c)                 # confidence → evidence count (k = 1)
n_ev   = evidence_weight · confidence_update_lr
p'     = (p·n + p_ev·n_ev) / (n + n_ev)
n'     = n + n_ev
c'     = n' / (n' + k)
```

- Batch all pending evidence per atom into one merge per epoch (removes per-row UPDATE churn and shrinks the H1 race window).
- Delete the never-called `pln_merge` or make it this function; make `update_truth` a thin wrapper.
- Add a `note TEXT` column to `evidence` and persist `believe()`'s justification string instead of discarding it. `Evidence` model gains the field; append-only stays true.

### 5.3 Wall-clock, protected-memory-aware consolidation (fixes C2, H1, H2, H3)

**Epoch serialization.** Wrap `run_epoch` in an advisory claim: `UPDATE personality SET epoch_running = 1 WHERE tenant_id=? AND space=? AND epoch_running = 0` — if 0 rows change, skip this cycle (another epoch is live, possibly in another process). Release in `finally`, with a staleness timeout column for crash recovery. Evidence application then happens inside one `execute_batch` transaction: fetch unprocessed → merge → flag, atomically.

**Wall-clock decay.** Replace per-epoch multiplicative decay with `factor = exp(−λ · Δt_hours)` computed from each atom's `updated_at`, where λ derives from the personality rate normalized to a reference interval (`λ = rate · 60 / reference_seconds`). Properties: idempotent under any reflect interval, spares just-written atoms, and two processes reflecting concurrently no longer double-decay (the second sees Δt ≈ 0).

**Fixed-point-safe valence propagation.** Propagate *toward the source's actual valence* with a bounded step, never eroding a stronger same-sign valence:

```
target_v = source_valence                        # not source × 0.1
step     = valence_propagation · source_intensity · (1 − mood_inertia)
v'       = v + step · (target_v − v)
   but if sign(v) == sign(target_v) and |v| > |target_v|: v' = v   # never weaken
```

Additionally, atoms carrying the error-avoidance LTI floor get a **valence floor**: `|v'| ≥ min(|v_at_creation|, 0.7)` so no propagation path can declassify a `critical_warning`. This is the single highest-leverage change for the engine's stated purpose.

**One-shot contradiction resolution + actual detection.** On resolution, stamp the relation atom's metadata (`resolved_at`) and skip stamped edges. Detection becomes a real pipeline stage: at claim-extraction time, when a new belief shares a subject entity with an existing belief, run a cheap language-agnostic check — embedding similarity high (same topic) while the claims carry opposing polarity signals (negation classification via GLiNER2 `classify_text`, or an optional NLI cross-encoder behind an extra) — and emit the `contradicts` edge the epoch already knows how to consume. Until detection ships, the docs should stop claiming the feature.

### 5.4 Retrieval integrity

- Resolve the recall floor as `explicit param → personality.min_confidence_to_surface → 0.1`, so presets govern surfacing as documented.
- Document (or gate behind a personality flag) the unconditional person-atom injection and the severe-negative floor bypass — both are sensible for the purpose but currently undisclosed behavior.
- Cap `boost_sti` at the same 3.0 ceiling used everywhere else.

### 5.5 Entity resolution correctness (H9)

Replace the blanket `concept` fallback with an explicit compatibility matrix: `{technology, concept, topic, skill}` mutually mergeable; `{person}`, `{organization}`, `{location}`, `{event}` only self-mergeable. Reject fuzzy merges when labels differ in digit/version tokens (`"Python 2" ≠ "Python 3"` — compare digit sequences before accepting WRatio). Lowercase the alias PK (lookup is already `LOWER()`-based) to stop case-variant aliases pointing at different atoms.

### 5.6 Spaces: reachable thresholds and commutative bridges (H4, H5)

- Renormalize contextual-similarity weights over *available* signals (the `embed_engine=None` path already does this): an atom pair with no types and no neighbors is scored on embedding alone, so identical content ≈ 1.0 and the 0.85 threshold means what it says.
- Order-normalize the bridge dedup key: `key = tuple(sorted((id_a, id_b)))`, and check both metadata orientations when scanning existing bridges. Skip bridge discovery when the *current* space is itself a bridge space (prevents `A_x_B_x_C` nesting).
- Count updates in `materialize_bridge`'s return (`created + updated`), matching its docstring.
- Performance: memoize neighbor-context embeddings per atom id for the duration of a set-op call, and compute the pair grid with numpy — the epoch-time bridge scan currently re-embeds an atom once per candidate pair.

### 5.7 Server hardening (C4, H7, H8)

- **Stream relay:** open the upstream stream *before* constructing `StreamingResponse` so upstream status propagates; guard every `choices` access (`if chunk.get("choices"):` else pass the frame through untouched — usage chunks reach the client and `[DONE]` handling stays intact); wrap the non-stream path in try/except that relays upstream error JSON with its real status.
- **Event loop:** move `estimate_valence` + dedup into the existing executor closure; convert REST handlers to sync `def` (FastAPI threadpools them); pre-warm the embedder and sentiment anchors at lifespan startup.
- **Security:** allowlist logged headers (content-type, x-smrti-*, masked authorization) instead of a one-entry denylist; gate `/llm-calls`, `/graph`, `/metrics`, and especially the `?db=` parameter behind an env-configured token or localhost check; bound `_instances` with a closing LRU and validate tenant/space header charset/length.
- **Dedup:** window the duplicate check (e.g. 24 h) and on a hit boost the existing atom's STI instead of dropping the signal entirely — repetition is evidence, and the current behavior throws it away.
- Either implement the documented confidence qualifier in `_format_memory` (e.g. `(confidence: high/medium/low)` suffix derived from `truth.confidence`) or delete the claim; implementing it is ~5 lines and genuinely useful to the downstream LLM.

### 5.8 Extraction robustness

- Parse LLM output with `json.JSONDecoder().raw_decode` scanning for the first balanced object; validate shapes/ranges (valence clamped, predicate length/charset capped) before DB writes; add `logging` at WARNING for every swallowed failure path.
- Move GLiNER spans and entity context out of the *system* prompt into a delimited data block in the user message, with per-label length caps — the current construction lets stored content steer future extractions.
- Run pronoun-merge before `_link_claims` resolves its id map (or re-resolve ids after merge) to kill the dangling-edge window.

### 5.9 smrti-town: reconnect the brain (C3)

Mechanical first step: fix every `r.content` → `r.atom.content` (etc.), delete the bare `except Exception: pass` wrappers, and add one integration test asserting `perceive()` returns non-empty memories after a `remember()`. Then, in order: wrap all Smrti calls in `asyncio.to_thread`; write TALK/action outcomes as episodes into speaker+listener spaces (without this the memory graph is decorative even after the fix); run a periodic `reflect()` per agent space; then either wire `lifecycle.py`/`culture.py`/`events.py` into the loop (fixing culture.py's two API mismatches) or delete them and rewrite the docs to describe the city-builder that actually exists.

### 5.10 CI

Add a `ci.yml` running `pytest` on push/PR to main (the model-cache step already exists in publish.yml to copy), plus a preset-JSON-vs-Python-dict equality test and a real (non-mocked) embedding-resolution test — three of the highest-impact regressions found here (L2 metric, preset drift, tier-3 death) would have been caught by exactly those tests.

---

## 6. Test-suite verification

`pytest tests/` — **see summary below** (run in a clean venv, Python 3.11, editable install with `[openai]` extra).

The suite is genuinely broad (36 files) but systematically avoids the failure modes above: vec-similarity tests patch identical vectors and raise thresholds; contradiction tests hand-create the `contradicts` edge the runtime never makes; bridge idempotency is only tested same-order; town tests don't assert memory contents; and no test runs two epochs concurrently or varies the reflect interval. The bugs in section 3 are invisible to it by construction.

---

## 7. Prioritized roadmap

| Priority | Work | Fixes |
|---|---|---|
| P0 | Vector layer v2 (cosine + space partition + content-hash re-embed + no relation vectors) | C1, H6, tier-3 death, KNN starvation |
| P0 | Fixed-point-safe valence propagation + protected valence floor | C2 — the engine's core promise |
| P0 | Proxy stream-relay hardening | C4 — silent memory loss in the flagship integration |
| P0 | smrti_town `RecallResult` fix + episode writes | C3 |
| P1 | Epoch serialization + wall-clock decay + one-shot contradictions + batched PLN evidence | H1, H2, H3, PLN claim |
| P1 | Security: header allowlist, gate `?db=`, bounded `_instances` | H8 |
| P1 | Executor discipline in all three server modes | H7 |
| P2 | Resolution compatibility matrix + threshold recalibration; commutative bridges; renormalized contextual similarity; personality floor in recall | H4, H5, H9 |
| P2 | Extraction parsing/validation/logging; prompt-injection hardening | medium cluster |
| P3 | Doc reconciliation (section 4), contradiction *detection*, per-PR CI, dedup windowing, confidence qualifier | remainder |

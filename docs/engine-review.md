# smrti engine review: proposal versus implementation

Reviewed at commit `3aecee5` (2026-09-03). Scope: the `smrti` package, the
benchmark harnesses, the `smrti-town` demo, and the claims made in `README.md`
and `CLAUDE.md`. Every defect listed under "Confirmed defects" was reproduced
by running the code; the reproduction script and its output are summarised in
the final section.

## 1. Verdict in brief

smrti is a well-built hybrid retrieval store (vector KNN + BM25 fused by RRF,
with an echo filter and a diversity cap) wrapped in vocabulary borrowed from
OpenCog's AtomSpace. The retrieval half is measured and honest. The half that
the README leads with, "what gets recalled and why is governed by graph
topology, PLN, attentional economics and emotional valence; similarity is one
signal among five", is not what the shipped engine does in any configuration a
user runs, and it is not what the published benchmark numbers measure:

- The benchmarks run with extraction off, zero consolidation epochs, and
  `min_confidence=0.0`. In that configuration STI is 0, LTI is 0 and confidence
  is a constant, so salience reduces to `0.35·similarity + constants` and the
  0.975 hit rate is the score of a vector+BM25 store. The README states the
  extraction half of this; it does not state the epoch half.
- Where the other signals do carry weight, they carry it in the wrong direction:
  a person concept with **zero** similarity to the query ranks first because
  recall injects the three highest-standing person atoms unconditionally and
  standing terms sum to 0.65 of the score (defect D5).
- Decay is a function of server uptime, not of time or of use. A memory loses
  98% of its attention and falls to the confidence floor within one hour of a
  running server, and loses nothing at all while the server is down (D4).
- Several documented mechanisms have no producer or no consumer: nothing ever
  creates a `contradicts` edge, so contradiction resolution never runs; nothing
  ever lowers a belief's probability, so `known_antipattern` ("a disproven
  belief") can only be asserted by the caller; the propagated mood is written
  every epoch and read by nothing that decides anything (D6, D9, section 4.4).
- `forget()` raises the STI of the memory it is forgetting and leaves it
  recallable (D1). The town demo's memory-weighted decisions never run because
  they read attributes `RecallResult` does not have, and the exception is
  swallowed (D10).

None of this makes the project unsound. It means the claims run ahead of the
engine, and the benchmark configuration hides the gap. Section 6 proposes how
to close it; most of the fixes are small.

## 2. The AtomSpace comparison

OpenCog's AtomSpace is a typed hypergraph knowledge store. What makes it what
it is:

| AtomSpace concept | What it is there | What smrti has |
| --- | --- | --- |
| Atoms: Nodes and Links | A hypergraph. Links are n-ary and can point at other Links, so a relation can be the subject of another relation. | Property graph. `relation` rows carry one `source_id` and one `target_id`. Binary edges only; nothing links to a link. |
| TruthValue (strength, confidence) | Confidence is `n/(n+k)` over evidence count `n`; revision sums counts. | `TruthValue.merge` implements exactly this with `k=1`, but it is used only when blending bridge atoms (`spaces/emergence.py:33`). The evidence path uses a different, ad hoc blend (`evolution/truth.py:7`). |
| PLN | A rule engine: deduction, induction, abduction, revision, over the hypergraph. | The revision rule only, and only for bridges. No inference of any kind; the closest thing is 1-hop candidate expansion at recall time. |
| ECAN (STI/LTI/VLTI) | An economy: a fixed STI fund, rent charged to atoms, wages paid for use by MindAgents, an attentional-focus boundary, Hebbian links, conservative importance diffusion. | Multiplicative decay (a proportional rent, fine), `+0.5` minted from nothing on every recall hit and every extraction re-mention, a 3.0 cap, and conservative 1-hop diffusion. No fund, no focus boundary, no VLTI. |
| Pattern matcher / query language | Declarative graph queries (BindLink, GetLink) that inference and agents run over the store. | KNN + BM25 entry points, 1-hop expansion, SQL. |

The name "AtomSpace-inspired" is honest at the vocabulary level: atoms, truth
values, STI/LTI and PLN revision are all here. The mechanisms that give those
words their meaning in OpenCog, hypergraph structure, inference over it, and an
attention economy with a budget, are not. A reader who knows AtomSpace will
expect more from the README's opening paragraph than the engine delivers; a
reader who does not will not miss it. The recommendation is to keep the
vocabulary and lower the claim (section 6.3).

## 3. Claim-by-claim audit

Claims are quoted or paraphrased from `README.md` and `CLAUDE.md`.

| # | Claim | Status | Evidence |
| --- | --- | --- | --- |
| C1 | "Similarity is one signal among five, not the ranking." | **Misleading as measured.** In the benchmarked configuration the other four are constants or zero. In the live configuration they outrank relevance (D5). | `bench/longmemeval/run.py` never calls `reflect()`; `retrieval/salience.py`; check 5. |
| C2 | "Truth values update via PLN revision" / "merged during epochs via PLN". | **False for the evidence path.** The epoch uses `update_truth`, a weighted moving average with `conf += w·(1−conf)`. The PLN formula exists but only blends bridge atoms. | `evolution/epoch.py` step 1 calls `update_truth`; `.merge(` has one caller. |
| C3 | "Bayesian truth values". | **Overstated.** No prior, likelihood or posterior anywhere; probability moves only by weighted averaging of asserted numbers. Beliefs cannot be lowered by anything the engine does (D9). | `evolution/truth.py`; check 9. |
| C4 | "Attentional economics (STI/LTI)". | **Partial.** Decay and conservative diffusion exist. Boosts mint STI; there is no budget and no focus set. STI is a "recalled-recently on a running server" flag, zero after an idle hour. | `core/atomspace.py::boost_sti`, `retrieval/fan_out.py:377`, check 4. |
| C5 | "Forgets what doesn't matter." | **Not for anything the user said.** User-authored episodes and beliefs are never pruned (`epoch.py:334`), decay only to the surfacing floor, and `forget()` leaves them recallable and boosts their attention (D1). Only concepts, goals and agent-authored atoms can leave the graph. | check 1. |
| C6 | "Old-but-critical errors outrank recent trivia." | **Holds narrowly.** With a stated valence of −0.9 an aged error scores 0.467 against fresh trivia at 0.435 on the balanced preset. The margin comes from the 0.5 LTI floor and the weight shift; with an estimated valence (which rarely exceeds ±0.5) it does not hold. Untested end to end. | `salience.py:36`; section 7 arithmetic. |
| C7 | "Never repeats a critical mistake." | **Depends entirely on the caller stating a valence.** The proxy never states one (by design), so a proxied conversation cannot mint a `critical_warning`. Reasonable, but the headline reads as if the engine detects mistakes. | `servers/proxy.py::_remember`; `retrieval/classify.py:34`. |
| C8 | `known_antipattern` = "a disproven belief". | **No path to disproof.** Evidence only ever carries 0.9 (mention) or the atom's own probability (reinforce, believe). Contradiction resolution lowers confidence, not probability, and never runs (D6). The only way to get p < 0.3 is to assert it. | check 9; `retrieval/classify.py:36`. |
| C9 | Epoch step 6, "resolve contradictions". | **Dead code.** No producer of `relation='contradicts'` in the engine; the extraction prompts never ask for one. | `grep -rn contradicts src/smrti` → the consumer only. |
| C10 | `min_confidence_to_surface`: "floor below which atoms are excluded from recall results". | **Not applied at recall.** `recall()` defaults to a hardcoded 0.1; the personality column is read only by the epoch's decay floor and prune predicate. The `deterministic` preset's 0.3 never gates a result (D3). | `__init__.py:221`; check 3. |
| C11 | `lti_promotion_threshold`: "cumulative STI required to increment LTI". | **Wrong description.** Promotion is `lti = max(lti, min(sti,1)·0.5)` when `sti > threshold`. Nothing is cumulative and nothing increments. | `epoch.py:270`. |
| C12 | Valence carries "tone and intensity". | **Intensity is redundant.** Every creation path sets `intensity = |valence|`; the estimator returns no intensity. Every `v < −0.5 and i > 0.5` conjunct is `v < −0.5`, and the salience valence term is `w_val·v²`. | `__init__.py:184,256`; check 2. |
| C13 | Intrinsic valence is "the tone the atom was written with, which propagation never touches". | **Only for atoms written through `AtomSpace.add_atom`.** Every extracted concept is inserted by the resolver with NULL intrinsic columns, so its judged tone is the drifting value. This is the exact failure the split was introduced to fix. | `extraction/resolve.py:243`; check 7. |
| C14 | Epoch step 7, "discover cross-domain connections". | **Misnamed.** Links atoms within cosine distance 0.4 (similarity ≥ 0.6) of a high-LTI atom. That is near-paraphrase linking, not cross-domain association. | `evolution/connections.py:63`. |
| C15 | Evidence is "an append-only observation log". | **Stores no observation.** `Evidence` has no text or source field; `believe(evidence="Team survey results")` discards the string and files a row whose `observed_probability` equals the probability just asserted. | check 8. |
| C16 | `remember` "subsumes `believe`". | **Two different atoms.** `remember(type="belief", p=0.95)` yields confidence 0.5 and no permanence; `believe(p=0.95)` yields confidence 0.95. The MCP handler routes to `believe`, the Python facade does not. | check 8. |
| C17 | Benchmarks measured with the `deterministic` preset. | **True but inert.** With zero epochs and `min_confidence=0.0`, no preset parameter except `w_similarity`, `w_confidence` and `agent_source_trust` can influence the result. | `bench/*/config.json`, `bench/longmemeval/run.py`. |
| C18 | smrti-town: "options with positively-valenced memories win … identical starting states diverge because memory diverges". | **Never executes.** `_place_valence`, `_person_valence` and the perception memory list read `r.valence`, `r.content`, `r.probability` on `RecallResult`, which has `atom`, `salience`, `similarity`. The `AttributeError` is caught and 0.0 returned. The town never calls `reflect()` either. | `smrti_town/agent.py:286,659,669`; D10. |
| C19 | CLAUDE.md: "one-time data repairs are recorded by name in an `applied_repairs` table". | **No such table.** The permanence repair moved into the epoch; the doc was not updated. | `grep -rn applied_repairs src/` → nothing. |
| C20 | Multi-tenant isolation, overlay reads are read-only. | **Holds.** Well pinned by `tests/test_tenancy_isolation.py`. | — |
| C21 | Hybrid recall: RRF fusion, echo damping, agent discount sparing similarity, diversity cap. | **Holds and is measured.** This is the part of the engine the benchmark actually exercises. | `retrieval/fan_out.py`, `retrieval/diversify.py`. |

## 4. Confirmed defects

Reproduced with a deterministic stand-in embedder (none of these depend on
embedding semantics). Script: `scratchpad/repro.py`, summarised in section 7.

### D1. `forget()` boosts what it forgets and does not forget it

`forget()` calls `self.recall()` (`__init__.py:302`), and `recall` applies
`sti_boost_on_access` to every result in the write space
(`retrieval/fan_out.py:377`). Forgetting an atom at STI 0 leaves it at STI 0.5.
Two forgets take it past the LTI promotion threshold. The confidence cut
(`× 0.3`) takes a `remember`-created episode from 0.5 to 0.15, which is above
the default surfacing floor of 0.1, so the memory is still returned by the next
recall. User-authored episodes and beliefs are exempt from pruning, so the atom
is permanent.

Observed: `sti 0.0 → 0.5`, `confidence 0.15`, still recalled at the default
floor.

### D2. `intensity` is a copy of `|valence|`

`remember`, `believe` and `_link_claims` all write `intensity = abs(valence)`;
`estimate_valence` returns a scalar. The second dimension of the emotional
model is never independently set, so every guard of the form
`valence < x and intensity > y` is `valence < x`, and the salience term
`w_val·|valence|·intensity` is `w_val·valence²`.

### D3. `min_confidence_to_surface` is not honoured at recall

`Smrti.recall`, the MCP tool, the REST model and the proxy all carry their own
literal floors (0.1, 0.1, 0.1, 0.3). `retrieve()` never reads the personality
column. With the `deterministic` preset (floor 0.3), an atom at confidence 0.15
surfaces.

### D4. Decay is a function of server uptime

`reflect_loop.py` runs an epoch every 60 s per live instance. Every epoch
multiplies STI by 0.9, confidence by 0.98, LTI by 0.99, unconditionally. Trace
for one user episode with STI 1.0 on the balanced preset:

| epochs | wall clock at default interval | STI | confidence | LTI |
| --- | --- | --- | --- | --- |
| 1 | 1 min | 0.900 | 0.490 | 0.450 (promoted) |
| 10 | 10 min | 0.349 | 0.409 | 0.411 |
| 60 | 1 h | 0.002 | 0.149 | 0.249 |
| 120 | 2 h | 0.000 | 0.100 (floor) | 0.136 |
| 480 | 8 h | 0.000 | 0.100 | 0.100 (floor) |

Consequences: (a) after an hour of a running server every memory has the same
STI and confidence, so the salience formula degenerates to similarity plus the
LTI term; (b) a database that is not being served does not age at all, so a
week-old memory opened today is "fresher" than a one-hour-old memory on a
server that stayed up; (c) on the proxy, instances are LRU-evicted at 64
(`proxy.py:51`), so a tenant/space pair that falls out of the LRU stops
consolidating while its neighbours continue. The personality's decay rates are
therefore tuned against a clock nobody chose. This is the single largest gap
between "salience over recency" and what runs.

### D5. Standing outranks relevance; the top person atoms always win

`retrieve()` adds the three highest-`sti+lti` person atoms to every candidate
set regardless of query (`fan_out.py:283`). They are then scored on their true
similarity, which for an unrelated query is ~0. On the balanced preset a person
concept with saturated STI (3.0, the cap every extraction re-mention pushes
toward), confidence 0.9 and LTI 0.8 scores 0.470 with similarity 0. A perfectly
relevant fresh episode (similarity 1.0, STI 0.5, confidence 0.5) scores 0.512;
at similarity 0.9 it scores 0.478; anything aged an hour scores at most 0.375.
Observed with a stand-in embedder: query "kubernetes ingress host header",
result 1 is `concept Alice` at similarity 0.000, salience 0.470, above eight
relevant episodes at 0.23. In the proxy this costs one of the five injected
slots on every turn (the `_enrich_content` helper exists to make that slot
readable, which suggests it was noticed). The underlying cause is structural:
relevance has no gating role, it is one additive term at weight 0.35, so any
atom whose standing terms sum past ~0.35 can outrank a relevant one it has
nothing to do with.

### D6. Contradiction resolution has no producer

Epoch step 6 selects `relation = 'contradicts'` (`epoch.py:278`). Nothing in
the engine writes that relation; the extraction prompts do not request it. The
step, its tests and its README line describe behaviour that cannot occur
unless an LLM happens to emit the predicate. HaluMem's 71% hallucination on
"dynamic update" questions is the user-visible face of this: there is no
supersession mechanism, so the older and newer versions of a fact are recalled
side by side with equal standing.

### D7. Extracted atoms have NULL intrinsic valence

`EntityResolver._create_atom` (`resolve.py:243`) inserts directly into `atoms`
without the `intrinsic_*` columns. `ATOM_OWN_VALENCE` falls back to the drifting
value for such rows, so every concept the pipeline creates absorbs the mood of
its neighbours and is judged on it. Observed: a resolver-created "Postgres"
concept linked to one −0.9 episode reads own valence −0.30 after 20 epochs.
`_link_claims` (`extract.py:490,506`) also writes negative claim valence into
the drifting column of the target and the episode "so it classifies as
critical_warning"; since the stated-valence gate was added that write can no
longer produce a `critical_warning`, and the comment is stale.

### D8. Two initial-confidence regimes for the same kind of atom

`remember(type="belief")` creates confidence 0.5 with no permanence handling;
`believe()` creates 0.3, or `probability` when ≥ 0.95, and files an evidence
row. The REST and MCP paths route `type=belief` to `believe()`; the Python
facade does not. Initial confidence across creation paths is: episode 0.5,
belief 0.3, extracted concept 0.6 (user) / 0.3 (agent), relation 0.5, healing
edge 0.2, discovered association 0.1. Since `w_confidence` is 0.20 to 0.40 of
the salience score, an extracted concept node outranks the user's own belief on
the confidence term by construction.

### D9. Nothing can lower a belief's probability

Every evidence row the engine writes carries either 0.9 (`_MENTION_PROBABILITY`)
or the atom's current probability (`believe`, `reinforce`). `update_truth`
therefore only ever pulls probability toward 0.9 or holds it. Two conflicting
beliefs stored side by side stay at their asserted probabilities through any
number of epochs. "Disproven" is not a state the engine can reach.

### D10. smrti-town's memory-driven decisions never run

`agent.py:286-288`, `659`, `669` read `r.content`, `r.valence`, `r.probability`
from `RecallResult`, which has neither. Each site is wrapped in
`try/except Exception`, so perception memories are always `[]` and place/person
valence is always 0.0. The README's central claim for the demo (memory shapes
behaviour, identical starts diverge) is not exercised. The town also never
calls `reflect()`, so no citizen memory consolidates.

### D11. The evidence string is discarded

`believe(statement, probability, evidence="…")` uses `evidence` as a boolean.
The `Evidence` model has no text column; `source_episode_id` is set only by the
resolver's mention rows. The "append-only observation log" records that
something was observed, never what.

## 5. Design misalignments beyond the defects

### 5.1 The benchmark measures a different engine than the one shipped

`bench/longmemeval/run.py` and `bench/halumem/run.py` construct a `Smrti`,
call `remember` per turn, and call `recall` once. No `reflect()`, no
extraction, `min_confidence=0.0`, `top_k=50`. In that state every atom has STI
0 (except the ones the single recall just boosted), LTI 0, confidence 0.5, and a
small squared estimated-valence term. The five-signal formula collapses to
`0.35·sim + 0.15` for user turns and `0.35·sim + 0.045` for assistant turns.
The baseline history shows the hit rate moving 0.55 → 0.90 when `top_k` went
10 → 50 (commit `f6002dc`) and 0.90 → 0.975 with the ranking fix. Both are
retrieval changes. Nothing has ever measured whether an epoch helps or hurts,
and D4/D5 together predict that on a live graph it hurts: standing accrues to
hubs, relevance does not gate, and the hubs are what surfaces.

### 5.2 Truth values on atoms that have no truth

`probability` and `confidence` are stored on every atom type. For an episode
("user asked about deployment") probability is a default of 0.8 (0.75 in the
proxy) that never changes; for a concept ("Alice") it is 0.8 pulled toward 0.9
by mentions. Only beliefs and relations carry a proposition. Yet confidence
weighs 0.20 to 0.40 of every recall score, so a number with no meaning for
most atoms is a fifth to two-fifths of the ranking. The mention-evidence path
also turns belief probability into a mention-frequency statistic.

### 5.3 Two truth-update formulas

`TruthValue.merge` (count-based, PLN revision with `k=1`) and `update_truth`
(`p' = (p·c + e·w)/(c + w)`, `c' = c + w(1−c)`) disagree on what confidence is.
Under the first, confidence 0.5 means one unit of evidence; under the second,
adding weight `w` of evidence to confidence `c` yields `c + w(1−c)`, which
exceeds the PLN value `(c + w(1−c)) / (1 + w(1−c))` for every `w > 0`. The epoch
uses the second; bridges use the first. One formula should serve both, and the
count-based one is the one the documentation describes.

### 5.4 Mood propagation has no consumer

`propagate_valence` moves the drifting `valence`/`intensity` pair each epoch,
controlled by `valence_propagation` and `mood_inertia`. After the intrinsic
split, every judgement (salience, severity, the confidence bypass, the prune
guard) reads the intrinsic pair. The drifting pair is read by: the next
propagation, the bridge-atom blend, and the `valence` field of the MCP/proxy
recall response. Two of the seventeen personality hyperparameters therefore
change nothing a user can observe except a number in an API response and which
atoms propagate next epoch. (For resolver-created atoms it does matter, which
is D7, not a feature.)

### 5.5 Automatic bridge materialisation

Every tenth epoch each space computes `space_overlap` against every other
non-bridge space in the tenant (`epoch.py:372`): `O(500²)` cosine pairs in pure
Python per space pair, plus one neighbourhood embedding per surviving candidate,
and it writes atoms into `{a}_x_{b}` spaces that no caller requested. On the
proxy every live `(tenant, space)` runs its own epoch, so a tenant with `S`
spaces does `S·(S−1)` overlap computations every ten minutes and materialises
bridges between every pair whose content overlaps, which for the README's own
team example (shared reads) is every pair. This is a side-effecting write
triggered by a background timer at a threshold (`min_jaccard=0.1`) the user did
not set. It should be an explicit call.

### 5.6 Rich-get-richer attention

Recall adds 0.5 STI to each returned atom in the write space; extraction adds
0.5 per re-mention; STI caps at 3.0; salience rewards STI. An atom that is
recalled is more likely to be recalled. ECAN counters this with rent against a
fixed fund so that hoarding is impossible; here the only counter is
multiplicative decay, which on a running server empties everyone's account at
the same rate (D4) and on an idle one never charges. Together with D5 the
outcome is that hubs (persons, the most-mentioned concepts) saturate and pin
the top of every recall.

### 5.7 Healing builds hubs

With one person atom in a space, every orphaned episode is linked to that
person and a low-confidence `associated` edge is created from the person to
every concept the episode mentions. The person becomes a hub with an edge to
everything, which then feeds the 100-per-direction expansion and D5. With
several persons, attribution is by cosine ≥ 0.3 between an episode embedding
and a *name* embedding, which on a MiniLM paraphrase model is close to chance.

### 5.8 Two temporal mechanisms, two storage locations

Relative dates are resolved twice: at write time by GLiNER + dateparser,
appended into the text (`temporal.py:123`), and again by the extraction LLM
into `metadata.$.temporal` (`extract.py:180`). Recall renders the second; the
first is embedded. Either alone would do; both together mean a memory can carry
two resolutions of the same span that need not agree.

## 6. Over-engineering flags

- **Five set operations, a three-signal contextual similarity with hand-tuned
  weights, bridge emergence with PLN-merged truth, and automatic epoch-time
  bridging**, in service of two MCP tools and one REST route. The set algebra
  is `O(|A|·|B|)` pure-Python cosine over 500×500 atom slices; the "homonym
  disambiguation" it exists for (Java the island vs Java the language) is a
  scenario with no test on real data. Keep `space_overlap` and an explicit
  `space_merge`; drop the automatic step (5.5) and the union / symmetric
  difference until someone needs them.
- **TruthValue on every atom type** (5.2). Beliefs and relations need one;
  episodes and concepts do not, and carrying one makes the ranking depend on a
  meaningless constant.
- **Seventeen personality hyperparameters** of which, in the steady state D4
  reaches within an hour, only `w_similarity`, `w_confidence` (as a constant),
  `w_lti` (at the floor) and `agent_source_trust` affect recall, and two
  (`valence_propagation`, `mood_inertia`) affect nothing observable (5.4). The
  "same history, different memories" claim rests on decay rates that no
  benchmark exercises.
- **Valence as two dimensions** (D2). One dimension is stored twice.
- **The mood propagation subsystem** (5.4) and the **contradiction resolution
  step** (D6) are complete features with tests and documentation and no
  producer or no consumer.
- **Two temporal pipelines** (5.8).
- **The `applied_repairs` table** exists in the documentation only (C19); the
  permanence lift was folded into the epoch, which is the better design, and
  the doc should say so.

Things that look heavy but are justified: the stable-rowid keying of the two
virtual tables (it solved a measured `O(N)` write), the per-loop lock and HTTP
client registries in `extract.py`, the CASE-guarded JSON expressions in
`provenance.py`, and the migration-with-backup path in `db.py`.

## 7. Recommendations

Ordered by how much each closes the gap between claim and behaviour.

### P0: make the engine do what the README says

1. **Decouple decay from server uptime (D4).** Store `last_accessed_at`, and
   compute decay as a function of elapsed real time: either lazily at read time
   (`sti·exp(−λ·Δt)`) or in the epoch as `rate^(Δt / unit)` using `updated_at`.
   Re-tune the six decay parameters as half-lives in hours, which is a unit a
   user can reason about. Alternatively make epochs event-driven (every N writes
   or recalls per space) so that "an epoch" means "some amount of use". Either
   fixes (a), (b) and (c) under D4.
2. **Make relevance a gate, not a term (D5, 5.6).** Two cheap options: drop the
   unconditional person injection from `retrieve()` (return anchors as a
   separate `entities` field if the proxy wants them), and multiply the standing
   terms by a relevance factor (for example `salience = sim · (1 + standing)`,
   or require `similarity ≥ τ` for anything that did not enter via KNN or
   BM25). Re-run the LongMemEval harness *with* epochs to confirm the standing
   terms are no longer able to evict evidence.
3. **Fix `forget()` (D1).** Retrieve without the STI boost (add a flag to
   `retrieve`, or reuse the candidate pool directly), sink confidence to below
   the surfacing floor deterministically, exclude `$.forgotten` atoms from
   recall, and let the pruner remove stamped user atoms. "Forget" should mean
   the memory stops surfacing.
4. **Honour `min_confidence_to_surface` (D3).** Make it the default floor in
   `retrieve()` when the caller passes none; let the server env var override it
   rather than replace it.
5. **Centralise atom creation (D7, D8).** Route the resolver, healing and
   connection discovery through `AtomSpace.add_atom` (or a thin
   `insert_atom_row` that always writes the intrinsic columns), and give each
   atom type one initial confidence. Make `remember(type="belief")` call
   `believe`, or remove `belief` from `remember`'s accepted types.
6. **Fix smrti-town (D10).** `r.atom.content`, `r.atom.valence.own`,
   `r.atom.truth.probability`; add a test that `_place_valence` returns a
   non-zero value for a negative memory; call `reflect()` from the tick loop if
   consolidation is meant to shape citizens.

### P1: give the documented mechanisms a producer and a consumer

7. **Contradiction and supersession (D6, D9).** Ask the claims prompt for
   `supersedes`/`contradicts` when the same subject and predicate appear with a
   new object, or detect it structurally in `_link_claims` (same
   `(source_id, predicate)`, different `target_id`, later `created_at`) and file
   negative evidence (`observed_probability` low) against the older claim.
   Either gives probability a way down, makes `known_antipattern` reachable, and
   is the missing lever behind HaluMem's dynamic-update failures.
8. **One truth formula (5.3).** Use the count-based revision everywhere:
   store `n = c/(1−c)` implicitly, add evidence weight to `n`, recompute `c`.
   Then "Bayesian" can be dropped from the docs and "PLN revision" becomes true.
9. **Store evidence content (D11).** Add `text` and `source` columns to
   `evidence`; `believe(evidence=…)` should record the string, and the
   resolver's mention rows should keep `source_episode_id` (they do) so a
   belief's provenance can be listed.
10. **Restrict truth values to atoms that carry a proposition (5.2)**, or at
    least stop weighting confidence for episodes and concepts in salience.
11. **Make bridging explicit (5.5).** Remove `_discover_bridges` from the epoch;
    keep `space_merge`.
12. **Rename or redefine connection discovery (C14).** If the intent is
    cross-domain association, require the two atoms to differ in entity type or
    in graph neighbourhood; if the intent is paraphrase linking, name it that.
13. **Collapse intensity (D2)** or give it an independent source. Until then
    remove the redundant conjuncts so the thresholds read as what they are.
14. **One temporal path (5.8).** The write-time NER path is deterministic and
    cheap; the LLM path covers idioms. Pick one storage location and have the
    other feed it.

### P2: measure and document the engine that exists

15. **Benchmark the shipped configuration.** Add harness modes that (a) run
    `reflect()` N times after ingest, (b) enable extraction, (c) use the proxy's
    `top_k=5` and the MCP default 10. Publish each row. If the epoch or the graph
    lowers hit rate, that is the finding the README should carry until fixed.
16. **Add scenario tests for the headline claims.** An aged critical error
    versus fresh trivia through `recall`, not `compute_salience`; a forgotten
    memory not surfacing; a person hub not outranking a relevant episode; the
    town's `_place_valence` returning a non-zero value. Replace the tautological
    `test_min_confidence_filter` (`len == 0 or all(...)` passes on an empty
    result).
17. **Correct the documentation**: "cumulative STI" (C11), the surfacing floor
    (C10), "PLN revision on evidence" (C2), "disproven" (C8), `applied_repairs`
    (C19), and the README opening paragraph. A version that matches the code:
    *"Memories are graph nodes with truth values, attention weights and
    valence. Retrieval fuses vector and lexical search, expands one hop through
    extracted relations, and ranks by a personality-weighted blend of relevance,
    attention, confidence and emotional tone."* That is still a strong pitch and
    every word of it is true today.

## 8. Method and what was not verified

- Static reading of every module under `src/smrti`, the benchmark harnesses,
  the town's `agent.py`, `README.md`, `CLAUDE.md`, and the test names.
- Empirical checks (`repro.py`) ran against the installed package with a
  deterministic hash-based stand-in for the embedding provider, because the
  review environment's proxy denies Hugging Face downloads. Every check probes
  bookkeeping, not similarity quality, so the substitution does not affect the
  conclusions. Checks and observed results:

  | check | observed |
  | --- | --- |
  | 1 forget boosts STI | STI 0.0 → 0.5, confidence 0.15, still recalled |
  | 2 intensity = \|valence\| | four of four atoms |
  | 3 preset floor ignored | deterministic floor 0.3, atom at 0.15 surfaced |
  | 4 decay trace over 480 epochs | table in D4 |
  | 5 person injection outranks relevance | similarity 0.000 ranked first at 0.470 |
  | 6 no `contradicts` producer | one consumer, zero producers |
  | 7 resolver atoms have NULL intrinsic valence | drifted to −0.30 in 20 epochs |
  | 8 `remember(type=belief)` vs `believe` | confidence 0.5 vs 0.95; evidence text absent |
  | 9 probability cannot fall | 0.8 after 15 epochs beside a conflicting 0.9 belief |
  | town attribute access | `AttributeError` on `content`, `valence`, `probability` |

- The full pytest suite was **not** run end to end here: the real embedding
  model could not be downloaded. Of the thirteen test modules tried, 100 tests
  that need no model passed; the remaining 87 failed or errored, every one of
  them on `httpx.ProxyError` from the denied download or on the fixture cleanup
  that cascades from it. CI runs the full suite with the real model; nothing in
  this review contradicts a passing suite, because none of the defects above
  have a test that would fail on them.
- The LongMemEval and HaluMem harnesses were not re-run (datasets and a judge
  model are required). Statements about what they measure come from reading the
  harness code and the baseline history in git.

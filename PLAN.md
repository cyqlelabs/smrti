# Engram: AtomSpace-Inspired Memory + Personality Engine for AI Agents

## Development Plan

**Project:** `engram` — *Living memory that thinks, feels, and remembers*
**PyPI:** `engram-memory` | **npm:** `engram-memory` | **GitHub:** `engram`

---

## 1. Vision & Problem Statement

### The Problem
Current AI agent memory systems rely on vector similarity search (cosine distance over embeddings), which has fundamental failure modes:

- **Negation trap:** "user loves X" and "user hates X" are embedding-neighbors
- **Temporal blindness:** "Alice was CEO" vs "Alice is CEO" are indistinguishable
- **Multi-hop failure:** relational queries ("does my wife like that restaurant?") require graph traversal, not chunk retrieval
- **Contextual conflation:** same words, different intent — no way to disambiguate
- **No epistemic state:** all facts treated as equally true, no confidence tracking

### The Insight (from OpenCog AtomSpace)
Memory is not a bag of facts — it's a **living epistemic structure** where:
- Beliefs strengthen/weaken with evidence (TruthValues)
- Attention flows toward what matters (AttentionValues / ECAN)
- New connections form between distant concepts (emergent associations)
- Emotional coloring shapes recall and response (valence)
- Personality crystallizes from patterns of attention + belief + emotional style

### The Solution
A portable, single-file memory engine that any AI agent can plug into via MCP, REST, or direct Python import. Built on SQLite + sqlite-vec. Ships as `pip install engram-memory`.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                   AGENT (Claude, Gemini, etc.)       │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────┐  │
│  │ MCP      │  │ REST     │  │ Python Library    │  │
│  │ (stdio)  │  │ (HTTP)   │  │ (direct import)   │  │
│  └────┬─────┘  └────┬─────┘  └────────┬──────────┘  │
│       │              │                 │              │
│       └──────────────┼─────────────────┘              │
│                      │                                │
│              ┌───────▼────────┐                       │
│              │  Engram Core   │                       │
│              │  (Python)      │                       │
│              └───────┬────────┘                       │
│                      │                                │
│       ┌──────────────┼──────────────┐                 │
│       │              │              │                 │
│  ┌────▼─────┐  ┌─────▼────┐  ┌─────▼──────┐         │
│  │ AtomSpace│  │ Salience │  │ Consolid.  │         │
│  │ (graph)  │  │ Retrieval│  │ Epoch      │         │
│  └────┬─────┘  └─────┬────┘  └─────┬──────┘         │
│       │              │              │                 │
│       └──────────────┼──────────────┘                 │
│                      │                                │
│              ┌───────▼────────┐                       │
│              │ SQLite + vec0  │ ← single .db file     │
│              │ (WAL mode)     │                       │
│              └────────────────┘                       │
└─────────────────────────────────────────────────────┘
```

### Key Architectural Decisions

| Decision | Choice | Rationale |
|---|---|---|
| **Storage** | SQLite + sqlite-vec | Single-file, zero-config, portable. Ships anywhere SQLite runs (Linux, macOS, Windows, WASM) |
| **Vector search** | sqlite-vec `vec0` virtual tables | KNN via `MATCH`, metadata filtering, auxiliary columns. No separate vector DB needed |
| **Embeddings** | FastEmbed (ONNX Runtime) as default, OpenAI/Anthropic API as optional | No PyTorch dependency (~50MB vs ~2GB). CPU inference, sub-ms latency. BAAI/bge-small-en-v1.5 default |
| **Graph traversal** | SQLite recursive CTEs | Sufficient for 1-3 hop traversals. Agents can't digest deeper context anyway |
| **Concurrency** | WAL mode + aiosqlite + single-writer queue | Multiple async readers, serialized writes. sqlite-vec loaded per connection |
| **Consolidation** | asyncio.to_thread background loop | Single-process, no message queue. Runs every N minutes. Deterministic, replayable |
| **Distribution** | PyPI (primary) + npm wrapper + Docker | `pip install engram-memory` is the golden path |

---

## 3. Data Model

### 3.1 SQLite Schema

```sql
-- ============================================================
-- ATOMS: The unified node/link table (inspired by AtomSpace)
-- ============================================================
CREATE TABLE atoms (
    id          TEXT PRIMARY KEY,    -- UUID
    type        TEXT NOT NULL,       -- 'concept', 'belief', 'episode', 'goal', 'relation'
    label       TEXT NOT NULL,       -- canonical name
    content     TEXT,                -- full text content (for episodes)

    -- TruthValue (from AtomSpace)
    probability REAL DEFAULT 0.5,   -- how true is this? (0.0-1.0)
    confidence  REAL DEFAULT 0.0,   -- how much evidence? (0.0-1.0)

    -- AttentionValue (simplified ECAN)
    sti         REAL DEFAULT 0.0,   -- Short-Term Importance (decays per epoch)
    lti         REAL DEFAULT 0.0,   -- Long-Term Importance (persistent salience)

    -- Emotional Valence
    valence     REAL DEFAULT 0.0,   -- -1.0 (negative) to 1.0 (positive)
    intensity   REAL DEFAULT 0.0,   -- 0.0 (neutral) to 1.0 (strong)

    -- For relation atoms (edges as first-class citizens)
    source_id   TEXT REFERENCES atoms(id),
    target_id   TEXT REFERENCES atoms(id),
    relation    TEXT,                -- 'supports', 'contradicts', 'associated', 'causes', 'part_of'

    -- Metadata
    agent_id    TEXT NOT NULL,       -- multi-agent support
    created_at  TEXT DEFAULT (datetime('now')),
    updated_at  TEXT DEFAULT (datetime('now')),
    metadata    TEXT DEFAULT '{}',   -- JSON for flexible properties

    -- Entity type constraint
    entity_type TEXT                 -- rigid upper ontology: 'person', 'project', 'tool',
                                    -- 'preference', 'constraint', 'location', 'organization'
);

CREATE INDEX idx_atoms_type ON atoms(type);
CREATE INDEX idx_atoms_agent ON atoms(agent_id);
CREATE INDEX idx_atoms_entity_type ON atoms(entity_type);
CREATE INDEX idx_atoms_source ON atoms(source_id);
CREATE INDEX idx_atoms_target ON atoms(target_id);
CREATE INDEX idx_atoms_label ON atoms(label);
CREATE INDEX idx_atoms_sti ON atoms(sti DESC);

-- ============================================================
-- VECTOR INDEX: sqlite-vec virtual table for semantic search
-- ============================================================
CREATE VIRTUAL TABLE vec_atoms USING vec0(
    atom_id     TEXT,                       -- FK to atoms.id
    embedding   float[384],                 -- BAAI/bge-small-en-v1.5 = 384 dims
    agent_id    TEXT partition key,          -- partition by agent
    +label      TEXT                         -- auxiliary: returned with results
);

-- ============================================================
-- EVIDENCE LEDGER: Append-only truth maintenance log
-- ============================================================
CREATE TABLE evidence (
    id                  TEXT PRIMARY KEY,
    atom_id             TEXT NOT NULL REFERENCES atoms(id),
    observed_probability REAL NOT NULL,
    weight              REAL DEFAULT 1.0,    -- reliability of observation
    source_episode_id   TEXT,                -- which episode produced this
    agent_id            TEXT NOT NULL,
    created_at          TEXT DEFAULT (datetime('now')),
    processed           INTEGER DEFAULT 0    -- 0=pending, 1=applied
);

CREATE INDEX idx_evidence_pending ON evidence(processed, agent_id);
CREATE INDEX idx_evidence_atom ON evidence(atom_id);

-- ============================================================
-- PERSONALITY: Per-agent hyperparameters
-- ============================================================
CREATE TABLE personality (
    agent_id                TEXT PRIMARY KEY,
    -- Belief dynamics
    confidence_decay_rate   REAL DEFAULT 0.02,   -- per epoch
    confidence_update_lr    REAL DEFAULT 0.3,     -- learning rate for new evidence
    min_confidence_to_surface REAL DEFAULT 0.1,   -- below this = effectively forgotten

    -- Attention dynamics
    sti_decay_rate          REAL DEFAULT 0.1,     -- per epoch (0.0=perfect memory, 1.0=goldfish)
    sti_boost_on_access     REAL DEFAULT 0.5,     -- how much STI increases when referenced
    sti_propagation_factor  REAL DEFAULT 0.15,    -- fraction passed to 1-hop neighbors
    lti_promotion_threshold REAL DEFAULT 0.7,     -- STI above this promotes to LTI

    -- Emotional dynamics
    valence_weight          REAL DEFAULT 0.2,     -- weight of valence in salience scoring
    valence_propagation     REAL DEFAULT 0.1,     -- emotional bleed to connected atoms
    mood_inertia            REAL DEFAULT 0.8,     -- how slowly agent mood shifts (0=reactive, 1=stoic)

    -- Salience weights (for retrieval scoring)
    w_similarity            REAL DEFAULT 0.35,    -- vector cosine similarity
    w_sti                   REAL DEFAULT 0.25,    -- short-term importance
    w_confidence            REAL DEFAULT 0.20,    -- truth confidence
    w_lti                   REAL DEFAULT 0.10,    -- long-term importance
    w_valence               REAL DEFAULT 0.10,    -- emotional resonance

    -- Meta
    preset_name             TEXT DEFAULT 'balanced',
    created_at              TEXT DEFAULT (datetime('now'))
);

-- ============================================================
-- PRAGMAS: Performance configuration
-- ============================================================
-- Applied at connection time, not stored in schema:
-- PRAGMA journal_mode=WAL;
-- PRAGMA synchronous=NORMAL;
-- PRAGMA busy_timeout=5000;
-- PRAGMA foreign_keys=ON;
-- PRAGMA cache_size=-64000;  -- 64MB cache
```

### 3.2 Entity Type System (Rigid Upper Ontology)

The LLM does NOT invent types. These are hardcoded:

| Entity Type | Description | Example |
|---|---|---|
| `person` | People the agent interacts with or knows about | "Alice", "the user's boss Dave" |
| `organization` | Companies, teams, groups | "Acme Corp", "the frontend team" |
| `project` | Work items, initiatives, repos | "Project Alpha", "the auth rewrite" |
| `tool` | Technologies, languages, frameworks | "Python", "React", "Kubernetes" |
| `preference` | User likes/dislikes/habits | "prefers dark mode", "hates meetings" |
| `constraint` | Rules, deadlines, limitations | "merge freeze after March 5th" |
| `location` | Places | "Seattle", "the downtown office" |
| `event` | Things that happened | "the deploy failure on Tuesday" |
| `concept` | Abstract ideas, domains | "microservices", "data privacy" |
| `goal` | Objectives, desired outcomes | "ship v2 by Q3", "learn Rust" |

Anything that doesn't fit goes into `concept` with descriptive `metadata` JSON.

---

## 4. Core Modules

### 4.1 Project Structure

```
engram/
├── pyproject.toml
├── LICENSE                     # MIT
├── src/
│   └── engram/
│       ├── __init__.py         # Public API: AtomSpace, Atom, retrieve, remember
│       ├── __main__.py         # `python -m engram serve`
│       ├── cli.py              # Typer CLI: serve mcp|rest, init, status, export
│       │
│       ├── core/
│       │   ├── db.py           # SQLite connection manager (WAL, sqlite-vec loading)
│       │   ├── models.py       # Pydantic: Atom, TruthValue, AttentionValue, Evidence
│       │   ├── atomspace.py    # Graph operations: add, link, traverse, query
│       │   └── embed.py        # EmbeddingProvider (FastEmbed default, OpenAI optional)
│       │
│       ├── extraction/
│       │   ├── resolve.py      # Entity resolution: exact → fuzzy → embedding → create
│       │   ├── prompts.py      # Extraction prompt templates for host LLM piggyback
│       │   ├── aliases.py      # Alias table management (Aho-Corasick for known entities)
│       │   └── gliner.py       # [v1.0] GLiNER zero-shot NER for offline/bulk extraction
│       │
│       ├── retrieval/
│       │   ├── salience.py     # Salience scoring formula (personality-weighted)
│       │   └── fan_out.py      # Entity-first retrieval: extract → match → expand → rank
│       │
│       ├── evolution/
│       │   ├── epoch.py        # Consolidation epoch: truth update, STI decay, LTI promote
│       │   ├── truth.py        # Bayesian truth maintenance (evidence → probability update)
│       │   ├── attention.py    # STI/LTI decay and propagation formulas
│       │   ├── valence.py      # Emotional propagation
│       │   └── connections.py  # Cross-domain association discovery ("charisma" engine)
│       │
│       ├── personality/
│       │   ├── params.py       # PersonalityProfile dataclass + presets
│       │   └── presets.py      # balanced, analytical, curious, empathetic, maverick
│       │
│       └── servers/
│           ├── mcp.py          # MCP server (stdio transport, tool definitions)
│           ├── rest.py         # FastAPI server (OpenAPI spec → works with Gemini/OpenAI)
│           └── tools.py        # Shared tool definitions (remember, recall, reflect, forget)
│
├── tests/
│   ├── test_atomspace.py
│   ├── test_extraction.py
│   ├── test_resolution.py
│   ├── test_retrieval.py
│   ├── test_evolution.py
│   ├── test_personality.py
│   └── test_mcp.py
│
└── presets/
    ├── balanced.json
    ├── analytical.json
    ├── curious.json
    ├── empathetic.json
    └── maverick.json
```

### 4.2 Pydantic Models

```python
from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional
import uuid

class AtomType(str, Enum):
    CONCEPT = "concept"
    BELIEF = "belief"
    EPISODE = "episode"
    GOAL = "goal"
    RELATION = "relation"

class EntityType(str, Enum):
    PERSON = "person"
    ORGANIZATION = "organization"
    PROJECT = "project"
    TOOL = "tool"
    PREFERENCE = "preference"
    CONSTRAINT = "constraint"
    LOCATION = "location"
    EVENT = "event"
    CONCEPT = "concept"
    GOAL = "goal"

class TruthValue(BaseModel):
    probability: float = Field(0.5, ge=0.0, le=1.0)
    confidence: float = Field(0.0, ge=0.0, le=1.0)

class AttentionValue(BaseModel):
    sti: float = Field(0.0, ge=0.0)    # Short-Term Importance
    lti: float = Field(0.0, ge=0.0)    # Long-Term Importance

class Valence(BaseModel):
    valence: float = Field(0.0, ge=-1.0, le=1.0)   # negative to positive
    intensity: float = Field(0.0, ge=0.0, le=1.0)   # strength

class Atom(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: AtomType
    label: str
    content: Optional[str] = None
    truth: TruthValue = Field(default_factory=TruthValue)
    attention: AttentionValue = Field(default_factory=AttentionValue)
    valence: Valence = Field(default_factory=Valence)
    entity_type: Optional[EntityType] = None
    # For relations
    source_id: Optional[str] = None
    target_id: Optional[str] = None
    relation: Optional[str] = None
    agent_id: str = "default"
    metadata: dict = Field(default_factory=dict)
```

---

## 5. Entity Extraction: The Piggyback Pattern

### 5.1 Design Philosophy: Don't Own the LLM — Borrow It

The host agent (Claude, GPT, Gemini) is **already processing the user's message**. Making it extract entities as a side-effect via tool_use costs zero additional infrastructure. Bundling a local LLM (~1-3GB RAM) inside a pip-installable SQLite utility is bloatware.

| Approach | Latency | RAM | Accuracy (messy text) | Offline | Verdict |
|---|---|---|---|---|---|
| spaCy NER | 10-20ms | 50MB | Poor on informal text | Yes | Dead end for conversational AI |
| GLiNER (zero-shot) | ~50ms | ~300MB | Good for entity spans | Yes | Great sieve, can't resolve pronouns |
| Local LLM (1.5B) | 150-250ms | 1-3GB | Excellent | Yes | Too heavy for a pip utility |
| **Host LLM piggyback** | **0ms extra** | **0** | **Excellent** | No | **v0.1 winner** |
| **Tiered (v1.0)** | 1-50ms | ~300MB | Excellent | Yes | **v1.0 winner** |

### 5.2 v0.1 Pipeline: Host LLM Piggyback

The host LLM calls `engram_remember` and `engram_believe` — extraction happens *inside* these tool calls. The LLM does the hard work (pronoun resolution, typo correction, claim extraction) because it's already loaded and processing the conversation.

```
User message arrives
        │
        ▼
Host LLM processes message (normal agent flow)
        │
        ├──→ Generates response to user
        │
        └──→ Calls engram_remember(content="User prefers Rust over Go")
                    │
                    ▼
            Engram ingestion pipeline:
            [1] Parse the content string
            [2] Resolve each entity against existing atoms:
                ├── Exact match on label/alias      (< 1ms)
                ├── RapidFuzz against same type      (< 5ms)
                └── Embedding cosine similarity      (< 20ms)
            [3] Upsert atoms + evidence
            [4] Embed async for future vector search
```

### 5.3 Extraction Prompt (Injected into Host LLM)

This prompt is embedded in the MCP tool descriptions so the host LLM naturally extracts structured knowledge when calling Engram tools:

```
When the user shares information worth remembering, call engram_remember.

RULES:
- Resolve ALL pronouns to explicit names ("he" → "Dave", "that project" → "Project Alpha")
- Correct obvious typos ("pythn" → "Python", "k8s" → "Kubernetes")
- Classify entities into EXACTLY these types:
  person, organization, project, tool, preference, constraint, location, event, concept, goal
- Break complex statements into atomic claims
- Include emotional valence when sentiment is expressed (-1.0 to 1.0)
```

For the offline fallback (Python library without host agent), the extraction prompt is more explicit:

```python
EXTRACTION_PROMPT = """You are Engram's extraction engine. Extract structured knowledge from the user's input.

RULES:
1. COREFERENCE RESOLUTION IS MANDATORY: Never extract pronouns ("he", "it", "they").
   Resolve them to explicit entity names based on conversation history.
2. FIXED TYPES ONLY: Classify entities into exactly these 10 types:
   person, organization, project, tool, preference, constraint, location, event, concept, goal
3. ATOMIC CLAIMS: Break complex sentences into simple (subject, predicate, object) triplets.
4. TYPO CORRECTION: Normalize entity names ("pythn" → "Python", "k8s" → "Kubernetes").
5. Output ONLY valid JSON matching this schema. No explanation.

EXAMPLE INPUT:
"My boss Dave said he's moving the React project to Next.js because he hates the build times."

EXAMPLE OUTPUT:
{
  "entities": [
    {"name": "Dave", "type": "person", "aliases": ["my boss", "he"]},
    {"name": "React project", "type": "project", "aliases": ["the React project"]},
    {"name": "Next.js", "type": "tool", "aliases": []},
    {"name": "build times", "type": "concept", "aliases": []}
  ],
  "claims": [
    {"subject": "Dave", "predicate": "is_migrating", "object": "React project", "to": "Next.js"},
    {"subject": "Dave", "predicate": "dislikes", "object": "build times", "valence": -0.7}
  ]
}"""
```

### 5.4 Entity Resolution (Tiered, Runs on Every Ingestion)

```python
from rapidfuzz import process, fuzz

class EntityResolver:
    """Resolves extracted entity names to existing atoms or creates new ones.
    Tiered: exact → alias → fuzzy → embedding → create."""

    def __init__(self, db, embed_engine, fuzzy_threshold: float = 85.0,
                 cosine_threshold: float = 0.3):
        self.db = db
        self.embed_engine = embed_engine
        self.fuzzy_threshold = fuzzy_threshold
        self.cosine_threshold = cosine_threshold

    def resolve(self, name: str, entity_type: str, agent_id: str) -> str:
        """Returns atom_id for the resolved entity."""

        # Tier 0: Exact label match (indexed, < 1ms)
        row = self.db.execute(
            "SELECT id FROM atoms WHERE LOWER(label) = LOWER(?) AND entity_type = ? AND agent_id = ?",
            (name, entity_type, agent_id)
        ).fetchone()
        if row:
            self._boost_sti(row["id"])
            return row["id"]

        # Tier 1: Alias table lookup (< 1ms)
        row = self.db.execute(
            "SELECT atom_id FROM aliases WHERE LOWER(alias) = LOWER(?) AND agent_id = ?",
            (name, agent_id)
        ).fetchone()
        if row:
            self._boost_sti(row["atom_id"])
            return row["atom_id"]

        # Tier 2: Fuzzy match via RapidFuzz against same entity_type (< 5ms)
        candidates = self.db.execute(
            "SELECT id, label FROM atoms WHERE entity_type = ? AND agent_id = ? AND type != 'relation'",
            (entity_type, agent_id)
        ).fetchall()
        if candidates:
            names_map = {r["id"]: r["label"] for r in candidates}
            match = process.extractOne(name, names_map, scorer=fuzz.WRatio)
            if match and match[1] >= self.fuzzy_threshold:
                matched_id = match[2]
                self._add_alias(matched_id, name, agent_id)
                self._boost_sti(matched_id)
                return matched_id

        # Tier 3: Embedding cosine similarity (< 20ms, catches semantic equivalents)
        query_vec = self.embed_engine.embed(name)
        vec_match = self.db.execute(
            """SELECT atom_id, distance FROM vec_atoms
               WHERE embedding MATCH ? AND agent_id = ?
               ORDER BY distance LIMIT 1""",
            (query_vec, agent_id)
        ).fetchone()
        if vec_match and vec_match["distance"] < self.cosine_threshold:
            self._add_alias(vec_match["atom_id"], name, agent_id)
            return vec_match["atom_id"]

        # Tier 4: No match — create new atom
        return self._create_atom(name, entity_type, agent_id)

    def _add_alias(self, atom_id: str, alias: str, agent_id: str):
        """Remember this alias for instant Tier 1 resolution next time."""
        self.db.execute(
            "INSERT OR IGNORE INTO aliases (atom_id, alias, agent_id) VALUES (?, ?, ?)",
            (atom_id, alias, agent_id)
        )

    def _boost_sti(self, atom_id: str):
        self.db.execute(
            "UPDATE atoms SET sti = sti + 0.5 WHERE id = ?", (atom_id,)
        )

    def _create_atom(self, name: str, entity_type: str, agent_id: str) -> str:
        # ... creates atom, embeds it, returns new id
        pass
```

### 5.5 Alias Table (New Schema Addition)

```sql
-- Fast exact-match resolution for known aliases, abbreviations, typos
CREATE TABLE aliases (
    alias       TEXT NOT NULL,
    atom_id     TEXT NOT NULL REFERENCES atoms(id),
    agent_id    TEXT NOT NULL,
    created_at  TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (alias, agent_id)
);

CREATE INDEX idx_aliases_atom ON aliases(atom_id);
```

When RapidFuzz or embedding resolution succeeds, the matched name is added to the alias table. Next time the same name appears, it's resolved in < 1ms via Tier 1 — the system learns from every resolution.

### 5.6 v1.0 Pipeline: Tiered Architecture (for Bulk Ingestion)

For ingesting chat logs, documents, or codebases where calling the host LLM per-chunk is too expensive:

```
Incoming text
      │
      ▼
[Tier 0] Alias table lookup (exact match)                  < 1ms
         Known entity? ──→ Done. Link to atom.
      │ no match
      ▼
[Tier 1] RapidFuzz against known entity labels              < 5ms
         Fuzzy match ≥ 85%? ──→ Done. Add alias.
      │ no match
      ▼
[Tier 2] GLiNER zero-shot NER (custom labels = 10 types)   ~50ms
         Finds new entity spans on CPU (~300MB model)
         ──→ Create new atoms for discovered entities
      │
      ▼
[Tier 3] Host LLM / injected callable (ASYNC, optional)
         Only for messages flagged as "complex":
         - Contains pronouns (he/she/they/it/my/that)
         - Contains opinion markers (love/hate/prefer/think/believe)
         ──→ Pronoun resolution + claim extraction
```

**GLiNER integration (v1.0 only, optional dependency):**

```python
from gliner import GLiNER

ENTITY_LABELS = ["person", "organization", "project", "tool", "preference",
                 "constraint", "location", "event", "concept", "goal"]

class GLiNERExtractor:
    """Offline zero-shot NER. ~50ms on CPU, ~300MB model, no GPU needed."""

    def __init__(self, model_name: str = "urchade/gliner_medium-v2.1"):
        self.model = GLiNER.from_pretrained(model_name)

    def extract(self, text: str, threshold: float = 0.4) -> list[dict]:
        entities = self.model.predict_entities(text, ENTITY_LABELS, threshold=threshold)
        return [{"name": e["text"], "type": e["label"], "score": e["score"]}
                for e in entities]
```

**Escalation heuristic — when to call the host LLM:**

The escalation decision must be **language-agnostic** — no hardcoded pronouns, opinion
words, or language-specific patterns. Instead, we measure structural signals from the
outputs of Tier 0-2 themselves:

```python
from dataclasses import dataclass

@dataclass
class ExtractionSignals:
    """Signals collected from Tier 0-2 results. All language-agnostic."""
    text_token_count: int        # whitespace-split token count of input
    entities_extracted: int      # entities found by GLiNER (Tier 2)
    entities_resolved: int       # entities that matched existing atoms (Tiers 0-1)
    avg_resolve_distance: float  # mean embedding distance of resolved entities to nearest atom
    short_token_ratio: float     # fraction of tokens with len <= 2 (catches pronouns/deictics in any language)

def compute_escalation_score(s: ExtractionSignals) -> float:
    """
    Returns 0.0-1.0. Score >= 0.5 → escalate to host LLM.

    Language-agnostic: uses only mathematical properties of the extraction
    results, not any word lists or regex patterns.
    """
    scores = []

    # 1. Resolution rate: what fraction of extracted entities are NEW (unresolved)?
    #    High novelty → the text references things we don't know about → LLM might help
    #    But also: if GLiNER found few entities, maybe there are implicit references
    if s.entities_extracted > 0:
        unresolved_rate = 1.0 - (s.entities_resolved / s.entities_extracted)
        scores.append(unresolved_rate * 0.2)

    # 2. Entity density: entities / tokens. Low density in long text = hidden references
    #    Short texts (< 5 tokens) are naturally low-density, so we scale by length
    if s.text_token_count >= 5:
        density = s.entities_extracted / s.text_token_count
        # Expect ~1 entity per 8 tokens in entity-rich text
        density_score = max(0.0, 1.0 - (density / 0.12))
        scores.append(density_score * 0.25)

    # 3. Short-token ratio: fraction of tokens with len <= 2
    #    Pronouns, deictics, and particles are short in virtually all languages
    #    (he, 他, él, er, il, 그, то, ...). High ratio = likely implicit references
    scores.append(min(1.0, s.short_token_ratio / 0.4) * 0.3)

    # 4. Embedding novelty: if resolved entities are far from known atoms,
    #    the text might be in unfamiliar territory where LLM context helps
    if s.avg_resolve_distance > 0:
        novelty = min(1.0, s.avg_resolve_distance / 0.6)
        scores.append(novelty * 0.15)

    # 5. Text length bonus: longer texts are more likely to contain
    #    complex references worth escalating
    length_factor = min(1.0, s.text_token_count / 50.0)
    scores.append(length_factor * 0.1)

    return min(1.0, sum(scores))


def compute_signals(text: str, extracted: list[dict], resolved_ids: list[str | None],
                    distances: list[float]) -> ExtractionSignals:
    """Compute escalation signals from Tier 0-2 results."""
    tokens = text.split()
    short_tokens = sum(1 for t in tokens if len(t) <= 2)
    return ExtractionSignals(
        text_token_count=len(tokens),
        entities_extracted=len(extracted),
        entities_resolved=sum(1 for r in resolved_ids if r is not None),
        avg_resolve_distance=sum(distances) / max(len(distances), 1),
        short_token_ratio=short_tokens / max(len(tokens), 1),
    )


# Usage in the tiered pipeline:
signals = compute_signals(text, gliner_entities, resolved_ids, distances)
score = compute_escalation_score(signals)
if score >= 0.5 and host_llm_available:
    # Escalate to Tier 3: host LLM for pronoun resolution + claim extraction
    await escalate_to_llm(text, context_history)
```

**Why this works across languages:**
- **No word lists:** Pronouns in Chinese (他/她/它), Spanish (él/ella), Arabic (هو/هي),
  Japanese (彼/彼女), Korean (그/그녀) are all short tokens — caught by `short_token_ratio`
- **Pro-drop languages** (Spanish, Japanese, Turkish): pronouns are often omitted entirely,
  causing low entity density → caught by density score
- **Agglutinative languages** (Turkish, Finnish): entity names are long compound words,
  so short tokens are even more diagnostic of pronouns/particles
- **The system learns:** as the alias table grows, resolution rate increases,
  and escalation naturally decreases over time

### 5.7 Offline Fallback (Python Library Mode)

When Engram is used without a host agent, the user injects their own LLM callable:

```python
from engram import Engram

# Option A: Use a local model via Ollama
mem = Engram(
    db_path="memory.db",
    extractor=lambda text: ollama.chat(
        model="qwen2.5:1.5b",
        messages=[{"role": "system", "content": EXTRACTION_PROMPT},
                  {"role": "user", "content": text}],
        format="json"
    )
)

# Option B: Use an API model
mem = Engram(
    db_path="memory.db",
    extractor=lambda text: litellm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": EXTRACTION_PROMPT},
                  {"role": "user", "content": text}],
        response_format={"type": "json_object"}
    )
)

# Option C: No LLM at all (v1.0 — GLiNER only, no claim extraction)
mem = Engram(db_path="memory.db", extractor="gliner")
```

Engram does NOT ship an LLM. It accepts one via dependency injection. The `extractor` parameter is a callable `(str) -> dict` that returns the extraction schema.

### 5.8 Extraction Data Flow Summary

```
┌─────────────────────────────────────────────────────────────┐
│                    EXTRACTION PIPELINE                       │
│                                                              │
│  v0.1: Host LLM Piggyback                                   │
│  ┌──────────┐    ┌───────────────┐    ┌──────────────────┐  │
│  │ Host LLM │──→ │ engram_remember│──→ │ EntityResolver   │  │
│  │ tool_use │    │ (MCP tool)    │    │ exact→fuzzy→embed│  │
│  └──────────┘    └───────────────┘    └──────────────────┘  │
│                                                              │
│  v1.0: Tiered (for bulk/offline)                             │
│  ┌──────────┐    ┌──────────┐    ┌────────┐    ┌────────┐  │
│  │ Alias    │──→ │ RapidFuzz│──→ │ GLiNER │──→ │ Host   │  │
│  │ Table    │    │ (fuzzy)  │    │ (NER)  │    │ LLM    │  │
│  │ < 1ms   │    │ < 5ms   │    │ ~50ms  │    │ (async)│  │
│  └──────────┘    └──────────┘    └────────┘    └────────┘  │
│        ↓               ↓              ↓             ↓       │
│        └───────────────┴──────────────┴─────────────┘       │
│                         ↓                                    │
│                  EntityResolver                              │
│              (resolve against memory)                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Retrieval: Salience-Scored Fan-Out

### 6.1 The Retrieval Pipeline

```
User query
    │
    ▼
[1] Extract entities from query (LLM tool call or NER)
    │
    ▼
[2] Match entities to atoms:
    ├── Exact label match (fast, indexed)
    └── KNN vector search on vec_atoms (top 50)
    │
    ▼
[3] Fan-out: 1-hop graph expansion via recursive CTE
    │
    ▼
[4] Score all candidates by SALIENCE:
    │
    │   S = (w_sim × cosine_similarity)
    │     + (w_sti × normalized_sti)
    │     + (w_conf × confidence)
    │     + (w_lti × normalized_lti)
    │     + (w_val × |valence| × intensity)
    │
    ▼
[5] Return top K atoms with full epistemic context
```

### 6.2 Salience SQL Query

```sql
-- Step 1: KNN entry points
WITH knn AS (
    SELECT atom_id, distance
    FROM vec_atoms
    WHERE embedding MATCH :query_embedding
      AND agent_id = :agent_id
    ORDER BY distance
    LIMIT 50
),
-- Step 2: 1-hop expansion (bounded spreading activation)
expanded AS (
    SELECT a.id, a.label, a.type, a.content,
           a.probability, a.confidence, a.sti, a.lti,
           a.valence, a.intensity,
           COALESCE(1.0 - k.distance, 0.0) AS similarity
    FROM atoms a
    LEFT JOIN knn k ON k.atom_id = a.id
    WHERE a.id IN (SELECT atom_id FROM knn)
       OR a.id IN (
           SELECT target_id FROM atoms
           WHERE source_id IN (SELECT atom_id FROM knn)
             AND type = 'relation'
       )
       OR a.id IN (
           SELECT source_id FROM atoms
           WHERE target_id IN (SELECT atom_id FROM knn)
             AND type = 'relation'
       )
),
-- Step 3: Score by salience
scored AS (
    SELECT *,
        (p.w_similarity * similarity)
      + (p.w_sti * MIN(sti / 2.0, 1.0))
      + (p.w_confidence * confidence)
      + (p.w_lti * MIN(lti / 2.0, 1.0))
      + (p.w_valence * ABS(valence) * intensity)
    AS salience
    FROM expanded e
    CROSS JOIN personality p
    WHERE p.agent_id = :agent_id
)
SELECT * FROM scored
WHERE type != 'relation'
ORDER BY salience DESC
LIMIT :top_k;
```

---

## 7. Evolution Mechanics

### 7.1 Truth Maintenance (Bayesian Update)

When new evidence arrives for an atom, update its TruthValue:

```python
def update_truth(current: TruthValue, evidence_prob: float, evidence_weight: float,
                 lr: float) -> TruthValue:
    """
    Weighted Bayesian-inspired update.
    lr = confidence_update_lr from personality params.
    """
    # Effective weight = evidence reliability × learning rate
    w = evidence_weight * lr

    # Update probability: weighted average biased by confidence
    new_prob = (current.probability * current.confidence + evidence_prob * w) / (current.confidence + w)

    # Update confidence: approaches 1.0 asymptotically
    new_conf = min(1.0, current.confidence + w * (1.0 - current.confidence))

    return TruthValue(probability=clamp(new_prob, 0.0, 1.0), confidence=new_conf)
```

### 7.2 Attention Decay (Per Consolidation Epoch)

```python
def decay_attention(atom: Atom, personality: PersonalityProfile) -> Atom:
    # STI decays exponentially
    atom.attention.sti *= (1.0 - personality.sti_decay_rate)

    # Promote to LTI if STI exceeds threshold
    if atom.attention.sti > personality.lti_promotion_threshold:
        atom.attention.lti = max(atom.attention.lti, atom.attention.sti * 0.5)

    # Confidence also decays slowly (forgetting curve)
    atom.truth.confidence *= (1.0 - personality.confidence_decay_rate)

    return atom
```

### 7.3 STI Propagation (Bounded Spreading Activation)

```python
def propagate_sti(atom_id: str, boost: float, personality: PersonalityProfile, db):
    """Spread a fraction of STI to 1-hop neighbors. No recursion = no oscillation."""
    spread = boost * personality.sti_propagation_factor
    if spread < 0.01:
        return  # below threshold, stop

    neighbors = db.execute("""
        SELECT target_id FROM atoms WHERE source_id = ? AND type = 'relation'
        UNION
        SELECT source_id FROM atoms WHERE target_id = ? AND type = 'relation'
    """, (atom_id, atom_id))

    for (neighbor_id,) in neighbors:
        db.execute("UPDATE atoms SET sti = MIN(sti + ?, 3.0) WHERE id = ?",
                   (spread, neighbor_id))
```

### 7.4 Cross-Domain Connection Discovery ("Charisma" Engine)

Runs during consolidation epoch. Finds atoms that are semantically similar (by embedding) but NOT already connected in the graph:

```python
async def discover_connections(agent_id: str, db, embed_engine):
    """Find surprising associations between unconnected high-LTI atoms."""
    # Get high-importance atoms
    high_lti = db.execute("""
        SELECT id, label, content FROM atoms
        WHERE agent_id = ? AND lti > 0.3 AND type != 'relation'
        ORDER BY lti DESC LIMIT 50
    """, (agent_id,))

    for atom in high_lti:
        # Find semantically similar atoms NOT already linked
        candidates = db.execute("""
            WITH knn AS (
                SELECT atom_id, distance FROM vec_atoms
                WHERE embedding MATCH :emb AND agent_id = :aid
                ORDER BY distance LIMIT 10
            )
            SELECT k.atom_id, k.distance FROM knn k
            WHERE k.atom_id != :id
              AND k.atom_id NOT IN (
                  SELECT target_id FROM atoms WHERE source_id = :id AND type = 'relation'
              )
              AND k.distance < 0.4  -- reasonably similar
        """, {"emb": atom.embedding, "aid": agent_id, "id": atom.id})

        for candidate in candidates:
            # Create a weak 'associated' link
            create_relation(atom.id, candidate.atom_id, "associated",
                          truth=TruthValue(probability=0.5, confidence=0.1))
```

### 7.5 Consolidation Epoch (The Main Loop)

```python
async def run_epoch(agent_id: str, db, personality: PersonalityProfile):
    """Single deterministic consolidation pass. Runs every ~60 seconds."""

    # 1. Process pending evidence
    pending = db.execute(
        "SELECT * FROM evidence WHERE processed = 0 AND agent_id = ? ORDER BY created_at",
        (agent_id,))
    for ev in pending:
        atom = get_atom(ev.atom_id)
        atom.truth = update_truth(atom.truth, ev.observed_probability,
                                  ev.weight, personality.confidence_update_lr)
        save_atom(atom)
        db.execute("UPDATE evidence SET processed = 1 WHERE id = ?", (ev.id,))

    # 2. Decay attention and confidence for ALL atoms
    db.execute("""
        UPDATE atoms SET
            sti = sti * (1.0 - ?),
            confidence = confidence * (1.0 - ?)
        WHERE agent_id = ?
    """, (personality.sti_decay_rate, personality.confidence_decay_rate, agent_id))

    # 3. Promote high-STI to LTI
    db.execute("""
        UPDATE atoms SET lti = MAX(lti, sti * 0.5)
        WHERE agent_id = ? AND sti > ?
    """, (agent_id, personality.lti_promotion_threshold))

    # 4. Resolve contradictions
    contradictions = db.execute("""
        SELECT a.id, a.source_id, a.target_id
        FROM atoms a
        WHERE a.type = 'relation' AND a.relation = 'contradicts' AND a.agent_id = ?
    """, (agent_id,))
    for c in contradictions:
        # Weaker belief loses confidence faster
        weaken_loser(c.source_id, c.target_id, personality)

    # 5. Discover cross-domain connections (every 10th epoch)
    if epoch_count % 10 == 0:
        await discover_connections(agent_id, db, embed_engine)

    # 6. Prune dead atoms (confidence near zero, low LTI)
    db.execute("""
        DELETE FROM atoms
        WHERE agent_id = ? AND confidence < ? AND lti < 0.05 AND type != 'episode'
    """, (agent_id, personality.min_confidence_to_surface))
```

---

## 8. Personality Presets

### 8.1 Preset Definitions

```json
{
  "balanced": {
    "confidence_decay_rate": 0.02,
    "confidence_update_lr": 0.3,
    "sti_decay_rate": 0.1,
    "sti_boost_on_access": 0.5,
    "sti_propagation_factor": 0.15,
    "lti_promotion_threshold": 0.7,
    "valence_weight": 0.2,
    "mood_inertia": 0.8,
    "w_similarity": 0.35,
    "w_sti": 0.25,
    "w_confidence": 0.20,
    "w_lti": 0.10,
    "w_valence": 0.10
  },

  "analytical": {
    "confidence_decay_rate": 0.01,
    "confidence_update_lr": 0.15,
    "sti_decay_rate": 0.05,
    "sti_propagation_factor": 0.05,
    "lti_promotion_threshold": 0.9,
    "valence_weight": 0.05,
    "mood_inertia": 0.95,
    "w_similarity": 0.30,
    "w_sti": 0.15,
    "w_confidence": 0.40,
    "w_lti": 0.10,
    "w_valence": 0.05,
    "_description": "Slow to change beliefs, low emotional reactivity, trusts high-confidence facts above all"
  },

  "curious": {
    "confidence_decay_rate": 0.03,
    "confidence_update_lr": 0.5,
    "sti_decay_rate": 0.2,
    "sti_boost_on_access": 0.8,
    "sti_propagation_factor": 0.3,
    "lti_promotion_threshold": 0.5,
    "valence_weight": 0.15,
    "mood_inertia": 0.5,
    "w_similarity": 0.25,
    "w_sti": 0.35,
    "w_confidence": 0.15,
    "w_lti": 0.10,
    "w_valence": 0.15,
    "_description": "High STI propagation = lateral thinking. Fast decay = moves on quickly. Surfaces novel low-confidence ideas"
  },

  "empathetic": {
    "confidence_decay_rate": 0.02,
    "confidence_update_lr": 0.4,
    "sti_decay_rate": 0.08,
    "sti_propagation_factor": 0.2,
    "valence_weight": 0.4,
    "valence_propagation": 0.25,
    "mood_inertia": 0.4,
    "w_similarity": 0.25,
    "w_sti": 0.20,
    "w_confidence": 0.10,
    "w_lti": 0.10,
    "w_valence": 0.35,
    "_description": "High valence weight = mood-matching. Low mood inertia = emotionally reactive. Remembers feelings"
  },

  "maverick": {
    "confidence_decay_rate": 0.005,
    "confidence_update_lr": 0.1,
    "sti_decay_rate": 0.15,
    "sti_propagation_factor": 0.35,
    "lti_promotion_threshold": 0.4,
    "valence_weight": 0.25,
    "mood_inertia": 0.7,
    "w_similarity": 0.20,
    "w_sti": 0.30,
    "w_confidence": 0.15,
    "w_lti": 0.15,
    "w_valence": 0.20,
    "_description": "Very slow confidence decay = stubborn core beliefs. High STI propagation = wild associations. Charismatic: connects disparate concepts with conviction"
  }
}
```

### 8.2 How Personality Traits Emerge

| Trait | Mechanism | Observable Behavior |
|---|---|---|
| **Stubborn** | Low `confidence_update_lr`, low `confidence_decay_rate` | Holds beliefs despite counter-evidence. Consistent across conversations |
| **Curious** | High `sti_propagation_factor`, high `sti_boost_on_access` | Mentions tangential topics. Asks probing questions about low-confidence atoms |
| **Empathetic** | High `valence_weight`, low `mood_inertia` | Mirrors emotional tone. Recalls emotionally-charged memories preferentially |
| **Analytical** | High `w_confidence`, low `valence_weight` | Cites evidence. Qualifies uncertain statements. Ignores emotional context |
| **Charismatic** | High `sti_propagation_factor` + cross-domain discovery | Draws surprising metaphors. Connects distant concepts with confidence |

---

## 9. Agent Interface: Tools

### 9.1 MCP Tool Definitions

The MCP server exposes these tools to any connected agent:

```python
TOOLS = [
    {
        "name": "engram_remember",
        "description": "Store a memory, belief, or observation. The system extracts entities, assigns truth values, and links to existing knowledge.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "The memory or observation to store"},
                "type": {"type": "string", "enum": ["belief", "episode", "goal"], "default": "episode"},
                "probability": {"type": "number", "description": "How true is this (0-1)", "default": 0.8},
                "valence": {"type": "number", "description": "Emotional tone (-1 to 1)", "default": 0.0}
            },
            "required": ["content"]
        }
    },
    {
        "name": "engram_recall",
        "description": "Retrieve relevant memories using salience-scored search. Returns memories with their truth values, confidence, and emotional context.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to recall"},
                "top_k": {"type": "integer", "default": 10},
                "min_confidence": {"type": "number", "default": 0.1}
            },
            "required": ["query"]
        }
    },
    {
        "name": "engram_reflect",
        "description": "Trigger a consolidation pass. Updates beliefs based on evidence, decays attention, discovers new connections. Returns a summary of changes.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "engram_believe",
        "description": "Assert or update a specific belief with a truth value. If the belief contradicts existing knowledge, creates a contradiction link.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "statement": {"type": "string"},
                "probability": {"type": "number"},
                "evidence": {"type": "string", "description": "Why you believe this"}
            },
            "required": ["statement", "probability"]
        }
    },
    {
        "name": "engram_forget",
        "description": "Lower confidence on a memory or belief. Does not hard-delete — the consolidation epoch handles pruning.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to forget"},
                "reason": {"type": "string"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "engram_personality",
        "description": "Get or set the agent's personality profile. Affects how memories are scored, how fast beliefs change, and emotional reactivity.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["get", "set", "preset"]},
                "preset": {"type": "string", "enum": ["balanced", "analytical", "curious", "empathetic", "maverick"]},
                "params": {"type": "object", "description": "Custom personality parameters"}
            },
            "required": ["action"]
        }
    },
    {
        "name": "engram_status",
        "description": "Get memory statistics: total atoms, active beliefs, emotional state, attention distribution.",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    }
]
```

### 9.2 REST API (for Gemini / OpenAI / Custom Agents)

FastAPI server exposes identical functionality via HTTP:

```
POST /remember        → engram_remember
POST /recall          → engram_recall
POST /reflect         → engram_reflect
POST /believe         → engram_believe
POST /forget          → engram_forget
GET  /personality     → get personality
PUT  /personality     → set personality
GET  /status          → memory stats
GET  /atoms/{id}      → get specific atom
GET  /graph           → export graph (JSON-LD or DOT format)
GET  /openapi.json    → OpenAPI spec (Gemini/OpenAI can auto-discover tools)
```

### 9.3 Python Library API

```python
from engram import Engram

# Initialize (creates SQLite DB if not exists)
mem = Engram(db_path="~/.engram/memory.db", personality="curious")

# Remember
mem.remember("User prefers Python over JavaScript", probability=0.9, valence=0.3)

# Recall (salience-scored)
results = mem.recall("What programming languages does the user know?", top_k=5)
for atom in results:
    print(f"{atom.label} (p={atom.truth.probability:.2f}, c={atom.truth.confidence:.2f})")

# Believe (with evidence)
mem.believe("User is working on a Rust project", probability=0.7,
            evidence="They mentioned cargo build in conversation")

# Reflect (manual consolidation trigger)
changes = mem.reflect()
print(f"Updated {changes.beliefs_updated} beliefs, discovered {changes.new_connections} connections")

# Personality
mem.set_personality("maverick")
```

---

## 10. Distribution Strategy

### 10.1 Python (Primary)

```toml
# pyproject.toml
[project]
name = "engram-memory"
version = "0.1.0"
description = "AtomSpace-inspired memory + personality engine for AI agents"
requires-python = ">=3.10"
dependencies = [
    "sqlite-vec>=0.1.0",
    "fastembed>=0.4.0",
    "rapidfuzz>=3.0",         # fuzzy entity resolution (Tier 2)
    "pydantic>=2.0",
    "aiosqlite>=0.20.0",
    "typer>=0.12.0",
    "mcp>=1.0.0",             # MCP Python SDK
    "fastapi>=0.115.0",
    "uvicorn>=0.30.0",
]

[project.scripts]
engram = "engram.cli:app"

[project.optional-dependencies]
openai = ["openai>=1.0"]     # optional: use OpenAI embeddings
gliner = ["gliner>=0.2.0"]   # optional: offline zero-shot NER (v1.0 tiered extraction)
ollama = ["ollama>=0.3.0"]   # optional: local LLM for offline claim extraction
```

```bash
# User installs
pip install engram-memory

# Run as MCP server (for Claude Code, etc.)
engram serve mcp

# Run as REST API (for Gemini, OpenAI, custom agents)
engram serve rest --port 8420

# Initialize with a personality
engram init --personality curious --agent-id my-agent
```

### 10.2 Claude Code Integration

```json
// ~/.claude/settings.json
{
  "mcpServers": {
    "engram": {
      "command": "engram",
      "args": ["serve", "mcp"],
      "env": {
        "ENGRAM_DB": "~/.engram/memory.db",
        "ENGRAM_PERSONALITY": "balanced"
      }
    }
  }
}
```

### 10.3 npm Wrapper (for JS/TS agents)

```bash
npx engram-memory serve rest --port 8420
```

Thin npm package that downloads the Python wheel and manages the process.

### 10.4 Docker

```bash
docker run -v ~/.engram:/data ghcr.io/you/engram serve rest --port 8420
```

---

## 11. Development Phases

### Phase 1: Foundation (Weeks 1-3)
**Goal:** Working memory store with vector search + entity resolution.

- [ ] SQLite schema + sqlite-vec setup, including `aliases` table (`core/db.py`)
- [ ] Pydantic models (`core/models.py`)
- [ ] FastEmbed integration (`core/embed.py`)
- [ ] Basic CRUD: add atom, get atom, link atoms (`core/atomspace.py`)
- [ ] Entity resolution: exact → alias → fuzzy (RapidFuzz) → embedding → create (`extraction/resolve.py`)
- [ ] Alias table that learns from every successful resolution (`extraction/aliases.py`)
- [ ] Extraction prompt templates for host LLM piggyback (`extraction/prompts.py`)
- [ ] Vector search via `vec0` MATCH (`retrieval/fan_out.py`)
- [ ] Simple salience scoring: similarity + recency only (`retrieval/salience.py`)
- [ ] CLI: `engram init`, `engram status`
- [ ] Tests for core operations, entity resolution, alias learning
- [ ] **Milestone:** `mem.remember()` and `mem.recall()` work. "Dave" resolves to "David Smith"

### Phase 2: MCP + REST Servers (Weeks 3-4)
**Goal:** Any agent can connect.

- [ ] MCP server with all tool definitions (`servers/mcp.py`)
- [ ] FastAPI REST server with OpenAPI spec (`servers/rest.py`)
- [ ] CLI: `engram serve mcp` and `engram serve rest`
- [ ] Test with Claude Code as MCP client
- [ ] Test REST API with curl / OpenAI function calling
- [ ] **Milestone:** Claude Code can `engram_remember` and `engram_recall`

### Phase 3: Truth Maintenance + Evidence (Weeks 5-6)
**Goal:** Beliefs evolve with evidence. Contradictions detected.

- [ ] Evidence ledger (append-only writes)
- [ ] Bayesian truth update formula (`evolution/truth.py`)
- [ ] Contradiction detection: when new belief opposes existing
- [ ] `engram_believe` tool with evidence tracking
- [ ] `engram_forget` (soft delete via confidence reduction)
- [ ] Consolidation epoch v1: process evidence + resolve contradictions
- [ ] **Milestone:** Agent can say "I used to think X, but now I believe Y because..."

### Phase 4: Attention + Personality (Weeks 7-9)
**Goal:** Personality shapes memory retrieval and evolution.

- [ ] Full salience formula with all 5 weights
- [ ] STI decay + boost + propagation (`evolution/attention.py`)
- [ ] LTI promotion logic
- [ ] Personality table + presets
- [ ] `engram_personality` tool
- [ ] Personality-weighted consolidation epoch
- [ ] Valence tracking + emotional propagation (`evolution/valence.py`)
- [ ] **Milestone:** Setting personality to "curious" visibly changes what the agent recalls

### Phase 5: Emergence + Polish (Weeks 10-12)
**Goal:** Cross-domain connections, charisma engine, GLiNER tier, production hardening.

- [ ] Cross-domain connection discovery (`evolution/connections.py`)
- [ ] Graph export (JSON-LD, DOT visualization)
- [ ] `engram_status` with rich memory statistics
- [ ] GLiNER integration for offline/bulk extraction (`extraction/gliner.py`, optional dep)
- [ ] Tiered extraction pipeline: alias → fuzzy → GLiNER → host LLM
- [ ] Escalation heuristic: pronoun/opinion detection for LLM-required messages
- [ ] Dependency-injected `extractor` callable for Python library mode
- [ ] Rate limiting, error handling, connection pooling
- [ ] PyPI publish, npm wrapper, Docker image
- [ ] **Milestone:** v0.1.0 public release

### Phase 6: Advanced (Post-Launch)
- [ ] Multi-agent shared memory (agent_id partitioning already built in)
- [ ] Memory import/export between agents
- [ ] WebAssembly build for browser-based agents
- [ ] Fine-tuned small model for entity extraction (replace LLM dependency)
- [ ] Visualization UI (graph explorer)
- [ ] Benchmarks vs Mem0, Graphiti, Zep

---

## 12. Key Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| sqlite-vec is pre-v1, breaking changes | High | Medium | Pin version, vendor if needed. Schema migration tooling from day 1 |
| FastEmbed model quality insufficient | Low | High | Pluggable embedding provider. Default to bge-small, allow OpenAI override |
| Entity extraction via LLM is unreliable | Medium | High | Host LLM piggyback (most accurate model available). Constrained to 10 fixed types. No open-ended triples in v0.1 |
| Entity resolution produces false merges | Medium | High | Conservative fuzzy threshold (85%). Embedding cosine threshold (0.3). Type-scoped matching. Alias table audit tool |
| Pronoun resolution fails without host LLM | Medium | Medium | Language-agnostic escalation heuristic (short-token ratio, entity density, resolution rate) routes to LLM tier. GLiNER fallback extracts entity spans without resolution |
| Personality tuning is opaque/hard to debug | Medium | Medium | Expose full salience breakdown in recall results. Logging of all epoch mutations |
| SQLite concurrent write contention | Medium | Medium | Single-writer queue pattern. WAL mode. Consolidation epoch runs in separate thread |
| Graph traversal too slow at scale | Low | Medium | Bounded to 1-hop. Index on source_id/target_id. Practical limit ~100K atoms per agent |

---

## 13. Success Criteria

1. **`pip install engram-memory && engram serve mcp`** works in under 60 seconds on a fresh machine
2. Claude Code can remember, recall, and reflect via MCP tools
3. Setting personality to "curious" vs "analytical" produces **measurably different** recall results for the same query
4. Truth maintenance correctly resolves "I like X" → (later) "I don't like X anymore"
5. Cross-domain connection discovery finds at least one non-obvious association per 100 stored memories
6. Single-file SQLite DB stays under 100MB for 50K atoms
7. Recall latency < 200ms for databases with 10K atoms

---

## Appendix A: Enhanced Schema — True Hypergraph with Outgoing Sets

The backend architect agent proposed a more faithful AtomSpace-inspired schema where links are true hypergraph edges with ordered membership, not just binary source/target columns. This is the **recommended production schema** — it supports N-ary relations, ordered outgoing sets, and full-text search.

### A.1 Outgoing Junction Table (Replaces source_id/target_id)

```sql
-- Links express their structure through ordered outgoing sets,
-- not binary source_id/target_id columns.
-- A link's "name" is NULL; it is identified by (atom_type, outgoing_set).
CREATE TABLE outgoing (
    link_id     INTEGER NOT NULL REFERENCES atoms(id) ON DELETE CASCADE,
    position    INTEGER NOT NULL,   -- 0-indexed ordinal
    target_id   INTEGER NOT NULL REFERENCES atoms(id) ON DELETE CASCADE,
    PRIMARY KEY (link_id, position)
) WITHOUT ROWID;

CREATE INDEX idx_outgoing_target ON outgoing(target_id, link_id);
```

This enables:
- **N-ary relations:** `EvaluationLink(predicate, subject, object)` has positions 0, 1, 2
- **Incoming-set queries:** "find all links pointing at atom X" via `idx_outgoing_target`
- **Faithful AtomSpace semantics:** nodes have names, links have NULL names + outgoing sets

### A.2 Full-Text Search (FTS5 for Hybrid Retrieval)

```sql
-- Keyword search alongside vector search
CREATE VIRTUAL TABLE atoms_fts USING fts5(
    name,
    content='atoms',
    content_rowid='id'
);

-- Auto-sync triggers
CREATE TRIGGER atoms_ai AFTER INSERT ON atoms BEGIN
    INSERT INTO atoms_fts(rowid, name) VALUES (new.id, new.name);
END;
CREATE TRIGGER atoms_ad AFTER DELETE ON atoms BEGIN
    INSERT INTO atoms_fts(atoms_fts, rowid, name) VALUES ('delete', old.id, old.name);
END;
CREATE TRIGGER atoms_au AFTER UPDATE OF name ON atoms BEGIN
    INSERT INTO atoms_fts(atoms_fts, rowid, name) VALUES ('delete', old.id, old.name);
    INSERT INTO atoms_fts(rowid, name) VALUES (new.id, new.name);
END;
```

### A.3 PLN Revision Rule (Replaces Simple Weighted Average)

The backend architect proposed using OpenCog's PLN (Probabilistic Logic Networks) revision rule for truth maintenance — more principled than a simple weighted average:

```python
@dataclass(slots=True)
class TruthValue:
    strength: float = 1.0       # probability
    confidence: float = 0.0     # weight of evidence

    def merge(self, other: "TruthValue") -> "TruthValue":
        """PLN revision rule: combine two independent truth estimates."""
        w_a = self.confidence / (1.0 - self.confidence + 1e-9)
        w_b = other.confidence / (1.0 - other.confidence + 1e-9)
        w_total = w_a + w_b
        if w_total < 1e-9:
            return TruthValue(
                strength=(self.strength + other.strength) / 2.0,
                confidence=0.0
            )
        merged_s = (w_a * self.strength + w_b * other.strength) / w_total
        merged_c = w_total / (w_total + 1.0)
        return TruthValue(strength=merged_s, confidence=merged_c)
```

This correctly handles:
- **Evidence accumulation:** confidence approaches 1.0 asymptotically
- **Conflicting evidence:** high-confidence evidence dominates low-confidence
- **Independence assumption:** two observations combine their weights

### A.4 Spreading Activation via Recursive CTE

The graph traversal agent produced a production-grade spreading activation query:

```sql
WITH RECURSIVE spread(atom_id, activation, depth, path) AS (
    SELECT :start_id, :initial_stimulus, 0, CAST(:start_id AS TEXT)
    UNION ALL
    SELECT o2.target_id,
           s.activation * :decay_factor * a_link.tv_strength,
           s.depth + 1,
           s.path || ',' || CAST(o2.target_id AS TEXT)
    FROM spread s
    JOIN outgoing o1 ON o1.target_id = s.atom_id
    JOIN atoms a_link ON a_link.id = o1.link_id AND a_link.is_deleted = 0
    JOIN outgoing o2 ON o2.link_id = o1.link_id AND o2.target_id != s.atom_id
    WHERE s.depth < :max_depth
      AND s.activation * :decay_factor > :min_activation
      AND INSTR(s.path, CAST(o2.target_id AS TEXT)) = 0
)
SELECT a.id, a.name, a.atom_type, a.tv_strength, a.tv_confidence,
       SUM(sp.activation) AS total_activation, MIN(sp.depth) AS min_depth
FROM spread sp
JOIN atoms a ON a.id = sp.atom_id
WHERE a.is_deleted = 0
GROUP BY a.id
ORDER BY total_activation DESC
LIMIT :limit;
```

Key: `tv_strength` on the link attenuates signal — low-confidence links transmit less activation, which is PLN-compatible.

### A.5 Thread-Safe Connection Pool

```python
class Database:
    def __init__(self, db_path: str, read_pool_size: int = 4):
        self._write_conn = self._create_connection()
        self._write_lock = threading.Lock()
        self._read_pool: Queue[sqlite3.Connection] = Queue(maxsize=read_pool_size)
        for _ in range(read_pool_size):
            conn = self._create_connection()
            conn.execute("PRAGMA query_only = ON")
            self._read_pool.put(conn)

    def _create_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA cache_size=-64000")
        return conn
```

Pattern: **1 write connection** (mutex-protected) + **N read connections** (pool). Readers never block the writer in WAL mode.

---

## Appendix B: Cross-Platform Distribution (Agent Research)

### B.1 MCP is the Universal Standard (2025-2026)

Per the agent research, MCP has been adopted by **all major platforms**:
- **Claude** — native support since November 2024
- **OpenAI** — integrated across ChatGPT since March 2025
- **Google Gemini** — confirmed support April 2025
- **Donated to Agentic AI Foundation** (Linux Foundation) — December 2025
- **10,000+ published MCP servers, 2,000+ in official registry**

This means: **MCP stdio is the primary distribution mechanism.** REST API is the fallback for platforms that don't yet support MCP natively.

### B.2 MCP Resources and Prompts (Not Just Tools)

The MCP spec supports three primitives. The plan currently only defines Tools. Add:

**Resources (read-only context data):**
```json
{
  "uri": "engram://entities",
  "name": "All Known Entities",
  "description": "List of all entities in memory with their types and confidence"
}
```
```json
{
  "uri": "engram://graph/{entity_name}",
  "name": "Entity Graph",
  "description": "Knowledge subgraph around a specific entity"
}
```
```json
{
  "uri": "engram://personality",
  "name": "Current Personality Profile",
  "description": "Active personality hyperparameters and their effects"
}
```

**Prompts (user-directed workflows):**
```json
{
  "name": "engram_summarize",
  "description": "Generate a summary of everything known about an entity",
  "arguments": [
    {"name": "entity", "required": true},
    {"name": "max_facts", "required": false}
  ]
}
```

### B.3 Cross-Platform Tool Schema (Minimal Compatible Format)

For a tool to work across Claude (MCP), OpenAI, Gemini, and LangChain, use this minimal JSON Schema format:

```json
{
  "name": "function_name",
  "description": "What this function does",
  "parameters": {
    "type": "object",
    "properties": {
      "param1": { "type": "string", "description": "..." }
    },
    "required": ["param1"]
  }
}
```

**Constraints for cross-platform compatibility:**
- Parameter names: alphanumeric + underscore only
- Types: `string`, `number`, `boolean`, `integer`, `object`, `array`, `enum`
- Avoid: `anyOf`, `oneOf`, `allOf`, `$ref`
- Root schema: always `type: "object"`

### B.4 Embedding Strategy Options

| Backend | Install Size | Speed | Quality | Offline |
|---|---|---|---|---|
| **FastEmbed** (ONNX, bge-small-en-v1.5) | ~50MB | Sub-ms/CPU | Good (384d) | Yes |
| **Ollama** (nomic-embed-text-v1.5) | ~270MB | ~5ms/CPU | Better (768d) | Yes |
| **OpenAI** (text-embedding-3-small) | 0 (API) | ~50ms (network) | Best (1536d) | No |

**Default:** FastEmbed for zero-config portability. Ollama for quality. OpenAI for max quality with API key.

---

## Appendix C: sqlite-vec Operational Limits

Per the deep research agent:

| Metric | Value |
|---|---|
| **Optimal range** | 10K-500K vectors, sub-second queries |
| **Tested limit** | 1M vectors (degraded: seconds per query) |
| **ANN indexing** | Not yet available (brute-force only) |
| **Concurrent writes** | Single-writer (SQLite constraint) |
| **Distance functions** | L2, L1, cosine, dot product, hamming |
| **Vector types** | float32, int8, bit (binary quantization) |
| **SIMD** | AVX (x86) + NEON (ARM) accelerated |

**Practical impact:** For agent memory, 100K atoms per agent is more than sufficient. At 384 dimensions (bge-small), this is ~150MB of vector data. Well within sqlite-vec's sweet spot.

**Mitigation for scale:** Use `atom_type` as partition key in `vec0` to reduce search space. Use metadata columns for pre-filtering. Use bit vectors for dimensional reduction if needed.

---

## Appendix D: Inspirations and References

- [OpenCog AtomSpace](https://github.com/opencog/atomspace) — hypergraph, TruthValues, ECAN
- [OpenCog Hyperon](https://hyperon.opencog.org/) — MeTTa, metagraph rewriting
- [Zep / Graphiti](https://github.com/getzep/graphiti) — temporal knowledge graphs for agents
- [Cognee](https://github.com/topoteretes/cognee) — ontology-enhanced GraphRAG
- [Mem0](https://mem0.ai) — graph memory layer for agents
- [sqlite-vec](https://github.com/asg017/sqlite-vec) — vector search extension for SQLite
- [FastEmbed](https://github.com/qdrant/fastembed) — lightweight ONNX embedding inference
- [MCP Specification](https://modelcontextprotocol.io/) — Model Context Protocol for agent tools

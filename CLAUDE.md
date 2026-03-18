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
- `db.py` — SQLite schema (WAL mode). Tables: `atoms`, `vec_atoms` (virtual vec0 for KNN), `evidence` (append-only observation log), `personality`, `aliases`
- `embed.py` — Thread-safe FastEmbed singleton (paraphrase-multilingual-MiniLM-L12-v2, 384 dims, ONNX CPU, 50+ languages)

**Retrieval (`retrieval/`):** `fan_out.py` does KNN → 1-hop graph expansion → salience scoring. `salience.py` formula: `w_sim×sim + w_sti×sti + w_conf×conf + w_lti×lti + w_val×|valence|×intensity`. When valence < -0.5, weight dynamically shifts from `w_sti` to `w_valence` so old-but-critical errors outrank recent trivia. `classify.py` classifies recall results into severity levels (`critical_warning`, `known_antipattern`, `context`) based on valence/intensity/probability thresholds.

**Evolution (`evolution/`):** `epoch.py` runs consolidation cycles: process evidence log → decay STI/LTI → promote high-LTI nodes → resolve contradictions → prune low-salience atoms. `truth.py` implements PLN (Probabilistic Logic Networks) for merging independent probability estimates.

**Extraction (`extraction/`):** `resolve.py` cascades entity resolution: exact match → alias lookup → fuzzy (RapidFuzz) → embedding similarity → create new. `sentiment.py` estimates valence via cosine similarity against positive/negative anchor embeddings — language-agnostic, used by all server modes as a fallback when callers don't provide explicit valence. `extract.py` — async LLM-based entity/claim extraction from text; called by all server modes when `SMRTI_EXTRACT=1`. Before each extraction call, `_build_entity_context()` queries the top salient concept, belief, and goal atoms (person, organization, project, tool, location, event, goal, preference, constraint) from the memory graph and injects them as a `[Known entities]` block so the LLM can resolve pronouns ("I" → person, "we" → organization) even when the name isn't in the current message — using persisted memory, not raw conversation history. The extraction prompt includes a formal JSON schema and six few-shot examples (tool migration, org/location, error/failure with negative valence, coreference with known entities, goal/preference/constraint, event/location). `max_tokens=1024` prevents silent truncation on complex inputs. Entity lookup is case-insensitive with `setdefault` collision safety so subject/object casing mismatches never silently drop relation edges.

**Personality (`personality/`):** `PersonalityProfile` dataclass with 16 hyperparameters. Six presets (`balanced`, `analytical`, `curious`, `empathetic`, `maverick`, `deterministic`) stored as JSON in `presets/` and loaded into the `personality` DB table per tenant/space pair. `deterministic` is optimized for agentic workflows: fast learning (lr=0.4) + slow decay (0.005), high LTI promotion threshold (0.85), laser-focus attention (boost=0.8, propagation=0.05), and similarity-gated confidence ranking.

**Servers (`servers/`):** `config.py` centralises all shared env-var defaults (`SMRTI_DB`, `SMRTI_PERSONALITY`, `SMRTI_TENANT_ID`, `SMRTI_SPACE`, `SMRTI_READ_SPACES`, `SMRTI_EXTRACT`, `SMRTI_EXTRACT_URL`, `SMRTI_EXTRACT_MODEL`, `SMRTI_IGNORE_PATTERNS`) read by all server modes. `mcp.py` wraps Smrti as MCP stdio tools; `handle_tool()` recall response includes `severity` and `intensity` fields. All three modes (MCP, REST, proxy) auto-estimate valence via `extraction/sentiment.py` when callers don't supply an explicit value, activating the error-avoidance memory path for negative content. All three modes call `extraction/extract.py` after every `remember` operation to extract entities/claims and build concept nodes + relation edges (enabled by default via `SMRTI_EXTRACT`; proxy uses request auth, proxy and REST forward the request `Authorization` header; MCP passes no auth — works as-is with local LLMs). `rest.py` is a FastAPI REST server. `proxy.py` is an OpenAI-compatible proxy with severity-aware memory injection: recalled memories are split into two sections — behavioral constraints (`YOU MUST NOT`, `AVOID` for `critical_warning`/`known_antipattern`) and background context (`Note:` for `context`), each with its own preamble — and contextual query reformulation (configurable via `SMRTI_QUERY_MODE`, `SMRTI_QUERY_CONTEXT_MSGS`, `SMRTI_QUERY_MAX_CHARS`). All severity levels include a `confidence` qualifier. `reflect_loop.py` runs periodic background consolidation across all server modes (interval controlled by `SMRTI_REFLECT_INTERVAL`, default 60s, 0 to disable). `tools.py` defines the 6 shared tool schemas (remember, recall, reflect, believe, forget, status).

## Key Design Decisions

- **Append-only evidence log:** Truth values are never mutated directly — observations are logged to `evidence` and merged during epochs via PLN
- **Multi-tenant isolation:** Every query is partitioned by `tenant_id` and `space`
- **Lazy embedding init:** FastEmbed model loads on first use, not at import time
- **Salience over recency:** Retrieval ranks by a weighted salience score, not timestamps
- **Error-avoidance memory:** Severe negative-valence atoms (valence < -0.7, intensity > 0.7) get an LTI floor of 0.5 on creation, preventing epoch pruning. Salience weights shift dynamically for negative-valence atoms. Recall results carry a `severity` classification used by all serving modes.

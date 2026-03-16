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
smrti serve proxy      # OpenAI-compatible proxy on :8421
```

## Architecture

Smrti is an AtomSpace-inspired memory engine for AI agents. It stores beliefs as graph nodes with Bayesian truth values, emotional valence, and attention weights in a single SQLite file with vector indexing (sqlite-vec).

**Entry point:** `src/smrti/__init__.py` — `Smrti` class is the public facade (`remember`, `recall`, `believe`, `reflect`, `forget`, `status`).

**Core layers (`core/`):**
- `models.py` — Pydantic data structures: `Atom`, `TruthValue`, `AttentionValue`, `Valence`, `Evidence`, `RecallResult`
- `atomspace.py` — Graph operations: add/update atoms, link atoms, boost STI
- `db.py` — SQLite schema (WAL mode). Tables: `atoms`, `vec_atoms` (virtual vec0 for KNN), `evidence` (append-only observation log), `personality`
- `embed.py` — Thread-safe FastEmbed singleton (BAAI/bge-small-en-v1.5, 384 dims, ONNX CPU)

**Retrieval (`retrieval/`):** `fan_out.py` does KNN → 1-hop graph expansion → salience scoring. `salience.py` formula: `w_sim×sim + w_sti×sti + w_conf×conf + w_lti×lti + w_val×|valence|×intensity`. When valence < -0.5, weight dynamically shifts from `w_sti` to `w_valence` so old-but-critical errors outrank recent trivia. `classify.py` classifies recall results into severity levels (`critical_warning`, `known_antipattern`, `context`) based on valence/intensity/probability thresholds.

**Evolution (`evolution/`):** `epoch.py` runs consolidation cycles: process evidence log → decay STI/LTI → promote high-LTI nodes → resolve contradictions → prune low-salience atoms. `truth.py` implements PLN (Probabilistic Logic Networks) for merging independent probability estimates.

**Extraction (`extraction/`):** `resolve.py` cascades entity resolution: exact match → alias lookup → fuzzy (RapidFuzz) → embedding similarity → create new.

**Personality (`personality/`):** `PersonalityProfile` dataclass with 16 hyperparameters. Five presets (`balanced`, `analytical`, `curious`, `empathetic`, `maverick`) stored as JSON in `presets/` and loaded into the `personality` DB table per tenant/space pair.

**Servers (`servers/`):** `mcp.py` wraps Smrti as MCP stdio tools; `handle_tool()` recall response includes `severity` and `intensity` fields. `rest.py` is a FastAPI REST server. `proxy.py` is an OpenAI-compatible proxy with severity-aware memory injection (XML tags: `<critical_warning>`, `<known_antipattern>`, `<context>`) and contextual query reformulation (configurable via `SMRTI_QUERY_MODE`, `SMRTI_QUERY_CONTEXT_MSGS`, `SMRTI_QUERY_MAX_CHARS`). `tools.py` defines the 6 shared tool schemas (remember, recall, reflect, believe, forget, status).

## Key Design Decisions

- **Append-only evidence log:** Truth values are never mutated directly — observations are logged to `evidence` and merged during epochs via PLN
- **Multi-tenant isolation:** Every query is partitioned by `tenant_id` and `space`
- **Lazy embedding init:** FastEmbed model loads on first use, not at import time
- **Salience over recency:** Retrieval ranks by a weighted salience score, not timestamps
- **Error-avoidance memory:** Severe negative-valence atoms (valence < -0.7, intensity > 0.7) get an LTI floor of 0.5 on creation, preventing epoch pruning. Salience weights shift dynamically for negative-valence atoms. Recall results carry a `severity` classification used by all serving modes.

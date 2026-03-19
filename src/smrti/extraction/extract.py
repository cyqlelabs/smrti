"""LLM-based entity and claim extraction from conversational text."""
from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, Optional

import httpx

from .prompts import AGENT_EXTRACTION_PROMPT, CLAIMS_ONLY_PROMPT, ENTITY_TYPES, EXTRACTION_PROMPT

if TYPE_CHECKING:
    from smrti import Smrti

_http: Optional[httpx.AsyncClient] = None

_VALID_TYPES = set(ENTITY_TYPES)

# Per-session locks to serialize extractions within the same (tenant_id, write_space)
_session_locks: dict[str, asyncio.Lock] = {}


def _apply_thinking_mode(body: dict, mode: str) -> None:
    """Mutate a chat-completion request body to control thinking mode.

    Supports llama.cpp / vLLM Qwen3-style chat_template_kwargs.
    mode="auto" leaves the body untouched.
    """
    if mode == "disabled":
        body.setdefault("chat_template_kwargs", {})["enable_thinking"] = False
    elif mode == "enabled":
        body.setdefault("chat_template_kwargs", {})["enable_thinking"] = True


def _get_http() -> httpx.AsyncClient:
    global _http
    if _http is None:
        _http = httpx.AsyncClient(timeout=httpx.Timeout(30.0))
    return _http


async def extract_knowledge(
    text: str,
    http: httpx.AsyncClient,
    upstream: str,
    auth: str,
    model: str,
    entity_context: str = "",
    source: str = "user",
    tenant_id: str = "",
) -> Optional[dict]:
    """Call the upstream LLM to extract entities and claims from text.

    Returns a dict with 'entities' and 'claims' lists, or None on failure.
    """
    from smrti.call_log import append as _log

    system_prompt = AGENT_EXTRACTION_PROMPT if source == "agent" else EXTRACTION_PROMPT
    if entity_context and source != "agent":
        user_content = (
            f"[Known entities — use these to resolve pronouns and references]\n"
            f"{entity_context}\n\n"
            f"[Text to extract]\n{text}"
        )
    else:
        user_content = text

    from smrti.servers import config as _cfg
    request_body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.0,
        "max_tokens": 4096,
    }
    _apply_thinking_mode(request_body, _cfg.EXTRACT_THINKING)
    entry: dict = {
        "kind": "extraction",
        "subkind": "full",
        "tenant_id": tenant_id,
        "upstream": upstream,
        "model": model,
        "source": source,
        "request": request_body,
        "status": 0,
        "response_raw": "",
        "response_parsed": None,
        "error": None,
        "duration_ms": 0.0,
    }
    t0 = time.monotonic()
    try:
        resp = await http.post(
            f"{upstream}/v1/chat/completions",
            headers={"Content-Type": "application/json", "Authorization": auth},
            json=request_body,
            timeout=60.0,
        )
        entry["status"] = resp.status_code
        data = resp.json()
        msg = data["choices"][0]["message"]
        raw = (msg.get("content") or msg.get("reasoning_content") or "").strip()
        entry["response_raw"] = raw[:2000]
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        parsed = json.loads(raw)
        entry["response_parsed"] = parsed
        return parsed
    except Exception as exc:
        entry["error"] = str(exc)
        return None
    finally:
        entry["duration_ms"] = round((time.monotonic() - t0) * 1000, 1)
        _log(entry)


_COREF_TYPES = {"person", "organization", "project", "tool", "location", "event", "goal", "preference", "constraint"}


def _get_salient_person(mem: "Smrti") -> tuple[str, str] | None:
    """Return (label, atom_id) of the most salient person atom in the current space, or None."""
    row = mem.db.fetchone(
        """SELECT id, label FROM atoms
           WHERE tenant_id = ? AND space = ? AND entity_type = 'person'
             AND source_id IS NULL AND type IN ('concept', 'belief', 'goal')
           ORDER BY (sti + lti) DESC LIMIT 1""",
        (mem.tenant_id, mem.write_space),
    )
    return (row["label"], row["id"]) if row else None


def _build_entity_context(mem: "Smrti") -> str:
    """Return a compact list of salient named entities from the memory graph.

    Used to ground coreference resolution in the extraction prompt — e.g. so
    that "I" resolves to "Nico" even when the name isn't in the current message.
    Covers all entity types that can plausibly be referenced by a pronoun or
    short noun phrase: person, organization, project, tool, location, event, goal.
    """
    rows = mem.db.fetchall(
        """SELECT label, entity_type
           FROM atoms
           WHERE tenant_id = ? AND space = ? AND type IN ('concept', 'belief', 'goal')
             AND source_id IS NULL AND entity_type IS NOT NULL
           ORDER BY (sti + lti) DESC
           LIMIT 30""",
        (mem.tenant_id, mem.write_space),
    )
    lines = []
    for row in rows:
        etype = row["entity_type"] or ""
        if etype in _COREF_TYPES:
            lines.append(f"- {row['label']} ({etype})")
    return "\n".join(lines)


# ── Reusable helpers ──────────────────────────────────────────────────────────


def _db_resolve_label(label: str, entity_ids: dict[str, str], mem: "Smrti") -> str | None:
    """Resolve a claim subject/object to an atom_id, falling back to DB lookup."""
    atom_id = entity_ids.get(label) or entity_ids.get(label.lower())
    if atom_id:
        return atom_id
    row = mem.db.fetchone(
        "SELECT id FROM atoms WHERE LOWER(label) = LOWER(?) AND tenant_id = ? AND space = ?",
        (label, mem.tenant_id, mem.write_space),
    )
    if row:
        entity_ids[label] = row["id"]
        entity_ids[label.lower()] = row["id"]
        return row["id"]
    return None


def _resolve_ner_entities(
    entities: list[dict],
    episode_id: str,
    mem: "Smrti",
) -> dict[str, str]:
    """Resolve a list of {"name", "type"} dicts via the entity cascade.

    Returns a mapping of name → atom_id. Also creates mentions edges.
    Handles pronoun entities: batch-merges where unambiguous, skips type="pronoun",
    and retroactively merges existing pronoun atoms for resolved named persons.
    """
    from .resolve import EntityResolver

    resolver = EntityResolver(mem.db, mem.embed)
    entity_ids: dict[str, str] = {}

    # Batch-merge pronoun entities before resolution
    try:
        from .ner import get_ner
        ner = get_ner()
        from .pronouns import merge_pronoun_entities_in_batch
        entities = merge_pronoun_entities_in_batch(
            entities, ner,
            db=mem.db, tenant_id=mem.tenant_id, spaces=[mem.write_space],
        )
    except Exception:
        ner = None

    for ent in entities:
        name = (ent.get("name") or "").strip()
        etype = ent.get("type", "concept")
        if not name:
            continue
        # Skip pronoun-typed entities that survived batch merge (ambiguous case)
        if etype == "pronoun":
            continue
        if etype not in _VALID_TYPES:
            etype = "concept"
        atom_id = resolver.resolve(name, etype, mem.tenant_id, mem.write_space, [mem.write_space])
        entity_ids[name] = atom_id
        entity_ids.setdefault(name.lower(), atom_id)
        for alias in ent.get("aliases", []):
            if alias and alias.lower() != name.lower():
                resolver.aliases.add(atom_id, alias, mem.tenant_id, mem.write_space)
                entity_ids.setdefault(alias, atom_id)
                entity_ids.setdefault(alias.lower(), atom_id)
        mem.atomspace.link_atoms(episode_id, atom_id, "mentions", mem.tenant_id, mem.write_space)

    # Retroactive merge: for each resolved named person, merge co-mentioned pronoun atoms
    if ner is not None:
        from .pronouns import find_and_merge_pronoun_atoms
        for ent in entities:
            etype = ent.get("type", "concept")
            name = (ent.get("name") or "").strip()
            if etype != "person" or not name:
                continue
            atom_id = entity_ids.get(name)
            if atom_id and not ner.classify_pronoun(name):
                find_and_merge_pronoun_atoms(
                    atom_id, episode_id, mem.db, ner, mem.tenant_id, mem.write_space,
                )

    return entity_ids


def _link_claims(claims: list[dict], entity_ids: dict[str, str], mem: "Smrti") -> None:
    """Create relation edges from claim triplets."""
    _resolver = None
    for claim in claims:
        subj_raw = claim.get("subject", "")
        obj_raw = claim.get("object", "")
        subj_id = _db_resolve_label(subj_raw, entity_ids, mem)
        obj_id = _db_resolve_label(obj_raw, entity_ids, mem)
        # Auto-create missing object atoms as concepts rather than silently dropping
        if not obj_id and obj_raw:
            if _resolver is None:
                from .resolve import EntityResolver
                _resolver = EntityResolver(mem.db, mem.embed)
            obj_id = _resolver.resolve(obj_raw, "concept", mem.tenant_id, mem.write_space, [mem.write_space])
            entity_ids[obj_raw] = obj_id
            entity_ids.setdefault(obj_raw.lower(), obj_id)
        if subj_id and obj_id and subj_id != obj_id:
            predicate = claim.get("predicate", "related_to")
            claim_valence = float(claim.get("valence") or 0.0)
            mem.atomspace.link_atoms(
                subj_id, obj_id, predicate,
                mem.tenant_id, mem.write_space,
                valence=claim_valence,
            )
            # Safety net: promote target atom to goal type on has_goal claims
            if predicate == "has_goal":
                _promote_to_goal(obj_id, mem)


def _promote_to_goal(atom_id: str, mem: "Smrti") -> None:
    """Promote an atom to goal type if it isn't already."""
    row = mem.db.fetchone("SELECT type, entity_type FROM atoms WHERE id = ?", (atom_id,))
    if row and row["type"] != "goal":
        mem.db.execute(
            "UPDATE atoms SET type = 'goal', entity_type = 'goal' WHERE id = ?",
            (atom_id,),
        )


# ── Full LLM extraction path (original) ──────────────────────────────────────


async def extract_and_link(
    episode_id: str,
    content: str,
    mem: "Smrti",
    auth: str,
    model: str,
    upstream: str,
    source: str = "user",
) -> None:
    """Extract entities/claims from content and link them to the episode atom.

    Shared by all serve modes (proxy, MCP, REST). Silently no-ops if the LLM
    call fails or returns no usable structure.
    """
    entity_context = await asyncio.get_running_loop().run_in_executor(
        None, _build_entity_context, mem
    )
    extracted = await extract_knowledge(content, _get_http(), upstream, auth, model, entity_context, source, mem.tenant_id)
    if not extracted:
        return

    def _sync_work() -> None:
        entity_ids = _resolve_ner_entities(extracted.get("entities", []), episode_id, mem)
        _link_claims(extracted.get("claims", []), entity_ids, mem)

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _sync_work)


# ── Claims-only LLM call ─────────────────────────────────────────────────────


async def extract_claims_only(
    text: str,
    entities: list[dict],
    upstream: str,
    auth: str,
    model: str,
    entity_context: str = "",
    tenant_id: str = "",
) -> Optional[dict]:
    """Call the LLM with a shorter claims-only prompt, given pre-extracted entities.

    Returns {"claims": [...]} or None on failure.
    """
    from smrti.call_log import append as _log

    entities_block = "\n".join(
        f"- {e['name']} ({e['type']})" for e in entities
        if e.get("name") and e.get("type") != "pronoun"
    )
    system_prompt = CLAIMS_ONLY_PROMPT.replace("{entities_block}", entities_block)

    user_content = text
    if entity_context:
        user_content = (
            f"[Known entities — use these to resolve pronouns and references]\n"
            f"{entity_context}\n\n"
            f"[Text to extract]\n{text}"
        )

    from smrti.servers import config as _cfg
    request_body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.0,
        "max_tokens": 4096,
    }
    _apply_thinking_mode(request_body, _cfg.EXTRACT_THINKING)
    entry: dict = {
        "kind": "extraction",
        "subkind": "claims_only",
        "tenant_id": tenant_id,
        "upstream": upstream,
        "model": model,
        "source": "hybrid",
        "request": request_body,
        "status": 0,
        "response_raw": "",
        "response_parsed": None,
        "error": None,
        "duration_ms": 0.0,
    }
    t0 = time.monotonic()
    try:
        resp = await _get_http().post(
            f"{upstream}/v1/chat/completions",
            headers={"Content-Type": "application/json", "Authorization": auth},
            json=request_body,
            timeout=60.0,
        )
        entry["status"] = resp.status_code
        data = resp.json()
        msg = data["choices"][0]["message"]
        raw = (msg.get("content") or msg.get("reasoning_content") or "").strip()
        entry["response_raw"] = raw[:2000]
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        parsed = json.loads(raw)
        entry["response_parsed"] = parsed
        return parsed
    except Exception as exc:
        entry["error"] = str(exc)
        return None
    finally:
        entry["duration_ms"] = round((time.monotonic() - t0) * 1000, 1)
        _log(entry)


# ── Hybrid dispatch ───────────────────────────────────────────────────────────


async def extract_and_link_hybrid(
    episode_id: str,
    content: str,
    mem: "Smrti",
    auth: str,
    model: str,
    upstream: str,
    source: str = "user",
    mode: str = "hybrid",
) -> None:
    """Hybrid extraction: GLiNER for entities, LLM only for claims when needed.

    Modes:
      - "llm"    — full LLM path (backward compatible)
      - "hybrid" — GLiNER entities + LLM claims when 2+ entities
      - "local"  — GLiNER entities only, no LLM calls
    source == "agent" always takes the full LLM path.
    """
    if source == "agent" or mode == "llm":
        await extract_and_link(episode_id, content, mem, auth, model, upstream, source)
        return

    # Try GLiNER for entity extraction
    ner_entities: list[dict] | None = None
    try:
        from smrti.extraction import ner as ner_mod

        ner_instance = ner_mod.get_ner()
        loop = asyncio.get_running_loop()
        ner_entities = await loop.run_in_executor(None, ner_instance.extract, content)
    except ImportError:
        if mode == "hybrid":
            await extract_and_link(episode_id, content, mem, auth, model, upstream, source)
            return
        # local mode with no gliner installed — nothing we can do
        return
    except Exception:
        if mode == "hybrid":
            await extract_and_link(episode_id, content, mem, auth, model, upstream, source)
            return
        return

    if not ner_entities:
        return

    # Resolve entities and create mentions edges
    def _sync_resolve() -> dict[str, str]:
        return _resolve_ner_entities(ner_entities, episode_id, mem)

    loop = asyncio.get_running_loop()
    entity_ids = await loop.run_in_executor(None, _sync_resolve)

    # In local mode, we're done — no LLM calls
    if mode == "local":
        return

    # Speaker injection: for user messages, if no person atom resolved (pronoun dropped
    # because "I" alias wasn't in the alias table), inject the most salient known person
    # so claims — especially goals, preferences, and actions — can be attributed to them.
    # This treats first-person pronouns as speaker metadata rather than entity aliases,
    # preventing graph fragmentation when alias registration was missed in hybrid mode.
    if source == "user":
        def _inject_speaker_if_missing() -> list[dict]:
            atom_ids = list(set(entity_ids.values()))
            if atom_ids:
                ph = ",".join("?" * len(atom_ids))
                row = mem.db.fetchone(
                    f"SELECT 1 FROM atoms WHERE id IN ({ph}) AND entity_type = 'person' AND tenant_id = ?",
                    (*atom_ids, mem.tenant_id),
                )
                if row:
                    return ner_entities  # person already in scope
            person = _get_salient_person(mem)
            if person:
                label, atom_id = person
                entity_ids[label] = atom_id
                entity_ids.setdefault(label.lower(), atom_id)
                return ner_entities + [{"name": label, "type": "person"}]
            return ner_entities

        ner_entities = await loop.run_in_executor(None, _inject_speaker_if_missing)

    # Hybrid mode: call LLM for claims only when 2+ unique entities
    unique_ids = set(entity_ids.values())
    if len(unique_ids) < 2:
        return

    entity_context = await loop.run_in_executor(None, _build_entity_context, mem)
    claims_result = await extract_claims_only(
        content, ner_entities, upstream, auth, model, entity_context, mem.tenant_id
    )
    if not claims_result:
        return

    def _sync_resolve_and_link() -> None:
        # Resolve new entities the LLM emitted: goals (new atoms) and
        # preference/constraint reclassifications (resolve to existing atom,
        # updating its entity_type so it becomes a belief atom).
        _ALLOWED_NEW_TYPES = {"goal", "preference", "constraint", "concept"}
        new_entities = claims_result.get("entities", [])
        if new_entities:
            from .resolve import EntityResolver
            resolver = EntityResolver(mem.db, mem.embed)
            for ent in new_entities:
                name = (ent.get("name") or "").strip()
                etype = ent.get("type", "")
                if not name or etype not in _ALLOWED_NEW_TYPES:
                    continue
                atom_id = resolver.resolve(name, etype, mem.tenant_id, mem.write_space, [mem.write_space])
                entity_ids[name] = atom_id
                entity_ids.setdefault(name.lower(), atom_id)
                if etype in ("preference", "constraint"):
                    # Reclassify the atom: concept → belief, update entity_type
                    mem.db.execute(
                        "UPDATE atoms SET type = 'belief', entity_type = ? WHERE id = ? AND type = 'concept'",
                        (etype, atom_id),
                    )
                elif etype == "concept":
                    mem.atomspace.link_atoms(episode_id, atom_id, "mentions", mem.tenant_id, mem.write_space)
        _link_claims(claims_result.get("claims", []), entity_ids, mem)

    await loop.run_in_executor(None, _sync_resolve_and_link)


# ── Serialized wrapper ────────────────────────────────────────────────────────


async def extract_and_link_serialized(
    episode_id: str,
    content: str,
    mem: "Smrti",
    auth: str,
    model: str,
    upstream: str,
    source: str = "user",
    mode: str = "hybrid",
) -> None:
    """Serialize extractions within the same (tenant_id, write_space) session.

    Acquires a per-session asyncio.Lock so that episode N's entities are fully
    committed before episode N+1's ``_build_entity_context()`` query runs.
    Cross-session concurrency is preserved (different keys = different locks).
    """
    key = f"{mem.tenant_id}:{mem.write_space}"
    if key not in _session_locks:
        _session_locks[key] = asyncio.Lock()
    async with _session_locks[key]:
        await extract_and_link_hybrid(
            episode_id, content, mem, auth, model, upstream, source, mode
        )

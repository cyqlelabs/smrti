"""LLM-based entity and claim extraction from conversational text."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from typing import TYPE_CHECKING, Optional

import httpx

from smrti.core.models import STRUCTURAL_RELATIONS, SUPERSEDED_PROBABILITY, Evidence
from smrti.core.provenance import (
    ATOM_FORGOTTEN,
    ATOM_METADATA_JSON,
    ATOM_SOURCE,
    SOURCE_AGENT,
    SUPERSEDED_BY,
    claim_current_sql,
    forgotten_sql,
)
from smrti.decisions import DecisionEngine
from smrti.decisions.extraction import route_extraction, verify_supersession

from .prompts import AGENT_EXTRACTION_PROMPT, CLAIMS_ONLY_PROMPT, ENTITY_TYPES, EXTRACTION_PROMPT

if TYPE_CHECKING:
    from smrti import Smrti

logger = logging.getLogger("smrti.extract")

_VALID_TYPES = set(ENTITY_TYPES)

_EXTRACT_TIMEOUT: float = float(os.environ.get("SMRTI_EXTRACT_TIMEOUT", "60.0"))

# Per-loop HTTP clients: an AsyncClient is bound to the event loop it was
# created on — reusing it from a new loop hangs. Entries for closed loops are
# evicted lazily so the registry stays bounded.
_http_clients: dict[int, tuple[asyncio.AbstractEventLoop, httpx.AsyncClient]] = {}


class _PerLoopLocks:
    """Session locks keyed by (event loop, session key).

    An asyncio.Lock is bound to the loop that created it — awaiting one from
    another loop hangs. Lock dicts are therefore kept per loop, and entries
    for closed loops are evicted lazily so the registry stays bounded.
    Dict-style access (``in``, ``[]``, ``clear``) spans all loops.
    """

    def __init__(self) -> None:
        self._entries: dict[int, tuple[asyncio.AbstractEventLoop, dict[str, asyncio.Lock]]] = {}

    def get_lock(self, key: str) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        for loop_id, (cached_loop, _) in list(self._entries.items()):
            if cached_loop.is_closed():
                self._entries.pop(loop_id, None)
        entry = self._entries.get(id(loop))
        if entry is None:
            entry = (loop, {})
            self._entries[id(loop)] = entry
        locks = entry[1]
        if key not in locks:
            locks[key] = asyncio.Lock()
        return locks[key]

    def __contains__(self, key: str) -> bool:
        return any(key in locks for _, locks in self._entries.values())

    def __getitem__(self, key: str) -> asyncio.Lock:
        for _, locks in self._entries.values():
            if key in locks:
                return locks[key]
        raise KeyError(key)

    def clear(self) -> None:
        self._entries.clear()


# Per-session locks to serialize extractions within the same (tenant_id, write_space)
_session_locks = _PerLoopLocks()


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
    loop = asyncio.get_running_loop()
    for loop_id, (cached_loop, _) in list(_http_clients.items()):
        if cached_loop.is_closed():
            _http_clients.pop(loop_id, None)
    entry = _http_clients.get(id(loop))
    if entry is None:
        entry = (loop, httpx.AsyncClient(timeout=httpx.Timeout(30.0)))
        _http_clients[id(loop)] = entry
    return entry[1]


async def aclose_extract_clients() -> None:
    """Close the current event loop's HTTP client, if one was created."""
    entry = _http_clients.pop(id(asyncio.get_running_loop()), None)
    if entry is not None:
        await entry[1].aclose()


async def _post_chat(
    http: httpx.AsyncClient, upstream: str, auth: str, request_body: dict
) -> httpx.Response:
    """POST a chat-completion request, retrying once without chat_template_kwargs.

    Some OpenAI-compatible validators reject unknown fields with a 4xx; the
    retry drops the thinking-mode hint so extraction still works.
    """
    headers = {"Content-Type": "application/json", "Authorization": auth}
    resp = await http.post(
        f"{upstream}/v1/chat/completions",
        headers=headers, json=request_body, timeout=_EXTRACT_TIMEOUT,
    )
    if 400 <= resp.status_code < 500 and "chat_template_kwargs" in request_body:
        retry_body = {k: v for k, v in request_body.items() if k != "chat_template_kwargs"}
        resp = await http.post(
            f"{upstream}/v1/chat/completions",
            headers=headers, json=retry_body, timeout=_EXTRACT_TIMEOUT,
        )
    return resp


_ITEM_SHAPES = {
    "entities": ("name", "type"),
    "claims": ("subject", "predicate", "object"),
    "temporal": ("text", "resolved"),
}

# An ISO calendar date and nothing else. The model is asked for one; anything
# it returns that is not one is discarded rather than stored, because a
# resolution the reader cannot trust is worse than none.
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _validate_extraction(parsed: object) -> Optional[dict]:
    """Validate the shape of a parsed LLM extraction response.

    The top level must be a dict; "entities"/"claims" must be lists of dicts
    with the expected string fields. Malformed items are dropped rather than
    crashing the pipeline.
    """
    if not isinstance(parsed, dict):
        return None
    out = dict(parsed)
    for key, fields in _ITEM_SHAPES.items():
        if key not in out:
            continue
        items = out[key]
        if not isinstance(items, list):
            out[key] = []
            continue
        out[key] = [
            item for item in items
            if isinstance(item, dict)
            and all(isinstance(item.get(f), str) for f in fields)
        ]
    return out


def _write_time(episode_id: str, mem: "Smrti") -> str:
    """When the episode was stored, as the LLM's base for relative dates."""
    row = mem.db.fetchone(
        "SELECT created_at FROM atoms WHERE id = ? AND tenant_id = ?",
        (episode_id, mem.tenant_id),
    )
    return (row["created_at"] if row and row["created_at"] else "") or ""


def _temporal_block(write_time: str) -> str:
    """The header that tells the model what "tomorrow" is relative to."""
    return f"[Write time]\n{write_time}\n\n" if write_time else ""


def _store_temporal(episode_id: str, mem: "Smrti", items: list) -> None:
    """Record the LLM's resolved dates on the episode.

    Metadata, not text: the episode was embedded when it was stored, and
    rewriting its content now would leave the vector describing something the
    row no longer says. Recall renders these beside the memory instead.
    """
    resolved = [
        {"text": item["text"], "resolved": item["resolved"]}
        for item in items
        if isinstance(item, dict)
        and isinstance(item.get("text"), str)
        and isinstance(item.get("resolved"), str)
        and _ISO_DATE.match(item["resolved"])
    ]
    if not resolved:
        return
    # Merge, never replace. The write-time tier (GLiNER + dateparser) files
    # its resolutions here first and is authoritative for the spans it
    # handled — it is deterministic where the model is not — so the model
    # only adds spans the parser could not reach. One list, one reading per
    # span.
    existing = _existing_temporal(episode_id, mem)
    seen = {item["text"].casefold() for item in existing}
    merged = existing + [
        item for item in resolved if item["text"].casefold() not in seen
    ]
    if merged == existing:
        return
    mem.db.execute(
        f"""UPDATE atoms SET metadata = json_set({ATOM_METADATA_JSON},
                                                 '$.temporal', json(?))
            WHERE id = ? AND tenant_id = ?""",
        (json.dumps(merged), episode_id, mem.tenant_id),
    )


def _existing_temporal(episode_id: str, mem: "Smrti") -> list[dict]:
    row = mem.db.fetchone(
        "SELECT metadata FROM atoms WHERE id = ? AND tenant_id = ?",
        (episode_id, mem.tenant_id),
    )
    if row is None:
        return []
    try:
        items = json.loads(row["metadata"] or "{}").get("temporal")
    except (TypeError, ValueError, AttributeError):
        return []
    if not isinstance(items, list):
        return []
    return [
        {"text": item["text"], "resolved": item["resolved"]}
        for item in items
        if isinstance(item, dict)
        and isinstance(item.get("text"), str)
        and isinstance(item.get("resolved"), str)
    ]


async def extract_knowledge(
    text: str,
    http: httpx.AsyncClient,
    upstream: str,
    auth: str,
    model: str,
    entity_context: str = "",
    source: str = "user",
    tenant_id: str = "",
    write_time: str = "",
) -> Optional[dict]:
    """Call the upstream LLM to extract entities and claims from text.

    Returns a dict with 'entities', 'claims' and 'temporal' lists, or None on
    failure. ``write_time`` is the base the model resolves relative dates
    against; the idioms no date parser reaches ("el finde que viene") are
    covered here, at no extra call, because this request was being made
    anyway.
    """
    from smrti.call_log import append as _log

    system_prompt = AGENT_EXTRACTION_PROMPT if source == "agent" else EXTRACTION_PROMPT
    if entity_context and source != "agent":
        user_content = (
            f"{_temporal_block(write_time)}"
            f"[Known entities — use these to resolve pronouns and references]\n"
            f"{entity_context}\n\n"
            f"[Text to extract]\n{text}"
        )
    else:
        user_content = f"{_temporal_block(write_time)}{text}" if write_time else text

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
        resp = await _post_chat(http, upstream, auth, request_body)
        entry["status"] = resp.status_code
        if not 200 <= resp.status_code < 300:
            entry["error"] = f"upstream returned HTTP {resp.status_code}"
            return None
        data = resp.json()
        msg = data["choices"][0]["message"]
        raw = (msg.get("content") or msg.get("reasoning_content") or "").strip()
        entry["response_raw"] = raw[:2000]
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        parsed = _validate_extraction(json.loads(raw))
        if parsed is None:
            entry["error"] = "response is not a JSON object"
            return None
        entry["response_parsed"] = parsed
        return parsed
    except Exception as exc:
        entry["error"] = str(exc)
        return None
    finally:
        entry["duration_ms"] = round((time.monotonic() - t0) * 1000, 1)
        _log(entry)


_COREF_TYPES = {"person", "organization", "project", "role", "tool", "technology", "skill", "location", "event", "topic", "media", "health", "goal", "preference", "constraint"}


def _get_sole_person(mem: "Smrti") -> tuple[str, str] | None:
    """Return (label, atom_id) of the only person atom in the write space.

    Returns None when zero or multiple person atoms exist — first-person
    claims are only attributed when the speaker is unambiguous.
    """
    rows = mem.db.fetchall(
        """SELECT id, label FROM atoms
           WHERE tenant_id = ? AND space = ? AND entity_type = 'person'
             AND source_id IS NULL AND type IN ('concept', 'belief', 'goal')
           LIMIT 2""",
        (mem.tenant_id, mem.write_space),
    )
    if len(rows) != 1:
        return None
    return (rows[0]["label"], rows[0]["id"])


# Bookkeeping relations have no place in the model's view of an entity.
_STRUCTURAL_RELATIONS = STRUCTURAL_RELATIONS

# How many recorded facts each known entity carries into the prompt.
_CONTEXT_CLAIMS_PER_ENTITY = 6


def _build_entity_context(mem: "Smrti") -> str:
    """Return a compact list of salient named entities from the memory graph.

    Used to ground coreference resolution in the extraction prompt — e.g. so
    that "I" resolves to "Nico" even when the name isn't in the current message.
    Covers all entity types that can plausibly be referenced by a pronoun or
    short noun phrase: person, organization, project, tool, location, event, goal.

    Each entity also carries what the graph already records about it
    (``Alice (person): lives_in Amsterdam; works_for Acme``). That is what
    lets the model say a new fact *replaces* an old one — the ``supersedes``
    field on a claim — which is the one way the graph ever learns that a
    belief has stopped being true. Superseded claims are left out: the model
    should see the current state, not the history.
    """
    rows = mem.db.fetchall(
        f"""SELECT id, label, entity_type
           FROM atoms
           WHERE tenant_id = ? AND space = ? AND type IN ('concept', 'belief', 'goal')
             AND source_id IS NULL AND entity_type IS NOT NULL
             AND NOT {ATOM_FORGOTTEN}
           ORDER BY (sti + lti) DESC
           LIMIT 30""",
        (mem.tenant_id, mem.write_space),
    )
    entities = [
        (row["id"], row["label"], row["entity_type"] or "")
        for row in rows
        if (row["entity_type"] or "") in _COREF_TYPES
    ]
    if not entities:
        return ""

    ids = [atom_id for atom_id, _, _ in entities]
    ph = ",".join("?" * len(ids))
    rel_ph = ",".join("?" * len(_STRUCTURAL_RELATIONS))
    claims = mem.db.fetchall(
        f"""SELECT r.source_id AS subject_id, r.relation AS predicate, t.label AS object
            FROM atoms r JOIN atoms t ON t.id = r.target_id
            WHERE r.type = 'relation' AND r.tenant_id = ? AND r.space = ?
              AND r.source_id IN ({ph})
              AND r.relation NOT IN ({rel_ph})
              AND {claim_current_sql('r')}
              AND NOT {forgotten_sql('t')}
            ORDER BY r.confidence DESC, r.created_at DESC""",
        (mem.tenant_id, mem.write_space, *ids, *_STRUCTURAL_RELATIONS),
    )
    facts: dict[str, list[str]] = {}
    for claim in claims:
        held = facts.setdefault(claim["subject_id"], [])
        if len(held) < _CONTEXT_CLAIMS_PER_ENTITY:
            held.append(f"{claim['predicate']} {claim['object']}")

    lines = []
    for atom_id, label, etype in entities:
        line = f"- {label} ({etype})"
        if facts.get(atom_id):
            line += ": " + "; ".join(facts[atom_id])
        lines.append(line)
    return "\n".join(lines)


# ── Reusable helpers ──────────────────────────────────────────────────────────


def _register_entity(entity_ids: dict[str, str], name: str, atom_id: str) -> None:
    """Record name → atom_id with a lowercase convenience key.

    The exact-case key is always written, so when two same-message names
    differ only in case and resolve to different atoms, exact references keep
    resolving to their own atom; the lowercase key is first-writer-wins.
    """
    entity_ids[name] = atom_id
    entity_ids.setdefault(name.lower(), atom_id)


def _db_resolve_label(label: str, entity_ids: dict[str, str], mem: "Smrti") -> str | None:
    """Resolve a claim subject/object to an atom_id, falling back to DB lookup."""
    atom_id = entity_ids.get(label) or entity_ids.get(label.lower())
    if atom_id:
        return atom_id
    row = mem.db.fetchone(
        "SELECT id FROM atoms WHERE u_lower(label) = u_lower(?) AND tenant_id = ? AND space = ? AND type NOT IN ('relation', 'episode')",
        (label, mem.tenant_id, mem.write_space),
    )
    if row:
        _register_entity(entity_ids, label, row["id"])
        return row["id"]
    return None


def _agent_trust(mem: "Smrti") -> float:
    """Read this space's agent_source_trust, falling back to the schema default."""
    row = mem.db.fetchone(
        "SELECT agent_source_trust FROM personality WHERE tenant_id = ? AND space = ?",
        (mem.tenant_id, mem.write_space),
    )
    if row is None or row["agent_source_trust"] is None:
        return 0.5
    return row["agent_source_trust"]


def _episode_text(mem: "Smrti", episode_id: str) -> str:
    """The text of the episode an extraction is running over, or nothing."""
    if not episode_id:
        return ""
    row = mem.db.fetchone(
        "SELECT content, label FROM atoms WHERE id = ? AND tenant_id = ?",
        (episode_id, mem.tenant_id),
    )
    if row is None:
        return ""
    return row["content"] or row["label"] or ""


def _decisions(mem: "Smrti") -> DecisionEngine | None:
    """The instance's decision engine, or None for a double that has none."""
    engine = getattr(mem, "decisions", None)
    return engine if isinstance(engine, DecisionEngine) else None


def _resolver(mem: "Smrti", source: str, episode_id: str, context: str | None = None) -> "EntityResolver":
    """An entity resolver for this extraction, carrying the sentence the
    mentions came from so an uncertain match can be verified against it."""
    from .resolve import EntityResolver

    return EntityResolver(
        mem.db, mem.embed,
        source=source, agent_trust=_agent_trust(mem), episode_id=episode_id,
        decisions=_decisions(mem),
        context=_episode_text(mem, episode_id) if context is None else context,
    )


def _resolve_ner_entities(
    entities: list[dict],
    episode_id: str,
    mem: "Smrti",
    source: str = "user",
    context: str | None = None,
) -> dict[str, str]:
    """Resolve a list of {"name", "type"} dicts via the entity cascade.

    Returns a mapping of name → atom_id. Also creates mentions edges.
    Handles pronoun entities: batch-merges where unambiguous, skips type="pronoun",
    and retroactively merges existing pronoun atoms for resolved named persons.
    ``context`` is the text the mentions came from; read from the episode
    when not given.
    """
    resolver = _resolver(mem, source, episode_id, context)
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
        _register_entity(entity_ids, name, atom_id)
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


# Lower an atom's tone to a claim's, in both pairs. The claim is the model's
# reading of what was said about the atom — an estimate, so it never marks
# the valence as stated — but it is a better estimate of the atom's own tone
# than the neutral default an extracted concept is born with, so it reaches
# the intrinsic pair that every judgement reads and not only the drifting one
# propagation moves. Only ever downward: a claim can make a memory graver,
# never lighter.
_LOWER_TONE_SQL = """UPDATE atoms SET
        valence = MIN(valence, ?),
        intensity = MAX(intensity, ?),
        intrinsic_valence = MIN(COALESCE(intrinsic_valence, valence), ?),
        intrinsic_intensity = MAX(COALESCE(intrinsic_intensity, intensity), ?)
    WHERE id = ?"""

def _lower_tone(mem: "Smrti", atom_id: str, valence: float) -> None:
    intensity = min(1.0, abs(valence))
    mem.db.execute(_LOWER_TONE_SQL, (valence, intensity, valence, intensity, atom_id))


def _link_claims(
    claims: list[dict],
    entity_ids: dict[str, str],
    mem: "Smrti",
    episode_id: str = "",
    source: str = "user",
) -> None:
    """Create relation edges from claim triplets.

    Each claim is processed independently — a malformed one (non-numeric
    valence, non-string labels) is logged and skipped, never aborting the
    batch. Valence is clamped to [-1, 1] and intensity to [0, 1] before any
    DB write.

    A claim carrying ``supersedes`` replaces an earlier claim with the same
    subject and predicate: the old edge is marked, linked to the new one by a
    ``contradicts`` edge naming it the loser, and filed negative evidence
    against — see :func:`_supersede`.
    """
    claim_resolver = None
    min_valence = 0.0
    for claim in claims:
        try:
            subj_raw = claim.get("subject", "")
            obj_raw = claim.get("object", "")
            subj_id = _db_resolve_label(subj_raw, entity_ids, mem)
            obj_id = _db_resolve_label(obj_raw, entity_ids, mem)
            # Auto-create missing object atoms as concepts rather than silently dropping
            if not obj_id and obj_raw:
                if claim_resolver is None:
                    claim_resolver = _resolver(mem, source, episode_id)
                obj_id = claim_resolver.resolve(obj_raw, "concept", mem.tenant_id, mem.write_space, [mem.write_space])
                _register_entity(entity_ids, obj_raw, obj_id)
            if subj_id and obj_id and subj_id != obj_id:
                predicate = claim.get("predicate", "related_to")
                claim_valence = max(-1.0, min(1.0, float(claim.get("valence") or 0.0)))
                # A claim edge read from an agent turn carries the agent's
                # provenance, as the atoms it joins do: it is what lets the
                # supersession writer tell a user's statement from a
                # model's guess about the same subject.
                new_edge = mem.atomspace.link_atoms(
                    subj_id, obj_id, predicate,
                    mem.tenant_id, mem.write_space,
                    valence=claim_valence,
                    metadata={"source": SOURCE_AGENT} if source == SOURCE_AGENT else None,
                )
                if _is_superseded(mem, new_edge):
                    _revive(mem, new_edge, obj_id, episode_id, source)
                # Lower the target's tone to the claim's straight away. Epoch
                # propagation cannot: relation atoms have no neighbours of
                # their own, so propagate_valence finds nothing to move.
                if claim_valence < -0.3:
                    _lower_tone(mem, obj_id, claim_valence)
                    if claim_valence < min_valence:
                        min_valence = claim_valence
                # Safety net: promote target atom to goal type on has_goal claims
                if predicate == "has_goal":
                    _promote_to_goal(obj_id, mem)
                old_label = claim.get("supersedes")
                if isinstance(old_label, str) and old_label.strip():
                    _supersede(
                        mem, subj_id, predicate, old_label.strip(), obj_id,
                        new_edge, entity_ids, episode_id, source,
                    )
        except Exception as exc:
            logger.warning("skipping malformed claim %r: %s", claim, exc)
            continue

    # Carry the gravest claim's tone back to the episode it came from. This is
    # still an estimate — it never sets VALENCE_STATED, so the episode cannot
    # become a behavioural constraint on the model's reading alone — but it is
    # the better estimate, and it ranks and protects the memory accordingly.
    if episode_id and min_valence < -0.3:
        _lower_tone(mem, episode_id, min_valence)


# What the supersession check filed on a claim edge it declined to let
# supersede: the verdict, so a later pass or a reader can see why both
# claims are still current.
SUPERSESSION_DEFERRED = "supersession_deferred"


def _supersede(
    mem: "Smrti",
    subj_id: str,
    predicate: str,
    old_label: str,
    new_obj_id: str,
    new_edge_id: str,
    entity_ids: dict[str, str],
    episode_id: str,
    source: str,
) -> None:
    """Record that the new claim replaces an older one about the same subject.

    This is the producer the contradiction step was missing, and the one path
    by which a belief's probability can fall. For the old claim edge, and for
    the old object too when it is a belief atom (a superseded preference or
    constraint): a ``contradicts`` edge from the new to the old naming the old
    one as the loser, so the epoch cuts it to ``SUPERSEDED_PROBABILITY``
    rather than adjudicating by confidence (the older claim is usually the
    more confident, having been mentioned more); negative evidence against
    it, so the log records why; and, on the edge, a ``superseded_by`` mark
    that drops it from the entity context and from the proxy's rendering of
    the entity. A superseded belief then reads as a known antipattern at
    recall: the thing the user used to prefer.
    """
    old_obj_id = _db_resolve_label(old_label, entity_ids, mem)
    if not old_obj_id or old_obj_id == new_obj_id:
        return
    edge_columns = f"id, relation, created_at, updated_at, {ATOM_SOURCE} AS author"
    old_edge = mem.db.fetchone(
        f"""SELECT {edge_columns} FROM atoms WHERE type = 'relation' AND source_id = ? AND target_id = ?
           AND relation = ? AND tenant_id = ? AND space = ? AND id != ?""",
        (subj_id, old_obj_id, predicate, mem.tenant_id, mem.write_space, new_edge_id),
    )
    if old_edge is None:
        # The earlier claim may have been extracted under a different wording
        # of the same predicate; any factual edge to the old object will do.
        rel_ph = ",".join("?" * len(_STRUCTURAL_RELATIONS))
        old_edge = mem.db.fetchone(
            f"""SELECT {edge_columns} FROM atoms WHERE type = 'relation' AND source_id = ? AND target_id = ?
                AND relation NOT IN ({rel_ph}) AND tenant_id = ? AND space = ? AND id != ?
                ORDER BY created_at DESC LIMIT 1""",
            (subj_id, old_obj_id, *_STRUCTURAL_RELATIONS, mem.tenant_id, mem.write_space, new_edge_id),
        )
    if old_edge is None:
        return
    old_edge_id = old_edge["id"]

    # Source trust is a rule, not a judgement: a model's reading of its own
    # reply never replaces what the user stated. The assistant guessing a
    # different employer is not a correction, and elevating it to one would
    # let the graph overwrite testimony with inference.
    if source == SOURCE_AGENT and old_edge["author"] != SOURCE_AGENT:
        logger.info(
            "agent claim %s %s does not supersede the user's %s", predicate, new_obj_id, old_label
        )
        return

    if not _supersession_allowed(
        mem, subj_id, predicate, old_label, new_obj_id, new_edge_id, old_edge, episode_id, source,
    ):
        return

    mem.db.execute(
        f"""UPDATE atoms SET metadata = json_set({ATOM_METADATA_JSON}, '$.{SUPERSEDED_BY}', ?)
            WHERE id = ? AND tenant_id = ? AND space = ?""",
        (new_edge_id, old_edge_id, mem.tenant_id, mem.write_space),
    )

    new_obj = mem.db.fetchone("SELECT label FROM atoms WHERE id = ?", (new_obj_id,))
    note = f"superseded: {predicate} {old_label} -> {new_obj['label'] if new_obj else new_obj_id}"
    trust = _agent_trust(mem) if source == SOURCE_AGENT else 1.0
    pairs = [(new_edge_id, old_edge_id)]
    # The object belief falls with the claim only when this claim was the
    # last one holding it. "Tea" is one atom however many people prefer it,
    # and cutting it because Alice switched to coffee told the graph that
    # Bob had too.
    old_obj = mem.db.fetchone("SELECT type FROM atoms WHERE id = ?", (old_obj_id,))
    if (
        old_obj
        and old_obj["type"] == "belief"
        and not _still_claimed(mem, old_obj_id, except_edge_id=old_edge_id)
    ):
        pairs.append((new_obj_id, old_obj_id))
    for winner, loser in pairs:
        mem.atomspace.link_atoms(
            winner, loser, "contradicts", mem.tenant_id, mem.write_space,
            metadata={"loser": loser},
        )
        mem.atomspace.add_evidence(Evidence(
            atom_id=loser,
            observed_probability=SUPERSEDED_PROBABILITY,
            weight=trust,
            source_episode_id=episode_id or None,
            text=note,
            source=source,
            tenant_id=mem.tenant_id,
            space=mem.write_space,
        ))


def _supersession_allowed(
    mem: "Smrti",
    subj_id: str,
    predicate: str,
    old_label: str,
    new_obj_id: str,
    new_edge_id: str,
    old_edge,
    episode_id: str,
    source: str,
) -> bool:
    """Whether the semantic check lets the new claim replace the old edge.

    True whenever the ``supersession`` task is off or could not answer —
    the extractor's ``supersedes`` field decided alone before the check
    existed and still does then. A verdict that keeps both claims is
    stamped on the new edge (``$.supersession_deferred``) so a reader can
    see why the older claim is still current. The network call runs
    outside any transaction; the old edge is read again afterwards, and
    if it changed or went away while the provider was thinking, the
    mutation is not made on a stale reading.
    """
    engine = _decisions(mem)
    if engine is None:
        return True
    subject = mem.db.fetchone("SELECT label FROM atoms WHERE id = ?", (subj_id,))
    new_obj = mem.db.fetchone("SELECT label FROM atoms WHERE id = ?", (new_obj_id,))
    verdict = verify_supersession(
        engine,
        episode_text=_episode_text(mem, episode_id),
        subject=subject["label"] if subject else subj_id,
        predicate=old_edge["relation"] or predicate,
        old_object=old_label,
        new_object=new_obj["label"] if new_obj else new_obj_id,
        old_stated_at=old_edge["created_at"] or "",
        old_author=old_edge["author"] or "user",
        new_author=source,
        tenant_id=mem.tenant_id,
        space=mem.write_space,
    )
    if verdict is None:
        return True
    current = mem.db.fetchone(
        f"""SELECT updated_at FROM atoms r WHERE r.id = ? AND r.tenant_id = ? AND r.space = ?
            AND {claim_current_sql('r')}""",
        (old_edge["id"], mem.tenant_id, mem.write_space),
    )
    if current is None or current["updated_at"] != old_edge["updated_at"]:
        logger.info("old claim %s changed while it was being verified; not superseding", old_edge["id"])
        return False
    if not verdict.applied or verdict.allow:
        return True
    mem.db.execute(
        f"""UPDATE atoms SET metadata = json_set({ATOM_METADATA_JSON}, '$.{SUPERSESSION_DEFERRED}', json(?))
            WHERE id = ? AND tenant_id = ? AND space = ?""",
        (
            json.dumps({"old_edge": old_edge["id"], "label": verdict.label,
                        "confidence": round(verdict.confidence, 3)}),
            new_edge_id, mem.tenant_id, mem.write_space,
        ),
    )
    return False


def _still_claimed(mem: "Smrti", obj_id: str, except_edge_id: str) -> bool:
    """Whether a current factual claim other than *except_edge_id* points at the object."""
    rel_ph = ",".join("?" * len(_STRUCTURAL_RELATIONS))
    return (
        mem.db.fetchone(
            f"""SELECT 1 FROM atoms r WHERE r.type = 'relation' AND r.target_id = ?
                AND r.id != ? AND r.relation NOT IN ({rel_ph})
                AND r.tenant_id = ? AND r.space = ? AND {claim_current_sql('r')}
                LIMIT 1""",
            (obj_id, except_edge_id, *_STRUCTURAL_RELATIONS, mem.tenant_id, mem.write_space),
        )
        is not None
    )


def _is_superseded(mem: "Smrti", edge_id: str) -> bool:
    row = mem.db.fetchone(
        f"SELECT 1 FROM atoms r WHERE r.id = ? AND NOT {claim_current_sql('r')}",
        (edge_id,),
    )
    return row is not None


# What a revived claim, and the belief it points at, are held at once more.
# The supersession cut them to SUPERSEDED_PROBABILITY by policy; this is the
# same policy run backwards, to the probability a claim edge is born with.
_REVIVED_PROBABILITY = 0.8


def _revive(
    mem: "Smrti", edge_id: str, obj_id: str, episode_id: str, source: str
) -> None:
    """Make a claim the graph had marked replaced current again.

    Stating a claim that a later one superseded — Berlin, then Paris, then
    Berlin again — reuses the original edge, and used to leave its mark in
    place: the update then superseded Paris as well, and neither city was
    current. The mark comes off, and the cut is reversed where it landed —
    the edge, and its object when that is a belief the supersession cut
    too: every ``contradicts`` edge still naming one of them the loser is
    deleted, since an unresolved one would cut the claim again at the next
    epoch and a resolved one has nothing left to say; the probability is
    lifted back by the policy write that lowered it, so a preference the
    user returned to stops reading as a known antipattern; and evidence of
    the revival is filed, so the log says why.
    """
    losers = mem.db.fetchall(
        """SELECT id, CASE WHEN json_valid(metadata)
                           THEN json_extract(metadata, '$.loser') END AS loser
           FROM atoms WHERE type = 'relation' AND relation = 'contradicts'
             AND tenant_id = ? AND space = ? AND target_id IN (?, ?)""",
        (mem.tenant_id, mem.write_space, edge_id, obj_id),
    )
    cut = [c for c in losers if c["loser"] in (edge_id, obj_id)]
    if cut:
        mem.atomspace.delete_atoms([c["id"] for c in cut], mem.tenant_id)
    revived = {edge_id, *(c["loser"] for c in cut)}
    trust = _agent_trust(mem) if source == SOURCE_AGENT else 1.0
    for atom_id in revived:
        mem.db.execute(
            f"""UPDATE atoms SET
                    probability = MAX(probability, ?),
                    metadata = json_remove({ATOM_METADATA_JSON}, '$.{SUPERSEDED_BY}')
                WHERE id = ? AND tenant_id = ? AND space = ?""",
            (_REVIVED_PROBABILITY, atom_id, mem.tenant_id, mem.write_space),
        )
        mem.atomspace.add_evidence(Evidence(
            atom_id=atom_id,
            observed_probability=_REVIVED_PROBABILITY,
            weight=trust,
            source_episode_id=episode_id or None,
            text="stated again after being superseded",
            source=source,
            tenant_id=mem.tenant_id,
            space=mem.write_space,
        ))


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
    entity_context: str | None = None,
) -> None:
    """Extract entities/claims from content and link them to the episode atom.

    Shared by all serve modes (proxy, MCP, REST). Silently no-ops if the LLM
    call fails or returns no usable structure. ``entity_context`` is the
    known-entities block when the caller already built it.
    """
    loop = asyncio.get_running_loop()
    if entity_context is None:
        entity_context = await loop.run_in_executor(None, _build_entity_context, mem)
    write_time = await loop.run_in_executor(None, _write_time, episode_id, mem)
    extracted = await extract_knowledge(
        content, _get_http(), upstream, auth, model, entity_context, source,
        mem.tenant_id, write_time,
    )
    if not extracted:
        return

    def _sync_work() -> None:
        entity_ids = _resolve_ner_entities(extracted.get("entities", []), episode_id, mem, source, content)
        _link_claims(extracted.get("claims", []), entity_ids, mem, episode_id, source)
        _store_temporal(episode_id, mem, extracted.get("temporal", []))

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
    write_time: str = "",
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

    user_content = f"{_temporal_block(write_time)}{text}" if write_time else text
    if entity_context:
        user_content = (
            f"{_temporal_block(write_time)}"
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
        resp = await _post_chat(_get_http(), upstream, auth, request_body)
        entry["status"] = resp.status_code
        if not 200 <= resp.status_code < 300:
            entry["error"] = f"upstream returned HTTP {resp.status_code}"
            return None
        data = resp.json()
        msg = data["choices"][0]["message"]
        raw = (msg.get("content") or msg.get("reasoning_content") or "").strip()
        entry["response_raw"] = raw[:2000]
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        parsed = _validate_extraction(json.loads(raw))
        if parsed is None:
            entry["error"] = "response is not a JSON object"
            return None
        entry["response_parsed"] = parsed
        return parsed
    except Exception as exc:
        entry["error"] = str(exc)
        return None
    finally:
        entry["duration_ms"] = round((time.monotonic() - t0) * 1000, 1)
        _log(entry)


# ── Hybrid dispatch ───────────────────────────────────────────────────────────


_warned_no_upstream = False


def _effective_mode(mode: str, upstream: str) -> str:
    """Downgrade the LLM-dependent modes to local when no upstream is set.

    Entities still get extracted locally; only claim extraction needs a model.
    Warns once so the operator learns why claims stopped appearing instead of
    discovering it in the call log.
    """
    global _warned_no_upstream
    if upstream or mode == "local":
        return mode
    if not _warned_no_upstream:
        _warned_no_upstream = True
        logger.warning(
            "no extraction endpoint configured (set SMRTI_EXTRACT_URL or "
            "SMRTI_UPSTREAM_URL) — extracting entities locally and skipping claims"
        )
    return "local"


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
    source == "agent" takes the full LLM path. The routing gate, when the
    ``routing`` decision task is on, runs before any of that and may route
    a message past the LLM (local entities only) or to it (a durable fact
    or correction with fewer than two entities).
    """
    mode = _effective_mode(mode, upstream)
    loop = asyncio.get_running_loop()

    # The routing gate. Before any model is called, three yes/no judgements
    # about the message decide whether the LLM claim extraction is worth
    # its call: chatter that holds nothing durable, corrects nothing and
    # adds nothing is routed past it (the episode is already stored, and
    # local NER still runs so it is linked to what it mentions), while a
    # durable fact or a correction is routed *to* it even where the entity
    # count below would not have called it — "never deploy on Fridays"
    # names no entity. Off, unavailable, or in shadow, nothing changes.
    entity_context: str | None = None
    route = None
    engine = _decisions(mem)
    if mode != "local" and engine is not None and engine.enabled("routing"):
        entity_context = await loop.run_in_executor(None, _build_entity_context, mem)
        route = await route_extraction(
            engine, content, source=source, entity_context=entity_context,
            tenant_id=mem.tenant_id, space=mem.write_space,
        )
    if route is not None and route.skip:
        # The episode is stored and embedded already; what the skip route
        # adds is the mentions edges local NER can find. In llm mode there
        # is no local NER — that mode is for machines that cannot hold the
        # tagger — so the episode stays reachable by its vector alone.
        if mode != "llm":
            await _link_local_entities(episode_id, content, mem, source)
        return

    if source == "agent" or mode == "llm":
        await extract_and_link(episode_id, content, mem, auth, model, upstream, source, entity_context)
        return

    # Try GLiNER for entity extraction
    ner_entities: list[dict] | None = None
    try:
        from smrti.extraction import ner as ner_mod

        ner_instance = ner_mod.get_ner()
        ner_entities = await loop.run_in_executor(None, ner_instance.extract, content)
    except ImportError:
        if mode == "hybrid":
            await extract_and_link(episode_id, content, mem, auth, model, upstream, source, entity_context)
            return
        # local mode with no gliner installed — nothing we can do
        return
    except Exception:
        if mode == "hybrid":
            await extract_and_link(episode_id, content, mem, auth, model, upstream, source, entity_context)
            return
        return

    if not ner_entities:
        # NER found nothing — fall through to full LLM extraction so standalone
        # directives/constraints that NER can't parse still get extracted.
        if mode == "hybrid":
            await extract_and_link(episode_id, content, mem, auth, model, upstream, source, entity_context)
        return

    # Resolve entities and create mentions edges
    def _sync_resolve() -> dict[str, str]:
        return _resolve_ner_entities(ner_entities, episode_id, mem, source, content)

    entity_ids = await loop.run_in_executor(None, _sync_resolve)

    # In local mode, we're done — no LLM calls
    if mode == "local":
        return

    # Speaker injection: for user messages, if no person atom resolved (pronoun dropped
    # because "I" alias wasn't in the alias table), inject the sole known person
    # so claims — especially goals, preferences, and actions — can be attributed to them.
    # Attribution only happens when exactly one person atom exists in the write
    # space; with several candidates the speaker is ambiguous and we skip it.
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
            person = _get_sole_person(mem)
            if person is None:
                logger.debug(
                    "speaker attribution skipped: no unambiguous person atom in %s/%s",
                    mem.tenant_id, mem.write_space,
                )
                return ner_entities
            label, atom_id = person
            _register_entity(entity_ids, label, atom_id)
            return ner_entities + [{"name": label, "type": "person"}]

        ner_entities = await loop.run_in_executor(None, _inject_speaker_if_missing)

    # Hybrid mode: call LLM for claims only when 2+ unique entities — unless
    # the routing gate found a durable fact or a correction the entity count
    # cannot see, in which case the full extraction runs on it.
    unique_ids = set(entity_ids.values())
    if len(unique_ids) < 2:
        if route is not None and route.force_llm:
            await extract_and_link(episode_id, content, mem, auth, model, upstream, source, entity_context)
        return

    if entity_context is None:
        entity_context = await loop.run_in_executor(None, _build_entity_context, mem)
    write_time = await loop.run_in_executor(None, _write_time, episode_id, mem)
    claims_result = await extract_claims_only(
        content, ner_entities, upstream, auth, model, entity_context, mem.tenant_id,
        write_time,
    )
    if not claims_result:
        return

    def _sync_resolve_and_link() -> None:
        # Resolve new entities the LLM emitted: goals (new atoms) and
        # preference/constraint reclassifications (resolve to existing atom,
        # updating its entity_type so it becomes a belief atom).
        _ALLOWED_NEW_TYPES = {"goal", "preference", "constraint", "role", "technology", "skill", "topic", "media", "health", "concept"}
        new_entities = claims_result.get("entities", [])
        if new_entities:
            resolver = _resolver(mem, source, episode_id, content)
            for ent in new_entities:
                name = (ent.get("name") or "").strip()
                etype = ent.get("type", "")
                if not name or etype not in _ALLOWED_NEW_TYPES:
                    continue
                atom_id = resolver.resolve(name, etype, mem.tenant_id, mem.write_space, [mem.write_space])
                _register_entity(entity_ids, name, atom_id)
                if etype in ("preference", "constraint"):
                    # Reclassify the atom: concept → belief, update entity_type
                    mem.db.execute(
                        "UPDATE atoms SET type = 'belief', entity_type = ? WHERE id = ? AND type = 'concept'",
                        (etype, atom_id),
                    )
                elif etype not in ("goal", "preference", "constraint"):
                    mem.atomspace.link_atoms(episode_id, atom_id, "mentions", mem.tenant_id, mem.write_space)
        _link_claims(claims_result.get("claims", []), entity_ids, mem, episode_id, source)
        _store_temporal(episode_id, mem, claims_result.get("temporal", []))

    await loop.run_in_executor(None, _sync_resolve_and_link)


async def _link_local_entities(episode_id: str, content: str, mem: "Smrti", source: str) -> None:
    """Local NER and resolution only — the skip route's "local processing".

    The episode stays, and it is linked to the entities it mentions, so a
    memory the gate judged not worth a claim is still reachable through
    the graph. No model is called; an NER failure is logged and swallowed.
    """
    try:
        from smrti.extraction import ner as ner_mod

        ner_instance = ner_mod.get_ner()
        loop = asyncio.get_running_loop()
        entities = await loop.run_in_executor(None, ner_instance.extract, content)
        if entities:
            await loop.run_in_executor(
                None, _resolve_ner_entities, entities, episode_id, mem, source, content
            )
    except Exception:
        logger.debug("local entity linking failed for episode %s", episode_id, exc_info=True)


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
    async with _session_locks.get_lock(key):
        await extract_and_link_hybrid(
            episode_id, content, mem, auth, model, upstream, source, mode
        )
